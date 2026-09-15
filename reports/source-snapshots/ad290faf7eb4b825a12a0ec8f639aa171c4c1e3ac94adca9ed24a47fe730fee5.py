"""Bounded paired original-versus-packed MoE probe. No learned weight updates."""
import collections
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))
def write(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n", encoding="utf-8")
def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def summary(a):
    return {"samples": a, "mean": statistics.mean(a), "median": statistics.median(a),
            "min": min(a), "max": max(a)}
def timeout():
    print("FAIL: bounded profiling deadline (120 seconds) exceeded", flush=True)
    os._exit(124)

def main():
    timer = threading.Timer(120, timeout)
    timer.daemon = True
    timer.start()
    manifest = read(HERE / "probe-manifest.json")
    for rel, sha in manifest["files"].items():
        assert digest(HERE / rel) == sha, rel
    checkpoint = ROOT / manifest["checkpoint"]
    assert digest(checkpoint) == manifest["checkpoint_sha256"]
    os.environ["AUTORESEARCH_CACHE_DIR"] = str(ROOT / ".autoresearch/cache")
    os.environ["AUTORESEARCH_DATASET"] = "tinystories"
    sys.path[:0] = [str(HERE / "source/project"), str(HERE / "source/upstream")]
    import torch
    import prepare
    prepare.MAX_SEQ_LEN = 512
    import train
    import autoresearch_model as packed_module
    spec = importlib.util.spec_from_file_location("original_model", HERE / "original_model.py")
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)
    train.WINDOW_PATTERN = "L"
    runtime = train.detect_runtime()
    train._configure_step_kernels(runtime)
    assert platform.system() == "Windows" and "5070 Ti" in torch.cuda.get_device_name()
    assert runtime.amp_dtype == torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    candidate = read(HERE / "moe.json")
    config = train.build_model_config(6, 8192, runtime, use_activation_checkpointing=True)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    pairs = {}
    for name, module in (("original", original_module), ("packed", packed_module)):
        with torch.device("meta"):
            model = module.model_class(train, candidate)(config)
        model.to_empty(device="cuda")
        model.init_weights(embed_dtype=runtime.amp_dtype)
        model.load_state_dict(state, strict=True)
        model.train()
        opt = model.setup_optimizer(unembedding_lr=train.UNEMBEDDING_LR,
            embedding_lr=train.EMBEDDING_LR, scalar_lr=train.SCALAR_LR,
            adam_betas=train.ADAM_BETAS, matrix_lr=.04, weight_decay=train.WEIGHT_DECAY)
        for g in opt.param_groups:
            g["lr"] = 0.
            if g["kind"] == "muon":
                g["momentum"] = .95
        pairs[name] = (model, opt)
    bank = torch.load(HERE / "batches.pt", weights_only=True).pin_memory()
    gpu = torch.empty_like(bank[0], device="cuda")
    def transfer(i):
        gpu.copy_(bank[i % len(bank)], non_blocking=True)
        return gpu[0], gpu[1]
    def forward(model, x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x, y)
    def sync():
        torch.cuda.synchronize()
    def update(name, offset):
        model, opt = pairs[name]
        for j in range(16):
            x, y = transfer(offset + j)
            loss = forward(model, x, y)
            (loss / 16).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    print("Preflight: saved depth-6 checkpoint loss/all-gradient and routing parity", flush=True)
    preflight = []
    for name, (model, opt) in pairs.items():
        x, y = transfer(0)
        loss = forward(model, x, y)
        loss.backward()
        preflight.append((loss.detach().cpu(),
                          {n: p.grad.detach().cpu() for n, p in model.named_parameters()},
                          model.routing_report()["train"]["counts"]))
        opt.zero_grad(set_to_none=True)
    torch.testing.assert_close(preflight[0][0], preflight[1][0], atol=1e-6, rtol=1e-5)
    max_grad_error = 0.
    for name, g in preflight[0][1].items():
        h = preflight[1][1][name]
        torch.testing.assert_close(g, h, atol=1e-6, rtol=1e-5)
        max_grad_error = max(max_grad_error, (g - h).abs().max().item())
    assert preflight[0][2] == preflight[1][2]
    del preflight
    print(f"PASS preflight; max gradient absolute difference {max_grad_error}", flush=True)
    gc.collect()
    gc.disable()
    for name in pairs:
        print(f"Warmup: {name}, three updates", flush=True)
        for i in range(3):
            update(name, i * 16)
    sync()
    samples = {name: [] for name in pairs}
    blocks = []
    memories = {name: [] for name in pairs}
    for block in range(6):
        order = ("original", "packed") if block % 2 == 0 else ("packed", "original")
        observed = {}
        for name in order:
            sync()
            idle = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for i in range(2):
                sync()
                start = time.perf_counter()
                loss = update(name, block * 32 + i * 16)
                sync()
                times.append(time.perf_counter() - start)
            assert torch.isfinite(loss).item()
            samples[name].extend(times)
            observed[name] = sum(times)
            memories[name].append({"idle_joint_bytes": idle,
                "peak_joint_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_joint_reserved_bytes": torch.cuda.max_memory_reserved(),
                "increment_above_joint_idle_bytes": torch.cuda.max_memory_allocated() - idle})
        gain = observed["original"] / observed["packed"] - 1.
        blocks.append({"index": block, "order": order, "seconds": observed, "throughput_gain": gain})
        print(f"Pair {block + 1}/6: {json.dumps(blocks[-1])}", flush=True)
    aggregate_gain = sum(samples["original"]) / sum(samples["packed"]) - 1.
    gate = all(b["throughput_gain"] > 0 for b in blocks) and aggregate_gain >= .05
    print(f"Uninstrumented aggregate gain={aggregate_gain:.6%}; consistent gate={gate}", flush=True)
    # Independent phase timings, not used to calculate throughput.
    phases = {}
    for name, (model, opt) in pairs.items():
        values = {k: [] for k in ("data", "forward", "backward", "optimizer", "zero_grad")}
        for i in range(3):
            totals = collections.defaultdict(float)
            for j in range(16):
                sync(); start = time.perf_counter()
                x, y = transfer(i * 16 + j)
                sync(); totals["data"] += time.perf_counter() - start
                start = time.perf_counter(); loss = forward(model, x, y)
                sync(); totals["forward"] += time.perf_counter() - start
                start = time.perf_counter(); (loss / 16).backward()
                sync(); totals["backward"] += time.perf_counter() - start
            start = time.perf_counter(); opt.step()
            sync(); totals["optimizer"] = time.perf_counter() - start
            start = time.perf_counter(); opt.zero_grad(set_to_none=True)
            sync(); totals["zero_grad"] = time.perf_counter() - start
            for k in values:
                values[k].append(totals[k])
        phases[name] = {k: summary(v) for k, v in values.items()}
    traces = {}
    for name, (model, opt) in pairs.items():
        print(f"Separate trace: {name}, two microbatches and one optimizer", flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=False) as prof:
            for j in range(2):
                x, y = transfer(j)
                with torch.profiler.record_function("phase/forward"):
                    loss = forward(model, x, y)
                    sync()
                with torch.profiler.record_function("phase/backward"):
                    (loss / 16).backward()
                    sync()
            with torch.profiler.record_function("phase/optimizer"):
                opt.step()
                sync()
            opt.zero_grad(set_to_none=True)
        prof.export_chrome_trace(str(HERE / (name + "-trace.json")))
        events = [{"name": e.key, "count": e.count, "input_shapes": e.input_shapes,
                   "self_cpu_us": e.self_cpu_time_total, "cpu_us": e.cpu_time_total,
                   "self_device_us": e.self_device_time_total}
                  for e in prof.key_averages(group_by_input_shape=True)]
        write(HERE / (name + "-operators.json"), events)
        raw = read(HERE / (name + "-trace.json"))["traceEvents"]
        kernels = [e for e in raw if e.get("ph") == "X" and e.get("cat") == "kernel"]
        synchronizations = [e for e in raw if e.get("ph") == "X"
                            and e.get("name") == "cudaStreamSynchronize"]
        counts = collections.Counter()
        for e in events:
            if e["name"].startswith("aten::"):
                counts[e["name"]] += e["count"]
        traces[name] = {"kernel_count": len(kernels), "kernel_sum_ms": sum(e["dur"] for e in kernels)/1000,
                        "stream_sync_count": len(synchronizations),
                        "stream_sync_ms": sum(e["dur"] for e in synchronizations)/1000,
                        "operator_counts": dict(counts)}
    results = {}
    for name, (model, opt) in pairs.items():
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor.cpu(), state[key]), (name, key)
        routing = model.routing_report()
        assert routing["train"]["dropped_tokens"] == 0
        assert routing["router_gradients_finite"] and all(g > 0 for g in routing["router_gradient_max_abs"])
        results[name] = {"timed_tokens": 12 * 16384, "seconds": sum(samples[name]),
                         "tokens_per_second": 12 * 16384 / sum(samples[name]),
                         "updates_seconds": summary(samples[name]), "phases": phases[name],
                         "trace": traces[name], "routing": routing, "parameters": model.parameter_report(),
                         "memory_joint_resident_scope": memories[name], "weights_unchanged": True}
    assert results["original"]["routing"]["train"]["counts"] == results["packed"]["routing"]["train"]["counts"]
    result = {"correctness_passed": True, "full_checkpoint_max_gradient_error": max_grad_error,
              "performance_gate_passed": gate, "aggregate_throughput_gain": aggregate_gain,
              "paired_blocks": blocks, "variants": results,
              "config": {k: str(v) if isinstance(v, torch.dtype) else v for k, v in vars(config).items()},
              "settings": "BF16, checkpoint on, same saved weights and inputs, 16 accumulation, lr zero",
              "limitations": "Both models/optimizer states resident; joint allocator peaks are not standalone VRAM. Phase/trace overhead excluded from throughput. Sequential short samples, uncontrolled GPU background load.",
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "nonzero_lr_updates": 0}
    write(HERE / "probe-result.json", result)
    for rel, sha in manifest["files"].items():
        assert digest(HERE / rel) == sha
    timer.cancel()
    print("COMPLETE: bounded comparison; gate=" + str(gate), flush=True)

if __name__ == "__main__":
    main()

