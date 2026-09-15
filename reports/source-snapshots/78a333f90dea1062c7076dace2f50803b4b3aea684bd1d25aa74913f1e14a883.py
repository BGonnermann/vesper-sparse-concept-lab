"""Bounded, frozen-weight profiling of the two preserved executed snapshots."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = {
    "dense": ROOT / "runs/autoresearch/20260914T231604Z-061092d4",
    "moe": ROOT / "runs/autoresearch/20260914T232147Z-ff9e4e60",
}
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))
def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def summary(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "samples": values}
def imports():
    os.environ["AUTORESEARCH_CACHE_DIR"] = str(ROOT / ".autoresearch/cache")
    os.environ["AUTORESEARCH_DATASET"] = "tinystories"
    sys.path[:0] = [str(HERE / "source/project"), str(HERE / "source/upstream")]
    import torch
    import prepare
    prepare.MAX_SEQ_LEN = 512
    import train
    import autoresearch_model
    train.WINDOW_PATTERN = "L"
    return torch, prepare, train, autoresearch_model

def prepare_probe():
    comparison = read(ROOT / "runs/autoresearch/moe-resume-20260914T231300Z/comparison.json")
    executions = {k: read(v / "execution.json") for k, v in RUNS.items()}
    for variant, execution in executions.items():
        assert execution["verified"] and execution["isolated"]
        for rel, sha in execution["files"].items():
            assert digest(RUNS[variant] / rel) == sha, (variant, rel)
    assert executions["dense"]["executed_files"] == executions["moe"]["executed_files"]
    assert read(RUNS["dense"] / "protocol.json") == read(RUNS["moe"] / "protocol.json")
    shutil.copytree(RUNS["moe"] / "source", HERE / "source")
    for variant, run in RUNS.items():
        (HERE / variant).mkdir()
        shutil.copy2(run / "candidate.json", HERE / variant / "candidate.json")
        shutil.copy2(run / "protocol.json", HERE / variant / "protocol.json")
    for rel, sha in comparison["data_hashes"].items():
        assert digest(ROOT / ".autoresearch/cache" / rel) == sha, rel
    torch, prepare, train, model_module = imports()
    assert platform.system() == "Windows" and "5070 Ti" in torch.cuda.get_device_name()
    tokenizer = prepare.Tokenizer.from_directory(dataset="tinystories")
    loader = prepare.make_dataloader(tokenizer, 2, 512, "train", device="cpu", dataset="tinystories")
    batches, timings = [], []
    for _ in range(64):
        start = time.perf_counter()
        x, y, _ = next(loader)
        timings.append(time.perf_counter() - start)
        batches.append(torch.stack((x.clone(), y.clone())))
    bank = torch.stack(batches)
    torch.save(bank, HERE / "batches.pt")
    manifest = {
        "source_hashes": {p.relative_to(HERE).as_posix(): digest(p)
                          for p in (HERE / "source").rglob("*") if p.is_file()},
        "executed_original_sources": executions["moe"]["executed_files"],
        "data_hashes": comparison["data_hashes"],
        "checkpoint_hashes": {k: digest(v / "checkpoint_pre_eval.pt") for k, v in RUNS.items()},
        "candidate_hashes": {k: digest(HERE / k / "candidate.json") for k in RUNS},
        "bank_sha256": digest(HERE / "batches.pt"),
        "bank_shape": list(bank.shape), "bank_dtype": str(bank.dtype),
        "vocab_size": tokenizer.get_vocab_size(),
        "protocol": read(RUNS["moe"] / "protocol.json"),
        "precision": "BF16 CUDA autocast, FP32 router softmax and matrix parameters",
        "activation_checkpointing": True, "accumulation": 16,
        "warmup_updates": 3, "timed_updates": 6, "decomposition_updates": 3,
        "trace_microbatches": 2, "worker_timeout_seconds": 120,
        "weights": "Each completed checkpoint; fixed weights, all optimizer lr=0, fresh warmed optimizer state",
        "throughput_scope": "Replay bank H2D + forward + backward + optimizer + zero_grad; no profiler/hooks or CPU packing",
        "loader_cpu_seconds": {"cold_first": timings[0], "steady": summary(timings[1:])},
        "probe_script_sha256": digest(Path(__file__)),
    }
    write(HERE / "manifest.json", manifest)
    print("Prepared identical sealed batch bank and verified source/config/data hashes.", flush=True)
    for variant in RUNS:
        command = [sys.executable, "-I", "-B", "-u", str(Path(__file__)), "--worker", variant]
        with (HERE / variant / "profile.log").open("w", encoding="utf-8") as log:
            print(f"Starting bounded {variant} profile: {HERE / variant / 'profile.log'}", flush=True)
            result = subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        assert result.returncode == 0, f"{variant} failed; see {HERE / variant / 'profile.log'}"
    preserved = read(HERE / "prior-preservation.json")
    for entry in preserved:
        assert digest(Path(entry["Path"])).upper() == entry["Hash"].upper(), entry["Path"]
    write(HERE / "preservation-result.json", {"verified_unchanged": len(preserved)})
    print("Both bounded profiles complete; prior artifacts unchanged.", flush=True)

def worker(variant):
    import gc
    manifest = read(HERE / "manifest.json")
    for rel, sha in manifest["source_hashes"].items():
        assert digest(HERE / rel) == sha
    assert digest(HERE / "batches.pt") == manifest["bank_sha256"]
    assert digest(Path(__file__)) == manifest["probe_script_sha256"]
    torch, prepare, train, model_module = imports()
    out = HERE / variant
    candidate = read(out / "candidate.json")
    assert digest(out / "candidate.json") == manifest["candidate_hashes"][variant]
    runtime = train.detect_runtime()
    train._configure_step_kernels(runtime)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    assert runtime.amp_dtype == torch.bfloat16
    config = train.build_model_config(6, manifest["vocab_size"], runtime, use_activation_checkpointing=True)
    with torch.device("meta"):
        model = model_module.model_class(train, candidate)(config)
    model.to_empty(device=runtime.device)
    model.init_weights(embed_dtype=runtime.amp_dtype)
    checkpoint = RUNS[variant] / "checkpoint_pre_eval.pt"
    assert digest(checkpoint) == manifest["checkpoint_hashes"][variant]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.train()
    optimizer = model.setup_optimizer(unembedding_lr=train.UNEMBEDDING_LR,
        embedding_lr=train.EMBEDDING_LR, scalar_lr=train.SCALAR_LR, adam_betas=train.ADAM_BETAS,
        matrix_lr=candidate["matrix_lr"], weight_decay=train.WEIGHT_DECAY)
    for group in optimizer.param_groups:
        group["lr"] = 0.0
        if group["kind"] == "muon":
            group["momentum"] = 0.95
    bank = torch.load(HERE / "batches.pt", weights_only=True).pin_memory()
    gpu = torch.empty_like(bank[0], device="cuda")
    def transfer(i):
        gpu.copy_(bank[i % len(bank)], non_blocking=True)
        return gpu[0], gpu[1]
    def forward(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x, y)
    def sync():
        torch.cuda.synchronize()
    def update(offset):
        for j in range(16):
            x, y = transfer(offset + j)
            loss = forward(x, y)
            (loss / 16).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss
    gc.collect()
    gc.disable()
    print(f"{variant}: three warmups, six uninstrumented updates", flush=True)
    for i in range(3):
        update(i * 16)
    sync()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(6):
        sync()
        start = time.perf_counter()
        loss = update(i * 16)
        sync()
        times.append(time.perf_counter() - start)
    assert torch.isfinite(loss).item()
    primary_memory = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
    primary = {"update_seconds": summary(times), "tokens": 6 * 16384,
               "seconds": sum(times), "tokens_per_second": 6 * 16384 / sum(times)}
    print(json.dumps(primary), flush=True)

    print("Separate synchronized phase decomposition (not throughput)", flush=True)
    phase_samples = {k: [] for k in ("data_h2d", "forward", "backward", "optimizer", "zero_grad")}
    for i in range(3):
        totals = collections.defaultdict(float)
        for j in range(16):
            sync(); start = time.perf_counter()
            x, y = transfer(i * 16 + j)
            sync(); totals["data_h2d"] += time.perf_counter() - start
            start = time.perf_counter()
            loss = forward(x, y)
            sync(); totals["forward"] += time.perf_counter() - start
            start = time.perf_counter()
            (loss / 16).backward()
            sync(); totals["backward"] += time.perf_counter() - start
        start = time.perf_counter()
        optimizer.step()
        sync(); totals["optimizer"] = time.perf_counter() - start
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        sync(); totals["zero_grad"] = time.perf_counter() - start
        for name in phase_samples:
            phase_samples[name].append(totals[name])
    # Standalone stock CUDA loader includes packing/tokenization and H2D; same first 32 batches.
    tokenizer = prepare.Tokenizer.from_directory(dataset="tinystories")
    loader = prepare.make_dataloader(tokenizer, 2, 512, "train", device="cuda", dataset="tinystories")
    loader_times = []
    for i in range(32):
        sync(); start = time.perf_counter()
        x, y, _ = next(loader)
        sync(); loader_times.append(time.perf_counter() - start)
        assert torch.equal(torch.stack((x.cpu(), y.cpu())), bank[i].cpu())
    del loader
    print("Separate two-microbatch operator trace", flush=True)
    handles, rows, scopes = [], [], {}
    current_phase = ["forward"]
    def instrument(name, module, expert=False):
        stack = []
        def enter(mod, args):
            ctx = torch.profiler.record_function("module/" + name + "/" + current_phase[0])
            ctx.__enter__()
            stack.append(ctx)
            if expert:
                rows.append({"module": name, "phase": current_phase[0],
                             "shape": list(args[0].shape), "dtype": str(args[0].dtype)})
        def leave(mod, args, output):
            stack.pop().__exit__(None, None, None)
        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(leave, always_call=True))
    for name, module in model.named_modules():
        if name.endswith(".attn") or name.endswith(".mlp") or name.endswith(".router"):
            instrument(name, module)
        elif ".mlp.experts." in name and name.split(".")[-1].isdigit():
            instrument(name, module, True)
    names = {id(p): n for n, p in model.named_parameters()}
    optimizer_groups = []
    for group in optimizer.param_groups:
        ns = [names[id(p)] for p in group["params"]]
        kind = ("expert" if any(".experts." in n for n in ns) else
                "router" if any(".router." in n for n in ns) else
                "dense_ffn" if any(".mlp." in n for n in ns) else "shared")
        optimizer_groups.append({"kind": group["kind"], "category": kind, "tensors": len(ns),
                                 "elements": sum(p.numel() for p in group["params"]),
                                 "shapes": [list(p.shape) for p in group["params"]], "names": ns})
        scopes[id(group)] = f"optimizer/{group['kind']}/{kind}"
    # Labels only; original unmodified optimizer methods perform all work.
    for method in ("_step_muon", "_step_adamw"):
        original = getattr(optimizer, method)
        def wrapped(group, call=original):
            with torch.profiler.record_function(scopes[id(group)]):
                return call(group)
        setattr(optimizer, method, wrapped)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, with_stack=False, profile_memory=False) as prof:
        for j in range(2):
            x, y = transfer(j)
            current_phase[0] = "forward"
            with torch.profiler.record_function("phase/forward"):
                loss = forward(x, y)
                sync()
            current_phase[0] = "backward"
            with torch.profiler.record_function("phase/backward"):
                (loss / 16).backward()
                sync()
        current_phase[0] = "optimizer"
        with torch.profiler.record_function("phase/optimizer"):
            optimizer.step()
            sync()
        optimizer.zero_grad(set_to_none=True)
    for handle in handles:
        handle.remove()
    prof.export_chrome_trace(str(out / "trace.json"))
    events = []
    for e in prof.key_averages(group_by_input_shape=True):
        events.append({"name": e.key, "count": e.count, "input_shapes": e.input_shapes,
                       "cpu_us": e.cpu_time_total, "self_cpu_us": e.self_cpu_time_total,
                       "device_us": e.device_time_total, "self_device_us": e.self_device_time_total})
    write(out / "operators.json", events)
    write(out / "expert-rows.json", rows)
    # Count original forward rows independently from checkpoint recomputation.
    row_totals = collections.defaultdict(int)
    for row in rows:
        if row["phase"] == "forward":
            row_totals[row["module"].split(".mlp")[0]] += row["shape"][0]
    if variant == "moe":
        assert len(row_totals) == 6 and all(n == 2048 for n in row_totals.values())
    routing = model.routing_report()
    assert routing["router_gradients_finite"] and routing["train"]["dropped_tokens"] == 0
    if variant == "moe":
        assert all(g > 0 for g in routing["router_gradient_max_abs"])
    for name, value in model.state_dict().items():
        assert torch.equal(value.cpu(), state[name]), f"Frozen checkpoint changed: {name}"
    executed = {}
    for module in (prepare, train, model_module):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(HERE / "source")
        executed[path.relative_to(HERE).as_posix()] = digest(path)
    result = {"variant": variant, "platform": platform.platform(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "config": vars(config), "amp_dtype": str(runtime.amp_dtype),
              "attention_backend": runtime.attention_backend,
              "checkpoint_sha256": manifest["checkpoint_hashes"][variant],
              "executed_files": executed, "batch_sha256": manifest["bank_sha256"],
              "frozen_weights_verified": True, "primary": primary, "primary_peak_bytes": primary_memory,
              "phases_seconds_per_update": {k: summary(v) for k, v in phase_samples.items()},
              "stock_cuda_loader_seconds_per_microbatch": {"cold_first": loader_times[0],
                  "steady": summary(loader_times[1:]), "matches_replay_bank": True},
              "parameters": model.parameter_report(), "optimizer_groups": optimizer_groups,
              "routing": routing, "trace_forward_rows_per_layer": dict(row_totals),
              "trace_note": "Two microbatches plus one optimizer; diagnostic overhead excluded from primary throughput",
              "nonzero_lr_updates": 0}
    write(out / "result.json", result)
    print(f"PASS: {variant}; fixed checkpoint, identical inputs, finite gradients, no dropped tokens.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=list(RUNS))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        prepare_probe()

