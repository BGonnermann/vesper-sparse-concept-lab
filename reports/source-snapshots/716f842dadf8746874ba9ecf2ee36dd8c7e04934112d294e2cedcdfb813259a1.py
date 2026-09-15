"""Three paired uninstrumented on/off measurements; standalone memory passes."""
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
variant = sys.argv[1]
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def write(p, v): p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n", encoding="utf-8")
def digest(p):
    with p.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()
def summary(a):
    return {"samples": a, "mean": statistics.mean(a), "min": min(a), "max": max(a)}
manifest = read(HERE / "probe-manifest.json")
for rel, sha in manifest["files"].items(): assert digest(HERE / rel) == sha, rel
checkpoint = ROOT / manifest["checkpoints"][variant]["path"]
assert digest(checkpoint) == manifest["checkpoints"][variant]["sha256"]
os.environ["AUTORESEARCH_CACHE_DIR"] = str(ROOT / ".autoresearch/cache")
os.environ["AUTORESEARCH_DATASET"] = "tinystories"
sys.path[:0] = [str(HERE / "source/project"), str(HERE / "source/upstream")]
import torch
import prepare
prepare.MAX_SEQ_LEN = 512
import train
import autoresearch_model
train.WINDOW_PATTERN = "L"
runtime = train.detect_runtime()
train._configure_step_kernels(runtime)
assert "5070 Ti" in torch.cuda.get_device_name()
assert runtime.amp_dtype == torch.bfloat16
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
candidate = read(HERE / (variant + ".json"))
protocols = {mode: read(HERE / ("protocol-" + mode + ".json")) for mode in ("on", "off")}
assert {k:v for k,v in protocols["on"].items() if k != "activation_checkpointing"} == {
    k:v for k,v in protocols["off"].items() if k != "activation_checkpointing"}
config = train.build_model_config(6, 8192, runtime, use_activation_checkpointing=True)
with torch.device("meta"):
    model = autoresearch_model.model_class(train, candidate)(config)
model.to_empty(device="cuda")
model.init_weights(embed_dtype=runtime.amp_dtype)
state = torch.load(checkpoint, weights_only=True, map_location="cpu")
model.load_state_dict(state, strict=True)
model.train()
optimizer = model.setup_optimizer(unembedding_lr=train.UNEMBEDDING_LR, embedding_lr=train.EMBEDDING_LR,
    scalar_lr=train.SCALAR_LR, adam_betas=train.ADAM_BETAS, matrix_lr=.04, weight_decay=train.WEIGHT_DECAY)
for group in optimizer.param_groups:
    group["lr"] = 0.
    if group["kind"] == "muon": group["momentum"] = .95
bank = torch.load(HERE / "batches.pt", weights_only=True).pin_memory()
gpu = torch.empty_like(bank[0], device="cuda")
def sync(): torch.cuda.synchronize()
def transfer(i):
    gpu.copy_(bank[i % len(bank)], non_blocking=True)
    return gpu[0], gpu[1]
def mode(name):
    model.config.use_activation_checkpointing = protocols[name]["activation_checkpointing"]
    assert model.parameter_report()["activation_checkpointing"] is (name == "on")
def forward(x, y=None):
    with torch.autocast("cuda", dtype=torch.bfloat16): return model(x, y)
def update(offset):
    for j in range(16):
        x, y = transfer(offset+j)
        loss = forward(x, y)
        (loss / 16).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss

print(f"{variant}: saved depth-6 checkpoint on/off output and all-gradient check", flush=True)
checks = []
for name in ("on", "off"):
    mode(name)
    x, y = transfer(0)
    with torch.no_grad(): logits = forward(x).cpu()
    loss = forward(x, y)
    loss.backward()
    checks.append((logits, loss.detach().cpu(), {n:p.grad.detach().cpu() for n,p in model.named_parameters()},
                   model.routing_report()["train"]["counts"]))
    optimizer.zero_grad(set_to_none=True)
torch.testing.assert_close(checks[0][0], checks[1][0], atol=1e-6, rtol=1e-5)
torch.testing.assert_close(checks[0][1], checks[1][1], atol=1e-6, rtol=1e-5)
max_gradient_error = 0.
for name, a in checks[0][2].items():
    b = checks[1][2][name]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    assert torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
    max_gradient_error = max(max_gradient_error, (a-b).abs().max().item())
# Counts are cumulative in one model: second pass must add the same assignments.
assert all([2*n for n in a] == b for a,b in zip(checks[0][3], checks[1][3]))
del checks, logits, loss, a, b
print(f"PASS correctness; maximum gradient absolute difference={max_gradient_error}", flush=True)
gc.collect(); gc.disable()
for name in ("on", "off"):
    mode(name)
    print(f"Warmup: checkpoint {name}, three updates", flush=True)
    for i in range(3): update(i*16)
sync()
samples = {"on": [], "off": []}
paired = []
routing_counts = {name: [[0]*4 for _ in range(6)] if variant == "moe" else [] for name in samples}
for pair in range(3):
    order = ("on", "off") if pair % 2 == 0 else ("off", "on")
    times = {}
    for name in order:
        mode(name)
        before = model.routing_report()["train"]["counts"]
        values = []
        for i in range(2):
            sync(); start = time.perf_counter()
            loss = update(pair*32+i*16)
            sync(); values.append(time.perf_counter()-start)
        assert torch.isfinite(loss).item()
        after = model.routing_report()["train"]["counts"]
        for l, (aa, bb) in enumerate(zip(before,after)):
            for e in range(4): routing_counts[name][l][e] += bb[e]-aa[e]
        samples[name].extend(values)
        times[name] = sum(values)
    paired.append({"pair": pair, "order": order, "seconds": times,
                   "off_throughput_gain": times["on"]/times["off"]-1})
    print(json.dumps(paired[-1]), flush=True)
assert routing_counts["on"] == routing_counts["off"]
memory = {}
print("Separate standalone memory passes, not throughput samples", flush=True)
for name in ("on", "off"):
    mode(name)
    optimizer.zero_grad(set_to_none=True)
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    sync()
    torch.cuda.reset_peak_memory_stats()
    free_samples = []
    for i in range(5):  # Three warmups plus two memory-probe updates.
        loss = update(i*16)
        sync()
        free, total = torch.cuda.mem_get_info()
        free_samples.append(free)
    memory[name] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "minimum_free_bytes_at_update_boundaries": min(free_samples),
                    "physical_bytes": total,
                    "scope": "Single model and warmed optimizer; unused cache released before this condition; memory pass excluded from throughput"}
    print(name, json.dumps(memory[name]), flush=True)
for key, value in model.state_dict().items():
    assert torch.equal(value.cpu(), state[key]), key
routing = model.routing_report()
assert routing["router_gradients_finite"] and routing["train"]["dropped_tokens"] == 0
if variant == "moe": assert all(v > 0 for v in routing["router_gradient_max_abs"])
gain = sum(samples["on"])/sum(samples["off"])-1
comfortable = (memory["off"]["peak_reserved_bytes"] < .75*memory["off"]["physical_bytes"]
               and memory["off"]["minimum_free_bytes_at_update_boundaries"] >= .25*memory["off"]["physical_bytes"])
gate = comfortable and gain >= .05 and all(p["off_throughput_gain"] > 0 for p in paired)
result = {"variant": variant, "correctness_passed": True, "max_gradient_error": max_gradient_error,
          "pairs": paired, "off_throughput_gain": gain, "comfortable_memory": comfortable,
          "gate_passed": gate, "weights_unchanged": True, "memory": memory,
          "modes": {name: {"protocol": protocols[name], "update_seconds": summary(values),
                            "timed_tokens": 98304, "timed_seconds": sum(values),
                            "tokens_per_second": 98304/sum(values),
                            "timed_routing_counts": routing_counts[name]}
                    for name,values in samples.items()},
          "parameters": model.parameter_report(), "routing": routing,
          "precision": str(runtime.amp_dtype), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
          "checkpoint_sha256": manifest["checkpoints"][variant]["sha256"],
          "batch_sha256": manifest["files"]["batches.pt"],
          "nonzero_lr_updates": 0,
          "limitations": "Three short pairs; same saved weights, fresh warmed optimizer; no profiler; replay omits CPU packing; background GPU load/clocks uncontrolled. Free VRAM sampled at update boundaries."}
for rel, sha in manifest["files"].items(): assert digest(HERE / rel) == sha, rel
write(HERE / (variant + "-probe-result.json"), result)
print(f"COMPLETE {variant}: off gain={100*gain:.2f}%, comfortable={comfortable}, gate={gate}", flush=True)

