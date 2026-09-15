"""Finalize only after the two smoke gates and two authorized baselines complete."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))
def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def write(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False) + "\n", encoding="utf-8")

probe = read(HERE / "probe-result.json")
assert probe["correctness_passed"] and probe["performance_gate_passed"]
baselines, smokes = {}, {}
for variant in ("dense", "moe"):
    for kind, collection in (("smoke", smokes), ("baseline", baselines)):
        folder = ROOT / "runs/autoresearch" / f"packed-20260915T002151Z-{variant}-{kind}"
        r = read(folder / "result.json")
        assert r["status"] == "completed" and r["returncode"] == 0
        assert r["execution"]["verified"] and r["execution"]["isolated"]
        assert r["execution"]["files"] == r["snapshot_files"]
        for rel, sha in r["snapshot_files"].items():
            assert digest(folder / rel) == sha, (folder, rel)
        collection[variant] = r
d, m = baselines["dense"], baselines["moe"]
fields = ("protocol", "upstream", "data_seal", "seed", "runner_sha256", "adapter_sha256",
          "model_sha256", "bootstrap_sha256", "record_version", "orchestrator_sha256",
          "packed_probe_sha256")
for f in fields:
    assert d[f] == m[f], f
for rel, sha in d["data_seal"].items():
    assert digest(ROOT / ".autoresearch/cache" / rel) == sha, rel
for name, r in baselines.items():
    for f in fields:
        assert r[f] == smokes[name][f], (name, f)
    assert r["candidate"] == smokes[name]["candidate"]
    assert r["candidate"]["depth"] == 6 and r["candidate"]["matrix_lr"] == .04
    assert r["protocol"]["training_seconds"] == 300
    assert r["artifacts"]["routing"]["train"]["dropped_tokens"] == 0
    assert r["artifacts"]["routing"]["eval"]["dropped_tokens"] == 0

training = {}
for name, r in baselines.items():
    t, model, routing = (r["artifacts"][k] for k in ("training", "model", "routing"))
    folder = Path(r["command"][4]).parent if False else ROOT / "runs/autoresearch" / f"packed-20260915T002151Z-{name}-baseline"
    memory = read(folder / "memory.json")
    assert t["training_tokens"] == t["optimizer_updates"] * 16384
    assert t["warmup_updates"] == 11
    assert t["timed_training_tokens"] == (t["optimizer_updates"] - 11) * 16384
    assert t["timed_training_seconds"] >= 300
    assert routing["train"]["tokens"] == t["training_tokens"]
    for layer in routing["train"]["counts"]:
        assert sum(layer) == t["training_tokens"]
    assert math.isfinite(r["metrics"]["val_bpb"])
    training[name] = {
        "validation_bpb": r["metrics"]["val_bpb"],
        "training_tokens": t["training_tokens"], "optimizer_updates": t["optimizer_updates"],
        "timed_tokens": t["timed_training_tokens"], "timed_seconds": t["timed_training_seconds"],
        "timed_tokens_per_second": t["timed_training_tokens"] / t["timed_training_seconds"],
        "wall_seconds": r["wall_seconds"], "parameters": model, "memory": memory,
        "allocated_mib": memory["peak_allocated_bytes"] / 2**20,
        "reserved_mib": memory["peak_reserved_bytes"] / 2**20,
        "routing": routing, "path": str(folder.relative_to(ROOT)),
    }
prior = read(HERE / "prior-preservation.json")
for entry in prior:
    assert digest(Path(entry["Path"])).upper() == entry["Hash"].upper(), entry["Path"]
original, packed = probe["variants"]["original"], probe["variants"]["packed"]
review = "Independent read-only review: no concrete correctness findings; residual host boundary transfer and small per-expert GEMMs noted."
result = {"status": "complete", "cpu_tests_passed": 24, "cuda_tests_passed": 8,
          "test_first_expected_failure": "Original dispatch made four allocations, not one",
          "gradient_max_absolute_error_depth6_checkpoint": probe["full_checkpoint_max_gradient_error"],
          "review": review, "preserved_prior_files": len(prior),
          "matched_training_fields": fields, "training": training,
          "profile": {"paired_gain": probe["aggregate_throughput_gain"],
                      "all_six_pairs_improved": all(b["throughput_gain"] > 0 for b in probe["paired_blocks"]),
                      "original_tps": original["tokens_per_second"], "packed_tps": packed["tokens_per_second"],
                      "original_trace": original["trace"], "packed_trace": packed["trace"]},
          "limits": ["Short paired profiling, unchanged checkpoint weights, no BPB in probe",
                     "One seed per fresh baseline; unequal dense/MoE parameter budgets",
                     "Timed throughput excludes 11 warmup updates; total tokens include them",
                     "Profiler/phase decomposition excluded from replay throughput",
                     "Both models resident for replay; its memory is joint, not standalone",
                     "Background GPU load/clocks not controlled",
                     "BPB contains cross-entropy only, excludes routing auxiliary loss"],
          "stopped_after": "One dense baseline and one optimized MoE baseline following correctness, paired timing and smoke gates; no cloud or additional features"}
write(HERE / "comparison.json", result)
td, tm = training["dense"], training["moe"]
rows = []
for k, label, form in [
    ("validation_bpb", "Validation BPB", ".6f"),
    ("timed_tokens_per_second", "Timed training tokens/s", ",.1f"),
    ("training_tokens", "Total tokens including warmup", ","),
    ("timed_tokens", "Timed tokens excluding warmup", ","),
    ("optimizer_updates", "Optimizer updates", ","),
    ("timed_seconds", "Timed training seconds", ".3f"),
    ("wall_seconds", "Launch-to-record wall seconds", ".3f"),
    ("allocated_mib", "Peak allocated VRAM, MiB", ".2f"),
    ("reserved_mib", "Peak reserved VRAM, MiB", ".2f")]:
    rows.append(f"| {label} | {format(td[k], form)} | {format(tm[k], form)} |")
for k in ("total_parameters", "active_parameters"):
    rows.append(f"| {k} | {td['parameters'][k]:,} | {tm['parameters'][k]:,} |")
phase_rows = []
for name in ("data", "forward", "backward", "optimizer", "zero_grad"):
    phase_rows.append(f"| {name} | {original['phases'][name]['mean']*1000:.2f} | {packed['phases'][name]['mean']*1000:.2f} |")
util = []
for i, (tr, ev) in enumerate(zip(tm["routing"]["train"]["fractions"], tm["routing"]["eval"]["fractions"])):
    util.append(f"| {i} | " + " / ".join(f"{100*x:.2f}%" for x in tr) + " | " +
                " / ".join(f"{100*x:.2f}%" for x in ev) + " |")
block_range = [100*b["throughput_gain"] for b in probe["paired_blocks"]]
report = f"""# Packed expert dispatch: correctness, profiling and training

## Outcome

Packed dispatch passed correctness and the preregistered timing gate: **{100*probe['aggregate_throughput_gain']:.2f}% higher replay throughput**, faster in all six paired blocks (gains {min(block_range):.2f}%-{max(block_range):.2f}%). Fresh dense and optimized MoE smoke gates passed, followed by exactly one 300-second baseline each.

This is an implementation speedup, **not proof that MoE is more efficient than dense**. Dense and MoE have unequal total parameter budgets and only one fresh seed/run each. Training quality conclusions are preliminary.

## Implementation and correctness

Only production feedforward dispatch and its recording label changed in @@scripts/autoresearch_model.py@@. Stable sorting packs tokens once, four expert calls consume contiguous slices (including empty slices), and one index-copy restores original order. A fixed-size histogram and one boundary transfer replace repeated dynamic nonzero/bincount paths.

Architecture, attention, expert selection/tie behavior, within-expert row order, full-softmax gate weights, FP32 router/auxiliary math, BF16 expert execution, output cast, auxiliary coefficient, optimizer policy, tokenizer, data and training settings remain unchanged.

- Test-first allocation regression failed as expected on original dispatch, then passed.
- **24 CPU tests and 8 CUDA tests passed.** New cases cover balanced, uneven, empty and tied routing; FP32 and CUDA BF16 inputs; outputs, auxiliary loss, every input/parameter gradient, finite nonzero router task gradients, and checkpoint replay.
- Original-reference comparison tolerance: absolute 1e-6, relative 1e-5; no tolerance loosening.
- Full saved depth-6 checkpoint preflight: **maximum absolute gradient difference {probe['full_checkpoint_max_gradient_error']}**, matching loss and routing counts.
- Existing tests also cover causality, validation excluding auxiliary loss, optimizer coverage, empty-expert tensor gradients/updates, accounting and snapshot isolation.
- {review}

## Profiling results, not training results

Original and optimized MoE loaded the **same saved checkpoint**, with identical replay inputs, context 512, microbatch 2, 16 accumulation microbatches, BF16 and checkpointing on. All learning rates were zero; checkpoint tensors were verified unchanged.

Three warmup updates/implementation; six counterbalanced paired blocks of two updates (12 timed updates, **196,608 timed tokens/implementation**). Gate required improvement in all six pairs and at least 5% aggregate throughput gain. Profiler, phase timing instrumentation and additional module hooks were absent from throughput measurements.

| Uninstrumented replay | Original MoE | Packed MoE |
|---|---:|---:|
| Tokens/s | {original['tokens_per_second']:,.1f} | {packed['tokens_per_second']:,.1f} |
| Total timed seconds | {original['seconds']:.4f} | {packed['seconds']:.4f} |
| Mean update ms | {original['updates_seconds']['mean']*1000:.2f} | {packed['updates_seconds']['mean']*1000:.2f} |

Separate trace: two microbatches plus one optimizer step per implementation:

| Trace observation | Original | Packed |
|---|---:|---:|
| Actual kernel launches | {original['trace']['kernel_count']:,} | {packed['trace']['kernel_count']:,} |
| Stream synchronizations | {original['trace']['stream_sync_count']} | {packed['trace']['stream_sync_count']} |
| Stream sync duration, ms | {original['trace']['stream_sync_ms']:.3f} | {packed['trace']['stream_sync_ms']:.3f} |
| Nonzero calls | {original['trace']['operator_counts'].get('aten::nonzero',0)} | {packed['trace']['operator_counts'].get('aten::nonzero',0)} |
| Bincount calls | {original['trace']['operator_counts'].get('aten::bincount',0)} | {packed['trace']['operator_counts'].get('aten::bincount',0)} |

Separate synchronized phase means (three updates each), ms/update:

| Phase | Original | Packed |
|---|---:|---:|
{chr(10).join(phase_rows)}

These phase/trace passes contain additional instrumentation/synchronization and **do not define throughput**. Replay omits CPU tokenization/packing; transfer is included. Both models and warmed optimizers were resident together: raw joint allocator peaks and increments are in @@probe-result.json@@, not misreported as standalone model VRAM. The training table below provides standalone VRAM.

Original and packed probe routing counts match exactly, all tokens are covered and both checkpoints remain unchanged. Per-expert utilization for both identical replay streams is preserved in @@probe-result.json@@.

## Fresh training results

Both: depth 6, width 384, context 512, 16,384 tokens/update, microbatch 2, checkpointing enabled, seed 42, matrix LR 0.04, existing **300-second** training protocol. MoE: four experts/top-1, auxiliary coefficient 0.01, router LR 0.001. No NCP or memory tables.

Protocol, data/tokenizer hashes, runtime, seed, runner/adapter/model/bootstrap hashes and actual executed source snapshots match across fresh runs. Candidate differences are only the established dense/MoE architecture fields. Each run executed its captured snapshot; smoke gates match the baseline identities.

| Metric | Fresh dense | Fresh packed MoE |
|---|---:|---:|
{chr(10).join(rows)}

BPB is token cross-entropy only, excluding routing auxiliary loss. Total tokens include 11 warmup updates; timed throughput uses only timed tokens/time. Wall time covers launch through result validation, excluding separate snapshot preparation. VRAM is PyTorch allocator peak, not total device usage or host RAM.

### Expert utilization

Percent of tokens assigned to experts 0 / 1 / 2 / 3, separately by layer:

| Layer | Training | Validation |
|---|---|---|
{chr(10).join(util)}

Zero dropped tokens in training and validation; every router gradient finite/nonzero. Training mean auxiliary loss: **{tm['routing']['training_auxiliary_loss']:.6f}**, weighted contribution **{tm['routing']['weighted_training_auxiliary_loss']:.6f}**, recorded separately from BPB. Dense has no experts.

## Remaining bottlenecks and interpretation

Backward, including full-block activation-checkpoint recomputation, remains the largest packed phase. Packing removed repeated dynamic dispatch paths but retained one host boundary transfer per layer invocation, four Python expert calls and small expert GEMMs. Checkpointing repeats that work. Optimizer work across all 48 expert matrices remains; active-parameter counts do not describe that cost.

No skipped experts/tokens, reduced precision, reduced context, altered training budget or changed architecture produced the measured speedup. Further improvements would need their own authorization and correctness/timing gates. No further optimization is included here.

Short sequential timing samples and baseline results can vary with background GPU load/clocks. Saved-weight replay does not measure learning quality; the fresh baseline table does. Historical original-MoE BPB is not a paired learning-quality control for this run.

## Preservation and artifacts

**{len(prior)} prior files verified unchanged**, including depth-4/6/8 results, the first MoE campaign and all earlier profiling artifacts.

- @@tests-red.log@@, @@tests-cpu.log@@, @@tests-cuda.log@@ and archived original/test source.
- @@probe.log@@, @@probe-manifest.json@@, @@probe-result.json@@, captured @@source/@@ and @@batches.pt@@.
- @@original-trace.json@@, @@packed-trace.json@@ and operator summaries.
- @@baseline-matching.json@@ and @@comparison.json@@: matched identities and complete metrics.
- Dense baseline: @@{td['path']}@@.
- Packed baseline: @@{tm['path']}@@.
- Each smoke/baseline retains @@run.log@@, result/config/source receipts, memory, routing and checkpoint artifacts.
- @@paired_probe.py@@, @@gated_trial.py@@ and @@final_report.py@@ reproduce the bounded procedure. Source SHA256s are authoritative; Git metadata was deliberately not queried.

All run logs were created and their exact PowerShell tail commands provided before launch. **Stopped after this report: no further architecture features, experiments or cloud jobs.**
"""
(HERE / "report.md").write_text(report.replace("@@", chr(96)), encoding="utf-8")
write(HERE / "completion.json", {"status": "complete", "prior_files_unchanged": len(prior),
                               "report_sha256": digest(HERE / "report.md"),
                               "baseline_count": 2, "smoke_count": 2, "cloud_jobs": 0})
print(json.dumps({k: {name: value for name, value in v.items() if name not in ("routing", "parameters", "memory")} for k, v in training.items()}, indent=2))
print("Expert training fractions:",tm["routing"]["train"]["fractions"])
print("PASS: complete; preserved",len(prior),"prior files.")

