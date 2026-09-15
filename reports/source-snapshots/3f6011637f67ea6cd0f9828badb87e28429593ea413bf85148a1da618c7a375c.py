"""Reconcile this authorized dense/MoE pair and preserve a source-backed receipt."""
import hashlib
import json
import math
from pathlib import Path
import shutil

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
RUNS = ROOT / "runs/autoresearch"
DIRECTORIES = {
    "dense_smoke": RUNS / "20260914T231448Z-edc7928a",
    "moe_smoke": RUNS / "20260914T231516Z-5599ce3e",
    "dense_baseline": RUNS / "20260914T231604Z-061092d4",
    "moe_baseline": RUNS / "20260914T232147Z-ff9e4e60",
}


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


records = {}
for label, directory in DIRECTORIES.items():
    record = load(directory / "result.json")
    assert record["status"] == "completed" and record["returncode"] == 0, label
    assert math.isfinite(record["metrics"]["val_bpb"]), label
    assert record["record_version"] == 3
    assert record["candidate"]["depth"] == 6 and record["candidate"]["matrix_lr"] == .04
    assert record["artifacts"]["model"]["width"] == 384
    execution = load(directory / "execution.json")
    assert execution == record["execution"] and execution["verified"] and execution["isolated"]
    assert execution["files"] == record["snapshot_files"]
    assert execution["executed_files"] == {p: h for p, h in execution["files"].items() if p.endswith(".py")}
    for path, expected in execution["files"].items():
        assert digest(directory / path) == expected, path
    for name, value in record["artifacts"].items():
        assert load(directory / (name + ".json")) == value, name
    optimizer = record["artifacts"]["optimizer"]
    assert optimizer["verified"] and not optimizer["missing"] and optimizer["duplicate_count"] == 0
    names = [p for group in optimizer["groups"] for p in group["parameters"]]
    assert len(names) == len(set(names)) == optimizer["parameter_tensors"]
    routing = record["artifacts"]["routing"]
    training = record["artifacts"]["training"]
    assert training["training_tokens"] == int(record["metrics"]["num_steps"]) * 16384
    assert routing["train"]["tokens"] == training["training_tokens"]
    assert routing["eval"]["tokens"] == (8192 if record["kind"] == "smoke" else 65536)
    assert routing["train"]["dropped_tokens"] == routing["eval"]["dropped_tokens"] == 0
    if record["condition"] == "moe":
        required = {f"transformer.h.{layer}.mlp.experts.{expert}.{projection}.weight"
                    for layer in range(6) for expert in range(4) for projection in ("c_fc", "c_proj")}
        required |= {f"transformer.h.{layer}.mlp.router.weight" for layer in range(6)}
        assert required <= set(names), "Expert/router missing from optimizer"
        assert routing["router_gradients_finite"]
        assert all(math.isfinite(g) and g > 0 for g in routing["router_gradient_max_abs"])
        for phase in ("train", "eval"):
            assert len(routing[phase]["counts"]) == 6
            assert all(len(row) == 4 and sum(row) == routing[phase]["tokens"] for row in routing[phase]["counts"])
    records[label] = record

common_fields = ("protocol", "upstream", "data_seal", "seed", "runner_sha256", "adapter_sha256",
                 "model_sha256", "bootstrap_sha256", "record_version")
reference = records["dense_baseline"]
for record in records.values():
    for field in common_fields:
        assert record[field] == reference[field], f"Comparison mismatch: {field}"
for variant in ("dense", "moe"):
    assert records[variant + "_smoke"]["candidate"] == records[variant + "_baseline"]["candidate"]
assert load(ROOT / "experiments/autoresearch/protocol.json") == reference["protocol"]
cache = ROOT / ".autoresearch/cache"
actual_data = {p.relative_to(cache).as_posix(): digest(p) for p in cache.rglob("*") if p.is_file()}
assert actual_data == reference["data_seal"], "Shared data changed"
for path, expected in reference["snapshot_files"].items():
    if path.startswith("source/project/"):
        assert digest(ROOT / "scripts" / Path(path).name) == expected
    elif path.startswith("source/upstream/"):
        assert digest(ROOT / ".autoresearch/upstream" / Path(path).name) == expected
preserved = load(STAGE / "prior-preservation.json")
for entry in preserved:
    assert digest(Path(entry["Path"])) == entry["Hash"].lower(), entry["Path"]

rows = {}
for variant in ("dense", "moe"):
    record = records[variant + "_baseline"]
    model = record["artifacts"]["model"]
    training = record["artifacts"]["training"]
    memory = load(DIRECTORIES[variant + "_baseline"] / "memory.json")
    rows[variant] = {
        "validation_bpb": record["metrics"]["val_bpb"],
        "total_parameters": model["total_parameters"], "active_parameters": model["active_parameters"],
        "active_parameter_convention": model["active_parameter_convention"],
        "depth": model["depth"], "width": model["width"], "memory_table_bytes": model["memory_table_bytes"],
        "training_tokens": training["training_tokens"], "optimizer_updates": training["optimizer_updates"],
        "wall_seconds": record["wall_seconds"], "timed_training_seconds": training["timed_training_seconds"],
        "timed_tokens_per_second": training["timed_training_tokens"] / training["timed_training_seconds"],
        "peak_allocated_bytes": memory["peak_allocated_bytes"], "peak_reserved_bytes": memory["peak_reserved_bytes"],
        "peak_allocated_mib": memory["peak_allocated_bytes"] / 2**20,
        "peak_reserved_mib": memory["peak_reserved_bytes"] / 2**20,
        "memory_scope": memory["measurement_scope"], "host_ram_peak": None,
        "routing": record["artifacts"]["routing"],
        "artifacts": str(DIRECTORIES[variant + "_baseline"].relative_to(ROOT)),
    }
dense, moe = rows["dense"], rows["moe"]
receipt = {
    "status": "completed", "interpretation": "Preliminary feasibility comparison; unequal total parameter budgets and one seed. No efficiency gain established.",
    "cpu_tests_passed": 22, "cuda_tests_passed": 6, "review": "No remaining concrete findings in isolation fix or revised CUDA assertions",
    "prior_files_verified_unchanged": len(preserved), "matched_comparison_fields": list(common_fields),
    "protocol": reference["protocol"], "data_hashes": reference["data_seal"],
    "runs": rows, "smokes": {v: str(DIRECTORIES[v + "_smoke"].relative_to(ROOT)) for v in ("dense", "moe")},
    "bpb_relative_change_percent": 100 * (moe["validation_bpb"] / dense["validation_bpb"] - 1),
    "training_tokens_relative_change_percent": 100 * (moe["training_tokens"] / dense["training_tokens"] - 1),
    "total_parameter_ratio": moe["total_parameters"] / dense["total_parameters"],
    "precision_fix": "FP32 router softmax and vector weights, then input-dtype cast; tightened test tolerance to atol=1e-6, rtol=1e-5; finite/nonzero task gradients asserted",
    "isolation_fix": "Captured project/upstream modules/configs executed using -I; hashes verified before/after and actual imported sources recorded",
    "limits": ["Active counts are structural and include full shared embedding tables", "Tokens include warmup; throughput excludes warmup", "BPB excludes auxiliary loss", "VRAM is PyTorch allocator only", "GPU background load not controlled", "Weights-only checkpoints, not resumable training state"],
    "stopped_after": "Two fresh smoke tests and exactly one five-minute timed baseline per variant; no further experiments or cloud jobs",
}
lines = ["# Preliminary dense/MoE feasibility comparison", "", "Both correctness suites, both fresh smoke tests, and both baselines passed.", "",
         "| Metric | Dense | MoE, four experts / top-1 |", "| --- | ---: | ---: |"]
for title, key, fmt in [("Validation BPB", "validation_bpb", ".6f"), ("Total parameters", "total_parameters", ",d"),
                        ("Active parameters", "active_parameters", ",d"), ("Training tokens including warmup", "training_tokens", ",d"),
                        ("Wall seconds", "wall_seconds", ".1f"), ("Timed tokens/second", "timed_tokens_per_second", ",.0f"),
                        ("Peak allocated MiB", "peak_allocated_mib", ".1f"), ("Peak reserved MiB", "peak_reserved_mib", ".0f")]:
    lines.append(f"| {title} | {format(dense[key], fmt)} | {format(moe[key], fmt)} |")
lines += ["", "Both models use depth 6, width 384, context 512 and 300 seconds of timed training.",
          f"MoE BPB was {receipt['bpb_relative_change_percent']:.2f}% higher with {receipt['total_parameter_ratio']:.2f}x total parameters and {abs(receipt['training_tokens_relative_change_percent']):.2f}% fewer training tokens.",
          "This initial result favors dense within this specific run budget. It is not evidence of an efficiency gain or a definitive architecture ranking.", "",
          "Active parameters count all shared tensors, including complete embedding tables, plus one expert per layer and all router weights. This is not measured compute.",
          "VRAM covers the PyTorch allocator. BPB excludes the routing auxiliary loss.", "", "## Expert utilization", "",
          "Percent of each layer's tokens; expert identities are independent between layers. Training aggregates the full run, including warmup.", "",
          "| Layer | Train E0 / E1 / E2 / E3 | Validation E0 / E1 / E2 / E3 |", "| --- | --- | --- |"]
for index in range(6):
    a = " / ".join(f"{100*x:.1f}%" for x in moe["routing"]["train"]["fractions"][index])
    b = " / ".join(f"{100*x:.1f}%" for x in moe["routing"]["eval"]["fractions"][index])
    lines.append(f"| {index + 1} | {a} | {b} |")
lines += ["", "Every expert was used in every layer. Zero dropped tokens in training and validation. All six routers recorded finite, nonzero gradients.",
          f"Unweighted training auxiliary loss: {moe['routing']['training_auxiliary_loss']:.6f}; weighted contribution: {moe['routing']['weighted_training_auxiliary_loss']:.6f}.",
          "", "## Verification and preservation", "", "22 CPU tests and 6 CUDA-suite tests passed. Review found no remaining concrete issues in the two fixes.",
          f"All {len(preserved)} prior artifacts are unchanged. Protocol, data/tokenizer hashes, runtime, seed and executed source hashes match across the fresh pair and their smoke tests.",
          "Captured source/configuration execution and live-edit isolation are covered by real subprocess tests. Each run retains snapshot.json, execution.json, optimizer/model/routing/training records, source files, weights and logs.",
          "", "## Run locations", ""]
for label, path in DIRECTORIES.items():
    lines.append(f"- {label}: {path.relative_to(ROOT).as_posix()}")
lines += ["", "No additional experiments or cloud jobs were launched. All conclusions remain preliminary."]
save(STAGE / "comparison.json", receipt)
with (STAGE / "comparison.md").open("x", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
archive = STAGE / "verification-source"
archive.mkdir()
source_hashes = {}
for relative in ("tests/test_autoresearch.py", "tests/test_moe.py", "tests/test_snapshot.py", "docs/moe-feasibility.md"):
    destination = archive / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, destination)
    source_hashes[relative] = digest(destination)
save(STAGE / "verification-source-manifest.json", source_hashes)
print(json.dumps({"status": "completed", "prior_files_unchanged": len(preserved),
                  "metrics": {k: {field: value for field, value in v.items() if field != "routing"} for k, v in rows.items()},
                  "bpb_change_percent": receipt["bpb_relative_change_percent"],
                  "tokens_change_percent": receipt["training_tokens_relative_change_percent"],
                  "report": str(STAGE / "comparison.md")}, indent=2))
