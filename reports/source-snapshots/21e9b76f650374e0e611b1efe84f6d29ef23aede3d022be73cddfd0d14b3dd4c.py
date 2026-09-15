"""CPU-only analysis of the completed bounded profile artifacts."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))
def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
results = {v: read(HERE / v / "result.json") for v in ("dense", "moe")}
manifest = read(HERE / "manifest.json")
d, m = results["dense"], results["moe"]
for field in ("config", "amp_dtype", "attention_backend", "executed_files", "batch_sha256"):
    assert d[field] == m[field], field
for rel, sha in manifest["source_hashes"].items():
    assert digest(HERE / rel) == sha, rel
assert digest(HERE / "batches.pt") == d["batch_sha256"] == manifest["bank_sha256"]
for v, r in results.items():
    assert r["frozen_weights_verified"] and r["nonzero_lr_updates"] == 0
    assert r["primary"]["tokens"] == 98304
    assert r["routing"]["router_gradients_finite"] and r["routing"]["train"]["dropped_tokens"] == 0

analysis = {"matched_fields": ["config", "amp_dtype", "attention_backend", "executed_files", "batch_sha256"],
            "source_and_batch_hashes_reverified": True,
            "preservation": read(HERE / "preservation-result.json"),
            "variants": {}, "phase_deltas_ms": {}}
for name in d["phases_seconds_per_update"]:
    analysis["phase_deltas_ms"][name] = 1000 * (
        m["phases_seconds_per_update"][name]["mean"] - d["phases_seconds_per_update"][name]["mean"])
for v, r in results.items():
    events = read(HERE / v / "operators.json")
    agg = {}
    for e in events:
        a = agg.setdefault(e["name"], {"name": e["name"], "count": 0, "cpu_us": 0,
                                     "self_cpu_us": 0, "device_us": 0, "self_device_us": 0})
        for k in ("count", "cpu_us", "self_cpu_us", "device_us", "self_device_us"):
            a[k] += e[k]
    rows = read(HERE / v / "expert-rows.json")
    row_stats = {}
    for phase in ("forward", "backward"):
        rr = [x for x in rows if x["phase"] == phase]
        row_stats[phase] = {"calls": len(rr), "total_rows": sum(x["shape"][0] for x in rr),
                           "min_rows": min((x["shape"][0] for x in rr), default=None),
                           "max_rows": max((x["shape"][0] for x in rr), default=None),
                           "empty_calls": sum(x["shape"][0] == 0 for x in rr)}
    module_totals = collections.defaultdict(lambda: collections.defaultdict(float))
    for e in events:
        n = e["name"]
        if n.startswith("module/"):
            typ = ("expert" if ".experts." in n else "router" if ".router/" in n
                   else "ffn" if ".mlp/" in n else "attention")
            phase = n.split("/")[-1]
            for k in ("cpu_us", "device_us", "count"):
                module_totals[typ + "/" + phase][k] += e[k]
    trace = read(HERE / v / "trace.json")
    gpu = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
    runtime = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"]
    sync_ancestors = collections.defaultdict(lambda: {"count": 0, "us": 0.0})
    host = [e for e in trace["traceEvents"] if e.get("ph") == "X"
            and e.get("cat") in ("cpu_op", "user_annotation")]
    # Attribute each runtime synchronization to its narrowest enclosing CPU op.
    for e in runtime:
        if "Synchronize" not in e["name"]:
            continue
        enclosing = [p for p in host if p.get("tid") == e.get("tid")
                     and p["ts"] <= e["ts"]
                     and p["ts"] + p["dur"] >= e["ts"] + e["dur"]]
        parent = min(enclosing, key=lambda p: p["dur"])["name"] if enclosing else "unattributed"
        sync_ancestors[parent]["count"] += 1
        sync_ancestors[parent]["us"] += e["dur"]
    opt_totals = collections.defaultdict(lambda: {"tensors": 0, "elements": 0, "groups": 0})
    for g in r["optimizer_groups"]:
        key = g["kind"] + "/" + g["category"]
        opt_totals[key]["groups"] += 1
        opt_totals[key]["tensors"] += g["tensors"]
        opt_totals[key]["elements"] += g["elements"]
    analysis["variants"][v] = {
        "primary": r["primary"],
        "phases_ms_per_update": {k: val["mean"] * 1000 for k, val in r["phases_seconds_per_update"].items()},
        "stock_loader_ms": {"cold_first": r["stock_cuda_loader_seconds_per_microbatch"]["cold_first"] * 1000,
            "steady": r["stock_cuda_loader_seconds_per_microbatch"]["steady"]["mean"] * 1000},
        "parameters": r["parameters"], "peak_mib": {k: n / 2**20 for k, n in r["primary_peak_bytes"].items()},
        "top_self_cpu": sorted(agg.values(), key=lambda e: e["self_cpu_us"], reverse=True)[:15],
        "top_self_device": sorted(agg.values(), key=lambda e: e["self_device_us"], reverse=True)[:12],
        "selected_ops": [e for n, e in agg.items() if n in (
            "aten::where", "aten::nonzero", "aten::bincount", "aten::index_select",
            "aten::index_add", "aten::mm", "aten::bmm", "aten::copy_", "aten::scatter_add_")
            or "Synchronize" in n or n.startswith("optimizer/") or n.startswith("phase/")],
        "module_inclusive_trace_us": dict(module_totals),
        "optimizer_parameter_work": dict(opt_totals), "expert_rows": row_stats,
        "sync_parent_trace": dict(sync_ancestors),
        "trace_kernel_count": len(gpu),
        "trace_kernel_sum_ms": sum(e["dur"] for e in gpu) / 1000,
        "gemm_shapes": [e for e in events if e["name"] in ("aten::mm", "aten::bmm")],
    }
analysis["replay_throughput_change_percent"] = 100 * (m["primary"]["tokens_per_second"] / d["primary"]["tokens_per_second"] - 1)
analysis["replay_update_time_ratio"] = m["primary"]["update_seconds"]["mean"] / d["primary"]["update_seconds"]["mean"]
(HERE / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
for v, a in analysis["variants"].items():
    print(v, json.dumps({k: val for k, val in a.items() if k not in ("gemm_shapes", "top_self_device")}, indent=2))
print("DELTA_MS", analysis["phase_deltas_ms"])

