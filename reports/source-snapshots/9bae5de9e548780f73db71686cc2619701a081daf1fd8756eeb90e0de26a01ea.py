"""Prepare logs/snapshots before explicitly launching one gated local trial."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import autoresearch as runner

def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def output(variant, kind):
    return ROOT / "runs/autoresearch" / ("packed-20260915T002151Z-" + variant + "-" + kind)

def prepare(variant, kind):
    probe = read(HERE / "probe-result.json")
    assert probe["correctness_passed"] and probe["performance_gate_passed"]
    manifest = read(HERE / "probe-manifest.json")
    assert runner.digest(ROOT / "scripts/autoresearch_model.py") == manifest["files"]["source/project/autoresearch_model.py"]
    setup = runner.verify_runtime()
    data = read(ROOT / ".autoresearch/data-seal.json")
    runner.verify_seal(ROOT / ".autoresearch/cache", data)
    candidate = runner.validate_candidate(read(HERE / (variant + ".json")))
    protocol = runner.validate_protocol(read(HERE / "protocol.json"))
    assert protocol["training_seconds"] == 300
    assert protocol == read(ROOT / "experiments/autoresearch/protocol.json")
    metadata = {"kind": kind, "condition": candidate["feedforward"], "record_version": 3,
                "candidate": candidate, "protocol": protocol, "upstream": setup, "data_seal": data,
                "seed": 42, "decision": "unreviewed", "git_commit": None, "git_dirty": None,
                "git_metadata_note": "Not queried; executed file SHA256 identities are authoritative.",
                "packed_probe_sha256": runner.digest(HERE / "probe-result.json"),
                "orchestrator_sha256": runner.digest(Path(__file__))}
    metadata.update({key: runner.digest(ROOT / "scripts" / name)
                     for name, key in runner.PROJECT_FILES.items()})
    if kind == "baseline":
        runner.require_smoke(ROOT / "runs/autoresearch", runner.smoke_fingerprint(metadata))
    out = output(variant, kind)
    out.mkdir(exist_ok=False)
    snapshots = runner.capture_run_snapshot(out, metadata)
    metadata.update(status="prepared", metrics=None, returncode=None,
                    snapshot_files=snapshots,
                    command=runner.snapshot_command(runner.runtime_python(), out, kind == "smoke"),
                    timeout_seconds=protocol["smoke_timeout_seconds" if kind == "smoke" else "baseline_timeout_seconds"])
    runner.write_json(out / "result.json", metadata)
    (out / "run.log").touch(exist_ok=False)
    print("Prepared log (no training launched): " + str(out / "run.log"), flush=True)

def launch(variant, kind):
    out = output(variant, kind)
    record = read(out / "result.json")
    assert record["status"] == "prepared", "Run already launched; never reuse artifacts."
    assert runner.digest(Path(__file__)) == record["orchestrator_sha256"]
    assert runner.digest(HERE / "probe-result.json") == record["packed_probe_sha256"]
    runner.verify_runtime()
    runner.verify_seal(ROOT / ".autoresearch/cache", record["data_seal"])
    for rel, sha in record["snapshot_files"].items():
        assert runner.digest(out / rel) == sha, rel
    if kind == "baseline":
        runner.require_smoke(ROOT / "runs/autoresearch", runner.smoke_fingerprint(record))
    record.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    runner.write_json(out / "result.json", record)
    start = time.monotonic()
    process = None
    try:
        with (out / "run.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(record["command"], cwd=out, env=runner.environment(),
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            print(f"Started {variant} {kind}; PID {process.pid}; log {out / 'run.log'}", flush=True)
            try:
                record["returncode"] = process.wait(timeout=record["timeout_seconds"])
            except subprocess.TimeoutExpired:
                runner.stop_process(process)
                record.update(status="timeout", returncode=process.returncode)
        if record["status"] == "running":
            if record["returncode"] != 0:
                record["status"] = "failed"
            else:
                record["metrics"] = runner.parse_summary((out / "run.log").read_text(encoding="utf-8"))
                record["artifacts"] = runner.validate_run_artifacts(out, record)
                record["execution"] = runner.validate_execution(out, record["snapshot_files"])
                record["status"] = "completed"
    except BaseException as exc:
        if process is not None and process.poll() is None:
            runner.stop_process(process)
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - start
        record["wall_scope"] = "Launch through validation of result artifacts; preparation recorded separately."
        runner.write_json(out / "result.json", record)
    print(json.dumps({"status": record["status"], "wall_seconds": record["wall_seconds"],
                      "metrics": record["metrics"], "result": str(out / "result.json")}), flush=True)
    return 0 if record["status"] == "completed" else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "launch"))
    parser.add_argument("variant", choices=("dense", "moe"))
    parser.add_argument("kind", choices=("smoke", "baseline"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.variant, args.kind)
    else:
        raise SystemExit(launch(args.variant, args.kind))

