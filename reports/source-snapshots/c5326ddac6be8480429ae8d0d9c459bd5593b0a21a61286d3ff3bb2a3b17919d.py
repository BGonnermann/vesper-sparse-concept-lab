"""Prepare immutable checkpoint-off trials; hold each log before announcing."""
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
action, variant, kind = sys.argv[1:]
assert action in ("prepare", "launch") and variant in ("dense", "moe") and kind in ("smoke", "baseline")
out = ROOT / "runs/autoresearch" / f"checkpoint-20260915T010035Z-{variant}-{kind}"
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def gate():
    for v in ("dense", "moe"):
        p = read(HERE / (v + "-probe-result.json"))
        assert p["correctness_passed"] and p["comfortable_memory"] and p["gate_passed"]
gate()
if action == "prepare":
    manifest = read(HERE / "probe-manifest.json")
    for name in runner.PROJECT_FILES:
        assert runner.digest(ROOT / "scripts" / name) == manifest["files"]["source/project/" + name]
    setup = runner.verify_runtime()
    seal = read(ROOT / ".autoresearch/data-seal.json")
    runner.verify_seal(ROOT / ".autoresearch/cache", seal)
    candidate = runner.validate_candidate(read(HERE / (variant + ".json")))
    protocol = runner.validate_protocol(read(HERE / "protocol-off.json"))
    assert protocol["activation_checkpointing"] is False and protocol["training_seconds"] == 300
    assert {k:v for k,v in protocol.items() if k != "activation_checkpointing"} == read(HERE / "prior-protocol.json")
    record = dict(kind=kind, condition=variant, record_version=3, candidate=candidate, protocol=protocol,
                  upstream=setup, data_seal=seal, seed=42, decision="unreviewed",
                  git_commit=None, git_dirty=None, git_metadata_note="Not queried; executed source hashes recorded.",
                  orchestrator_sha256=runner.digest(Path(__file__)),
                  checkpoint_probe_hashes={v:runner.digest(HERE / (v+"-probe-result.json")) for v in ("dense","moe")})
    record.update({key:runner.digest(ROOT / "scripts" / name) for name,key in runner.PROJECT_FILES.items()})
    if kind == "baseline": runner.require_smoke(ROOT / "runs/autoresearch", runner.smoke_fingerprint(record))
    out.mkdir(exist_ok=False)
    snapshots = runner.capture_run_snapshot(out, record)
    record.update(status="prepared", metrics=None, returncode=None, snapshot_files=snapshots,
                  command=runner.snapshot_command(runner.runtime_python(), out, kind=="smoke"),
                  timeout_seconds=protocol["smoke_timeout_seconds" if kind=="smoke" else "baseline_timeout_seconds"])
    runner.write_json(out / "result.json", record)
    (out / "run.log").touch(exist_ok=False)
    print(f"Prepared snapshot/log, not launched: {out}", flush=True)
else:
    record = read(out / "result.json")
    assert record["status"] == "prepared"
    assert record["orchestrator_sha256"] == runner.digest(Path(__file__))
    for v, sha in record["checkpoint_probe_hashes"].items():
        assert runner.digest(HERE / (v+"-probe-result.json")) == sha
    for rel,sha in record["snapshot_files"].items():
        assert runner.digest(out / rel) == sha
    runner.verify_runtime()
    runner.verify_seal(ROOT / ".autoresearch/cache", record["data_seal"])
    if kind == "baseline": runner.require_smoke(ROOT / "runs/autoresearch", runner.smoke_fingerprint(record))
    process = None
    start = None
    try:
        with (out / "run.log").open("a", encoding="utf-8") as log:
            print(f"LOG READY; training not started: {out / 'run.log'}", flush=True)
            deadline = time.monotonic()+180
            while not (out / "start.marker").exists():
                if time.monotonic()>deadline: raise TimeoutError("Announcement gate expired")
                time.sleep(.1)
            record.update(status="running",started_at=datetime.now(timezone.utc).isoformat())
            runner.write_json(out / "result.json", record)
            start = time.monotonic()
            process = subprocess.Popen(record["command"],cwd=out,env=runner.environment(),
                                       stdout=log,stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            print(f"Started {variant} {kind}; PID={process.pid}",flush=True)
            try: record["returncode"] = process.wait(timeout=record["timeout_seconds"])
            except subprocess.TimeoutExpired:
                runner.stop_process(process)
                record.update(status="timeout",returncode=process.returncode)
        if record["status"]=="running":
            if record["returncode"]!=0: record["status"]="failed"
            else:
                record["metrics"]=runner.parse_summary((out/"run.log").read_text(encoding="utf-8"))
                record["artifacts"]=runner.validate_run_artifacts(out,record)
                record["execution"]=runner.validate_execution(out,record["snapshot_files"])
                record["status"]="completed"
    except BaseException as exc:
        if process is not None and process.poll() is None: runner.stop_process(process)
        record.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["wall_seconds"]=time.monotonic()-start if start is not None else 0.
        record["wall_scope"]="Actual launch through result validation; announcement/preparation excluded."
        runner.write_json(out/"result.json",record)
    print(json.dumps({"status":record["status"],"wall_seconds":record["wall_seconds"],"metrics":record["metrics"]}),flush=True)
    raise SystemExit(0 if record["status"]=="completed" else 1)

