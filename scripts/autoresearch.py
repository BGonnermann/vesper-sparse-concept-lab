"""Local autoresearch setup and bounded run recording; standard library only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".autoresearch"
RUNTIME = STATE / "upstream"
CONFIG = ROOT / "experiments" / "autoresearch"
RUNTIME_FILES = ("train.py", "prepare.py", "pyproject.toml", ".python-version", "uv.lock")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def seal_files(directory):
    return {p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_seal(directory, expected):
    if seal_files(directory) != expected:
        raise ValueError("Prepared files changed. Restore the cache or explicitly prepare a new dataset seal.")


def validate_candidate(value):
    if set(value) != {"depth", "matrix_lr"}:
        raise ValueError("Candidate must contain only depth and matrix_lr.")
    if type(value["depth"]) is not int or not 2 <= value["depth"] <= 8:
        raise ValueError("depth must be an integer from 2 through 8.")
    lr = value["matrix_lr"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 0.1:
        raise ValueError("matrix_lr must be finite and in (0, 0.1].")
    return value


def validate_protocol(value):
    numeric = {"sequence_length", "tokens_per_update", "microbatch_size", "eval_batch_size",
               "eval_tokens", "smoke_eval_tokens", "training_seconds", "smoke_timeout_seconds",
               "baseline_timeout_seconds"}
    if set(value) != numeric | {"protocol_id"} or not isinstance(value["protocol_id"], str):
        raise ValueError("Protocol fields are missing or unknown.")
    if any(type(value[key]) is not int or value[key] <= 0 for key in numeric):
        raise ValueError("Protocol numeric values must be positive integers.")
    if value["tokens_per_update"] % (value["sequence_length"] * value["microbatch_size"]):
        raise ValueError("Tokens per update must be divisible by the microbatch token count.")
    for key in ("eval_tokens", "smoke_eval_tokens"):
        if value[key] % (value["sequence_length"] * value["eval_batch_size"]):
            raise ValueError("Evaluation token budget must contain complete batches.")
    if not value["training_seconds"] < value["baseline_timeout_seconds"] <= 3600:
        raise ValueError("Baseline deadline must exceed training budget and be at most 3600 seconds.")
    if value["smoke_timeout_seconds"] > 600:
        raise ValueError("Smoke deadline must be at most 600 seconds.")
    return value


def require_smoke(directory, fingerprint):
    for path in directory.glob("*/result.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if result.get("kind") == "smoke" and result.get("status") == "completed":
            if all(result.get(key) == value for key, value in fingerprint.items()):
                return
    raise ValueError("Run smoke successfully for this candidate, protocol, data, and runtime before baseline.")


def parse_summary(log):
    if re.search(r"\bloss:\s*[+-]?(?:nan|inf)\b", log, re.IGNORECASE):
        raise ValueError("Nonfinite training loss.")
    fields = {"val_bpb", "peak_vram_mb", "num_steps", "training_seconds", "total_seconds",
              "total_tokens_M", "num_params_M", "depth", "train_batch_size", "eval_batch_size"}
    metrics = {}
    for line in log.splitlines():
        key, separator, raw = line.partition(":")
        if separator and key in fields:
            if key in metrics:
                raise ValueError("Duplicate final metric: " + key)
            value = float(raw.strip())
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid final metric: " + key)
            metrics[key] = value
    if not {"val_bpb", "peak_vram_mb", "num_steps"} <= metrics.keys():
        raise ValueError("Training did not emit the required final metrics.")
    if metrics["num_steps"] < 1 or not metrics["num_steps"].is_integer():
        raise ValueError("Training did not complete an integer number of steps.")
    return metrics


def stop_process(process):
    # Kill the owned process tree; never terminate unrelated Python/GPU processes.
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass  # Always fall back to terminating the owned root process below.
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def run_trial(command, cwd, output_dir, timeout_seconds, metadata, env=None):
    output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(metadata, status="running", metrics=None, returncode=None,
                  started_at=datetime.now(timezone.utc).isoformat(), command=command,
                  timeout_seconds=timeout_seconds)
    for name in ("candidate", "protocol"):
        if name in metadata:
            write_json(output_dir / (name + ".json"), metadata[name])
    write_json(output_dir / "result.json", record)
    started = time.monotonic()
    process = None
    try:
        with (output_dir / "run.log").open("w", encoding="utf-8") as log:
            options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, **options)
            try:
                record["returncode"] = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
                stop_process(process)
                record["returncode"] = process.returncode
        if record["status"] == "running":
            if record["returncode"] != 0:
                record["status"] = "failed"
            else:
                try:
                    record["metrics"] = parse_summary((output_dir / "run.log").read_text(encoding="utf-8", errors="replace"))
                    record["status"] = "completed"
                except ValueError as error:
                    record.update(status="invalid", error=str(error))
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        if process is not None:
            stop_process(process)
            record["returncode"] = process.returncode
    except (OSError, subprocess.SubprocessError) as error:
        if record["status"] == "running":
            record["status"] = "failed"
        record["error"] = str(error)
    finally:
        record["wall_seconds"] = time.monotonic() - started
        write_json(output_dir / "result.json", record)
    return record


def checked(command, cwd=ROOT, timeout=900, capture=False, env=None):
    return subprocess.run(command, cwd=cwd, timeout=timeout, check=True, env=env,
                          text=True, encoding="utf-8", capture_output=capture)


def runtime_python():
    return RUNTIME / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def environment():
    return dict(os.environ, AUTORESEARCH_CACHE_DIR=str(STATE / "cache"),
                AUTORESEARCH_DATASET="tinystories", AUTORESEARCH_DISABLE_AUTOTUNE="1",
                UV_PROJECT_ENVIRONMENT=str(RUNTIME / ".venv"), PYTHONUNBUFFERED="1")


def setup():
    pin = json.loads((CONFIG / "upstream.json").read_text())
    for executable in ("git", "uv"):
        if shutil.which(executable) is None:
            raise ValueError(f"Install {executable} first, then rerun setup.")
    STATE.mkdir(exist_ok=True)
    if not RUNTIME.exists():
        checked(["git", "clone", "--no-checkout", pin["repository"], str(RUNTIME)])
        checked(["git", "checkout", "--detach", pin["revision"]], cwd=RUNTIME)
    actual = checked(["git", "rev-parse", "HEAD"], cwd=RUNTIME, capture=True).stdout.strip()
    if actual != pin["revision"]:
        raise ValueError("Existing upstream checkout has a different revision; it was left unchanged.")
    changed = checked(["git", "diff", "HEAD", "--", *RUNTIME_FILES[:-1]], cwd=RUNTIME, capture=True).stdout
    if changed:
        raise ValueError("Upstream code has local edits; setup will not overwrite them.")
    # The fork's checked-in uv.lock is stale against its pyproject. Resolve once,
    # then seal the actual resulting lock and use the venv directly for trials.
    checked(["uv", "sync", "--python", "3.11"], cwd=RUNTIME, env=environment())
    checked(["uv", "pip", "check", "--python", str(runtime_python())], env=environment())
    write_json(STATE / "setup.json", {"upstream": pin,
               "runtime_files": {p: digest(RUNTIME / p) for p in RUNTIME_FILES}})
    print("Setup ready. Next: doctor, then prepare, then smoke.")


def verify_runtime():
    manifest = json.loads((STATE / "setup.json").read_text())
    for path, expected in manifest["runtime_files"].items():
        if digest(RUNTIME / path) != expected:
            raise ValueError(f"Upstream runtime changed: {path}. Restore it before testing.")
    if not runtime_python().is_file():
        raise ValueError("Runtime Python is missing; run setup first.")
    return manifest


def prepare():
    verify_runtime()
    checked([str(runtime_python()), str(RUNTIME / "prepare.py"), "--dataset", "tinystories"],
            cwd=RUNTIME, timeout=1800, env=environment())
    seal = seal_files(STATE / "cache")
    if not seal:
        raise ValueError("Preparation produced no files.")
    write_json(STATE / "data-seal.json", seal)
    print("Dataset and tokenizer sealed. Next: smoke.")


def run(kind, repeat):
    manifest = verify_runtime()
    protocol = validate_protocol(json.loads((CONFIG / "protocol.json").read_text()))
    candidate = validate_candidate(json.loads((CONFIG / "candidate.json").read_text()))
    data_seal = json.loads((STATE / "data-seal.json").read_text())
    verify_seal(STATE / "cache", data_seal)
    head = checked(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    dirty = checked(["git", "status", "--porcelain"], capture=True).stdout.strip()
    metadata = {"kind": kind, "condition": "dense", "candidate": candidate, "protocol": protocol,
                "git_commit": head, "git_dirty": bool(dirty), "upstream": manifest,
                "data_seal": data_seal, "seed": 42, "decision": "unreviewed",
                "runner_sha256": digest(Path(__file__)),
                "adapter_sha256": digest(ROOT / "scripts/autoresearch_train.py")}
    if kind == "baseline":
        require_smoke(ROOT / "runs" / "autoresearch", {key: metadata[key] for key in
                      ("candidate", "protocol", "upstream", "data_seal", "runner_sha256", "adapter_sha256")})
    for _ in range(repeat):
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        output = ROOT / "runs" / "autoresearch" / tag
        command = [str(runtime_python()), "-u", str(ROOT / "scripts/autoresearch_train.py"),
                   "--runtime", str(RUNTIME), "--candidate", str(output / "candidate.json"),
                   "--protocol", str(output / "protocol.json")]
        if kind == "smoke":
            command.append("--smoke-test")
        timeout = protocol["smoke_timeout_seconds" if kind == "smoke" else "baseline_timeout_seconds"]
        print(f"Starting {kind}. Deadline {timeout}s. Log: {output / 'run.log'}", flush=True)
        result = run_trial(command, output, output, timeout, metadata, env=environment())
        print(f"{result['status']}: {output / 'result.json'}", flush=True)
        if result["metrics"]:
            print(f"Validation BPB: {result['metrics']['val_bpb']}; wall time: {result['wall_seconds']:.1f}s")
        if result["status"] != "completed":
            print("Stopped. Keep result.json and run.log for diagnosis.")
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "doctor", "prepare", "smoke", "baseline"])
    parser.add_argument("--repeat", type=int, choices=range(1, 4), default=1,
                        help="Repeat the same candidate 1–3 times; seed remains 42.")
    args = parser.parse_args()
    try:
        if args.action == "setup":
            setup()
        elif args.action == "prepare":
            prepare()
        elif args.action == "doctor":
            verify_runtime()
            checked([str(runtime_python()), str(ROOT / "scripts/autoresearch_train.py"), "--doctor"],
                    timeout=120, env=environment())
        else:
            return run(args.action, args.repeat)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print("If not set up yet, run the setup action first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
