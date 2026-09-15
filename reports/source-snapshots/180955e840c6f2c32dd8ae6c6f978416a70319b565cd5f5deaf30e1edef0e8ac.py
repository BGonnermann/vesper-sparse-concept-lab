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
PROJECT_FILES = {"autoresearch.py": "runner_sha256", "autoresearch_train.py": "adapter_sha256",
                 "autoresearch_model.py": "model_sha256", "autoresearch_bootstrap.py": "bootstrap_sha256",
                 "autoresearch_memory.py": "memory_sha256", "autoresearch_ncp.py": "ncp_sha256"}


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
    fields = {"depth", "matrix_lr", "feedforward"}
    if 'ncp' in value:
        fields.add('ncp')
        n = value['ncp']
        expected = {'kind','chunk_size','layers','entries','after_layer','prediction_weight','vq_weight','ce_weight','lr','feedback_scale','mode'}
        if not isinstance(n,dict) or set(n)!=expected or n['kind']!='ncp_v1':
            raise ValueError('NCP requires the complete ncp_v1 configuration.')
        if value.get('feedforward')!='dense' or 'memory' in value:
            raise ValueError('NCP campaign initially permits dense alone.')
        for key, allowed in [('chunk_size',(2,4,8)),('layers',(1,2)),('entries',(16,32,64,128))]:
            if type(n[key]) is not int or n[key] not in allowed:
                raise ValueError('Invalid NCP '+key)
        if type(n['after_layer']) is not int or type(value.get('depth')) is not int or not 0 <= n['after_layer'] < value['depth']:
            raise ValueError('Invalid NCP insertion layer')
        if n['mode'] not in ('feedback','auxiliary'):
            raise ValueError('Invalid NCP mode')
        for key in ('prediction_weight','vq_weight','ce_weight','lr','feedback_scale'):
            if type(n[key]) not in (int,float) or not math.isfinite(n[key]) or not 0 <= n[key] <= 1:
                raise ValueError('Invalid NCP '+key)
        if n['lr']<=0 or n['lr']>.003:
            raise ValueError('Invalid NCP learning rate')
    if "memory" in value:
        fields.add("memory")
        memory = value["memory"]
        if (not isinstance(memory, dict) or not {"kind", "bos_token_id"} <= set(memory)
                or not set(memory) <= {"kind", "bos_token_id", "after_layer", "lr"}
                or memory["kind"] != "ngram_v1" or type(memory["bos_token_id"]) is not int
                or memory["bos_token_id"] < 0):
            raise ValueError("Memory must select frozen ngram_v1 with an explicit BOS token ID.")
        layer = memory.get("after_layer", 1)
        if (type(layer) is not int or type(value.get("depth")) is not int
                or not 0 <= layer < value["depth"]):
            raise ValueError("Memory after_layer must identify an existing zero-based block.")
        memory_lr = memory.get("lr", .001)
        if type(memory_lr) not in (int, float) or not math.isfinite(memory_lr) or not 0 < memory_lr <= .01:
            raise ValueError("Memory lr must be finite and in (0, .01].")
    if value.get("feedforward") == "moe":
        fields |= {"num_experts", "top_k", "aux_loss_weight", "router_lr"}
    if set(value) != fields or value.get("feedforward") not in ("dense", "moe"):
        raise ValueError("Candidate must select dense or moe with exactly its supported settings.")
    if type(value["depth"]) is not int or not 2 <= value["depth"] <= 12:
        raise ValueError("depth must be an integer from 2 through 12.")
    lr = value["matrix_lr"]
    if type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 0.1:
        raise ValueError("matrix_lr must be finite and in (0, 0.1].")
    if value["feedforward"] == "moe":
        if type(value["num_experts"]) is not int or value["num_experts"] not in (2, 4) or type(value["top_k"]) is not int or value["top_k"] != 1:
            raise ValueError("Packed MoE requires two or four experts and top-1 routing.")
        for key in ("aux_loss_weight", "router_lr"):
            number = value[key]
            if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= 0.1:
                raise ValueError(f"{key} must be finite and in (0, 0.1].")
    return value


def smoke_fingerprint(metadata):
    return {key: metadata.get(key) for key in ("condition", "candidate", "protocol", "upstream", "data_seal",
            "seed", *PROJECT_FILES.values(), "record_version")}


def capture_run_snapshot(output_dir, metadata):
    """Copy the complete local import closure and reject changes since fingerprinting."""
    manifest = {}
    for area in ("project", "upstream"):
        (output_dir / "source" / area).mkdir(parents=True)
    sources = [(ROOT / "scripts" / name, "project/" + name, metadata[key])
               for name, key in PROJECT_FILES.items()]
    sources += [(RUNTIME / name, "upstream/" + name, expected)
                for name, expected in metadata["upstream"]["runtime_files"].items()]
    for original, relative, expected in sources:
        captured = output_dir / "source" / relative
        shutil.copy2(original, captured)
        actual = digest(captured)
        if actual != expected:
            raise ValueError(f"Source changed between fingerprint and capture: {original}")
        manifest["source/" + relative] = actual
    for name in ("candidate", "protocol"):
        write_json(output_dir / (name + ".json"), metadata[name])
        manifest[name + ".json"] = digest(output_dir / (name + ".json"))
    write_json(output_dir / "snapshot.json", manifest)
    return manifest


def snapshot_command(python, output_dir, smoke=False):
    output_dir = output_dir.resolve()
    command = [str(python), "-I", "-u", str(output_dir / "source/project/autoresearch_bootstrap.py"),
               "--runtime", str(output_dir / "source/upstream"),
               "--candidate", str(output_dir / "candidate.json"),
               "--protocol", str(output_dir / "protocol.json")]
    return command + (["--smoke-test"] if smoke else [])


def validate_execution(output_dir, expected):
    execution = json.loads((output_dir / "execution.json").read_text(encoding="utf-8"))
    python_sources = {name: digest_value for name, digest_value in expected.items() if name.endswith(".py")}
    if not execution["verified"] or not execution["isolated"] or execution["files"] != expected or execution["executed_files"] != python_sources:
        raise ValueError("Executed source/configuration hashes do not match the captured run.")
    for name, expected_hash in expected.items():
        if digest(output_dir / name) != expected_hash:
            raise ValueError(f"Captured file changed during execution: {name}")
    return execution


def validate_run_artifacts(directory, record):
    artifacts = {name: json.loads((directory / (name + ".json")).read_text(encoding="utf-8"))
                 for name in ("model", "optimizer", "routing", "training")}
    model, optimizer, routing, training = (artifacts[name] for name in ("model", "optimizer", "routing", "training"))
    candidate, protocol = record["candidate"], record["protocol"]
    if (model.get("activation_checkpointing") is not protocol["activation_checkpointing"]
            or training.get("activation_checkpointing") is not protocol["activation_checkpointing"]):
        raise ValueError("Executed checkpointing does not match the captured protocol.")
    if model["feedforward"] != record["condition"] or routing["feedforward"] != record["condition"]:
        raise ValueError("Recorded model/routing condition does not match the candidate.")
    for key in ("total_parameters", "active_parameters"):
        if type(model[key]) is not int or model[key] <= 0:
            raise ValueError("Invalid exact parameter count.")
    if model["active_parameters"] > model["total_parameters"] or model["depth"] != candidate["depth"]:
        raise ValueError("Invalid model dimensions/counts.")
    if not optimizer["verified"] or optimizer["missing"] or optimizer["duplicate_count"] or optimizer["unexpected_count"]:
        raise ValueError("Optimizer coverage is incomplete.")
    expected_tokens = int(record["metrics"]["num_steps"]) * protocol["tokens_per_update"]
    if training["training_tokens"] != expected_tokens or routing["train"]["tokens"] != expected_tokens:
        raise ValueError("Training token accounting mismatch.")
    eval_tokens = protocol["smoke_eval_tokens"] if record["kind"] == "smoke" else protocol["eval_tokens"]
    if routing["eval"]["tokens"] != eval_tokens:
        raise ValueError("Evaluation token accounting mismatch.")
    for key in ("training_cross_entropy", "training_auxiliary_loss", "weighted_training_auxiliary_loss"):
        if not math.isfinite(routing[key]) or routing[key] < 0:
            raise ValueError("Nonfinite or negative loss statistic.")
    for phase in ("train", "eval"):
        if routing[phase]["dropped_tokens"] != 0:
            raise ValueError("Routing dropped tokens.")
        if candidate["feedforward"] == "moe":
            counts = routing[phase]["counts"]
            if len(counts) != candidate["depth"] or any(len(layer) != candidate["num_experts"] or sum(layer) != routing[phase]["tokens"]
                    or any(type(n) is not int or n < 0 for n in layer) for layer in counts):
                raise ValueError("Expert routing counts do not cover every token exactly once per layer.")
    if candidate["feedforward"] == "moe":
        gradients = routing["router_gradient_max_abs"]
        if not routing["router_gradients_finite"] or len(gradients) != candidate["depth"] or any(not math.isfinite(g) or g <= 0 for g in gradients):
            raise ValueError("Router gradients were absent, zero or nonfinite.")
        if routing["aux_loss_weight"] != candidate["aux_loss_weight"]:
            raise ValueError("Auxiliary coefficient changed.")
    return artifacts


def validate_protocol(value):
    fixed = value.get("stopping_rule") == "optimizer_updates"
    extra = {"stopping_rule", "optimizer_updates", "seed", "schedule", "batch_tape", "batch_tape_sha256"} if fixed else set()
    numeric = {"sequence_length", "tokens_per_update", "microbatch_size", "eval_batch_size",
               "eval_tokens", "smoke_eval_tokens", "training_seconds", "smoke_timeout_seconds",
               "baseline_timeout_seconds"}
    if set(value) != numeric | {"protocol_id", "activation_checkpointing"} | extra or not isinstance(value["protocol_id"], str):
        raise ValueError("Protocol fields are missing or unknown.")
    if type(value["activation_checkpointing"]) is not bool:
        raise ValueError("activation_checkpointing must be an explicit boolean.")
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
    if fixed:
        if type(value["optimizer_updates"]) is not int or value["optimizer_updates"] != 512:
            raise ValueError("Fixed comparison requires exactly 512 updates.")
        if type(value["seed"]) is not int or value["seed"] not in (42, 43, 44):
            raise ValueError("Fixed comparison requires seed 42, 43 or 44.")
        if value["schedule"] != {"clock": "optimizer_step", "progress": "zero_based_step / 512", "lr_warmup_updates": 0, "decay_start_step": 256, "final_lr_fraction": 0.0, "measurement_warmup_updates": 11, "muon_momentum_warmup_updates": 300}:
            raise ValueError("Unexpected fixed-step schedule.")
        if value["activation_checkpointing"] or value["tokens_per_update"] != 16384:
            raise ValueError("Fixed comparison requires checkpointing off and 16384 tokens/update.")
        if not isinstance(value["batch_tape"], str) or not re.fullmatch("[0-9a-f]{64}", value["batch_tape_sha256"]):
            raise ValueError("A hashed batch tape is required.")
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
            if metadata.get("record_version", 0) >= 3:
                record["snapshot_files"] = capture_run_snapshot(output_dir, metadata)
                command = snapshot_command(command[0], output_dir, metadata["kind"] == "smoke")
                record["command"] = command
                write_json(output_dir / "result.json", record)
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
                    if record.get("record_version", 0) >= 2:
                        record["artifacts"] = validate_run_artifacts(output_dir, record)
                    if record.get("record_version", 0) >= 3:
                        record["execution"] = validate_execution(output_dir, record["snapshot_files"])
                    record["status"] = "completed"
                except (ValueError, KeyError, TypeError, OSError) as error:
                    record.update(status="invalid", error=str(error))
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        if process is not None:
            stop_process(process)
            record["returncode"] = process.returncode
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        if record["status"] == "running":
            record["status"] = "failed"
        record["error"] = str(error)
        with (output_dir / "run.log").open("a", encoding="utf-8") as log:
            log.write(f"\nRunner error: {error}\n")
    finally:
        record["wall_seconds"] = time.monotonic() - started
        write_json(output_dir / "result.json", record)
        emit_compact_report(output_dir)
    return record


def emit_compact_report(output_dir):
    """Reporting failure must never discard or relabel a training result."""
    if not output_dir.resolve().is_relative_to((ROOT / "runs/autoresearch").resolve()):
        return  # Synthetic test fixtures are not research experiments.
    try:
        checked([sys.executable, str(ROOT / "scripts/experiment_reports.py"), "one", str(output_dir)],
                timeout=60, capture=True)
    except (OSError, subprocess.SubprocessError) as error:
        write_json(output_dir / "reporting-error.json", {
            "status": "report_pending", "error_type": type(error).__name__,
            "retry": "python scripts/experiment_reports.py backfill"})
        print("Compact report pending; result preserved. Retry experiment_reports.py backfill.", flush=True)


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


def run(kind, repeat, candidate_path=None, protocol_path=None):
    manifest = verify_runtime()
    protocol = validate_protocol(json.loads((Path(protocol_path) if protocol_path else CONFIG / "protocol.json").read_text()))
    candidate = validate_candidate(json.loads((Path(candidate_path) if candidate_path else CONFIG / "candidate.json").read_text()))
    data_seal = json.loads((STATE / "data-seal.json").read_text())
    verify_seal(STATE / "cache", data_seal)
    head = checked(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    dirty = checked(["git", "status", "--porcelain"], capture=True).stdout.strip()
    metadata = {"kind": kind, "condition": candidate["feedforward"], "record_version": 3, "candidate": candidate, "protocol": protocol,
                "git_commit": head, "git_dirty": bool(dirty), "upstream": manifest,
                "data_seal": data_seal, "seed": 42, "decision": "unreviewed",
                "runner_sha256": digest(Path(__file__)),
                "adapter_sha256": digest(ROOT / "scripts/autoresearch_train.py"),
                "model_sha256": digest(ROOT / "scripts/autoresearch_model.py"),
                "bootstrap_sha256": digest(ROOT / "scripts/autoresearch_bootstrap.py"),
                "memory_sha256": digest(ROOT / "scripts/autoresearch_memory.py")}
    if kind == "baseline":
        require_smoke(ROOT / "runs" / "autoresearch", smoke_fingerprint(metadata))
    for _ in range(repeat):
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        output = ROOT / "runs" / "autoresearch" / tag
        command = snapshot_command(runtime_python(), output, kind == "smoke")
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
    parser.add_argument("--candidate", type=Path, default=CONFIG / "candidate.json",
                        help="Explicit dense or MoE candidate JSON; copied into each run.")
    parser.add_argument("--protocol", type=Path, default=CONFIG / "protocol.json",
                        help="Explicit protocol including checkpointing; captured and smoke-gated.")
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
            return run(args.action, args.repeat, args.candidate, args.protocol)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print("If not set up yet, run the setup action first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
