"""Append-compatible test logger; retain child exit status and output."""
import os
from pathlib import Path
import subprocess
import sys
import time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
mode = sys.argv[1]
env = dict(os.environ, MOE_TEST_DEVICE="cuda" if mode == "cuda" else "cpu")
args = (["tests/test_checkpointing.py", "CheckpointConfigTests.test_explicit_boolean_is_supported_and_not_coerced"]
        if mode == "red" else ["-m", "unittest", "discover", "-s", "tests", "-v"])
path = HERE / (sys.argv[2] if len(sys.argv) > 2 else "tests-" + mode + ".log")
with path.open("x", encoding="utf-8") as log:
    print(f"LOG READY, test not started: {path}", flush=True)
    gate = path.with_suffix(".start")
    deadline = time.monotonic() + 120
    while not gate.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Log announcement gate timed out")
        time.sleep(.1)
    result = subprocess.run([sys.executable, "-B", *args], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, timeout=90)
print(f"{mode}: exit={result.returncode}; log={path}")
raise SystemExit(result.returncode)
