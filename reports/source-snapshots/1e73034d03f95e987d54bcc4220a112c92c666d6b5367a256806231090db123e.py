"""Hold a fresh log writer, announce it, then start a bounded child on a marker."""
from pathlib import Path
import subprocess
import sys
import time
HERE = Path(__file__).resolve().parent
label = sys.argv[1]
path = HERE / (label + ".log")
with path.open("x", encoding="utf-8") as log:
    print(f"LOG READY; child not started: {path}", flush=True)
    deadline = time.monotonic()+180
    while not path.with_suffix(".start").exists():
        if time.monotonic()>deadline: raise TimeoutError("Announcement gate expired")
        time.sleep(.1)
    try:
        p=subprocess.run([sys.executable,"-I","-B","-u",str(HERE/"checkpoint_probe.py"),sys.argv[2]],
                         cwd=HERE,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    except subprocess.TimeoutExpired:
        log.write("\nFAIL: probe exceeded 180-second limit\n")
        raise
print(f"{label}: exit={p.returncode}",flush=True)
raise SystemExit(p.returncode)

