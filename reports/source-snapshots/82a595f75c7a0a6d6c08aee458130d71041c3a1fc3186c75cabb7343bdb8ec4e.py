"""Hold the final report log until announced; no training runs here."""
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
path=HERE/"report-checks.log"
with path.open("x",encoding="utf-8") as log:
    print(f"LOG READY; report checks not started: {path}",flush=True)
    deadline=time.monotonic()+180
    while not (HERE/"report-checks.start").exists():
        if time.monotonic()>deadline:raise TimeoutError("Report announcement gate expired")
        time.sleep(.1)
    p=subprocess.run([sys.executable,"-I","-B","-u",str(HERE/"final_report.py")],
                     cwd=HERE,stdout=log,stderr=subprocess.STDOUT,timeout=90)
print(f"Report checks exit={p.returncode}",flush=True)
raise SystemExit(p.returncode)

