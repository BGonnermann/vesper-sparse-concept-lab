"""Run the canonical campaign checks."""
from pathlib import Path
import runpy

if __name__=="__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1]/"tests"/"test_ncp_no_future.py"),run_name="__main__")
