"""Execute only captured project/upstream modules and record their verified hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_snapshot(root, manifest):
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_hash(path) != expected:
            raise RuntimeError(f"Captured file changed or escaped snapshot: {name}")


def main():
    if not sys.flags.isolated:
        raise RuntimeError("Captured experiments require Python isolated mode (-I).")
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    verify_snapshot(root, manifest)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args, _ = parser.parse_known_args()
    for actual, expected in ((args.runtime, root / "source/upstream"),
                             (args.candidate, root / "candidate.json"),
                             (args.protocol, root / "protocol.json")):
        if actual.resolve() != expected:
            raise RuntimeError("Execution paths must refer to the captured configuration and runtime.")
    os.chdir(root)
    # -I excludes CWD, PYTHONPATH and user site-packages. Only captured source
    # directories are added; installed dependencies remain in the pinned venv.
    sys.path[:0] = [str(root / "source/project"), str(root / "source/upstream")]
    entry = root / "source/project/autoresearch_train.py"
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        verify_snapshot(root, manifest)
        executed = {entry, Path(__file__).resolve()}
        for module in tuple(sys.modules.values()):
            origin = getattr(module, "__file__", None)
            if origin:
                path = Path(origin).resolve()
                if path.is_relative_to(root / "source"):
                    executed.add(path)
        executed_hashes = {}
        for path in sorted(executed):
            name = path.relative_to(root).as_posix()
            if name not in manifest:
                raise RuntimeError(f"Unrecorded module executed inside the snapshot: {name}")
            executed_hashes[name] = file_hash(path)
        receipt = {"verified": True, "isolated": True, "python": sys.executable,
                   "files": manifest, "executed_files": executed_hashes,
                   "dependency_scope": "Project/upstream Python sources and configs captured; installed packages remain in the recorded runtime venv"}
        (root / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
