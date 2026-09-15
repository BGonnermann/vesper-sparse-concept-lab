"""Real subprocess regression for captured entrypoint, imports and configuration."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("snapshot_runner", ROOT / "scripts/autoresearch.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

ENTRY = '''import argparse, json
from pathlib import Path
import autoresearch, autoresearch_model, autoresearch_memory, autoresearch_ncp, train, prepare
parser = argparse.ArgumentParser()
parser.add_argument("--runtime")
parser.add_argument("--candidate")
parser.add_argument("--protocol")
parser.add_argument("--smoke-test", action="store_true")
args = parser.parse_args()
try:
    import uncaptured_live_helper
except ModuleNotFoundError:
    pass
else:
    raise RuntimeError("Imported an uncaptured live helper")
print(json.dumps([autoresearch.VALUE, autoresearch_model.VALUE, train.VALUE, prepare.VALUE,
                  json.loads(Path(args.candidate).read_text())["depth"],
                  json.loads(Path(args.protocol).read_text())["tokens"]]))
'''


class SnapshotTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory).resolve()
        live = root / "live"
        scripts = live / "scripts"
        runtime = live / "upstream"
        scripts.mkdir(parents=True)
        runtime.mkdir()
        (scripts / "autoresearch.py").write_text("VALUE = 7\n")
        (scripts / "autoresearch_model.py").write_text("VALUE = 11\n")
        (scripts / "autoresearch_memory.py").write_text("VALUE = 19\n")
        (scripts / "autoresearch_ncp.py").write_text("VALUE = 23\n")
        (scripts / "autoresearch_train.py").write_text(ENTRY)
        (scripts / "autoresearch_bootstrap.py").write_bytes((ROOT / "scripts/autoresearch_bootstrap.py").read_bytes())
        (runtime / "train.py").write_text("VALUE = 13\n")
        (runtime / "prepare.py").write_text("VALUE = 17\n")
        for name in ("pyproject.toml", ".python-version", "uv.lock"):
            (runtime / name).write_text("fixture\n")
        metadata = {key: runner.digest(scripts / name) for name, key in runner.PROJECT_FILES.items()}
        metadata.update(candidate={"depth": 6}, protocol={"tokens": 16384},
                        upstream={"runtime_files": {name: runner.digest(runtime / name) for name in runner.RUNTIME_FILES}})
        output = root / "run"
        output.mkdir()
        with patch.object(runner, "ROOT", live), patch.object(runner, "RUNTIME", runtime):
            manifest = runner.capture_run_snapshot(output, metadata)
        return live, output, manifest, metadata

    def test_live_edits_cannot_change_captured_entry_imports_or_config(self):
        with tempfile.TemporaryDirectory() as directory:
            live, output, manifest, metadata = self.fixture(directory)
            for path in (live / "scripts").glob("*.py"):
                path.write_text("raise RuntimeError('LIVE PROJECT EXECUTED')\n")
            for path in (live / "upstream").glob("*.py"):
                path.write_text("raise RuntimeError('LIVE UPSTREAM EXECUTED')\n")
            (live / "candidate.json").write_text('{"depth":8}')
            (live / "protocol.json").write_text('{"tokens":1}')
            (live / "uncaptured_live_helper.py").write_text("VALUE = 99\n")
            metadata["candidate"]["depth"] = 8
            metadata["protocol"]["tokens"] = 1
            env = dict(os.environ, PYTHONPATH=str(live))
            result = subprocess.run(runner.snapshot_command(sys.executable, output, smoke=True),
                                    cwd=live, env=env, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [7, 11, 13, 17, 6, 16384])
            receipt = runner.validate_execution(output, manifest)
            self.assertEqual(len(receipt["executed_files"]), 8)
            self.assertEqual(receipt["executed_files"]["source/project/autoresearch_model.py"],
                             manifest["source/project/autoresearch_model.py"])

    def test_tampered_snapshot_is_rejected_before_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            live, output, _, _ = self.fixture(directory)
            (output / "source/project/autoresearch_model.py").write_text("VALUE = 999\n")
            result = subprocess.run(runner.snapshot_command(sys.executable, output),
                                    cwd=live, text=True, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Captured file changed", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_source_change_between_fingerprint_and_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            live, _, _, metadata = self.fixture(directory)
            (live / "scripts/autoresearch_model.py").write_text("VALUE = 999\n")
            second = Path(directory) / "second"
            second.mkdir()
            with patch.object(runner, "ROOT", live), patch.object(runner, "RUNTIME", live / "upstream"):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    runner.capture_run_snapshot(second, metadata)

    def test_moe_and_changed_executed_source_cannot_reuse_dense_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = dict(condition="dense", candidate={"depth": 6, "feedforward": "dense"},
                            protocol={}, upstream={}, data_seal={}, seed=42, runner_sha256="r",
                            adapter_sha256="a", model_sha256="m", bootstrap_sha256="b", record_version=3)
            (root / "smoke").mkdir()
            runner.write_json(root / "smoke/result.json", dict(metadata, kind="smoke", status="completed"))
            runner.require_smoke(root, runner.smoke_fingerprint(metadata))
            for change in ({"condition": "moe", "candidate": {"depth": 6, "feedforward": "moe"}},
                           {"model_sha256": "changed"}, {"bootstrap_sha256": "changed"},
                           {"data_seal": {"dataset": "changed"}}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    runner.require_smoke(root, runner.smoke_fingerprint(dict(metadata, **change)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
