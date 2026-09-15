"""Runner tests use real short subprocesses; no CUDA or dataset download."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import subprocess

SPEC = importlib.util.spec_from_file_location("runner", Path(__file__).parents[1] / "scripts/autoresearch.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

GOOD = "val_bpb: 1.25\npeak_vram_mb: 2048.0\nnum_steps: 3\ntraining_seconds: 0.0\n"


class ResultsTests(unittest.TestCase):
    def test_smoke_zero_timer_is_valid_but_not_a_throughput_measurement(self):
        self.assertEqual(runner.parse_summary(GOOD)["val_bpb"], 1.25)

    def test_missing_nonfinite_and_duplicate_metrics_are_rejected(self):
        for text in ["", GOOD.replace("1.25", "nan"), GOOD.replace("1.25", "inf"),
                     GOOD.replace("2048.0", "-1"), GOOD + "val_bpb: 0.1\n"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                runner.parse_summary(text)

    def test_success_saves_log_and_record_with_original_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = runner.run_trial([sys.executable, "-c", "print(" + repr(GOOD) + ")"],
                                      root, root / "trial", 5, {"kind": "smoke"})
            self.assertEqual(result["status"], "completed")
            saved = json.loads((root / "trial/result.json").read_text())
            self.assertEqual(saved["kind"], "smoke")
            self.assertEqual(saved["metrics"]["val_bpb"], 1.25)
            self.assertIn("peak_vram_mb", (root / "trial/run.log").read_text())

    def test_failed_process_is_not_promoted_by_valid_looking_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = runner.run_trial([sys.executable, "-c", "print(" + repr(GOOD) + ");raise SystemExit(7)"],
                                      root, root / "trial", 5, {})
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["returncode"], 7)
            self.assertIsNone(result["metrics"])

    def test_timeout_terminates_child_and_preserves_partial_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            before = time.monotonic()
            result = runner.run_trial([sys.executable, "-u", "-c", "import time;print('started');time.sleep(60)"],
                                      root, root / "trial", 0.5, {})
            self.assertEqual(result["status"], "timeout")
            self.assertLess(time.monotonic() - before, 10)
            self.assertIn("started", (root / "trial/run.log").read_text())

    def test_zero_exit_without_metrics_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = runner.run_trial([sys.executable, "-c", "print('no metrics')"], root, root / "trial", 5, {})
            self.assertEqual(result["status"], "invalid")

    def test_existing_trial_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "trial").mkdir()
            with self.assertRaises(FileExistsError):
                runner.run_trial([sys.executable, "-c", "pass"], root, root / "trial", 5, {})

    def test_changed_or_added_dataset_file_invalidates_seal(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").write_text("original")
            seal = runner.seal_files(root)
            runner.verify_seal(root, seal)
            (root / "data").write_text("changed")
            with self.assertRaises(ValueError):
                runner.verify_seal(root, seal)
            (root / "data").write_text("original")
            (root / "extra").write_text("unexpected")
            with self.assertRaises(ValueError):
                runner.verify_seal(root, seal)

    def test_candidate_rejects_unknown_and_nonfinite_settings(self):
        for candidate in [{"depth": 4, "matrix_lr": float("nan")},
                          {"depth": 4, "matrix_lr": .04, "eval_tokens": 1},
                          {"depth": 0, "matrix_lr": .04}]:
            with self.assertRaises(ValueError):
                runner.validate_candidate(candidate)

    def test_protocol_rejects_partial_batches_and_unbounded_timeouts(self):
        protocol = {"protocol_id": "test", "activation_checkpointing": True, "sequence_length": 512, "tokens_per_update": 16384,
                    "microbatch_size": 2, "eval_batch_size": 2, "eval_tokens": 65536,
                    "smoke_eval_tokens": 8192, "training_seconds": 300,
                    "smoke_timeout_seconds": 180, "baseline_timeout_seconds": 900}
        runner.validate_protocol(protocol)
        for change in [{"tokens_per_update": 513}, {"eval_tokens": 12},
                       {"smoke_timeout_seconds": -1}, {"baseline_timeout_seconds": 999999}]:
            with self.assertRaises(ValueError):
                runner.validate_protocol(dict(protocol, **change))

    def test_baseline_requires_successful_smoke_for_same_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fingerprint = {"candidate": {"depth": 4}, "adapter_sha256": "abc"}
            with self.assertRaises(ValueError):
                runner.require_smoke(root, fingerprint)
            (root / "one").mkdir()
            (root / "one/result.json").write_text(json.dumps(dict(fingerprint, kind="smoke", status="completed")))
            runner.require_smoke(root, fingerprint)
            with self.assertRaises(ValueError):
                runner.require_smoke(root, dict(fingerprint, adapter_sha256="changed"))

    def test_windows_taskkill_timeout_still_kills_owned_child(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with patch.object(runner.os, "name", "nt"), patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("taskkill", 15)):
                runner.stop_process(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
