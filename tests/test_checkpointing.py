"""Explicit checkpoint configuration and dense/packed on-off numerical parity."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import autoresearch as runner
import test_moe as fixtures


class CheckpointConfigTests(unittest.TestCase):
    def protocol(self, flag):
        value = json.loads((ROOT / "experiments/autoresearch/protocol.json").read_text())
        return dict(value, activation_checkpointing=flag)

    def test_explicit_boolean_is_supported_and_not_coerced(self):
        for flag in (True, False):
            with self.subTest(flag=flag):
                try:
                    result = runner.validate_protocol(self.protocol(flag))
                except ValueError as exc:
                    self.fail(f"Explicit checkpoint boolean must be supported: {exc}")
                self.assertIs(result["activation_checkpointing"], flag)
        for flag in (0, 1, None, "false"):
            with self.subTest(invalid=flag), self.assertRaises(ValueError):
                runner.validate_protocol(self.protocol(flag))
        missing = self.protocol(True)
        missing.pop("activation_checkpointing")
        with self.assertRaises(ValueError):
            runner.validate_protocol(missing)

    def test_checkpoint_off_cannot_reuse_on_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "on").mkdir()
            runner.write_json(p / "on/result.json", {"kind": "smoke", "status": "completed",
                                                    "protocol": self.protocol(True)})
            runner.require_smoke(p, {"protocol": self.protocol(True)})
            with self.assertRaises(ValueError):
                runner.require_smoke(p, {"protocol": self.protocol(False)})


class CheckpointAgreementTests(unittest.TestCase):
    def test_dense_and_packed_outputs_and_all_gradients_agree(self):
        f = fixtures.ModelTests()
        f.device = os.environ.get("MOE_TEST_DEVICE", "cpu")
        f.dtype = torch.bfloat16 if f.device == "cuda" else torch.float32
        torch.set_num_threads(2)
        for variant in ("dense", "moe"):
            with self.subTest(variant=variant):
                off = f.model(variant, checkpoint=False)
                on = f.model(variant, checkpoint=True)
                with torch.no_grad():
                    for name, p in off.named_parameters():
                        if name.endswith("c_proj.weight"):
                            p.normal_(std=.02)
                on.load_state_dict(off.state_dict())
                x, y = f.data()
                with torch.no_grad(), f.context():
                    a, b = off(x), on(x)
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                with f.context():
                    la, lb = off(x, y), on(x, y)
                la.backward()
                lb.backward()
                torch.testing.assert_close(la, lb, atol=1e-6, rtol=1e-5)
                for (name, p), (_, q) in zip(off.named_parameters(), on.named_parameters()):
                    self.assertIsNotNone(p.grad, name)
                    self.assertIsNotNone(q.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all().item(), name)
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
                self.assertEqual(off.routing_report()["train"], on.routing_report()["train"])
                self.assertIs(off.parameter_report()["activation_checkpointing"], False)
                self.assertIs(on.parameter_report()["activation_checkpointing"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
