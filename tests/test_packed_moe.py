"""Packed dispatch parity against the pre-optimization routing algorithm."""
import copy
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from autoresearch_model import Top1FeedForward


class OriginalDispatch(Top1FeedForward):
    """Frozen pre-optimization forward, independent of the packed forward."""
    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        with torch.autocast(device_type=x.device.type, enabled=False):
            probabilities = F.softmax(self.router(flat.float()), dim=-1)
        selected = probabilities.argmax(dim=-1)
        counts = torch.bincount(selected, minlength=self.num_experts)
        fractions = counts.to(probabilities.dtype) / flat.shape[0]
        auxiliary = self.num_experts * (fractions.detach() * probabilities.mean(dim=0)).sum()
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            indices = torch.where(selected == expert_id)[0]
            values = expert(flat.index_select(0, indices))
            weights = probabilities[indices, expert_id].unsqueeze(-1)
            values = (values * weights).to(output.dtype)
            output = output.index_add(0, indices, values)
        return output.reshape_as(x), auxiliary, counts


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(32, 128, bias=False)
        self.c_proj = nn.Linear(128, 32, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class PackedDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get("MOE_TEST_DEVICE", "cpu")
        torch.set_num_threads(2)
        torch.manual_seed(501)

    def fixture(self, cls=Top1FeedForward, ties=False):
        torch.manual_seed(501)
        model = cls(Expert(), Expert, 32).to(self.device)
        with torch.no_grad():
            model.router.weight.zero_()
            if not ties:
                model.router.weight[:, :4].copy_(2 * torch.eye(4, device=self.device))
        return model

    def inputs(self, assignments, dtype):
        # Noncontiguous, with distinct rows to detect wrong restoration/order.
        x = torch.randn(2, 32, 8, device=self.device).transpose(1, 2) * .1
        x[:, :, :4] = 0
        for i, expert in enumerate(assignments):
            x[i // 8, i % 8, expert] = 1 + i / 32
        return x.to(dtype).detach().requires_grad_(True)

    def test_all_experts_share_one_packed_input_allocation(self):
        # A regression to separate per-expert gathers violates the packing contract.
        model = self.fixture()
        chunks = []
        handles = [e.register_forward_pre_hook(lambda m, args: chunks.append(args[0]))
                   for e in model.experts]
        x = self.inputs([3, 0, 1, 2] * 4, torch.float32)
        model(x)
        for handle in handles:
            handle.remove()
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len({t.untyped_storage().data_ptr() for t in chunks}), 1,
                         "Experts must receive slices of one packed allocation")
        for expert, chunk in enumerate(chunks):
            expected = x.reshape(16, 32)[torch.tensor([3, 0, 1, 2] * 4, device=self.device) == expert]
            torch.testing.assert_close(chunk, expected, atol=0, rtol=0)

    def test_outputs_all_gradients_and_checkpoint_routes_match_original(self):
        cases = [
            ("balanced", [3, 0, 1, 2] * 4, [4, 4, 4, 4], False),
            ("uneven", [3, 0, 3, 1, 3, 0, 2, 3, 0, 3, 1, 3, 0, 3, 3, 3], [4, 2, 1, 9], False),
            ("empty", [3, 0] * 8, [8, 0, 0, 8], False),
            ("ties", [3, 0, 1, 2] * 4, [16, 0, 0, 0], True),
        ]
        dtypes = [torch.float32] + ([torch.bfloat16] if self.device == "cuda" else [])
        for label, assignments, expected_counts, ties in cases:
            for dtype in dtypes:
                for checked in (False, True):
                    with self.subTest(case=label, dtype=dtype, checkpoint=checked):
                        reference = self.fixture(OriginalDispatch, ties)
                        packed = self.fixture(Top1FeedForward, ties)
                        packed.load_state_dict(reference.state_dict())
                        x = self.inputs(assignments, dtype)
                        target = torch.linspace(-1, 1, x.numel(), device=self.device).reshape_as(x)
                        observations = []
                        for model, use_checkpoint in ((reference, False), (packed, checked)):
                            z = x.detach().clone().requires_grad_(True)
                            routes = []
                            handle = model.router.register_forward_hook(
                                lambda m, args, out: routes.append(out.detach().argmax(-1).cpu()))
                            with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                                output, auxiliary, counts = (checkpoint(model, z, use_reentrant=False)
                                                              if use_checkpoint else model(z))
                                task = (output.float() * target).sum()
                            self.assertEqual(output.dtype, dtype)
                            self.assertEqual(auxiliary.dtype, torch.float32)
                            self.assertEqual(counts.tolist(), expected_counts)
                            task_gradient = torch.autograd.grad(task, model.router.weight, retain_graph=True)[0]
                            self.assertTrue(torch.isfinite(task_gradient).all().item())
                            self.assertGreater(task_gradient.abs().max().item(), 0)
                            gradients = torch.autograd.grad(task + .01 * auxiliary, (z, *model.parameters()))
                            handle.remove()
                            self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))
                            self.assertTrue(all(torch.equal(routes[0], r) for r in routes))
                            if use_checkpoint:
                                self.assertGreaterEqual(len(routes), 2)
                            # Empty paths must yield tensor zeros, not None.
                            for i, (_, p) in enumerate(model.named_parameters()):
                                if ".experts." in "." + _:
                                    expert = int(_.split(".")[1])
                                    if expected_counts[expert] == 0:
                                        self.assertEqual(gradients[i + 1].count_nonzero().item(), 0)
                            observations.append((output.detach(), auxiliary.detach(), task_gradient, gradients, routes[0]))
                        a, b = observations
                        for left, right in zip(a[:3], b[:3]):
                            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
                        for left, right in zip(a[3], b[3]):
                            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
                        self.assertTrue(torch.equal(a[4], b[4]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
