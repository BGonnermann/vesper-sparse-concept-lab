"""Causal, optimizer and accounting contracts against the pinned real model."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / ".autoresearch/upstream"))
import train
import autoresearch as runner

MODEL_SPEC = importlib.util.find_spec("autoresearch_model")
if MODEL_SPEC is not None:
    import autoresearch_model as implementation


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODEL_SPEC is None:
            raise AssertionError("MoE model adapter has not been implemented")
        cls.device = os.environ.get("MOE_TEST_DEVICE", "cpu")
        cls.dtype = torch.bfloat16 if cls.device == "cuda" else torch.float32
        torch.set_num_threads(2)

    def model(self, variant="moe", checkpoint=False):
        torch.manual_seed(17)
        config = train.GPTConfig(sequence_len=8, vocab_size=32, n_layer=2,
                                 n_head=1, n_kv_head=1, n_embd=32,
                                 window_pattern="L", compute_dtype=self.dtype,
                                 use_activation_checkpointing=checkpoint)
        candidate = {"depth": 2, "matrix_lr": .04, "feedforward": variant}
        if variant == "moe":
            candidate.update(num_experts=4, top_k=1, aux_loss_weight=.01, router_lr=.001)
        cls = implementation.model_class(train, candidate)
        with torch.device("meta"):
            model = cls(config)
        model.to_empty(device=self.device)
        model.init_weights(embed_dtype=self.dtype)
        return model

    def context(self):
        return torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda")

    def data(self):
        x = torch.arange(16, device=self.device).reshape(2, 8)
        return x, (x + 1) % 32

    def test_shapes_causal_prefix_and_validation_excludes_auxiliary(self):
        model = self.model().eval()
        # Make expert projections nonzero so causality exercises expert outputs.
        with torch.no_grad():
            for block in model.transformer.h:
                for expert in block.mlp.experts:
                    expert.c_proj.weight.normal_(std=.02)
        x, y = self.data()
        altered = x.clone()
        altered[:, 4:] = (altered[:, 4:] + 7) % 32
        with torch.no_grad(), self.context():
            logits = model(x)
            changed = model(altered)
            self.assertEqual(tuple(logits.shape), (2, 8, 32))
            torch.testing.assert_close(logits[:, :4], changed[:, :4], atol=1e-5, rtol=1e-5)
            expected = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none")
            torch.testing.assert_close(model(x, y, reduction="none"), expected)
            model.aux_loss_weight = 1000.0
            torch.testing.assert_close(model(x, y), expected.mean())

    def test_every_parameter_is_optimized_and_empty_experts_have_finite_gradients(self):
        model = self.model(checkpoint=True)
        optimizer = model.setup_optimizer(matrix_lr=.04)
        expected = {id(p) for p in model.parameters()}
        actual = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), expected)
        with torch.no_grad():
            for block in model.transformer.h:
                block.mlp.router.weight.zero_()  # ties route all tokens to expert 0
        x, y = self.data()
        with self.context():
            loss = model(x, y)
        loss.backward()
        before = {name: p.detach().clone() for name, p in model.named_parameters() if "router" in name}
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(parameter).all().item(), name)
            if name in before:
                self.assertGreater((parameter - before[name]).abs().max().item(), 0, name)
        counts = model.routing_report()["train"]["counts"]
        self.assertEqual(counts, [[16, 0, 0, 0], [16, 0, 0, 0]])

    def test_task_loss_and_balance_each_train_router_and_every_expert_is_reachable(self):
        model = self.model()
        mlp = model.transformer.h[0].mlp
        with torch.no_grad():
            mlp.router.weight.zero_()
            mlp.router.weight[:, :4].copy_(torch.eye(4, device=self.device) * 2)
            for expert in mlp.experts:
                expert.c_proj.weight.normal_(std=.02)
        values = torch.zeros(4, 32, device=self.device)
        values[:, :4] = torch.eye(4, device=self.device)
        with self.context():
            output, auxiliary, counts = mlp(values)
            probabilities = torch.softmax(F.linear(values.float(), mlp.router.weight.float()), -1)
            expected = torch.stack([mlp.experts[i](values[i:i+1])[0] * probabilities[i, i] for i in range(4)])
        self.assertEqual(counts.tolist(), [1, 1, 1, 1])
        torch.testing.assert_close(output, expected, atol=.001, rtol=.01)
        task_gradient = torch.autograd.grad(output.float().square().sum(), mlp.router.weight)[0]
        self.assertGreater(task_gradient.abs().sum().item(), 0)
        with self.context():
            _, unbalanced, _ = mlp(values[:1].repeat(8, 1))
        balance_gradient = torch.autograd.grad(unbalanced, mlp.router.weight)[0]
        self.assertGreater(balance_gradient.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(balance_gradient).all().item())
        self.assertGreater(unbalanced.item(), auxiliary.item())

    def test_checkpoint_equivalence_and_statistics_are_not_double_counted(self):
        plain = self.model(checkpoint=False)
        checked = self.model(checkpoint=True)
        checked.load_state_dict(plain.state_dict())
        x, y = self.data()
        with self.context():
            a, b = plain(x, y), checked(x, y)
        a.backward()
        b.backward()
        torch.testing.assert_close(a, b)
        for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, atol=.0001, rtol=.01)
        self.assertEqual(checked.routing_report()["train"]["counts"], plain.routing_report()["train"]["counts"])
        self.assertEqual(checked.routing_report()["train"]["tokens"], 16)
        for counts in checked.routing_report()["train"]["counts"]:
            self.assertEqual(sum(counts), 16)

    def test_dense_matches_upstream_and_shared_initial_weights_match_moe(self):
        dense = self.model("dense")
        moe = self.model("moe")
        raw = train.GPT(dense.config).to(self.device)
        raw.init_weights(embed_dtype=self.dtype)
        raw.load_state_dict(dense.state_dict())
        x, y = self.data()
        with self.context():
            torch.testing.assert_close(dense(x, y), raw(x, y))
        torch.testing.assert_close(dense.transformer.wte.weight, moe.transformer.wte.weight)
        for a, b in zip(dense.transformer.h, moe.transformer.h):
            torch.testing.assert_close(a.attn.c_q.weight, b.attn.c_q.weight)
        dense_count = dense.parameter_report()
        moe_count = moe.parameter_report()
        self.assertEqual(dense_count["total_parameters"], dense_count["active_parameters"])
        self.assertEqual(moe_count["active_parameters"], dense_count["total_parameters"] + 256)
        self.assertEqual(moe_count["total_parameters"] - moe_count["active_parameters"], 49152)


class VariantConfigTests(unittest.TestCase):
    def test_selectable_variants_and_unknown_options_are_rejected(self):
        dense = {"depth": 6, "matrix_lr": .04, "feedforward": "dense"}
        moe = dict(dense, feedforward="moe", num_experts=4, top_k=1, aux_loss_weight=.01, router_lr=.001)
        self.assertEqual(runner.validate_candidate(dense), dense)
        self.assertEqual(runner.validate_candidate(moe), moe)
        for changed in [dict(moe, top_k=2), dict(moe, num_experts=0),
                        dict(moe, aux_loss_weight=0), dict(moe, router_lr=float("nan")),
                        dict(dense, capacity=1), dict(dense, feedforward="unknown")]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.validate_candidate(changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
