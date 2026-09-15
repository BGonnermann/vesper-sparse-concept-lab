"""Tiny correctness gates for bounded expert-count/memory-placement/LR variants."""
import io
import os
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / ".autoresearch/upstream")]
import train
import autoresearch as runner
from autoresearch_model import model_class
from autoresearch_train import batch_order


def candidate(variant="moe", experts=2, memory=None):
    result = {"depth": 3, "matrix_lr": .04, "feedforward": variant}
    if variant == "moe":
        result.update(num_experts=experts, top_k=1, aux_loss_weight=.01, router_lr=.001)
    if memory is not None:
        result["memory"] = dict(kind="ngram_v1", bos_token_id=0, **memory)
    return result


class CampaignConfigTests(unittest.TestCase):
    def test_expert_count_and_memory_options_are_strict(self):
        for experts in (2, 4):
            for placement in (0, 1, 2):
                value = candidate(experts=experts, memory={"after_layer": placement, "lr": .0005})
                self.assertEqual(runner.validate_candidate(value), value)
        default = candidate(memory={})
        self.assertEqual(runner.validate_candidate(default), default)
        invalid = [candidate(experts=n) for n in (0, 1, 3, 5, True, 2.0)]
        invalid.append(dict(candidate(), top_k=2))
        invalid.extend(candidate(memory={"after_layer": n}) for n in (-1, 3, True, 1.0, "1"))
        invalid.extend(candidate(memory={"lr": lr}) for lr in (0, -.001, .0101, float("nan"), float("inf"), True, ".001"))
        invalid.append(candidate(memory={"unapproved_feature": True}))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.validate_candidate(value)
        self.assertEqual(runner.validate_candidate(candidate(memory={"lr": .01}))["memory"]["lr"], .01)

    def test_seed44_preserves_fixed_budget_and_independent_order(self):
        protocol = {
            "protocol_id": "vesper-tinystories-equal-token-v1",
            "activation_checkpointing": False, "sequence_length": 512,
            "tokens_per_update": 16384, "microbatch_size": 2, "eval_batch_size": 2,
            "eval_tokens": 65536, "smoke_eval_tokens": 8192, "training_seconds": 300,
            "smoke_timeout_seconds": 180, "baseline_timeout_seconds": 900,
            "stopping_rule": "optimizer_updates", "optimizer_updates": 512, "seed": 44,
            "batch_tape": "fixture-tape.pt", "batch_tape_sha256": "0" * 64,
            "schedule": {"clock": "optimizer_step", "progress": "zero_based_step / 512",
                         "lr_warmup_updates": 0, "decay_start_step": 256,
                         "final_lr_fraction": 0.0, "measurement_warmup_updates": 11,
                         "muon_momentum_warmup_updates": 300},
        }
        validated = runner.validate_protocol(protocol)
        self.assertEqual(validated["seed"], 44)
        self.assertEqual(validated["optimizer_updates"] * validated["tokens_per_update"], 8388608)
        order = batch_order(8192, 44)
        torch.manual_seed(901)
        torch.rand(20)
        self.assertEqual(order, batch_order(8192, 44))
        self.assertNotEqual(order, batch_order(8192, 42))
        self.assertEqual(sorted(order), list(range(8192)))


class CampaignNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get("MOE_TEST_DEVICE", "cpu")
        cls.dtype = torch.bfloat16 if cls.device == "cuda" else torch.float32
        torch.set_num_threads(2)

    def context(self):
        return torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda")

    def model(self, value, checkpoint=False):
        torch.manual_seed(42)
        config = train.GPTConfig(sequence_len=8, vocab_size=32, n_layer=3,
                                n_head=1, n_kv_head=1, n_embd=32, window_pattern="L",
                                compute_dtype=self.dtype, use_activation_checkpointing=checkpoint)
        with torch.device("meta"):
            model = model_class(train, value)(config)
        model.to_empty(device=self.device)
        model.init_weights(embed_dtype=self.dtype)
        return model

    def inputs(self):
        return torch.tensor([[0, 1, 2, 3, 0, 4, 5, 6], [0, 7, 8, 9, 10, 0, 11, 12]],
                            device=self.device, dtype=torch.int64)

    def activate(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("c_proj.weight"):
                    p.normal_(std=.02)

    def close(self, a, b):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_legacy_defaults_and_control_initialization_pairing(self):
        for variant in ("dense", "moe"):
            for experts in ((2, 4) if variant == "moe" else (2,)):
                with self.subTest(variant=variant, experts=experts):
                    implicit = self.model(candidate(variant, experts, {})).eval()
                    explicit = self.model(candidate(variant, experts, {"after_layer": 1, "lr": .001})).eval()
                    for name, p in implicit.state_dict().items():
                        self.assertTrue(torch.equal(p, explicit.state_dict()[name]), name)
                    with torch.no_grad(), self.context():
                        self.close(implicit(self.inputs()), explicit(self.inputs()))
                    control = self.model(candidate(variant, experts))
                    altered = self.model(candidate(variant, experts, {"after_layer": 2, "lr": .0005}))
                    other = dict(altered.named_parameters())
                    for name, p in control.named_parameters():
                        self.assertTrue(torch.equal(p, other[name]), name)

    def test_placement_causality_validation_and_state_roundtrip(self):
        ids = self.inputs()
        targets = (ids + 1) % 32
        for variant in ("dense", "moe"):
            for after in (0, 2):
                with self.subTest(variant=variant, after=after):
                    value = candidate(variant, 2, {"after_layer": after, "lr": .0005})
                    model = self.model(value).eval()
                    self.activate(model)
                    events = []
                    hooks = [block.register_forward_hook(
                        lambda module, args, output, i=i: events.append(i))
                        for i, block in enumerate(model.transformer.h)]
                    hooks.append(model.ngram_memory.register_forward_hook(
                        lambda module, args, output: events.append("memory")))
                    try:
                        with torch.no_grad(), self.context():
                            logits = model(ids)
                    finally:
                        for hook in hooks:
                            hook.remove()
                    expected_events = [0, 1, 2]
                    expected_events.insert(after + 1, "memory")
                    self.assertEqual(events, expected_events)
                    self.assertEqual(tuple(logits.shape), (2, 8, 32))
                    self.assertEqual(model.parameter_report()["memory_after_layer"], after)
                    self.assertEqual(model.parameter_report()["memory_lr"], .0005)
                    with torch.no_grad(), self.context():
                        for cut in (1, 4, 7):
                            changed = ids.clone()
                            changed[:, cut:] = (changed[:, cut:] + 13) % 32
                            self.close(logits[:, :cut], model(changed)[:, :cut])
                        expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
                        model.aux_loss_weight = 1000.
                        self.close(model(ids, targets, reduction="none"), expected)
                        self.close(model(ids, targets), expected.mean())
                    buffer = io.BytesIO()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    restored = self.model(value).eval()
                    restored.load_state_dict(torch.load(buffer, map_location=self.device, weights_only=True), strict=True)
                    with torch.no_grad(), self.context():
                        self.close(logits, restored(ids))

    def test_task_gradients_optimizer_coverage_and_checkpoint_agreement(self):
        ids = self.inputs()
        for variant in ("dense", "moe"):
            for after in (0, 2):
                with self.subTest(variant=variant, after=after):
                    value = candidate(variant, 2, {"after_layer": after, "lr": .0005})
                    plain = self.model(value)
                    self.activate(plain)
                    checked = self.model(value, checkpoint=True)
                    checked.load_state_dict(plain.state_dict())
                    for model in (plain, checked):
                        model.aux_loss_weight = 0.  # Router gradients must come from the task.
                        optimizer = model.setup_optimizer(matrix_lr=.04)
                        all_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
                        self.assertEqual(len(all_ids), len(set(all_ids)))
                        self.assertEqual(set(all_ids), {id(p) for p in model.parameters()})
                        memory_ids = {id(p) for p in model.ngram_memory.parameters()}
                        groups = [g for g in optimizer.param_groups if memory_ids & {id(p) for p in g["params"]}]
                        self.assertEqual(len(groups), 1)
                        group = groups[0]
                        self.assertEqual({id(p) for p in group["params"]}, memory_ids)
                        self.assertEqual(group["kind"], "adamw")
                        self.assertEqual(group["initial_lr"], .0005)
                        self.assertEqual(group["lr"], .0005)
                        self.assertEqual(group["betas"], (.9, .999))
                        self.assertEqual(group["eps"], 1e-8)
                        self.assertEqual(group["weight_decay"], 0.)
                    with self.context():
                        a = plain(ids, (ids + 1) % 32)
                        b = checked(ids, (ids + 1) % 32)
                    a.backward()
                    b.backward()
                    self.close(a, b)
                    for (name, p), (other_name, q) in zip(plain.named_parameters(), checked.named_parameters()):
                        self.assertEqual(name, other_name)
                        self.assertIsNotNone(p.grad, name)
                        self.assertIsNotNone(q.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all().item(), name)
                        self.assertTrue(torch.isfinite(q.grad).all().item(), name)
                        self.close(p.grad, q.grad)
                        if "router" in name or name.startswith("ngram_memory."):
                            self.assertGreater(p.grad.abs().sum().item(), 0., name)
                    self.assertEqual(plain.routing_report()["train"], checked.routing_report()["train"])
                    if variant == "moe":
                        report = checked.routing_report()["train"]
                        self.assertEqual(report["tokens"], 16)
                        self.assertEqual(report["dropped_tokens"], 0)
                        for counts in report["counts"]:
                            self.assertEqual(len(counts), 2)
                            self.assertEqual(sum(counts), 16)

    def test_two_expert_empty_route_is_dropless_and_finite(self):
        model = self.model(candidate())
        self.activate(model)
        with torch.no_grad():
            for block in model.transformer.h:
                block.mlp.router.weight.zero_()
        ids = self.inputs()
        with self.context():
            loss = model(ids, (ids + 1) % 32)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        report = model.routing_report()["train"]
        self.assertEqual(report["counts"], [[16, 0], [16, 0], [16, 0]])
        self.assertEqual(report["dropped_tokens"], 0)
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all().item(), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
