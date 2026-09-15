"""Bounded frozen-plan gates. No training-data fitting or long GPU runs."""
import copy
import io
import os
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / ".autoresearch/upstream"))
import train
from autoresearch_memory import CausalNgramMemory, IncrementalLookupState, suffix_keys
from autoresearch_model import model_class


def scalar_reference(ids, bos, boundaries=None):
    """Independent Python integer specification; no implementation hash helper."""
    rows = ids.tolist()
    addresses, masks = [], []
    for row_index, row in enumerate(rows):
        row_addresses, row_masks = [], []
        for position in range(len(row)):
            keys, validities = [], []
            for n in (2, 3):
                start = position + 1 - n
                suffix = row[max(0, start):position + 1]
                valid = start >= 0 and bos not in suffix
                if boundaries is not None and start >= 0:
                    valid = valid and not any(boundaries[row_index][start:position + 1])
                h = n
                if valid:
                    for u in suffix:
                        h = (h * 1000003 + (u + 1)) % 2147483647
                keys.append(h % 8192 if valid else -1)
                validities.append(valid)
            row_addresses.append(keys)
            row_masks.append(validities)
        addresses.append(row_addresses)
        masks.append(row_masks)
    shape = (*ids.shape, 2)
    return (torch.tensor(addresses, dtype=torch.int64, device=ids.device).reshape(shape),
            torch.tensor(masks, dtype=torch.bool, device=ids.device).reshape(shape))


def reference_memory(memory, hidden, ids):
    keys, valid = scalar_reference(ids, memory.bos_token_id)
    vectors = []
    for row in range(ids.shape[0]):
        values = []
        for position in range(ids.shape[1]):
            pair = []
            for order, table in enumerate((memory.E2, memory.E3)):
                key = int(keys[row, position, order])
                pair.append(table[key] if key >= 0 else table.new_zeros(64))
            values.append(torch.cat(pair))
        vectors.append(torch.stack(values))
    looked_up = torch.stack(vectors).to(hidden.dtype)
    projected = F.linear(looked_up, memory.Wv)
    with torch.autocast(hidden.device.type, enabled=False):
        gate = torch.sigmoid(F.linear(train.norm(hidden.float()), memory.wg[None], memory.bg[None]))
    return hidden + (gate * projected).to(hidden.dtype), gate


class NgramMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get("MOE_TEST_DEVICE", "cpu")
        cls.dtype = torch.bfloat16 if cls.device == "cuda" else torch.float32
        torch.set_num_threads(2)

    def context(self):
        return torch.autocast(self.device, dtype=self.dtype, enabled=self.device == "cuda")

    def memory(self, width=32, seed=8442):
        value = CausalNgramMemory(width, 0, 32768, train.norm).to(self.device)
        value.initialize(seed)
        return value

    def model(self, variant="dense", memory=True, checkpoint=False, seed=42):
        torch.manual_seed(seed)
        candidate = {"depth": 3, "matrix_lr": .04, "feedforward": variant}
        if variant == "moe":
            candidate.update(num_experts=4, top_k=1, aux_loss_weight=.01, router_lr=.001)
        if memory:
            candidate["memory"] = {"kind": "ngram_v1", "bos_token_id": 0}
        config = train.GPTConfig(sequence_len=8, vocab_size=32, n_layer=3,
                                 n_head=1, n_kv_head=1, n_embd=32,
                                 window_pattern="L", compute_dtype=self.dtype,
                                 use_activation_checkpointing=checkpoint)
        with torch.device("meta"):
            result = model_class(train, candidate)(config)
        result.to_empty(device=self.device)
        result.init_weights(embed_dtype=self.dtype)
        return result

    def inputs(self):
        return torch.tensor([[0, 1, 2, 3, 0, 4, 5, 6], [7, 8, 9, 0, 0, 10, 11, 12]],
                            device=self.device, dtype=torch.int64)

    def assert_numerical(self, actual, expected, label):
        delta = (actual.float() - expected.float()).abs()
        maximum = delta.max().item() if delta.numel() else 0.
        relative = (delta / expected.float().abs().clamp_min(1e-8)).max().item() if delta.numel() else 0.
        print(f"ngram reference {label}: max_abs={maximum:.9g}, max_rel={relative:.9g}, dtype={actual.dtype}")
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_hash_shapes_scalar_reference_boundaries_and_collisions(self):
        cases = [torch.empty(2, 0, dtype=torch.int64), torch.tensor([[1], [0]]),
                 torch.tensor([[1, 2], [0, 0]]), torch.tensor([[1, 2, 3], [5, 5, 5]]),
                 self.inputs().cpu()]
        for ids in cases:
            ids = ids.to(self.device)
            actual = suffix_keys(ids, 0, 32768)
            expected = scalar_reference(ids, 0)
            for a, b in zip(actual, expected):
                self.assertTrue(torch.equal(a, b))
        ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)
        boundary = torch.tensor([[False, False, True, False, False]], device=self.device)
        for a, b in zip(suffix_keys(ids, 0, 32768, boundary), scalar_reference(ids, 0, boundary.tolist())):
            self.assertTrue(torch.equal(a, b))
        # Differing last IDs by 8192 collide here without a modulus wrap.
        collision = torch.tensor([[1, 2], [1, 8194]], device=self.device)
        keys, valid = suffix_keys(collision, 0, 32768)
        self.assertTrue(valid[:, -1, 0].all())
        self.assertEqual(keys[0, -1, 0].item(), keys[1, -1, 0].item())
        vectors = self.memory().lookup(collision)
        torch.testing.assert_close(vectors[0, -1, :64], vectors[1, -1, :64])
        for invalid in (torch.tensor([[-1]]), torch.tensor([[32768]])):
            with self.assertRaises(ValueError):
                suffix_keys(invalid, 0, 32768)

    def test_matching_precision_output_and_gradient_reference(self):
        memory = self.memory()
        reference = copy.deepcopy(memory)
        ids = self.inputs()
        hidden = torch.randn(2, 8, 32, device=self.device, dtype=self.dtype, requires_grad=True)
        other_hidden = hidden.detach().clone().requires_grad_()
        weights = torch.randn_like(hidden)
        with self.context():
            actual = memory(hidden, ids)
            expected, _ = reference_memory(reference, other_hidden, ids)
        self.assertEqual(actual.dtype, hidden.dtype)
        self.assertTrue(all(p.dtype == torch.float32 for p in memory.parameters()))
        self.assert_numerical(actual, expected, "outputs")
        (actual.float() * weights).sum().backward()
        (expected.float() * weights).sum().backward()
        self.assert_numerical(hidden.grad, other_hidden.grad, "hidden gradients")
        for (name, p), (_, q) in zip(memory.named_parameters(), reference.named_parameters()):
            self.assert_numerical(p.grad, q.grad, name + " gradients")

    def test_prefix_interventions_keys_gates_outputs_and_document_reset(self):
        memory = self.memory()
        ids = self.inputs()
        hidden = torch.randn(2, 8, 32, device=self.device, dtype=self.dtype)
        for cut in (1, 3, 4, 5, 7):
            altered = ids.clone()
            altered[:, cut:] = (altered[:, cut:] + 13) % 32
            altered_hidden = hidden.clone()
            altered_hidden[:, cut:] = torch.randn_like(hidden[:, cut:])
            with self.context():
                gates, residual = memory.components(hidden, ids)
                gates_b, residual_b = memory.components(altered_hidden, altered)
            for a, b in zip(memory.keys(ids), memory.keys(altered)):
                self.assertTrue(torch.equal(a[:, :cut], b[:, :cut]))
            self.assert_numerical(gates[:, :cut], gates_b[:, :cut], "causal gates")
            self.assert_numerical(residual[:, :cut], residual_b[:, :cut], "causal residual")
        before = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 7]], device=self.device)
        changed = torch.tensor([[8, 9, 10, 0, 4, 5, 6, 7]], device=self.device)
        for a, b in zip(memory.keys(before), memory.keys(changed)):
            self.assertTrue(torch.equal(a[:, 3:], b[:, 3:]))
        self.assert_numerical(memory.lookup(before)[:, 3:], memory.lookup(changed)[:, 3:], "BOS vectors")

    def test_logits_targets_and_future_activation_gradients(self):
        ids = self.inputs()
        for variant in ("dense", "moe"):
            model = self.model(variant).eval()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith("c_proj.weight"):
                        parameter.normal_(std=.02)
            with self.context():
                expected = model(ids)
                first_loss = model(ids, (ids + 1) % 32, reduction="none")
                second_loss = model(ids, (ids + 2) % 32, reduction="none")
                repeated = model(ids)
            self.assert_numerical(expected, repeated, "targets cannot alter logits")
            self.assertFalse(torch.equal(first_loss, second_loss))
            torch.testing.assert_close(first_loss, F.cross_entropy(expected.flatten(0, 1),
                                       ((ids + 1) % 32).flatten(), reduction="none"))
            for cut in (1, 3, 4, 5, 7):
                changed = ids.clone()
                changed[:, cut:] = (changed[:, cut:] + 13) % 32
                with self.context():
                    changed_logits = model(changed)
                self.assert_numerical(expected[:, :cut], changed_logits[:, :cut], "causal logits")
            captured = []
            def retain_embedding(module, args, output):
                output.retain_grad()
                captured.append(output)
            handle = model.transformer.wte.register_forward_hook(retain_embedding)
            try:
                with self.context():
                    logits = model(ids)
                logits[:, 3, 1].sum().backward()
            finally:
                handle.remove()
            self.assertEqual(captured[0].grad[:, 4:].abs().sum().item(), 0.)
            self.assertGreater(captured[0].grad[:, :4].abs().sum().item(), 0.)

    def test_selected_rows_task_gradients_invalid_zero_and_optimizer(self):
        ids = self.inputs()
        for variant in ("dense", "moe"):
            model = self.model(variant)
            optimizer = model.setup_optimizer(matrix_lr=.04)
            all_ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(set(all_ids), {id(p) for p in model.parameters()})
            memory = model.ngram_memory
            memory_ids = {id(p) for p in memory.parameters()}
            for group in optimizer.param_groups:
                if memory_ids & {id(p) for p in group["params"]}:
                    self.assertEqual(group["kind"], "adamw")
                    self.assertEqual(group["initial_lr"], .001)
                    self.assertEqual(group["betas"], (.9, .999))
                    self.assertEqual(group["eps"], 1e-8)
                    self.assertEqual(group["weight_decay"], 0.)
            with torch.no_grad():
                if variant == "moe":
                    for block in model.transformer.h:
                        for expert in block.mlp.experts:
                            expert.c_proj.weight.normal_(std=.02)
            model.aux_loss_weight = 0.  # Isolate router and memory task gradients.
            with self.context():
                loss = model(ids, (ids + 1) % 32)
            loss.backward()
            keys, valid = memory.keys(ids)
            for i, table in enumerate((memory.E2, memory.E3)):
                selected = keys[..., i][valid[..., i]].unique()
                unused = torch.ones(8192, dtype=torch.bool, device=self.device)
                unused[selected] = False
                self.assertGreater(table.grad[selected].abs().sum().item(), 0.)
                self.assertEqual(table.grad[unused].abs().sum().item(), 0.)
            for p in memory.parameters():
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0.)
            if variant == "moe":
                for block in model.transformer.h:
                    self.assertGreater(block.mlp.router.weight.grad.abs().sum().item(), 0.)
                self.assertEqual(model.routing_report()["train"]["dropped_tokens"], 0)
            before = {name: p.detach().clone() for name, p in memory.named_parameters()}
            optimizer.step()
            for name, p in memory.named_parameters():
                self.assertTrue(torch.isfinite(p).all())
                self.assertGreater((p - before[name]).abs().sum().item(), 0.)
            optimizer.zero_grad(set_to_none=True)
            hidden = torch.randn(2, 3, 32, device=self.device, dtype=self.dtype)
            invalid = torch.zeros(2, 3, dtype=torch.int64, device=self.device)
            with self.context():
                output = memory(hidden, invalid)
            torch.testing.assert_close(output, hidden, atol=0, rtol=0)
            output.float().sum().backward()
            self.assertEqual(memory.E2.grad.abs().sum().item(), 0.)
            self.assertEqual(memory.E3.grad.abs().sum().item(), 0.)
            # No optimizer step here: historical Adam momentum can move an unused row.

    def test_checkpoint_outputs_gradients_keys_and_counters(self):
        ids = self.inputs()
        for variant in ("dense", "moe"):
            for enabled in (False, True):
                plain = self.model(variant, memory=enabled)
                checked = self.model(variant, memory=enabled, checkpoint=True)
                checked.load_state_dict(plain.state_dict())
                with self.context():
                    a = plain(ids, (ids + 1) % 32)
                    b = checked(ids, (ids + 1) % 32)
                a.backward()
                b.backward()
                self.assert_numerical(a, b, "checkpoint loss")
                for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
                    self.assertIsNotNone(p.grad, name)
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
                self.assertEqual(plain.routing_report(), checked.routing_report())
                if enabled:
                    self.assertIsNone(checked.ngram_memory.last_diagnostics)
                    for a, b in zip(plain.ngram_memory.keys(ids), checked.ngram_memory.keys(ids)):
                        self.assertTrue(torch.equal(a, b))

    def test_initialization_pairing_parameter_costs_and_read_only_validation(self):
        dense = self.model("dense", memory=False)
        dg = self.model("dense")
        moe = self.model("moe", memory=False)
        mg = self.model("moe")
        for control, added in ((dense, dg), (moe, mg)):
            other = dict(added.named_parameters())
            for name, p in control.named_parameters():
                self.assertTrue(torch.equal(p, other[name]), name)
            self.assertEqual(added.num_scaling_params()["total"], sum(p.numel() for p in added.parameters()))
            self.assertEqual(added.parameter_report()["active_parameters"] - control.parameter_report()["active_parameters"], 4257)
        for a, b in zip(dg.ngram_memory.parameters(), mg.ngram_memory.parameters()):
            self.assertTrue(torch.equal(a, b))
        changed = self.model(seed=43)
        self.assertFalse(torch.equal(changed.ngram_memory.E2, dg.ngram_memory.E2))
        self.assertEqual(dg.ngram_memory.initialization_seed, 8442)
        large = self.memory(width=384)
        self.assertEqual(sum(p.numel() for p in large.parameters()), 1098113)
        self.assertEqual(large.E2.numel() * 4 + large.E3.numel() * 4, 4194304)
        before = {k: v.clone() for k, v in mg.state_dict().items()}
        mg.eval()
        ids = self.inputs()
        with torch.no_grad(), self.context():
            logits = mg(ids)
            actual = mg(ids, (ids + 1) % 32, reduction="none")
            expected = F.cross_entropy(logits.flatten(0, 1), ((ids + 1) % 32).flatten(), reduction="none")
            mg.ngram_memory.diagnostics(torch.randn(2, 8, 32, device=self.device, dtype=self.dtype), ids)
        torch.testing.assert_close(actual, expected)
        for name, value in mg.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertIsNone(mg.ngram_memory.last_diagnostics)

    def test_incremental_reset_reorder_and_lookup_state_roundtrip(self):
        ids = self.inputs()
        state = IncrementalLookupState(2, 0, 32768)
        full_keys, full_valid = suffix_keys(ids, 0, 32768)
        for position in range(ids.shape[1]):
            keys, valid = state.step(ids[:, position], identities=["a", "b"])
            self.assertTrue(torch.equal(keys, full_keys[:, position]))
            self.assertTrue(torch.equal(valid, full_valid[:, position]))
        saved = copy.deepcopy(state.state_dict())
        restored = IncrementalLookupState(2, 0, 32768)
        restored.load_state_dict(saved)
        state.reorder([1, 0])
        next_ids = torch.tensor([13, 14], device=self.device)
        actual = state.step(next_ids, identities=["b", "a"])
        expected = restored.step(next_ids.flip(0), identities=["a", "b"])
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b.flip(0)))
        state.reset([0])
        _, valid = state.step(next_ids, identities=["new", "a"])
        self.assertFalse(valid[0].any())
        state.step(next_ids, identities=["other", "a"])
        self.assertEqual(state.histories[0], [13])

    def test_cpu_optimizer_rng_sampler_resume_fixture(self):
        # Lookup-module/Adam fixture only, not a whole-model training-resume claim.
        torch.manual_seed(42)
        memory = CausalNgramMemory(8, 0, 32, train.norm)
        memory.initialize(8442)
        optimizer = torch.optim.AdamW(memory.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        sampler = torch.Generator().manual_seed(42)
        def update(module, opt, generator):
            ids = torch.randint(1, 32, (2, 4), generator=generator)
            hidden = torch.randn(2, 4, 8)
            opt.zero_grad(set_to_none=True)
            loss = module(hidden, ids).square().mean()
            loss.backward()
            opt.step()
            return loss.detach(), ids
        update(memory, optimizer, sampler)
        receipt = {"model": memory.state_dict(), "optimizer": optimizer.state_dict(),
                   "rng": torch.get_rng_state(), "sampler": sampler.get_state()}
        buffer = io.BytesIO()
        torch.save(receipt, buffer)
        expected_loss, expected_ids = update(memory, optimizer, sampler)
        expected_state = copy.deepcopy(memory.state_dict())
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        restored = CausalNgramMemory(8, 0, 32, train.norm)
        restored.load_state_dict(saved["model"])
        resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_sampler = torch.Generator()
        resumed_sampler.set_state(saved["sampler"])
        torch.set_rng_state(saved["rng"])
        actual_loss, actual_ids = update(restored, resumed_optimizer, resumed_sampler)
        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(torch.equal(actual_loss, expected_loss))
        for name, value in restored.state_dict().items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
