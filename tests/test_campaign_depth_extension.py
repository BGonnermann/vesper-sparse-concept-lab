"""CPU/meta-only gates for the prospective dense depth-10/12 extension.

These establish configuration and exact structural counts, not GPU fit or speed.
No real parameter storage, optimizer, forward pass, or CUDA call is required.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / ".autoresearch/upstream")]
import autoresearch as runner
from autoresearch_model import model_class
import train


class CampaignDepthExtensionTests(unittest.TestCase):
    def test_dense_extended_depth_validation_remains_bounded_and_typed(self):
        for depth in (2, 6, 8, 10, 12):
            value = {"depth": depth, "matrix_lr": .04, "feedforward": "dense"}
            with self.subTest(depth=depth):
                self.assertEqual(runner.validate_candidate(value), value)
        for depth in (0, 1, 13, 100, True, False, 10.0, "10", None):
            value = {"depth": depth, "matrix_lr": .04, "feedforward": "dense"}
            with self.subTest(invalid_depth=depth), self.assertRaises(ValueError):
                runner.validate_candidate(value)

    def test_upstream_shapes_and_exact_dense_parameter_counts_on_meta(self):
        runtime = SimpleNamespace(use_activation_checkpointing=False,
                                  attention_backend="sdpa", amp_dtype=torch.bfloat16)
        # Independent structural formula: attention 4w^2 and MLP 8w^2 per
        # layer, input/output/value embeddings, two depth-length scalars,
        # and one 32-input per-head gate per value-embedding layer.
        cases = ((8, 512, 4, 50332176),
                 (10, 640, 5, 85852980),
                 (12, 768, 6, 135267480))
        with patch.object(train, "MAX_SEQ_LEN", 512), patch.object(train, "WINDOW_PATTERN", "L"):
            for depth, width, heads, expected in cases:
                with self.subTest(depth=depth):
                    config = train.build_model_config(depth, 8192, runtime, False)
                    self.assertEqual(config.n_layer, depth)
                    self.assertEqual(config.n_embd, width)
                    self.assertEqual(config.n_head, heads)
                    self.assertEqual(config.n_kv_head, heads)
                    self.assertEqual(config.n_embd // config.n_head, 128)
                    self.assertEqual(config.sequence_len, 512)
                    self.assertEqual(config.window_pattern, "L")
                    self.assertEqual(config.compute_dtype, torch.bfloat16)
                    self.assertFalse(config.use_activation_checkpointing)
                    candidate = {"depth": depth, "matrix_lr": .04, "feedforward": "dense"}
                    with torch.device("meta"):
                        model = model_class(train, candidate)(config)
                    self.assertTrue(all(p.device.type == "meta" for p in model.parameters()))
                    self.assertEqual(len(model.transformer.h), depth)
                    self.assertEqual(tuple(model.transformer.wte.weight.shape), (8192, width))
                    self.assertEqual(tuple(model.lm_head.weight.shape), (8192, width))
                    self.assertEqual(len(model.value_embeds), depth // 2)
                    for index, block in enumerate(model.transformer.h):
                        for name in ("c_q", "c_k", "c_v", "c_proj"):
                            self.assertEqual(tuple(getattr(block.attn, name).weight.shape), (width, width))
                        self.assertEqual(tuple(block.mlp.c_fc.weight.shape), (4 * width, width))
                        self.assertEqual(tuple(block.mlp.c_proj.weight.shape), (width, 4 * width))
                        self.assertEqual(model.window_sizes[index], (512, 0))
                        if index % 2 == 1:
                            self.assertEqual(tuple(block.attn.ve_gate.weight.shape), (heads, 32))
                            self.assertEqual(tuple(model.value_embeds[str(index)].weight.shape), (8192, width))
                        else:
                            self.assertIsNone(block.attn.ve_gate)
                    independent = (12 * depth * width * width
                                   + (2 + depth // 2) * 8192 * width
                                   + 2 * depth + (depth // 2) * 32 * heads)
                    self.assertEqual(independent, expected)
                    self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
                    self.assertEqual(model.num_scaling_params()["total"], expected)
                    report = model.parameter_report()
                    self.assertEqual(report["depth"], depth)
                    self.assertEqual(report["width"], width)
                    self.assertEqual(report["total_parameters"], expected)
                    self.assertEqual(report["active_parameters"], expected)
                    self.assertEqual(report["memory_table_bytes"], 0)
                    self.assertEqual(report["memory_parameters"], 0)
                    self.assertEqual(report["num_experts"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
