"""Prospective tests for explicit dense width and reserved confirmation seeds."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'.autoresearch/upstream')]
import autoresearch as r
import autoresearch_model as models
import train


class DimensionChecks(unittest.TestCase):
    def test_reserved_confirmation_seeds(self):
        import json
        protocol=json.loads((ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json').read_text())
        for seed in (42,43,44,45,46):
            r.validate_protocol(dict(protocol,seed=seed))
        for seed in (True,41,47,45.0):
            with self.assertRaises(ValueError): r.validate_protocol(dict(protocol,seed=seed))

    def test_width_override_is_bounded(self):
        for depth in (6,12):
            for width in (384,768):
                r.validate_candidate(dict(depth=depth,model_width=width,matrix_lr=.04,feedforward='dense'))
        for changes in ({'model_width':512},{'model_width':True},{'depth':8},{'model_width':384.0}):
            candidate=dict(depth=6,model_width=384,matrix_lr=.04,feedforward='dense');candidate.update(changes)
            with self.assertRaises(ValueError): r.validate_candidate(candidate)

    def test_native_dimensions_and_default_compatibility(self):
        runtime=SimpleNamespace(use_activation_checkpointing=False,attention_backend='sdpa',amp_dtype=torch.bfloat16)
        totals={}
        for depth in (6,12):
            candidate=dict(depth=depth,matrix_lr=.04,feedforward='dense')
            original=train.build_model_config(depth,8192,runtime,False)
            before=copy.deepcopy(original)
            self.assertIs(models.with_model_width(original,candidate),original)
            for width in (384,768):
                explicit=dict(candidate,model_width=width)
                config=models.with_model_width(original,explicit)
                self.assertEqual(original,before)
                self.assertEqual((config.n_layer,config.n_embd,config.n_head,config.n_kv_head),(depth,width,width//128,width//128))
                self.assertEqual((config.sequence_len,config.vocab_size,config.compute_dtype),(original.sequence_len,8192,torch.bfloat16))
                with torch.device('meta'): model=models.model_class(train,explicit)(config)
                self.assertEqual(tuple(model.lm_head.weight.shape),(8192,width))
                for block in model.transformer.h:
                    self.assertEqual(tuple(block.attn.c_q.weight.shape),(width,width))
                    self.assertEqual(tuple(block.mlp.c_fc.weight.shape),(4*width,width))
                totals[(depth,width)]=model.parameter_report()['total_parameters']
        self.assertEqual(totals[(6,384)],26345772)
        self.assertEqual(totals[(12,768)],135267480)
        self.assertLess(totals[(6,384)],totals[(12,384)])
        self.assertLess(totals[(12,384)],totals[(12,768)])
        self.assertLess(totals[(6,384)],totals[(6,768)])
        self.assertLess(totals[(6,768)],totals[(12,768)])


if __name__=='__main__': unittest.main(verbosity=2)
