"""Behavioral gates for causal multi-token concept prediction."""
import copy
import io
import os
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / '.autoresearch/upstream')]
import train
import autoresearch as runner
from autoresearch_model import model_class


def settings(**changes):
    return dict(kind='ncp_v1', chunk_size=4, layers=2, entries=16, after_layer=0,
                prediction_weight=1., vq_weight=1., ce_weight=0., lr=.001,
                feedback_scale=1., mode='feedback', **changes)


class NCPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = os.environ.get('MOE_TEST_DEVICE', 'cpu')
        cls.dtype = torch.bfloat16 if cls.device == 'cuda' else torch.float32
        torch.set_num_threads(2)

    def context(self):
        return torch.autocast(self.device, dtype=self.dtype, enabled=self.device == 'cuda')

    def model(self, ncp=True, **changes):
        torch.manual_seed(42)
        candidate = dict(depth=3, matrix_lr=.04, feedforward='dense')
        if ncp:
            candidate['ncp'] = settings()
            candidate['ncp'].update(changes)
        config = train.GPTConfig(sequence_len=20, vocab_size=32, n_layer=3,
            n_head=2, n_kv_head=2, n_embd=32, window_pattern='L',
            compute_dtype=self.dtype, use_activation_checkpointing=False)
        with torch.device('meta'):
            model = model_class(train, candidate)(config)
        model.to_empty(device=self.device)
        model.init_weights(embed_dtype=self.dtype)
        return model

    def test_mechanism_exists_and_backbone_initialization_is_paired(self):
        model, dense = self.model(), self.model(False)
        self.assertIsNotNone(getattr(model, 'ncp', None), 'Missing concept mechanism')
        for name, p in dense.named_parameters():
            self.assertTrue(torch.equal(p, dict(model.named_parameters())[name]), name)
        self.assertGreater(model.parameter_report()['ncp_parameters'], 0)
        self.assertEqual(model.num_scaling_params()['total'], sum(p.numel() for p in model.parameters()))

    def test_scalar_feedback_alignment_and_prefix_causality(self):
        model = self.model().eval()
        self.assertIsNotNone(getattr(model, 'ncp', None), 'Missing concept mechanism')
        ncp = model.ncp
        predicted = torch.arange(5., device=self.device).reshape(1,5,1).expand(1,5,32)
        actual = ncp.broadcast(predicted, 20)
        for t in range(20):
            expected = 0. if t < 3 else float((t-3)//4)
            self.assertTrue(torch.equal(actual[0,t], actual.new_full((32,), expected)))
        # Make all backbone residual projections nonzero to exercise attention.
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith('c_proj.weight'):
                    p.normal_(std=.03)
        ids = torch.randint(0,32,(2,20),device=self.device)
        with torch.no_grad(), self.context():
            full = model(ids)
            for length in range(1,21):
                prefix = model(ids[:,:length])
                torch.testing.assert_close(prefix, full[:,:length], atol=2e-3 if self.device=='cuda' else 1e-6, rtol=1e-3)
            changed = ids.clone(); changed[:,9:] = (changed[:,9:]+1)%32
            torch.testing.assert_close(model(changed)[:,:9], full[:,:9], atol=1e-6, rtol=1e-5)

    def test_future_target_independence_and_gradient_isolation(self):
        model = self.model()
        self.assertIsNotNone(getattr(model, 'ncp', None), 'Missing concept mechanism')
        x = torch.randn(2,20,32,device=self.device,requires_grad=True)
        targets = torch.randn_like(x,requires_grad=True)
        with self.context():
            a, losses = model.ncp(x, compute_loss=True, target_hidden=targets)
            b, other = model.ncp(x, compute_loss=True, target_hidden=targets+100)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertNotEqual(float(losses['prediction'].detach()), float(other['prediction'].detach()))
        losses['vq'].backward(retain_graph=True)
        self.assertIsNone(x.grad)
        self.assertIsNone(targets.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.ncp.codebook.parameters()))
        model.zero_grad(set_to_none=True)
        losses['prediction'].backward()
        self.assertIsNone(targets.grad)
        self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertGreater(float(model.ncp.head.weight.grad.abs().sum()),0)

    def test_token_feedback_gradient_and_auxiliary_only_ablation(self):
        ids = torch.randint(0,32,(2,20),device=self.device)
        for mode in ['feedback','auxiliary']:
            model = self.model(mode=mode)
            self.assertIsNotNone(getattr(model, 'ncp', None), 'Missing concept mechanism')
            with self.context():
                loss = model(ids).float().square().mean()
            loss.backward()
            gradient = model.ncp.head.weight.grad
            if mode == 'feedback':
                self.assertIsNotNone(gradient)
                self.assertGreater(float(gradient.abs().sum()),0)
            else:
                self.assertTrue(gradient is None or float(gradient.abs().sum())==0)

    def test_optimizer_roundtrip_and_eval_immutability(self):
        model = self.model()
        self.assertIsNotNone(getattr(model, 'ncp', None), 'Missing concept mechanism')
        optimizer = model.setup_optimizer(matrix_lr=.04)
        self.assertTrue(model.optimizer_report['verified'])
        ids = torch.randint(0,32,(2,20),device=self.device)
        with self.context():
            model(ids, ids.roll(-1,1)).backward()
        for name,p in model.ncp.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        optimizer.step(); model.zero_grad(set_to_none=True); model.eval()
        before = {n:t.clone() for n,t in model.state_dict().items()}
        with torch.no_grad(), self.context():
            logits = model(ids)
            ce = model(ids,ids,reduction='none')
            expected = F.cross_entropy(logits.flatten(0,1),ids.flatten(),reduction='none')
        torch.testing.assert_close(ce, expected)
        for n,t in model.state_dict().items(): self.assertTrue(torch.equal(t,before[n]),n)
        stream = io.BytesIO(); torch.save(model.state_dict(),stream); stream.seek(0)
        restored = self.model().eval(); restored.load_state_dict(torch.load(stream,weights_only=True))
        with torch.no_grad(), self.context(): torch.testing.assert_close(restored(ids),logits,rtol=0,atol=0)

    def test_candidate_validation(self):
        candidate = dict(depth=6,matrix_lr=.04,feedforward='dense',ncp=settings())
        self.assertEqual(runner.validate_candidate(candidate),candidate)
        for key,value in [('chunk_size',1),('layers',0),('entries',3),('mode','teacher'),('prediction_weight',float('nan')),('after_layer',6)]:
            invalid = copy.deepcopy(candidate); invalid['ncp'][key]=value
            with self.assertRaises(ValueError): runner.validate_candidate(invalid)

    def test_codebook_can_fit_detached_varied_chunks_without_collapsing(self):
        model = self.model()
        self.assertIsNotNone(getattr(model, 'ncp', None), 'Missing concept mechanism')
        ncp = model.ncp
        # Fixed synthetic targets exercise dictionary learning independently of LM.
        target = torch.randn(8,20,32,device=self.device)
        optimizer = torch.optim.AdamW(ncp.codebook.parameters(),lr=.01,weight_decay=0)
        first = None
        for _ in range(30):
            codes = ncp.codebook()
            pooled = ncp.pool(target)
            selected,_ = ncp.quantize(pooled,codes)
            loss = F.mse_loss(selected.flatten(-2),pooled)
            if first is None: first=float(loss.detach())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertLess(float(loss.detach()),first*.8)
        counts = torch.tensor(ncp.diagnostics(target)['target_counts'])
        self.assertTrue(((counts>0).sum(-1)>=4).all())


if __name__ == '__main__': unittest.main(verbosity=2)
