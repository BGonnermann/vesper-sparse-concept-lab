"""Causal and optimizer checks for source mixing and a real capacity control."""
import copy
import io
import unittest

import torch
from torch.nn import functional as F
import test_ncp as base
import autoresearch as runner
from autoresearch_model import model_class


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base.NCPTests.setUpClass()

    def setUp(self):
        self.helper=base.NCPTests()

    def test_search_chunk_and_insertion_boundaries_remain_causal(self):
        for chunk in (2,4,8):
            for layer in (0,1,2):
                with self.subTest(chunk=chunk,layer=layer):
                    model=self.helper.model(pool_normalization='rms',chunk_size=chunk,after_layer=layer).eval()
                    with torch.no_grad():
                        for name,p in model.named_parameters():
                            if name.endswith('c_proj.weight'): p.normal_(std=.03)
                    ids=torch.randint(0,32,(2,20),device=self.helper.device)
                    changed=ids.clone();changed[:,9:]=(changed[:,9:]+1)%32
                    with torch.no_grad(),self.helper.context():
                        full=model(ids)
                        torch.testing.assert_close(model(changed)[:,:9],full[:,:9],atol=1e-6,rtol=1e-5)
                        for length in sorted({1,chunk-1,chunk,chunk+1,2*chunk-1,19}):
                            torch.testing.assert_close(model(ids[:,:length]),full[:,:length],
                                atol=2e-3 if self.helper.device=='cuda' else 1e-6,rtol=1e-3)

    def test_raw_mixing_matches_zero_logit_behavior(self):
        raw=self.helper.model(mixing='raw_logits').ncp
        soft=self.helper.model().ncp
        pooled=torch.ones(1,2,32,device=self.helper.device)
        with torch.no_grad(),self.helper.context():
            raw.head.weight.zero_(); soft.head.weight.zero_()
            prediction,_=raw.predict(pooled,raw.codebook())
            torch.testing.assert_close(prediction,torch.zeros_like(prediction),rtol=0,atol=0)
            expected=soft.codebook().mean(1).flatten().expand(1,2,32)
            torch.testing.assert_close(soft.predict(pooled,soft.codebook())[0],expected)

    def test_variants_are_causal_target_independent_and_trainable(self):
        for change in ({'mixing':'raw_logits'},{'feedback_scale':8.},{'prediction_weight':0.}):
            with self.subTest(change=change):
                model=self.helper.model(pool_normalization='rms',**change)
                with torch.no_grad():
                    for name,p in model.named_parameters():
                        if name.endswith('c_proj.weight'): p.normal_(std=.03)
                x=torch.randn(2,20,32,device=self.helper.device,requires_grad=True)
                target=torch.randn_like(x,requires_grad=True)
                with self.helper.context():
                    prediction,loss=model.ncp(x,True,target)
                    changed,other=model.ncp(x,True,target+10)
                torch.testing.assert_close(prediction,changed,rtol=0,atol=0)
                loss['total'].backward()
                self.assertIsNone(target.grad)
                model.zero_grad(set_to_none=True)
                ids=torch.randint(0,32,(2,20),device=self.helper.device)
                optimizer=model.setup_optimizer(matrix_lr=.04)
                with self.helper.context(): model(ids,ids.roll(-1,1)).backward()
                for name,p in model.ncp.named_parameters():
                    self.assertIsNotNone(p.grad,name)
                    self.assertTrue(torch.isfinite(p.grad).all(),name)
                    self.assertGreater(float(p.grad.abs().sum()),0,name)
                optimizer.step(); model.eval()
                with torch.no_grad(),self.helper.context():
                    full=model(ids)
                    for length in (1,3,4,7,8,13,19):
                        torch.testing.assert_close(model(ids[:,:length]),full[:,:length],
                            atol=2e-3 if self.helper.device=='cuda' else 1e-6,rtol=1e-3)

    def test_extended_configuration_bounds(self):
        candidate=dict(depth=6,matrix_lr=.04,feedforward='dense',ncp=base.settings())
        candidate['ncp'].update(mixing='raw_logits',feedback_scale=8.,prediction_weight=0.)
        self.assertEqual(runner.validate_candidate(candidate),candidate)
        for key,value in [('mixing','teacher'),('feedback_scale',8.1)]:
            invalid=copy.deepcopy(candidate); invalid['ncp'][key]=value
            with self.assertRaises(ValueError): runner.validate_candidate(invalid)

    def capacity(self):
        candidate=dict(depth=3,matrix_lr=.04,feedforward='dense',
            capacity=dict(kind='residual_mlp_v1',hidden=416,after_layer=0,lr=.001))
        runner.validate_candidate(candidate)
        config=base.train.GPTConfig(sequence_len=20,vocab_size=32,n_layer=3,
            n_head=2,n_kv_head=2,n_embd=32,window_pattern='L',
            compute_dtype=self.helper.dtype,use_activation_checkpointing=False)
        torch.manual_seed(42)
        with torch.device('meta'): model=model_class(base.train,candidate)(config)
        model.to_empty(device=self.helper.device); model.init_weights(embed_dtype=self.helper.dtype)
        return model

    def test_capacity_is_exact_paired_causal_trainable_and_persistent(self):
        model=self.capacity(); ncp=self.helper.model(); dense=self.helper.model(False)
        report=model.parameter_report()
        self.assertEqual(report['capacity_parameters'],ncp.parameter_report()['ncp_parameters'])
        self.assertEqual(report['total_parameters'],ncp.parameter_report()['total_parameters'])
        self.assertEqual(model.num_scaling_params()['total'],report['total_parameters'])
        for name,p in dense.named_parameters():
            self.assertTrue(torch.equal(p,dict(model.named_parameters())[name]),name)
        optimizer=model.setup_optimizer(matrix_lr=.04)
        self.assertTrue(model.optimizer_report['verified'])
        ids=torch.randint(0,32,(2,20),device=self.helper.device)
        with self.helper.context(): model(ids,ids.roll(-1,1)).backward()
        for name,p in model.capacity.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.abs().sum()),0,name)
        optimizer.step(); model.eval()
        before={name:t.clone() for name,t in model.state_dict().items()}
        with torch.no_grad(),self.helper.context():
            full=model(ids)
            for length in (1,3,8,19):
                torch.testing.assert_close(model(ids[:,:length]),full[:,:length],
                    atol=2e-3 if self.helper.device=='cuda' else 1e-6,rtol=1e-3)
            ce=model(ids,ids,reduction='none')
            torch.testing.assert_close(ce,F.cross_entropy(full.flatten(0,1),ids.flatten(),reduction='none'))
        for name,t in model.state_dict().items(): self.assertTrue(torch.equal(t,before[name]),name)
        stream=io.BytesIO(); torch.save(model.state_dict(),stream); stream.seek(0)
        restored=self.capacity().eval(); restored.load_state_dict(torch.load(stream,weights_only=True))
        with torch.no_grad(),self.helper.context(): torch.testing.assert_close(restored(ids),full,rtol=0,atol=0)


if __name__=='__main__': unittest.main(verbosity=2)
