"""Behavioral checks for token-only training of the frozen latent-feedback path."""
import copy
import os
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'.autoresearch/upstream')]
import train
from autoresearch_model import model_class
from autoresearch_ncp import NextConcept


class NoFutureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.device=os.environ.get('MOE_TEST_DEVICE','cpu')
        self.dtype=torch.bfloat16 if self.device=='cuda' else torch.float32
        self.settings=dict(kind='ncp_v1',chunk_size=4,layers=2,entries=16,after_layer=0,
            prediction_weight=0.,vq_weight=0.,ce_weight=0.,lr=.001,feedback_scale=1.,
            mode='feedback',pool_normalization='rms')
        torch.manual_seed(42)

    def context(self):
        return torch.autocast(self.device,dtype=self.dtype,enabled=self.device=='cuda')

    def test_zero_future_objective_has_zero_parameter_and_input_gradients(self):
        module=NextConcept(32,2,self.settings).to(self.device)
        module.initialize(12645)
        x=torch.randn(2,20,32,device=self.device,requires_grad=True)
        target=torch.randn_like(x,requires_grad=True)
        with self.context():
            _,losses=module(x,compute_loss=True,target_hidden=target)
        self.assertEqual(float(losses['total']),0.)
        losses['total'].backward()
        self.assertIsNone(target.grad)
        self.assertEqual(int(torch.count_nonzero(x.grad)),0)
        for name,p in module.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertEqual(int(torch.count_nonzero(p.grad)),0,name)

    def test_real_token_loss_and_all_gradients_ignore_future_target_changes(self):
        config=train.GPTConfig(sequence_len=20,vocab_size=32,n_layer=3,n_head=2,n_kv_head=2,
            n_embd=32,window_pattern='L',compute_dtype=self.dtype,use_activation_checkpointing=False)
        candidate=dict(depth=6,matrix_lr=.04,feedforward='dense',ncp=self.settings)
        with torch.device('meta'):
            model=model_class(train,candidate)(config)
        model.to_empty(device=self.device)
        model.init_weights(embed_dtype=self.dtype)
        models=[model,copy.deepcopy(model),copy.deepcopy(model)]
        x=torch.randint(0,32,(2,20),device=self.device)
        y=torch.randint(0,32,(2,20),device=self.device)
        changed_forward=models[1].ncp.forward
        def altered(hidden,compute_loss=False,target_hidden=None):
            return changed_forward(hidden,compute_loss=compute_loss,target_hidden=hidden.detach().flip(1)+7)
        models[1].ncp.forward=altered
        values=[]
        for index,item in enumerate(models):
            item.train()
            with self.context():
                loss=(F.cross_entropy(item(x).flatten(0,1),y.flatten()) if index==2 else item(x,y))
            loss.backward()
            values.append(loss.detach())
        for index in (1,2):
            torch.testing.assert_close(values[0],values[index],rtol=0,atol=0)
            for (name,left),(other,right) in zip(models[0].named_parameters(),models[index].named_parameters()):
                self.assertEqual(name,other)
                if left.grad is None or right.grad is None:
                    self.assertIsNone(left.grad,name);self.assertIsNone(right.grad,name)
                else:
                    torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0,msg=name)
        for prefix in ('ncp.head.','ncp.codebook.','ncp.blocks.'):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0)
                for name,p in models[0].named_parameters() if name.startswith(prefix)),prefix)


if __name__=='__main__':
    unittest.main(verbosity=2)
