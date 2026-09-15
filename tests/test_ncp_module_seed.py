"""CPU-only real NextConcept initialization through the actual training adapter."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import autoresearch as r
import autoresearch_train as adapter
from autoresearch_ncp import NextConcept


class ReachedForward(Exception): pass


class Fixture(torch.nn.Module):
    enabled=True
    last=None
    def __init__(self,config):
        super().__init__()
        self.backbone=torch.nn.Parameter(torch.empty(32,32))
        self.ncp=NextConcept(32,2,dict(chunk_size=4,entries=16,layers=1,mode='feedback',
            prediction_weight=1.,vq_weight=1.,ce_weight=0.,feedback_scale=1.)) if self.enabled else None
        Fixture.last=self
    def init_weights(self,**kwargs):
        torch.nn.init.normal_(self.backbone)
        if self.ncp is not None:self.ncp.initialize(12600+torch.initial_seed())
    def setup_optimizer(self,**kwargs):
        self.before_optimizer={k:v.detach().clone() for k,v in self.state_dict().items()}
        self.rng_after_init=torch.rand(8)
        return SimpleNamespace(param_groups=[])
    def parameter_report(self):return {'total_parameters':sum(p.numel() for p in self.parameters())}
    @property
    def optimizer_report(self):return {'verified':True}
    def forward(self,x,y):
        self.first_batch=x.detach().clone()
        raise ReachedForward()


class DenseFixture(Fixture): enabled=False


class ModuleSeedTests(unittest.TestCase):
    def probe(self,seed,module=None,dense=False):
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);tape=torch.arange(512,dtype=torch.int64)[:,None,None,None].expand(512,2,1,4).clone()
            tape_path=directory/'tape.pt';torch.save(tape,tape_path)
            protocol=dict(seed=seed,batch_order_seed=42,tokens_per_update=4,sequence_length=4,
                optimizer_updates=512,batch_tape=str(tape_path),batch_tape_sha256=r.digest(tape_path),schedule={})
            if module is not None:protocol['ncp_initialization_seed']=module
            train=SimpleNamespace(GPT=DenseFixture if dense else Fixture,UNEMBEDDING_LR=.004,
                EMBEDDING_LR=.6,SCALAR_LR=.5,ADAM_BETAS=(.8,.95),MATRIX_LR=.04,WEIGHT_DECAY=.2)
            runtime=SimpleNamespace(device='cpu',device_type='cpu',amp_dtype=torch.bfloat16)
            try:
                os.chdir(directory)
                with patch('torch.cuda.synchronize'),patch.object(torch.Tensor,'pin_memory',lambda self:self),\
                        patch.object(adapter,'batch_order',wraps=adapter.batch_order) as order,contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError if dense and module is not None else ReachedForward):
                        adapter.fixed_training(train,protocol,runtime,None,None,1,False)
                if dense and module is not None:return
                order.assert_called_once_with(512,42)
                receipt=directory/'ncp-initialization.json'
                return dict(state=Fixture.last.before_optimizer,rng=Fixture.last.rng_after_init,
                    batch=Fixture.last.first_batch,initial=json.loads((directory/'initialization.json').read_text(encoding='utf-8')),
                    receipt=json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else None)
            finally:os.chdir(cwd)

    def test_override_before_optimizer_preserves_backbone_rng_and_order(self):
        a=self.probe(42);b=self.probe(45);ab=self.probe(42,45);ba=self.probe(45,42)
        for base,cross,anchor,module in [(a,ab,b,45),(b,ba,a,42)]:
            torch.testing.assert_close(base['state']['backbone'],cross['state']['backbone'],rtol=0,atol=0)
            torch.testing.assert_close(base['rng'],cross['rng'],rtol=0,atol=0)
            torch.testing.assert_close(base['batch'],cross['batch'],rtol=0,atol=0)
            for name,value in anchor['state'].items():
                if name.startswith('ncp.'):
                    torch.testing.assert_close(value,cross['state'][name],rtol=0,atol=0)
            self.assertFalse(torch.equal(base['state']['ncp.head.weight'],cross['state']['ncp.head.weight']))
            self.assertEqual(cross['receipt']['ncp_initialization_seed'],module)
            self.assertEqual(cross['receipt']['effective_seed'],12600+module)
            basis=cross['state']['ncp.codebook.basis'].contiguous().view(torch.uint8).numpy().tobytes()
            self.assertEqual(cross['receipt']['basis_sha256'],hashlib.sha256(basis).hexdigest())

    def test_default_compatibility(self):
        a=self.probe(42);b=self.probe(42,42)
        self.assertIsNone(a['receipt']);self.assertEqual(a['initial'],b['initial'])
        for name,value in a['state'].items():torch.testing.assert_close(value,b['state'][name],rtol=0,atol=0)
        torch.testing.assert_close(a['rng'],b['rng'],rtol=0,atol=0)

    def test_dense_override_rejected(self):self.probe(42,45,dense=True)

    def test_protocol_bounds(self):
        p=json.loads((ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json').read_text(encoding='utf-8'))
        for init in (42,45):
            for module in (42,45):r.validate_protocol(dict(p,seed=init,batch_order_seed=42,ncp_initialization_seed=module))
        for changes in [dict(ncp_initialization_seed=True),dict(ncp_initialization_seed=43),
                        dict(seed=43,ncp_initialization_seed=42),dict(seed=45,ncp_initialization_seed=42),
                        dict(batch_order_seed=45,ncp_initialization_seed=42)]:
            with self.assertRaises(ValueError):r.validate_protocol(dict(p,**changes))


if __name__=='__main__':unittest.main(verbosity=2)
