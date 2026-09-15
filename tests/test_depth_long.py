"""Long-budget stream and schedule correctness, independent of model quality."""
import contextlib
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
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import autoresearch as r
from autoresearch_train import fixed_training,batch_order,fixed_schedule
from test_ncp_order import RandomToy


class LongerBudgetTests(unittest.TestCase):
    def test_protocol_accepts_new_campaign_but_keeps_old_contract(self):
        old=json.loads((ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json').read_text())
        for n in (512,1024,2048):
            p=dict(old,protocol_id='vesper-tinystories-depth-long-v1',optimizer_updates=n,seed=101,
                   tape_microbatches=32768,schedule=dict(old['schedule'],progress=f'zero_based_step / {n}',decay_start_step=n//2))
            self.assertEqual(r.validate_protocol(p),p)
        with self.assertRaises(ValueError): r.validate_protocol(dict(old,optimizer_updates=1024))
        for bad in (True,42,-1):
            with self.assertRaises(ValueError): r.validate_protocol(dict(p,seed=bad))
        with self.assertRaises(ValueError): r.validate_protocol(dict(p,tape_microbatches=8192))

    def test_real_loop_uses_nested_prefix_of_larger_tape(self):
        cwd=Path.cwd(); receipts=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            tape=torch.arange(16,dtype=torch.int64)[:,None,None,None].expand(16,2,1,2).clone()
            source=root/'tape.pt';torch.save(tape,source)
            for n in (4,8,16):
                out=root/str(n);out.mkdir()
                protocol=dict(seed=101,tokens_per_update=2,sequence_length=2,optimizer_updates=n,
                    tape_microbatches=16,batch_tape=str(source),batch_tape_sha256=r.digest(source),schedule={})
                train=SimpleNamespace(GPT=RandomToy,UNEMBEDDING_LR=.004,EMBEDDING_LR=.6,SCALAR_LR=.5,
                    ADAM_BETAS=(.8,.95),MATRIX_LR=.04,WEIGHT_DECAY=.2)
                runtime=SimpleNamespace(device='cpu',device_type='cpu',amp_dtype=torch.bfloat16)
                try:
                    os.chdir(out)
                    with patch('torch.cuda.synchronize'),patch.object(torch.Tensor,'pin_memory',lambda self:self),contextlib.redirect_stdout(io.StringIO()):
                        result=fixed_training(train,protocol,runtime,None,None,1,False)
                    row=json.loads((out/'batches.json').read_text())
                    self.assertEqual(result['step'],n)
                    self.assertEqual(row['consumed_indices'],batch_order(16,101)[:n])
                    self.assertEqual(row['training_tokens'],2*n)
                    receipts.append(json.loads((out/'initialization.json').read_text()))
                finally: os.chdir(cwd)
            self.assertEqual(receipts[0],receipts[1]);self.assertEqual(receipts[1],receipts[2])

    def test_normalized_schedule_has_absolute_momentum_warmup(self):
        for n in (512,1024,2048):
            schedule=fixed_schedule(n)
            self.assertEqual(schedule[0]['lr_multiplier'],1)
            self.assertEqual(schedule[n//2]['lr_multiplier'],1)
            self.assertAlmostEqual(schedule[-1]['lr_multiplier'],2/n)
            self.assertAlmostEqual(schedule[300]['muon_momentum'],.95)
            self.assertAlmostEqual(schedule[n//2]['muon_weight_decay'],.1)


if __name__=='__main__': unittest.main()
