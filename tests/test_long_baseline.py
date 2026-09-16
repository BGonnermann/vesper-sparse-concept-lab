import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from long_baseline_runtime import Sampler,rng_state,restore_rng,save_state,load_state,fixed_schedule
from foundation_train import make_tape
from foundation_eval import evaluate,selected_inputs

class Bytes:
    def encode_ordinary(self,text):return list(text.encode())
    def decode(self,ids):return bytes(ids).decode()
    def decode_single_token_bytes(self,i):return bytes([i])

class LongTests(unittest.TestCase):
    def rows(self):return [dict(id=f'{d}{i}',domain=d,text=('ab '+str(i)+' ')*300) for d in ('general','technical') for i in range(3)]
    def test_batches_identical_to_frozen_pilot(self):
        rows=self.rows();tape,meta=make_tape(rows,Bytes(),256,301,.8,3);s=Sampler(rows,Bytes(),256,301,.8)
        np.testing.assert_array_equal(np.stack([s.next() for _ in range(48)]),tape)
        self.assertEqual(s.identity,meta['stream_sha256']);self.assertEqual(dict(s.offsets),meta['source_tokens'])
    def test_sampler_resume(self):
        a=Sampler(self.rows(),Bytes(),256,301,.8)
        for _ in range(17):a.next()
        state=a.state_dict();expected=[a.next() for _ in range(23)];b=Sampler(self.rows(),Bytes(),256,301,.8);b.load_state_dict(state)
        np.testing.assert_array_equal(np.stack(expected),np.stack([b.next() for _ in range(23)]))
    def test_sampler_rejects_different_identity(self):
        a=Sampler(self.rows(),Bytes(),256,301,.8);state=a.state_dict();state['identity']={}
        with self.assertRaises(AssertionError):a.load_state_dict(state)
    def test_rng_round_trip(self):
        state=rng_state();expected=(random.random(),np.random.rand(),torch.rand(4));restore_rng(state)
        self.assertEqual(expected[0],random.random());self.assertEqual(expected[1],np.random.rand());self.assertTrue(torch.equal(expected[2],torch.rand(4)))
    def test_checkpoint_optimizer_resume(self):
        device=os.environ.get('MOE_TEST_DEVICE','cpu');torch.manual_seed(301)
        model=torch.nn.Linear(4,4).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.ones(2,4,device=device)
        def step(m,o):o.zero_grad();m(x).square().mean().backward();o.step()
        step(model,opt)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.pt';save_state(p,dict(step=1,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state()))
            step(model,opt);expected={k:v.clone() for k,v in model.state_dict().items()}
            fresh=torch.nn.Linear(4,4).to(device);new=torch.optim.AdamW(fresh.parameters(),lr=.01);state=load_state(p)
            fresh.load_state_dict(state['model']);new.load_state_dict(state['optimizer']);restore_rng(state['rng']);step(fresh,new)
            for k,v in fresh.state_dict().items():self.assertTrue(torch.equal(v,expected[k]))
    def test_corrupt_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.pt';save_state(p,dict(step=1,tensor=torch.ones(3)));p.write_bytes(p.read_bytes()+b'bad')
            with self.assertRaises(AssertionError):load_state(p)
    def test_schedule_unchanged_policy(self):
        s=fixed_schedule(4096);self.assertEqual(s[0]['lr_multiplier'],1);self.assertEqual(s[2048]['lr_multiplier'],1)
        self.assertEqual(s[-1]['lr_multiplier'],2/4096);self.assertEqual(s[300]['muon_momentum'],.95)
    def test_eval_exact_bytes_and_no_train_inputs(self):
        class Uniform(torch.nn.Module):
            def forward(self,x,y,reduction='none'):return torch.full_like(y,math.log(256),dtype=torch.float32)
        rows=[dict(id='x',domain='general',split='validation',text='héllo\n世界')]
        inputs=selected_inputs(rows);before=json.dumps(inputs);result=evaluate(Uniform(),Bytes(),inputs,256,device=os.environ.get('MOE_TEST_DEVICE','cpu'))
        self.assertAlmostEqual(result['aggregate_bpb'],8,places=5);self.assertEqual(before,json.dumps(inputs))
        rows[0]['split']='train'
        with self.assertRaises(ValueError):selected_inputs(rows)

if __name__=='__main__':unittest.main()
