"""Long-run plumbing only: preserves foundation pilot batches and optimizer policy."""
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from foundation_data import ROOT,digest,verify,write
from foundation_train import records
from foundation_tokenizer import load
from autoresearch_model import model_class,with_model_width
from autoresearch_train import fixed_schedule
import autoresearch as r


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def validate_profile(path):
    p=read(path)
    assert p==json.loads(subprocess.check_output(['git','show','4c5b255:experiments/mainline/foundation-general-v2.json'],cwd=ROOT,text=True)),'Accepted foundation commit mismatch'
    assert digest(ROOT/p['tokenizer']['path'])==p['tokenizer']['sha256']
    m=verify(ROOT/p['data']['path']);assert m['fingerprint']==p['data']['fingerprint']
    assert p['training']==dict(sequence_length=512,microbatch_size=2,tokens_per_update=16384,optimizer_updates=512,activation_checkpointing=False,matrix_lr=.04,amp='bfloat16')
    return p,m

class Sampler:
    """O(corpus + one update) RAM; identical microbatches to make_tape, not a new sampling policy."""
    def __init__(self,rows,enc,bos,seed,weight):
        self.streams={};self.weight=weight;self.rng=random.Random(seed);self.offsets=collections.Counter();self.count=0
        for domain in ('general','technical'):
            ordered=sorted((r for r in rows if r['domain']==domain),key=lambda r:r['id']);random.Random(seed).shuffle(ordered)
            self.streams[domain]=np.asarray([i for row in ordered for i in [bos]+enc.encode_ordinary(row['text'])],dtype=np.int32)
            if len(self.streams[domain])<1025:raise ValueError('Insufficient stream')
        self.identity={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in self.streams.items()}
    def next(self):
        domain='general' if self.rng.random()<self.weight else 'technical';s=self.streams[domain];batch=np.empty((2,513),dtype=np.int32)
        for b in range(2):
            start=self.offsets[domain];batch[b]=s[np.arange(start,start+513)%len(s)];self.offsets[domain]+=512
        self.count+=1;return batch
    def state_dict(self):return dict(rng=self.rng.getstate(),offsets=dict(self.offsets),count=self.count,identity=self.identity,weight=self.weight)
    def load_state_dict(self,state):
        assert state['identity']==self.identity and state['weight']==self.weight
        assert sum(state['offsets'].values())==state['count']*1024
        self.rng.setstate(state['rng']);self.offsets=collections.Counter(state['offsets']);self.count=state['count']
    def progress(self):return {k:dict(tokens=self.offsets[k],stream_tokens=len(v),passes=self.offsets[k]/len(v)) for k,v in self.streams.items()}

def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])

def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu()),'Restored tensor differs'
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else:assert a==b

def save_state(path,state):
    path=Path(path);tmp=path.with_suffix('.partial');torch.save(cpu_tree(state),tmp)
    with tmp.open('r+b') as f:os.fsync(f.fileno())
    h=digest(tmp);os.replace(tmp,path)
    receipt=dict(path=str(path.resolve()),sha256=h,bytes=path.stat().st_size,step=state['step'])
    write(str(path)+'.json',receipt);return receipt

def load_state(path):
    path=Path(path);receipt=read(str(path)+'.json');assert digest(path)==receipt['sha256'],'Checkpoint corruption'
    state=torch.load(path,map_location='cpu',weights_only=True);assert state['step']==receipt['step']
    return state

def build(profile,seed):
    enc,identity,prepare=load(ROOT/profile['tokenizer']['path']);assert identity['sha256']==profile['tokenizer']['sha256']
    import train
    train.MAX_SEQ_LEN=512;train.WINDOW_PATTERN='L';torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
    runtime=train.detect_runtime();assert runtime.device_type=='cuda' and runtime.amp_dtype==torch.bfloat16;train._configure_step_kernels(runtime)
    candidate=read(ROOT/profile['architecture_config']);assert candidate==dict(depth=12,model_width=768,matrix_lr=.04,feedforward='dense')
    config=with_model_width(train.build_model_config(12,enc.n_vocab,runtime,False),candidate)
    with torch.device('meta'):model=model_class(train,candidate)(config)
    model.to_empty(device=runtime.device);model.init_weights(embed_dtype=runtime.amp_dtype)
    assert model.parameter_report()['total_parameters']==profile['total_parameters']
    optimizer=model.setup_optimizer(unembedding_lr=train.UNEMBEDDING_LR,embedding_lr=train.EMBEDDING_LR,scalar_lr=train.SCALAR_LR,adam_betas=train.ADAM_BETAS,matrix_lr=.04,weight_decay=train.WEIGHT_DECAY)
    assert model.optimizer_report['verified']
    return model,optimizer,enc,enc.encode_single_token(prepare.BOS_TOKEN)

def update(model,optimizer,sampler,item,buffer,host):
    torch.cuda.synchronize();start=time.perf_counter();values=[]
    model.train()
    for _ in range(16):
        host.copy_(torch.from_numpy(sampler.next()));buffer.copy_(host,non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(buffer[:,:-1].contiguous(),buffer[:,1:].contiguous())
        if not bool(torch.isfinite(loss)):raise ValueError('Non-finite loss')
        (loss/16).backward();values.append(float(loss.detach()))
    # Observe, do not clip or alter gradients. Float64 scalar reduction avoids norm overflow.
    norms=[p.grad.detach().float().norm().double() for p in model.parameters() if p.grad is not None]
    norm=torch.stack(norms).square().sum().sqrt()
    if not bool(torch.isfinite(norm)):raise ValueError('Non-finite gradients')
    for group in optimizer.param_groups:
        group['lr']=group['initial_lr']*item['lr_multiplier']
        if group['kind']=='muon':group.update(momentum=item['muon_momentum'],weight_decay=item['muon_weight_decay'])
    optimizer.step();model.zero_grad(set_to_none=True);torch.cuda.synchronize()
    return dict(loss=sum(values)/16,gradient_norm=float(norm),seconds=time.perf_counter()-start,matrix_lr=.04*item['lr_multiplier'],lr_multiplier=item['lr_multiplier'],muon_momentum=item['muon_momentum'],muon_weight_decay=item['muon_weight_decay'])

def source_identity():
    names=['long_baseline_runtime.py','long_baseline.py','foundation_train.py','foundation_eval.py','foundation_tokenizer.py','foundation_data.py','autoresearch.py','autoresearch_model.py','autoresearch_train.py']
    return {n:digest(ROOT/'scripts'/n) for n in names}
