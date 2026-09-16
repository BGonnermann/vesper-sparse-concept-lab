"""Sequential native-Windows supervision, calibration, and a single fixed reference run."""
import argparse
import ctypes
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
import torch
import autoresearch as r
from foundation_data import ROOT,digest,write
from long_baseline import HOME,log
from long_baseline_runtime import read,load_state
import verify_foundation_campaign as monitor

monitor.log=log

def checked(command,path,timeout):
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:monitor.checked(command,path,timeout=timeout)
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def launch(config,name,extra=(),timeout=1200):
    checked([str(r.runtime_python()),'-B',str(ROOT/'scripts/long_baseline.py'),'--config',str(config),'--output',str(HOME/name),*map(str,extra)],HOME/(name+('-resume' if extra and '--resume' in extra else '')+'.log'),timeout)

def preflight():
    from ncp_campaign import gpu_lock
    with gpu_lock():
        for device in ('cpu','cuda'):
            env=r.environment();env['MOE_TEST_DEVICE']=device
            monitor.checked([str(r.runtime_python()),'-B','-m','unittest','discover','-s','tests','-p','test_long_baseline.py','-v'],HOME/f'tests-focused-v2-{device}.log',env=env,timeout=180)
            log('Focused '+device+' tests passed')
    base=dict(profile='experiments/mainline/foundation-general-v2.json',profile_sha256=digest(ROOT/'experiments/mainline/foundation-general-v2.json'),seed=301,updates=4,evaluation_steps=[4],sample_steps=[],milestone_steps=[4],checkpoint_every=4,purpose='resume_preflight')
    config=HOME/'resume-preflight-v2.json';write(config,base)
    launch(config,'resume-straight-v2')
    launch(config,'resume-split-v2',('--until',2))
    launch(config,'resume-split-v2',('--resume',HOME/'resume-split-v2/checkpoint-000002.pt'))
    a=load_state(HOME/'resume-straight-v2/checkpoint-000004.pt');b=load_state(HOME/'resume-split-v2/checkpoint-000004.pt')
    assert a['sampler']==b['sampler'];assert a['scheduler']==b['scheduler']
    def compare(a,b):
        if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and torch.equal(a,b),'Tensor resume divergence'
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:compare(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):compare(x,y)
        else:assert a==b
    compare(a['rng'],b['rng'])
    # Restored states are asserted exactly inside the training process before any updates.
    # Native BF16 training itself is not bitwise deterministic: independent uninterrupted
    # repeats also diverged. Preserve that failed strict test rather than claim bitwise training.
    va=a['evaluations'][-1]['evaluation'];vb=b['evaluations'][-1]['evaluation'];delta=abs(va['aggregate_bpb']-vb['aggregate_bpb'])
    loss_delta=max(abs(x['loss']-y['loss']) for x,y in zip(a['losses'],b['losses']))
    assert delta<1e-4 and loss_delta<1e-4,'Numerical continuation exceeds preflight tolerance'
    write(HOME/'resume-verification.json',dict(status='verified',native_cuda_full_model=True,restored_model_optimizer_rng_bit_identical=True,continued_trajectory_bit_identical=False,sampler_and_scheduler_identical=True,steps=4,interrupted_after=2,validation_absolute_difference=delta,maximum_loss_difference=loss_delta,numerical_tolerance=1e-4,limitation='Independent uninterrupted BF16 repeats also differ; exact checkpoint restoration does not imply a bitwise deterministic future trajectory. Initial failed strict test retained.'))
    del a,b
    log('Full model CUDA resume verified: restored states exact; continuation within1e-4loss/BPB. Native BF16 independent training repeats are not bitwise deterministic.')
    calibration=dict(base,updates=96,evaluation_steps=[0,32,96],sample_steps=[96],milestone_steps=[96],checkpoint_every=96,purpose='calibration')
    config=HOME/'calibration-config.json';write(config,calibration);launch(config,'calibration')
    log('Calibration complete. Select fixed budget from measured cost before launching reference.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preflight','train']);p.add_argument('--config',type=Path);a=p.parse_args()
    if a.action=='preflight':preflight()
    else:
        assert read(HOME/'resume-verification.json')['status']=='verified'
        schedule=read(HOME/'schedule.json');remaining=(datetime.fromisoformat(schedule['training_cutoff'])-datetime.now().astimezone()).total_seconds()
        cfg=read(a.config);assert cfg['purpose']=='long_reference' and cfg['estimated_total_seconds']<remaining
        launch(a.config,'reference',timeout=int(remaining));log('Reference training complete. Final validation replay/test and reporting phase; no additional training.')
