"""Fresh-process inference; final test can only open once on the fixed final endpoint."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import torch
from long_baseline_runtime import *
from long_baseline import log,samples
from foundation_eval import evaluate,selected_inputs


def weights_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def execute(run,output,mode,checkpoint=None):
    if output.exists():raise ValueError('Preserve prior output; use a fresh path. Never repeat the final-test stage.')
    cfg=read(run/'config.json');profile,manifest=validate_profile(ROOT/cfg['profile']);result=read(run/'result.json')
    checkpoint=checkpoint or Path(result['latest']['path']);state=load_state(checkpoint)
    assert state['config_sha256']==digest(run/'config.json')==result['config_sha256']
    assert state['source_hashes']==source_identity() and state['upstream']==r.verify_runtime()
    assert state['dataset_fingerprint']==manifest['fingerprint'] and state['tokenizer_sha256']==profile['tokenizer']['sha256']
    if mode=='finalize':
        assert result['status']=='completed' and state['step']==cfg['updates']==result['step'],'Final fixed-budget checkpoint only'
        marker=run/'test-opened.json'
        if marker.exists():raise ValueError('Test stage already opened; preserve its outcome and do not rerun')
    model,optimizer,enc,bos=build(profile,cfg['seed']);del optimizer
    assert all(bool(torch.isfinite(t).all()) for t in state['model'].values())
    model.load_state_dict(state['model'],strict=True);step=state['step'];del state
    before=weights_hash(model);receipt=dict(status='running',mode=mode,step=step,tokens=step*16384,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        config_sha256=digest(run/'config.json'),dataset_fingerprint=manifest['fingerprint'],tokenizer_sha256=profile['tokenizer']['sha256'],evaluator_sha256=digest(__file__),weights_before=before)
    if mode=='generate':
        generated=samples(model,enc,bos);assert weights_hash(model)==before
        receipt.update(status='verified',samples=generated,weights_after=before);write(output,receipt);log(f'Fresh-process generation verified at step={step}: {output}');return
    output.mkdir(parents=True)
    validation=selected_inputs(records(ROOT/profile['data']['path']/'validation.jsonl'))
    assert validation==read(run/'validation-inputs.json'),'Evaluation input drift'
    value=evaluate(model,enc,validation,bos);write(output/'validation.json',value)
    recorded=next(x for x in read(run/'evaluations.json') if x['step']==step)['evaluation']
    assert value['input_fingerprint']==recorded['input_fingerprint'] and abs(value['aggregate_bpb']-recorded['aggregate_bpb'])<1e-6,'Fresh validation replay mismatch'
    receipt['validation_absolute_difference']=abs(value['aggregate_bpb']-recorded['aggregate_bpb'])
    if mode=='finalize':
        inputs=selected_inputs(records(ROOT/profile['data']['path']/'test.jsonl'))
        freeze=dict(checkpoint_sha256=receipt['checkpoint_sha256'],step=step,config_sha256=receipt['config_sha256'],evaluator_sha256=receipt['evaluator_sha256'],opened_at=datetime.now().astimezone().isoformat(),input_sha256=hashlib.sha256(''.join(x['sha256'] for x in inputs).encode()).hexdigest())
        write(marker,freeze);write(output/'freeze.json',freeze);log(f'FINAL TEST OPENED ONCE: step={step}, checkpoint={receipt["checkpoint_sha256"]}')
        test=evaluate(model,enc,inputs,bos);write(output/'test.json',test)
        assert test['input_fingerprint']==freeze['input_sha256'];assert weights_hash(model)==before
        receipt['test_sha256']=digest(output/'test.json')
    generated=samples(model,enc,bos);write(output/'samples.json',generated)
    assert weights_hash(model)==before and digest(checkpoint)==receipt['checkpoint_sha256']
    assert verify(ROOT/profile['data']['path'])['fingerprint']==manifest['fingerprint']
    receipt.update(status='verified',weights_after=before,validation_sha256=digest(output/'validation.json'),samples_sha256=digest(output/'samples.json'))
    write(output/'verification.json',receipt);log(f'Fresh-process {mode} verified: {output}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['generate','validation','finalize'],required=True);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    from ncp_campaign import gpu_lock
    with gpu_lock():execute(a.run,a.output,a.mode,a.checkpoint)
