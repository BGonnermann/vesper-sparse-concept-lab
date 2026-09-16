"""Fixed-budget, resumable dense reference training. No search or test-set selection."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import torch
from long_baseline_runtime import *
from foundation_eval import selected_inputs,evaluate,generate
import foundation_eval as evaluation

HOME=ROOT/'runs/foundation_long_20260916'
SAMPLE_PROMPTS=evaluation.PROMPTS+['Write one sentence explaining what a dictionary is.']

def log(message):
    text=datetime.now().astimezone().isoformat()+' '+message
    with (HOME/'progress.log').open('a',encoding='utf-8') as f:f.write(text+'\n')
    print(text,flush=True)

def samples(model,enc,bos):
    old=evaluation.PROMPTS
    try:evaluation.PROMPTS=SAMPLE_PROMPTS;return generate(model,enc,bos,seed=20260915,new_tokens=64)
    finally:evaluation.PROMPTS=old

def train_run(config_path,out,resume=None,until=None):
    cfg=read(config_path);profile,manifest=validate_profile(ROOT/cfg['profile']);assert digest(ROOT/cfg['profile'])==cfg['profile_sha256']
    config_hash=digest(config_path);sources=source_identity();upstream=r.verify_runtime()
    if resume is None:out.mkdir(parents=True,exist_ok=False)
    elif not out.exists():raise ValueError('Resume output missing')
    if shutil.disk_usage(out).free<40*2**30:raise RuntimeError('Disk reserve plus checkpoint headroom')
    attempt=len(list(out.glob('attempt-*.json')))+1;attempt_path=out/f'attempt-{attempt:02d}.json'
    receipt=dict(status='running',config_sha256=config_hash,started_at=datetime.now().astimezone().isoformat(),resume=str(resume),source_hashes=sources,upstream=upstream,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    for name,h in sources.items():
        target=out/'source'/name;target.parent.mkdir(exist_ok=True)
        if not target.exists():shutil.copy2(ROOT/'scripts'/name,target)
        assert digest(target)==h
    write(attempt_path,receipt);log(f'START {out.name} attempt={attempt} seed={cfg["seed"]} updates={cfg["updates"]}')
    started=time.time();losses=[];evaluations=[];step=0;update_seconds=0.;checkpoint_seconds=0.;evaluation_seconds=0.;sample_seconds=0.;peak_allocated=0;peak_reserved=0
    try:
        model,optimizer,enc,bos=build(profile,cfg['seed']);sampler=Sampler(records(ROOT/profile['data']['path']/'train.jsonl'),enc,bos,cfg['seed'],profile['data']['general_weight'])
        inputs=selected_inputs(records(ROOT/profile['data']['path']/'validation.jsonl'))
        write(out/'validation-inputs.json',inputs);write(out/'model.json',model.parameter_report());write(out/'optimizer.json',model.optimizer_report)
        schedule=fixed_schedule(cfg['updates'])
        if resume is not None:
            state=load_state(resume);assert state['config_sha256']==config_hash and state['source_hashes']==sources and state['upstream']==upstream
            assert state['dataset_fingerprint']==manifest['fingerprint'] and state['tokenizer_sha256']==profile['tokenizer']['sha256']
            assert state['scheduler']==dict(updates=cfg['updates'],policy='autoresearch_train.fixed_schedule',next_step=state['step'])
            model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);sampler.load_state_dict(state['sampler']);restore_rng(state['rng'])
            assert_tree_equal(model.state_dict(),state['model']);assert_tree_equal(optimizer.state_dict(),state['optimizer'])
            assert_tree_equal(sampler.state_dict(),state['sampler']);assert_tree_equal(rng_state(),state['rng'])
            log('RESTORE VERIFIED: exact model/optimizer tensors and dtypes, sampler, and all RNG states')
            step=state['step'];assert sampler.count==step*16
            losses=state['losses'];evaluations=state['evaluations'];started=state['started'];update_seconds=state['update_seconds'];checkpoint_seconds=state['checkpoint_seconds'];evaluation_seconds=state['evaluation_seconds'];sample_seconds=state['sample_seconds'];peak_allocated=state['peak_allocated'];peak_reserved=state['peak_reserved']
            assert len(losses)==step;del state
            log(f'RESUMED {out.name} update={step} tokens={step*16384}; identical frozen configuration')
        write(out/'config.json',cfg)
        host=torch.empty((2,513),dtype=torch.int32,pin_memory=True);buffer=torch.empty((2,513),dtype=torch.long,device='cuda')
        evaluation_points=set(cfg['evaluation_steps']);sample_points=set(cfg['sample_steps']);milestones=set(cfg['milestone_steps']);stop=until if until is not None else cfg['updates']
        assert step<stop<=cfg['updates']
        if until is not None:assert cfg['purpose']=='resume_preflight','No quality-based or arbitrary early stop of the reference run'
        if step==0 and 0 in evaluation_points:
            value=evaluate(model,enc,inputs,bos);evaluation_seconds+=value['seconds'];evaluations.append(dict(step=0,tokens=0,wall_seconds=time.time()-started,evaluation=value));write(out/'evaluations.json',evaluations);log(f'VALIDATION {out.name} step=0 BPB={value["aggregate_bpb"]:.6f}')
        torch.cuda.reset_peak_memory_stats()
        with (out/f'updates-attempt-{attempt:02d}.jsonl').open('x',encoding='utf-8') as stream:
            while step<stop:
                value=update(model,optimizer,sampler,schedule[step],buffer,host);step+=1;update_seconds+=value['seconds']
                peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated());peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
                value.update(step=step,tokens=step*16384,tokens_per_parameter=step*16384/profile['total_parameters'],wall_seconds=time.time()-started,tokens_per_second=16384/value['seconds'],peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,data_progress=sampler.progress())
                losses.append(value);stream.write(json.dumps(value)+'\n');stream.flush()
                if step%32==0:log(f'step {step}/{cfg["updates"]} tokens={step*16384} loss={value["loss"]:.6f} grad={value["gradient_norm"]:.4f} tok/s={value["tokens_per_second"]:.0f}')
                checkpoint_due=step%cfg['checkpoint_every']==0 or step in evaluation_points or step in sample_points or step in milestones or step==stop
                if not checkpoint_due:continue
                if shutil.disk_usage(out).free<35*2**30:raise RuntimeError('Checkpoint disk reserve')
                if source_identity()!=sources:raise ValueError('Training source changed')
                assert digest(config_path)==config_hash,'Configuration changed'
                assert verify(ROOT/profile['data']['path'])['fingerprint']==manifest['fingerprint'],'Dataset changed'
                assert digest(ROOT/profile['tokenizer']['path'])==profile['tokenizer']['sha256'],'Tokenizer changed'
                if step in evaluation_points:
                    ev=evaluate(model,enc,inputs,bos);evaluation_seconds+=ev['seconds'];evaluations.append(dict(step=step,tokens=step*16384,wall_seconds=time.time()-started,evaluation=ev));write(out/'evaluations.json',evaluations)
                    log(f'VALIDATION {out.name} step={step} BPB={ev["aggregate_bpb"]:.6f} general={ev["domains"]["general"]["bpb"]:.6f} technical={ev["domains"]["technical"]["bpb"]:.6f}')
                if step in sample_points:
                    t=time.perf_counter();write(out/f'samples-{step:06d}.json',samples(model,enc,bos));sample_seconds+=time.perf_counter()-t;log(f'SAMPLES {out.name} step={step}')
                # Save at an optimizer boundary, after diagnostics which use their own RNG.
                target=out/f'checkpoint-{step:06d}.pt';t=time.perf_counter()
                state=dict(schema=1,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),sampler=sampler.state_dict(),rng=rng_state(),
                    config_sha256=config_hash,source_hashes=sources,upstream=upstream,dataset_fingerprint=manifest['fingerprint'],tokenizer_sha256=profile['tokenizer']['sha256'],
                    scheduler=dict(updates=cfg['updates'],policy='autoresearch_train.fixed_schedule',next_step=step),losses=losses,evaluations=evaluations,started=started,
                    update_seconds=update_seconds,checkpoint_seconds=checkpoint_seconds,evaluation_seconds=evaluation_seconds,sample_seconds=sample_seconds,peak_allocated=peak_allocated,peak_reserved=peak_reserved)
                saved=save_state(target,state);del state;checkpoint_seconds+=time.perf_counter()-t
                previous=read(out/'latest.json') if (out/'latest.json').exists() else None
                write(out/'latest.json',saved);log(f'CHECKPOINT {out.name} step={step} sha256={saved["sha256"]}')
                # Only this run's generated non-milestone checkpoints; retain latest and predecessor.
                retain={step,previous['step'] if previous else step}|milestones
                scored=[x for x in evaluations if x['step']>0]
                if scored:
                    best=min(scored,key=lambda x:x['evaluation']['aggregate_bpb'])['step'];retain.add(best)
                    write(out/'best-validation.json',read(out/f'checkpoint-{best:06d}.pt.json'))
                for old in out.glob('checkpoint-*.pt'):
                    n=int(old.stem.split('-')[-1])
                    if n not in retain:
                        old.unlink();log(f'RETENTION {out.name} pruned own superseded checkpoint step={n}; hash receipt retained')
                write(out/'losses.json',losses)
                torch.cuda.reset_peak_memory_stats()
        assert verify(ROOT/profile['data']['path'])['fingerprint']==manifest['fingerprint']
        assert source_identity()==sources and digest(ROOT/profile['tokenizer']['path'])==profile['tokenizer']['sha256']
        result=dict(status='completed' if step==cfg['updates'] else 'interrupted_for_preflight',step=step,tokens=step*16384,tokens_per_parameter=step*16384/profile['total_parameters'],
            wall_seconds=time.time()-started,update_seconds=update_seconds,checkpoint_seconds=checkpoint_seconds,evaluation_seconds=evaluation_seconds,sample_seconds=sample_seconds,
            tokens_per_second=step*16384/update_seconds,steady_tokens_per_second=(step-11)*16384/sum(x['seconds'] for x in losses[11:]) if step>11 else None,
            peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,latest=read(out/'latest.json'),data_progress=sampler.progress(),parameters=model.parameter_report(),config_sha256=config_hash,source_hashes=sources)
        write(out/'result.json',result);receipt.update(status=result['status'],step=step);log(f'END {out.name}: {result["status"]} tokens={step*16384}')
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc),step=step);log(f'FAILED {out.name}: {exc!r}');raise
    finally:
        receipt['finished_at']=datetime.now().astimezone().isoformat();write(attempt_path,receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--until',type=int)
    a=p.parse_args()
    from ncp_campaign import gpu_lock
    with gpu_lock():train_run(a.config,a.output,a.resume,a.until)
