"""Dense-only pilot; fixed tokens, sealed corpus, per-domain evaluation and complete receipts."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import autoresearch as r
from autoresearch_model import model_class, with_model_width
from autoresearch_train import fixed_schedule
from foundation_data import ROOT,digest,sha,verify,write
from foundation_tokenizer import load
from foundation_eval import selected_inputs,evaluate,generate


def records(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]

def make_tape(rows,enc,bos,seed,weight,updates):
    """Source weight is microbatch/token mass, NOT document count. Deterministic epochs."""
    domains=('general','technical');streams={};lengths={}
    for domain in domains:
        ordered=sorted((r for r in rows if r['domain']==domain),key=lambda r:r['id'])
        random.Random(seed).shuffle(ordered)
        streams[domain]=np.asarray([i for row in ordered for i in [bos]+enc.encode_ordinary(row['text'])],dtype=np.int32)
        if len(streams[domain])<1025:raise ValueError('Insufficient training tokens for '+domain)
        lengths[domain]=len(streams[domain])
    rng=random.Random(seed);offsets=Counter();counts=Counter();tape=np.empty((updates*16,2,513),dtype=np.int32)
    for j in range(len(tape)):
        domain='general' if rng.random()<weight else 'technical';s=streams[domain]
        for b in range(2):
            start=offsets[domain];tape[j,b]=s[np.arange(start,start+513)%len(s)];offsets[domain]+=512
        counts[domain]+=1024
    return tape,dict(seed=seed,general_weight=weight,source_tokens=dict(counts),unique_stream_tokens=lengths,
        stream_sha256={k:sha(v.tobytes()) for k,v in streams.items()},stream_passes={k:offsets[k]/lengths[k] for k in domains},
        tape_sha256=sha(tape.tobytes()),policy='One seeded document permutation per domain; cyclic packed stream with BOS; Bernoulli source per 1024-target-token microbatch')

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    row=dict(status='running',seed=args.seed,updates=args.updates,training_tokens=args.updates*16384,
             general_weight=args.general_weight,checkpointing=args.checkpointing,
             git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             dirty_paths=subprocess.check_output(['git','status','--short'],text=True).splitlines(),
             source_hashes={p.name:digest(p) for p in (ROOT/'scripts').glob('*.py') if p.name.startswith(('foundation_','autoresearch'))})
    for name in row['source_hashes']:
        destination=out/'source'/name;destination.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts'/name,destination)
    write(out/'result.json',row)
    try:
        if shutil.disk_usage(out).free<30*2**30:raise RuntimeError('30 GiB disk reserve')
        manifest=verify(args.data);row['dataset_fingerprint']=manifest['fingerprint'];write(out/'data-manifest.json',manifest)
        enc,tokenizer,prepare=load(args.tokenizer);row['tokenizer']=tokenizer
        bos=enc.encode_single_token(prepare.BOS_TOKEN)
        rows=records(Path(args.data)/'train.jsonl');validation=records(Path(args.data)/'validation.jsonl')
        inputs=selected_inputs(validation);write(out/'evaluation-inputs.json',inputs)
        tape,stream=make_tape(rows,enc,bos,args.seed,args.general_weight,args.updates);write(out/'stream.json',stream)
        row['stream']=stream;row['preparation_seconds']=time.perf_counter()-start
        import train
        train.MAX_SEQ_LEN=512;train.WINDOW_PATTERN='L'
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        runtime=train.detect_runtime();assert runtime.device_type=='cuda' and runtime.amp_dtype==torch.bfloat16
        train._configure_step_kernels(runtime)
        candidate=json.loads((ROOT/'experiments/mainline/dense-v1.json').read_text());assert candidate==dict(depth=12,model_width=768,matrix_lr=.04,feedforward='dense')
        config=with_model_width(train.build_model_config(12,enc.n_vocab,runtime,args.checkpointing),candidate)
        with torch.device('meta'):model=model_class(train,candidate)(config)
        model.to_empty(device=runtime.device);model.init_weights(embed_dtype=runtime.amp_dtype)
        row['candidate']=candidate;row['parameters']=model.parameter_report();write(out/'model.json',row['parameters'])
        row['initialization_sha256']=sha(b''.join(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() for p in model.parameters()))
        optimizer=model.setup_optimizer(unembedding_lr=train.UNEMBEDDING_LR,embedding_lr=train.EMBEDDING_LR,
            scalar_lr=train.SCALAR_LR,adam_betas=train.ADAM_BETAS,matrix_lr=.04,weight_decay=train.WEIGHT_DECAY)
        write(out/'optimizer.json',model.optimizer_report)
        row['hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
        row['runtime']=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,upstream=r.verify_runtime())
        schedule=fixed_schedule(args.updates);write(out/'schedule.json',schedule)
        tape=torch.from_numpy(tape).pin_memory();buffer=torch.empty((2,513),dtype=torch.long,device='cuda')
        model.train();torch.cuda.reset_peak_memory_stats();elapsed_all=0.;timed=0.;losses=[]
        for item in schedule:
            step=item['step'];torch.cuda.synchronize();t0=time.perf_counter();values=[]
            for j in range(step*16,(step+1)*16):
                buffer.copy_(tape[j],non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(buffer[:,:-1].contiguous(),buffer[:,1:].contiguous())
                if not bool(torch.isfinite(loss)):raise ValueError('Non-finite training loss')
                (loss/16).backward();values.append(float(loss.detach()))
            for group in optimizer.param_groups:
                group['lr']=group['initial_lr']*item['lr_multiplier']
                if group['kind']=='muon':group.update(momentum=item['muon_momentum'],weight_decay=item['muon_weight_decay'])
            optimizer.step();model.zero_grad(set_to_none=True);torch.cuda.synchronize();dt=time.perf_counter()-t0
            elapsed_all+=dt
            if step>=11:timed+=dt
            loss_value=sum(values)/16;losses.append(dict(step=step+1,mean_loss=loss_value,min_loss=min(values),max_loss=max(values),seconds=dt))
            print(f'step {step+1}/{args.updates} loss={loss_value:.6f} dt={dt:.3f}',flush=True)
            if (step+1)%32==0:write(out/'losses.json',losses)
        write(out/'losses.json',losses)
        row['training']=dict(all_update_seconds=elapsed_all,timed_seconds=timed,all_update_tokens_per_second=args.updates*16384/elapsed_all,
            timed_tokens_per_second=max(0,args.updates-11)*16384/timed if timed else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            min_loss=min(x['min_loss'] for x in losses),max_loss=max(x['max_loss'] for x in losses),finite=True)
        torch.save(model.state_dict(),out/'checkpoint.pt');row['checkpoint_sha256']=digest(out/'checkpoint.pt')
        row['validation']=evaluate(model,enc,inputs,bos);write(out/'validation.json',row['validation'])
        write(out/'samples.json',generate(model,enc,bos))
        # Legacy TinyStories regression only makes sense with its original vocabulary.
        if args.tokenizer=='current':
            prepare.MAX_SEQ_LEN=512;tok=prepare.Tokenizer(enc,'tinystories')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                row['tinystories_legacy_bpb']=prepare.evaluate_bpb(model,tok,2,device='cuda',dataset='tinystories',eval_tokens=65536)
        else:row['tinystories_legacy_bpb']=None
        if verify(args.data)['fingerprint']!=row['dataset_fingerprint']:raise ValueError('Dataset changed during run')
        if row['checkpoint_sha256']!=digest(out/'checkpoint.pt'):raise ValueError('Checkpoint changed during evaluation')
        for name,h in row['source_hashes'].items():
            if digest(ROOT/'scripts'/name)!=h:raise ValueError('Source changed during run: '+name)
        row['status']='completed'
    except BaseException as exc:
        row.update(status='failed',error=repr(exc));raise
    finally:
        row['wall_seconds']=time.perf_counter()-start;write(out/'result.json',row)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/foundation-v1');p.add_argument('--output',required=True)
    p.add_argument('--tokenizer',default='current');p.add_argument('--seed',type=int,default=201);p.add_argument('--updates',type=int,default=512)
    p.add_argument('--general-weight',type=float,default=.8);p.add_argument('--checkpointing',action='store_true');a=p.parse_args()
    if not 0<=a.general_weight<=1 or a.updates<=0:p.error('Invalid mixture weight or update count')
    from ncp_campaign import gpu_lock
    with gpu_lock():main(a)
