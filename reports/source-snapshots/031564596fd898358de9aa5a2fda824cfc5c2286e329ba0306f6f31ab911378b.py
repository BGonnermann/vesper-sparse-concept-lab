"""One sequential longer-training campaign, with fixed matrix and captured execution."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import autoresearch as r
import experiment_reports as reports
from ncp_campaign import gpu_lock

HERE=r.ROOT/'runs/autoresearch/depth-long-20260915'
START=datetime.fromisoformat('2026-09-15T19:50:45+00:00')
DEADLINE=datetime.fromisoformat('2026-09-16T03:50:45+00:00')
RESERVE=4500
SOURCE=Path(__file__).read_bytes()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def remaining(): return (DEADLINE-datetime.now(timezone.utc)).total_seconds()
def sources(): return {name:r.digest(r.ROOT/'scripts'/name) for name in r.PROJECT_FILES}
def log(message):
    text=datetime.now(timezone.utc).isoformat()+' '+message
    with (HERE/'campaign.log').open('a',encoding='utf-8') as f: f.write(text+'\n')
    print(text,flush=True)
def records(): return [read(p) for p in sorted(HERE.glob('trial-*/result.json'))]
def candidate(depth): return r.validate_candidate(dict(depth=depth,model_width=768,matrix_lr=.04,feedforward='dense'))
def protocol(seed,budget):
    if (HERE/'matrix-plan.json').exists():
        p=read(HERE/'matrix-plan.json')['protocols'][str(budget)]
        p['seed']=seed
        return r.validate_protocol(p)
    p=read(r.ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json')
    n=512*budget
    p.update(protocol_id='vesper-tinystories-depth-long-v1',seed=seed,optimizer_updates=n,
        tape_microbatches=32768,batch_tape=str(HERE/'batches.pt'),batch_tape_sha256=read(HERE/'stream-result.json')['tape_sha256'],
        baseline_timeout_seconds=2400)
    p['schedule'].update(progress=f'zero_based_step / {n}',decay_start_step=n//2)
    return r.validate_protocol(p)

def prepare_stream():
    import os
    import torch
    r.verify_runtime(); seal=read(r.STATE/'data-seal.json');r.verify_seal(r.STATE/'cache',seal)
    assert shutil.disk_usage(r.ROOT).free>35*2**30
    source=HERE/'stream-prepare.py'
    assert not source.exists() and not (HERE/'batches.pt').exists()
    source.write_bytes((r.RUNTIME/'prepare.py').read_bytes())
    os.environ.update(r.environment())
    spec=importlib.util.spec_from_file_location('prepare',source);prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    torch.set_num_threads(2);prepare.MAX_SEQ_LEN=512
    tokenizer=prepare.Tokenizer.from_directory(dataset='tinystories')
    def loader(): return prepare.make_dataloader(tokenizer,2,512,'train',device='cpu',dataset='tinystories')
    batches=loader();tape=torch.empty((32768,2,2,512),dtype=torch.int64);hashes=[];epochs=set()
    started=time.monotonic()
    for i in range(len(tape)):
        x,y,epoch=next(batches);tape[i,0].copy_(x);tape[i,1].copy_(y);epochs.add(epoch)
        hashes.append(hashlib.sha256(tape[i].numpy().tobytes()).hexdigest())
        if (i+1)%8192==0: log(f'Stream packed {i+1}/32768 microbatches')
    assert torch.equal(tape[:,0,:,1:],tape[:,1,:,:-1])
    old=torch.load(r.ROOT/'runs/autoresearch/equal-token-20260915/batches.pt',map_location='cpu',weights_only=True)
    assert torch.equal(tape[:len(old)],old),'New stream does not extend previous loader stream'
    torch.save(tape,HERE/'batches.pt')
    old_hashes={hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in old}
    del old,batches
    batches=loader()
    for i in range(len(tape)):
        x,y,epoch=next(batches)
        assert torch.equal(x,tape[i,0]) and torch.equal(y,tape[i,1]),i
    r.verify_seal(r.STATE/'cache',seal)
    r.write_json(HERE/'batch-hashes.json',dict(hashes=hashes))
    row=dict(kind='training_stream',status='completed',tape_sha256=r.digest(HERE/'batches.pt'),
        source_sha256=r.digest(source),data_seal=seal,microbatches=32768,training_tokens=33554432,
        shape=list(tape.shape),epochs=sorted(epochs),duplicate_microbatches=len(hashes)-len(set(hashes)),
        previous_tape_matching_microbatches=sum(h in old_hashes for h in hashes),
        verified_reconstruction_microbatches=32768,first8192_match_previous_tape=True,
        batch_hashes_sha256=r.digest(HERE/'batch-hashes.json'),wall_seconds=time.monotonic()-started,
        interpretation='One loader epoch, no replayed indices. Independent runs reuse the same tape; content repetition below batch granularity is not excluded.')
    r.write_json(HERE/'stream-result.json',row);log('Stream sealed and independently reconstructed')

def preflight():
    with gpu_lock():
        for device in ('cpu','cuda'):
            target=HERE/f'preflight-{device}.log'
            with target.open('x',encoding='utf-8') as f:
                done=subprocess.run([str(r.runtime_python()),'-B','-m','unittest','discover','-s','tests','-v'],
                    cwd=r.ROOT,env=dict(r.environment(),MOE_TEST_DEVICE=device),stdout=f,stderr=subprocess.STDOUT,timeout=300)
            assert done.returncode==0,('Preflight failed',target)
        r.write_json(HERE/'preflight-result.json',dict(status='completed',source_hashes=sources(),
            tests={p.name:r.digest(p) for p in (r.ROOT/'tests').glob('test_*.py')},
            logs={d:r.digest(HERE/f'preflight-{d}.log') for d in ('cpu','cuda')}))
    log('CPU and CUDA test gates passed')

def freeze():
    assert not (HERE/'matrix-plan.json').exists()
    gate=read(HERE/'preflight-result.json');assert gate['source_hashes']==sources()
    (HERE/'controller.py').write_bytes(SOURCE)
    shutil.copy2(r.ROOT/'CAMPAIGN_DEPTH_PLAN.md',HERE/'PLAN.md')
    plan=dict(kind='depth_long_matrix',status='frozen',started_at=START.isoformat(),deadline=DEADLINE.isoformat(),
        reserve_seconds=RESERVE,initial_seeds=[101,102,103],additional_seed_policy='104,105,... complete six-cell blocks admitted by time only',
        candidates={str(d):candidate(d) for d in (6,12)},budgets=[1,2,4],
        protocols={str(b):protocol(101,b) for b in (1,2,4)},source_hashes=sources(),
        data_seal=read(r.STATE/'data-seal.json'),upstream=r.verify_runtime(),
        controller_sha256=r.digest(HERE/'controller.py'),plan_sha256=r.digest(HERE/'PLAN.md'),
        decision_source_sha256=r.digest(r.ROOT/'scripts/depth_report.py'),
        preflight_sha256=r.digest(HERE/'preflight-result.json'),stream_sha256=r.digest(HERE/'stream-result.json'),
        initial_free_disk_bytes=shutil.disk_usage(r.ROOT).free,frozen_at=datetime.now(timezone.utc).isoformat())
    r.write_json(HERE/'matrix-plan.json',plan)
    r.write_json(HERE/'result.json',dict(kind='depth_long_campaign',status='running',plan_sha256=r.digest(HERE/'matrix-plan.json')))
    log('Experiment matrix frozen before training')

def publish():
    def git(*args):return subprocess.check_output(['git',*args],cwd=r.ROOT,text=True,stderr=subprocess.STDOUT,timeout=60).strip()
    assert git('remote','get-url','--push','origin')=='https://github.com/VesperEngineering/vesper-sparse-concept-lab.git'
    assert git('branch','--show-current')=='research/overnight-20260915'
    assert all(x.startswith('reports/') for x in git('diff','--cached','--name-only').splitlines())
    reports.emit(HERE);reports.index()
    paths=['reports/README.md','reports/index.json','reports/source-snapshots']
    paths += [str(p.relative_to(r.ROOT)) for p in (r.ROOT/'reports/experiments').glob('depth-long-20260915*')]
    git('add','--',*paths)
    if git('diff','--cached','--name-only'):
        git('-c','user.name=VesperEngineering','-c','user.email=admin@novaaetus.com','commit','-m',f'Report longer-training campaign through {len(records())} attempts')
    def execute(command,**kwargs):return subprocess.run(['git','-c','http.sslBackend=openssl',*command[1:]],timeout=60,**kwargs)
    value=reports.publish(execute=execute)
    r.write_json(HERE/'publication-result.json',dict(status='completed',**value));log('PUBLISHED '+value['publication_commit'])

def validate(out,row,plan):
    from autoresearch_train import batch_order,fixed_schedule
    assert row['candidate']==plan['candidates'][str(row['depth'])]
    p=protocol(row['seed'],row['budget']);assert row['protocol']==p
    assert row['data_seal']==plan['data_seal'] and row['upstream']==plan['upstream']
    for name,h in plan['source_hashes'].items(): assert row['snapshot_files']['source/project/'+name]==h
    r.validate_execution(out,row['snapshot_files']);r.validate_run_artifacts(out,row)
    batches=read(out/'batches.json');n=512*row['budget'];count=n*16
    assert batches['consumed_indices']==batch_order(32768,row['seed'])[:count]
    assert len(set(batches['consumed_indices']))==count and batches['training_tokens']==n*16384
    hashes=read(HERE/'batch-hashes.json')['hashes'];chain=hashlib.sha256()
    for i in batches['consumed_indices']:chain.update(bytes.fromhex(hashes[i]))
    assert chain.hexdigest()==batches['consumed_batch_hash_chain']
    assert row['metrics']['num_steps']==n
    assert read(out/'schedule.json')['updates']==fixed_schedule(n)
    assert read(out/'evaluation-immutability.json')['verified']
    assert r.digest(out/'checkpoint_pre_eval.pt')==row['checkpoint_sha256']

def trial(depth,budget,seed):
    plan=read(HERE/'matrix-plan.json');assert sources()==plan['source_hashes']
    assert hashlib.sha256(SOURCE).hexdigest()==plan['controller_sha256'],'Loaded controller differs from freeze'
    assert r.digest(r.ROOT/'scripts/depth_report.py')==plan['decision_source_sha256'],'Decision source changed'
    assert r.digest(HERE/'stream-result.json')==plan['stream_sha256'],'Stream receipt changed'
    assert r.digest(HERE/'batch-hashes.json')==read(HERE/'stream-result.json')['batch_hashes_sha256']
    assert r.digest(HERE/'preflight-result.json')==plan['preflight_sha256']
    assert read(HERE/'preflight-result.json')['tests']=={p.name:r.digest(p) for p in (r.ROOT/'tests').glob('test_*.py')}
    assert shutil.disk_usage(r.ROOT).free>20*2**30,'Disk floor'
    assert remaining()>RESERVE+60
    out=HERE/f'trial-{len(records())+1:04d}-D{depth}-b{budget}-s{seed}';out.mkdir()
    (out/'orchestrator.py').write_bytes(SOURCE)
    row=dict(kind='fixed_updates',status='prepared',condition='dense',label=f'D{depth}-W768',depth=depth,budget=budget,seed=seed,
        candidate=plan['candidates'][str(depth)],protocol=protocol(seed,budget),data_seal=plan['data_seal'],upstream=plan['upstream'],
        orchestrator_sha256=r.digest(out/'orchestrator.py'),plan_sha256=r.digest(HERE/'matrix-plan.json'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip())
    row.update({key:r.digest(r.ROOT/'scripts'/name) for name,key in r.PROJECT_FILES.items()})
    r.write_json(out/'selection.json',dict(hypothesis='Frozen dense depth by training budget; no score-dependent configuration changes',depth=depth,budget=budget,seed=seed))
    process=None;start=time.monotonic();samples=[]
    log('START '+out.name+' log='+str(out/'run.log'))
    with gpu_lock():
        try:
            assert r.verify_runtime()==plan['upstream'],'Upstream identity changed'
            r.verify_seal(r.STATE/'cache',plan['data_seal'])
            row['snapshot_files']=r.capture_run_snapshot(out,row);row['status']='running';r.write_json(out/'result.json',row)
            with (out/'run.log').open('x',encoding='utf-8') as f:
                process=subprocess.Popen(r.snapshot_command(r.runtime_python(),out),cwd=out,env=r.environment(),
                    stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                end=time.monotonic()+min(2400,remaining()-RESERVE);heartbeat=time.monotonic()+60
                while process.poll() is None:
                    if time.monotonic()>end:raise TimeoutError('Bounded training timeout')
                    try:samples.append(subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip())
                    except (OSError,subprocess.SubprocessError):pass
                    if time.monotonic()>heartbeat:
                        lines=(out/'run.log').read_text(errors='replace').splitlines()
                        log(out.name+' '+next((x for x in reversed(lines) if x.startswith('step ')),'initializing'));heartbeat=time.monotonic()+60
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:pass
            row['returncode']=process.returncode;assert process.returncode==0,'Training child failed'
            row['metrics']=r.parse_summary((out/'run.log').read_text())
            row['checkpoint_sha256']=r.digest(out/'checkpoint_pre_eval.pt')
            validate(out,row,plan)
            for peer in records():
                if peer['status']!='completed' or peer['seed']!=seed:continue
                folder=HERE/peer['trial']
                if peer['depth']==depth:
                    assert read(folder/'initialization.json')==read(out/'initialization.json'),'Initialization policy changed across budgets'
                if peer['budget']==budget:
                    assert read(folder/'batches.json')==read(out/'batches.json'),'Paired data mismatch'
                    assert read(folder/'schedule.json')['updates']==read(out/'schedule.json')['updates']
            row.update(status='completed',memory=read(out/'memory.json'))
            log(f'END {out.name} BPB={row["metrics"]["val_bpb"]:.6f}')
        except BaseException as exc:
            if process is not None and process.poll() is None:r.stop_process(process)
            row.update(status='failed',error=repr(exc));log('FAILED '+out.name+' '+repr(exc))
        finally:
            row.update(trial=out.name,wall_seconds=time.monotonic()-start)
            r.write_json(out/'gpu-samples.json',dict(samples=samples,scope='Sampled whole-board memory includes desktop/driver allocations'))
            r.write_json(out/'result.json',row);reports.emit(out)
    return row

def run():
    plan=read(HERE/'matrix-plan.json');assert plan['source_hashes']==sources()
    seed=101
    while True:
        good=[x for x in records() if x['status']=='completed']
        complete={s for s in {x['seed'] for x in good} if sum(x['seed']==s for x in good)==6}
        if seed in complete:seed+=1;continue
        if seed>=104:
            durations=[sum(x['wall_seconds'] for x in good if x['seed']==s) for s in complete]
            estimate=max(durations)*1.25+360 if durations else 4800
            if remaining()<RESERVE+estimate:
                log(f'No further whole seed admitted: remaining={remaining():.0f}s estimate={estimate:.0f}s reserve={RESERVE}s');break
        r.write_json(HERE/f'seed-{seed}-admission.json',dict(seed=seed,remaining_seconds=remaining(),admitted_at=datetime.now(timezone.utc).isoformat(),rule='Complete six frozen cells before next seed; admission uses time only'))
        for budget in (1,2,4):
            for depth in ((6,12) if (seed+budget)%2==0 else (12,6)):
                matches=[x for x in records() if (x['seed'],x['depth'],x['budget'])==(seed,depth,budget)]
                if any(x['status']=='completed' for x in matches):continue
                for attempt in range(len(matches),3):
                    row=trial(depth,budget,seed)
                    from depth_report import write
                    write()
                    try:publish()
                    except Exception as exc:log('Publication deferred, local evidence retained: '+repr(exc))
                    if row['status']=='completed':break
                    # An unclassified failure requires primary-agent diagnosis, never silent retry.
                    raise RuntimeError('Preserved failure requires diagnosis before bounded retry: '+row['trial'])
                else:raise RuntimeError('Three attempts exhausted for required cell')
        seed+=1
    r.write_json(HERE/'training-closed.json',dict(status='completed',closed_at=datetime.now(timezone.utc).isoformat(),complete_seeds=sorted(complete),reason='Next complete paired seed would encroach on reporting reserve'))
    log('Training closed; final evaluation freeze and verification next')

if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare_stream()
    elif action=='preflight':preflight()
    elif action=='freeze':freeze()
    elif action=='publish':publish()
    elif action=='run':run()
    else:raise ValueError(action)
