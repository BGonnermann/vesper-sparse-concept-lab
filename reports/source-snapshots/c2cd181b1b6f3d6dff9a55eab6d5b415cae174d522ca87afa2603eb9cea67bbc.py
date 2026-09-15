"""Sequential, deadline-bounded NCP campaign using captured fixed-token execution."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from contextlib import contextmanager
from functools import wraps
import msvcrt

import autoresearch as r
import experiment_reports as reports

ROOT=r.ROOT
HERE=ROOT/'runs/autoresearch/ncp-20260915'
DEADLINE=datetime.fromisoformat('2026-09-15T19:41:32+00:00')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def remaining(): return (DEADLINE-datetime.now(timezone.utc)).total_seconds()
def can_launch(seconds): return seconds>=2100
def sources(): return {n:r.digest(ROOT/'scripts'/n) for n in r.PROJECT_FILES}


def log(message):
    line=datetime.now(timezone.utc).isoformat()+' '+message
    with (HERE/'campaign.log').open('a',encoding='utf-8') as f: f.write(line+'\n')
    print(line,flush=True)


@contextmanager
def gpu_lock(path=None):
    """OS-owned byte lock: retained file, automatically released on process exit."""
    path=HERE/'gpu-run.lock' if path is None else Path(path)
    with path.open('a+b') as handle:
        if handle.tell()==0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:
            raise RuntimeError('Another campaign GPU owner holds '+str(path)) from exc
        try: yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def exclusive_gpu(function):
    @wraps(function)
    def wrapped(*args,**kwargs):
        with gpu_lock(): return function(*args,**kwargs)
    return wrapped


def promising(record, control_bpb):
    return (record['status']=='completed' and not (record.get('ncp_health') or {}).get('collapsed',False)
        and record['metrics']['val_bpb']<=control_bpb-.001)


def base_ncp():
    return dict(kind='ncp_v1',chunk_size=4,layers=2,entries=64,after_layer=0,
        prediction_weight=1.,vq_weight=1.,ce_weight=0.,lr=.001,feedback_scale=1.,mode='feedback')


def candidate(label):
    value=dict(depth=6,matrix_lr=.04,feedforward='dense')
    if label=='D12': value['depth']=12
    elif label!='D6':
        value['ncp']=base_ncp()
        if label=='AUX': value['ncp']['mode']='auxiliary'
        elif label!='NCP':
            value=read(HERE/'candidates'/f'{label}.json')
    return r.validate_candidate(value)


def health(path):
    data=read(path)
    if not data['enabled']: return None
    samples=data['samples']
    counts=[[sum(s['target_counts'][i][j] for s in samples) for j in range(len(samples[0]['target_counts'][i]))] for i in range(len(samples[0]['target_counts']))]
    perplexities=[]
    for row in counts:
        probabilities=[x/sum(row) for x in row if x]
        perplexities.append(math.exp(-sum(p*math.log(p) for p in probabilities)))
    used=[sum(x>0 for x in row) for row in counts]
    return dict(collapsed=any(x<4 for x in used) or min(perplexities)<2,
        used_entries=used,target_perplexity=perplexities,target_counts=counts,
        mean_prediction_entropy=sum(s['mean_prediction_entropy'] for s in samples)/len(samples),
        mean_next_concept_accuracy=sum(s['next_concept_accuracy'] for s in samples)/len(samples),
        mean_next_concept_mse=sum(s['next_concept_mse'] for s in samples)/len(samples),
        mean_feedback_rms=sum(s['feedback_rms'] for s in samples)/len(samples),
        mean_hidden_rms=sum(s['hidden_rms'] for s in samples)/len(samples),
        training_loss_means=data['training_loss_means'],
        final_update_gradient_norms=data['final_update_gradient_norms'],
        final_update_parameter_delta_norms=data['final_update_parameter_delta_norms'])


@exclusive_gpu
def preflight():
    started=time.monotonic()
    receipt=dict(kind='ncp_correctness',status='running')
    try:
        r.verify_runtime(); r.verify_seal(ROOT/'.autoresearch/cache',read(ROOT/'.autoresearch/data-seal.json'))
        for device in ('cpu','cuda'):
            path=HERE/f'preflight-{device}-{int(time.time())}.log'
            log('Correctness '+device+' log='+str(path))
            with path.open('x',encoding='utf-8') as f:
                result=subprocess.run([str(r.runtime_python()),'-B','-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
                    env=dict(r.environment(),MOE_TEST_DEVICE=device),stdout=f,stderr=subprocess.STDOUT,timeout=240)
            if result.returncode: raise RuntimeError('Correctness failed: '+str(path))
            receipt[device]=dict(log=str(path),sha256=r.digest(path))
        path=HERE/f'fit-{int(time.time())}.log'
        with path.open('x',encoding='utf-8') as f:
            result=subprocess.run([str(r.runtime_python()),'-B','-u',str(ROOT/'scripts/ncp_gate.py')],cwd=ROOT,
                env=r.environment(),stdout=f,stderr=subprocess.STDOUT,timeout=300)
        if result.returncode: raise RuntimeError('Synthetic fit failed: '+str(path))
        receipt.update(status='completed',passed=True,source_hashes=sources(),
            test_hashes={p.name:r.digest(p) for p in (ROOT/'tests').glob('test_*.py')},fit_sha256=r.digest(HERE/'fit-result.json'))
        receipt['fit_file']=read(HERE/'fit-result.json')['artifact_file']
        log('CPU/CUDA correctness and synthetic fit passed')
    except BaseException as exc:
        receipt.update(status='failed',passed=False,error=repr(exc)); raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started
        r.write_json(HERE/f'preflight-{int(time.time())}-result.json',receipt)
        r.write_json(HERE/'active-preflight.json',receipt)
        reports.emit(HERE); reports.index()


@exclusive_gpu
def trial(label, seed, hypothesis, phase='screen', control='D6'):
    if not can_launch(remaining()):
        log('Report reserve reached; no new child'); return None
    gate=read(HERE/'active-preflight.json')
    assert gate['passed'] and gate['source_hashes']==sources(),'Shared source gate mismatch'
    assert gate['test_hashes']=={p.name:r.digest(p) for p in (ROOT/'tests').glob('test_*.py')},'Test gate mismatch'
    free=shutil.disk_usage(ROOT).free
    if free<20*2**30: raise RuntimeError('Disk below 20GiB reserve')
    records=sorted(HERE.glob('trial-*/result.json'))
    out=HERE/f'trial-{len(records)+1:04d}-{label}-s{seed}'
    out.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),out/'orchestrator.py')
    selection=dict(label=label,seed=seed,phase=phase,hypothesis=hypothesis,control=control,
        selected_at=datetime.now(timezone.utc).isoformat(),remaining_seconds=remaining(),free_disk_bytes=free)
    r.write_json(out/'selection.json',selection)
    log(f'START {out.name}: {hypothesis}; log={out / "run.log"}')
    metadata=dict(kind='fixed_updates',status='prepared',metrics=None,label=label,seed=seed,phase=phase,selection=selection)
    process=None; start=time.monotonic(); gpu_samples=[]
    try:
        setup=r.verify_runtime(); seal=read(ROOT/'.autoresearch/data-seal.json')
        r.verify_seal(ROOT/'.autoresearch/cache',seal)
        config=candidate(label)
        protocol=read(ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json'); protocol['seed']=seed
        r.validate_protocol(protocol)
        metadata.update(condition='dense',candidate=config,protocol=protocol,upstream=setup,data_seal=seal,
            record_version=7,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            orchestrator_sha256=r.digest(Path(__file__)),preflight_sha256=r.digest(HERE/'active-preflight.json'))
        metadata.update({key:r.digest(ROOT/'scripts'/name) for name,key in r.PROJECT_FILES.items()})
        snapshot=r.capture_run_snapshot(out,metadata); metadata['snapshot_files']=snapshot
        metadata['status']='running'; r.write_json(out/'result.json',metadata)
        with (out/'run.log').open('x',encoding='utf-8') as f:
            process=subprocess.Popen(r.snapshot_command(r.runtime_python(),out),cwd=out,
                env=r.environment(),stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            end=time.monotonic()+min(900,remaining()-1200)
            next_heartbeat=time.monotonic()+60
            while process.poll() is None:
                if time.monotonic()>end: raise TimeoutError('900-second child deadline')
                try:
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                    gpu_samples.append(gpu)
                except (OSError,subprocess.SubprocessError): pass
                if time.monotonic()>=next_heartbeat:
                    lines=(out/'run.log').read_text(errors='replace').splitlines()
                    log(out.name+' '+next((x for x in reversed(lines) if x.startswith('step ')), 'child initializing/evaluating'))
                    next_heartbeat=time.monotonic()+60
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: pass
        metadata['returncode']=process.returncode
        if process.returncode: raise RuntimeError('Training child exit '+str(process.returncode))
        metadata['metrics']=r.parse_summary((out/'run.log').read_text())
        metadata['artifacts']=r.validate_run_artifacts(out,metadata)
        metadata['execution']=r.validate_execution(out,snapshot)
        from autoresearch_train import batch_order,fixed_schedule
        batches=read(out/'batches.json'); training=metadata['artifacts']['training']
        assert training['optimizer_updates']==metadata['metrics']['num_steps']==512
        assert training['training_tokens']==batches['training_tokens']==8388608
        assert batches['consumed_indices']==batch_order(8192,seed)
        assert read(out/'schedule.json')['updates']==fixed_schedule()
        assert read(out/'evaluation-immutability.json')['verified']
        checkpoint=out/'checkpoint_pre_eval.pt'
        assert checkpoint.exists(),'Required checkpoint not retained'
        metadata['checkpoint_sha256']=r.digest(checkpoint)
        for p in records:
            peer=read(p)
            if peer['status']!='completed' or peer['seed']!=seed: continue
            assert batches==read(p.parent/'batches.json'),'Paired data order mismatch'
            assert protocol==peer['protocol'],'Paired protocol mismatch'
            if peer['label']==control and peer['candidate']['depth']==config['depth']:
                old=read(p.parent/'initialization.json')['parameters']; new=read(out/'initialization.json')['parameters']
                common={n:h for n,h in old.items() if n in new and not n.startswith('ncp.')}
                assert common and all(new[n]==h for n,h in common.items()),'Shared initialization mismatch'
                r.write_json(out/'initialization-pairing.json',dict(control=str(p.parent),verified_tensors=len(common)))
        metadata.update(memory=read(out/'memory.json'),ncp_health=health(out/'ncp-diagnostics.json'),status='completed')
        if metadata['ncp_health'] is not None:
            r.write_json(out/'ncp-health.json',metadata['ncp_health'])
        log(f'END {out.name}: BPB={metadata["metrics"]["val_bpb"]:.6f}; collapsed={metadata["ncp_health"]["collapsed"] if metadata["ncp_health"] else None}')
    except BaseException as exc:
        if process is not None and process.poll() is None: r.stop_process(process)
        metadata.update(status='failed',error=repr(exc))
        log('FAILED '+out.name+' '+repr(exc))
    finally:
        metadata['wall_seconds']=time.monotonic()-start
        if (out/'checkpoint_failure.pt').exists():
            metadata['partial_checkpoint_sha256']=r.digest(out/'checkpoint_failure.pt')
        r.write_json(out/'gpu-samples.json',dict(samples=gpu_samples,scope='Whole-board sampled values include desktop/other processes; not continuous peak'))
        r.write_json(out/'result.json',metadata); r.emit_compact_report(out)
        try:
            from ncp_report import write
            write()
        except Exception as exc:
            log('Campaign summary retry required; local trial retained: '+repr(exc))
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['preflight','trial'])
    parser.add_argument('--label'); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--hypothesis'); parser.add_argument('--phase',default='screen'); parser.add_argument('--control',default='D6')
    args=parser.parse_args()
    if args.action=='preflight': preflight()
    else:
        assert args.label and args.hypothesis
        result=trial(args.label,args.seed,args.hypothesis,args.phase,args.control)
        if result and result['status']!='completed': raise SystemExit(1)
