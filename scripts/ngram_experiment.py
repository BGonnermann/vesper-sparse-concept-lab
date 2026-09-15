"""Frozen four-condition n-gram study: gated preflight, captured runs, compact reports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'runs/autoresearch/ngram-20260915'
REFERENCE = ROOT / 'runs/autoresearch/equal-token-20260915'
sys.path.insert(0,str(ROOT/'scripts'))
import autoresearch as r
import experiment_reports as reports
ORDER = ('D','DG','M','MG')


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


def announce(path):
    log=path.open('x',encoding='utf-8')
    print('LOG READY; child not started: '+str(path),flush=True)
    deadline=time.monotonic()+180
    while not path.with_suffix('.start').exists():
        if time.monotonic()>deadline: raise TimeoutError('Log announcement gate expired')
        time.sleep(.1)
    return log


def offline_ngrams(tape,bos,vocab):
    """Exact suffix occupancy/collisions on training tape only, outside all run timers."""
    import numpy as np
    x=tape[:,0].numpy().reshape(-1,512)
    result={}
    for n in (2,3):
        shape=(x.shape[0],x.shape[1]-n+1)
        valid=np.ones(shape,dtype=bool); encoded=np.zeros(shape,dtype=np.int64)
        for offset in range(n):
            part=x[:,offset:offset+shape[1]]
            valid &= part!=bos
            encoded=encoded*vocab+part
        unique,frequency=np.unique(encoded[valid],return_counts=True)
        h=np.full(unique.shape,n,dtype=np.int64)
        for offset in range(n):
            token=(unique//(vocab**(n-1-offset)))%vocab
            h=(h*1000003+token+1)%2147483647
        addresses=h%8192
        distinct_per_bucket=np.bincount(addresses,minlength=8192)
        accesses=np.bincount(addresses,weights=frequency,minlength=8192).astype(np.int64)
        occupied=int(np.count_nonzero(distinct_per_bucket))
        result[str(n)]={'valid_positions':int(valid.sum()),'all_positions':int(x.size),
            'valid_rate':float(valid.sum()/x.size),'distinct_suffixes':int(unique.size),
            'accessed_rows':occupied,'address_occupancy':occupied/8192,
            'collision_definition':'Distinct valid suffixes beyond the first in each occupied bucket',
            'colliding_extra_suffixes':int(unique.size)-occupied,
            'buckets_with_collisions':int((distinct_per_bucket>1).sum()),
            'distinct_suffixes_per_bucket':distinct_per_bucket.tolist(),
            'training_accesses_per_bucket':accesses.tolist()}
    return result


def prepare():
    import torch
    sys.path.insert(0,str(r.RUNTIME))
    import prepare as data
    import train
    from autoresearch_model import model_class
    from autoresearch_train import batch_order, fixed_schedule
    torch.set_num_threads(2)
    revision=read(HERE/'plan-revision.json')
    assert r.digest(HERE/'frozen-plan.md')==revision['plan_sha256']==r.digest(ROOT/'docs/ngram-memory-plan.md')
    r.verify_runtime(); seal=read(ROOT/'.autoresearch/data-seal.json');r.verify_seal(ROOT/'.autoresearch/cache',seal)
    protocol=r.validate_protocol(read(REFERENCE/'protocol-42.json'))
    assert r.digest(Path(protocol['batch_tape']))==protocol['batch_tape_sha256']=='07e33edbaa75fd2b06e8478436c55a0ab83382035a15e3dd4648026f7a8fb689'
    assert protocol['seed']==42 and protocol['optimizer_updates']==512 and protocol['baseline_timeout_seconds']==900
    tape=torch.load(protocol['batch_tape'],map_location='cpu',weights_only=True)
    tokenizer=data.Tokenizer.from_directory(dataset='tinystories')
    bos=tokenizer.get_bos_token_id();vocab=tokenizer.get_vocab_size()
    assert tape.shape==(8192,2,2,512) and tape.dtype==torch.int64
    assert int(tape.min())>=0 and int(tape.max())<vocab
    assert bool((tape[:,0,:,0]==bos).all())
    assert torch.equal(tape[:,0,:,1:],tape[:,1,:,:-1])
    order=batch_order(8192,42);expected=read(REFERENCE/'order-42.json');assert order==expected['indices']
    h=hashlib.sha256()
    for i in order:h.update(hashlib.sha256(tape[i].numpy().tobytes()).digest())
    assert h.hexdigest()==expected['batch_hash_chain']
    r.write_json(HERE/'boundary-and-batches.json',{'verified':True,'bos_token_id':bos,'vocab_size':vocab,
        'boundary_policy':'Captured packer prepends BOS to every document, including cropped documents; every row starts BOS; no separate boundary metadata',
        'packer_sha256':r.digest(r.RUNTIME/'prepare.py'),'tape_sha256':protocol['batch_tape_sha256'],
        'batch_hash_chain':h.hexdigest(),'microbatches':8192,'training_tokens':8388608})
    r.write_json(HERE/'ngram-tape-diagnostics.json',offline_ngrams(tape,bos,vocab))
    r.write_json(HERE/'protocol.json',protocol)
    assert fixed_schedule()==read(REFERENCE/'expected-schedule.json')
    initial={};groups={}
    for label in ORDER:
        variant='moe' if label.startswith('M') else 'dense'
        candidate=read(ROOT/f'experiments/autoresearch/{variant}-depth6.json')
        if label.endswith('G'):candidate['memory']={'kind':'ngram_v1','bos_token_id':bos}
        r.validate_candidate(candidate);r.write_json(HERE/f'{label}.json',candidate)
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        runtime=train.detect_runtime();train.MAX_SEQ_LEN=512
        config=train.build_model_config(6,vocab,runtime,False)
        with torch.device('meta'):model=model_class(train,candidate)(config)
        model.to_empty(device='cuda');model.init_weights(embed_dtype=torch.bfloat16)
        hashes={n:hashlib.sha256(p.detach().cpu().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}
        initial[label]=hashes;r.write_json(HERE/f'initialization-{label}.json',hashes)
        optimizer=model.setup_optimizer(unembedding_lr=train.UNEMBEDDING_LR,embedding_lr=train.EMBEDDING_LR,
            scalar_lr=train.SCALAR_LR,adam_betas=train.ADAM_BETAS,matrix_lr=.04,weight_decay=train.WEIGHT_DECAY)
        groups[label]=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
        expected_params={'D':26345772,'DG':27443885,'M':47588652,'MG':48686765}
        expected_active={'D':26345772,'DG':26395437,'M':26354988,'MG':26404653}
        report=model.parameter_report()
        assert report['total_parameters']==expected_params[label] and report['active_parameters']==expected_active[label]
        assert model.num_scaling_params()['total']==expected_params[label]
        r.write_json(HERE/f'model-{label}.json',report)
        del optimizer,model
    for control,memory in (('D','DG'),('M','MG')):
        assert all(initial[memory][name]==value for name,value in initial[control].items())
        assert groups[control]==groups[memory][:-1], 'Backbone optimizer groups changed'
    added=lambda label:{k:v for k,v in initial[label].items() if k.startswith('ngram_memory.')}
    assert added('DG') and added('DG')==added('MG')
    for label,variant in (('D','dense'),('M','moe')):
        assert initial[label]==read(REFERENCE/f'initialization-{variant}-42.json')
    print('PASS: frozen plan, boundaries, full tape/order, all parameter counts, paired initialization and unchanged backbone optimizer groups',flush=True)


def preflight(label='preflight'):
    path=HERE/(label+'.log')
    record={'kind':'correctness_preflight','status':'running','plan_revision':read(HERE/'plan-revision.json')}
    try:
        with announce(path) as log:
            for device in ('cpu','cuda'):
                child=subprocess.run([str(r.runtime_python()),'-B','-m','unittest','discover','-s','tests','-v'],
                    cwd=ROOT,env=dict(r.environment(),MOE_TEST_DEVICE=device),stdout=log,stderr=subprocess.STDOUT,timeout=180)
                log.write(f'\n{device} correctness exit: {child.returncode}\n');log.flush()
                if child.returncode:raise RuntimeError(device+' correctness failed; inspect '+str(path))
            child=subprocess.run([str(r.runtime_python()),'-B','-u',str(Path(__file__)),'prepare'],
                cwd=ROOT,env=r.environment(),stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if child.returncode:raise RuntimeError('Frozen plan/data/model preflight failed')
        record.update(status='completed',passed=True,log=path.name,log_sha256=r.digest(path),
            source_hashes={n:r.digest(ROOT/'scripts'/n) for n in r.PROJECT_FILES},
            test_hashes={p.name:r.digest(p) for p in (ROOT/'tests').glob('test_*.py')})
        r.write_json(HERE/'preflight.json',record)
    except BaseException as exc:
        record.update(status='failed',passed=False,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        r.write_json(HERE/(label+'-result.json'),record)
        reports.emit(HERE);reports.index()
    print('ALL PREFLIGHT GATES PASSED',flush=True)


def trial(label):
    if label not in ORDER:raise ValueError('Only D, DG, M and MG authorized')
    for prior in ORDER[:ORDER.index(label)]:assert read(HERE/prior/'result.json')['status']=='completed'
    gate=read(HERE/'preflight.json');assert gate['passed']
    assert gate['source_hashes']=={n:r.digest(ROOT/'scripts'/n) for n in r.PROJECT_FILES}
    assert gate['log_sha256']==r.digest(HERE/gate['log'])
    revision=read(HERE/'plan-revision.json');assert revision['plan_sha256']==r.digest(HERE/'frozen-plan.md')
    setup=r.verify_runtime();seal=read(ROOT/'.autoresearch/data-seal.json');r.verify_seal(ROOT/'.autoresearch/cache',seal)
    candidate=r.validate_candidate(read(HERE/f'{label}.json'));protocol=r.validate_protocol(read(HERE/'protocol.json'))
    out=HERE/label;out.mkdir(exist_ok=False)
    record=dict(kind='fixed_updates',condition=candidate['feedforward'],label=label,candidate=candidate,
        protocol=protocol,seed=42,record_version=5,upstream=setup,data_seal=seal,
        git_commit=revision['git_revision'],git_dirty=True,plan_revision=revision,
        git_metadata_note='Base revision only; uncommitted implementation identified by captured executed hashes, not later publication commit.',
        orchestrator_sha256=r.digest(Path(__file__)),preflight_sha256=r.digest(HERE/'preflight.json'))
    record.update({key:r.digest(ROOT/'scripts'/n) for n,key in r.PROJECT_FILES.items()})
    snapshot=r.capture_run_snapshot(out,record);record.update(snapshot_files=snapshot,status='prepared')
    r.write_json(out/'result.json',record);process=None;start=None
    try:
        with announce(out/'run.log') as log:
            start=time.monotonic();record['status']='running';r.write_json(out/'result.json',record)
            process=subprocess.Popen(r.snapshot_command(r.runtime_python(),out),cwd=out,env=r.environment(),
                stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            record['returncode']=process.wait(timeout=900)
        if record['returncode']:raise RuntimeError('Training exited '+str(record['returncode']))
        record['metrics']=r.parse_summary((out/'run.log').read_text())
        record['artifacts']=r.validate_run_artifacts(out,record)
        record['execution']=r.validate_execution(out,snapshot)
        t=record['artifacts']['training'];batches=read(out/'batches.json')
        assert t['optimizer_updates']==record['metrics']['num_steps']==512
        assert t['training_tokens']==batches['training_tokens']==8388608 and t['timed_training_tokens']==8208384
        expected=read(REFERENCE/'order-42.json')
        assert batches['consumed_indices']==expected['indices'] and batches['consumed_batch_hash_chain']==expected['batch_hash_chain']
        assert read(out/'initialization.json')['parameters']==read(HERE/f'initialization-{label}.json')
        assert read(out/'schedule.json')['updates']==read(REFERENCE/'expected-schedule.json')
        assert record['artifacts']['model']==read(HERE/f'model-{label}.json')
        record['memory']=read(out/'memory.json');record['status']='completed'
    except BaseException as exc:
        if process is not None and process.poll() is None:r.stop_process(process)
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['wall_seconds']=time.monotonic()-start if start else 0
        r.write_json(out/'result.json',record)
        r.emit_compact_report(out)
    print(json.dumps({k:record[k] for k in ('status','metrics','wall_seconds','memory')}),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='preflight':preflight(sys.argv[2] if len(sys.argv)>2 else 'preflight')
    else:trial(sys.argv[1])
