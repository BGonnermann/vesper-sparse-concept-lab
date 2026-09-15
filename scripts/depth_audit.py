"""Fresh byte and pairing audit for the longer-training campaign."""
from datetime import datetime,timezone
import hashlib
import shutil
import subprocess

import autoresearch as r
import depth_campaign as c


def audit(final=False):
    plan=c.read(c.HERE/'matrix-plan.json')
    assert plan['source_hashes']==c.sources()
    assert r.digest(c.HERE/'PLAN.md')==plan['plan_sha256']
    assert r.digest(c.HERE/'controller.py')==plan['controller_sha256']
    assert r.digest(r.ROOT/'scripts/depth_report.py')==plan['decision_source_sha256']
    assert r.verify_runtime()==plan['upstream']
    r.verify_seal(r.STATE/'cache',plan['data_seal'])
    stream=c.read(c.HERE/'stream-result.json')
    assert r.digest(c.HERE/'stream-result.json')==plan['stream_sha256']
    assert r.digest(c.HERE/'batches.pt')==stream['tape_sha256']
    assert r.digest(c.HERE/'batch-hashes.json')==stream['batch_hashes_sha256']
    assert r.digest(c.HERE/'stream-prepare.py')==stream['source_sha256']
    assert stream['verified_reconstruction_microbatches']==32768
    rows=c.records();successful=[];failed=[];snapshots={}
    for row in rows:
        assert row['status'] in ('completed','failed'),'Active trial during final audit'
        out=c.HERE/row['trial']
        if row['status']!='completed':failed.append(dict(trial=row['trial'],error=row.get('error')));continue
        c.validate(out,row,plan)
        for name,h in row['snapshot_files'].items():
            if name.endswith('.py'):
                snapshots[h]=name
        successful.append(dict(trial=row['trial'],seed=row['seed'],depth=row['depth'],budget=row['budget'],result_sha256=r.digest(out/'result.json')))
    by={(x['seed'],x['depth'],x['budget']):x for x in successful}
    assert len(by)==len(successful)
    complete=[]
    for seed in sorted({x['seed'] for x in successful}):
        if not all((seed,d,b) in by for d in (6,12) for b in (1,2,4)):continue
        complete.append(seed)
        for budget in (1,2,4):
            a,b=[c.HERE/by[seed,d,budget]['trial'] for d in (6,12)]
            assert c.read(a/'batches.json')==c.read(b/'batches.json')
            assert c.read(a/'schedule.json')['updates']==c.read(b/'schedule.json')['updates']
        for depth in (6,12):
            initial=None;previous=[]
            for budget in (1,2,4):
                folder=c.HERE/by[seed,depth,budget]['trial']
                state=c.read(folder/'initialization.json')
                if initial is not None:assert initial==state
                initial=state
                order=c.read(folder/'batches.json')['consumed_indices']
                assert order[:len(previous)]==previous;previous=order
    assert all(s in complete for s in (101,102,103)),'Mandatory paired seeds incomplete'
    evaluation=None
    if final:
        frozen=c.read(c.HERE/'evaluation-freeze.json');evaluation=c.read(c.HERE/'final-evaluation-result.json')
        assert frozen['matrix_sha256']==r.digest(c.HERE/'matrix-plan.json')
        assert evaluation['status']=='completed' and evaluation['freeze_sha256']==r.digest(c.HERE/'evaluation-freeze.json')
        assert {x['trial']:x['result_sha256'] for x in frozen['checkpoints']}=={x['trial']:x['result_sha256'] for x in successful}
        for name,key in [('evaluation-driver.py','evaluator_sha256'),('closeout-driver.py','closeout_sha256')]:
            assert r.digest(c.HERE/name)==frozen[key];snapshots[frozen[key]]=name
        expected={('val',x['trial']) for x in frozen['checkpoints']}|{('test',x) for x in frozen['test_trials']}
        actual={(x['split'],x['trial']) for x in evaluation['outcomes']}
        assert actual==expected and len(actual)==len(evaluation['outcomes'])
        split_batches={}
        for outcome in evaluation['outcomes']:
            path=c.HERE/outcome['result_path'];log=c.HERE/outcome['log_path'];child=c.read(path)
            assert r.digest(path)==outcome['result_sha256'] and r.digest(log)==outcome['log_sha256']
            original=c.read(c.HERE/outcome['trial']/'result.json')
            assert child['status']=='completed' and child['checkpoint_sha256']==original['checkpoint_sha256']
            assert child['result_sha256']==r.digest(c.HERE/outcome['trial']/'result.json')
            assert child['source_hashes']==original['snapshot_files']
            assert child['script_sha256']==frozen['evaluator_sha256'] and child['freeze_sha256']==evaluation['freeze_sha256']
            assert child['persistent_state_immutable'] and child['state_before']==child['state_after']
            assert child['dispatch']['actual_splits']==[outcome['split']]
            assert child['accounting']['tokens']==65536 and child['accounting']['batches']==64 and child['accounting']['target_bytes']>0
            if outcome['split']=='val':assert child['absolute_validation_bpb_difference']<=1e-6
            signature=(child['batch_sha256'],child['accounting']['target_bytes'])
            assert split_batches.setdefault(outcome['split'],signature)==signature
            assert child['bpb']==outcome['bpb']
        assert split_batches['val'][0]!=split_batches['test'][0],'Test and validation batches must differ'
    committed=[]
    for h,name in snapshots.items():
        path=f'reports/source-snapshots/{h}.py'
        data=subprocess.check_output(['git','show','HEAD:'+path],cwd=r.ROOT,timeout=15)
        assert hashlib.sha256(data).hexdigest()==h,('Committed source bytes changed',path)
        committed.append(path)
    result=dict(kind='depth_long_audit',status='completed',measured_at=datetime.now(timezone.utc).isoformat(),
        successful=successful,failed=failed,complete_seeds=complete,committed_sources=committed,
        all_sources_committed_and_byte_verified=True,
        final_evaluation_verified=final,evaluation_count=len(evaluation['outcomes']) if evaluation else 0,
        trial_result_hashes={x['trial']:r.digest(c.HERE/x['trial']/'result.json') for x in rows},
        matrix_sha256=r.digest(c.HERE/'matrix-plan.json'),script_sha256=r.digest(__import__('pathlib').Path(__file__)),
        checkpoint_bytes=sum(p.stat().st_size for p in c.HERE.glob('trial-*/checkpoint*.pt')),
        free_disk_bytes=shutil.disk_usage(r.ROOT).free)
    r.write_json(c.HERE/'audit-result.json',result)
    c.log(f'AUDIT PASS: {len(successful)} checkpoints, {len(complete)} complete paired seeds')
    return result


if __name__=='__main__':
    with c.gpu_lock():audit('--final' in __import__('sys').argv)
