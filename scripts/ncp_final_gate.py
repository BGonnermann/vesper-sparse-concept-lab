"""Require current complete evidence before labeling the campaign completed."""
from datetime import datetime,timezone

import autoresearch as r
import ncp_campaign as c


def validate_inventory(rows,audit,hashes):
    terminal={'completed','failed','timeout','invalid','interrupted'}
    assert all(row['status'] in terminal for row in rows),'Active or prepared trial remains'
    complete={row['trial'] for row in rows if row['status']=='completed'}
    other={row['trial'] for row in rows if row['status']!='completed'}
    assert audit['status']=='completed' and audit['all_sources_committed_and_byte_verified']
    assert {row['trial'] for row in audit['verified_trials']}==complete,'Audit completed inventory differs'
    assert {row['trial'] for row in audit['preserved_noncompleted']}==other,'Audit failure inventory differs'
    assert audit['trial_result_hashes']==hashes,'Trial results changed since audit'


def ready(rows):
    with c.gpu_lock():
        audit=c.read(c.HERE/'audit-result.json')
        hashes={p.parent.name:r.digest(p) for p in c.HERE.glob('trial-*/result.json')}
        validate_inventory(rows,audit,hashes)
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(audit['measured_at_utc'])).total_seconds()
        assert 0<=age<=900,'Final audit is older than15minutes'
        gate=c.read(c.HERE/'active-preflight.json')
        assert gate['passed'] and gate['source_hashes']==c.sources()
        assert gate['test_hashes']=={p.name:r.digest(p) for p in (r.ROOT/'tests').glob('test_*.py')}
        from ncp_frozen_pairs import collect
        frozen=collect(c.read(c.HERE/'confirmation-selection.json'))
        assert all(item['completed_pairs']==4 for item in frozen['summaries'].values())
        for name in ('depth-grid-result.json','crossed-order-result.json','module-initialization-result.json',
                     'feedback-interventions-result.json'):
            assert c.read(c.HERE/name)['status']=='completed',name
        no_future=c.HERE/'no-future-result.json'
        if no_future.exists():
            assert c.read(no_future)['status'] in ('completed','failed')
        else:
            assert c.read(c.HERE/'no-future-budget-result.json')['status']=='skipped'
        assert c.read(c.HERE/'batch-evidence-result.json')['status']=='completed'
        assert c.read(c.HERE/'basis-evidence-result.json')['status']=='completed'
        for name in ('tape-origin-audit-result.json','split-audit-result.json','official-source-byte-audit-result.json'):
            assert c.read(c.HERE/name)['status']=='completed',name
        replay=c.read(c.HERE/'checkpoint-replay-result.json')
        assert replay['status']=='completed'
        complete={row['trial'] for row in rows if row['status']=='completed'}
        assert {row['trial'] for row in replay['outcomes']}==complete
        assert len(replay['outcomes'])==len(complete)==replay['completed_checkpoints']==replay['expected_checkpoints']
        baseline=c.read(c.HERE/'concept-baselines-result.json')
        assert baseline['status']=='completed' and {row['seed'] for row in baseline['outcomes']}=={43,44,45,46}
        for aggregate,plan_path,driver_path in (
            (replay,c.HERE/'checkpoint-replay/plan.json',c.HERE/'checkpoint-replay-driver.py'),
            (baseline,c.HERE/'concept-baselines-plan.json',c.HERE/'concept-baselines-driver.py')):
            assert r.digest(plan_path)==aggregate['plan_sha256']
            plan=c.read(plan_path)
            assert r.digest(driver_path)==plan['script_sha256']
            for row in aggregate['outcomes']:
                path=c.HERE/row['result_path']
                assert r.digest(path)==row['result_sha256'],'Probe receipt changed'
                child=c.read(path)
                assert child['status']=='completed' and child['result_sha256']==hashes[row['trial']]
                assert child['script_sha256']==plan['script_sha256'] and child['settings_immutable']
                original=c.read(c.HERE/row['trial']/'result.json')
                assert child['checkpoint_sha256']==original['checkpoint_sha256']
                assert child['source_hashes']==original['snapshot_files']
                assert abs(child['bpb']-original['metrics']['val_bpb'])<=1e-6
                assert child['persistent_state_immutable'] and child['state_before']==child['state_after']
                assert child['accounting']['tokens']==65536 and child['accounting']['batches']==64
                assert r.digest(c.HERE/row['log_path'])==row['log_sha256']
    return dict(audit_sha256=r.digest(c.HERE/'audit-result.json'),audit_age_seconds=age,
        attempted_trials=len(rows),verified_completed_trials=len(audit['verified_trials']))
