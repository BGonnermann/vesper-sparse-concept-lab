"""Complete the predeclared dense depth/width grid without tuning its scores."""
import statistics
import sys

import autoresearch as r
import ncp_campaign as c
import ncp_search as search

LABELS=('D6','D12-W384','D6-W768','D12')
SEEDS=(43,44)


def validate(directory,plan,label,seed):
    row=c.read(directory/'result.json')
    assert row['status']=='completed' and (row['label'],row['seed'])==(label,seed)
    assert row['candidate']==plan['candidates'][label]
    protocol=dict(plan['protocol'],seed=seed)
    assert row['protocol']==protocol and row['data_seal']==plan['data_seal'] and row['upstream']==plan['upstream']
    reused=plan['reused_controls'].get(f'{label}:{seed}')
    if reused:
        assert directory.name==reused['trial'] and r.digest(directory/'result.json')==reused['result_sha256']
        sources=reused['source_hashes']
    else: sources=plan['source_hashes']
    for name,digest in sources.items(): assert row['snapshot_files']['source/project/'+name]==digest
    assert c.read(directory/'model.json')['total_parameters']==plan['parameter_counts'][label]
    r.validate_execution(directory,row['snapshot_files']);r.validate_run_artifacts(directory,row)


def summary():
    plan=c.read(c.HERE/'depth-grid-plan.json') if (c.HERE/'depth-grid-plan.json').exists() else None
    found={}
    for path in c.HERE.glob('trial-*/result.json'):
        row=c.read(path);key=(row['label'],row['seed'])
        if key[0] not in LABELS or key[1] not in SEEDS or row['status']!='completed': continue
        assert key not in found,('Repeated depth-grid condition',key)
        if plan: validate(path.parent,plan,*key)
        assert row['candidate']==c.candidate(key[0])
        model=c.read(path.parent/'model.json')
        found[key]=(path,row,model)
    comparisons=[]
    for seed in SEEDS:
        for axis,left,right in [('depth_at_width384','D12-W384','D6'),('depth_at_width768','D12','D6-W768'),
                ('width_at_depth6','D6-W768','D6'),('width_at_depth12','D12','D12-W384')]:
            if (left,seed) not in found or (right,seed) not in found: continue
            a,x,m=found[(left,seed)];b,y,n=found[(right,seed)]
            assert x['protocol']==y['protocol'] and x['data_seal']==y['data_seal']
            assert c.read(a.parent/'batches.json')==c.read(b.parent/'batches.json')
            assert c.read(a.parent/'schedule.json')['updates']==c.read(b.parent/'schedule.json')['updates']
            for path,row in ((a,x),(b,y)): r.validate_execution(path.parent,row['snapshot_files'])
            comparisons.append(dict(axis=axis,seed=seed,candidate=left,control=right,
                candidate_trial=a.parent.name,control_trial=b.parent.name,
                candidate_result_sha256=r.digest(a),control_result_sha256=r.digest(b),
                candidate_bpb=x['metrics']['val_bpb'],control_bpb=y['metrics']['val_bpb'],
                delta_bpb=x['metrics']['val_bpb']-y['metrics']['val_bpb'],
                candidate_parameters=m['total_parameters'],control_parameters=n['total_parameters'],
                update_time_ratio=c.read(a.parent/'fixed-training.json')['all_update_seconds']/c.read(b.parent/'fixed-training.json')['all_update_seconds']))
    means={axis:statistics.mean(x['delta_bpb'] for x in comparisons if x['axis']==axis) for axis in {x['axis'] for x in comparisons}}
    return dict(kind='dense_depth_width_grid',status='completed' if len(found)==8 else 'running',
        comparisons=comparisons,mean_deltas=means,complete_conditions=len(found),expected_conditions=8,
        scope='Predeclared depth6/12 by width384/768, seeds43/44, matrixLR .04, fixed tokens; no configuration selection')


def main(publish=False):
    path=c.HERE/'depth-grid-plan.json'
    if path.exists(): plan=c.read(path)
    else:
        fit=c.read(c.HERE/'fit-result.json')['results']
        reused={}
        for file in c.HERE.glob('trial-*/result.json'):
            row=c.read(file)
            if row['label'] in ('D6','D12') and row['seed'] in SEEDS and row['status']=='completed':
                key=f'{row["label"]}:{row["seed"]}'
                assert key not in reused
                reused[key]=dict(trial=file.parent.name,result_sha256=r.digest(file),
                    source_hashes={name:row['snapshot_files']['source/project/'+name] for name in r.PROJECT_FILES})
        plan=dict(kind='depth_width_grid_plan',seeds=list(SEEDS),
            candidates={label:c.candidate(label) for label in LABELS},source_hashes=c.sources(),
            protocol=c.read(r.ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json'),
            data_seal=c.read(r.ROOT/'.autoresearch/data-seal.json'),upstream=r.verify_runtime(),reused_controls=reused,
            parameter_counts={label:fit[label]['model']['total_parameters'] for label in LABELS},
            forecasts={label:fit[label]['conservative_trial_forecast_seconds'] for label in ('D12-W384','D6-W768')},
            planned_at=c.datetime.now(c.timezone.utc).isoformat(),
            hypothesis='Separate the already confirmed joint depth/width gain into fixed-width depth and fixed-depth width comparisons; no tuning')
        r.write_json(path,plan)
    pending=[(label,seed) for label in ('D12-W384','D6-W768') for seed in SEEDS
        if not any(x['label']==label and x['seed']==seed for x in search.records())]
    if sum(plan['forecasts'][label]+10 for label,seed in pending)>c.remaining()-5400:
        c.log('Optional depth grid deferred by90-minute report reserve forecast')
        r.write_json(c.HERE/'depth-grid-entry-result.json',dict(kind='depth_grid_entry',status='not_entered',reason='Budget forecast'))
        return
    for seed in SEEDS:
        for label,control in [('D12-W384','D6'),('D6-W768','D12')]:
            assert c.sources()==plan['source_hashes']
            assert c.candidate(label)==plan['candidates'][label]
            assert dict(c.read(r.ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json'),seed=seed)==dict(plan['protocol'],seed=seed)
            assert c.read(r.ROOT/'.autoresearch/data-seal.json')==plan['data_seal'] and r.verify_runtime()==plan['upstream']
            peers=[x for x in search.records() if x['label']==label and x['seed']==seed]
            if peers:
                assert len(peers)==1 and peers[0]['status']=='completed' and peers[0]['candidate']==plan['candidates'][label]
                directory=next(p.parent for p in c.HERE.glob('trial-*/result.json') if c.read(p)==peers[0])
                validate(directory,plan,label,seed)
                continue
            if plan['forecasts'][label]+10>c.remaining()-5400:
                c.log('Depth grid reached confirmation/report reserve; remaining cells unavailable');return
            result=c.trial(label,seed,plan['hypothesis'],'depth_width_confirmation',control)
            if result is None: return
            if result['status']!='completed': raise RuntimeError('Preserved depth-grid failure needs diagnosis')
            directory=next(p.parent for p in c.HERE.glob('trial-*/result.json') if c.read(p)==result)
            validate(directory,plan,label,seed)
            r.write_json(c.HERE/'depth-grid-result.json',summary())
            from ncp_report import write
            write()
            if publish:
                try: search.publish()
                except Exception as exc: c.log('Publication deferred; depth-grid result retained: '+repr(exc))


if __name__=='__main__': main('--publish' in sys.argv)
