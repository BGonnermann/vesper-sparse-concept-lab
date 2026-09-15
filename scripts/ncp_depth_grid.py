"""Complete the predeclared dense depth/width grid without tuning its scores."""
import statistics
import sys

import autoresearch as r
import ncp_campaign as c
import ncp_search as search

LABELS=('D6','D12-W384','D6-W768','D12')
SEEDS=(43,44)


def summary():
    found={}
    for path in c.HERE.glob('trial-*/result.json'):
        row=c.read(path);key=(row['label'],row['seed'])
        if key[0] not in LABELS or key[1] not in SEEDS or row['status']!='completed': continue
        assert key not in found,('Repeated depth-grid condition',key)
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
        plan=dict(kind='depth_width_grid_plan',seeds=list(SEEDS),
            candidates={label:c.candidate(label) for label in LABELS},source_hashes=c.sources(),
            planned_at=c.datetime.now(c.timezone.utc).isoformat(),
            hypothesis='Separate the already confirmed joint depth/width gain into fixed-width depth and fixed-depth width comparisons; no tuning')
        r.write_json(path,plan)
    for seed in SEEDS:
        for label,control in [('D12-W384','D6'),('D6-W768','D12')]:
            assert c.sources()==plan['source_hashes']
            assert c.candidate(label)==plan['candidates'][label]
            peers=[x for x in search.records() if x['label']==label and x['seed']==seed]
            if peers:
                assert len(peers)==1 and peers[0]['status']=='completed' and peers[0]['candidate']==plan['candidates'][label]
                continue
            result=c.trial(label,seed,plan['hypothesis'],'depth_width_confirmation',control)
            if result is None: return
            if result['status']!='completed': raise RuntimeError('Preserved depth-grid failure needs diagnosis')
            r.write_json(c.HERE/'depth-grid-result.json',summary())
            from ncp_report import write
            write()
            if publish:
                try: search.publish()
                except Exception as exc: c.log('Publication deferred; depth-grid result retained: '+repr(exc))


if __name__=='__main__': main('--publish' in sys.argv)
