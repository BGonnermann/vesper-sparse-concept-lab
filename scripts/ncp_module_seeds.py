"""Two conditional NCP-module initializer swaps; no architecture or seed selection."""
import copy
import hashlib
from pathlib import Path
import sys

import autoresearch as r
import ncp_campaign as c
import ncp_crossed_order as crossed
import ncp_search as search

LEVELS = (42, 45)
DRIVER_SOURCE = Path(__file__).read_bytes()
RESERVE = 5400
FORECAST = 930


def strip_seeds(value):
    return {k:v for k,v in value.items() if k not in ('seed','batch_order_seed','ncp_initialization_seed')}


def entry():
    path = c.HERE / 'module-initialization-entry-result.json'
    if path.exists():
        result = c.read(path)
        assert r.digest(c.HERE/'crossed-order-result.json') == result['crossed_result_sha256']
        return result
    evidence = crossed.summary(c.read(c.HERE/'crossed-order-plan.json'))
    assert evidence['status']=='completed'
    d = {(x['initialization_seed'],x['batch_order_seed']):x['delta_bpb'] for x in evidence['cells']}
    qualified = all(d[45,order]>d[42,order] for order in LEVELS) and evidence['contrasts']['initialization_45_minus_42']>.002
    result = dict(kind='module_initialization_entry',status='qualified' if qualified else 'not_entered',
        measured_at=c.datetime.now(c.timezone.utc).isoformat(), evidence=evidence,
        crossed_result_sha256=r.digest(c.HERE/'crossed-order-result.json'),
        rule='Init45 less favorable under both orders and mean initializer contrast>0.002; predeclared commit13d0225',
        reason='Isolate NCP-module versus backbone initialization at fixed order42' if qualified else 'Predeclared initializer-sensitivity rule failed')
    r.write_json(path,result)
    c.log('Module-initialization stage '+result['status']+': '+result['reason'])
    return result


def basis_hash(directory):
    import torch
    record = c.read(directory/'result.json')
    assert r.digest(directory/'checkpoint_pre_eval.pt')==record['checkpoint_sha256']
    state = torch.load(directory/'checkpoint_pre_eval.pt',map_location='cpu',weights_only=True)
    return hashlib.sha256(state['ncp.codebook.basis'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def anchor(plan,condition,seed):
    identity = plan['anchors'][f'{condition}:{seed}']
    path = c.HERE / identity['trial']
    assert crossed.identity(path)==identity
    crossed.verify_artifacts(path)
    return path


def freeze():
    if entry()['status']!='qualified':
        return None
    path = c.HERE/'module-initialization-plan.json'
    if path.exists():
        plan = c.read(path)
        assert (c.HERE/'module-initialization-driver.py').read_bytes()==DRIVER_SOURCE
        assert r.digest(c.HERE/'module-initialization-driver.py')==plan['driver_sha256']
        assert r.digest(c.HERE/'module-initialization-entry-result.json')==plan['entry_sha256']
        return plan
    frozen = c.read(c.HERE/'confirmation-selection.json')
    cross = c.read(c.HERE/'crossed-order-result.json')
    anchors = {}
    bases = {}
    for cell in cross['cells']:
        if cell['batch_order_seed']!=42:
            continue
        seed = cell['initialization_seed']
        for condition in ('D6','NCP'):
            directory = c.HERE / cell['runs'][condition]['trial']
            record = crossed.verify_artifacts(directory)
            assert record['candidate']==(c.candidate('D6') if condition=='D6' else frozen['variants'][frozen['label']])
            assert strip_seeds(record['protocol'])==strip_seeds(frozen['confirmation_protocol'])
            assert record['data_seal']==frozen['confirmation_data_seal'] and record['upstream']==frozen['confirmation_upstream']
            anchors[f'{condition}:{seed}']=crossed.identity(directory)
            if condition=='NCP':
                bases[str(seed)]=basis_hash(directory)
    assert len(anchors)==4
    candidate = copy.deepcopy(frozen['variants'][frozen['label']])
    candidate_path = c.HERE/'candidates/MODULE-NCP.json'
    if candidate_path.exists(): assert c.read(candidate_path)==candidate
    else: r.write_json(candidate_path,candidate)
    protocol = copy.deepcopy(frozen['confirmation_protocol'])
    for backbone,module in ((42,45),(45,42)):
        r.validate_protocol(dict(protocol,seed=backbone,batch_order_seed=42,ncp_initialization_seed=module))
    for name,value in frozen['confirmation_source_hashes'].items():
        if name not in ('autoresearch.py','autoresearch_train.py'):
            assert c.sources()[name]==value, ('Mechanism source changed',name)
    archive = c.HERE/'module-initialization-driver.py'
    assert not archive.exists()
    archive.write_bytes(DRIVER_SOURCE)
    plan = dict(kind='module_initialization_plan',candidate=candidate,candidate_sha256=r.digest(candidate_path),
        anchors=anchors,anchor_basis_hashes=bases,protocol=protocol,source_hashes=c.sources(),
        data_seal=c.read(r.ROOT/'.autoresearch/data-seal.json'),upstream=r.verify_runtime(),
        expected_parameters=frozen['expected_parameters'],driver_sha256=r.digest(archive),
        frozen_at=c.datetime.now(c.timezone.utc).isoformat(),entry_sha256=r.digest(c.HERE/'module-initialization-entry-result.json'),
        cells=[[42,45],[45,42]],order_seed=42,per_run_forecast_seconds=FORECAST,
        hypothesis='Swap only the NCP module initializer to separate module from backbone initialization at fixed order42; frozen architecture/objectives; no seed selection',
        interpretation='Two initializer levels and one order level; descriptive mechanism diagnosis, not independent replication or held-out evidence')
    assert plan['data_seal']==frozen['confirmation_data_seal'] and plan['upstream']==frozen['confirmation_upstream']
    r.write_json(path,plan)
    return plan


def existing(backbone):
    found = [p.parent for p in c.HERE.glob('trial-*/result.json')
             if (c.read(p)['label'],c.read(p)['seed'])==('MODULE-NCP',backbone)]
    assert len(found)<=1
    return found[0] if found else None


def validate(directory,plan,backbone,module):
    record = crossed.verify_artifacts(directory)
    assert record['label']=='MODULE-NCP' and record['phase']=='module_initialization_diagnostic'
    assert record['candidate']==plan['candidate'] and r.digest(directory/'candidate.json')==plan['candidate_sha256']
    assert record['protocol']==dict(plan['protocol'],seed=backbone,batch_order_seed=42,ncp_initialization_seed=module)
    assert record['data_seal']==plan['data_seal'] and record['upstream']==plan['upstream']
    for name,value in plan['source_hashes'].items():
        assert record['snapshot_files']['source/project/'+name]==value
    dense = anchor(plan,'D6',backbone)
    module_anchor = anchor(plan,'NCP',module)
    parameters = c.read(directory/'initialization.json')['parameters']
    assert {k:v for k,v in parameters.items() if not k.startswith('ncp.')}==c.read(dense/'initialization.json')['parameters']
    assert {k:v for k,v in parameters.items() if k.startswith('ncp.')}=={
        k:v for k,v in c.read(module_anchor/'initialization.json')['parameters'].items() if k.startswith('ncp.')}
    receipt = c.read(directory/'ncp-initialization.json')
    assert receipt['ncp_initialization_seed']==module and receipt['effective_seed']==12600+module
    assert receipt['backbone_initialization_seed']==backbone
    assert receipt['basis_sha256']==plan['anchor_basis_hashes'][str(module)]==basis_hash(directory)
    assert crossed.payload(c.read(directory/'batches.json'))==crossed.payload(c.read(dense/'batches.json'))
    assert c.read(directory/'schedule.json')==c.read(module_anchor/'schedule.json')
    assert c.read(directory/'model.json')['total_parameters']==plan['expected_parameters']
    return record


def summary(plan):
    cells=[]
    for backbone in LEVELS:
        dense = anchor(plan,'D6',backbone)
        dense_bpb = c.read(dense/'result.json')['metrics']['val_bpb']
        for module in LEVELS:
            directory = anchor(plan,'NCP',backbone) if backbone==module else existing(backbone)
            if directory is None: continue
            record = crossed.verify_artifacts(directory) if backbone==module else validate(directory,plan,backbone,module)
            cells.append(dict(backbone_seed=backbone,ncp_initialization_seed=module,batch_order_seed=42,
                trial=directory.name,result_sha256=r.digest(directory/'result.json'),dense_trial=dense.name,
                bpb=record['metrics']['val_bpb'],dense_bpb=dense_bpb,delta_bpb=record['metrics']['val_bpb']-dense_bpb,
                collapsed=record['ncp_health']['collapsed'],reused=backbone==module))
    contrasts=None
    if len(cells)==4:
        d={(x['backbone_seed'],x['ncp_initialization_seed']):x['delta_bpb'] for x in cells}
        contrasts=dict(module_45_minus_42=(d[42,45]+d[45,45]-d[42,42]-d[45,42])/2,
            backbone_45_minus_42=(d[45,42]+d[45,45]-d[42,42]-d[42,45])/2,
            interaction_difference_of_differences=d[45,45]-d[45,42]-d[42,45]+d[42,42])
    return dict(kind='module_initialization_diagnostic',status='completed' if len(cells)==4 else 'running',
        cells=cells,contrasts=contrasts,expected_cells=4,interpretation=plan['interpretation'],
        plan_sha256=r.digest(c.HERE/'module-initialization-plan.json'))


def main(publish=False):
    plan=freeze()
    if plan is None: return
    pending=[]
    for backbone,module in plan['cells']:
        directory=existing(backbone)
        if directory is None: pending.append((backbone,module))
        else: validate(directory,plan,backbone,module)
    if len(pending)*FORECAST>c.remaining()-RESERVE:
        r.write_json(c.HERE/'module-initialization-budget-result.json',dict(kind='module_initialization_budget',status='not_entered',reason='Whole-stage timeout forecast would cross90-minute reserve'))
        return
    for backbone,module in pending:
        assert c.sources()==plan['source_hashes'] and r.verify_runtime()==plan['upstream']
        assert c.read(r.ROOT/'.autoresearch/data-seal.json')==plan['data_seal']
        assert c.read(r.ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json')==plan['protocol']
        assert r.digest(c.HERE/'candidates/MODULE-NCP.json')==plan['candidate_sha256']
        assert c.remaining()>RESERVE+FORECAST
        result=c.trial('MODULE-NCP',backbone,plan['hypothesis']+f'; backbone{backbone}, module{module}',
            'module_initialization_diagnostic','D6',batch_order_seed=42,ncp_initialization_seed=module)
        assert result and result['status']=='completed', 'Preserved module diagnostic failure needs diagnosis'
        validate(existing(backbone),plan,backbone,module)
        r.write_json(c.HERE/'module-initialization-result.json',summary(plan))
        from ncp_report import write
        write()
        if publish:
            try: search.publish()
            except Exception as exc: c.log('Publication deferred; module evidence retained: '+repr(exc))


if __name__=='__main__':
    if '--entry-only' in sys.argv: entry()
    elif '--freeze-only' in sys.argv: freeze()
    else: main('--publish' in sys.argv)
