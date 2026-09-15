"""Freeze a selection-seed candidate, then execute predeclared paired controls."""
import copy
import sys
from pathlib import Path

import autoresearch as r
import ncp_campaign as c
import ncp_search as search
from ncp_interactions import fingerprint
import ncp_frozen_contract as contract

SEEDS=[45,46,43,44]
DRIVER_SOURCE=Path(__file__).read_bytes()


def conditions(record,model):
    chosen=copy.deepcopy(record['candidate'])
    assert chosen['depth']==6 and chosen['ncp']['mode']=='feedback'
    width=model['width']; added=model['ncp_parameters']
    assert added>0 and added%(2*width)==0
    auxiliary=copy.deepcopy(chosen); auxiliary['ncp']['mode']='auxiliary'
    unsupervised=copy.deepcopy(chosen)
    unsupervised['ncp'].update(prediction_weight=0.,ce_weight=0.)
    capacity={k:v for k,v in chosen.items() if k!='ncp'}
    capacity['capacity']=dict(kind='residual_mlp_v1',hidden=added//(2*width),
        after_layer=chosen['ncp']['after_layer'],lr=chosen['ncp']['lr'])
    return {record['label']:chosen,'FROZEN-AUX':auxiliary,'FROZEN-CAP':capacity,'FROZEN-NOPRED':unsupervised}


def freeze():
    destination=c.HERE/'confirmation-selection.json'
    if destination.exists():
        receipt=c.read(destination)
        assert (c.HERE/'confirmation-driver.py').read_bytes()==DRIVER_SOURCE
        assert r.digest(c.HERE/'confirmation-driver.py')==receipt['driver_sha256']
        return receipt
    history=search.records()
    assert not any(x['phase']=='ncp_confirmation' for x in history)
    baseline=next(x for x in history if x['label']=='D6' and x['seed']==42 and x['status']=='completed')
    eligible=[x for x in history if x['seed']==42 and x['status']=='completed'
        and x.get('candidate',{}).get('ncp',{}).get('mode')=='feedback'
        and not x['ncp_health']['collapsed']
        and (x['candidate']['ncp']['prediction_weight']>0 or x['candidate']['ncp']['ce_weight']>0)]
    selected=min(eligible,key=lambda x:x['metrics']['val_bpb'])
    assert c.promising(selected,baseline['metrics']['val_bpb']),'No candidate meets the declared screening threshold'
    matches=[p for p in c.HERE.glob('trial-*/result.json') if c.read(p)==selected]
    assert len(matches)==1
    trial=matches[0].parent; model=c.read(trial/'model.json')
    variants=conditions(selected,model)
    variant_hashes={}
    for label,candidate in variants.items():
        r.validate_candidate(candidate)
        path=c.HERE/'candidates'/f'{label}.json'
        if path.exists(): assert fingerprint(c.read(path))==fingerprint(candidate)
        else: r.write_json(path,candidate)
        variant_hashes[label]=r.digest(path)
    driver=c.HERE/'confirmation-driver.py'
    if driver.exists(): assert driver.read_bytes()==DRIVER_SOURCE
    else: driver.write_bytes(DRIVER_SOURCE)
    reused={}
    for seed in SEEDS:
        peers=[p for p in c.HERE.glob('trial-*/result.json') if (c.read(p)['label'],c.read(p)['seed'])==('D6',seed)]
        if peers:
            assert len(peers)==1 and c.read(peers[0])['status']=='completed'
            record=c.read(peers[0])
            reused[str(seed)]=dict(trial=peers[0].parent.name,result_sha256=r.digest(peers[0]),
                source_hashes={name:record['snapshot_files']['source/project/'+name] for name in r.PROJECT_FILES})
    receipt=dict(kind='frozen_ncp_selection',status='completed',label=selected['label'],
        seeds=SEEDS,primary_seeds=[45,46],sensitivity_seeds=[43,44],
        selection_trial=trial.name,selection_result_sha256=r.digest(matches[0]),
        candidate_sha256=r.digest(trial/'candidate.json'),source_hashes=selected['snapshot_files'],
        selected_bpb=selected['metrics']['val_bpb'],control_bpb=baseline['metrics']['val_bpb'],
        variants=variants,variant_file_hashes=variant_hashes,
        expected_parameters=model['total_parameters'],expected_added_parameters=model['ncp_parameters'],
        driver_sha256=r.digest(driver),
        confirmation_source_hashes=c.sources(),confirmation_data_seal=c.read(r.ROOT/'.autoresearch/data-seal.json'),
        confirmation_protocol=c.read(r.ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json'),
        reused_controls=reused,dense_parameters=c.read(c.HERE/'trial-0001-D6-s42/model.json')['total_parameters'],
        frozen_at=c.datetime.now(c.timezone.utc).isoformat(),
        success_rule='Valid noncollapsed NCP beats D6 on both primary seeds45/46; report all four seeds and all mixed signs; no reselection')
    r.write_json(destination,receipt)
    c.log('FROZEN '+selected['label']+' from '+trial.name+' for seeds45,46,43,44; no reselection')
    return receipt


def run(publish=False):
    chosen=freeze()
    for seed in chosen['seeds']:
        # Fixed condition order, including fresh dense controls on45/46.
        for label in ['D6',chosen['label'],'FROZEN-AUX','FROZEN-CAP','FROZEN-NOPRED']:
            if label!='D6':
                assert r.digest(c.HERE/'candidates'/f'{label}.json')==chosen['variant_file_hashes'][label]
            peers=[p.parent for p in c.HERE.glob('trial-*/result.json') if (c.read(p)['label'],c.read(p)['seed'])==(label,seed)]
            if peers:
                assert len(peers)==1,'Existing attempts require explicit diagnosis, not an automatic repeat'
                contract.validate(peers[0],chosen,label,seed)
                continue
            assert c.sources()==chosen['confirmation_source_hashes'],'Confirmation source contract changed'
            contract.protocol(chosen,seed)
            hypothesis=(f'Frozen four-seed confirmation: {label}, seed{seed}; selection {chosen["selection_trial"]}; '
                'compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection')
            result=c.trial(label,seed,hypothesis,'ncp_confirmation','D6')
            if result is None: return
            if result['status']!='completed': raise RuntimeError('Preserved confirmation failure needs diagnosis')
            path=next(p.parent for p in c.HERE.glob('trial-*/result.json') if c.read(p)==result)
            contract.validate(path,chosen,label,seed)
            if publish:
                try: search.publish()
                except Exception as exc: c.log('Publication deferred; confirmation retained: '+repr(exc))


if __name__=='__main__':
    if '--freeze-only' in sys.argv: freeze()
    else: run('--publish' in sys.argv)
