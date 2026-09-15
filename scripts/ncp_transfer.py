"""Conditional, untuned transfer of the frozen mechanism to dense depth12."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import autoresearch as r
import ncp_campaign as c
import ncp_confirmation as confirmation
import ncp_frozen_contract as contract
from ncp_frozen_pairs import collect
import ncp_search as search

DRIVER_SOURCE=Path(__file__).read_bytes()
LABELS=('D12','TRANSFER-NCP','TRANSFER-CAP','TRANSFER-AUX','TRANSFER-NOPRED')
RESERVE=5400


def model_report(candidate):
    import torch
    sys.path.insert(0,str(r.RUNTIME))
    import train
    from autoresearch_model import model_class,with_model_width
    runtime=SimpleNamespace(use_activation_checkpointing=False,attention_backend='sdpa',amp_dtype=torch.bfloat16)
    config=with_model_width(train.build_model_config(candidate['depth'],8192,runtime,False),candidate)
    with torch.device('meta'): model=model_class(train,candidate)(config)
    return model.parameter_report()


def prepare():
    path=c.HERE/'transfer-selection.json'
    if path.exists():
        plan=c.read(path)
        assert (c.HERE/'transfer-driver.py').read_bytes()==DRIVER_SOURCE
        return plan
    original=c.read(c.HERE/'confirmation-selection.json');evidence=collect(original)
    assert all(x['completed_pairs']==4 for x in evidence['summaries'].values()),'Finish all frozen D6 pairs first'
    if not evidence['summaries']['D6']['primary_same_sign_improvement']:
        r.write_json(c.HERE/'transfer-entry-result.json',dict(kind='transfer_entry',status='not_entered',
            reason='Frozen D6 mechanism did not pass the predeclared primary same-sign rule',evidence=evidence))
        c.log('Conditional depth12 transfer not entered: D6 primary replication rule failed')
        return None
    candidate=copy.deepcopy(original['variants'][original['label']]);candidate['depth']=12
    model=model_report(candidate)
    variants=confirmation.conditions(dict(label='TRANSFER-NCP',candidate=candidate),model,prefix='TRANSFER')
    assert model_report(variants['TRANSFER-CAP'])['total_parameters']==model['total_parameters']
    hashes={}
    for label,value in variants.items():
        r.validate_candidate(value);file=c.HERE/'candidates'/f'{label}.json'
        if file.exists(): assert c.read(file)==value
        else: r.write_json(file,value)
        hashes[label]=r.digest(file)
    archive=c.HERE/'transfer-driver.py'
    if archive.exists(): assert archive.read_bytes()==DRIVER_SOURCE
    else: archive.write_bytes(DRIVER_SOURCE)
    plan=dict(kind='frozen_depth12_transfer',status='completed',label='TRANSFER-NCP',dense_label='D12',ablation_prefix='TRANSFER',
        seeds=[45,46],primary_seeds=[45,46],sensitivity_seeds=[],variants=variants,variant_file_hashes=hashes,
        expected_parameters=model['total_parameters'],expected_added_parameters=model['ncp_parameters'],
        dense_parameters=model_report(c.candidate('D12'))['total_parameters'],reused_controls={},
        confirmation_protocol=original['confirmation_protocol'],confirmation_source_hashes=c.sources(),
        confirmation_data_seal=c.read(r.ROOT/'.autoresearch/data-seal.json'),confirmation_upstream=r.verify_runtime(),
        original_selection_sha256=r.digest(c.HERE/'confirmation-selection.json'),entry_evidence=evidence,
        driver_sha256=r.digest(archive),frozen_at=c.datetime.now(c.timezone.utc).isoformat(),
        comparison_interpretation='Conditional second-backbone test with unchanged frozen mechanism; same seed identities and validation corpus, not additional independent seeds or held-out generalization',
        schedule_policy='Paired conditions in order dense,NCP,CAP,AUX,NOPRED; require budget for first three; optional last two use only remaining time, never their scores')
    r.write_json(path,plan)
    return plan


def main(publish=False):
    plan=prepare()
    if plan is None: return
    r.write_json(c.HERE/'fit-candidates.json',plan['variants'])
    c.preflight()
    fit=c.read(c.HERE/'fit-result.json')['results']
    for label in plan['variants']:
        assert fit[label]['model']['total_parameters']==plan['expected_parameters']
    forecasts={label:fit[label]['conservative_trial_forecast_seconds'] for label in LABELS}
    completed=lambda label,seed:[p.parent for p in c.HERE.glob('trial-*/result.json')
        if (c.read(p)['label'],c.read(p)['seed'])==(label,seed)]
    minimum=sum(forecasts[label]+10 for label in LABELS[:3] for seed in plan['seeds'] if not completed(label,seed))
    if minimum>c.remaining()-RESERVE:
        c.log('Depth12 transfer budget gate: paired dense/NCP/CAP forecasts do not fit before90-minute reserve')
        r.write_json(c.HERE/'transfer-entry-result.json',dict(kind='transfer_entry',status='not_entered',
            reason='Minimum paired condition forecasts exceed remaining research budget',forecast_seconds=minimum,remaining_seconds=c.remaining()))
        return
    omitted=[]
    for label in LABELS:
        pending=[seed for seed in plan['seeds'] if not completed(label,seed)]
        if label in LABELS[3:] and sum(forecasts[label]+10 for seed in pending)>c.remaining()-RESERVE:
            omitted.append(label);c.log('Optional paired transfer control omitted by time forecast: '+label);continue
        for seed in plan['seeds']:
            peers=completed(label,seed)
            if peers:
                assert len(peers)==1;contract.validate(peers[0],plan,label,seed);continue
            assert c.sources()==plan['confirmation_source_hashes']
            live=c.read(r.ROOT/'runs/autoresearch/equal-token-20260915/protocol-42.json');live['seed']=seed
            assert live==contract.protocol(plan,seed)
            assert c.read(r.ROOT/'.autoresearch/data-seal.json')==plan['confirmation_data_seal']
            assert r.verify_runtime()==plan['confirmation_upstream']
            if label!='D12': assert r.digest(c.HERE/'candidates'/f'{label}.json')==plan['variant_file_hashes'][label]
            result=c.trial(label,seed,'Untuned conditional depth12 transfer of frozen D6 mechanism; '+plan['schedule_policy'],
                'frozen_depth12_transfer','D12')
            if result is None: return
            if result['status']!='completed': raise RuntimeError('Preserved transfer failure requires diagnosis')
            contract.validate(completed(label,seed)[0],plan,label,seed)
            r.write_json(c.HERE/'transfer-pairs-result.json',dict(kind='transfer_paired_evidence',**collect(plan),omitted_by_budget=omitted))
            from ncp_report import write
            write()
            if publish:
                try: search.publish()
                except Exception as exc: c.log('Publication deferred; transfer retained: '+repr(exc))
    r.write_json(c.HERE/'transfer-pairs-result.json',dict(kind='transfer_paired_evidence',**collect(plan),omitted_by_budget=omitted))


if __name__=='__main__': main('--publish' in sys.argv)
