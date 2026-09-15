"""Read-only confirmation-contract checks against retained screening evidence."""
import copy

import autoresearch as r
import ncp_campaign as c
import ncp_confirmation as f
import ncp_frozen_pairs as p


def main():
    directory=c.HERE/'trial-0015-R-feedback_scale0.1-s42'
    dense=c.HERE/'trial-0001-D6-s42'
    record=c.read(directory/'result.json');control=c.read(dense/'result.json')
    source=lambda row:{name:row['snapshot_files']['source/project/'+name] for name in r.PROJECT_FILES}
    frozen=dict(label=record['label'],variants=f.conditions(record,c.read(directory/'model.json')),
        seeds=[42],primary_seeds=[42],variant_file_hashes={record['label']:r.digest(directory/'candidate.json')},
        confirmation_protocol=record['protocol'],confirmation_data_seal=record['data_seal'],
        confirmation_source_hashes=source(record),expected_parameters=c.read(directory/'model.json')['total_parameters'],
        dense_parameters=c.read(dense/'model.json')['total_parameters'],
        reused_controls={'42':dict(trial=dense.name,result_sha256=r.digest(dense/'result.json'),source_hashes=source(control))})
    result=p.collect(frozen)
    assert len(result['pairs'])==1 and abs(result['pairs'][0]['delta_bpb']+.001116)<1e-9
    bad=copy.deepcopy(frozen);bad['variants'][record['label']]['ncp']['feedback_scale']=.3
    cases=[bad]
    bad=copy.deepcopy(frozen);bad['confirmation_protocol']['eval_tokens']*=2;cases.append(bad)
    bad=copy.deepcopy(frozen);bad['confirmation_source_hashes']['autoresearch_ncp.py']='0'*64;cases.append(bad)
    bad=copy.deepcopy(frozen);bad['expected_parameters']+=1;cases.append(bad)
    for bad in cases:
        rejected=False
        try: p.collect(bad)
        except (AssertionError,ValueError): rejected=True
        assert rejected,'Changed frozen contract was accepted'
    assert frozen['variants']['FROZEN-CAP']['capacity']['hidden']==4832
    assert frozen['variants']['FROZEN-AUX']['ncp']['mode']=='auxiliary'
    assert frozen['variants']['FROZEN-NOPRED']['ncp']['prediction_weight']==0
    print('PASS: saved pair; exact configuration, protocol, data, execution, counts, schedule, optimizer and initialization; four altered contracts rejected')


if __name__=='__main__': main()
