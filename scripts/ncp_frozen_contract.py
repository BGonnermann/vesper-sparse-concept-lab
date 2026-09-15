"""Exact prospective identity contract for frozen confirmation runs."""
import copy

import autoresearch as r
import ncp_campaign as c


def protocol(frozen,seed):
    assert seed in frozen['seeds']
    value=copy.deepcopy(frozen['confirmation_protocol']);value['seed']=seed
    return r.validate_protocol(value)


def validate(directory,frozen,label,seed):
    record=c.read(directory/'result.json')
    assert record['status']=='completed' and (record['label'],record['seed'])==(label,seed)
    expected=c.candidate('D6') if label=='D6' else frozen['variants'][label]
    assert record['candidate']==expected,('Frozen configuration mismatch',directory)
    assert record['protocol']==protocol(frozen,seed),('Frozen protocol mismatch',directory)
    assert record['data_seal']==frozen['confirmation_data_seal'],('Frozen data identity mismatch',directory)
    assert record['upstream']==frozen['confirmation_upstream'],('Frozen upstream identity mismatch',directory)
    reused=frozen['reused_controls'].get(str(seed)) if label=='D6' else None
    if reused:
        assert directory.name==reused['trial'] and r.digest(directory/'result.json')==reused['result_sha256']
        expected_sources=reused['source_hashes']
    else: expected_sources=frozen['confirmation_source_hashes']
    for name,digest in expected_sources.items():
        assert record['snapshot_files']['source/project/'+name]==digest,('Frozen source mismatch',directory,name)
    r.validate_execution(directory,record['snapshot_files'])
    r.validate_run_artifacts(directory,record)
    count=c.read(directory/'model.json')['total_parameters']
    assert count==(frozen['dense_parameters'] if label=='D6' else frozen['expected_parameters'])
    if label!='D6':
        assert r.digest(directory/'candidate.json')==frozen['variant_file_hashes'][label]
    return record
