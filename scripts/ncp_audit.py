"""Fresh verification of saved campaign evidence; never modifies trial artifacts."""
import hashlib
import json
import math
from pathlib import Path
import subprocess

import autoresearch as r
from autoresearch_train import batch_order,fixed_schedule
from ncp_campaign import HERE,read


def audit():
    verified=[]; failures=[]; by_seed={}
    for p in sorted(HERE.glob('trial-*/result.json')):
        record=read(p); directory=p.parent
        for relative,expected in record.get('snapshot_files',{}).items():
            assert r.digest(directory/relative)==expected,(directory,relative)
        if record['status']!='completed':
            failures.append(dict(trial=directory.name,status=record['status'],error=record.get('error'),
                partial_checkpoint_verified=(r.digest(directory/'checkpoint_failure.pt')==record['partial_checkpoint_sha256']) if record.get('partial_checkpoint_sha256') else None))
            continue
        r.validate_execution(directory,record['snapshot_files'])
        artifacts=r.validate_run_artifacts(directory,record)
        assert math.isfinite(record['metrics']['val_bpb'])
        assert r.digest(directory/'checkpoint_pre_eval.pt')==record['checkpoint_sha256']
        before=read(directory/'evaluation-immutability.json')
        assert before['verified'] and before['before_sha256']==before['after_sha256']
        batches=read(directory/'batches.json'); seed=record['seed']
        assert batches['training_tokens']==8388608
        assert batches['consumed_indices']==batch_order(8192,seed)
        assert read(directory/'schedule.json')['updates']==fixed_schedule()
        assert artifacts['training']['optimizer_updates']==512
        assert artifacts['training']['timed_training_tokens']==8208384
        if seed in by_seed:
            assert batches==by_seed[seed],(directory,'batch ordering differs')
        else: by_seed[seed]=batches
        verified.append(dict(trial=directory.name,source_execution=True,checkpoint=True,
            evaluation_immutable=True,token_budget=True,paired_batch_order=True))
    # Check Git's committed bytes, not only working-tree bytes, for published snapshots.
    archive_hashes={}
    for p in HERE.glob('trial-*/source/**/*.py'):
        digest=r.digest(p); name='reports/source-snapshots/'+digest+'.py'
        local=r.ROOT/name
        assert local.exists() and r.digest(local)==digest,name
        if name not in archive_hashes:
            result=subprocess.run(['git','show','HEAD:'+name],cwd=r.ROOT,capture_output=True)
            archive_hashes[name]=bool(result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==digest)
    result=dict(kind='campaign_evidence_audit',status='completed',verified_trials=verified,
        preserved_noncompleted=failures,committed_source_snapshots=archive_hashes,
        all_sources_committed_and_byte_verified=all(archive_hashes.values()),
        measured_campaign_logical_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file() and '.git' not in p.parts),
        measured_checkpoint_bytes=sum(p.stat().st_size for p in HERE.glob('trial-*/*.pt')))
    r.write_json(HERE/'audit-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('verified_trials','preserved_noncompleted','committed_source_snapshots')},indent=2))
    return result


if __name__=='__main__': audit()
