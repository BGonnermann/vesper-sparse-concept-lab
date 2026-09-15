"""Fresh verification of saved campaign evidence; never modifies trial artifacts."""
import hashlib
import ast
import json
import math
from pathlib import Path
import subprocess
import shutil
from datetime import datetime,timezone

import autoresearch as r
from autoresearch_train import batch_order,fixed_schedule
from ncp_campaign import HERE,read


def audit():
    verified=[]; failures=[]; by_seed={}; initializations={}; concept_initializations={}; capacity_initializations={}; controllers=[]
    for p in sorted(HERE.glob('trial-*/result.json')):
        record=read(p); directory=p.parent
        if record.get('orchestrator_sha256'):
            controller=directory/'orchestrator.py'
            origin='trial_archive'
            if not controller.exists():
                controller=directory/'orchestrator-recovered-from-git.py'
                origin='git_recovery'
            if controller.exists():
                assert r.digest(controller)==record['orchestrator_sha256'],(directory,'controller archive hash')
            controllers.append(dict(trial=directory.name,sha256=record['orchestrator_sha256'],
                archive_verified=controller.exists(),archive_origin=origin if controller.exists() else 'unavailable',
                limitation=('Recovered after execution from Git, matching original recorded SHA256' if origin=='git_recovery' and controller.exists()
                    else None if controller.exists() else 'Early trial recorded controller hash without preserving controller bytes; training child source is separately archived and verified')))
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
        order_seed=record['protocol'].get('batch_order_seed',seed)
        assert batches.get('batch_order_seed',batches['seed'])==order_seed
        assert batches['training_tokens']==8388608
        assert batches['consumed_indices']==batch_order(8192,order_seed)
        assert read(directory/'schedule.json')['updates']==fixed_schedule()
        assert artifacts['training']['optimizer_updates']==512
        assert artifacts['training']['timed_training_tokens']==8208384
        batch_payload={k:v for k,v in batches.items() if k not in ('seed','batch_order_seed')}
        if order_seed in by_seed:
            assert batch_payload==by_seed[order_seed],(directory,'batch ordering differs')
        else: by_seed[order_seed]=batch_payload
        parameters=read(directory/'initialization.json')['parameters']
        backbone={name:value for name,value in parameters.items() if not name.startswith(('ncp.','capacity.'))}
        backbone_key=(seed,record['candidate']['depth'],read(directory/'model.json')['width'])
        if backbone_key in initializations:
            assert backbone==initializations[backbone_key],(directory,'backbone initialization differs')
        else: initializations[backbone_key]=backbone
        settings=record['candidate'].get('ncp')
        if settings:
            # These settings determine parameter shapes and initialization RNG use.
            concept_key=(*backbone_key,settings['layers'],settings['entries'])
            concept={name:value for name,value in parameters.items() if name.startswith('ncp.')}
            if concept_key in concept_initializations:
                assert concept==concept_initializations[concept_key],(directory,'matched NCP initialization differs')
            else: concept_initializations[concept_key]=concept
        if record['candidate'].get('capacity'):
            capacity_key=(*backbone_key,record['candidate']['capacity']['hidden'])
            capacity={name:value for name,value in parameters.items() if name.startswith('capacity.')}
            if capacity_key in capacity_initializations:
                assert capacity==capacity_initializations[capacity_key],(directory,'matched capacity initialization differs')
            else: capacity_initializations[capacity_key]=capacity
        verified.append(dict(trial=directory.name,source_execution=True,checkpoint=True,
            evaluation_immutable=True,token_budget=True,paired_batch_order=True,matched_initialization=True))
    # Check Git's committed bytes, not only working-tree bytes, for published snapshots.
    archive_hashes={}
    archived_sources=list(HERE.glob('trial-*/source/**/*.py'))+list(HERE.glob('trial-*/*.py'))+list(HERE.glob('*.py'))
    for p in archived_sources:
        digest=r.digest(p); name='reports/source-snapshots/'+digest+'.py'
        local=r.ROOT/name
        assert local.exists() and r.digest(local)==digest,name
        if name not in archive_hashes:
            result=subprocess.run(['git','show','HEAD:'+name],cwd=r.ROOT,capture_output=True)
            archive_hashes[name]=bool(result.returncode==0 and hashlib.sha256(result.stdout).hexdigest()==digest)
    correction_path=HERE/'orchestrator-source-correction-result.json'
    controller_correction=None
    if correction_path.exists():
        correction=read(correction_path);directory=HERE/correction['trial']
        reference=directory/'orchestrator-loaded-reference.py';archived=directory/'orchestrator.py'
        assert r.digest(reference)==correction['loaded_controller_reference_sha256']
        assert r.digest(archived)==correction['archived_file_sha256']
        functions=lambda path:{n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        left,right=functions(reference),functions(archived)
        for name in ('trial','preflight','candidate','health'):
            assert ast.dump(ast.Module(body=left[name].body,type_ignores=[]))==ast.dump(ast.Module(body=right[name].body,type_ignores=[]))
        controller_correction=dict(verified=True,trial=correction['trial'],scope=correction['loaded_version_evidence'])
    result=dict(kind='campaign_evidence_audit',status='completed',verified_trials=verified,
        measured_at_utc=datetime.now(timezone.utc).isoformat(),free_disk_bytes=shutil.disk_usage(r.ROOT).free,
        controller_archive_correction=controller_correction,controller_archives=controllers,
        preserved_noncompleted=failures,committed_source_snapshots=archive_hashes,
        all_sources_committed_and_byte_verified=all(archive_hashes.values()),
        measured_campaign_logical_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
        measured_checkpoint_bytes=sum(p.stat().st_size for p in HERE.glob('trial-*/*.pt')))
    r.write_json(HERE/'audit-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('verified_trials','preserved_noncompleted','committed_source_snapshots')},indent=2))
    return result


if __name__=='__main__': audit()
