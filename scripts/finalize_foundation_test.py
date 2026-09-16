"""Freeze all pilot checkpoints after training stops; one sequential final test stage."""
from datetime import datetime
import json
from pathlib import Path
import time
import autoresearch as r
from foundation_data import ROOT,digest,sha,write,verify
from foundation_baseline import HOME,log
from foundation_eval import selected_inputs
from ncp_campaign import gpu_lock
from verify_foundation_campaign import checked

NAMES=[f'pilot-{arm}-s{seed}' for seed in (201,202,203) for arm in ('control','mixture','tokenizer')]+[f'data-{arm}-s{seed}' for seed in (211,212,213) for arm in ('old','expanded')]

def main():
    deadline=datetime.fromisoformat(json.loads((HOME/'schedule.json').read_text())['planned_stop'])
    while True:
        if (deadline-datetime.now().astimezone()).total_seconds()<1800:raise TimeoutError('Final-test admission reserve')
        if all((HOME/(name+'-replay.json')).exists() for name in NAMES):break
        for name in NAMES:
            p=HOME/(name+'-supervisor.json')
            if p.exists() and json.loads(p.read_text())['status']=='failed':raise RuntimeError('Training failure; final test blocked')
        time.sleep(15)
    time.sleep(3)
    freeze_path=HOME/'final-test-freeze.json'
    if not freeze_path.exists():
        with gpu_lock():
            identities={}
            for name in NAMES:
                folder=HOME/name;row=json.loads((folder/'result.json').read_text());replay=json.loads((HOME/(name+'-replay.json')).read_text())
                assert row['status']=='completed' and replay['status']=='verified'
                assert digest(folder/'result.json')==replay['run_result_sha256']
                assert row['checkpoint_sha256']==digest(folder/'checkpoint.pt')
                identities[name]=dict(seed=row['seed'],checkpoint_sha256=row['checkpoint_sha256'],result_sha256=digest(folder/'result.json'),dataset_fingerprint=row['dataset_fingerprint'])
            data=ROOT/'data/foundation-v1';manifest=verify(data)
            inputs=selected_inputs([json.loads(x) for x in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()])
            write(HOME/'final-test-inputs.json',inputs)
            write(freeze_path,dict(status='frozen',frozen_at=datetime.now().astimezone().isoformat(),runs=identities,
                evaluator_sha256=digest(ROOT/'scripts/evaluate_foundation_test.py'),controller_sha256=digest(__file__),
                plan_sha256=digest(ROOT/'docs/foundation-final-test.md'),test_data_directory=str(data),test_file_sha256=manifest['outputs']['test.jsonl'],
                input_fingerprint=sha(''.join(x['sha256'] for x in inputs).encode()),
                rule='No subsequent training/reselection; all15test outcomes reported; both validation and test gates including mean improvement >2 paired SD'))
        log('ALL TRAINING FROZEN. Final test stage sealed for all15checkpoints; no further campaign training is permitted.')
    for name in NAMES:
        output=HOME/(name+'-test.json')
        if output.exists():assert json.loads(output.read_text())['status']=='verified';continue
        checked([str(r.runtime_python()),'-B',str(ROOT/'scripts/evaluate_foundation_test.py'),str(HOME/name),'--freeze',str(freeze_path),'--output',str(output)],HOME/(name+'-test.log'))
        log('Final held-out test completed: '+name)
    log('Final test stage complete for all15models. Reporting/verification only; no training or reselection.')

if __name__=='__main__':main()
