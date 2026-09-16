"""Six fresh-seed data-only runs and replay; same tokenizer, held-out bytes and technical stream."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
import autoresearch as r
from foundation_baseline import HOME,log
from foundation_data import ROOT,digest,verify,write
from foundation_expand import read_rows
from verify_foundation_campaign import checked

OLD=ROOT/'data/foundation-v1'
NEW=ROOT/'data/foundation-expanded-v1'
TOKENIZER=str(HOME/'tokenizer/candidate.json')

def validate_corpus_pair(old,new):
    a=verify(old);b=verify(new)
    assert b['parent_fingerprint']==a['fingerprint']
    assert b['ingestor_sha256']==digest(ROOT/'scripts/foundation_expand.py')
    for name in ('validation.jsonl','test.jsonl'):assert a['outputs'][name]==b['outputs'][name]
    select=lambda root:sorted((x for x in read_rows(Path(root)/'train.jsonl') if x['domain']=='technical'),key=lambda x:x['id'])
    assert select(old)==select(new),'Technical training data changed'
    assert a['outputs']['train.jsonl']!=b['outputs']['train.jsonl']
    return a,b

def launch(name,seed,data,updates=512):
    cutoff=datetime.fromisoformat(json.loads((HOME/'schedule.json').read_text())['experiment_cutoff'])
    if (cutoff-datetime.now().astimezone()).total_seconds()<1200:raise TimeoutError('Do not consume reporting reserve')
    out=HOME/name
    if out.exists():raise RuntimeError('Preserve existing attempt '+name)
    log('START '+name);started=time.monotonic()
    command=[str(r.runtime_python()),'-B',str(ROOT/'scripts/foundation_train.py'),'--output',str(out),'--data',str(data),
             '--seed',str(seed),'--updates',str(updates),'--general-weight','.8','--tokenizer',TOKENIZER]
    receipt=dict(status='running',command=command,controller_sha256=digest(__file__))
    write(HOME/(name+'-supervisor.json'),receipt)
    try:
        checked(command,HOME/(name+'.log'),timeout=1200)
        result=json.loads((out/'result.json').read_text());assert result['status']=='completed'
        assert result['training_tokens']==updates*16384
        receipt.update(status='completed',result_sha256=digest(out/'result.json'))
        log('END '+name+' aggregate BPB='+str(result['validation']['aggregate_bpb']))
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));log('FAILED '+name+' '+repr(exc));raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started;write(HOME/(name+'-supervisor.json'),receipt)

def main():
    old,new=validate_corpus_pair(OLD,NEW)
    frozen=dict(seeds=[211,212,213],updates=512,old_fingerprint=old['fingerprint'],new_fingerprint=new['fingerprint'],
        tokenizer_sha256=digest(TOKENIZER),controller_sha256=digest(__file__),plan_sha256=digest(ROOT/'docs/foundation-data-extension.md'))
    plan=HOME/'expanded-plan.json'
    if plan.exists():assert json.loads(plan.read_text())==frozen
    else:write(plan,frozen)
    if not (HOME/'expanded-smoke').exists():launch('expanded-smoke',211,NEW,updates=2)
    assert json.loads((HOME/'expanded-smoke/result.json').read_text())['status']=='completed'
    for seed in frozen['seeds']:
        order=[('old',OLD),('expanded',NEW)] if seed%2 else [('expanded',NEW),('old',OLD)]
        for arm,data in order:
            name=f'data-{arm}-s{seed}'
            if (HOME/name).exists():
                assert json.loads((HOME/name/'result.json').read_text())['status']=='completed'
            else:
                assert digest(__file__)==frozen['controller_sha256']
                validate_corpus_pair(OLD,NEW);launch(name,seed,data)
    log('Data-only six-run comparison completed. Freeze training; sequential independent replays follow.')
    for seed in frozen['seeds']:
        for arm,data in [('old',OLD),('expanded',NEW)]:
            name=f'data-{arm}-s{seed}';output=HOME/(name+'-replay.json')
            if output.exists():assert json.loads(output.read_text())['status']=='verified';continue
            checked([str(r.runtime_python()),'-B',str(ROOT/'scripts/evaluate_foundation.py'),str(HOME/name),'--data',str(data),'--output',str(output)],HOME/(name+'-replay.log'))
            log('Verified data-only checkpoint '+name)
    log('Data extension verified; no test split model scores opened and no promotion.')

if __name__=='__main__':main()
