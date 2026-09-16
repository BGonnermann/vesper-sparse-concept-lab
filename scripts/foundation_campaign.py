"""Sequential native-Windows supervisor. No retries; retained failures; thermal/deadline guard."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import autoresearch as r
from foundation_data import digest,write
from foundation_baseline import HOME,log


def launch(name,seed,updates,weight,tokenizer,timeout=1200):
    schedule=json.loads((HOME/'schedule.json').read_text());cutoff=datetime.fromisoformat(schedule['experiment_cutoff'])
    if (cutoff-datetime.now().astimezone()).total_seconds()<timeout:raise RuntimeError('Insufficient experiment budget; preserve reporting reserve')
    if shutil.disk_usage(HOME).free<31*2**30:raise RuntimeError('Disk floor')
    out=HOME/name;log('START '+name);samples=[];hot=0;process=None;started=time.monotonic()
    command=[str(r.runtime_python()),'-B',str(r.ROOT/'scripts/foundation_train.py'),'--output',str(out),
             '--seed',str(seed),'--updates',str(updates),'--general-weight',str(weight),'--tokenizer',tokenizer]
    receipt=dict(command=command,started=datetime.now().astimezone().isoformat(),status='running')
    write(HOME/(name+'-supervisor.json'),receipt)
    try:
        with (HOME/(name+'.log')).open('x',encoding='utf-8') as f:
            process=subprocess.Popen(command,cwd=r.ROOT,env=r.environment(),stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            next_heartbeat=time.monotonic()+60
            while process.poll() is None:
                sample=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                samples.append(dict(seconds=time.monotonic()-started,sample=sample))
                temp=int(sample.split(',')[0]);hot=hot+1 if temp>=85 else 0
                if hot>=3:raise RuntimeError('Sustained >=85C')
                if time.monotonic()-started>timeout:raise TimeoutError('Per-job hard deadline')
                if datetime.now().astimezone()>=cutoff:raise TimeoutError('Campaign experiment cutoff')
                if time.monotonic()>=next_heartbeat:
                    lines=(HOME/(name+'.log')).read_text(errors='replace').splitlines()
                    log(name+' '+next((line for line in reversed(lines) if line.startswith('step ')),'preparing'))
                    write(HOME/(name+'-gpu.json'),samples);next_heartbeat=time.monotonic()+60
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
        if process.returncode:raise RuntimeError('Training child exited '+str(process.returncode))
        result=json.loads((out/'result.json').read_text());assert result['status']=='completed'
        assert result['training_tokens']==updates*16384 and sum(result['stream']['source_tokens'].values())==updates*16384
        assert result['parameters']['feedforward']=='dense' and result['parameters']['total_parameters']==135267480
        assert result['parameters']['active_parameters']==135267480
        receipt.update(status='completed',result_sha256=digest(out/'result.json'))
        log('END '+name+' aggregate BPB='+str(result['validation']['aggregate_bpb']))
    except BaseException as exc:
        if process is not None and process.poll() is None:r.stop_process(process)
        receipt.update(status='failed',error=repr(exc));log('FAILED '+name+' '+repr(exc));raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started;write(HOME/(name+'-supervisor.json'),receipt);write(HOME/(name+'-gpu.json'),samples)


def run():
    smoke=json.loads((HOME/'smoke-v2/result.json').read_text());assert smoke['status']=='completed'
    plan=HOME/'pilot-plan.json'
    candidate=str(HOME/'tokenizer/candidate.json')
    arms={'control':(.8,'current'),'mixture':(.5,'current'),'tokenizer':(.8,candidate)}
    frozen=dict(seeds=[201,202,203],updates=512,arms=arms,candidate_sha256=digest(candidate),
        selection_rule='See docs/foundation-campaign.md; no promotion from pilot alone',
        source_hashes={p.name:digest(p) for p in (r.ROOT/'scripts').glob('foundation_*.py')})
    if plan.exists():
        assert json.loads(plan.read_text())==json.loads(json.dumps(frozen)),'Frozen pilot changed'
    else:write(plan,frozen)
    for seed in frozen['seeds']:
        order=list(arms) if seed%2 else list(reversed(arms))
        for arm in order:
            name=f'pilot-{arm}-s{seed}'
            if (HOME/name).exists():
                result=json.loads((HOME/name/'result.json').read_text())
                if result['status']!='completed':raise RuntimeError('Retained failed attempt requires diagnosis, not automatic overwrite: '+name)
                continue
            for filename,h in frozen['source_hashes'].items():assert digest(r.ROOT/'scripts'/filename)==h
            launch(name,seed,512,*arms[arm])
    log('Predeclared nine-run pilot complete. No mainline promotion; proceed to independent verification.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['smoke','run']);a=p.parse_args()
    if a.command=='smoke':launch('smoke-v2',201,2,.8,'current',timeout=240)
    else:run()
