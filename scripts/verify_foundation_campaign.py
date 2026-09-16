"""Wait for the bounded pilot, then sequentially verify/replay every completed checkpoint."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
import autoresearch as r
from foundation_baseline import HOME,log
from foundation_data import write
from ncp_campaign import gpu_lock


def checked(command,path,timeout=240,env=None):
    samples=[];hot=0;start=time.monotonic();process=None;heartbeat=start+60
    with Path(path).open('x',encoding='utf-8') as f:
        try:
            process=subprocess.Popen(command,cwd=r.ROOT,env=env or r.environment(),stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            while process.poll() is None:
                sample=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                samples.append(sample);hot=hot+1 if int(sample.split(',')[0])>=85 else 0
                if hot>=3:raise RuntimeError('Verification thermal limit')
                if time.monotonic()-start>timeout:raise TimeoutError('Verification subprocess deadline')
                if time.monotonic()>=heartbeat:
                    lines=Path(path).read_text(errors='replace').splitlines()
                    log(Path(path).stem+' '+next((x for x in reversed(lines) if x.startswith('step ')),'working'))
                    heartbeat=time.monotonic()+60
                    write(str(path)+'.health.json',dict(samples=samples,wall_seconds=time.monotonic()-start))
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            if process.returncode:raise RuntimeError(f'Verification failed: {path}')
        finally:
            if process is not None and process.poll() is None:r.stop_process(process)
            write(str(path)+'.health.json',dict(samples=samples,wall_seconds=time.monotonic()-start))


def main():
    names=[f'pilot-{arm}-s{seed}' for seed in (201,202,203) for arm in ('control','mixture','tokenizer')]
    deadline=datetime.fromisoformat(json.loads((HOME/'schedule.json').read_text())['planned_stop'])
    while True:
        if (deadline-datetime.now().astimezone()).total_seconds()<1200:raise TimeoutError('Verification reserve exhausted')
        receipts=[json.loads((HOME/(n+'-supervisor.json')).read_text()) for n in names if (HOME/(n+'-supervisor.json')).exists()]
        if any(x['status']=='failed' for x in receipts):raise RuntimeError('Pilot failure requires diagnosis; no automatic retries')
        if len(receipts)==9 and all(x['status']=='completed' for x in receipts):break
        time.sleep(20)
    log('Phase 6 complete; verification phase: rerun CPU/CUDA tests and independently replay all nine checkpoints.')
    with gpu_lock():
        for device in ('cpu','cuda'):
            checked([str(r.runtime_python()),'-B','-m','unittest','discover','-s','tests','-v'],HOME/f'tests-final-{device}.log',env=dict(r.environment(),MOE_TEST_DEVICE=device))
            log('Final '+device+' test suite passed')
    for name in names:
        receipt=HOME/(name+'-replay.json')
        if receipt.exists():
            assert json.loads(receipt.read_text())['status']=='verified'
            continue
        checked([str(r.runtime_python()),'-B',str(r.ROOT/'scripts/evaluate_foundation.py'),str(HOME/name),'--output',str(receipt)],HOME/(name+'-replay.log'))
        log('Verified immutable checkpoint replay + exact-byte TinyStories regression: '+name)
    checked([str(r.runtime_python()),'-B',str(r.ROOT/'scripts/report_foundation.py')],HOME/'final-report.log')
    log('Verified validation-stage paired report written: reports/foundation-v1/PILOT.md; no promotion from validation alone.')

if __name__=='__main__':main()
