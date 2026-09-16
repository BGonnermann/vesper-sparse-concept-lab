"""Reproduce the frozen dense TinyStories baseline with captured execution."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
import autoresearch as r
from ncp_campaign import gpu_lock

HOME=r.ROOT/'runs/foundation_campaign'
def log(message):
    line=datetime.now().astimezone().isoformat()+' '+message
    with (HOME/'progress.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
    print(line,flush=True)

def main():
    reference=r.ROOT/'runs/autoresearch/depth-long-20260915/trial-0002-D12-b1-s101/result.json'
    prior=json.loads(reference.read_text()); out=HOME/'baseline-reproduction';out.mkdir(exist_ok=False)
    row=dict(kind='fixed_updates',condition='dense',status='prepared',candidate=prior['candidate'],protocol=prior['protocol'],
             seed=101,reference=str(reference),reference_sha256=r.digest(reference),
             git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             data_seal=json.loads((r.STATE/'data-seal.json').read_text()),upstream=r.verify_runtime())
    row.update({key:r.digest(r.ROOT/'scripts'/name) for name,key in r.PROJECT_FILES.items()})
    start=time.monotonic();process=None;samples=[];hot=0
    with gpu_lock():
        try:
            if shutil.disk_usage(HOME).free<30*2**30:raise RuntimeError('Disk reserve')
            r.verify_seal(r.STATE/'cache',row['data_seal'])
            row['snapshot_files']=r.capture_run_snapshot(out,row)
            log('Baseline reproduction started: D12 W768 seed101, 8,388,608 tokens')
            with (out/'run.log').open('x',encoding='utf-8') as f:
                process=subprocess.Popen(r.snapshot_command(r.runtime_python(),out),cwd=out,env=r.environment(),stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                while process.poll() is None:
                    sample=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
                    samples.append(dict(seconds=time.monotonic()-start,sample=sample))
                    temp=int(sample.split(',')[0]);hot=hot+1 if temp>=85 else 0
                    if hot>=3:raise RuntimeError('Sustained GPU temperature >=85C')
                    if time.monotonic()-start>900:raise TimeoutError('Baseline 900-second deadline')
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
            if process.returncode:raise RuntimeError(f'Child exit {process.returncode}')
            row['metrics']=r.parse_summary((out/'run.log').read_text())
            r.validate_execution(out,row['snapshot_files']);r.validate_run_artifacts(out,row)
            row['checkpoint_sha256']=r.digest(out/'checkpoint_pre_eval.pt')
            row['memory']=json.loads((out/'memory.json').read_text())
            row['absolute_reference_bpb_difference']=abs(row['metrics']['val_bpb']-prior['metrics']['val_bpb'])
            if row['absolute_reference_bpb_difference']>.005:raise RuntimeError('Reproduction differs by >0.005 BPB; stop comparisons')
            row['status']='completed';log('Baseline reproduced: '+str(row['metrics']))
        except BaseException as exc:
            if process is not None and process.poll() is None:r.stop_process(process)
            row.update(status='failed',error=repr(exc));log('Baseline FAILED: '+repr(exc));raise
        finally:
            row['wall_seconds']=time.monotonic()-start
            r.write_json(out/'gpu-samples.json',samples);r.write_json(out/'result.json',row)
if __name__=='__main__':main()
