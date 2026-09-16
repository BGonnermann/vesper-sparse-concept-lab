"""Monitored single-trial entry point for a NEW study using a versioned foundation profile."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time
import autoresearch as r
from foundation_baseline import HOME
from foundation_data import ROOT,digest,verify,write
import verify_foundation_campaign as supervision


def validate_output(output):
    output=Path(output).resolve()
    if output.is_relative_to(HOME.resolve()):
        raise ValueError('The completed campaign is frozen; use a NEW study directory, not runs/foundation_campaign')
    if output.exists():raise ValueError('Preserve existing trial; choose a new output directory')
    return output

def run(profile_path,output,seed,timeout):
    output=validate_output(output);profile=json.loads(Path(profile_path).read_text())
    if not 1<=timeout<=2400:raise ValueError('Timeout must be1..2400seconds')
    t=profile['training'];assert (t['sequence_length'],t['microbatch_size'],t['tokens_per_update'])==(512,2,16384)
    assert t['activation_checkpointing'] is False and t['matrix_lr']==.04 and t['amp']=='bfloat16'
    assert json.loads((ROOT/profile['architecture_config']).read_text())==dict(depth=12,model_width=768,matrix_lr=.04,feedforward='dense')
    data=ROOT/profile['data']['path'];tokenizer=ROOT/profile['tokenizer']['path']
    assert verify(data)['fingerprint']==profile['data']['fingerprint']
    assert digest(tokenizer)==profile['tokenizer']['sha256']
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<31*2**30:raise RuntimeError('Disk reserve')
    log_path=output.parent/(output.name+'.log');receipt_path=output.parent/(output.name+'-supervisor.json')
    if log_path.exists() or receipt_path.exists():raise ValueError('Preserve earlier attempt logs; choose a new name')
    def log(message):
        line=datetime.now().astimezone().isoformat()+' '+message
        with (output.parent/'progress.log').open('a',encoding='utf-8') as f:f.write(line+'\n')
        print(line,flush=True)
    supervision.log=log
    command=[str(r.runtime_python()),'-B',str(ROOT/'scripts/foundation_train.py'),'--output',str(output),'--data',str(data),
        '--tokenizer',str(tokenizer),'--seed',str(seed),'--updates',str(t['optimizer_updates']),'--general-weight',str(profile['data']['general_weight'])]
    started=time.monotonic();receipt=dict(status='running',command=command,profile_sha256=digest(profile_path),controller_sha256=digest(__file__))
    write(receipt_path,receipt);log('START '+str(output))
    try:
        supervision.checked(command,log_path,timeout=timeout)
        result=json.loads((output/'result.json').read_text());assert result['status']=='completed'
        assert result['dataset_fingerprint']==profile['data']['fingerprint'] and result['tokenizer']['sha256']==profile['tokenizer']['sha256']
        assert result['parameters']['total_parameters']==profile['total_parameters']
        assert result['training_tokens']==t['optimizer_updates']*t['tokens_per_update']
        receipt.update(status='completed',result_sha256=digest(output/'result.json'));log('END '+str(output))
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));log('FAILED '+repr(exc));raise
    finally:
        receipt['wall_seconds']=time.monotonic()-started;write(receipt_path,receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,default=ROOT/'experiments/mainline/foundation-general-v2.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--timeout',type=int,default=1200)
    a=p.parse_args();run(a.profile,a.output,a.seed,a.timeout)
