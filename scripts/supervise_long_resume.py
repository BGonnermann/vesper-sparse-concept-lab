"""Resume under the same health watchdog with an explicit remaining wall-clock deadline."""
import argparse
from datetime import datetime
from pathlib import Path
from long_baseline_runtime import ROOT,read,digest
from long_baseline import HOME,log
from long_baseline_campaign import checked
import autoresearch as r
from foundation_data import write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--deadline',required=True);a=p.parse_args()
    deadline=datetime.fromisoformat(a.deadline)
    if deadline.tzinfo is None:raise ValueError('Deadline needs explicit timezone offset')
    remaining=(deadline-datetime.now().astimezone()).total_seconds()
    if remaining<=0:raise ValueError('Recovery deadline expired; no training launched')
    assert digest(a.config)==digest(a.run/'config.json'),'Do not change configuration on recovery'
    checkpoint=a.checkpoint or Path(read(a.run/'latest.json')['path'])
    assert digest(checkpoint)==read(str(checkpoint)+'.json')['sha256'],'Checkpoint corruption'
    number=len(list(HOME.glob('recovery-*.json')))+1;record=HOME/f'recovery-{number:02d}.json';out=HOME/f'recovery-{number:02d}.log'
    receipt=dict(status='running',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),deadline=a.deadline,controller_sha256=digest(__file__))
    write(record,receipt);log('RECOVERY ADMITTED: '+str(checkpoint))
    try:
        checked([str(r.runtime_python()),'-B',str(ROOT/'scripts/long_baseline.py'),'--config',str(a.config),'--output',str(a.run),'--resume',str(checkpoint)],out,int(remaining))
        receipt['status']='completed'
    except BaseException as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:write(record,receipt)
