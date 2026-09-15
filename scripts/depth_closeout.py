"""Freeze selection, replay validation and perform one final held-out stage."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import autoresearch as r
import depth_campaign as c
import depth_report as report
import experiment_reports as reports


def freeze():
    assert (c.HERE/'training-closed.json').exists(),'Training must close before test freeze'
    assert not (c.HERE/'evaluation-freeze.json').exists(),'Preserve existing freeze'
    rows=report.collect();decision=report.recommendation(rows);seeds=decision['complete_seeds']
    assert len(seeds)>=3
    assert len(rows)==len(seeds)*6,'Incomplete seed block must be resolved before held-out stage'
    selected=[x for x in rows if (x['depth'],x['budget']) in ((6,4),(12,2),(12,4))]
    means={d:statistics.mean(x['bpb'] for x in rows if x['depth']==d and x['budget']==(4 if d==6 else 2)) for d in (6,12)}
    times={d:statistics.mean(x['update_seconds'] for x in rows if x['depth']==d and x['budget']==(4 if d==6 else 2)) for d in (6,12)}
    candidate=min((6,12),key=lambda d:(means[d],times[d]))
    evaluator=c.HERE/'evaluation-driver.py';evaluator.write_bytes((r.ROOT/'scripts/depth_evaluation.py').read_bytes())
    closer=c.HERE/'closeout-driver.py';closer.write_bytes(Path(__file__).read_bytes())
    checkpoints=[dict(trial=x['trial'],seed=x['seed'],depth=x['depth'],budget=x['budget'],
        checkpoint_sha256=x['checkpoint_sha256'],result_sha256=x['result_sha256']) for x in rows]
    value=dict(status='frozen',frozen_at=datetime.now(timezone.utc).isoformat(),matrix_sha256=r.digest(c.HERE/'matrix-plan.json'),
        candidate_depth=candidate,candidate_budget=4 if candidate==6 else 2,
        reference_depth=18-candidate,reference_budget=2 if candidate==6 else 4,
        context_condition='D12@4x',selection_uses='validation only; mean over every complete seed, tie-break measured time',
        validation_recommendation=decision,checkpoints=checkpoints,test_trials=[x['trial'] for x in selected],
        evaluator_sha256=r.digest(evaluator),closeout_sha256=r.digest(closer),
        test_tokens=65536,test_split_rows=[0,10000],validation_split_rows=[10000,20000],train_split_rows=[20000,None],
        interpretation='One final held-out stage, all seeds, no training/tuning/reselection afterward.65,536 packed test tokens are a subset of the10,000-row held-out split.')
    r.write_json(c.HERE/'evaluation-freeze.json',value)
    c.log('Candidate/reference and all test checkpoint identities frozen before test results')


def evaluate(retry_failed=False):
    freeze=c.read(c.HERE/'evaluation-freeze.json')
    assert r.digest(c.HERE/'evaluation-driver.py')==freeze['evaluator_sha256']
    assert r.digest(c.HERE/'closeout-driver.py')==r.digest(Path(__file__))==freeze['closeout_sha256']
    assert r.digest(c.HERE/'matrix-plan.json')==freeze['matrix_sha256']
    assert {x['trial']:x['result_sha256'] for x in freeze['checkpoints']}=={x['trial']:r.digest(c.HERE/x['trial']/'result.json') for x in c.records() if x['status']=='completed'}
    directory=c.HERE/'final-evaluation';directory.mkdir(exist_ok=True)
    outcomes=[]
    with c.gpu_lock():
      try:
        r.verify_seal(r.STATE/'cache',c.read(c.HERE/'matrix-plan.json')['data_seal'])
        for split in ('val','test'):
            selected=freeze['checkpoints'] if split=='val' else [x for x in freeze['checkpoints'] if x['trial'] in freeze['test_trials']]
            for item in selected:
                prefix=f"{split}-{item['trial']}"
                previous=sorted(directory.glob(prefix+'-attempt*-result.json'))
                completed=[p for p in previous if c.read(p).get('status')=='completed']
                assert len(completed)<=1
                if completed:
                    output=completed[0];log=output.with_name(output.name.replace('-result.json','.log'))
                    result=c.read(output)
                    assert result['freeze_sha256']==r.digest(c.HERE/'evaluation-freeze.json')
                    assert result['result_sha256']==item['result_sha256'] and result['checkpoint_sha256']==item['checkpoint_sha256']
                    outcomes.append(dict(trial=item['trial'],seed=item['seed'],depth=item['depth'],budget=item['budget'],
                        split=split,bpb=result['bpb'],result_path=output.relative_to(c.HERE).as_posix(),result_sha256=r.digest(output),
                        log_path=log.relative_to(c.HERE).as_posix(),log_sha256=r.digest(log)))
                    continue
                logs=list(directory.glob(prefix+'-attempt*.log'))
                assert not logs or retry_failed,'Preserved failure requires diagnosis and explicit bounded retry'
                attempt=len(logs)+1;assert attempt<=3,'Evaluation attempt limit reached'
                output=directory/f"{prefix}-attempt{attempt}-result.json";log=directory/f"{prefix}-attempt{attempt}.log"
                assert c.remaining()>600,'Final report reserve reached'
                c.log(f"Final {split} evaluation {item['trial']}")
                with log.open('x',encoding='utf-8') as f:
                    done=subprocess.run([str(r.runtime_python()),'-B',str(c.HERE/'evaluation-driver.py'),
                        '--run',str(c.HERE/item['trial']),'--output',str(output),'--freeze',str(c.HERE/'evaluation-freeze.json'),'--split',split],
                        cwd=r.ROOT,env=r.environment(),stdout=f,stderr=subprocess.STDOUT,timeout=min(180,c.remaining()-600))
                assert done.returncode==0,('Final evaluation failed; inspect preserved evidence',log)
                result=c.read(output);assert result['status']=='completed'
                outcomes.append(dict(trial=item['trial'],seed=item['seed'],depth=item['depth'],budget=item['budget'],
                    split=split,bpb=result['bpb'],result_path=output.relative_to(c.HERE).as_posix(),result_sha256=r.digest(output),
                    log_path=log.relative_to(c.HERE).as_posix(),log_sha256=r.digest(log)))
                r.write_json(c.HERE/'final-evaluation-result.json',dict(status='running',freeze_sha256=r.digest(c.HERE/'evaluation-freeze.json'),outcomes=outcomes))
      except BaseException as exc:
        r.write_json(c.HERE/'final-evaluation-result.json',dict(status='failed',error=repr(exc),
            freeze_sha256=r.digest(c.HERE/'evaluation-freeze.json'),outcomes=outcomes))
        raise
      finally:
        r.verify_seal(r.STATE/'cache',c.read(c.HERE/'matrix-plan.json')['data_seal'])
    r.write_json(c.HERE/'final-evaluation-result.json',dict(status='completed',freeze_sha256=r.digest(c.HERE/'evaluation-freeze.json'),outcomes=outcomes,
        completed_at=datetime.now(timezone.utc).isoformat(),interpretation='Validation replay precedes one frozen held-out stage. No training, tuning or reselection based on test results.'))
    c.log('All validation replays and the single frozen test stage completed')
    shutil.copy2(c.HERE/'final-evaluation-result.json',directory/'result.json')
    reports.emit(directory)


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='evaluate':evaluate('--retry-failed' in sys.argv)
    else:raise ValueError(sys.argv[1])
