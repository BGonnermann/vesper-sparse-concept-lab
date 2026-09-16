"""Bounded post-training verification; never launches training or selects test checkpoints."""
from datetime import datetime
import time
from foundation_data import ROOT,write
from long_baseline import HOME,log
from long_baseline_runtime import read
from long_baseline_campaign import checked
import autoresearch as r


def main():
    status=dict(status='waiting_for_fixed_reference');write(HOME/'finish-status.json',status)
    deadline=datetime.fromisoformat(read(HOME/'schedule.json')['stop'])
    while True:
        if datetime.now().astimezone()>=deadline:raise TimeoutError('Eight-hour ceiling')
        attempts=list((HOME/'reference').glob('attempt-*.json'))
        if attempts and read(sorted(attempts)[-1])['status']=='failed':raise RuntimeError('Training failure requires diagnosis; no blind recovery')
        result=HOME/'reference/result.json'
        controller=(HOME/'reference-controller.log').read_text(errors='replace')
        if result.exists() and read(result)['status']=='completed' and 'Reference training complete.' in controller:break
        time.sleep(30)
    log('Reporting reserve: fixed training complete; starting fresh-process final validation and one final test evaluation')
    status['status']='final_evaluation';write(HOME/'finish-status.json',status)
    checked([str(r.runtime_python()),'-B',str(ROOT/'scripts/evaluate_long_baseline.py'),'--run',str(HOME/'reference'),'--mode','finalize','--output',str(HOME/'final-evaluation')],HOME/'final-evaluation.log',600)
    status['status']='charts_and_report';write(HOME/'finish-status.json',status)
    checked([str(r.runtime_python()),'-B',str(ROOT/'scripts/report_long_baseline.py')],HOME/'final-report.log',600)
    status.update(status='verified_pending_sample_review_and_git_closeout',completed_at=datetime.now().astimezone().isoformat());write(HOME/'finish-status.json',status)
    log('Final validation/test, fresh-process generation, all seven charts and numeric report complete. Manual sample/chart inspection and Git closeout remain.')

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(HOME/'finish-failure.json',dict(status='failed',error=repr(exc)));log('FINALIZATION FAILED: '+repr(exc));raise
