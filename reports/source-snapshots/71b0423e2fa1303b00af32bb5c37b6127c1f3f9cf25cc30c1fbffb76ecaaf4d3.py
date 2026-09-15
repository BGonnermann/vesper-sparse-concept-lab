"""One-shot final report closure; never launches training or a subsequent campaign."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import autoresearch as r
import ncp_campaign as c
import experiment_reports as reports

FINALIZE_AT=datetime.fromisoformat('2026-09-15T19:35:00+00:00')
SOURCE=Path(__file__).read_bytes()


def prepare():
    archive=c.HERE/'deadline-closeout-driver.py'
    if archive.exists(): assert archive.read_bytes()==SOURCE
    else: archive.write_bytes(SOURCE)
    plan=dict(kind='deadline_closeout_plan',status='prepared',finalize_at=FINALIZE_AT.isoformat(),
        deadline=c.DEADLINE.isoformat(),script_sha256=r.digest(archive),training_updates=0,
        steps=['fresh evidence audit','findings refresh','gated final report','publication','committed report/index verification'],
        interpretation='One campaign only. No training or automatic subsequent campaign. Final report is saved before the deadline; publication failure preserves local results.')
    path=c.HERE/'deadline-closeout-plan.json'
    if path.exists(): assert c.read(path)==plan
    else: r.write_json(path,plan)


def run_step(name,arguments,timeout):
    assert c.remaining()>timeout+20,'Insufficient time for bounded closeout step'
    log=c.HERE/('deadline-'+name+'.log')
    c.log('Final closure step '+name+'; log='+str(log))
    with log.open('x',encoding='utf-8') as handle:
        process=subprocess.run([str(r.runtime_python()),'-B',*arguments],cwd=r.ROOT,env=r.environment(),
            stdout=handle,stderr=subprocess.STDOUT,timeout=timeout)
    assert process.returncode==0,('Closeout step failed; preserved log',log)
    return dict(log=log.name,sha256=r.digest(log),returncode=process.returncode)


def main():
    prepare()
    final_path=c.HERE/'deadline-closeout-result.json'
    assert not final_path.exists(),'A previous closeout result exists; inspect it rather than repeating'
    c.log('Reporting reserve: closure scheduled19:35UTC, campaign deadline19:41:32UTC; no more GPU jobs')
    heartbeat=time.monotonic()
    while datetime.now(timezone.utc)<FINALIZE_AT:
        seconds=(FINALIZE_AT-datetime.now(timezone.utc)).total_seconds()
        time.sleep(max(0,min(40,seconds)))
        if time.monotonic()-heartbeat>=300:
            c.log('Reporting reserve: evidence preserved; waiting for final deadline audit; no GPU jobs')
            heartbeat=time.monotonic()
    result=dict(kind='deadline_closeout',status='running',started_at=datetime.now(timezone.utc).isoformat(),
        deadline=c.DEADLINE.isoformat(),steps={},training_updates=0)
    r.write_json(final_path,result)
    try:
        result['steps']['audit']=run_step('audit',['scripts/ncp_audit.py'],120)
        result['steps']['findings']=run_step('findings',['scripts/ncp_findings.py'],30)
        result['steps']['report']=run_step('report',['scripts/ncp_report.py','--final'],30)
        result['local_final_report_saved']=True
        r.write_json(final_path,result)
        result['steps']['publication']=run_step('publication',['scripts/ncp_search.py','--publish-only'],120)
        publication=c.read(c.HERE/'publication-result.json')
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True,timeout=15).strip()
        assert publication['publication_commit']==head and publication['verified_remote']
        identifier=reports.identity(c.HERE,r.ROOT)[0]
        prefix='reports/experiments/'+identifier+'/'
        def committed(path):
            return subprocess.check_output(['git','show','HEAD:'+path],cwd=r.ROOT,timeout=15)
        report=json.loads(committed(prefix+'report.json'))
        index=json.loads(committed('reports/index.json'))
        assert report['outcome']=='completed'
        entry=next(row for row in index['experiments'] if row['experiment_id']==identifier)
        assert entry['outcome']=='completed'
        hashes={}
        for filename in ('CAMPAIGN.md','FINDINGS.md'):
            data=committed(prefix+filename)
            local=(r.ROOT/prefix/filename).read_bytes().replace(b'\r\n',b'\n')
            assert data==local
            hashes[filename]=hashlib.sha256(data).hexdigest()
        assert committed(prefix+'CAMPAIGN.md').startswith(b'# NCP campaign final report')
        result.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),
            publication_commit=head,verified_remote=True,committed_report_hashes=hashes,
            committed_progress_index_completed=True,
            interpretation='Final report and completed progress index are committed and remotely verified. Scientific work is complete; stop at the existing deadline, with no subsequent campaign.')
        c.log('FINAL REPORT VERIFIED '+head+'; no further campaign will launch')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),failed_at=datetime.now(timezone.utc).isoformat())
        c.log('Final closure failure preserved: '+repr(exc))
        raise
    finally:
        r.write_json(final_path,result)


if __name__=='__main__':
    if '--prepare-only' in sys.argv: prepare()
    else: main()
