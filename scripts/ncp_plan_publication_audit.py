"""Separate local plan freezing from the first Git publication of each exact hash."""
import json
import subprocess
from pathlib import Path

import autoresearch as r
import ncp_campaign as c
import experiment_reports as reports


def main():
    files=['confirmation-selection.json','crossed-order-plan.json','module-initialization-plan.json',
        'no-future-plan.json','feedback-interventions-plan.json','concept-baselines-plan.json','depth-grid-plan.json']
    expected={name:r.digest(c.HERE/name) for name in files}
    report_path='reports/experiments/'+reports.identity(c.HERE,r.ROOT)[0]+'/report.json'
    commits=subprocess.check_output(['git','log','--reverse','--format=%H','--',report_path],cwd=r.ROOT,text=True).splitlines()
    found={}
    for commit in commits:
        raw=subprocess.check_output(['git','show',commit+':'+report_path],cwd=r.ROOT)
        evidence=json.loads(raw)['evidence']
        for name,value in expected.items():
            if name not in found and evidence.get(name,{}).get('sha256')==value:
                stamp=subprocess.check_output(['git','show','-s','--format=%cI',commit],cwd=r.ROOT,text=True).strip()
                plan=c.read(c.HERE/name)
                found[name]=dict(sha256=value,first_matching_publication_commit=commit,commit_time=stamp,
                    local_recorded_freeze_time=plan.get('frozen_at',plan.get('selected_at')))
        if len(found)==len(expected): break
    assert set(found)==set(expected),'A final plan hash has not been published'
    r.write_json(c.HERE/'plan-publication-audit-result.json',dict(kind='plan_publication_audit',status='completed',
        plans=found,script_sha256=r.digest(Path(__file__)),
        measured_at=c.datetime.now(c.timezone.utc).isoformat(),
        interpretation='Git evidence identifies the first committed report containing each exact current plan-file hash. Local freeze timestamps and first publication are distinct. Some diagnostic plans were frozen locally before execution but first included in Git reporting after results. Earlier progress messages describing every freeze-only/publish-only call as publication of the new plan were too strong. Existing hypotheses and implementation commits may predate these plan-hash publications; no training or inference result is changed.'))
    for name,item in found.items(): print(name,item['commit_time'],item['first_matching_publication_commit'][:8])


if __name__=='__main__':
    main()
