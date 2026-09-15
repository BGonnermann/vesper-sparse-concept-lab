"""Render final held-out findings only after complete evidence verification."""
from datetime import datetime,timezone
import hashlib
import statistics

import autoresearch as r
import depth_campaign as c
import depth_report as report
import experiment_reports as reports


def write():
    audit=c.read(c.HERE/'audit-result.json')
    assert audit['status']=='completed' and audit['final_evaluation_verified']
    assert 0<=(datetime.now(timezone.utc)-datetime.fromisoformat(audit['measured_at'])).total_seconds()<900
    assert audit['trial_result_hashes']=={x['trial']:r.digest(c.HERE/x['trial']/'result.json') for x in c.records()}
    frozen=c.read(c.HERE/'evaluation-freeze.json');evaluations=c.read(c.HERE/'final-evaluation-result.json')
    decision=report.write(final=True)
    assert decision==frozen['validation_recommendation'],'Validation selection changed after test freeze'
    root=r.ROOT/'reports/experiments'/reports.identity(c.HERE,r.ROOT)[0]
    path=root/'CAMPAIGN.md';text=path.read_text(encoding='utf-8')
    results=[x for x in evaluations['outcomes'] if x['split']=='test']
    by={(x['seed'],x['depth'],x['budget']):x for x in results}
    lines=[f"Frozen validation-selected candidate: D{frozen['candidate_depth']}@{frozen['candidate_budget']}x. Reference: D{frozen['reference_depth']}@{frozen['reference_budget']}x. D12@4x is a predeclared maximum-budget context condition. The recommendation above remains based on validation only.",'',
        '| Seed | D6@4x test BPB | D12@2x test BPB | D6@4x minus D12@2x | D12@4x test BPB | D12@4x minus D6@4x |',
        '|---:|---:|---:|---:|---:|---:|']
    deltas=[];same=[]
    for seed in decision['complete_seeds']:
        a,b,d=[by[seed,depth,budget]['bpb'] for depth,budget in ((6,4),(12,2),(12,4))]
        deltas.append(a-b);same.append(d-a)
        lines.append(f'| {seed} | {a:.6f} | {b:.6f} | {a-b:+.6f} | {d:.6f} | {d-a:+.6f} |')
    lines+=['',f"Test efficiency-pair mean difference: {statistics.mean(deltas):+.6f} BPB; {sum(x<0 for x in deltas)}/{len(deltas)} favor D6@4x. Same-budget test mean D12@4x minus D6@4x: {statistics.mean(same):+.6f}; {sum(x<0 for x in same)}/{len(same)} favor D12.",'',
        'One frozen test stage used65,536 packed tokens from test rows0–9999, not the full10,000-document split. All seeds are shown. No checkpoint selection, tuning or training followed test results. This stage does not establish performance beyond this TinyStories split.']
    text=text.replace('Pending. Exact candidate/reference checkpoint identities will be frozen before test evaluation; no subsequent training or reselection.','\n'.join(lines))
    replay=[c.read(c.HERE/x['result_path']) for x in evaluations['outcomes'] if x['split']=='val']
    text+='\n\n## Verification and retained storage\n\n'
    text+=f"All {len(replay)} completed checkpoints reproduced their validation BPB within1e-6. Maximum absolute difference: {max(x['absolute_validation_bpb_difference'] for x in replay):.3g}. The final audit verified consumed stream prefixes, paired schedules, same-architecture initial weights, optimizer coverage, source archives against committed Git bytes, evaluator split dispatch, byte accounting, immutable weights and the exact test inventory.\n\n"
    text+=f"New checkpoint storage: {audit['checkpoint_bytes']/2**30:.2f} GiB; free disk at audit: {audit['free_disk_bytes']/2**30:.2f} GiB. Existing artifacts and baseline were preserved.\n\n"
    text+='The pre-training review found and fixed a live-protocol reconstruction gap before any training. The first frozen controller/plan copies remain preserved. Synthetic red tests and their successful fixes are retained. Final-stage infrastructure was reviewed before any held-out scores.\n\n'
    text+='## Scope and next step\n\n'
    text+='The recommendation applies to the overlapping measured training-time range and this dataset, tokenizer, optimizer policy and native Windows runtime. D12@4x has no D6@8x comparator, so the largest-time efficiency frontier remains unmeasured. If longer budgets continue improving, another separately authorized study would be needed to locate diminishing returns. No convergence, trading-performance or general-coding claim is made. No production checkpoint was replaced. No subsequent campaign is launched.\n'
    path.write_text(text,encoding='utf-8')
    r.write_json(c.HERE/'result.json',dict(kind='depth_long_campaign',status='completed',
        recommendation=decision['recommendation'],complete_seeds=decision['complete_seeds'],
        completed_trials=len(report.collect()),attempted_trials=len(c.records()),
        audit_sha256=r.digest(c.HERE/'audit-result.json'),freeze_sha256=r.digest(c.HERE/'evaluation-freeze.json'),
        evaluation_sha256=r.digest(c.HERE/'final-evaluation-result.json'),deadline=c.DEADLINE.isoformat(),
        report_saved_at=datetime.now(timezone.utc).isoformat(),report_sha256=r.digest(path)))
    reports.emit(c.HERE);reports.index()
    c.log('Final report saved with held-out stage and complete audit')


if __name__=='__main__':write()
