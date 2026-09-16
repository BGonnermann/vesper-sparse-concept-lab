"""Publish a measured long-run reference; consumes receipts, never trains or evaluates."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from chart_long_baseline import export,render
from foundation_data import ROOT,digest,write

HOME=ROOT/'runs/foundation_long_20260916'
DEST=ROOT/'reports/foundation-long-v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def mean_metrics(values):
    return dict(aggregate_bpb=statistics.mean(v['aggregate_bpb'] for v in values),domains={d:statistics.mean(v['domains'][d]['bpb'] for v in values) for d in ('general','technical')})
def change(value,reference):return dict(delta=value-reference,percent=100*(value-reference)/reference)

def main():
    run=HOME/'reference';cfg=read(run/'config.json');result=read(run/'result.json');assert result['status']=='completed' and result['tokens']==cfg['training_tokens']
    final=HOME/'final-evaluation';verification=read(final/'verification.json');assert verification['status']=='verified'
    assert verification['checkpoint_sha256']==result['latest']['sha256']==digest(result['latest']['path'])
    assert digest(run/'config.json')==result['config_sha256']
    assert verification['test_sha256']==digest(final/'test.json') and verification['validation_sha256']==digest(final/'validation.json')
    vals=read(run/'evaluations.json');losses=read(run/'losses.json');assert len(losses)==cfg['updates']
    assert [x['step'] for x in vals]==cfg['evaluation_steps'];assert vals[-1]['step']==cfg['updates']
    val=read(final/'validation.json');test=read(final/'test.json');pilot_root=ROOT/'runs/foundation_campaign'
    pv=[];pt=[]
    for seed in (211,212,213):
        p=read(pilot_root/f'data-expanded-s{seed}/result.json');assert p['status']=='completed' and p['dataset_fingerprint']==verification['dataset_fingerprint']
        pv.append(p['validation']);pt.append(read(pilot_root/f'data-expanded-s{seed}-test.json')['evaluation'])
    pilot_val=mean_metrics(pv);pilot_test=mean_metrics(pt)
    matched=next(x for x in vals if x['step']==512)['evaluation']
    comparisons={}
    for name,metric,reference in [('matched_tokens_validation',matched,pilot_val),('final_validation',val,pilot_val),('final_test',test,pilot_test)]:
        comparisons[name]=dict(aggregate=change(metric['aggregate_bpb'],reference['aggregate_bpb']),domains={d:change(metric['domains'][d]['bpb'],reference['domains'][d]) for d in ('general','technical')})
    best=min((x for x in vals if x['step']>0),key=lambda x:x['evaluation']['aggregate_bpb'])
    late=min(vals,key=lambda x:abs(x['step']-.75*cfg['updates']))
    gain=late['evaluation']['aggregate_bpb']-val['aggregate_bpb'];best_gap=val['aggregate_bpb']-best['evaluation']['aggregate_bpb']
    if best_gap>.02:assessment='Endpoint validation regressed more than0.02BPB from its best: evidence of overfitting/instability on this fixed validation set.'
    elif gain>.01:assessment='The model continued improving during the final quarter on this fixed validation set; it did not reach a clear aggregate plateau by this descriptive criterion.'
    else:assessment='Late aggregate improvement was below0.01BPB: a plateau/diminishing-returns signal on this fixed validation set.'
    domain_gaps={d:val['domains'][d]['bpb']-min(x['evaluation']['domains'][d]['bpb'] for x in vals) for d in ('general','technical')}
    slow=None;streak=[]
    for a,b in zip(vals,vals[1:]):
        if b['step']-a['step']!=1024:streak=[];continue
        improvement=a['evaluation']['aggregate_bpb']-b['evaluation']['aggregate_bpb'];streak=(streak+[b]) if improvement<.01 else []
        if len(streak)==3:slow=streak[0]['tokens'];break
    health=read(HOME/'reference.log.health.json');temps=[float(x.split(',')[0]) for x in health['samples']];board=[float(x.split(',')[1]) for x in health['samples']];util=[float(x.split(',')[2]) for x in health['samples']]
    attempts=[read(p) for p in sorted(run.glob('attempt-*.json'))]
    executed_updates=sum(sum(1 for _ in p.open(encoding='utf-8')) for p in run.glob('updates-attempt-*.jsonl'))
    checkpoints=[]
    for path in sorted(run.glob('checkpoint-*.pt')):
        receipt=read(str(path)+'.json');assert digest(path)==receipt['sha256'];checkpoints.append(receipt)
    reference=dict(schema=1,status='verified',training_code_revision=attempts[0]['git_revision'],config=cfg,config_sha256=result['config_sha256'],dataset_fingerprint=verification['dataset_fingerprint'],tokenizer_sha256=verification['tokenizer_sha256'],
        result=result,final_validation=val,final_test=test,pilot_joint_validation=pilot_val,pilot_joint_test=pilot_test,comparisons=comparisons,best_validation=dict(step=best['step'],tokens=best['tokens'],bpb=best['evaluation']['aggregate_bpb']),
        assessment=assessment,last_quarter_gain=gain,endpoint_gap_from_best=best_gap,domain_endpoint_gaps=domain_gaps,first_three_small_gain_intervals_start_tokens=slow,
        numerical_resume=read(HOME/'resume-verification.json'),attempts=attempts,actual_completed_update_work_tokens=executed_updates*16384,discarded_update_tokens=(executed_updates-cfg['updates'])*16384,
        supervised_wall_seconds=health['wall_seconds'],max_sampled_gpu_temperature_c=max(temps),max_sampled_whole_board_memory_mib=max(board),mean_sampled_gpu_utilization_percent=statistics.mean(util),
        retained_checkpoints=checkpoints,runtime=read(HOME/'runtime-identity.json'),limitations=['One seed, no superiority confidence interval','Frozen test was already opened by the prior campaign','Extensive repeated data; no unique-token or general-capability claim','Matched pilot token checkpoint uses different seed and LR/schedule history','Native BF16 training trajectories are not bitwise deterministic; restored state is exact'])
    DEST.mkdir(parents=True,exist_ok=True);write(DEST/'baseline-reference.json',reference)
    export(run,DEST/'chart-data');render(DEST/'chart-data',DEST/'charts')
    for name in ('validation.json','test.json','verification.json','freeze.json'):shutil.copy2(final/name,DEST/('final-'+name))
    write(DEST/'checkpoint-manifest.json',dict(final=result['latest'],retained=checkpoints,pruned_hash_receipts_path=str(run),policy='Latest+predecessor+three milestones+best validation; old campaign artifacts untouched'))
    shutil.copy2(run/'config.json',DEST/'config.json');shutil.copy2(HOME/'resume-verification.json',DEST/'resume-verification.json')
    for path in sorted(run.glob('samples-*.json')):shutil.copy2(path,DEST/path.name)
    lines=['# Long-run dense reference baseline','', '**'+assessment+'**','',
        '## Verified facts','',
        '- Fresh initialization, predeclared seed301. Accepted foundation-general-v2 architecture, tokenizer, corpus and80/20mixture retained. No architecture search, alternative tokenizer, mixture or mechanism.',
        f"- **{result['parameters']['total_parameters']:,}total and structurally active parameters**; external memory table **0bytes**. Active counts include full shared embedding tables, not measured per-token FLOPs.",
        f"- Completed **{result['tokens']:,}tokens**, **{result['tokens_per_parameter']:.4f}tokens/parameter**, **{result['step']:,}updates**. {len(vals)}scheduled validation points. Test evaluated once on the final fixed-budget checkpoint.",
        f"- Training revision `{attempts[0]['git_revision']}`. Exact config and identities: [baseline reference](baseline-reference.json), [checkpoint manifest](checkpoint-manifest.json).",'',
        '## Measurements','',
        '| Metric | General | Technical | Aggregate |','|---|---:|---:|---:|',
        f"| Final validation BPB | {val['domains']['general']['bpb']:.6f} | {val['domains']['technical']['bpb']:.6f} | {val['aggregate_bpb']:.6f} |",
        f"| Final test BPB | {test['domains']['general']['bpb']:.6f} | {test['domains']['technical']['bpb']:.6f} | {test['aggregate_bpb']:.6f} |",'',
        '**Lower BPB is better.** The test is the fixed foundation regression set, not a newly untouched benchmark.','',
        '| Comparison with direct joint accepted-pilot mean (seeds211/212/213) | BPB change | Percentage change |','|---|---:|---:|']
    for name,c in comparisons.items():lines.append(f"| {name} | {c['aggregate']['delta']:+.6f} | {c['aggregate']['percent']:+.2f}% |")
    lines += ['', 'These are descriptive comparisons, not paired superiority claims. The8,388,608-token checkpoint matches token count but not seed or schedule history. Prior tokenizer and pool gains are **not added together**. Per-domain absolute and percentage comparisons are in the reference JSON.', '',
        f"Measured training-update time: **{result['update_seconds']/3600:.3f}h**; trainer wall **{result['wall_seconds']/3600:.3f}h**; supervised wall including process startup/exit **{health['wall_seconds']/3600:.3f}h**. Throughput **{result['tokens_per_second']:,.0f}tok/s** over all updates; steady **{result['steady_tokens_per_second']:,.0f}tok/s**. Allocator peaks **{result['peak_allocated_bytes']/2**20:.1f}MiB allocated /{result['peak_reserved_bytes']/2**20:.1f}MiB reserved**.", '',
        f"Evaluation time **{result['evaluation_seconds']:.1f}s**; checkpoint saves **{result['checkpoint_seconds']:.1f}s**; milestone generation **{result['sample_seconds']:.1f}s**. Maximum sampled temperature **{max(temps):.0f}°C**; sampled whole-board memory max **{max(board):.0f}MiB**; mean sampled GPU utilization **{statistics.mean(util):.1f}%** (startup and diagnostics included; samples are not exact board peaks).",'',
        f"Actual domain passes: general **{result['data_progress']['general']['passes']:.3f}**, technical **{result['data_progress']['technical']['passes']:.3f}**. The pool has11,378,155one-pass token positions includingBOS, not469M unique tokens. Full-model preflight/calibration work is separately retained; it is not part of the reference's token budget.", '',
        '## Learning curve and inferences','',
        f"Best recorded validation: **{best['evaluation']['aggregate_bpb']:.6f}BPB**, step **{best['step']}**, **{best['tokens']:,}tokens**. Endpoint gap from best: **{best_gap:+.6f}BPB**. Final-quarter aggregate gain: **{gain:+.6f}BPB**.",
        f"Domain endpoint gaps from their own best observations: general **{domain_gaps['general']:+.6f}**, technical **{domain_gaps['technical']:+.6f}BPB**.",
        ('First three consecutive1024-update intervals each gaining less than0.01BPB begin at **'+f'{slow:,}'+'tokens**. This descriptive slowing criterion is not a stopping rule or confidence interval.' if slow else 'The three-consecutive-small-gain-interval criterion was not reached.'),
        'Token exposure and the original budget-relative LR/weight-decay schedule change together. Late gains cannot be attributed solely to more tokens. Repeated-data overfitting and insufficient unique data can coexist; this is not evidence of broad assistant capability.', '',
        '## Charts and underlying data','',
        '[All chart data](chart-data/) retains every update and validation. Re-rendering requires no training. SVG and300DPI PNG versions are included.','']
    for path in sorted((DEST/'charts').glob('*.png')):lines.append(f'![{path.stem}](charts/{path.name})')
    lines += ['', '## Qualitative sample observations','',
        'Fixed six prompts, seed20260915, top-k40, temperature0.8 and64generated tokens; samples at8.39M,234.88M and469.76Mtokens. Prose, factual-looking text, code, mathematics and instruction-like text are included. Base model, no instruction tuning. See [sample inspection](SAMPLES.md); syntax and repetition observations are not functional or factual accuracy claims.', '',
        '## Failures, interruptions and limitations','',
        f"Reference attempts: **{len(attempts)}**; discarded/replayed successful-update tokens: **{reference['discarded_update_tokens']:,}**. Failed-attempt logs are preserved. One seed cannot establish superiority or bound long-run seed variance.",
        'The initial strict resume trajectory comparison failed. Independent uninterrupted BF16 repeats also diverged. Exact restoration of model/optimizer/RNG/sampler was then verified before continuation; four-step continuation loss/BPB differences were below0.0001. That short-test tolerance is not a full-run reproducibility bound. No kernel or optimizer policy was changed to force matching.',
        f"Final validation fresh-process replay difference: **{verification['validation_absolute_difference']:.9g}BPB**; weights unchanged. Final checkpoint was loaded and used for fresh-process generation.", '',
        '## Work not completed','',
        '- No multi-seed long-run comparison, broad capability benchmark or8GB-card deployment measurement.',
        '- No new sources, architecture search, optimizer tuning, alternate mixture/tokenizer or production-weight promotion.',
        '- No proof of semantic/external-benchmark decontamination. No claim that hundreds of millions of repeated tokens equal that much unique training data.', '',
        '## Exact reference for the next adaptive campaign','',
        f"Primary fixed-budget anchor: **{val['aggregate_bpb']:.6f}validation BPB** at **{result['tokens']:,}tokens**, with the per-domain targets above and **{health['wall_seconds']/3600:.3f}h supervised wall** on the recorded runtime. The complete best-so-far frontier is also a reference: do not claim a gain by comparing a selected cheap checkpoint only with an inferior endpoint. Compare matched-token and matched-wall tracks separately; record total/active parameters, table size and memory costs. Never optimize against the opened test scores.", '',
        '## Next three highest-value actions','',
        '1. Use this frozen configuration/frontier as the adaptive control; confirm small gains with matched fresh-seed controls rather than treating this one seed as a superiority distribution.',
        '2. Prioritize a separately declared unique-data expansion with new held-out evidence if repetition/plateau dominates; do not silently change this reference pool.',
        '3. Add bounded task-level evaluations before claiming useful code, mathematics, instruction-following or factual accuracy.', '',
        '## Reproduction and inference','',
        'The original run directory must not be overwritten. Training/resume are for recovery or a separately named replication, not a request to restart the completed campaign. See [protocol](../../docs/foundation-long-v1.md) for calibration, source locks and resume limitations.','',
        '```powershell',"$py = '.autoresearch/upstream/.venv/Scripts/python.exe'",
        '& $py scripts/long_baseline_campaign.py train --config experiments/mainline/foundation-long-v1.json',
        '# Recovery only, same frozen config and source bytes:',
        '& $py scripts/supervise_long_resume.py --config experiments/mainline/foundation-long-v1.json --run runs/foundation_long_20260916/reference --deadline "2026-09-16T12:46:33.205990-04:00"',
        '# Final evaluation command used ONCE (will refuse a second test opening):',
        '& $py scripts/evaluate_long_baseline.py --run runs/foundation_long_20260916/reference --mode finalize --output runs/foundation_long_20260916/final-evaluation',
        '# Rebuild all charts solely from committed numeric data:',
        '& $py scripts/chart_long_baseline.py --data reports/foundation-long-v1/chart-data --output reports/foundation-long-v1/charts',
        '# Load final checkpoint in a fresh process and generate fixed diagnostic samples:',
        '& $py scripts/evaluate_long_baseline.py --run runs/foundation_long_20260916/reference --checkpoint runs/foundation_long_20260916/reference/checkpoint-028672.pt --mode generate --output runs/foundation_long_20260916/manual-generation-01.json',
        '```','',
        'Large checkpoints, raw corpora and cached shards remain outside Git. Checkpoint locations and SHA256 hashes are in the manifest. Final closeout clock, tests and commits are recorded separately.']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    curve=['# Every scheduled validation (lower BPB is better)','','| Step | Tokens | Wall hours | Aggregate | General | Technical |','|---:|---:|---:|---:|---:|---:|']
    for x in vals:
        v=x['evaluation'];curve.append(f"| {x['step']} | {x['tokens']:,} | {x['wall_seconds']/3600:.3f} | {v['aggregate_bpb']:.6f} | {v['domains']['general']['bpb']:.6f} | {v['domains']['technical']['bpb']:.6f} |")
    (DEST/'CURVE.md').write_text('\n'.join(curve)+'\n',encoding='utf-8')
    print(assessment,flush=True)

if __name__=='__main__':main()
