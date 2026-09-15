"""Artifact-only campaign conclusions and bounded reporting checks."""
import json
from pathlib import Path
import statistics as stats

import autoresearch as r
import experiment_reports as reports
import overnight_campaign as campaign
from overnight_report import final_status

HERE = campaign.HERE


def summarize(rows):
    assert rows
    return dict(runs=len(rows), bpb_mean=stats.mean(x['bpb'] for x in rows),
        bpb_min=min(x['bpb'] for x in rows), bpb_max=max(x['bpb'] for x in rows),
        bpb_sample_sd=stats.stdev(x['bpb'] for x in rows) if len(rows)>1 else None,
        mean_timed_tokens_per_second=stats.mean(x['timed_tokens_per_second'] for x in rows),
        mean_update_wall_seconds=stats.mean(x['all_update_seconds'] for x in rows),
        mean_child_wall_seconds=stats.mean(x['wall_seconds'] for x in rows),
        peak_allocated_mib=max(x['allocated_mib'] for x in rows),
        peak_reserved_mib=max(x['reserved_mib'] for x in rows),
        total_parameters=sorted({x['total_parameters'] for x in rows}),
        active_parameters=sorted({x['active_parameters'] for x in rows}),
        width=sorted({x['width'] for x in rows}),
        memory_table_bytes=sorted({x['memory_table_bytes'] for x in rows}))


def main():
    assert final_status(False,[],[],[]) == 'running'
    assert final_status(True,[],[],[{'seed':43},{'seed':44}]) == 'completed'
    assert final_status(True,[{'status':'failed'}],[],[]) == 'failed'
    for active, pairs in [([{'run':'active'}],[{'seed':43},{'seed':44}]),([],[])]:
        try:
            final_status(True,[],active,pairs)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid final state accepted')
    result = campaign.read(HERE/'result.json')
    assert result['status'] == 'completed'
    rows = [row for path in sorted(HERE.glob('rows-*-result.json')) for row in campaign.read(path)['rows']]
    assert len(rows) == result['runs']
    assert all(row['tokens']==8388608 for row in rows)
    groups = {label:summarize([x for x in rows if x['label']==label and x['seed']==42])
              for label in sorted({x['label'] for x in rows if x['seed']==42})}
    confirmation = campaign.read(HERE/'confirmation-selection.json')['winner']
    candidate, control = confirmation['label'], confirmation['control']
    independent = []
    for seed in (43,44):
        a = [x for x in rows if x['label']==candidate and x['seed']==seed]
        b = [x for x in rows if x['label']==control and x['seed']==seed]
        n = min(len(a),len(b))
        independent.append(dict(seed=seed,candidate=summarize(a),control=summarize(b),
            difference_all_repeats=stats.mean(x['bpb'] for x in a)-stats.mean(x['bpb'] for x in b),
            first_pair_difference=a[0]['bpb']-b[0]['bpb'],balanced_repeats_per_condition=n,
            balanced_prefix_difference=stats.mean(x['bpb'] for x in a[:n])-stats.mean(x['bpb'] for x in b[:n])))
    paths = [p for p in HERE.rglob('*') if p.is_file()]
    audit = dict(kind='campaign_conclusions',status='completed',reporting_checks=5,
        trials=len(rows),unique_configurations=len(groups),total_training_tokens=sum(x['tokens'] for x in rows),
        seed42_groups=groups,independent_confirmation=independent,
        independent_seed_mean_difference=stats.mean(x['difference_all_repeats'] for x in independent),
        first_pair_independent_mean_difference=stats.mean(x['first_pair_difference'] for x in independent),
        balanced_independent_mean_difference=stats.mean(x['balanced_prefix_difference'] for x in independent),
        campaign_artifact_bytes=sum(p.stat().st_size for p in paths),
        campaign_checkpoint_bytes=sum(p.stat().st_size for p in paths if p.suffix=='.pt'),
        scripts_sha256={name:r.digest(campaign.ROOT/'scripts'/name) for name in
            ('overnight_campaign.py','overnight_report.py','overnight_conclusions.py')})
    r.write_json(HERE/'conclusions-result.json',audit)
    lines = ['# Campaign conclusions', '',
        f"{len(rows)} completed full trials across {len(groups)} distinct configurations. No candidate-count cap. "
        'Every full trial used 512 updates and 8,388,608 tokens. Synthetic gates were separate and are not research scores.', '',
        '## Independent confirmation', '',
        f'Frozen candidate: {candidate}; control: {control}. Both have 135,267,480 total/active parameters and width768. Only matrix LR differs.', '',
        '| Seed | Candidate BPB mean | Control BPB mean | Difference | Candidate/control repeats | First-pair difference |',
        '|---|---:|---:|---:|---|---:|']
    for pair in independent:
        a,b=pair['candidate'],pair['control']
        lines.append(f"| {pair['seed']} | {a['bpb_mean']:.6f} | {b['bpb_mean']:.6f} | {pair['difference_all_repeats']:+.6f} | {a['runs']}/{b['runs']} | {pair['first_pair_difference']:+.6f} |")
    lines += ['', f"Mean difference across the two independent seeds: {audit['independent_seed_mean_difference']:+.6f} BPB. "
        f"First-pair-only sensitivity: {audit['first_pair_independent_mean_difference']:+.6f}; balanced-repeat sensitivity: {audit['balanced_independent_mean_difference']:+.6f}.", '',
        'The seeds disagree on direction. Do not promote LR0.03 as a reliable improvement. Seed42 was used for selection and is not a third independent confirmation. '
        'Repeats estimate execution variability, not additional seeds. Unequal repeat counts are disclosed and sensitivity calculations do not treat repeats as independent samples.', '',
        '## Seed42 screen and repeat means', '',
        '| Label | Repeats | BPB mean [min,max] | Timed tok/s mean | Update wall s | Child wall s | Alloc/reserved MiB peak | Total/active parameters | Width |',
        '|---|---:|---|---:|---:|---:|---|---|---:|']
    for label in ['D','M','DG','M-experts2','M-router_lr0.003','DG-layer0','D-depth8','D-depth10','D-depth12','D12-lr0.03']:
        g=groups[label]
        lines.append(f"| {label} | {g['runs']} | {g['bpb_mean']:.6f} [{g['bpb_min']:.6f},{g['bpb_max']:.6f}] | {g['mean_timed_tokens_per_second']:.1f} | {g['mean_update_wall_seconds']:.1f} | {g['mean_child_wall_seconds']:.1f} | {g['peak_allocated_mib']:.1f}/{g['peak_reserved_mib']:.1f} | {g['total_parameters'][0]:,}/{g['active_parameters'][0]:,} | {g['width'][0]} |")
    lines += ['', 'Larger dense depth/width produced the largest observed quality improvement, with more parameters and slower updates. '
        'This is equal-token evidence, not an equal-time or equal-compute efficiency claim. No independent depth8/depth12 confirmation pair was run: the reserved independent comparison tested LR within depth12. '
        'MoE/router/memory findings remain selection-seed screens. Expert utilization and every configuration are preserved in the per-run reports and row receipts.', '',
        'Timed throughput excludes the first11 updates; all training tokens include them. Synchronized update wall time includes CPU dispatch and optimizer work, not just GPU kernel-busy time. '
        'Allocator peaks do not measure total board usage.', '',
        '## Correctness, diagnostics and provenance', '',
        'Initial CPU/CUDA suites passed52 tests each; expanded-depth suites passed54 each. Full-context depth10/12 synthetic gates verified causality, outputs, gradients, optimizer coverage, save/load and comfortable GPU fit. '
        'Previously gated LR settings were combined only after separate shape and LR screens; the inherited gate receipt explicitly says no rerun was performed for that configuration-only amendment.', '',
        'All training trials completed. A reporting-only identity assertion failed because schedule.json also contains intentionally different optimizer groups. '
        'The diagnostic was preserved; the repaired check compares exact schedule definitions/updates, protocol, batch identities and data seals. Full optimizer groups remain captured. '
        'Historical selection boilerplate saying no combined winners was stale for the four depth12/LR trials; their prospective hypotheses and matching controls explicitly identify the combination. '
        'The controller wording was corrected after training selection without rewriting those historical receipts.', '',
        'Repeated larger-depth runs vary despite identical recorded initialization and source hashes; no specific numerical kernel is established as the cause. '
        'The historical M-control discrepancy remains unresolved. Do not claim deterministic training, equivalence, or statistical proof.', '',
        'Prior artifacts are hash-verified separately. Exact executed sources are archived by hash; publication commits are never retroactive run provenance. '
        'NCP was deferred rather than added to fill a category. No dependency upgrades, cloud jobs, paid services or deletion.', '',
        '## Retention', '',
        f"Campaign artifacts: {audit['campaign_artifact_bytes']/2**30:.2f} GiB; checkpoints: {audit['campaign_checkpoint_bytes']/2**30:.2f} GiB. "
        'Preserve all artifacts now. Proposed later policy, requiring separate deletion approval: keep compact reports, hashes, source snapshots, controls, independent-seed checkpoints and failures indefinitely; '
        'archive redundant same-seed checkpoints and large traces after a30-day review. Never remove datasets or provenance needed to reproduce retained results.', '']
    body='\n'.join(lines)
    (HERE/'conclusions.md').write_text(body,encoding='utf-8')
    reports.emit(HERE);reports.index()
    report_id,_=reports.identity(HERE,campaign.ROOT)
    (campaign.ROOT/'reports/experiments'/report_id/'conclusions.md').write_text(body,encoding='utf-8')
    print(json.dumps({key:audit[key] for key in ('trials','unique_configurations','total_training_tokens','independent_seed_mean_difference','first_pair_independent_mean_difference','balanced_independent_mean_difference','campaign_artifact_bytes')},indent=2))


if __name__=='__main__':
    main()
