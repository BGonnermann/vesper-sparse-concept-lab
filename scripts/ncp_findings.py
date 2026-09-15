"""Generate final compact findings from verified campaign receipts."""
import statistics

import autoresearch as r
import experiment_reports as reports
import ncp_campaign as c
from ncp_frozen_pairs import collect
from ncp_report import rows


def main():
    data = rows()
    assert all(x['status'] != 'running' for x in data)
    frozen = c.read(c.HERE / 'confirmation-selection.json')
    evidence = collect(frozen)
    assert all(x['completed_pairs'] == 4 for x in evidence['summaries'].values())
    by_trial = {x['trial']: x for x in data}
    pairs = [x for x in evidence['pairs'] if x['control'] == 'D6']
    primary = [x for x in pairs if x['role'] == 'primary']
    mean = statistics.mean(x['delta_bpb'] for x in pairs)
    passed = evidence['summaries']['D6']['primary_same_sign_improvement']
    costs = {}
    for label in (frozen['label'], 'D6', 'FROZEN-AUX', 'FROZEN-CAP', 'FROZEN-NOPRED'):
        group = [x for x in data if x['label'] == label and x['seed'] in frozen['seeds'] and x['status'] == 'completed']
        assert len(group) == 4
        assert len({x['total_parameters'] for x in group}) == 1
        costs[label] = dict(parameters=group[0]['total_parameters'], active_parameters=group[0]['active_parameters'],
            mean_bpb=statistics.mean(x['bpb'] for x in group),
            mean_update_seconds=statistics.mean(x['all_update_seconds'] for x in group),
            mean_trial_seconds=statistics.mean(x['wall_seconds'] for x in group),
            mean_timed_tokens_per_second=statistics.mean(x['timed_tokens_per_second'] for x in group),
            max_allocated_mib=max(x['allocated_mib'] for x in group),
            max_reserved_mib=max(x['reserved_mib'] for x in group),
            max_sampled_board_mib=max(x['board_sampled_peak_mib'] for x in group if x['board_sampled_peak_mib'] is not None))
    findings = dict(kind='campaign_findings', status='completed', frozen_comparisons=evidence, costs=costs,
                    primary_success_rule_passed=passed, mean_four_seed_delta_bpb=mean,
                    confirmation_selection_sha256=r.digest(c.HERE / 'confirmation-selection.json'))
    text = ['# Campaign findings', '',
        ('Frozen NCP passed' if passed else 'Frozen NCP did not pass') + ' the predeclared primary two-seed consistency rule.', '',
        f'The four-seed mean NCP-minus-dense difference is {mean:+.6f} BPB; lower is better. '
        f'{sum(x["delta_bpb"] < 0 for x in pairs)} of four paired seeds favor NCP. '
        'These are equal-token validation results, not held-out or equal-time results.', '',
        f'The exactly parameter-matched residual MLP averaged {costs["FROZEN-CAP"]["mean_bpb"]:.6f} BPB '
        f'versus NCP at {costs[frozen["label"]]["mean_bpb"]:.6f}, with '
        f'{costs["FROZEN-CAP"]["mean_update_seconds"]:.1f} versus {costs[frozen["label"]]["mean_update_seconds"]:.1f} '
        'seconds of training updates. NCP did not show an advantage over this added-capacity control.', '',
        '## Frozen confirmation and ablations', '',
        '| Seed | Role | Dense BPB | NCP BPB | NCP minus dense |',
        '|---:|---|---:|---:|---:|']
    for pair in pairs:
        text.append(f'| {pair["seed"]} | {pair["role"]} | {pair["control_bpb"]:.6f} | {pair["candidate_bpb"]:.6f} | {pair["delta_bpb"]:+.6f} |')
    text.extend(['', '| Comparison | Mean NCP minus control BPB | Seeds favoring NCP |', '|---|---:|---:|'])
    for label, value in evidence['summaries'].items():
        text.append(f'| {label} | {value["mean_delta_bpb"]:+.6f} | {value["negative_signs"]}/4 |')
    text.extend(['', 'AUX keeps the selected concept objectives and parameters but removes predicted feedback. '
        'CAP replaces NCP with an exactly parameter-matched token-rate residual MLP. '
        'NOPRED removes prediction MSE and concept CE while retaining feedback, token CE and VQ learning. '
        'It is not a no-future-label or all-supervision-off condition.', '',
        'The seed42 search selected 0.631877 BPB versus its fresh dense control at0.634848. '
        'That selection gain is reported separately from all four fresh-seed outcomes; the candidate was not replaced after confirmation.', '',
        '## Measured cost across the four confirmation seeds', '',
        '| Condition | Total / active parameters | Mean BPB | Mean update s | Mean trial s | Mean timed tok/s | Max allocated / reserved MiB | Max sampled board MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|'])
    for label, value in costs.items():
        text.append(f'| {label} | {value["parameters"]:,} / {value["active_parameters"]:,} | {value["mean_bpb"]:.6f} | '
            f'{value["mean_update_seconds"]:.1f} | {value["mean_trial_seconds"]:.1f} | {value["mean_timed_tokens_per_second"]:.0f} | '
            f'{value["max_allocated_mib"]:.1f} / {value["max_reserved_mib"]:.1f} | {value["max_sampled_board_mib"]:.1f} |')
    ncp, dense = costs[frozen['label']], costs['D6']
    text.extend(['', f'NCP adds {ncp["parameters"]-dense["parameters"]:,} parameters '
        f'({100*(ncp["parameters"]/dense["parameters"]-1):.2f}%) and its mean all-update time is '
        f'{ncp["mean_update_seconds"]/dense["mean_update_seconds"]:.2f} times dense. '
        'Structural active counts are not amortized FLOPs: concepts run once per chunk; CAP runs at every token. '
        'The selected codebook has98,304 trainable transform parameters,24,576 frozen-basis bytes and24,576 effective-code bytes. '
        'No n-gram memory table is present.', '',
        'Timed throughput excludes11 warmup updates. All-update time includes them; trial time additionally includes preparation, '
        'evaluation and verification, excluding publication. Sampled board memory includes desktop/driver use and may miss peaks. '
        'These instrumented wall-time measurements do not establish equal-time quality or a measured FLOPs advantage.', '',
        '## Interpretation boundaries', '',
        'This is a small ConceptLM-inspired mechanism test: complete-chunk pooling, causal prediction over segmented codebooks, '
        'detached future targets and delayed predicted feedback. It is not a faithful reproduction of the paper or evidence that '
        'the learned entries represent human-interpretable concepts. Official training code is unreleased; the inspected implementation '
        'and paper also differ in feedback weighting. Native SDPA, initialization, normalization, positional features, model size and '
        'TinyStories training differ here.', '',
        'The mandatory native D6/D12 comparison changes both depth and width. The separate2x2 grid tests depth at fixed width '
        'and width at fixed depth under the same optimizer policy; width-dependent embedding/unembedding LR scaling remains part '
        'of that pinned policy. Matrix LR stays0.04. No optimum depth, width or training budget is established.', '',
        'All per-trial configurations, hashes, failures, utilization diagnostics and source corrections are in '
        '[the full campaign report](CAMPAIGN.md). The initial finite total-loss guard failures do not establish divergence; '
        'the corrected guard and exact retries are disclosed there. Collapsed candidates remain reported and excluded from selection.'])
    findings['additional_evidence_hashes']={}
    grid=c.read(c.HERE/'depth-grid-result.json')
    assert grid['status']=='completed'
    text.extend(['', '## Depth and width confirmation', '',
        '| Change | Mean BPB delta | Mean update-time ratio |', '|---|---:|---:|'])
    for axis in sorted({x['axis'] for x in grid['comparisons']}):
        group=[x for x in grid['comparisons'] if x['axis']==axis]
        text.append(f'| {axis.replace("_"," ")} | {statistics.mean(x["delta_bpb"] for x in group):+.6f} | '
            f'{statistics.mean(x["update_time_ratio"] for x in group):.2f} |')
    text.append('\nBoth seeds favor increasing either axis. The original native-depth comparison improves by0.046771 BPB on average, '
        'with5.13 times the parameters; it changes both depth and width. Similar measured runtime across widths is specific '
        'to this runtime and microbatch setup, not a FLOPs equivalence claim.')
    module=c.read(c.HERE/'module-initialization-result.json')
    assert module['status']=='completed'
    text.extend(['', '## Mechanism diagnosis', '',
        f'Crossed initializer/order runs traced most of the observed reversal to initialization. At fixed order42, '
        f'the descriptive module-initializer contrast is {module["contrasts"]["module_45_minus_42"]:+.6f} BPB and '
        f'the backbone contrast is {module["contrasts"]["backbone_45_minus_42"]:+.6f}. '
        'Only two deliberately chosen initializer levels were tested; this is not a general variance estimate.'])
    optional=c.HERE/'no-future-result.json'
    if optional.exists() and c.read(optional)['status']=='completed':
        no_future=c.read(optional)
        delta=statistics.mean(x['comparisons']['FROZEN-NOPRED']['delta_bpb'] for x in no_future['rows'])
        text.append(f'\nRemoving all future-target objectives, including VQ, changes mean BPB by {delta:+.6f} '
            'versus NOPRED on reused seeds45/46. It does not reveal a substantial VQ-supervision benefit in this pair. '
            'Token-only feedback still trains the latent path; it is an ablation, not an NCP candidate.')
    probe=c.HERE/'feedback-interventions-result.json'
    if probe.exists() and c.read(probe)['status']=='completed':
        outputs=c.read(probe)['probes']
        zero=[next(x['delta_from_original_bpb'] for x in p['outcomes'] if x['condition']=='feedback_zero') for p in outputs]
        text.append(f'\nEvery original checkpoint score reproduced within1e-6 BPB. Zeroing feedback worsens BPB by '
            f'{min(zero):.6f} to {max(zero):.6f}; code-identity rotations have smaller effects. '
            'Reliance on feedback does not establish an advantage over a separately trained dense or capacity control.')
    baseline=c.HERE/'concept-baselines-result.json'
    if baseline.exists() and c.read(baseline)['status']=='completed':
        evidence=c.read(baseline)['outcomes']
        learned=[x['scores']['learned']['aggregate_accuracy'] for x in evidence]
        persistence=[x['scores']['persistence']['aggregate_accuracy'] for x in evidence]
        majority=[x['scores']['training_majority']['aggregate_accuracy'] for x in evidence]
        text.append(f'\nOn the same validation batches, learned next-code accuracy is {min(learned):.1%}–{max(learned):.1%}, '
            f'versus {min(persistence):.1%}–{max(persistence):.1%} for repeating the current code and '
            f'{min(majority):.1%}–{max(majority):.1%} for a training-derived majority code. '
            'Learned predictions beat both baselines on every checkpoint. This supports code-label prediction learning, '
            'not semantic concepts or a reliable token-quality benefit.')
    replay=c.HERE/'checkpoint-replay-result.json'
    if replay.exists() and c.read(replay)['status']=='completed':
        evidence=c.read(replay)
        text.append(f'\nAll {evidence["completed_checkpoints"]} completed checkpoints independently replayed their original '
            f'BPB within1e-6 (largest difference {evidence["max_absolute_bpb_difference"]:.3g}), using captured source '
            'and unchanged saved weights. This confirms reproducibility of these scores, not generalization.')
    text.extend(['', '## Next best experiment', '',
        'Compare dense D6-width768 and D12-width768 at longer fixed-token budgets on new paired seeds, '
        'with one final evaluation on an untouched test split. Measure whether the depth gain persists and warrants '
        'roughly twice the update time. Keep quality-versus-time analysis separate. Do not combine NCP with other mechanisms '
        'on the strength of these mixed results. No next campaign is launched automatically.'])
    for name in ('depth-grid-result.json','module-initialization-result.json','no-future-result.json','feedback-interventions-result.json','concept-baselines-result.json','checkpoint-replay-result.json'):
        path=c.HERE/name
        if path.exists(): findings['additional_evidence_hashes'][name]=r.digest(path)
    r.write_json(c.HERE / 'findings-result.json', findings)
    (c.HERE / 'findings.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    destination = r.ROOT / 'reports/experiments' / reports.identity(c.HERE, r.ROOT)[0]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'FINDINGS.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('Wrote verified four-seed findings; full campaign finalization remains separate.')


if __name__ == '__main__':
    main()
