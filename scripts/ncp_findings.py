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
    r.write_json(c.HERE / 'findings-result.json', findings)
    (c.HERE / 'findings.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    destination = r.ROOT / 'reports/experiments' / reports.identity(c.HERE, r.ROOT)[0]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'FINDINGS.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print('Wrote verified four-seed findings; full campaign finalization remains separate.')


if __name__ == '__main__':
    main()
