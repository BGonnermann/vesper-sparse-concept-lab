"""Verify and publish local compact evidence for the inference-only diagnosis."""
import json
from pathlib import Path

import experiment_reports as reports
from ngram_diagnose import ROOT, HERE, STUDY, read, sha, write


def main():
    labels = ['DG-retry1', 'MG', 'M-old', 'M-new', 'M-new-old-source', 'M-old-new-source']
    analyses = {label: read(HERE / label / 'analysis.json') for label in labels}
    validation_hashes = set()
    for label, analysis in analyses.items():
        assert read(HERE / label / 'result.json')['status'] == 'completed'
        assert analysis['reproduction_passed'] and analysis['training_updates'] == 0
        enabled = analysis['evaluations']['final_enabled']['bpb']
        assert abs(enabled - analysis['recorded_bpb']) <= analysis['absolute_tolerance'] == 0.000002
        assert analysis['evaluations']['final_enabled_repeat']['bpb'] == enabled
        for evaluation in analysis['evaluations'].values():
            assert evaluation['parameter_immutability_verified']
            assert evaluation['validation_tokens'] == 65536
            validation_hashes.add(evaluation['validation_tape_sha256'])
    assert len(validation_hashes) == 1
    for own, cross in [('M-old', 'M-old-new-source'), ('M-new', 'M-new-old-source')]:
        assert analyses[own]['checkpoint_sha256'] == analyses[cross]['checkpoint_sha256']
        assert analyses[own]['evaluations']['final_enabled']['bpb'] == analyses[cross]['evaluations']['final_enabled']['bpb']
    batches = [read(STUDY / label / 'batches.json') for label in ['D', 'DG', 'M', 'MG']]
    assert all(batch == batches[0] for batch in batches)
    assert sorted(batches[0]['consumed_indices']) == list(range(8192))
    assert batches[0]['training_tokens'] == 8388608
    collision = read(HERE / 'collisions.json')
    assert collision['training_tape_sha256'] == batches[0]['batch_tape_sha256']
    prior = read(HERE / 'prior-artifacts.json')
    changed = [name for name, digest in prior.items() if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    write(HERE / 'preservation.json', {'checked_files': len(prior), 'changed_or_missing': changed})
    assert not changed, 'Prior artifacts changed; stop publication'
    rows = []
    for label, probe in [('DG', 'DG-retry1'), ('MG', 'MG')]:
        ev = analyses[probe]['evaluations']
        rows.append({'condition': label, 'enabled_bpb': ev['final_enabled']['bpb'],
                     'disabled_bpb': ev['final_residual_disabled']['bpb'],
                     'disabled_minus_enabled_bpb': ev['final_residual_disabled']['bpb'] - ev['final_enabled']['bpb']})
    reconciliation = read(HERE / 'reconciliation.json')
    configuration = read(HERE / 'DG-retry1' / 'diagnostic-config.json')
    result = {'kind': 'inference_diagnostic_comparison', 'status': 'completed', 'seed': 42,
              'rows': rows, 'training': {'new_updates': 0, 'new_training_tokens': 0},
              'settings': {'evaluation_tokens': 65536, 'batch_size': 2, 'sequence_length': 512,
                           'activation_checkpointing': False, 'window_pattern': 'L',
                           'precision': 'BF16 AMP, FP32 gate, high matmul precision',
                           'validation_tape_sha256': next(iter(validation_hashes)),
                           'absolute_bpb_tolerance': 0.000002},
              'data_hashes': configuration['data_seal'],
              'source_hashes': {k: v['executed_source_hashes'] for k, v in analyses.items()},
              'checkpoint_hashes': {k: v['checkpoint_sha256'] for k, v in analyses.items()},
              'interpretation': 'Inference ablation of jointly trained models; not independently trained controls. Preliminary single-seed evidence.'}
    write(HERE / 'result.json', result)
    write(HERE / 'analysis.json', {'comparison': result, 'collisions': collision,
                                 'reconciliation': reconciliation,
                                 'checks': {'enabled_reproduction': True, 'repeat_bpb_exact': True,
                                            'cross_source_bpb_exact': True, 'fixed_validation_hash': True,
                                            'complete_training_tape_permutation': True,
                                            'prior_artifacts_unchanged': len(prior)}})
    lines = ['# N-gram checkpoint diagnosis', '',
             'Outcome: memory has a small positive inference contribution in both jointly trained models. '
             'The seed-42 M training discrepancy remains unexplained. No new training, optimizer, backward pass, tuning, or cloud job was run.', '',
             '## Fixed-validation inference ablation', '',
             '| Model | Recorded BPB | Enabled BPB | Residual disabled BPB | Disabled - enabled |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        label = row['condition']; probe = 'DG-retry1' if label == 'DG' else label
        lines.append(f"| {label} | {analyses[probe]['recorded_bpb']:.6f} | {row['enabled_bpb']:.9f} | {row['disabled_bpb']:.9f} | {row['disabled_minus_enabled_bpb']:+.9f} |")
    lines += ['', 'Both enabled evaluations reproduce their recorded values within the predeclared absolute tolerance '
              '0.000002 BPB (six-decimal recording plus small FP32 reduction variation). Repeated, unhooked evaluations '
              'give exactly the same aggregate BPB. All six successful probes use the same 65,536-token validation '
              'tape, batch size 2, context 512, full attention, BF16 AMP, checkpointing off, and unchanged captured byte-normalized cross-entropy. '
              'Routing auxiliary loss is excluded. Model parameter hashes are unchanged by every evaluation.', '',
              'Residual removal is an **inference ablation of a jointly trained model**, not a replacement for a separately '
              'trained control. The hook computes the existing memory branch, then returns its input unchanged. These '
              'instrumented evaluations do not measure inference throughput. Previous training comparisons remain '
              'DG-D = -0.000670 and MG-M = +0.000506 BPB; they measure a different quantity. Small ablation deltas '
              'do not prove the memory layer caused the complete between-training difference.', '',
              '## Gate and residual measurements', '',
              'Memory exists only after zero-based block 1. Initialization below is reconstructed using the captured '
              'seeds and verified against every original initialization parameter hash; it is not a final-checkpoint '
              'measurement. Final measurements load the original pre-evaluation checkpoints. All distributions use '
              'the fixed validation tape, not the earlier 32-training-microbatch diagnostic sample.', '',
              '| Condition / state | Gate mean (all / valid suffix) | Gate p1 / median / p99 (all) | Hidden RMS | Gated residual RMS | Residual / hidden RMS | Effective addition / hidden RMS |',
              '|---|---|---|---:|---:|---:|---:|']
    for label, probe in [('DG', 'DG-retry1'), ('MG', 'MG')]:
        for state in ['initial_enabled', 'final_enabled']:
            m = analyses[probe]['evaluations'][state]['memory']; q = m['gate']['quantiles_0_1_25_50_75_99_100']
            lines.append(f"| {label} / {state.split('_')[0]} | {m['gate']['mean']:.9f} / {m['gate_valid_suffix']['mean']:.9f} | {q[1]:.9f} / {q[3]:.9f} / {q[5]:.9f} | {m['hidden_rms']['rms']:.6f} | {m['residual_rms']['rms']:.6f} | {100*m['global_residual_to_hidden_rms_ratio']:.4f}% | {100*m['global_effective_to_hidden_rms_ratio']:.4f}% |")
    lines += ['', 'Each gate distribution covers 65,536 positions; 64,985 (99.1592%) have at least one valid suffix. '
              'Invalid suffixes contribute zero lookup residual regardless of their gate. Initialization gates are '
              'constant sigmoid(-2) = 0.119202919; tiny mean-versus-extrema discrepancies are FP32 aggregation rounding.', '',
              '| Final condition | Gate min / max (all) | Gate p1 / median / p99 (valid suffix) | Token residual/hidden p1 / median / p99 / max |',
              '|---|---|---|---|']
    for label, probe in [('DG', 'DG-retry1'), ('MG', 'MG')]:
        m = analyses[probe]['evaluations']['final_enabled']['memory']
        q = m['gate_valid_suffix']['quantiles_0_1_25_50_75_99_100']
        r = m['residual_to_hidden_ratio']['quantiles_0_1_25_50_75_99_100']
        lines.append(f"| {label} | {m['gate']['min']:.6f} / {m['gate']['max']:.6f} | {q[1]:.6f} / {q[3]:.6f} / {q[5]:.6f} | {100*r[1]:.4f}% / {100*r[3]:.4f}% / {100*r[5]:.4f}% / {100*r[6]:.4f}% |")
    lines += ['', 'The gate is nearly saturated open, but this does not make the memory branch dominant: its global '
              'RMS is under 1% of the incoming backbone RMS. Residual RMS includes gating and precision casting; '
              'effective addition RMS also includes rounding when the residual is added to hidden states. These '
              'norms do not determine importance to logits or gradient interference. The positive inference ablations '
              'establish a small net contribution on this validation slice, not broad generalization.', '',
              '### Backbone block-output RMS by layer', '',
              '| Condition / state | Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for label, probe in [('DG', 'DG-retry1'), ('MG', 'MG')]:
        for state in ['initial_enabled', 'final_enabled']:
            layer = analyses[probe]['evaluations'][state]['backbone_layer_output_rms']
            lines.append('| ' + label + ' / ' + state.split('_')[0] + ' | ' + ' | '.join(f"{layer[str(i)]['rms']:.6f}" for i in range(6)) + ' |')
    lines += ['', 'Block 1 is measured immediately **before** memory insertion; later blocks include downstream effects. '
              'There are no separate gates or memory tables at other layers. Full per-token quantiles and valid-only '
              'gate distributions are retained in each probe analysis.', '',
              '## Training collision exposure', '',
              'Counted the verified shared training tape once: all four training runs consumed the same permutation '
              'of all 8,192 microbatches (8,388,608 input tokens). Keys use only row-local causal input suffixes; '
              'BOS-containing and incomplete suffixes are excluded, exactly as in captured memory code. Order-2 '
              'and order-3 tables are separate, each with 8,192 addresses.', '',
              '| Order | Valid occurrences | Distinct suffixes | Occupied / colliding buckets | Any-collision exposure | Non-dominant occurrence fraction | Competing bucket mass |',
              '|---|---:|---:|---|---:|---:|---:|']
    for order, row in collision['orders'].items():
        lines.append(f"| {order} | {row['valid_occurrences']:,} | {row['distinct_ngrams']:,} | {row['occupied_buckets']:,} / {row['buckets_with_multiple_ngrams']:,} | {100*row['frequency_weighted_any_collision_exposure']:.2f}% | {100*row['non_dominant_occurrence_fraction']:.4f}% | {100*row['frequency_weighted_other_ngram_bucket_mass']:.4f}% |")
    lines += ['', 'Let f_i be suffix frequency, F_b total frequency in bucket b, and N total valid occurrences.', '',
              '- Any-collision exposure: sum_i f_i * 1[bucket has another distinct suffix] / N.',
              '- Non-dominant fraction: 1 - sum_b max_i(f_i in b) / N. Tied dominant choices have equal mass.',
              '- Competing bucket mass: sum_i f_i * (1 - f_i/F_bucket(i)) / N; frequency-weighted probability of a different suffix conditional on sampling within the accessed bucket.',
              '- Independent global-pair probability, with replacement: (sum_b F_b^2 - sum_i f_i^2) / N^2; 0.000122328 for order 2 and 0.000119102 for order 3.', '',
              'Distinct suffixes per bucket have min/median/max 20/44/68 (order 2) and 146/194/251 (order 3). '
              'This measures extensive address sharing, especially for trigrams. It does **not** measure conflicting '
              'gradients, lost useful information, or the quality impact of a larger table. Frequent suffixes dominate '
              'some buckets, and learned shared representations can be useful. Full occupancy alone does not establish '
              'that either table is too small.', '',
              '## Reconciling the two seed-42 M controls', '',
              '| Checkpoint | Earlier captured source BPB | New captured source BPB |',
              '|---|---:|---:|',
              f"| Earlier equal-token M | {analyses['M-old']['evaluations']['final_enabled']['bpb']:.9f} | {analyses['M-old-new-source']['evaluations']['final_enabled']['bpb']:.9f} |",
              f"| New n-gram-study M | {analyses['M-new-old-source']['evaluations']['final_enabled']['bpb']:.9f} | {analyses['M-new']['evaluations']['final_enabled']['bpb']:.9f} |", '',
              'Cross-source evaluations reproduce each checkpoint exactly at the aggregate BPB level, including '
              'repeat evaluations. This rules out a measured aggregate BPB change from these forward/evaluation '
              'source swaps on this tape; it does not establish bitwise equality of every intermediate or backward pass.', '',
              '- All 88 initialization tensor hashes match, including experts and routers.',
              '- Candidate, protocol, complete batch order/hash chain, step schedule and optimizer-group receipts match exactly. Dataset/tokenizer seals and actual tape hash match.',
              '- Both runs consumed 512 updates / 8,388,608 tokens, including warmup; step-driven warmup/decay, BF16 AMP/Muon, checkpointing off and full attention match.',
              '- Captured upstream training/data source and bootstrap match. Project model/adapter/runner changed for optional memory; this M has no memory. No intentional no-memory tensor-math change was identified.',
              '- Printed last-microbatch losses first differ at update 17: new 3.994706 versus old 3.994785. The first 16 match only at six printed decimals; all subsequent 496 differ.',
              '- All 88 final checkpoint tensors differ. The -0.001666 recorded BPB difference reflects distinct training trajectories, not just validation formatting.',
              '- Historical deterministic-algorithm flags, selected kernels/driver details and GPU load were incompletely recorded. Current repeated inference is stable; it cannot establish historical training determinism.', '',
              '**Cause remains unexplained.** Numerical reduction/order sensitivity is plausible, not proven. '
              'Matching seeds and initialization are not a determinism guarantee. The shift exceeds either original '
              'memory-versus-control BPB delta, so single-seed architectural conclusions remain especially tentative.', '',
              '## One recommended next experiment (not implemented)', '',
              'Run a bounded M reproducibility replay before changing memory capacity: two fresh-process repeats '
              'for each historical captured source, seed 42, identical initialization and the first 32 optimizer '
              'updates of the existing tape/schedule. Record exact per-step parameter/gradient hashes and selected '
              'kernel/environment/determinism settings, with deterministic algorithms enforced and unsupported '
              'operations treated as a recorded failure, not silently bypassed. This four-short-run study would '
              'test within-source repeatability and locate cross-source divergence near update 17. It would not '
              'retroactively prove the original cause. No such replay has been started.', '',
              '## Provenance, repair and preservation', '',
              'The first DG probe failed enabled reproduction by 0.002079311 BPB because the diagnostic constructor '
              "omitted the captured adapter's WINDOW_PATTERN='L' override. That failed probe, its source snapshot, "
              'log and partial measurements are preserved and excluded from conclusions. DG-retry1 restores '
              'the original attention setting; no checkpoint, trained model code, data or tolerance was changed.', '',
              'Each successful process used Python isolated mode with captured project/upstream modules, verified '
              'source/config hashes, checkpoint hashes, initialization hashes and sealed data. Diagnostic scripts '
              'are separately captured and hashed; the new publishing commit is not claimed as the source used '
              'for historical training. Detailed source comparisons and checkpoint identities are in analysis.json '
              'and the child receipts.', '',
              f'Fresh preservation verification: {len(prior):,} pre-existing artifact files match their starting SHA-256 hashes. '
              'Local checkpoints, datasets, logs and large artifacts remain outside normal Git. Only compact '
              'reports and diagnostic code are published. Measurement overhead is outside training, and no '
              'throughput or VRAM improvement is claimed.', '', '## Probe receipts', '']
    for label in ['DG'] + labels:
        identity, _ = reports.identity(HERE / label, ROOT)
        lines.append(f'- [{label}](../{identity}/README.md)')
    body = '\n'.join(lines) + '\n'
    (HERE / 'diagnostic-report.md').write_text(body, encoding='utf-8')
    parent, _ = reports.identity(HERE, ROOT)
    reports.emit(HERE)
    (ROOT / 'reports/experiments' / parent / 'diagnostic-report.md').write_text(body, encoding='utf-8')
    for label in ['DG'] + labels:
        reports.emit(HERE / label)
    reports.index()
    print(json.dumps({'report_id': parent, 'preserved_artifacts': len(prior), 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
