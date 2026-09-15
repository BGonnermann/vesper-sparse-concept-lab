"""Artifact-only reporting for frozen-checkpoint and crossed-order diagnostics."""
import ncp_campaign as c


def gather(result):
    for name, key in [('feedback-interventions-result.json', 'feedback_interventions'),
                      ('crossed-order-result.json', 'initializer_order_diagnostic'),
                      ('crossed-order-entry-result.json', 'initializer_order_entry')]:
        path = c.HERE / name
        if path.exists():
            result[key] = c.read(path)


def append(text, result):
    if 'feedback_interventions' in result:
        evidence = result['feedback_interventions']
        assert evidence['status'] == 'completed'
        text.extend(['', '## Frozen checkpoint inference interventions', '',
            'No retraining. Each original checkpoint score was reproduced within1e-6 BPB before changing feedback. '
            'All conditions consumed identical validation batches and target bytes. Trainable weights remained fixed; '
            'persistent state stayed immutable during each evaluation and was restored afterward. '
            'The deterministic row rotation changes which learned code each head entry refers to while preserving the code set.', '',
            '| Seed | Original BPB | Feedback zero BPB | Delta | Rotated codes BPB | Delta |',
            '|---:|---:|---:|---:|---:|---:|'])
        for probe in evidence['probes']:
            assert probe['status']=='completed' and probe['state_restored'] and probe['trainable_weights_unchanged']
            items = {x['condition']: x for x in probe['outcomes']}
            original, zero, rotated = (items[x] for x in ('original', 'feedback_zero', 'codebook_rows_rotated'))
            text.append(f'| {probe["seed"]} | {original["bpb"]:.6f} | {zero["bpb"]:.6f} | '
                f'{zero["delta_from_original_bpb"]:+.6f} | {rotated["bpb"]:.6f} | {rotated["delta_from_original_bpb"]:+.6f} |')
        text.append('\nPositive deltas indicate reliance on the trained feedback path or code identity. '
            'They do not establish semantic concepts or show that NCP improves over training a dense model. '
            'Instrumented probe timing includes CPU accounting and is not a throughput comparison.')
    if 'initializer_order_diagnostic' in result:
        evidence = result['initializer_order_diagnostic']
        text.extend(['', '## Initializer versus batch-order diagnosis', '', evidence['interpretation'], '',
            '| Initialization seed | Batch-order seed | Dense BPB | NCP BPB | NCP minus dense | Evidence |',
            '|---:|---:|---:|---:|---:|---|'])
        for cell in evidence['cells']:
            text.append(f'| {cell["initialization_seed"]} | {cell["batch_order_seed"]} | '
                f'{cell["runs"]["D6"]["bpb"]:.6f} | {cell["runs"]["NCP"]["bpb"]:.6f} | {cell["delta_bpb"]:+.6f} | '
                + ('Reused diagonal' if cell['initialization_seed']==cell['batch_order_seed'] else 'New off-diagonal pair') + ' |')
        if evidence['contrasts'] is not None:
            text.extend(['', evidence['contrast_definition'], '', '| Descriptive contrast | BPB |', '|---|---:|'])
            for key, value in evidence['contrasts'].items():
                text.append(f'| {key.replace("_", " ")} | {value:+.6f} |')
        text.append('\nThe optional order-seed field changes only the tape permutation; the architecture and objectives stay frozen. '
            'Actual-loop tests verify default compatibility, unchanged initialization when order changes, '
            'unchanged order when initialization changes, and complete tape coverage. '
            'New runs are excluded from selection and independent-seed confirmation summaries.')
    if 'initializer_order_entry' in result:
        entry = result['initializer_order_entry']
        text.append('\nCrossed-order entry: '+entry['status']+'. '+entry.get('reason',''))
