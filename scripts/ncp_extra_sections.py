"""Artifact-only reporting for frozen-checkpoint and crossed-order diagnostics."""
import ncp_campaign as c


def gather(result):
    for name, key in [('feedback-interventions-result.json', 'feedback_interventions'),
                      ('crossed-order-result.json', 'initializer_order_diagnostic'),
                      ('crossed-order-entry-result.json', 'initializer_order_entry'),
                      ('module-initialization-result.json', 'module_initialization_diagnostic'),
                      ('module-initialization-entry-result.json', 'module_initialization_entry'),
                      ('module-initialization-budget-result.json', 'module_initialization_budget'),
                      ('no-future-result.json','no_future_diagnostic'),
                      ('no-future-budget-result.json','no_future_budget')]:
        path = c.HERE / name
        if path.exists():
            result[key] = c.read(path)


def append(text, result):
    if 'no_future_diagnostic' in result:
        evidence=result['no_future_diagnostic']
        text.extend(['', '## Token-only latent feedback: all future losses off', '', evidence['interpretation'], '',
            '| Seed | Token-only BPB | Control | Control BPB | Token-only minus control |',
            '|---:|---:|---|---:|---:|'])
        for row in evidence['rows']:
            for label,control in row['comparisons'].items():
                text.append(f'| {row["seed"]} | {row["bpb"]:.6f} | {label} | {control["bpb"]:.6f} | {control["delta_bpb"]:+.6f} |')
        text.append('\nNOPRED retains future-derived VQ learning; this condition removes it too. '
            'Actual token-loss and parameter-gradient checks show independence from future targets with all three weights zero. '
            'Target-code utilization remains descriptive and is not a validity gate for this token-only objective.')
        for failure in evidence['failures']:
            text.append('\nPreserved token-only failure: '+str(failure))
    if 'no_future_budget' in result:
        text.append('\nToken-only stage: '+result['no_future_budget']['status']+'. '+result['no_future_budget']['reason'])
    if 'module_initialization_diagnostic' in result:
        evidence=result['module_initialization_diagnostic']
        text.extend(['', '## Backbone versus NCP module initialization', '', evidence['interpretation'], '',
            '| Backbone seed | NCP module seed | Order seed | NCP BPB | Dense BPB | Delta | Collapsed | Reused |',
            '|---:|---:|---:|---:|---:|---:|---|---|'])
        for cell in evidence['cells']:
            text.append(f'| {cell["backbone_seed"]} | {cell["ncp_initialization_seed"]} | {cell["batch_order_seed"]} | '
                f'{cell["bpb"]:.6f} | {cell["dense_bpb"]:.6f} | {cell["delta_bpb"]:+.6f} | {cell["collapsed"]} | {cell["reused"]} |')
        if evidence['contrasts']:
            text.extend(['', '| Descriptive contrast in NCP minus dense | BPB |', '|---|---:|'])
            for key,value in evidence['contrasts'].items():
                text.append(f'| {key.replace("_", " ")} | {value:+.6f} |')
        text.append('\nOnly the explicit NCP initializer changes before optimizer construction. Backbone parameters and '
            'batch order match their controls; NCP parameters and the frozen codebook basis match their module-seed anchors. '
            'These cells cannot reselect the candidate or count as independent confirmation seeds.')
    for key in ('module_initialization_entry','module_initialization_budget'):
        if key in result:
            entry=result[key]
            text.append('\nModule-initialization stage: '+entry['status']+'. '+entry.get('reason',''))
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
