"""Frozen initializer/order factorial diagnosis; four new runs, no selection."""
import copy
import hashlib
from pathlib import Path
import sys

import autoresearch as r
import ncp_campaign as c
import ncp_search as search
from autoresearch_train import batch_order, fixed_schedule

LEVELS = (42, 45)
CONDITIONS = ('D6', 'NCP')
OFF_DIAGONALS = ((42, 45), (45, 42))
RESERVE_SECONDS = 5400
# Each child already has a 900-second hard timeout. Allow additional controller work.
RUN_FORECAST_SECONDS = 930
DRIVER_SOURCE = Path(__file__).read_bytes()


def without_seeds(value):
    return {k: v for k, v in value.items() if k not in ('seed', 'batch_order_seed')}


def payload(value):
    """Batch identity excludes only labels for the two independent RNG seeds."""
    return without_seeds(value)


def identity(directory):
    record = c.read(directory / 'result.json')
    names = ('result.json', 'candidate.json', 'protocol.json', 'snapshot.json',
             'execution.json', 'checkpoint_pre_eval.pt', 'initialization.json',
             'batches.json', 'schedule.json', 'model.json', 'fixed-training.json',
             'training.json', 'evaluation-immutability.json')
    return dict(trial=directory.name, files={name: r.digest(directory / name) for name in names},
                source_hashes=record['snapshot_files'])


def verify_artifacts(directory):
    record = c.read(directory / 'result.json')
    assert record['status'] == 'completed'
    r.validate_execution(directory, record['snapshot_files'])
    artifacts = r.validate_run_artifacts(directory, record)
    assert r.digest(directory / 'checkpoint_pre_eval.pt') == record['checkpoint_sha256']
    assert c.read(directory / 'candidate.json') == record['candidate']
    assert c.read(directory / 'protocol.json') == record['protocol']
    training = artifacts['training']
    assert training['optimizer_updates'] == 512 and training['training_tokens'] == 8388608
    batches = c.read(directory / 'batches.json')
    order_seed = record['protocol'].get('batch_order_seed', record['protocol']['seed'])
    assert record['seed'] == record['protocol']['seed']
    assert batches['training_tokens'] == 8388608
    assert batches['consumed_indices'] == batch_order(8192, order_seed)
    assert c.read(directory / 'schedule.json')['updates'] == fixed_schedule()
    immutability = c.read(directory / 'evaluation-immutability.json')
    assert immutability['verified'] and immutability['before_sha256'] == immutability['after_sha256']
    return record


def diagonal_path(plan, condition, seed):
    item = plan['diagonals'][f'{condition}:{seed}']
    directory = c.HERE / item['trial']
    assert identity(directory) == item, ('Reused diagonal changed', directory)
    return directory


def freeze():
    path = c.HERE / 'crossed-order-plan.json'
    if path.exists():
        plan = c.read(path)
        assert hashlib.sha256(DRIVER_SOURCE).hexdigest() == plan['driver_sha256']
        assert (c.HERE / 'crossed-order-driver.py').read_bytes() == DRIVER_SOURCE
        assert r.digest(c.HERE / 'confirmation-selection.json') == plan['frozen_selection_sha256']
        for condition in CONDITIONS:
            for seed in LEVELS:
                diagonal_path(plan, condition, seed)
        return plan
    from ncp_frozen_pairs import collect
    frozen = c.read(c.HERE / 'confirmation-selection.json')
    assert all(x['completed_pairs'] == 4 for x in collect(frozen)['summaries'].values()), 'Complete mandatory confirmation first'
    for name, value in frozen['confirmation_source_hashes'].items():
        if name not in ('autoresearch.py', 'autoresearch_train.py'):
            assert c.sources()[name] == value, ('Mechanism source changed during order diagnosis', name)
    candidates = {'D6': c.candidate('D6'), 'NCP': copy.deepcopy(frozen['variants'][frozen['label']])}
    diagonals = {}
    for condition in CONDITIONS:
        for seed in LEVELS:
            if condition == 'NCP' and seed == 42:
                directory = c.HERE / frozen['selection_trial']
            else:
                label = 'D6' if condition == 'D6' else frozen['label']
                matches = [p.parent for p in c.HERE.glob('trial-*/result.json')
                           if (c.read(p)['label'], c.read(p)['seed']) == (label, seed)]
                assert len(matches) == 1, ('Ambiguous diagonal', condition, seed)
                directory = matches[0]
            record = verify_artifacts(directory)
            assert record['candidate'] == candidates[condition]
            assert record['protocol']['seed'] == record['protocol'].get('batch_order_seed', seed) == seed
            assert without_seeds(record['protocol']) == without_seeds(frozen['confirmation_protocol'])
            assert record['data_seal'] == frozen['confirmation_data_seal']
            assert record['upstream'] == frozen['confirmation_upstream']
            diagonals[f'{condition}:{seed}'] = identity(directory)
    protocol = copy.deepcopy(frozen['confirmation_protocol'])
    for init, order in OFF_DIAGONALS:
        r.validate_protocol(dict(protocol, seed=init, batch_order_seed=order))
    hashes = {}
    for condition, candidate in candidates.items():
        file = c.HERE / 'candidates' / f'CROSS-{condition}.json'
        if file.exists():
            assert c.read(file) == candidate
        else:
            r.write_json(file, candidate)
        hashes[condition] = r.digest(file)
    archive = c.HERE / 'crossed-order-driver.py'
    if archive.exists():
        assert archive.read_bytes() == DRIVER_SOURCE
    else:
        archive.write_bytes(DRIVER_SOURCE)
    plan = dict(kind='initializer_order_diagnostic_plan', initialization_seeds=list(LEVELS),
                batch_order_seeds=list(LEVELS), off_diagonals=[list(x) for x in OFF_DIAGONALS],
                candidates=candidates, candidate_file_hashes=hashes, diagonals=diagonals,
                protocol_template=protocol, source_hashes=c.sources(), upstream=r.verify_runtime(),
                data_seal=c.read(r.ROOT / '.autoresearch/data-seal.json'),
                frozen_selection_sha256=r.digest(c.HERE / 'confirmation-selection.json'),
                driver_sha256=r.digest(archive), frozen_at=c.datetime.now(c.timezone.utc).isoformat(),
                per_run_forecast_seconds=RUN_FORECAST_SECONDS, report_reserve_seconds=RESERVE_SECONDS,
                hypothesis='Separate initialization and training-order sensitivity of the already frozen NCP-minus-dense difference; no reselection',
                source_qualification='Diagonal runs retain their original executed sources. New adapter sources introduce an optional order seed; omission must pass historical-behavior gates.',
                interpretation='Two deliberately chosen seed levels diagnose the observed reversal; not four independent replications or held-out evidence. Utilization samples follow order seed.')
    assert plan['data_seal'] == frozen['confirmation_data_seal']
    assert plan['upstream'] == frozen['confirmation_upstream']
    r.write_json(path, plan)
    return plan


def existing(condition, init, order):
    matches = []
    for path in c.HERE.glob('trial-*/result.json'):
        record = c.read(path)
        if (record['label'], record['seed']) != (f'CROSS-{condition}', init):
            continue
        assert record.get('protocol', {}).get('batch_order_seed') == order, ('Crossed label/order collision', path)
        matches.append(path.parent)
    assert len(matches) <= 1, ('Repeated diagnostic cell', condition, init, order)
    return matches[0] if matches else None


def validate(directory, plan, condition, init, order):
    record = verify_artifacts(directory)
    assert (record['label'], record['seed'], record['phase']) == (f'CROSS-{condition}', init, 'initializer_order_diagnostic')
    assert record['candidate'] == plan['candidates'][condition]
    assert r.digest(directory / 'candidate.json') == plan['candidate_file_hashes'][condition]
    assert record['protocol'] == dict(plan['protocol_template'], seed=init, batch_order_seed=order)
    assert record['data_seal'] == plan['data_seal'] and record['upstream'] == plan['upstream']
    for name, value in plan['source_hashes'].items():
        assert record['snapshot_files']['source/project/' + name] == value
    initial_diagonal = diagonal_path(plan, condition, init)
    order_diagonal = diagonal_path(plan, condition, order)
    assert c.read(directory / 'initialization.json')['parameters'] == c.read(initial_diagonal / 'initialization.json')['parameters']
    assert c.read(directory / 'model.json')['total_parameters'] == c.read(initial_diagonal / 'model.json')['total_parameters']
    assert payload(c.read(directory / 'batches.json')) == payload(c.read(order_diagonal / 'batches.json'))
    assert c.read(directory / 'schedule.json') == c.read(initial_diagonal / 'schedule.json')
    return record


def summary(plan):
    cells = []
    for init in LEVELS:
        for order in LEVELS:
            runs = {}
            for condition in CONDITIONS:
                directory = diagonal_path(plan, condition, init) if init == order else existing(condition, init, order)
                if directory is None:
                    continue
                record = verify_artifacts(directory) if init == order else validate(directory, plan, condition, init, order)
                runs[condition] = dict(trial=directory.name, result_sha256=r.digest(directory / 'result.json'),
                                       bpb=record['metrics']['val_bpb'])
            if len(runs) != 2:
                continue
            dense = c.HERE / runs['D6']['trial']; ncp = c.HERE / runs['NCP']['trial']
            a = c.read(dense / 'initialization.json')['parameters']
            b = c.read(ncp / 'initialization.json')['parameters']
            assert a == {k: v for k, v in b.items() if not k.startswith('ncp.')}
            assert payload(c.read(dense / 'batches.json')) == payload(c.read(ncp / 'batches.json'))
            cells.append(dict(initialization_seed=init, batch_order_seed=order, runs=runs,
                              delta_bpb=runs['NCP']['bpb'] - runs['D6']['bpb']))
    contrasts = None
    if len(cells) == 4:
        d = {(x['initialization_seed'], x['batch_order_seed']): x['delta_bpb'] for x in cells}
        contrasts = dict(initialization_45_minus_42=((d[45,42]+d[45,45])-(d[42,42]+d[42,45]))/2,
                         order_45_minus_42=((d[42,45]+d[45,45])-(d[42,42]+d[45,42]))/2,
                         interaction_difference_of_differences=d[45,45]-d[45,42]-d[42,45]+d[42,42])
    return dict(kind='initializer_order_diagnostic', status='completed' if len(cells)==4 else 'running',
                cells=cells, contrasts=contrasts, expected_cells=4, new_training_runs=4,
                contrast_definition='Contrasts operate on NCP-minus-dense BPB; positive means a less favorable NCP effect.',
                interpretation=plan['interpretation'], plan_sha256=r.digest(c.HERE / 'crossed-order-plan.json'))


def main(publish=False):
    plan = freeze()
    pending = []
    for init, order in OFF_DIAGONALS:
        for condition in CONDITIONS:
            directory = existing(condition, init, order)
            if directory is None:
                pending.append((condition, init, order))
            else:
                validate(directory, plan, condition, init, order)
    forecast = len(pending) * plan['per_run_forecast_seconds']
    if forecast > c.remaining() - RESERVE_SECONDS:
        r.write_json(c.HERE / 'crossed-order-entry-result.json', dict(kind='crossed_order_entry', status='not_entered',
                     reason='All remaining off-diagonal runs must fit before the 90-minute reserve', forecast_seconds=forecast))
        c.log('Crossed-order diagnostic deferred by whole-stage time gate')
        return
    for condition, init, order in pending:
        assert c.sources() == plan['source_hashes'] and r.verify_runtime() == plan['upstream']
        assert c.read(r.ROOT / '.autoresearch/data-seal.json') == plan['data_seal']
        live = c.read(r.ROOT / 'runs/autoresearch/equal-token-20260915/protocol-42.json')
        assert without_seeds(live) == without_seeds(plan['protocol_template'])
        assert r.digest(c.HERE / 'candidates' / f'CROSS-{condition}.json') == plan['candidate_file_hashes'][condition]
        assert c.remaining() > RESERVE_SECONDS + plan['per_run_forecast_seconds'], 'Preserve report reserve'
        result = c.trial(f'CROSS-{condition}', init, plan['hypothesis'] + f'; initializer {init}, batch order {order}',
                         'initializer_order_diagnostic', 'CROSS-D6', batch_order_seed=order)
        if result is None:
            return
        assert result['status'] == 'completed', 'Preserved diagnostic failure requires diagnosis; no automatic repeat'
        validate(existing(condition, init, order), plan, condition, init, order)
        r.write_json(c.HERE / 'crossed-order-result.json', summary(plan))
        from ncp_report import write
        write()
        if publish:
            try:
                search.publish()
            except Exception as exc:
                c.log('Publication deferred; crossed-order result retained: ' + repr(exc))
    r.write_json(c.HERE / 'crossed-order-result.json', summary(plan))


if __name__ == '__main__':
    if '--freeze-only' in sys.argv:
        freeze()
    else:
        main('--publish' in sys.argv)
