"""Two frozen latent-capacity diagnostics without future-target loss gradients."""
import copy
from pathlib import Path
import sys

import autoresearch as r
import ncp_campaign as c
import ncp_crossed_order as crossed
from ncp_frozen_pairs import collect
from ncp_module_seeds import basis_hash
import ncp_search as search

LABEL = 'FROZEN-NOFUTURE'
PHASE = 'future_supervision_diagnostic'
SEEDS = (45, 46)
RESERVE = 4500
FORECAST = 930
DRIVER_SOURCE = Path(__file__).read_bytes()


def existing(seed):
    found = [p.parent for p in c.HERE.glob('trial-*/result.json')
             if (c.read(p)['label'], c.read(p)['seed']) == (LABEL, seed)]
    assert len(found) <= 1, 'Ambiguous no-future trial'
    return found[0] if found else None


def anchor(plan, label, seed):
    identity = plan['anchors'][f'{label}:{seed}']
    directory = c.HERE / identity['trial']
    assert crossed.identity(directory) == identity
    crossed.verify_artifacts(directory)
    return directory


def freeze():
    path = c.HERE / 'no-future-plan.json'
    archive = c.HERE / 'no-future-driver.py'
    frozen_path = c.HERE / 'confirmation-selection.json'
    if path.exists():
        plan = c.read(path)
        assert archive.read_bytes() == DRIVER_SOURCE
        assert r.digest(archive) == plan['driver_sha256']
        assert r.digest(frozen_path) == plan['frozen_selection_sha256']
        return plan
    frozen = c.read(frozen_path)
    pairs = collect(frozen)
    assert all(x['completed_pairs'] == 4 for x in pairs['summaries'].values())
    candidate = copy.deepcopy(frozen['variants'][frozen['label']])
    for key in ('prediction_weight', 'ce_weight', 'vq_weight'):
        candidate['ncp'][key] = 0.0
    assert candidate['ncp']['mode'] == 'feedback'
    candidate_path = c.HERE / 'candidates' / (LABEL + '.json')
    if candidate_path.exists():
        assert c.read(candidate_path) == candidate
    else:
        r.write_json(candidate_path, candidate)
    anchors = {}
    bases = {}
    for seed in SEEDS:
        rows = [x for x in pairs['pairs'] if x['seed'] == seed]
        for label in ('D6', 'FROZEN-NOPRED', 'FROZEN-CAP'):
            pair = next(x for x in rows if x['control'] == label)
            anchors[f'{label}:{seed}'] = crossed.identity(c.HERE / pair['control_trial'])
        ncp = c.HERE / rows[0]['candidate_trial']
        anchors[f'NCP:{seed}'] = crossed.identity(ncp)
        bases[str(seed)] = basis_hash(ncp)
    for name, value in frozen['confirmation_source_hashes'].items():
        if name not in ('autoresearch.py', 'autoresearch_train.py'):
            assert c.sources()[name] == value, ('Mechanism changed', name)
    assert not archive.exists()
    archive.write_bytes(DRIVER_SOURCE)
    plan = dict(kind='no_future_plan', label=LABEL, phase=PHASE, seeds=list(SEEDS),
        candidate=candidate, candidate_sha256=r.digest(candidate_path), anchors=anchors,
        anchor_basis_hashes=bases, protocol=frozen['confirmation_protocol'],
        source_hashes=c.sources(), upstream=r.verify_runtime(),
        data_seal=c.read(r.ROOT / '.autoresearch/data-seal.json'),
        expected_parameters=frozen['expected_parameters'], driver_sha256=r.digest(archive),
        frozen_selection_sha256=r.digest(frozen_path),
        frozen_at=c.datetime.now(c.timezone.utc).isoformat(),
        per_run_forecast_seconds=FORECAST, report_reserve_seconds=RESERVE,
        hypothesis='Remove VQ supervision remaining in frozen NOPRED: all three future-target loss weights zero, predicted feedback and token CE retained; no reselection',
        interpretation='Same two previously observed confirmation seeds; mechanistic diagnostic, not new independent replication or held-out evidence. Future-target branches may execute, but their losses have zero training weight. Token CE can still train the predicted latent path and codebook transforms.')
    assert plan['upstream'] == frozen['confirmation_upstream']
    assert plan['data_seal'] == frozen['confirmation_data_seal']
    r.write_json(path, plan)
    return plan


def validate(directory, plan, seed):
    record = crossed.verify_artifacts(directory)
    assert (record['label'], record['phase'], record['seed']) == (LABEL, PHASE, seed)
    assert record['candidate'] == plan['candidate']
    assert r.digest(directory / 'candidate.json') == plan['candidate_sha256']
    assert record['protocol'] == dict(plan['protocol'], seed=seed)
    assert record['upstream'] == plan['upstream'] and record['data_seal'] == plan['data_seal']
    for name, value in plan['source_hashes'].items():
        assert record['snapshot_files']['source/project/' + name] == value
    ncp = anchor(plan, 'NCP', seed)
    for name in ('initialization.json', 'schedule.json', 'batches.json'):
        assert c.read(directory / name) == c.read(ncp / name), name
    assert basis_hash(directory) == plan['anchor_basis_hashes'][str(seed)]
    assert c.read(directory / 'model.json')['total_parameters'] == plan['expected_parameters']
    return record


def summary(plan):
    rows = []
    failures = []
    for seed in SEEDS:
        directory = existing(seed)
        if directory is None:
            continue
        record = c.read(directory / 'result.json')
        if record['status'] != 'completed':
            failures.append(dict(seed=seed, trial=directory.name, status=record['status'],
                                 result_sha256=r.digest(directory / 'result.json')))
            continue
        record = validate(directory, plan, seed)
        controls = {}
        for label in ('NCP', 'FROZEN-NOPRED', 'FROZEN-CAP', 'D6'):
            ref = anchor(plan, label, seed)
            other = c.read(ref / 'result.json')
            controls[label] = dict(trial=ref.name, bpb=other['metrics']['val_bpb'],
                delta_bpb=record['metrics']['val_bpb'] - other['metrics']['val_bpb'])
        rows.append(dict(seed=seed, trial=directory.name, bpb=record['metrics']['val_bpb'],
            result_sha256=r.digest(directory / 'result.json'), comparisons=controls,
            collapsed=record['ncp_health']['collapsed']))
    return dict(kind='no_future_diagnostic', status='failed' if failures else 'completed' if len(rows) == 2 else 'running',
        rows=rows, failures=failures, expected_seeds=list(SEEDS),
        plan_sha256=r.digest(c.HERE / 'no-future-plan.json'), interpretation=plan['interpretation'])


def main(publish=False):
    plan = freeze()
    pending = []
    for seed in SEEDS:
        directory = existing(seed)
        if directory is None:
            pending.append(seed)
        else:
            r.write_json(c.HERE / 'no-future-result.json', summary(plan))
            validate(directory, plan, seed)  # Preserve and stop on existing failures.
    if len(pending) * FORECAST > c.remaining() - RESERVE:
        r.write_json(c.HERE / 'no-future-budget-result.json', dict(kind='no_future_budget', status='skipped',
            pending_seeds=pending, reason='Whole remaining stage cannot fit before final75-minute reserve'))
        return
    for seed in pending:
        assert c.sources() == plan['source_hashes'] and r.verify_runtime() == plan['upstream']
        assert c.read(r.ROOT / '.autoresearch/data-seal.json') == plan['data_seal']
        assert c.read(r.ROOT / 'runs/autoresearch/equal-token-20260915/protocol-42.json') == plan['protocol']
        assert r.digest(c.HERE / 'candidates' / (LABEL + '.json')) == plan['candidate_sha256']
        assert c.remaining() > RESERVE + FORECAST
        result = c.trial(LABEL, seed, plan['hypothesis'], PHASE, 'D6')
        r.write_json(c.HERE / 'no-future-result.json', summary(plan))
        assert result and result['status'] == 'completed', 'Preserved no-future failure needs diagnosis'
        validate(existing(seed), plan, seed)
        from ncp_report import write
        write()
        if publish:
            try:
                search.publish()
            except Exception as exc:
                c.log('Publication deferred; no-future evidence retained: ' + repr(exc))
    r.write_json(c.HERE / 'no-future-result.json', summary(plan))


if __name__ == '__main__':
    if '--freeze-only' in sys.argv:
        freeze()
    else:
        main('--publish' in sys.argv)
