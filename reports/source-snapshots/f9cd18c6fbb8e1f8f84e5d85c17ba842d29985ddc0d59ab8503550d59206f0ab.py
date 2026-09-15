"""Bounded adaptive equal-token campaign; no dependency management or cloud jobs."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import autoresearch as r
import experiment_reports as reports

ROOT = r.ROOT
HERE = ROOT / 'runs/autoresearch/overnight-20260915'
REFERENCE = ROOT / 'runs/autoresearch/equal-token-20260915'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def remaining():
    contract = read(HERE / 'contract.json')
    assert r.digest(HERE / 'frozen-plan-v2.md') == contract['plan_sha256']
    return (datetime.fromisoformat(contract['deadline']) - datetime.now(timezone.utc)).total_seconds()


def source_hashes():
    return {n: r.digest(ROOT / 'scripts' / n) for n in r.PROJECT_FILES}


def announce(path):
    log = path.open('x', encoding='utf-8')
    print('LOG READY; child not started: ' + str(path), flush=True)
    deadline = time.monotonic() + 180
    while not path.with_suffix('.start').exists():
        if time.monotonic() > deadline:
            log.close()
            raise TimeoutError('Log announcement expired; no child launched')
        time.sleep(.1)
    return log


def space():
    dense = read(ROOT / 'experiments/autoresearch/dense-depth6.json')
    moe = read(ROOT / 'experiments/autoresearch/moe-depth6.json')
    memory = read(ROOT / 'runs/autoresearch/ngram-20260915/DG.json')
    result = {}
    def add(name, config, control, hypothesis, family):
        result[name] = dict(candidate=r.validate_candidate(config), control=control,
                            hypothesis=hypothesis, family=family)
    add('D', dense, None, 'Fresh fast dense reference', 'control')
    add('M', moe, None, 'Fresh four-expert packed sparse reference', 'control')
    add('DG', memory, 'D', 'Fresh original memory reference against dense', 'control')
    for lr in [.02, .06, .01, .03, .05, .08]:
        c = copy.deepcopy(dense); c['matrix_lr'] = lr
        add('D-lr' + str(lr), c, 'D', f'Matrix LR {lr} may improve optimization at unchanged token budget', 'dense')
    for depth in [4, 8, 2, 3, 5, 7]:
        c = copy.deepcopy(dense); c['depth'] = depth
        add('D-depth' + str(depth), c, 'D', f'Depth {depth} and coupled width may improve the quality/time frontier', 'dense')
    c = copy.deepcopy(moe); c['num_experts'] = 2
    add('M-experts2', c, 'M', 'Fewer experts may reduce dispatch and all-expert optimizer cost', 'moe')
    for key, values in [('router_lr', [.0003, .003]), ('aux_loss_weight', [.003, .03])]:
        for value in values:
            c = copy.deepcopy(moe); c[key] = value
            add('M-' + key + str(value), c, 'M', f'Change only {key} to {value} to improve routing learning', 'moe')
    for layer in [3, 0, 2, 4, 5]:
        c = copy.deepcopy(memory); c['memory']['after_layer'] = layer
        add('DG-layer' + str(layer), c, 'DG', f'Memory after block {layer} may use a more effective hidden representation', 'memory')
    for lr in [.003, .0001, .0003, .01]:
        c = copy.deepcopy(memory); c['memory']['lr'] = lr
        add('DG-lr' + str(lr), c, 'DG', f'Memory-only AdamW LR {lr} may improve its short-run adaptation', 'memory')
    return result


def completed():
    records = []
    for path in sorted(HERE.glob('trial-*/result.json')):
        record = read(path)
        if record['status'] != 'completed':
            raise RuntimeError('Unresolved prior trial; stop and diagnose ' + str(path))
        record['_path'] = str(path.parent)
        records.append(record)
    return records


def aggregate(records, label):
    import statistics
    rows = [x for x in records if x['label'] == label and x['seed'] == 42]
    return dict(bpb=statistics.mean(x['metrics']['val_bpb'] for x in rows),
                seconds=statistics.mean(read(Path(x['_path']) / 'fixed-training.json')['all_update_seconds'] for x in rows),
                repeats=len(rows))


def rank(records, choices):
    available = {x['label'] for x in records if x['seed'] == 42}
    options = []
    for label in available:
        control = choices[label]['control']
        if control not in available:
            continue
        candidate, reference = aggregate(records, label), aggregate(records, control)
        delta = candidate['bpb'] - reference['bpb']
        ratio = candidate['seconds'] / reference['seconds']
        quality = delta < 0 and ratio <= 1.10
        faster = ratio <= .90 and delta <= .001
        options.append(dict(label=label, control=control, delta_bpb=delta, time_ratio=ratio,
                            qualifies=quality or faster, quality=quality,
                            observed_candidate=candidate, observed_control=reference))
    return sorted(options, key=lambda x: (not x['qualifies'], not x['quality'], x['delta_bpb'], x['time_ratio'], x['label']))


def select(records, choices):
    available = {x['label'] for x in records}
    for label in ['D', 'M', 'DG']:
        if label not in available:
            return label, 42, 'screen', 'Fresh matched controls precede candidate selection', None
    ranked = rank(records, choices)
    freeze = HERE / 'confirmation-selection.json'
    if remaining() <= 6300 or freeze.exists():
        if not freeze.exists():
            winner = ranked[0]
            r.write_json(freeze, dict(selected_at=datetime.now(timezone.utc).isoformat(),
                                     winner=winner, evidence=ranked,
                                     note='Frozen before confirmation seeds; if no option qualifies, exploratory replication only'))
        winner = read(freeze)['winner']
        plan = [(winner['control'], 43), (winner['label'], 43), (winner['control'], 44), (winner['label'], 44)]
        confirmations = [x for x in records if x['phase'] == 'confirmation']
        label, seed = plan[len(confirmations) % len(plan)]
        return label, seed, 'confirmation', 'Frozen candidate/control confirmation; repetitions beyond first pair per seed measure execution variability, not new seeds', winner
    pending = [label for label in choices if label not in available]
    if pending:
        d, m, dg = (aggregate(records, label) for label in ['D', 'M', 'DG'])
        favored = 'moe' if m['bpb'] < d['bpb'] - .001 and m['seconds'] <= 1.1*d['seconds'] else ('memory' if dg['bpb'] < d['bpb'] - .0005 and dg['seconds'] <= 1.1*d['seconds'] else 'dense')
        family_order = [favored] + [x for x in ['dense', 'moe', 'memory'] if x != favored]
        # Alternate LR direction and shape after observing each completed result.
        if favored == 'dense' and records[-1]['label'].startswith('D-lr'):
            shaped = [x for x in pending if x.startswith('D-depth')]
            if shaped:
                return shaped[0], 42, 'screen', 'After observing dense LR outcome, test the independent depth/width cost axis', dict(D=d, M=m, DG=dg, last_trial=records[-1]['label'])
        label = min(pending, key=lambda x: family_order.index(choices[x]['family']))
        return label, 42, 'screen', f'Observed control quality/time favors {favored}; select next one-factor alternative without combining winners', dict(D=d, M=m, DG=dg, ranking=ranked)
    # No count stop: acquire repeatability evidence on the strongest candidates.
    top = [x for x in ranked if x['qualifies']][:3] or ranked[:1]
    target = min(top, key=lambda x: aggregate(records, x['label'])['repeats'])
    label = target['control'] if aggregate(records, target['control'])['repeats'] < aggregate(records, target['label'])['repeats'] else target['label']
    return label, 42, 'replication', 'Search space screened; adaptive repeat of a promising tradeoff or its control to estimate execution noise', target


def preflight():
    assert remaining() > 6300
    r.write_json(HERE / 'search-space.json', space())
    r.verify_runtime(); r.verify_seal(ROOT / '.autoresearch/cache', read(ROOT / '.autoresearch/data-seal.json'))
    logpath = HERE / 'preflight.log'
    result = dict(kind='correctness_preflight', status='running', passed=False)
    try:
        with announce(logpath) as log:
            for device in ['cpu', 'cuda']:
                child = subprocess.run([str(r.runtime_python()), '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                    cwd=ROOT, env=dict(r.environment(), MOE_TEST_DEVICE=device), stdout=log, stderr=subprocess.STDOUT, timeout=240)
                log.write(f'\n{device} correctness exit: {child.returncode}\n'); log.flush()
                if child.returncode:
                    raise RuntimeError(device + ' correctness failed')
        result.update(status='completed', passed=True, source_hashes=source_hashes(),
                      test_hashes={p.name:r.digest(p) for p in (ROOT / 'tests').glob('test_*.py')},
                      log_sha256=r.digest(logpath), search_sha256=r.digest(HERE / 'search-space.json'))
    except BaseException as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        r.write_json(HERE / 'preflight-result.json', result)
        reports.emit(HERE); reports.index()
    print('ALL CPU/CUDA CORRECTNESS GATES PASSED', flush=True)


def trial():
    if remaining() < 2700:
        print('REPORT RESERVE REACHED; no further trial started', flush=True)
        return
    gate = read(HERE / 'preflight-result.json')
    assert gate['passed'] and gate['source_hashes'] == source_hashes()
    assert gate['log_sha256'] == r.digest(HERE / 'preflight.log')
    assert gate['search_sha256'] == r.digest(HERE / 'search-space.json')
    assert gate['test_hashes'] == {p.name:r.digest(p) for p in (ROOT / 'tests').glob('test_*.py')}
    choices = read(HERE / 'search-space.json'); records = completed()
    label, seed, phase, reason, evidence = select(records, choices)
    selection = dict(label=label, seed=seed, phase=phase, reason=reason, evidence=evidence,
                     hypothesis=choices[label]['hypothesis'], control=choices[label]['control'],
                     selected_at=datetime.now(timezone.utc).isoformat(), remaining_seconds=remaining())
    out = HERE / f'trial-{len(records)+1:04d}-{label}-s{seed}'
    out.mkdir(exist_ok=False)
    r.write_json(out / 'selection.json', selection)
    setup = r.verify_runtime(); seal = read(ROOT / '.autoresearch/data-seal.json')
    r.verify_seal(ROOT / '.autoresearch/cache', seal)
    candidate = r.validate_candidate(choices[label]['candidate'])
    protocol = read(REFERENCE / 'protocol-42.json'); protocol['seed'] = seed
    r.validate_protocol(protocol)
    metadata = dict(kind='fixed_updates', condition=candidate['feedforward'], candidate=candidate,
                    protocol=protocol, label=label, seed=seed, phase=phase, selection=selection,
                    record_version=6, upstream=setup, data_seal=seal,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
                    provenance_note='Executed captured hashes identify this run; publishing commit is not retroactive provenance',
                    orchestrator_sha256=r.digest(Path(__file__)), preflight_sha256=r.digest(HERE / 'preflight-result.json'),
                    status='prepared', metrics=None)
    metadata.update({key:r.digest(ROOT / 'scripts' / name) for name,key in r.PROJECT_FILES.items()})
    snapshot = r.capture_run_snapshot(out, metadata); metadata['snapshot_files'] = snapshot
    r.write_json(out / 'result.json', metadata)
    process = None; start = None
    try:
        with announce(out / 'run.log') as log:
            assert remaining() >= 2700, 'Do not consume final report reserve'
            metadata['status'] = 'running'; start = time.monotonic()
            r.write_json(out / 'result.json', metadata)
            process = subprocess.Popen(r.snapshot_command(r.runtime_python(), out), cwd=out,
                env=r.environment(), stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            metadata['returncode'] = process.wait(timeout=900)
        if metadata['returncode']:
            raise RuntimeError('Training child failed with exit ' + str(metadata['returncode']))
        metadata['metrics'] = r.parse_summary((out / 'run.log').read_text())
        metadata['artifacts'] = r.validate_run_artifacts(out, metadata)
        metadata['execution'] = r.validate_execution(out, snapshot)
        training = metadata['artifacts']['training']; batches = read(out / 'batches.json')
        from autoresearch_train import batch_order, fixed_schedule
        assert training['optimizer_updates'] == metadata['metrics']['num_steps'] == 512
        assert training['training_tokens'] == batches['training_tokens'] == 8388608
        assert training['timed_training_tokens'] == 8208384
        assert batches['consumed_indices'] == batch_order(8192, seed)
        assert read(out / 'schedule.json')['updates'] == fixed_schedule()
        peers = [x for x in records if x['seed'] == seed]
        if peers:
            other = Path(peers[0]['_path'])
            assert batches == read(other / 'batches.json')
            assert protocol == read(other / 'protocol.json')
        control = choices[label]['control']
        controls = [x for x in peers if x['label'] == control]
        if controls and candidate['depth'] == controls[0]['candidate']['depth']:
            old = read(Path(controls[0]['_path']) / 'initialization.json')['parameters']
            new = read(out / 'initialization.json')['parameters']
            # Two-expert construction changes extra-expert RNG progression; do not
            # falsely require full cross-count pairing beyond recorded hashes.
            if label != 'M-experts2':
                assert all(new[n] == h for n,h in old.items() if n in new), 'Shared initialization mismatch'
        metadata.update(memory=read(out / 'memory.json'), status='completed')
    except BaseException as exc:
        if process is not None and process.poll() is None:
            r.stop_process(process)
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        metadata['wall_seconds'] = time.monotonic() - start if start else 0
        r.write_json(out / 'result.json', metadata); r.emit_compact_report(out)
    print(json.dumps({'run':out.name, 'label':label, 'seed':seed, 'phase':phase,
                      'bpb':metadata['metrics']['val_bpb'], 'wall_seconds':metadata['wall_seconds'],
                      'remaining_seconds':remaining()}), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'preflight':
        preflight()
    elif sys.argv[1] == 'next':
        trial()
    else:
        raise ValueError('Use preflight or next; no unbounded launcher')
