"""Frozen validation concept prediction versus persistence and training-majority baselines."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(run, output):
    run = run.resolve()
    record = read(run / 'result.json')
    result = dict(kind='concept_baseline_probe', status='running', trial=run.name, seed=record['seed'],
        script_sha256=digest(__file__), result_sha256=digest(run / 'result.json'),
        checkpoint_sha256=record['checkpoint_sha256'], training_health_sha256=digest(run / 'ncp-health.json'),
        optimizer_updates=0, condition_reached='setup')
    write(output, result)
    try:
        manifest = read(run / 'snapshot.json')
        assert record['status'] == 'completed' and manifest == record['snapshot_files']
        protocol = record['protocol']
        assert read(run / 'protocol.json') == protocol and read(run / 'candidate.json') == record['candidate']
        assert protocol['eval_tokens'] == 65536 and protocol['eval_batch_size'] == 2
        assert 65536 % (2 * protocol['sequence_length']) == 0

        def verify():
            for name, value in manifest.items():
                assert digest(run / name) == value, name
            assert digest(run / 'checkpoint_pre_eval.pt') == result['checkpoint_sha256']
            assert digest(run / 'result.json') == result['result_sha256']
            assert digest(run / 'ncp-health.json') == result['training_health_sha256']

        verify()
        sys.path[:0] = [str(run / 'source/project'), str(run / 'source/upstream')]
        import torch
        import prepare
        prepare.MAX_SEQ_LEN = protocol['sequence_length']
        import train
        from autoresearch_model import model_class
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        train.MAX_SEQ_LEN = protocol['sequence_length']
        train.WINDOW_PATTERN = 'L'
        runtime = train.detect_runtime()
        assert runtime.device_type == 'cuda' and runtime.amp_dtype == torch.bfloat16
        train._configure_step_kernels(runtime)
        state = torch.load(run / 'checkpoint_pre_eval.pt', map_location='cpu', weights_only=True)
        config = train.build_model_config(record['candidate']['depth'], state['transformer.wte.weight'].shape[0],
                                          runtime, protocol['activation_checkpointing'])
        assert config.n_embd == state['transformer.wte.weight'].shape[1]
        with torch.device('meta'):
            model = model_class(train, record['candidate'])(config)
        model.to_empty(device=runtime.device)
        model.init_weights(embed_dtype=runtime.amp_dtype)
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        ncp = model.ncp
        assert ncp is not None and ncp.settings['mode'] == 'feedback'
        settings = copy.deepcopy(ncp.settings)

        def state_hash():
            h = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                h.update(name.encode())
                h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            return h.hexdigest()

        def sources():
            loaded = {}
            for name, module in tuple(sys.modules.items()):
                path = getattr(module, '__file__', None)
                if path is None:
                    continue
                try:
                    relative = Path(path).resolve().relative_to(run).as_posix()
                except ValueError:
                    continue
                if relative.startswith('source/') and relative.endswith('.py'):
                    assert relative in manifest and digest(run / relative) == manifest[relative]
                    loaded[name] = dict(path=relative, sha256=manifest[relative])
            assert all(x in loaded for x in ('train', 'prepare', 'autoresearch_model', 'autoresearch_ncp'))
            return loaded

        loaded = sources()
        before = state_hash()
        original = read(run / 'evaluation-immutability.json')
        assert original['verified'] and before == original['before_sha256'] == original['after_sha256']
        training_counts = torch.tensor(read(run / 'ncp-health.json')['target_counts'], dtype=torch.long)
        assert training_counts.shape == (ncp.segments, ncp.entries)
        assert (training_counts.sum(1) > 0).all()
        majority = training_counts.argmax(1).to(runtime.device)
        correct = {name: torch.zeros(ncp.segments, dtype=torch.long, device=runtime.device)
                   for name in ('learned', 'persistence', 'training_majority')}
        counts = 0
        captured = []

        def capture(module, inputs, outputs):
            # Hold the already-consumed input; never change the module's output.
            captured.append(inputs[0].detach().clone())

        hook = ncp.register_forward_hook(capture)
        tokenizer = prepare.Tokenizer.from_directory(dataset='tinystories')
        byte_lengths = prepare.get_token_bytes(device='cpu', dataset='tinystories')
        steps = 65536 // (2 * protocol['sequence_length'])
        expected = hashlib.sha256()
        expected_bytes = 0
        loader = prepare.make_dataloader(tokenizer, 2, protocol['sequence_length'], 'val',
                                        device='cpu', dataset='tinystories')
        for _ in range(steps):
            x, y, _ = next(loader)
            for tensor in (x, y):
                expected.update(tensor.contiguous().numpy().tobytes())
            expected_bytes += int(byte_lengths[y].sum())
        del loader, x, y
        batch_hash = hashlib.sha256()
        accounting = dict(tokens=0, target_bytes=0, batches=0)
        forward = model.forward

        def counted(x, y=None, **kwargs):
            nonlocal counts
            assert y is not None and kwargs == {'reduction': 'none'} and not captured
            value = forward(x, y, **kwargs)
            assert len(captured) == 1
            # These labels/diagnostics run only after predictive token loss exists.
            hidden = captured.pop()
            with torch.no_grad():
                pooled = ncp.pool(hidden)
                codes = ncp.codebook()
                _, logits = ncp.predict(pooled, codes)
                _, indices = ncp.quantize(pooled, codes)
                target = indices[:, 1:]
                correct['learned'].add_((logits[:, :-1].argmax(-1) == target).sum((0, 1)))
                correct['persistence'].add_((indices[:, :-1] == target).sum((0, 1)))
                correct['training_majority'].add_((majority[None, None, :] == target).sum((0, 1)))
                counts += target.shape[0] * target.shape[1]
            for tensor in (x, y):
                batch_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            accounting['tokens'] += y.numel()
            accounting['target_bytes'] += int(byte_lengths[y.detach().cpu()].sum())
            accounting['batches'] += 1
            return value

        model.forward = counted
        result.update(condition_reached='evaluation', state_before=before, loaded_modules=loaded)
        write(output, result)
        with torch.no_grad(), torch.autocast('cuda', dtype=runtime.amp_dtype):
            bpb = prepare.evaluate_bpb(model, tokenizer, 2, device=runtime.device, dataset='tinystories', eval_tokens=65536)
        model.forward = forward
        hook.remove()
        assert not captured and ncp.settings == settings
        assert before == state_hash() and loaded == sources()
        assert accounting == dict(tokens=65536, target_bytes=expected_bytes, batches=steps)
        assert batch_hash.hexdigest() == expected.hexdigest()
        assert counts == steps * 2 * (protocol['sequence_length'] // ncp.chunk_size - 1)
        assert math.isfinite(bpb) and abs(bpb - record['metrics']['val_bpb']) <= 1e-6
        verify()
        scores = {name: dict(correct_per_segment=values.tolist(),
            accuracy_per_segment=(values.double() / counts).tolist(),
            aggregate_accuracy=float(values.sum()) / (counts * ncp.segments)) for name, values in correct.items()}
        result.update(status='completed', condition_reached='verified', bpb=bpb, original_bpb=record['metrics']['val_bpb'],
            absolute_bpb_difference=abs(bpb - record['metrics']['val_bpb']), scores=scores,
            comparisons_per_segment=counts, segments=ncp.segments, entries=ncp.entries,
            training_majority_indices=majority.tolist(), training_target_counts=training_counts.tolist(),
            accounting=accounting, batch_sha256=batch_hash.hexdigest(), data_seal=record['data_seal'],
            source_hashes=manifest, state_after=state_hash(), persistent_state_immutable=True, settings_immutable=True,
            interpretation='Next complete chunk code identity under the trained codebook, not semantics. Persistence uses current chunk assignments; majority is fixed from saved training-health counts with lowest-index tie breaking. Validation labels enter diagnostics only after token prediction. These are the same four confirmation checkpoints and validation data, not new independent seeds or held-out evidence.')
        write(output, result)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        write(output, result)
        raise


def controller(freeze_only=False, publish=False):
    import autoresearch as r
    import ncp_campaign as c
    import experiment_reports as reports
    from ncp_frozen_pairs import collect
    from ncp_crossed_order import identity
    frozen_path = c.HERE / 'confirmation-selection.json'
    plan_path = c.HERE / 'concept-baselines-plan.json'
    archive = c.HERE / 'concept-baselines-driver.py'
    source = Path(__file__).read_bytes()
    if plan_path.exists():
        plan = read(plan_path)
        assert archive.read_bytes() == source and digest(archive) == plan['script_sha256']
        assert digest(frozen_path) == plan['frozen_selection_sha256']
    else:
        frozen = read(frozen_path)
        pairs = collect(frozen)
        assert all(x['completed_pairs'] == 4 for x in pairs['summaries'].values())
        anchors = []
        for seed in (45, 46, 43, 44):
            pair = next(x for x in pairs['pairs'] if x['seed'] == seed and x['control'] == 'D6')
            path = c.HERE / pair['candidate_trial']
            anchors.append(dict(seed=seed, identity=identity(path), training_health_sha256=digest(path / 'ncp-health.json')))
        assert not archive.exists()
        archive.write_bytes(source)
        plan = dict(kind='concept_baselines_plan', anchors=anchors, script_sha256=digest(archive),
            frozen_selection_sha256=digest(frozen_path), data_seal=read(r.ROOT / '.autoresearch/data-seal.json'),
            child_timeout_seconds=120, minimum_launch_remaining_seconds=1500,
            frozen_at=c.datetime.now(c.timezone.utc).isoformat(), optimizer_updates=0,
            hypothesis='Compare learned next-chunk code prediction against persistence and training-frequency majority, without fitting on validation')
        assert plan['data_seal'] == frozen['confirmation_data_seal']
        write(plan_path, plan)
    if freeze_only:
        return
    outcomes = []
    blocked = False

    def aggregate():
        write(c.HERE / 'concept-baselines-result.json', dict(kind='concept_baselines',
            status='completed' if len(outcomes) == 4 and all(x['status'] == 'completed' for x in outcomes) else 'incomplete',
            plan_sha256=digest(plan_path), outcomes=outcomes, optimizer_updates=0))
        reports.emit(c.HERE)
        reports.index()

    for item in plan['anchors']:
        run = c.HERE / item['identity']['trial']
        assert identity(run) == item['identity']
        assert digest(run / 'ncp-health.json') == item['training_health_sha256']
        assert read(run / 'result.json')['data_seal'] == plan['data_seal']
        output = c.HERE / f'concept-baselines-s{item["seed"]}-result.json'
        log = c.HERE / f'concept-baselines-s{item["seed"]}.log'
        if output.exists():
            value = read(output)
            assert value['script_sha256'] == plan['script_sha256']
            assert value['result_sha256'] == item['identity']['files']['result.json']
            assert value['checkpoint_sha256'] == item['identity']['files']['checkpoint_pre_eval.pt']
            assert value['training_health_sha256'] == item['training_health_sha256']
        elif blocked or c.remaining() < 1500:
            outcomes.append(dict(seed=item['seed'], trial=run.name, status='skipped',
                reason='Prior probe failure' if blocked else 'Final25-minute reserve'))
            aggregate()
            continue
        else:
            failure = None
            c.log(f'START concept baselines {run.name}; log={log}')
            with c.gpu_lock():
                try:
                    r.verify_seal(r.ROOT / '.autoresearch/cache', plan['data_seal'])
                    with log.open('x', encoding='utf-8') as handle:
                        process = subprocess.run([str(r.runtime_python()), '-B', '-I', str(archive), '--run', str(run),
                            '--output', str(output)], cwd=r.ROOT, env=r.environment(), stdout=handle, stderr=subprocess.STDOUT, timeout=120)
                    assert process.returncode == 0, f'Child exit {process.returncode}'
                except BaseException as exc:
                    failure = repr(exc)
                try:
                    r.verify_seal(r.ROOT / '.autoresearch/cache', plan['data_seal'])
                except BaseException as exc:
                    failure = str(failure) + '; post-probe seal: ' + repr(exc)
            value = read(output) if output.exists() else dict(trial=run.name, seed=item['seed'])
            if failure:
                value.update(status='failed', error=failure, script_sha256=plan['script_sha256'],
                    result_sha256=item['identity']['files']['result.json'],
                    checkpoint_sha256=item['identity']['files']['checkpoint_pre_eval.pt'],
                    training_health_sha256=item['training_health_sha256'])
                write(output, value)
            c.log(f'END concept baselines {run.name}: {value["status"]}')
        blocked = blocked or value['status'] != 'completed'
        compact = {key: value[key] for key in ('trial', 'seed', 'status', 'bpb', 'original_bpb', 'absolute_bpb_difference',
            'scores', 'comparisons_per_segment', 'segments', 'accounting', 'batch_sha256', 'error') if key in value}
        compact.update(result_path=output.name, result_sha256=digest(output),
                       log_path=log.name, log_sha256=digest(log) if log.exists() else None)
        outcomes.append(compact)
        aggregate()
    aggregate()
    if publish:
        import ncp_search as search
        try:
            search.publish()
        except Exception as exc:
            c.log('Publication deferred; concept baseline evidence retained: ' + repr(exc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.run:
        assert args.output is not None
        child(args.run, args.output)
    else:
        assert args.output is None
        controller(args.freeze_only, args.publish)
