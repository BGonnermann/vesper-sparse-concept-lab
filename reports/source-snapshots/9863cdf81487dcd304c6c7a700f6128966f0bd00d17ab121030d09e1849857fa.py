"""Independent captured-source evaluation replay; no optimizer or training."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(run, output):
    run = run.resolve()
    record = read(run / 'result.json')
    progress = dict(kind='checkpoint_evaluation_replay', status='running', trial=run.name,
        checkpoint_sha256=record['checkpoint_sha256'], result_sha256=digest(run / 'result.json'),
        script_sha256=digest(__file__), condition_reached='setup', optimizer_updates=0)
    write(output, progress)
    try:
        assert record['status'] == 'completed'
        manifest = read(run / 'snapshot.json')
        assert manifest == record['snapshot_files']
        checkpoint = run / 'checkpoint_pre_eval.pt'

        def verify_files():
            for name, value in manifest.items():
                assert digest(run / name) == value, name
            assert digest(checkpoint) == record['checkpoint_sha256']
            assert digest(run / 'result.json') == progress['result_sha256']

        verify_files()
        protocol = record['protocol']
        assert read(run / 'protocol.json') == protocol
        assert read(run / 'candidate.json') == record['candidate']
        assert protocol['eval_tokens'] == 65536
        assert protocol['eval_tokens'] % (protocol['sequence_length'] * protocol['eval_batch_size']) == 0
        sys.path[:0] = [str(run / 'source/project'), str(run / 'source/upstream')]
        import torch
        import prepare
        prepare.MAX_SEQ_LEN = protocol['sequence_length']
        import train
        import autoresearch_model as extension
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('high')
        train.MAX_SEQ_LEN = protocol['sequence_length']
        train.WINDOW_PATTERN = 'L'
        runtime = train.detect_runtime()
        assert runtime.device_type == 'cuda' and runtime.amp_dtype == torch.bfloat16
        train._configure_step_kernels(runtime)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        candidate = record['candidate']
        config = train.build_model_config(candidate['depth'], state['transformer.wte.weight'].shape[0],
                                          runtime, protocol['activation_checkpointing'])
        if 'model_width' in candidate:
            config = extension.with_model_width(config, candidate)
        assert config.n_embd == state['transformer.wte.weight'].shape[1]
        assert config.n_embd == read(run / 'model.json')['width']
        with torch.device('meta'):
            model = extension.model_class(train, candidate)(config)
        model.to_empty(device=runtime.device)
        model.init_weights(embed_dtype=runtime.amp_dtype)
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        assert sum(p.numel() for p in model.parameters()) == read(run / 'model.json')['total_parameters']

        def state_hash():
            h = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                h.update(name.encode())
                h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            return h.hexdigest()

        def sources():
            loaded = {}
            for name, module in tuple(sys.modules.items()):
                filename = getattr(module, '__file__', None)
                if not filename:
                    continue
                try:
                    relative = Path(filename).resolve().relative_to(run).as_posix()
                except ValueError:
                    continue
                if relative.startswith('source/') and relative.endswith('.py'):
                    assert relative in manifest and digest(run / relative) == manifest[relative]
                    loaded[name] = dict(path=relative, sha256=manifest[relative])
            assert all(name in loaded for name in ('train', 'prepare', 'autoresearch_model', 'autoresearch_ncp'))
            return loaded

        loaded = sources()
        before = state_hash()
        original_evaluation = read(run / 'evaluation-immutability.json')
        assert original_evaluation['verified']
        assert before == original_evaluation['before_sha256'] == original_evaluation['after_sha256']
        ncp = getattr(model, 'ncp', None)
        settings = copy.deepcopy(ncp.settings) if ncp is not None else None
        tokenizer = prepare.Tokenizer.from_directory(dataset='tinystories')
        byte_lengths = prepare.get_token_bytes(device='cpu', dataset='tinystories')
        expected_hash = hashlib.sha256()
        expected_bytes = 0
        loader = prepare.make_dataloader(tokenizer, protocol['eval_batch_size'], protocol['sequence_length'],
                                        'val', device='cpu', dataset='tinystories')
        steps = protocol['eval_tokens'] // (protocol['eval_batch_size'] * protocol['sequence_length'])
        for _ in range(steps):
            x, y, _ = next(loader)
            for tensor in (x, y):
                expected_hash.update(tensor.contiguous().numpy().tobytes())
            expected_bytes += int(byte_lengths[y].sum())
        del loader, x, y
        accounting = dict(tokens=0, target_bytes=0, batches=0)
        actual_hash = hashlib.sha256()
        forward = model.forward

        def counted(x, y=None, **kwargs):
            assert y is not None and kwargs == {'reduction': 'none'}
            for tensor in (x, y):
                actual_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            accounting['tokens'] += y.numel()
            accounting['target_bytes'] += int(byte_lengths[y.detach().cpu()].sum())
            accounting['batches'] += 1
            return forward(x, y, **kwargs)

        model.forward = counted
        progress.update(condition_reached='evaluation', loaded_modules=loaded, state_before=before)
        write(output, progress)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=runtime.amp_dtype):
            bpb = prepare.evaluate_bpb(model, tokenizer, protocol['eval_batch_size'], device=runtime.device,
                                      dataset='tinystories', eval_tokens=protocol['eval_tokens'])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        model.forward = forward
        after = state_hash()
        assert before == after
        assert ncp is None or ncp.settings == settings
        assert accounting == dict(tokens=65536, target_bytes=expected_bytes, batches=steps)
        assert actual_hash.hexdigest() == expected_hash.hexdigest()
        assert math.isfinite(bpb)
        difference = abs(bpb - record['metrics']['val_bpb'])
        progress.update(bpb=bpb, original_bpb=record['metrics']['val_bpb'], absolute_bpb_difference=difference,
                        accounting=accounting, batch_sha256=actual_hash.hexdigest(), evaluation_seconds=elapsed)
        write(output, progress)
        assert difference <= 1e-6, ('BPB replay mismatch', bpb, record['metrics']['val_bpb'])
        assert sources() == loaded
        verify_files()
        progress.update(status='completed', condition_reached='verified', state_after=after,
            persistent_state_immutable=True, settings_immutable=True, data_seal=record['data_seal'],
            snapshot_sha256=digest(run / 'snapshot.json'), source_hashes=manifest,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            interpretation='Independent execution of the original validation protocol, not independent data or training. Evaluation order reconstructed from captured deterministic loader and sealed data; historical runs did not record evaluation batch hashes. Instrumented time includes CPU accounting, not a throughput comparison.')
        write(output, progress)
    except BaseException as exc:
        progress.update(status='failed', error=repr(exc))
        write(output, progress)
        raise


def controller(freeze_only=False, publish=False):
    import autoresearch as r
    import ncp_campaign as c
    import experiment_reports as reports
    from ncp_crossed_order import identity, verify_artifacts
    directory = c.HERE / 'checkpoint-replay'
    directory.mkdir(exist_ok=True)
    archive = directory / 'driver.py'
    root_archive = c.HERE / 'checkpoint-replay-driver.py'
    plan_path = directory / 'plan.json'
    source = Path(__file__).read_bytes()
    if root_archive.exists():
        assert root_archive.read_bytes() == source
    else:
        root_archive.write_bytes(source)
    if plan_path.exists():
        plan = read(plan_path)
        assert archive.read_bytes() == source and digest(archive) == plan['script_sha256']
    else:
        assert not archive.exists()
        completed, excluded = [], []
        for path in sorted(c.HERE.glob('trial-*/result.json')):
            record = read(path)
            assert record['status'] in ('completed', 'failed'), 'Freeze only after every trial reaches a terminal state'
            if record['status'] != 'completed':
                excluded.append(dict(trial=path.parent.name, status=record['status'], result_sha256=digest(path)))
                continue
            verify_artifacts(path.parent)
            completed.append(identity(path.parent))
        assert completed
        seal = read(r.ROOT / '.autoresearch/data-seal.json')
        assert all(read(c.HERE / x['trial'] / 'result.json')['data_seal'] == seal for x in completed)
        archive.write_bytes(source)
        plan = dict(kind='checkpoint_replay_plan', trials=completed, excluded=excluded, data_seal=seal,
            script_sha256=digest(archive), child_timeout_seconds=120, minimum_launch_remaining_seconds=1500,
            frozen_at=c.datetime.now(c.timezone.utc).isoformat(), optimizer_updates=0)
        write(plan_path, plan)
    if freeze_only:
        return
    current = {p.parent.name: digest(p) for p in c.HERE.glob('trial-*/result.json')}
    expected = {x['trial']: x['files']['result.json'] for x in plan['trials']}
    expected.update({x['trial']: x['result_sha256'] for x in plan['excluded']})
    assert current == expected, 'Trial inventory changed after freeze'
    outcomes = []

    def aggregate():
        good = [x for x in outcomes if x['status'] == 'completed']
        compact = []
        for outcome in outcomes:
            item = {key: outcome[key] for key in ('trial', 'status', 'bpb', 'original_bpb',
                'absolute_bpb_difference', 'accounting', 'batch_sha256', 'reason', 'error') if key in outcome}
            for kind, path in (('result', directory / (outcome['trial'] + '-result.json')),
                               ('log', directory / (outcome['trial'] + '.log'))):
                if path.exists():
                    item[kind + '_path'] = path.relative_to(c.HERE).as_posix()
                    item[kind + '_sha256'] = digest(path)
            compact.append(item)
        result = dict(kind='checkpoint_replay', status='completed' if len(good) == len(plan['trials']) else 'failed' if any(x['status'] == 'failed' for x in outcomes) else 'incomplete',
            plan_sha256=digest(plan_path), expected_checkpoints=len(plan['trials']), completed_checkpoints=len(good),
            outcomes=compact, excluded_attempts=plan['excluded'], optimizer_updates=0,
            max_absolute_bpb_difference=max((x['absolute_bpb_difference'] for x in outcomes if 'absolute_bpb_difference' in x), default=None))
        write(c.HERE / 'checkpoint-replay-result.json', result)
        write(directory / 'result.json', result)
        reports.emit(directory)
        reports.index()

    blocked = False
    for item in plan['trials']:
        run = c.HERE / item['trial']
        assert identity(run) == item
        output = directory / (run.name + '-result.json')
        if output.exists():
            result = read(output)
            assert result['script_sha256'] == plan['script_sha256']
            assert result['checkpoint_sha256'] == item['files']['checkpoint_pre_eval.pt']
            assert result['result_sha256'] == item['files']['result.json']
            outcomes.append(result)
            blocked = blocked or result['status'] != 'completed'
            aggregate()
            continue
        if blocked or c.remaining() < 1500:
            outcomes.append(dict(trial=run.name, status='skipped', reason='Prior replay failure' if blocked else 'Final report reserve'))
            aggregate()
            continue
        log = directory / (run.name + '.log')
        c.log(f'START checkpoint replay {run.name}; log={log}')
        failure = None
        with c.gpu_lock():
            try:
                r.verify_seal(r.ROOT / '.autoresearch/cache', plan['data_seal'])
                with log.open('x', encoding='utf-8') as handle:
                    process = subprocess.run([str(r.runtime_python()), '-B', '-I', str(archive), '--run', str(run),
                        '--output', str(output)], cwd=r.ROOT, env=r.environment(), stdout=handle,
                        stderr=subprocess.STDOUT, timeout=120)
                assert process.returncode == 0, f'Child exit {process.returncode}'
            except BaseException as exc:
                failure = repr(exc)
            try:
                r.verify_seal(r.ROOT / '.autoresearch/cache', plan['data_seal'])
            except BaseException as exc:
                failure = str(failure) + '; post-replay seal: ' + repr(exc)
        result = read(output) if output.exists() else dict(trial=run.name)
        if failure:
            result.update(status='failed', error=failure, script_sha256=plan['script_sha256'],
                checkpoint_sha256=item['files']['checkpoint_pre_eval.pt'], result_sha256=item['files']['result.json'])
            blocked = True
        result['log_sha256'] = digest(log) if log.exists() else None
        write(output, result)
        outcomes.append(result)
        aggregate()
        c.log(f'END checkpoint replay {run.name}: {result["status"]}')
    aggregate()
    if publish:
        import ncp_search as search
        try:
            search.publish()
        except Exception as exc:
            c.log('Publication deferred; replay evidence retained: ' + repr(exc))


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
