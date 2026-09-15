"""Captured-checkpoint inference interventions; never trains or rewrites weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    record = json.loads((run / 'result.json').read_text())
    progress = dict(kind='frozen_checkpoint_inference_intervention', status='running', trial=run.name,
                    seed=record['seed'], checkpoint_sha256=record['checkpoint_sha256'],
                    script_sha256=digest(__file__), data_seal=record['data_seal'], outcomes=[], condition_reached='setup')
    args.output.write_text(json.dumps(progress, indent=2) + '\n', encoding='utf-8')
    manifest = json.loads((run / 'snapshot.json').read_text())
    for name, value in manifest.items():
        assert digest(run / name) == value, name
    checkpoint = run / 'checkpoint_pre_eval.pt'
    assert record['status'] == 'completed'
    assert record['candidate']['ncp']['mode'] == 'feedback'
    assert digest(checkpoint) == record['checkpoint_sha256']
    sys.path[:0] = [str(run / 'source/project'), str(run / 'source/upstream')]
    import torch
    import prepare
    protocol = record['protocol']
    prepare.MAX_SEQ_LEN = protocol['sequence_length']
    import train
    from autoresearch_model import model_class
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    train.MAX_SEQ_LEN = protocol['sequence_length']
    train.WINDOW_PATTERN = 'L'
    runtime = train.detect_runtime()
    assert runtime.device_type == 'cuda'
    assert runtime.amp_dtype == torch.bfloat16, 'Probe precision differs from the campaign BF16 protocol'
    train._configure_step_kernels(runtime)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = train.build_model_config(record['candidate']['depth'], state['transformer.wte.weight'].shape[0], runtime, False)
    with torch.device('meta'):
        model = model_class(train, record['candidate'])(config)
    model.to_empty(device='cuda')
    model.init_weights(embed_dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True)
    model.eval()
    del state
    tokenizer = prepare.Tokenizer.from_directory(dataset='tinystories')
    byte_lengths = prepare.get_token_bytes(device='cpu', dataset='tinystories')
    original_settings = copy.deepcopy(model.ncp.settings)
    original_basis = model.ncp.codebook.basis.detach().clone()
    forward = model.forward

    def tensor_hash(items):
        h = hashlib.sha256()
        for name, tensor in items:
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def source_receipt():
        loaded = {}
        for name, module in tuple(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if not path:
                continue
            path = Path(path).resolve()
            try:
                relative = path.relative_to(run).as_posix()
            except ValueError:
                continue
            if relative.startswith('source/') and path.suffix == '.py':
                assert relative in manifest and digest(path) == manifest[relative]
                loaded[name] = dict(path=relative, sha256=manifest[relative])
        assert all(name in loaded for name in ('train', 'prepare', 'autoresearch_model', 'autoresearch_ncp'))
        return loaded

    loaded_before = source_receipt()
    original_state = tensor_hash(model.state_dict().items())
    original_weights = tensor_hash(model.named_parameters())
    outcomes = []
    for condition in ('original', 'feedback_zero', 'codebook_rows_rotated'):
        progress.update(condition_reached=condition, outcomes=outcomes)
        args.output.write_text(json.dumps(progress, indent=2) + '\n', encoding='utf-8')
        model.ncp.settings = copy.deepcopy(original_settings)
        with torch.no_grad():
            model.ncp.codebook.basis.copy_(original_basis)
            if condition == 'feedback_zero':
                model.ncp.settings['feedback_scale'] = 0.
            elif condition == 'codebook_rows_rotated':
                model.ncp.codebook.basis.copy_(original_basis.roll(1, dims=1))
        before = tensor_hash(model.state_dict().items())
        settings_before = copy.deepcopy(model.ncp.settings)
        assert tensor_hash(model.named_parameters()) == original_weights
        loader = prepare.make_dataloader(tokenizer, protocol['eval_batch_size'], prepare.MAX_SEQ_LEN,
                                        'val', device='cuda', dataset='tinystories')
        x, _, _ = next(loader)
        changed = x.clone()
        cut = x.shape[1] // 2
        changed[:, cut:] = (changed[:, cut:] + 1) % config.vocab_size
        with torch.no_grad(), torch.autocast('cuda', dtype=runtime.amp_dtype):
            left, right = forward(x), forward(changed)
        torch.testing.assert_close(left[:, :cut], right[:, :cut], rtol=0, atol=0)
        del loader, left, right, changed, x
        accounting = dict(tokens=0, target_bytes=0, batches=0)
        batch_hash = hashlib.sha256()

        def counted_forward(x, y=None, **kwargs):
            assert y is not None and kwargs == {'reduction': 'none'}
            for tensor in (x, y):
                batch_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            accounting['tokens'] += y.numel()
            accounting['target_bytes'] += int(byte_lengths[y.detach().cpu()].sum())
            accounting['batches'] += 1
            return forward(x, y, **kwargs)

        model.forward = counted_forward
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype=runtime.amp_dtype):
            bpb = prepare.evaluate_bpb(model, tokenizer, protocol['eval_batch_size'], device='cuda',
                                       dataset='tinystories', eval_tokens=protocol['eval_tokens'])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        model.forward = forward
        after = tensor_hash(model.state_dict().items())
        assert before == after and model.ncp.settings == settings_before
        assert accounting['tokens'] == protocol['eval_tokens']
        item = dict(condition=condition, bpb=bpb, evaluation_seconds=elapsed, **accounting,
                    batch_sha256=batch_hash.hexdigest(), state_before=before, state_after=after,
                    trainable_weights_sha256=original_weights, settings=settings_before,
                    prefix_causality_verified=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if outcomes:
            assert all(item[k] == outcomes[0][k] for k in ('batch_sha256', 'tokens', 'target_bytes', 'batches'))
            item['delta_from_original_bpb'] = bpb - outcomes[0]['bpb']
        else:
            assert abs(bpb - record['metrics']['val_bpb']) <= 1e-6, (bpb, record['metrics']['val_bpb'])
        outcomes.append(item)
        print(json.dumps(item), flush=True)
    model.ncp.settings = original_settings
    with torch.no_grad():
        model.ncp.codebook.basis.copy_(original_basis)
    assert tensor_hash(model.state_dict().items()) == original_state
    assert source_receipt() == loaded_before
    for name, value in manifest.items():
        assert digest(run / name) == value
    assert digest(checkpoint) == record['checkpoint_sha256']
    result = dict(kind='frozen_checkpoint_inference_intervention', status='completed', trial=run.name,
                  seed=record['seed'], label=record['label'], optimizer_updates=0, outcomes=outcomes,
                  checkpoint_sha256=record['checkpoint_sha256'], result_sha256=digest(run / 'result.json'),
                  source_hashes=manifest, loaded_modules=loaded_before, script_sha256=digest(__file__),
                  data_seal=record['data_seal'],
                  state_restored=True, trainable_weights_unchanged=True,
                  interpretation='Inference interventions, not retrained ablations; rotation changes code identity, not codebook contents. No semantic-concept or independent-generalization claim. Instrumented evaluation time includes CPU accounting and is not a throughput comparison.')
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
