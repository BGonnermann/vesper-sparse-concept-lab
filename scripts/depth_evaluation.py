"""Captured-source checkpoint evaluation with explicitly audited split dispatch."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def evaluate_split(prepare,model,tokenizer,batch,split,**kwargs):
    if split not in ('val','test'):raise ValueError('Only validation or frozen final test evaluation')
    original=prepare.make_dataloader
    receipt=dict(requested_split=split,actual_splits=[])
    def dispatch(tokenizer,batch,length,requested,**options):
        assert requested=='val','Upstream evaluator changed its split contract'
        receipt['actual_splits'].append(split)
        return original(tokenizer,batch,length,split,**options)
    prepare.make_dataloader=dispatch
    try:
        value=prepare.evaluate_bpb(model,tokenizer,batch,**kwargs)
        assert receipt['actual_splits']==[split]
        return value,receipt
    finally:prepare.make_dataloader=original


def child(run, output, split, freeze):
    run = run.resolve()
    frozen=read(freeze)
    assert frozen['status']=='frozen'
    expected=next(x for x in frozen['checkpoints'] if x['trial']==run.name)
    assert digest(run/'result.json')==expected['result_sha256']
    if split=='test':assert run.name in frozen['test_trials']
    assert digest(__file__)==frozen['evaluator_sha256']
    record = read(run / 'result.json')
    assert record['checkpoint_sha256']==expected['checkpoint_sha256']
    progress = dict(kind='checkpoint_evaluation_replay', status='running', trial=run.name,
        split=split,freeze_sha256=digest(freeze),checkpoint_sha256=record['checkpoint_sha256'], result_sha256=digest(run / 'result.json'),
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
        assert prepare.DATASET_CONFIGS['tinystories']['splits']=={
            'test':tuple(frozen['test_split_rows']), 'val':tuple(frozen['validation_split_rows']),
            'train':tuple(frozen['train_split_rows'])},'Captured split boundaries differ from freeze'
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
                                        split, device='cpu', dataset='tinystories')
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
            bpb,dispatch = evaluate_split(prepare,model, tokenizer, protocol['eval_batch_size'], split, device=runtime.device,
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
        progress.update(bpb=bpb, original_validation_bpb=record['metrics']['val_bpb'], absolute_validation_bpb_difference=difference if split=='val' else None, dispatch=dispatch,
                        accounting=accounting, batch_sha256=actual_hash.hexdigest(), evaluation_seconds=elapsed)
        write(output, progress)
        if split=='val':
            assert difference <= 1e-6, ('BPB replay mismatch', bpb, record['metrics']['val_bpb'])
        assert sources() == loaded
        verify_files()
        progress.update(status='completed', condition_reached='verified', state_after=after,
            split_rows=frozen['test_split_rows'] if split=='test' else frozen['validation_split_rows'],
            persistent_state_immutable=True, settings_immutable=True, data_seal=record['data_seal'],
            snapshot_sha256=digest(run / 'snapshot.json'), source_hashes=manifest,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            interpretation='Frozen checkpoint, explicit split dispatch through the original BPB metric. CPU/GPU evaluation order and byte accounting match. Instrumented evaluation time is not a training-throughput comparison.')
        write(output, progress)
    except BaseException as exc:
        progress.update(status='failed', error=repr(exc))
        write(output, progress)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],required=True)
    args=parser.parse_args()
    assert not args.output.exists(),'Preserve existing evaluation evidence'
    child(args.run,args.output,args.split,args.freeze)
