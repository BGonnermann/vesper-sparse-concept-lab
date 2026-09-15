"""Synthetic full-context correctness and fit gates; no corpus training."""
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

import autoresearch as r
from autoresearch_model import model_class

HERE = r.ROOT / 'runs/autoresearch/ncp-20260915'
from ncp_campaign import candidate as make_candidate
sys.path.insert(0, str(r.RUNTIME))
import train


def main():
    stamp=str(time.time_ns())
    source_path=HERE/f'fit-{stamp}.py'
    source_path.write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    runtime = train.detect_runtime(); train._configure_step_kernels(runtime)
    train.MAX_SEQ_LEN = 512; train.WINDOW_PATTERN = 'L'
    results = {}
    for label in ('D6','NCP','AUX','D12','N-RMS'):
        candidate = make_candidate(label)
        depth=candidate['depth']
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        config = train.build_model_config(depth, 8192, runtime, False)
        with torch.device('meta'):
            model = model_class(train, candidate)(config)
        model.to_empty(device=runtime.device); model.init_weights(embed_dtype=runtime.amp_dtype)
        # Exercise active attention and feedforward paths in the causal fixture.
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=.01)
                block.mlp.c_proj.weight.normal_(std=.01)
        optimizer = model.setup_optimizer(unembedding_lr=train.UNEMBEDDING_LR,
            embedding_lr=train.EMBEDDING_LR, scalar_lr=train.SCALAR_LR,
            adam_betas=train.ADAM_BETAS, matrix_lr=.04, weight_decay=train.WEIGHT_DECAY)
        assert model.optimizer_report['verified']
        x = torch.randint(0, 8192, (2,512), device=runtime.device)
        y = torch.randint(0, 8192, (2,512), device=runtime.device)
        changed = x.clone(); changed[:,256:] = (changed[:,256:] + 1) % 8192
        model.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=runtime.amp_dtype):
            logits = model(x); future_changed = model(changed)
            assert logits.shape == (2,512,8192)
            torch.testing.assert_close(logits[:,:256], future_changed[:,:256], atol=1e-6, rtol=1e-5)
            ce = F.cross_entropy(logits.flatten(0,1), y.flatten(), reduction='none')
            torch.testing.assert_close(model(x,y,reduction='none').flatten(), ce, atol=1e-6, rtol=1e-5)
        del logits, future_changed, ce, changed
        model.train()
        with torch.autocast('cuda', dtype=runtime.amp_dtype):
            loss = model(x,y)
        loss.backward()
        gradients = {n:float(p.grad.detach().float().norm()) for n,p in model.named_parameters() if p.grad is not None}
        assert len(gradients) == len(list(model.parameters()))
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters())
        assert any(value > 0 for value in gradients.values())
        model.zero_grad(set_to_none=True)
        timings = []
        for step in range(6):
            torch.cuda.synchronize(); start = time.perf_counter()
            for _ in range(16):
                with torch.autocast('cuda', dtype=runtime.amp_dtype):
                    loss = model(x,y)
                (loss/16).backward()
            optimizer.step(); model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            assert torch.isfinite(loss).item()
            if step >= 3:
                timings.append(elapsed)
            print(f'depth {depth} synthetic update {step+1}/6: {elapsed:.4f}s', flush=True)
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        allocated = torch.cuda.max_memory_allocated(); reserved = torch.cuda.max_memory_reserved()
        free,total = torch.cuda.mem_get_info()
        forecast = max(timings)*512 + 45
        assert free >= 2*2**30 and reserved < .75*total, 'Insufficient comfortable VRAM margin'
        assert forecast < 900, 'Conservative runtime forecast exceeds existing timeout'
        model.eval()
        checkpoint = io.BytesIO()
        with torch.no_grad(), torch.autocast('cuda', dtype=runtime.amp_dtype):
            expected = model(x).clone()
            torch.save(model.state_dict(), checkpoint)
            parameter = next(model.parameters()); parameter.add_(1)
            assert not torch.equal(model(x), expected), 'Roundtrip mutation did not affect output'
            checkpoint.seek(0)
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
            torch.testing.assert_close(model(x), expected, atol=1e-6, rtol=1e-5)
        results[label] = dict(passed=True, candidate=candidate, model=model.parameter_report(),
            optimizer_coverage=model.optimizer_report, gradient_norms=gradients,
            measured_update_seconds=timings, conservative_trial_forecast_seconds=forecast,
            peak_allocated_bytes=allocated, peak_reserved_bytes=reserved, free_bytes=free,
            synthetic_optimizer_updates=6, synthetic_update_tokens=98304,
            corpus_training_tokens=0, checkpoint_sha256=hashlib.sha256(checkpoint.getvalue()).hexdigest(),
            checkpoint_storage='In-memory synthetic roundtrip; full research checkpoints are retained on disk',
            interpretation='Randomized synthetic correctness/fit fixture; not a trained research candidate or quality score')
        print(f'PASS depth {depth}: params={model.parameter_report()["total_parameters"]}, forecast={forecast:.1f}s, reserved={reserved/2**20:.1f}MiB',flush=True)
        del model, optimizer, expected, x, y, loss, parameter
    receipt=dict(kind='synthetic_correctness_fit',status='completed',artifact_file=f'fit-{stamp}-result.json',
        results=results,source_file=source_path.name,source_sha256=r.digest(source_path),gpu=torch.cuda.get_device_name(0))
    r.write_json(HERE/receipt['artifact_file'],receipt)
    r.write_json(HERE/'fit-result.json',receipt)


if __name__ == '__main__':
    main()
