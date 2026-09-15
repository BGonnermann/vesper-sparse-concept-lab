"""Bridge to the pinned upstream training code; run through autoresearch.py."""
import argparse
import json
import math
from pathlib import Path
import platform
import sys


def fixed_schedule(updates=512):
    """Original optimizer policy expressed against consumed update budget."""
    return [dict(step=i, progress=i / updates,
                 lr_multiplier=min(1.0, 2 * (1 - i / updates)),
                 muon_momentum=.85 + .1 * min(i / 300, 1),
                 muon_weight_decay=.2 * (1 - i / updates)) for i in range(updates)]


def batch_order(count, seed):
    import torch
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()


def fixed_training(train, protocol, runtime, tokenizer, config, device_batch_size, smoke_test):
    import hashlib
    import time
    import torch
    from autoresearch import digest, write_json
    if smoke_test:
        raise ValueError("Fixed-update runs use the recorded correctness preflight, not extra smoke training.")
    t_start = time.time()
    seed = protocol["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    with torch.device("meta"):
        model = train.GPT(config)
    model.to_empty(device=runtime.device)
    model.init_weights(embed_dtype=runtime.amp_dtype)
    initial = {name: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
               for name, p in model.named_parameters()}
    write_json(Path("initialization.json"), {"seed": seed, "extra_expert_seed": 4200 + seed,
                                           "parameters": initial})
    optimizer = model.setup_optimizer(unembedding_lr=train.UNEMBEDDING_LR,
        embedding_lr=train.EMBEDDING_LR, scalar_lr=train.SCALAR_LR, adam_betas=train.ADAM_BETAS,
        matrix_lr=train.MATRIX_LR, weight_decay=train.WEIGHT_DECAY)
    tape_path = Path(protocol["batch_tape"])
    if digest(tape_path) != protocol["batch_tape_sha256"]:
        raise ValueError("Batch tape hash changed.")
    tape = torch.load(tape_path, map_location="cpu", weights_only=True)
    accumulation = protocol["tokens_per_update"] // (device_batch_size * protocol["sequence_length"])
    count = protocol["optimizer_updates"] * accumulation
    if tuple(tape.shape) != (count, 2, device_batch_size, protocol["sequence_length"]) or tape.dtype != torch.int64:
        raise ValueError("Batch tape shape/dtype does not match the exact training budget.")
    order = batch_order(count, seed)
    schedule = fixed_schedule(protocol["optimizer_updates"])
    write_json(Path("schedule.json"), {"definition": protocol["schedule"], "updates": schedule,
                                      "initial_optimizer_groups": [{k:v for k,v in g.items() if k != "params"} for g in optimizer.param_groups]})
    consumed = hashlib.sha256()
    # Hash the CPU tensors handed to the transfer path, outside timed GPU updates.
    batch_hashes = [hashlib.sha256(tape[i].numpy().tobytes()).hexdigest() for i in range(count)]
    device_buffer = torch.empty(tape.shape[1:], dtype=tape.dtype, device=runtime.device)
    tape = tape.pin_memory()
    model.train()
    timed = 0.0
    all_seconds = 0.0
    consumed_indices = []
    t_start_training = time.time()
    for item in schedule:
        step = item["step"]
        indices = order[step * accumulation:(step + 1) * accumulation]
        for index in indices:
            consumed.update(bytes.fromhex(batch_hashes[index]))
            consumed_indices.append(index)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for index in indices:
            device_buffer.copy_(tape[index], non_blocking=True)
            x, y = device_buffer.unbind(0)
            with torch.autocast(runtime.device_type, dtype=runtime.amp_dtype):
                loss = model(x, y)
            (loss / accumulation).backward()
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * item["lr_multiplier"]
            if group["kind"] == "muon":
                group["momentum"] = item["muon_momentum"]
                group["weight_decay"] = item["muon_weight_decay"]
        optimizer.step()
        model.zero_grad(set_to_none=True)
        value = loss.item()
        if not math.isfinite(value) or value > 100:
            raise RuntimeError(f"Invalid training loss at step {step}: {value}")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        all_seconds += elapsed
        if step >= 11:
            timed += elapsed
        print(f"step {step + 1}/512 | loss: {value:.6f} | lr_mult: {item['lr_multiplier']:.8f} | dt: {elapsed:.4f}s", flush=True)
    if len(consumed_indices) != count or len(set(consumed_indices)) != count:
        raise RuntimeError("Fixed-update batch coverage failed.")
    write_json(Path("batches.json"), {"seed": seed, "policy": "seeded permutation of fixed prepacked microbatches",
        "batch_tape_sha256": protocol["batch_tape_sha256"], "consumed_indices": consumed_indices,
        "consumed_batch_hash_chain": consumed.hexdigest(), "microbatches": count,
        "training_tokens": count * device_batch_size * protocol["sequence_length"]})
    write_json(Path("fixed-training.json"), {"seed": seed, "stopping_rule": "optimizer_updates",
        "optimizer_updates": len(schedule), "all_update_seconds": all_seconds,
        "schedule_sha256": digest(Path("schedule.json")), "batch_hash_chain": consumed.hexdigest()})
    return dict(model=model, num_params=model.num_scaling_params()["total"],
                num_flops_per_token=model.estimate_flops(), total_training_time=timed,
                step=len(schedule), t_start=t_start, t_start_training=t_start_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    import torch

    info = {"python": sys.version, "os": platform.platform(), "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        free, total = torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                    free_vram_bytes=free, total_vram_bytes=total,
                    bf16_supported=torch.cuda.is_bf16_supported())
    print(json.dumps(info, indent=2), flush=True)
    if not info["cuda_available"]:
        print("CUDA is unavailable. Check the NVIDIA driver and installed PyTorch build.")
        return 1
    if args.doctor:
        parameter = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
        optimizer = torch.optim.AdamW([parameter])
        dtype = torch.bfloat16 if info["bf16_supported"] else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            loss = (parameter @ parameter.T).float().square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError("Nonfinite values during GPU probe.")
        print("PASS: CUDA matrix multiplication, backward, and optimizer step.")
        return 0

    if not all((args.runtime, args.candidate, args.protocol)):
        parser.error("runtime, candidate, and protocol are required for training")
    candidate = json.loads(args.candidate.read_text())
    protocol = json.loads(args.protocol.read_text())
    Path("environment.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(args.runtime.resolve()))
    import prepare
    # Patch before importing train: it imports these constants by value.
    prepare.MAX_SEQ_LEN = protocol["sequence_length"]
    prepare.EVAL_TOKENS = protocol["eval_tokens"]
    prepare.TIME_BUDGET = protocol["training_seconds"]
    import train
    from autoresearch import validate_candidate, validate_protocol, write_json
    from autoresearch_model import model_class
    validate_candidate(candidate)
    validate_protocol(protocol)
    train.GPT = model_class(train, candidate)
    print(f"Feedforward variant: {candidate['feedforward']}", flush=True)
    print(f"Activation checkpointing: {protocol['activation_checkpointing']}", flush=True)
    print("Training loss includes configured routing auxiliary loss; validation BPB is cross-entropy only.", flush=True)
    observed = {}
    original_training = train._run_training_once

    def training(*positional, **kwargs):
        result = (fixed_training(train, protocol, *positional, **kwargs)
                  if protocol.get("stopping_rule") == "optimizer_updates" else original_training(*positional, **kwargs))
        model = result["model"]
        if model.config.use_activation_checkpointing is not protocol["activation_checkpointing"]:
            raise RuntimeError("Runtime checkpointing differs from captured protocol.")
        observed["model"] = model
        write_json(Path("model.json"), model.parameter_report())
        write_json(Path("optimizer.json"), model.optimizer_report)
        write_json(Path("routing.json"), model.routing_report())
        write_json(Path("training.json"), {
            "activation_checkpointing": model.config.use_activation_checkpointing,
            "optimizer_updates": result["step"],
            "training_tokens": result["step"] * protocol["tokens_per_update"],
            "warmup_updates": min(result["step"], 11),
            "timed_training_seconds": result["total_training_time"],
            "timed_training_tokens": max(result["step"] - 11, 0) * protocol["tokens_per_update"],
        })
        return result

    train._run_training_once = training
    train.DEPTH = candidate["depth"]
    train.MATRIX_LR = candidate["matrix_lr"]
    train.TOTAL_BATCH_SIZE = protocol["tokens_per_update"]
    train.WINDOW_PATTERN = "L"
    batch = protocol["microbatch_size"]
    # Upstream batch constants do not constrain GPU-profile/autotune candidates.
    train._build_train_candidates = lambda runtime: [(batch, protocol["activation_checkpointing"])]
    train._autotune_train_candidate = lambda *args: None
    train._build_eval_batch_candidates = lambda *args: [protocol["eval_batch_size"]]
    # Avoid presenting the fork's approximate hardware MFU as our measurement.
    train._get_gpu_peak_flops = lambda name: None
    original_evaluate = prepare.evaluate_bpb

    def evaluate(model, tokenizer, batch_size, **kwargs):
        value = original_evaluate(model, tokenizer, protocol["eval_batch_size"],
                                  device=kwargs["device"], dataset=kwargs["dataset"],
                                  eval_tokens=protocol["smoke_eval_tokens"] if args.smoke_test else protocol["eval_tokens"])
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite validation BPB.")
        write_json(Path("routing.json"), model.routing_report())
        return value

    train.evaluate_bpb = evaluate
    sys.argv = ["train.py", "--dataset", "tinystories"] + (["--smoke-test"] if args.smoke_test else [])
    torch.cuda.reset_peak_memory_stats()
    status = train.main()
    Path("memory.json").write_text(json.dumps({
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "measurement_scope": "PyTorch allocator; excludes other processes and driver allocations"
    }, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
