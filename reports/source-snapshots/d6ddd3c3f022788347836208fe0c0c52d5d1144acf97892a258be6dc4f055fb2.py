"""Bridge to the pinned upstream training code; run through autoresearch.py."""
import argparse
import json
import math
from pathlib import Path
import platform
import sys


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
        result = original_training(*positional, **kwargs)
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
