# checkpoint-20260915T010035Z

Outcome: **complete**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
report-checks.log:
Profile summary: [{'variant': 'dense', 'gain': 0.5998938144731729, 'on_tps': 29667.48299658156, 'off_tps': 47464.82253721887, 'max_gradient_error': 0.0}, {'variant': 'moe', 'gain': 0.771504550834121, 'on_tps': 12347.04557836565, 'off_tps': 21872.84743143106, 'max_gradient_error': 0.0}]

tests-cpu-run.log:
test_failed_process_is_not_promoted_by_valid_looking_output (test_autoresearch.ResultsTests.test_failed_process_is_not_promoted_by_valid_looking_output) ... ok
test_protocol_rejects_partial_batches_and_unbounded_timeouts (test_autoresearch.ResultsTests.test_protocol_rejects_partial_batches_and_unbounded_timeouts) ... ok
test_timeout_terminates_child_and_preserves_partial_output (test_autoresearch.ResultsTests.test_timeout_terminates_child_and_preserves_partial_output) ... ok
test_windows_taskkill_timeout_still_kills_owned_child (test_autoresearch.ResultsTests.test_windows_taskkill_timeout_still_kills_owned_child) ... ok
Ran 27 tests in 3.931s
OK

tests-cuda-run.log:
test_failed_process_is_not_promoted_by_valid_looking_output (test_autoresearch.ResultsTests.test_failed_process_is_not_promoted_by_valid_looking_output) ... ok
test_protocol_rejects_partial_batches_and_unbounded_timeouts (test_autoresearch.ResultsTests.test_protocol_rejects_partial_batches_and_unbounded_timeouts) ... ok
test_timeout_terminates_child_and_preserves_partial_output (test_autoresearch.ResultsTests.test_timeout_terminates_child_and_preserves_partial_output) ... ok
test_windows_taskkill_timeout_still_kills_owned_child (test_autoresearch.ResultsTests.test_windows_taskkill_timeout_still_kills_owned_child) ... ok
Ran 27 tests in 5.270s
OK

tests-red-retry.log:
ValueError: Protocol fields are missing or unknown.
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
AssertionError: Explicit checkpoint boolean must be supported: Protocol fields are missing or unknown.
Traceback (most recent call last):
    with self.assertRaises(ValueError):
AssertionError: ValueError not raised
FAILED (failures=3)
```

## Retained narrative: plan.md

# Controlled checkpointing experiment

Add strict boolean activation_checkpointing to captured protocol; default true; separate false protocol. No changes to microbatch 2, context 512, BF16, tokens/update 16384, depth6, model/optimizer, data or 300-second budget.
CPU/CUDA output and all-gradient agreement, plus saved depth6 checkpoint parity.
Three counterbalanced paired measurements per model, two updates per condition per pair and three warmup updates. Same weights/inputs, zero learning rates. No profiler in timings. Standalone peak-memory measurements, with the other model mode released.
Gate: all three off measurements faster and aggregate gain >=5% for each model; off peak reserved <75% physical VRAM and observed free VRAM >=25% physical VRAM. No OOM permitted. Profiling bounded to 180 seconds per model. No tuning/repeating failed timing gates.
If both pass: fresh checkpoint-off dense/MoE smoke gates then exactly one 300-second baseline each. Logs created and tail commands announced before every run. Up to three routine correctness repairs. Preserve all prior artifacts. Stop after report; no cloud or extra features.


## Retained narrative: report.md

# Activation checkpointing: controlled speed/memory experiment

## Outcome

Checkpointing off passed correctness, comfortable-memory and consistent-throughput gates for both dense and packed MoE. Exactly one fresh 300-second checkpoint-off baseline per model completed after fresh off smoke gates.

- **dense: +59.99% throughput**, all three pairs improved (58.31%-61.58%). Extra allocated memory: 146.42 MiB; extra reserved: 134.00 MiB. Minimum free VRAM sampled during the off memory pass: 12.68 GiB.
- **moe: +77.15% throughput**, all three pairs improved (71.04%-82.09%). Extra allocated memory: 172.66 MiB; extra reserved: 152.00 MiB. Minimum free VRAM sampled during the off memory pass: 13.69 GiB.

These are controlled **replay speed** results, not learning-quality claims. The fresh training comparison below is preliminary: one seed, unequal dense/MoE parameter budgets, and no fresh checkpoint-on baseline controls.

## Configuration and correctness

The captured protocol now requires a strict boolean `activation_checkpointing`. Default `experiments/autoresearch/protocol.json` remains true. `protocol-no-checkpoint.json` is false; every other protocol field is unchanged. The runner accepts `--protocol`, includes the complete protocol in smoke fingerprints/snapshots, and checks the actual model/training checkpoint flag against it. Old artifacts are untouched.

Both implementations remain depth 6/width 384, context 512, microbatch **2**, 16,384 tokens/update with 16 accumulation microbatches, BF16 expert/model execution and FP32 MoE router math. Attention, optimizer, routing/auxiliary loss, tokenizer/data and evaluation budgets are unchanged. No batch-size tuning.

**27 CPU and 27 CUDA tests passed.** On/off tests compare logits, loss, all parameter gradients and routing counts, including nonzero expert projections. Absolute tolerance 1e-6, relative 1e-5; no tolerance loosening. Existing tests cover causal behavior, empty experts, router task gradients, optimizer coverage and snapshot isolation.

Full depth-6/context-512 saved-checkpoint comparisons also passed. Maximum gradient absolute differences: dense **0.0**, MoE **0.0**. Checkpoint tensors were unchanged after profiling. Independent review found no concrete configuration/correctness issue.

The first log could not be opened for writing during two logging attempts. Those attempts were not accepted as test evidence. Fresh log writers were held open before announcements, then a start marker released each child. The subsequent expected test-first configuration failure and all passing results are preserved.

## Paired uninstrumented profiling

Same saved weights and replay batches within each model's on/off comparison. Three warmup updates per condition, then exactly **three paired measurements of two updates/condition**: 98,304 timed tokens per condition. Order on/off, off/on, on/off. No profiler or extra measurement hooks in timed updates; learning rates zero, full optimizer work retained.

| Model | Checkpointing | Replay tokens/s | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|
| dense | on | 29,667.5 | 435.14 | 484.00 |
| dense | off | 47,464.8 | 581.56 | 618.00 |
| moe | on | 12,347.0 | 723.90 | 796.00 |
| moe | off | 21,872.8 | 896.55 | 948.00 |

Memory came from separate standalone passes: one resident model/optimizer, unused allocator cache released before each condition, three warmup and two memory updates. Cache clearing and memory instrumentation were excluded from reported throughput. Peak statistics cover each whole memory pass. Reserved memory therefore does not simply carry over from off into on.

The gate required all three paired off results faster, aggregate gain >=5%, off reserved memory <75% of physical VRAM, and free VRAM >=25% at update boundaries. Both passed with wide headroom. CUDA allocator numbers exclude other processes/driver memory. Background allocations changed during these runs, so free-memory differences are not attributed solely to checkpointing.

Replay includes transfer, forward, backward, optimizer and gradient clearing, but omits CPU tokenization/packing. Raw paired samples, frozen-weight checks, identical routing counts, minimum free-memory readings and captured protocols are in each model's probe result.

## Fresh checkpoint-off training

Both fresh smoke gates passed. Actual captured model/training receipts confirm checkpointing **false**. Protocol, seed 42, data/tokenizer hashes, runtime and executed source identities match across dense/MoE runs. Only their established architecture-specific candidate fields differ. Other settings match the prior protocol exactly.

| Metric | Dense off | Packed MoE off |
|---|---:|---:|
| Validation BPB | 0.584336 | 0.650426 |
| Timed tokens/s | 47,658.5 | 25,413.0 |
| Total tokens including warmup | 14,483,456 | 7,815,168 |
| Timed tokens excluding warmup | 14,303,232 | 7,634,944 |
| Optimizer updates | 884 | 477 |
| Timed seconds | 300.119 | 300.435 |
| Launch-to-record wall seconds | 309.625 | 323.750 |
| Peak allocated MiB | 581.69 | 902.83 |
| Peak reserved MiB | 622.00 | 956.00 |

Dense parameters: total/active **26,345,772/26,345,772**. MoE parameters: **47,588,652/26,354,988**. No memory tables.

Timed throughput excludes the first 11 warmup updates; total tokens include them. The protocol targets 300 seconds and stops at an update boundary, so actual timed seconds can slightly exceed 300. Wall seconds exclude log-announcement waiting and snapshot preparation. BPB contains cross-entropy only, not routing auxiliary loss.

### Expert utilization

Share of tokens routed to experts 0 / 1 / 2 / 3:

| Layer | Training | Validation |
|---|---|---|
| 0 | 31.11% / 30.78% / 20.60% / 17.51% | 27.69% / 25.48% / 21.06% / 25.77% |
| 1 | 19.78% / 29.43% / 22.63% / 28.16% | 21.64% / 27.98% / 24.17% / 26.21% |
| 2 | 24.75% / 24.11% / 25.12% / 26.01% | 27.36% / 22.90% / 25.46% / 24.28% |
| 3 | 21.58% / 22.28% / 31.56% / 24.58% | 16.49% / 30.39% / 23.76% / 29.36% |
| 4 | 23.63% / 26.30% / 22.57% / 27.50% | 28.45% / 19.02% / 25.13% / 27.40% |
| 5 | 33.17% / 23.49% / 13.62% / 29.72% | 32.85% / 19.32% / 16.26% / 31.57% |

Zero dropped tokens in training/validation; all routers have finite nonzero gradients. Dense has no experts. MoE mean training auxiliary loss **1.042084**, weighted contribution **0.010421**, recorded separately from validation BPB.

## Interpretation and limits

Turning checkpointing off retains activations instead of recomputing blocks during backward. Here the extra memory was modest relative to GPU capacity and saved substantial repeated attention/MLP/dispatch work. Packed MoE still pays for four smaller expert GEMMs, routing/dispatch and optimizer work across all expert matrices. Similar active parameter counts do not imply equal training cost.

The paired measurements support a speed/memory tradeoff on this GPU at this exact batch/context. Three pairs, uncontrolled clocks/background load, fixed saved-weight replay and one fresh training seed do not establish general performance or a statistically robust quality gain. Previous checkpoint-on baseline BPBs are historical, not contemporaneous controls. The training table must not be conflated with the frozen-weight profiling table.

## Preservation and stop

**402 prior files verified unchanged**, including all earlier depth, MoE, packed-dispatch and profiling artifacts. No batch-size tuning, new architecture features or cloud jobs. Stopped after this report.

Artifacts in this directory: `comparison.json`, `probe-manifest.json`, `dense-probe-result.json`, `moe-probe-result.json`, paired sample logs, captured `source/`, on/off protocols, tests and preservation receipts. Reproducible orchestration: `checkpoint_probe.py`, `arm_probe.py`, `gated_trial.py`, `final_report.py`.

Fresh training logs/results: `runs\autoresearch\checkpoint-20260915T010035Z-dense-baseline` and `runs\autoresearch\checkpoint-20260915T010035Z-moe-baseline`. Both retain captured source/configuration identities, memory/routing reports and weights-only checkpoints. Every child run's log and exact PowerShell tail command were made available before it started.
