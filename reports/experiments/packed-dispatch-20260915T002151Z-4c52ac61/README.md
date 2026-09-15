# packed-dispatch-20260915T002151Z

Outcome: **complete**. Budget family: **wall_time_budget**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
tests-cpu.log:
test_failed_process_is_not_promoted_by_valid_looking_output (test_autoresearch.ResultsTests.test_failed_process_is_not_promoted_by_valid_looking_output) ... ok
test_protocol_rejects_partial_batches_and_unbounded_timeouts (test_autoresearch.ResultsTests.test_protocol_rejects_partial_batches_and_unbounded_timeouts) ... ok
test_timeout_terminates_child_and_preserves_partial_output (test_autoresearch.ResultsTests.test_timeout_terminates_child_and_preserves_partial_output) ... ok
test_windows_taskkill_timeout_still_kills_owned_child (test_autoresearch.ResultsTests.test_windows_taskkill_timeout_still_kills_owned_child) ... ok
Ran 24 tests in 2.649s
OK

tests-cuda.log:
Ran 8 tests in 2.343s
OK

tests-red.log:
Traceback (most recent call last):
AssertionError: 4 != 1 : Experts must receive slices of one packed allocation
FAILED (failures=1)
```

## Retained narrative: plan.md

# Packed dispatch experiment

Preserve architecture, four experts/top-1 assignment, router probabilities, auxiliary loss, precision, protocol and all old artifacts.
First prove packing allocation regression fails before modifying production. Then run full CPU and CUDA correctness suites, including original algorithm output/all-gradient parity, uneven/empty/tied routing and checkpoint replay.
Bounded performance gate: three warmup updates per implementation; six counterbalanced paired blocks of two uninstrumented updates each. Same saved MoE checkpoint and batch bank, zero learning rates. Require every paired block faster and aggregate speedup at least 5 percent. Trace separately: two microbatches and one optimizer per implementation.
Only if correctness and timing gates pass: fresh dense and packed MoE smoke gates, then exactly one 300-second baseline each. Create and announce every log before its run. Stop on a failed gate, diagnose, preserve artifacts, report. No added features or cloud. Up to three routine correctness repair iterations; do not tune architecture or rerun a failed performance gate.


## Retained narrative: report.md

# Packed expert dispatch: correctness, profiling and training

## Outcome

Packed dispatch passed correctness and the preregistered timing gate: **35.09% higher replay throughput**, faster in all six paired blocks (gains 29.88%-41.85%). Fresh dense and optimized MoE smoke gates passed, followed by exactly one 300-second baseline each.

This is an implementation speedup, **not proof that MoE is more efficient than dense**. Dense and MoE have unequal total parameter budgets and only one fresh seed/run each. Training quality conclusions are preliminary.

## Implementation and correctness

Only production feedforward dispatch and its recording label changed in `scripts/autoresearch_model.py`. Stable sorting packs tokens once, four expert calls consume contiguous slices (including empty slices), and one index-copy restores original order. A fixed-size histogram and one boundary transfer replace repeated dynamic nonzero/bincount paths.

Architecture, attention, expert selection/tie behavior, within-expert row order, full-softmax gate weights, FP32 router/auxiliary math, BF16 expert execution, output cast, auxiliary coefficient, optimizer policy, tokenizer, data and training settings remain unchanged.

- Test-first allocation regression failed as expected on original dispatch, then passed.
- **24 CPU tests and 8 CUDA tests passed.** New cases cover balanced, uneven, empty and tied routing; FP32 and CUDA BF16 inputs; outputs, auxiliary loss, every input/parameter gradient, finite nonzero router task gradients, and checkpoint replay.
- Original-reference comparison tolerance: absolute 1e-6, relative 1e-5; no tolerance loosening.
- Full saved depth-6 checkpoint preflight: **maximum absolute gradient difference 0.0**, matching loss and routing counts.
- Existing tests also cover causality, validation excluding auxiliary loss, optimizer coverage, empty-expert tensor gradients/updates, accounting and snapshot isolation.
- Independent read-only review: no concrete correctness findings; residual host boundary transfer and small per-expert GEMMs noted.

## Profiling results, not training results

Original and optimized MoE loaded the **same saved checkpoint**, with identical replay inputs, context 512, microbatch 2, 16 accumulation microbatches, BF16 and checkpointing on. All learning rates were zero; checkpoint tensors were verified unchanged.

Three warmup updates/implementation; six counterbalanced paired blocks of two updates (12 timed updates, **196,608 timed tokens/implementation**). Gate required improvement in all six pairs and at least 5% aggregate throughput gain. Profiler, phase timing instrumentation and additional module hooks were absent from throughput measurements.

| Uninstrumented replay | Original MoE | Packed MoE |
|---|---:|---:|
| Tokens/s | 12,298.9 | 16,614.5 |
| Total timed seconds | 15.9858 | 11.8336 |
| Mean update ms | 1332.15 | 986.13 |

Separate trace: two microbatches plus one optimizer step per implementation:

| Trace observation | Original | Packed |
|---|---:|---:|
| Actual kernel launches | 6,416 | 4,988 |
| Stream synchronizations | 144 | 24 |
| Stream sync duration, ms | 5.125 | 0.122 |
| Nonzero calls | 96 | 0 |
| Bincount calls | 24 | 0 |

Separate synchronized phase means (three updates each), ms/update:

| Phase | Original | Packed |
|---|---:|---:|
| data | 1.09 | 1.18 |
| forward | 440.81 | 299.30 |
| backward | 874.27 | 651.42 |
| optimizer | 19.10 | 19.78 |
| zero_grad | 0.26 | 0.24 |

These phase/trace passes contain additional instrumentation/synchronization and **do not define throughput**. Replay omits CPU tokenization/packing; transfer is included. Both models and warmed optimizers were resident together: raw joint allocator peaks and increments are in `probe-result.json`, not misreported as standalone model VRAM. The training table below provides standalone VRAM.

Original and packed probe routing counts match exactly, all tokens are covered and both checkpoints remain unchanged. Per-expert utilization for both identical replay streams is preserved in `probe-result.json`.

## Fresh training results

Both: depth 6, width 384, context 512, 16,384 tokens/update, microbatch 2, checkpointing enabled, seed 42, matrix LR 0.04, existing **300-second** training protocol. MoE: four experts/top-1, auxiliary coefficient 0.01, router LR 0.001. No NCP or memory tables.

Protocol, data/tokenizer hashes, runtime, seed, runner/adapter/model/bootstrap hashes and actual executed source snapshots match across fresh runs. Candidate differences are only the established dense/MoE architecture fields. Each run executed its captured snapshot; smoke gates match the baseline identities.

| Metric | Fresh dense | Fresh packed MoE |
|---|---:|---:|
| Validation BPB | 0.630091 | 0.706799 |
| Timed training tokens/s | 28,897.2 | 15,460.2 |
| Total tokens including warmup | 8,863,744 | 4,833,280 |
| Timed tokens excluding warmup | 8,683,520 | 4,653,056 |
| Optimizer updates | 541 | 295 |
| Timed training seconds | 300.497 | 300.969 |
| Launch-to-record wall seconds | 312.797 | 319.031 |
| Peak allocated VRAM, MiB | 434.89 | 729.39 |
| Peak reserved VRAM, MiB | 530.00 | 798.00 |
| total_parameters | 26,345,772 | 47,588,652 |
| active_parameters | 26,345,772 | 26,354,988 |

BPB is token cross-entropy only, excluding routing auxiliary loss. Total tokens include 11 warmup updates; timed throughput uses only timed tokens/time. Wall time covers launch through result validation, excluding separate snapshot preparation. VRAM is PyTorch allocator peak, not total device usage or host RAM.

### Expert utilization

Percent of tokens assigned to experts 0 / 1 / 2 / 3, separately by layer:

| Layer | Training | Validation |
|---|---|---|
| 0 | 33.16% / 31.32% / 21.31% / 14.20% | 32.22% / 30.79% / 20.96% / 16.03% |
| 1 | 20.69% / 29.19% / 21.67% / 28.45% | 20.40% / 27.49% / 25.10% / 27.01% |
| 2 | 23.70% / 25.10% / 26.73% / 24.47% | 31.47% / 26.35% / 20.68% / 21.51% |
| 3 | 24.55% / 19.02% / 33.87% / 22.55% | 21.26% / 27.19% / 25.11% / 26.44% |
| 4 | 23.75% / 28.10% / 21.66% / 26.49% | 30.94% / 22.54% / 23.29% / 23.23% |
| 5 | 36.49% / 24.35% / 12.78% / 26.38% | 36.28% / 18.10% / 16.11% / 29.51% |

Zero dropped tokens in training and validation; every router gradient finite/nonzero. Training mean auxiliary loss: **1.056940**, weighted contribution **0.010569**, recorded separately from BPB. Dense has no experts.

## Remaining bottlenecks and interpretation

Backward, including full-block activation-checkpoint recomputation, remains the largest packed phase. Packing removed repeated dynamic dispatch paths but retained one host boundary transfer per layer invocation, four Python expert calls and small expert GEMMs. Checkpointing repeats that work. Optimizer work across all 48 expert matrices remains; active-parameter counts do not describe that cost.

No skipped experts/tokens, reduced precision, reduced context, altered training budget or changed architecture produced the measured speedup. Further improvements would need their own authorization and correctness/timing gates. No further optimization is included here.

Short sequential timing samples and baseline results can vary with background GPU load/clocks. Saved-weight replay does not measure learning quality; the fresh baseline table does. Historical original-MoE BPB is not a paired learning-quality control for this run.

## Preservation and artifacts

**263 prior files verified unchanged**, including depth-4/6/8 results, the first MoE campaign and all earlier profiling artifacts.

- `tests-red.log`, `tests-cpu.log`, `tests-cuda.log` and archived original/test source.
- `probe.log`, `probe-manifest.json`, `probe-result.json`, captured `source/` and `batches.pt`.
- `original-trace.json`, `packed-trace.json` and operator summaries.
- `baseline-matching.json` and `comparison.json`: matched identities and complete metrics.
- Dense baseline: `runs\autoresearch\packed-20260915T002151Z-dense-baseline`.
- Packed baseline: `runs\autoresearch\packed-20260915T002151Z-moe-baseline`.
- Each smoke/baseline retains `run.log`, result/config/source receipts, memory, routing and checkpoint artifacts.
- `paired_probe.py`, `gated_trial.py` and `final_report.py` reproduce the bounded procedure. Source SHA256s are authoritative; Git metadata was deliberately not queried.

All run logs were created and their exact PowerShell tail commands provided before launch. **Stopped after this report: no further architecture features, experiments or cloud jobs.**
