# equal-token-20260915

Outcome: **completed**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
preflight.log:
test_failed_process_is_not_promoted_by_valid_looking_output (test_autoresearch.ResultsTests.test_failed_process_is_not_promoted_by_valid_looking_output) ... ok
test_protocol_rejects_partial_batches_and_unbounded_timeouts (test_autoresearch.ResultsTests.test_protocol_rejects_partial_batches_and_unbounded_timeouts) ... ok
test_timeout_terminates_child_and_preserves_partial_output (test_autoresearch.ResultsTests.test_timeout_terminates_child_and_preserves_partial_output) ... ok
test_windows_taskkill_timeout_still_kills_owned_child (test_autoresearch.ResultsTests.test_windows_taskkill_timeout_still_kills_owned_child) ... ok
ok
Ran 30 tests in 5.162s
OK
cuda correctness exit: 0
```

## Retained narrative: report.md

# Equal-token dense vs packed MoE: preliminary

Four captured-source runs; seeds 42 and 43; depth 6 / width 384; activation checkpointing off.
Every run completed 512 updates x 16,384 = **8,388,608 training tokens**, including all 11 timing-warmup updates.

| Seed | Model | Validation BPB | Timed tokens/s | Wall s | Allocated / reserved MiB |
|---|---|---:|---:|---:|---:|
| 42 | dense | 0.634848 | 49,388.8 | 176.55 | 581.69 / 622.00 |
| 42 | moe | 0.638883 | 24,639.2 | 349.66 | 902.07 / 956.00 |
| 43 | dense | 0.637774 | 49,502.0 | 175.83 | 581.69 / 622.00 |
| 43 | moe | 0.634744 | 24,995.5 | 344.17 | 904.45 / 956.00 |

## Paired BPB differences (MoE minus dense; negative favors MoE)
- Seed 42: +0.004035
- Seed 43: -0.003030
- Mean: +0.000503

## Protocol and evidence
- Same sealed TinyStories data/tokenizer, BF16 precision, context 512, microbatch 2, accumulation 16, full causal attention, optimizer hyperparameters and 65,536-token BPB evaluation.
- LR uses zero-based step i / 512, not a timer: no LR warmup; multiplier 1 through i=256, then 2*(1-i/512). Last multiplier is 1/256. The endpoint at i=512 is not an extra update.
- Muon momentum remains .85 + .1*min(i/300,1); Muon weight decay is .2*(1-i/512). Each schedule.json records all 512 entries and initial optimizer groups.
- The first 11 updates are measurement warmup, not additional training or LR warmup. Timed throughput uses exactly 8,208,384 tokens over the remaining 501 uninstrumented updates.
- A shared tape preserves the existing tokenizer and best-fit packing. A separate seeded CPU generator permutes its 8,192 microbatches; paired runs consumed identical indices and batch hash chains. Seeds 42/43 use different orders.
- Tape packing/hashing is preparation, outside reported timed throughput and run wall time. Run wall time includes startup, initialization hashing, tape loading, warmup, training, checkpoint save, evaluation and result checks. Timed updates include host-to-device transfers, forward/backward and optimizer work; no profiler.
- Full initialization hashes match the preflight for every run. Shared dense/MoE weights match within a seed; embeddings, routers and additional expert input weights change across seeds. Zero-initialized projections intentionally remain zero.
- Project/upstream modules execute from verified snapshots under isolated Python. Protocol, schedule, dataset/tokenizer hashes and actual batch receipts match across paired runs.
- Preserved all 560 previously recorded artifact files byte-for-byte.

## Parameters and utilization
- Dense: 26,345,772 total and active parameters. MoE: 47,588,652 total; 26,354,988 structural active parameters. No memory table. Active counts do not include dispatch overhead or all-expert optimizer work.
- MoE seed 42, train expert shares by layer (E0/E1/E2/E3), zero dropped tokens:
  Layer 0: 22.48% / 20.99% / 33.00% / 23.53%
  Layer 1: 26.03% / 24.62% / 27.05% / 22.30%
  Layer 2: 24.76% / 25.87% / 24.60% / 24.77%
  Layer 3: 23.51% / 22.47% / 24.12% / 29.89%
  Layer 4: 26.61% / 24.16% / 28.23% / 21.00%
  Layer 5: 26.59% / 26.65% / 17.16% / 29.59%
- MoE seed 43, train expert shares by layer (E0/E1/E2/E3), zero dropped tokens:
  Layer 0: 25.28% / 24.57% / 24.47% / 25.68%
  Layer 1: 24.62% / 22.84% / 26.02% / 26.53%
  Layer 2: 26.06% / 25.16% / 23.02% / 25.77%
  Layer 3: 25.83% / 22.55% / 30.54% / 21.08%
  Layer 4: 26.00% / 24.22% / 24.78% / 25.00%
  Layer 5: 22.23% / 13.38% / 30.09% / 34.30%

## Interpretation
Equal tokens are not equal total parameters or compute. MoE retains packing, per-layer host boundary synchronization, smaller expert GEMMs and optimizer work across all experts. This experiment does not separately profile those costs.
Two paired seeds provide preliminary evidence only, not a robust estimate of variance, generalization, or an efficiency gain. BPB excludes auxiliary routing loss. No further tuning, architecture features, training runs or cloud jobs were started.
