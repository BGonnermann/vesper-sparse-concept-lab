# moe-resume-20260914T231300Z

Outcome: **completed**. Budget family: **stage_or_unknown**.

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
Ran 22 tests in 3.478s
OK

tests-cuda.log:
Ran 6 tests in 3.246s
OK
```

## Retained narrative: comparison.md

# Preliminary dense/MoE feasibility comparison

Both correctness suites, both fresh smoke tests, and both baselines passed.

| Metric | Dense | MoE, four experts / top-1 |
| --- | ---: | ---: |
| Validation BPB | 0.628327 | 0.742139 |
| Total parameters | 26,345,772 | 47,588,652 |
| Active parameters | 26,345,772 | 26,354,988 |
| Training tokens including warmup | 8,945,664 | 3,850,240 |
| Wall seconds | 312.9 | 321.8 |
| Timed tokens/second | 29,205 | 12,224 |
| Peak allocated MiB | 434.9 | 729.3 |
| Peak reserved MiB | 530 | 798 |

Both models use depth 6, width 384, context 512 and 300 seconds of timed training.
MoE BPB was 18.11% higher with 1.81x total parameters and 56.96% fewer training tokens.
This initial result favors dense within this specific run budget. It is not evidence of an efficiency gain or a definitive architecture ranking.

Active parameters count all shared tensors, including complete embedding tables, plus one expert per layer and all router weights. This is not measured compute.
VRAM covers the PyTorch allocator. BPB excludes the routing auxiliary loss.

## Expert utilization

Percent of each layer's tokens; expert identities are independent between layers. Training aggregates the full run, including warmup.

| Layer | Train E0 / E1 / E2 / E3 | Validation E0 / E1 / E2 / E3 |
| --- | --- | --- |
| 1 | 31.7% / 31.4% / 22.6% / 14.3% | 29.9% / 30.3% / 23.6% / 16.3% |
| 2 | 20.0% / 30.3% / 20.7% / 29.0% | 18.8% / 31.4% / 24.6% / 25.2% |
| 3 | 23.3% / 24.1% / 28.8% / 23.8% | 29.7% / 25.2% / 23.9% / 21.2% |
| 4 | 23.9% / 18.4% / 35.9% / 21.8% | 22.4% / 24.5% / 26.3% / 26.8% |
| 5 | 24.2% / 29.1% / 20.0% / 26.6% | 33.2% / 22.8% / 21.6% / 22.4% |
| 6 | 37.6% / 24.9% / 12.7% / 24.8% | 36.6% / 18.7% / 14.8% / 30.0% |

Every expert was used in every layer. Zero dropped tokens in training and validation. All six routers recorded finite, nonzero gradients.
Unweighted training auxiliary loss: 1.063108; weighted contribution: 0.010631.

## Verification and preservation

22 CPU tests and 6 CUDA-suite tests passed. Review found no remaining concrete issues in the two fixes.
All 77 prior artifacts are unchanged. Protocol, data/tokenizer hashes, runtime, seed and executed source hashes match across the fresh pair and their smoke tests.
Captured source/configuration execution and live-edit isolation are covered by real subprocess tests. Each run retains snapshot.json, execution.json, optimizer/model/routing/training records, source files, weights and logs.

## Run locations

- dense_smoke: runs/autoresearch/20260914T231448Z-edc7928a
- moe_smoke: runs/autoresearch/20260914T231516Z-5599ce3e
- dense_baseline: runs/autoresearch/20260914T231604Z-061092d4
- moe_baseline: runs/autoresearch/20260914T232147Z-ff9e4e60

No additional experiments or cloud jobs were launched. All conclusions remain preliminary.
