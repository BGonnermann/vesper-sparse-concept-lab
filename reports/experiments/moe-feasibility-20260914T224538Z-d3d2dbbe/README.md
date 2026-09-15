# moe-feasibility-20260914T224538Z

Outcome: **stopped**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
cuda-dtype-diagnostic-retry1.log:
  "note": "Diagnosis only; no trial or optimizer update. Correctness suite remains failed."

cuda-dtype-diagnostic.log:
Traceback (most recent call last):
RuntimeError: Float did not match BFloat16

tests-cpu.log:
test_failed_process_is_not_promoted_by_valid_looking_output (test_autoresearch.ResultsTests.test_failed_process_is_not_promoted_by_valid_looking_output) ... ok
test_protocol_rejects_partial_batches_and_unbounded_timeouts (test_autoresearch.ResultsTests.test_protocol_rejects_partial_batches_and_unbounded_timeouts) ... ok
test_timeout_terminates_child_and_preserves_partial_output (test_autoresearch.ResultsTests.test_timeout_terminates_child_and_preserves_partial_output) ... ok
test_windows_taskkill_timeout_still_kills_owned_child (test_autoresearch.ResultsTests.test_windows_taskkill_timeout_still_kills_owned_child) ... ok
Ran 18 tests in 2.822s
OK

tests-cuda.log:
Traceback (most recent call last):
    raise error_metas[0].to_error(msg)
AssertionError: The values for attribute 'dtype' do not match: torch.float32 != torch.bfloat16.
Ran 6 tests in 2.989s
FAILED (failures=1)

tests-red.log:
Traceback (most recent call last):
    raise AssertionError("MoE model adapter has not been implemented")
AssertionError: MoE model adapter has not been implemented
ERROR: test_selectable_variants_and_unknown_options_are_rejected (__main__.VariantConfigTests.test_selectable_variants_and_unknown_options_are_rejected)
Traceback (most recent call last):
    raise ValueError("Candidate must contain only depth and matrix_lr.")
ValueError: Candidate must contain only depth and matrix_lr.
FAILED (errors=2)
```

## Retained narrative: stopped.md

# Stopped at the CUDA correctness gate

Implemented a local four-expert top-1 dropless feedforward model, probability
gating, explicit load balancing, router AdamW, optimizer identity coverage,
checkpoint-safe routing counts, separate validation cross-entropy, selectable
depth-6 configs, and version-2 runner records/source hashes and smoke gates.
Pinned upstream and protocol were not edited. No NCP or n-gram memory added.

## Test evidence

- tests-red.log: expected pre-implementation failures, missing adapter/schema.
- tests-cpu.log: 18 tests passed, including all existing runner tests.
- tests-cuda.log: 5 of 6 tests passed; the independent weighted-output reference
  in tests/test_moe.py:116 had BF16 dtype while the production output was FP32.
  Production preserves input dtype and multiplies by FP32 vector gate weights.
  The test used autocast probabilities and scalar multiplication, retaining BF16.
- cuda-dtype-diagnostic.log: first diagnostic also encountered scalar dtype
  promotion; its failure log is retained.
- cuda-dtype-diagnostic-retry1.log: corrected FP32 vector-weighted reference
  matched production exactly, max absolute difference 0.0; routing [1,1,1,1];
  task router gradient absolute sum 0.0030932407826185226, finite and nonzero.
  This diagnostic does not turn the failed suite into a passing suite.

## Review finding and remaining work

The runner saves source snapshots but launches live scripts, leaving a race
between recorded source hashes and executed code. Before training, launch from
verified snapshots and cover this behavior with a runner test. Add explicit
variant/model-source gate and malformed routing-record regression tests.

Correct the CUDA test reference precision and weighting shape, then rerun CPU
and CUDA suites. No correction or rerun of the failed suite was attempted after
the user's stop-on-failure boundary. No smoke or baseline trial was launched.
The requested dense/MoE comparison remains unexecuted and no BPB, training time,
throughput or memory comparison is claimed.

After a separately authorized continuation clears correctness, the remaining
scope is fresh dense and MoE smoke tests, then exactly one 300-second timed
baseline each, live-log commands, and a matched-settings comparison. Do not
reuse old dense smoke gates or infer efficiency from unequal parameter budgets.

## Preservation

prior-preservation.json lists hashes for all 57 prior files.
preservation-check.json confirms all unchanged, including depth-4/6/8 results,
logs, weights, configurations and earlier comparison receipts. Current changed
code/config/test sources are archived under source/ with source-manifest.json.
The global candidate is now depth 6 dense; explicit dense-depth6.json and
moe-depth6.json candidates are available. No cloud jobs or extra experiments.
