# profile-20260914T233639Z/retry

Outcome: **complete**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Retained narrative: plan.md

# Bounded profiling retry
Authorized: fix torch.dtype result serialization; finish dense/MoE profiling and report.
Changes: serialize config dtype as its explicit string; use fresh retry directory.
Preserve original failed script, trace, batches, logs and all prior training artifacts.
Per model: 3 warmups, 6 uninstrumented replay updates, 3 phase-decomposition updates, 2 traced microbatches and 1 traced optimizer step. Worker timeout: 120 seconds.
Frozen completed checkpoints, learning rates zero, identical 64-batch bank, BF16, context 512, microbatch 2, accumulation 16, checkpointing enabled.
No architecture changes, nonzero-learning-rate training, baseline campaign or cloud.
Acceptance: results serialize, both probes pass, matched inputs/settings/source hashes, fixed weights and artifact preservation verified, measured bottleneck and smallest proposed fix reported.
Serialization retry bound: one; diagnose and stop if another probe failure occurs.


## Retained narrative: profiling-report.md

# Bounded dense/MoE throughput investigation

## Finding and recommendation

**Backward, including activation-checkpoint recomputation, is the largest measured regression:** +551.04 ms/update, about 67.65% of the additional time in the separately synchronized phase measurements. Forward contributes +256.69 ms (31.51%); optimizer contributes +6.63 ms (0.81%). Data loading is not the bottleneck here.

The trace supports a fragmented, host-launch-heavy MoE path: four Python-dispatched experts, smaller GEMMs, extra gathers/scatters/casts, dynamic dispatch synchronization, and repeated dispatch under checkpointing. It does not show every token being processed by every expert. These measurements identify the bottleneck family, not an exact uninstrumented attribution to each operation.

**Smallest justified next fix, proposed only:** replace the four independent dynamic dispatch/gather/scatter paths inside `Top1FeedForward.forward` with one packed token permutation per layer, four contiguous expert slices, and one restoration to original token order. Use fixed-four-bin counts rather than dynamically sized `bincount`; transfer slice boundaries together if eager Python needs host integers. Preserve top-1 routing, FP32 differentiable gate weights, auxiliary loss, empty-expert gradient semantics, attention and checkpointing policy.

This is a localized implementation change, not a new architecture or optimizer policy. It still leaves four smaller expert MLPs and extra optimizer work. No gain is claimed before a separately authorized correctness check and short timing comparison. **No optimization was implemented.**

## Controls and preservation

- Reference runs: dense `20260914T231604Z-061092d4`; MoE `20260914T232147Z-ff9e4e60`. Latest comparison is in `runs/autoresearch/moe-resume-20260914T231300Z`.
- Native Windows, local RTX 5070 Ti, PyTorch 2.9.1+cu128.
- Both workers imported identical copies of the actually executed source snapshots, not live project modules. Original source/configuration/data/tokenizer hashes were verified. Imported paths/hashes and replay-bank hashes were checked again in analysis.
- Both: depth 6, width 384, context 512, microbatch 2, 16 accumulation microbatches, 16,384 tokens/update, BF16 autocast, identical SDPA backend, activation checkpointing enabled. Router softmax stays FP32.
- Same 64-batch bank and replay order. First 32 stock-loader batches matched the bank exactly.
- Each completed checkpoint was loaded with fresh, warmed optimizer state. All learning rates were zero, but optimizer kernels still ran. Checkpoint tensors were verified bitwise unchanged afterward.
- Per model: 3 warmup updates, 6 uninstrumented timing updates, 3 separately synchronized phase updates, then 2 traced microbatches and 1 traced optimizer step. Hard timeout: 120 seconds/worker. Total diagnostic tokens/model: 198,656, including warmups and tracing.
- Serialization fix: convert only configuration `torch.dtype` values to explicit strings. Other unsupported types still fail. Original failed script, logs, trace and batches remain intact.
- **225 prior files verified unchanged**, including all depth-4/6/8 results and the failed probe.

## Uninstrumented replay throughput

No profiler or additional module hooks in this pass. Existing routing-statistics/gradient hooks remained, matching the implementation. Includes pinned-host transfer, forward, backward, optimizer and gradient zeroing. Excludes CPU packing/tokenization, measured separately.

| Metric | Dense | MoE |
|---|---:|---:|
| Timed tokens | 98,304 | 98,304 |
| Total timed seconds | 3.1037 | 8.2129 |
| Mean update time | 517.28 ms | 1,368.81 ms |
| Update range | 486.58-537.49 ms | 1,286.67-1,478.90 ms |
| Tokens/s | 31,673.5 | 11,969.5 |
| Total parameters | 26,345,772 | 47,588,652 |
| Structural active parameters/token | 26,345,772 | 26,354,988 |
| Peak allocated VRAM | 434.89 MiB | 725.25 MiB |
| Peak reserved VRAM | 530.00 MiB | 774.00 MiB |
| Memory-table bytes | 0 | 0 |

MoE update time is **2.65x**, with **62.21% lower replay throughput**. VRAM is the uninstrumented pass's PyTorch allocator peak, excluding other processes/driver allocations.

Historical five-minute training throughput was 29,204.9 versus 12,223.6 tokens/s; BPB was 0.628327 versus 0.742139. This short replay reproduces the direction, not an exact replication of training rates. No new BPB was measured.

## Separate phase timings

Means of three diagnostic updates with synchronization at phase boundaries. Extra synchronization changes overlap. **Do not use their sum to calculate primary throughput**, or equate their deltas with an exact decomposition of the separate uninstrumented pass.

| Phase per 16,384-token update | Dense | MoE | Difference |
|---|---:|---:|---:|
| Replay data transfer | 1.05 ms | 1.16 ms | +0.11 ms |
| Forward | 180.46 ms | 437.15 ms | +256.69 ms |
| Backward plus checkpoint recomputation | 351.00 ms | 902.04 ms | +551.04 ms |
| Optimizer | 11.48 ms | 18.10 ms | +6.63 ms |
| Zero gradients | 0.12 ms | 0.26 ms | +0.14 ms |

Standalone original CUDA loader, including packing/tokenization and transfer: steady mean **0.444 versus 0.485 ms/microbatch**; cold first batch **82.49 versus 86.01 ms**. Steady loading is approximately 7.11 versus 7.76 ms per 16 microbatches without overlap. This short sample does not cover the entire corpus/refill distribution.

## Diagnostic trace evidence

Same two microbatches and one optimizer step. These numbers contain profiler overhead and are **not throughput measurements**.

| Observation | Dense | MoE |
|---|---:|---:|
| Actual CUDA kernel events | 2,664 | 6,416 |
| `aten::mm` calls | 318 | 654 |
| `aten::bmm` calls | 90 | 150 |
| `aten::nonzero` calls | 0 | 96 |
| `aten::bincount` calls | 0 | 24 |
| `aten::index_add` calls | 0 | 96 |
| `cudaStreamSynchronize` calls | 0 | 144 |
| Sum of actual kernel durations | 16.74 ms | 30.54 ms |

- **Dispatch/synchronization:** executed `source/project/autoresearch_model.py:37` loops over four experts; line 38 uses dynamic `where/nonzero`. Of 144 stream synchronizations, 96 are enclosed by `nonzero`; another 48 are copy/scalar-read paths inside `bincount`. Their total traced duration is 5.20 ms. Sync overhead alone does not explain the whole regression.
- **Small GEMMs/launch overhead:** expert hooks show **137-485 rows per expert**, mean 256, versus 1,024 rows per dense FFN microbatch; width 384, hidden width 1,536. Actual kernel count rises 2.41x. Combined `cudaLaunchKernel/cuLaunchKernel` CPU API time is **34.44 versus 77.33 ms** in the trace. Pure Python interpreter time was not isolated.
- **Copies and assembly:** lines 39-43 gather separate subsets, multiply weights, cast and perform four out-of-place `index_add` operations. `aten::copy_` calls rise from **515 to 1,467**. Not all copies are dispatch: casts, shared model work and optimizer work contribute.
- **Checkpoint repetition:** line 183 checkpoints whole blocks. Hooks observe **48 original-forward expert invocations plus 48 recomputation invocations** in two microbatches. Routing and expert forward work repeat during backward.
- **No accidental all-expert token computation:** each layer processes exactly **2,048 expert input rows total across all experts over two original microbatches**, not 8,192. Recompute totals separately match. No empty expert was observed in this sample. Source invokes an empty expert on a zero-row tensor when needed, preserving optimizer-compatible gradients, not computing unselected token rows.
- **Optimizer:** executed `source/upstream/train.py:765` stacks all group gradients/parameters and updates every matrix. Dense FFNs: **12 matrices / 7,077,888 parameters**. MoE: **48 matrices / 28,311,552 expert parameters**, plus six router matrices / 9,216 parameters in AdamW. This extra work is real but not dominant.
- **Routing health:** zero dropped tokens and finite, nonzero router gradients in every layer. These include the auxiliary objective; this is not a new task-only-gradient test.

Raw key-averaged user annotations contain both CPU and GPU annotation entries. Do not treat their counts as invocation counts or inclusive device times as summed GPU compute. The table counts actual `kernel` trace events; expert counts use independent hook records. Kernel-duration sums are neither elapsed time nor utilization.

## Expected costs versus implementation overhead

Expected under this architecture/policy: router/auxiliary loss, token grouping, gate multiplication, smaller GEMMs, checkpoint recomputation, gradients/state/updates for all expert matrices receiving tokens, and larger parameter storage.

Implementation-specific opportunities: repeated Python dispatch, four dynamic index extractions, dynamic histogram sizing, repeated gathers/full-size output assembly, conversion kernels, and their repetition during recomputation. Packed dispatch is the smallest coherent first optimization proposed. Grouped/fused expert GEMMs would be a larger follow-up, not the first change recommended here.

Matching structural active parameters does not match training cost: that count excludes dispatch/control overhead, ignores GEMM shapes and does not limit optimizer work to one expert per layer.

## Limits and stop boundary

Preliminary short sequential samples, one GPU, separate saved checkpoint weights, uncontrolled background load/clocks, fresh optimizer states and zero learning rates. The earlier failed dense-only probe measured 28,015.1 tokens/s before serialization failed; it is preserved and not pooled into this retry. Variation reinforces that no precise optimization benefit is established.

All sources, configurations, logs, timing samples, traces, expert rows and hashes are preserved. No architecture changes, additional baselines, longer training or cloud jobs. **Stop after this report.**

## Artifacts

- `probe.log`; `dense/profile.log`; `moe/profile.log`.
- Each variant's `result.json`, `trace.json`, `operators.json` and `expert-rows.json`.
- `manifest.json`, `source/`, `batches.pt`: identities and replay inputs.
- `profile_probe.py`, `analyze.py`, `analysis.json`: reproducible probe/analysis.
- `prior-preservation.json` and `preservation-result.json`: preservation receipts.
