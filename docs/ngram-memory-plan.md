# Causal n-gram memory: bounded next experiment

**Status: proposal only. No memory implementation or training is authorized by this document.**
The current task publishes research and prepares this plan. A separate approval is required to implement it or start the four proposed runs. Native Windows / RTX 5070 Ti; existing TinyStories only; no NCP, tokenizer changes, cloud work, or tuning sweep.

## Evidence and hypothesis

The repository's [architecture](architecture.md), [experiment roadmap](experiments.md), and [initial research review](research.md) propose independently switchable mechanisms. Their original envelopes and combined-feature hypotheses are proposals, not descriptions of the completed small runs. This plan narrows the next experiment to memory alone.

Engram motivates deterministic suffix n-gram lookup with context-dependent gating. Its published architecture also includes tokenizer compression, multiple hash heads, convolution, and specialized integration. We do not reproduce those components or transfer its large-model efficiency claims to this project. The primary architecture reference inspected for this plan is [Engram, revision v1, sections 2.1-2.3](https://arxiv.org/html/2601.07372v1). No third-party code is copied; any later reuse requires a pinned repository revision and license check.

Our completed [equal-token comparison](../reports/experiments/equal-token-20260915-87990217/README.md) found MoE-minus-dense BPB differences of +0.004035 and -0.003030 for seeds 42 and 43. Opposite signs and two seeds are not evidence of equivalence or a reliable MoE advantage. That comparison motivates retaining both backbones as controls rather than assuming memory helps MoE specifically.

**Hypothesis:** a small, trainable lookup of observed two- and three-token suffixes lowers held-out BPB at a fixed training-token budget by representing recurring local patterns. Test the memory effect separately in dense and packed MoE backbones. The initial screen does not establish that any benefit exceeds random-seed variation or the benefit of simply adding parameters.

## Frozen proposed mechanism

Depth 6, model width 384. Insert one memory residual immediately after decoder block index 1 (the second block), before the next block's existing residual mixing. Leave the original `x0` embedding skip unchanged. Do not add a memory copy at every layer. A memory-disabled branch must retain the existing dense/packed MoE computations exactly.

At position `t`, the model predicts target token `x[t+1]`. Lookup keys use only **input** suffixes `(x[t-1], x[t])` and `(x[t-2], x[t-1], x[t])`. Current input `x[t]` is observed, not a future target. Never pass `targets`, validation answers, logits, or a shifted target tensor into the key function.

- Two separate tables `E2` and `E3`, each `8192 x 64`; one hash per order. Keep raw tokenizer IDs, including case and whitespace distinctions. No token normalization, learned address selection, external retrieval corpus, or table growth.
- Versioned hash `polynomial-mod-v1`: initialize signed-int64 `h = n`; for each token ID `u` in left-to-right suffix order, set `h = (h * 1000003 + (u + 1)) % 2147483647`; address is `h % 8192`. Validate token ID bounds and use int64 intermediates so the documented operations do not overflow. Do not use Python's process-randomized `hash()`.
- If a suffix is invalid, its 64-vector contribution is exactly zero and produces no table gradient. Do not assign invalid histories to a trainable padding row. Valid addresses may use every table row.
- Concatenate both vectors to `m[t]` of width 128. Set `v[t] = Wv m[t]`, with biasless `Wv` of shape `384 x 128`.
- A scalar context gate is `g[t] = sigmoid(wg dot norm(h[t]) + bg)`, using only the same-position causal hidden state and existing parameter-free normalization. Add `g[t] * v[t]` to `h[t]`; preserve the residual dtype explicitly. There is no memory auxiliary loss.
- Proposed initialization: table entries independent normal standard deviation 0.02; `Wv` independent normal standard deviation `1/sqrt(128)`; `wg = 0`, `bg = -2`. Nonzero table/projection initialization avoids blocking table task gradients at the start. Gate initialization is intentionally small, not an identity guarantee.
- Initialize added parameters under a separately recorded, seed-derived RNG stream. Shared backbone initialization must match memory-off controls exactly; memory parameters must match between dense+memory and MoE+memory for seed 42. Future seeds must change the nonconstant added initialization without affecting data order.
- FP32 table, projection and gate parameters; table lookup values cast consistently with the BF16 compute path; FP32 gate calculation followed by an explicit residual-dtype cast. Test against a reference with the same casts and operations. No approximate hash or mixed-precision integer computation.
- Put all added parameters exactly once into explicit AdamW groups: initial LR 0.001, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay 0, and the common step LR multiplier. Existing backbone/MoE optimizer groups remain unchanged. Use ordinary dense gradients in this first version; do not route table matrices into Muon or introduce a sparse optimizer simultaneously.

The table is static in addressing and size, **not frozen during training**. Only the training objective updates its parameters. Evaluation is read-only: no counts-based fitting, optimizer steps, adaptive table writes, or held-out cache accumulation.

## Boundary and future-token rules

Reset history independently for each batch row at sequence start and at every BOS/document boundary. Exclude BOS itself from suffixes. The first ordinary token after a boundary has no valid lookup; the second permits only a bigram; the third permits both. Do not read a preceding row, previous microbatch, or previous packed document to fill incomplete context. If packing contains explicit boundary metadata, honor it in addition to BOS; fail preflight if the tape's boundary convention cannot be established from its captured packer/tokenizer.

At generation time, keep at most two preceding ordinary input IDs per sequence. Reset on BOS, new request, or changed sequence identity. Batched reordering must reorder the memory histories too. Full-prefix and token-by-token lookup must agree. Until a model KV-cache path is supported, do not claim full cached-decoding compatibility merely from a lookup-state test.

Existing causal attention may attend to earlier packed documents. Preserve that behavior identically in all four conditions rather than changing attention masking in this experiment. The **lookup history** must not cross BOS; the gate's hidden state can still contain the backbone's existing past-document context. Future-token isolation is mandatory for the whole model, but whole-model independence from past documents is not claimed.

## Exact proposed parameter costs

Counts below are arithmetic design costs, not measured runtime memory or FLOPs.

| Component | Parameters | FP32 parameter bytes |
|---|---:|---:|
| Two `8192 x 64` tables | 1,048,576 | 4,194,304 |
| Biasless `128 -> 384` projection | 49,152 | 196,608 |
| Scalar gate, weight plus bias | 385 | 1,540 |
| Total addition | **1,098,113** | **4,392,452** |

Table weights alone are exactly **4 MiB**, the proposed table-weight cap. Added FP32 weights, gradients and two Adam moments total an estimated **17,569,808 bytes / 16.756 MiB**, excluding temporary optimizer buffers, activations, allocator overhead, and checkpoint files. Measure actual peak VRAM; this estimate is not a fit result. Dense-gradient AdamW also scans unselected table rows, so sparse lookup does not imply sparse optimizer cost.

| Condition | Total parameters | Structural active parameters, maximum valid suffix |
|---|---:|---:|
| D: dense | 26,345,772 | 26,345,772 |
| M: packed MoE | 47,588,652 | 26,354,988 |
| DG: dense + memory | 27,443,885 | 26,395,437 |
| MG: packed MoE + memory | 48,686,765 | 26,404,653 |

The active addition counts at most two selected 64-value rows plus every projection/gate parameter: **49,665**. Preserve the existing convention counting full shared token embedding tables and one expert per layer; separately label this selected-memory-row extension. Fewer rows are selected at boundaries. Neither structural count measures FLOPs, all-row optimizer work, or kernel launch cost. Parameter totals must be checked against the actual implementation before any run.

## Correctness gates before training

Run bounded CPU FP32 and CUDA BF16 tests, retaining logs and exact source hashes. Compare with an independently written scalar-key/reference lookup; use matching precision and operations. Start FP32 comparisons at `atol=1e-6, rtol=1e-5`; set BF16 limits from observed reference quantization error and document absolute/relative errors rather than arbitrarily loosening a failure. Integer keys and validity masks must agree exactly.

1. Check shapes, scalar/reference hash equality, empty prefixes, sequence lengths 1/2/3, repeated tokens, all-BOS rows, consecutive BOS, intentional hash collisions, and independent batch rows.
2. Change every suffix after a chosen position and assert earlier keys, gates, memory outputs and logits are unchanged within declared numerical limits. Include boundaries next to the intervention and several cut positions. Use nonzero initialized projections so a leak cannot be hidden behind a zero residual.
3. At fixed input, change `targets` and verify forward logits and lookup keys are unchanged. Different targets are allowed to change loss and its gradients. Instrument input hidden/embedding activations and verify a logit at `t` has zero gradient with respect to future-position activations, not merely shared embedding parameter rows.
4. Hold a post-BOS input segment fixed while changing the preceding document; its lookup keys/vectors must remain identical. Test the gate independently with fixed hidden inputs. Do not demand that full-model logits ignore earlier documents under the unchanged attention policy.
5. Check finite, nonzero task gradients for selected table rows, projection and gate on nondegenerate examples; unselected table gradients and invalid-suffix contributions must be zero before the optimizer step. Explicitly distinguish zero gradients from possible Adam momentum updates after a row was selected in an earlier step.
6. Verify optimizer coverage, no duplicates, correct table optimizer groups, and an actual small synthetic optimizer update. Preserve MoE router task gradients, auxiliary loss behavior, routing counts and zero dropped tokens.
7. Verify memory on/off checkpoint recomputation matches outputs, gradients, keys and counters; ensure recomputation does not count a lookup twice or mutate history. The proposed training runs themselves keep checkpointing **off**.
8. Test full-prefix versus incremental lookup/reset behavior, permutation of batched generation histories, save/load round-trip, and a small CPU optimizer/RNG/sampler-state resume fixture. Record unsupported whole-model cache/resume functionality explicitly rather than claiming it passes.
9. Confirm validation has no parameter changes and no fit-time dependence on held-out text. BPB must include only the existing token cross-entropy/byte accounting, not MoE auxiliary loss or memory diagnostics.
10. Verify 512-update stopping, the full step schedule, initialization pairing, captured-source isolation, tape identity and actual consumed-batch hash chain. A live source edit after snapshot creation must not affect the trial.

Stop and diagnose any failed gate before training. No substitute run, retuning, or longer campaign is implied by a failure.

## Proposed four-run matched screen

One paired seed, **42**, four fresh conditions in frozen order **D, DG, M, MG**. All use depth 6 / width 384 and activation checkpointing off. One seed bounds this mechanism screen to four runs and avoids presenting the old two-seed study as a memory control. A second seed is a later, separately approved confirmation, not an automatic extension. Fixed order leaves possible temporal system-load confounding, which the report must acknowledge.

Use exactly the existing equal-token protocol, not the earlier 300-second stopping protocol:

- Existing sealed TinyStories training/validation splits, tokenizer, BF16 precision, full causal attention, context 512, microbatch 2, accumulation 16, and 16,384 tokens/update.
- Exactly **512 optimizer updates = 8,388,608 training tokens per run**, including all warmup updates. Across the four proposed runs, the total budget is 33,554,432 training tokens. The inherited `training_seconds: 300` metadata field is not an active stop condition in fixed-update mode; record that explicitly.
- Reuse the existing 8,192-microbatch tape and seed-42 order only after verifying their hashes against captured receipts. Existing tape SHA256: `07e33edbaa75fd2b06e8478436c55a0ab83382035a15e3dd4648026f7a8fb689`. All four models must consume the identical order and token hash chain; no hidden extra fetch/update.
- For zero-based optimizer step `i` from 0 to 511, LR multiplier is 1 through `i=256`, then `2*(1-i/512)`. There is **no LR warmup**. Last multiplier is `1/256`; `i=512` is not a 513th update. Record all 512 values and each group's base LR.
- Keep existing Muon matrix LR 0.04 and backbone optimizer settings. Muon momentum is `.85 + .1*min(i/300,1)`; Muon weight decay is `.2*(1-i/512)`. Packed MoE keeps four top-1 experts, router LR 0.001 and auxiliary coefficient 0.01.
- Keep final BPB evaluation at batch 2 and exactly 65,536 validation tokens, identical byte accounting and token ordering. Do not use validation to choose memory placement, size, hash, seed, or optimizer settings.
- First 11 updates are timing warmup, not extra updates. Report timed throughput on the remaining 501 updates / **8,208,384 tokens**. Include H2D, forward, backward and optimizer work; no profiler in throughput measurements. Record run wall time separately from tape preparation, and include checkpoint/evaluation overhead in run wall time as in the reference.
- Keep the existing 900-second external per-run failure timeout; it does not replace fixed-update stopping. If a run cannot finish safely within it, preserve failure artifacts and stop rather than altering budget or microbatch. No separate baseline-smoke campaign is proposed; bounded synthetic correctness/preflight checks are the gate.
- Capture executable project/upstream sources and resolved configuration before each run. Record actual executed-file hashes, data/tokenizer/tape/order hashes, initialization hashes, schedule and optimizer groups, seed, precision, device/runtime, and all failures. Create each log and provide its exact PowerShell `Get-Content -Tail 10 -Wait` command before starting.

## Report and stop criteria

Report every condition's BPB, total/structural active parameters, table bytes, total/timed tokens, timed throughput, complete wall time, peak allocated/reserved VRAM, and MoE utilization. Memory diagnostics: valid bigram/trigram rate, accessed rows, address occupancy, exact distinct training suffixes per occupied bucket where feasible, collision definition/count, gate distribution, table gradient norms, and update norms. Accumulate expensive uniqueness/collision diagnostics outside timed updates from the frozen training tape; do not fit anything from validation.

Primary contrasts are `BPB(DG)-BPB(D)` and `BPB(MG)-BPB(M)`; negative favors memory. Report the interaction `(MG-M)-(DG-D)` without claiming statistical significance. Dense-vs-MoE contrasts remain secondary. Keep this comparison separate from unequal-time-budget or different-schedule reports; old runs are historical context, not fresh paired controls.

Equal tokens do not mean equal parameters or compute. An improved BPB here cannot distinguish a special n-gram advantage from additional capacity, and one seed cannot establish reproducibility. A later parameter-matched non-memory control and additional seeds would be necessary before an efficiency claim, but are **not** part of this plan.

The proposed scope ends after four successful runs and one preliminary report, or after any failed gate/run with preserved diagnostics. No NCP, architectural additions, table-size sweep, batch tuning, deletion, or cloud jobs.
