# N-gram checkpoint diagnosis

Outcome: memory has a small positive inference contribution in both jointly trained models. The seed-42 M training discrepancy remains unexplained. No new training, optimizer, backward pass, tuning, or cloud job was run.

## Fixed-validation inference ablation

| Model | Recorded BPB | Enabled BPB | Residual disabled BPB | Disabled - enabled |
|---|---:|---:|---:|---:|
| DG | 0.634178 | 0.634178090 | 0.634274460 | +0.000096369 |
| MG | 0.637723 | 0.637722916 | 0.637860352 | +0.000137436 |

Both enabled evaluations reproduce their recorded values within the predeclared absolute tolerance 0.000002 BPB (six-decimal recording plus small FP32 reduction variation). Repeated, unhooked evaluations give exactly the same aggregate BPB. All six successful probes use the same 65,536-token validation tape, batch size 2, context 512, full attention, BF16 AMP, checkpointing off, and unchanged captured byte-normalized cross-entropy. Routing auxiliary loss is excluded. Model parameter hashes are unchanged by every evaluation.

Residual removal is an **inference ablation of a jointly trained model**, not a replacement for a separately trained control. The hook computes the existing memory branch, then returns its input unchanged. These instrumented evaluations do not measure inference throughput. Previous training comparisons remain DG-D = -0.000670 and MG-M = +0.000506 BPB; they measure a different quantity. Small ablation deltas do not prove the memory layer caused the complete between-training difference.

## Gate and residual measurements

Memory exists only after zero-based block 1. Initialization below is reconstructed using the captured seeds and verified against every original initialization parameter hash; it is not a final-checkpoint measurement. Final measurements load the original pre-evaluation checkpoints. All distributions use the fixed validation tape, not the earlier 32-training-microbatch diagnostic sample.

| Condition / state | Gate mean (all / valid suffix) | Gate p1 / median / p99 (all) | Hidden RMS | Gated residual RMS | Residual / hidden RMS | Effective addition / hidden RMS |
|---|---|---|---:|---:|---:|---:|
| DG / initial | 0.119202964 / 0.119202957 | 0.119202919 / 0.119202919 / 0.119202919 | 1.199920 | 0.002377 | 0.1981% | 0.1988% |
| DG / final | 0.999423683 / 0.999645829 | 0.991979003 / 0.999889731 / 0.999990225 | 18.968975 | 0.155359 | 0.8190% | 0.8320% |
| MG / initial | 0.119202964 / 0.119202957 | 0.119202919 / 0.119202919 / 0.119202919 | 1.199920 | 0.002377 | 0.1981% | 0.1988% |
| MG / final | 0.997475386 / 0.998815715 | 0.953995109 / 0.999770582 / 0.999988914 | 25.245697 | 0.194687 | 0.7712% | 0.7839% |

Each gate distribution covers 65,536 positions; 64,985 (99.1592%) have at least one valid suffix. Invalid suffixes contribute zero lookup residual regardless of their gate. Initialization gates are constant sigmoid(-2) = 0.119202919; tiny mean-versus-extrema discrepancies are FP32 aggregation rounding.

| Final condition | Gate min / max (all) | Gate p1 / median / p99 (valid suffix) | Token residual/hidden p1 / median / p99 / max |
|---|---|---|---|
| DG | 0.817658 / 0.999996 | 0.996257 / 0.999891 / 0.999990 | 0.1887% / 0.6815% / 2.2593% / 4.1118% |
| MG | 0.581888 / 0.999997 | 0.982083 / 0.999774 / 0.999989 | 0.1441% / 0.6485% / 2.3316% / 5.1835% |

The gate is nearly saturated open, but this does not make the memory branch dominant: its global RMS is under 1% of the incoming backbone RMS. Residual RMS includes gating and precision casting; effective addition RMS also includes rounding when the residual is added to hidden states. These norms do not determine importance to logits or gradient interference. The positive inference ablations establish a small net contribution on this validation slice, not broad generalization.

### Backbone block-output RMS by layer

| Condition / state | Block 0 | Block 1 | Block 2 | Block 3 | Block 4 | Block 5 |
|---|---:|---:|---:|---:|---:|---:|
| DG / initial | 1.099961 | 1.199920 | 1.299823 | 1.399711 | 1.499640 | 1.599708 |
| DG / final | 11.847523 | 18.968975 | 19.995457 | 26.629229 | 26.682266 | 33.788265 |
| MG / initial | 1.099961 | 1.199920 | 1.299823 | 1.399711 | 1.499640 | 1.599708 |
| MG / final | 12.961676 | 25.245697 | 25.862635 | 35.666332 | 30.063601 | 40.048302 |

Block 1 is measured immediately **before** memory insertion; later blocks include downstream effects. There are no separate gates or memory tables at other layers. Full per-token quantiles and valid-only gate distributions are retained in each probe analysis.

## Training collision exposure

Counted the verified shared training tape once: all four training runs consumed the same permutation of all 8,192 microbatches (8,388,608 input tokens). Keys use only row-local causal input suffixes; BOS-containing and incomplete suffixes are excluded, exactly as in captured memory code. Order-2 and order-3 tables are separate, each with 8,192 addresses.

| Order | Valid occurrences | Distinct suffixes | Occupied / colliding buckets | Any-collision exposure | Non-dominant occurrence fraction | Competing bucket mass |
|---|---:|---:|---|---:|---:|---:|
| 2 | 8,291,989 | 361,176 | 8,192 / 8,192 | 100.00% | 33.7452% | 46.0495% |
| 3 | 8,244,258 | 1,593,279 | 8,192 / 8,192 | 100.00% | 63.7415% | 78.1570% |

Let f_i be suffix frequency, F_b total frequency in bucket b, and N total valid occurrences.

- Any-collision exposure: sum_i f_i * 1[bucket has another distinct suffix] / N.
- Non-dominant fraction: 1 - sum_b max_i(f_i in b) / N. Tied dominant choices have equal mass.
- Competing bucket mass: sum_i f_i * (1 - f_i/F_bucket(i)) / N; frequency-weighted probability of a different suffix conditional on sampling within the accessed bucket.
- Independent global-pair probability, with replacement: (sum_b F_b^2 - sum_i f_i^2) / N^2; 0.000122328 for order 2 and 0.000119102 for order 3.

Distinct suffixes per bucket have min/median/max 20/44/68 (order 2) and 146/194/251 (order 3). This measures extensive address sharing, especially for trigrams. It does **not** measure conflicting gradients, lost useful information, or the quality impact of a larger table. Frequent suffixes dominate some buckets, and learned shared representations can be useful. Full occupancy alone does not establish that either table is too small.

## Reconciling the two seed-42 M controls

| Checkpoint | Earlier captured source BPB | New captured source BPB |
|---|---:|---:|
| Earlier equal-token M | 0.638882944 | 0.638882944 |
| New n-gram-study M | 0.637216994 | 0.637216994 |

Cross-source evaluations reproduce each checkpoint exactly at the aggregate BPB level, including repeat evaluations. This rules out a measured aggregate BPB change from these forward/evaluation source swaps on this tape; it does not establish bitwise equality of every intermediate or backward pass.

- All 88 initialization tensor hashes match, including experts and routers.
- Candidate, protocol, complete batch order/hash chain, step schedule and optimizer-group receipts match exactly. Dataset/tokenizer seals and actual tape hash match.
- Both runs consumed 512 updates / 8,388,608 tokens, including warmup; step-driven warmup/decay, BF16 AMP/Muon, checkpointing off and full attention match.
- Captured upstream training/data source and bootstrap match. Project model/adapter/runner changed for optional memory; this M has no memory. No intentional no-memory tensor-math change was identified.
- Printed last-microbatch losses first differ at update 17: new 3.994706 versus old 3.994785. The first 16 match only at six printed decimals; all subsequent 496 differ.
- All 88 final checkpoint tensors differ. The -0.001666 recorded BPB difference reflects distinct training trajectories, not just validation formatting.
- Historical deterministic-algorithm flags, selected kernels/driver details and GPU load were incompletely recorded. Current repeated inference is stable; it cannot establish historical training determinism.

**Cause remains unexplained.** Numerical reduction/order sensitivity is plausible, not proven. Matching seeds and initialization are not a determinism guarantee. The shift exceeds either original memory-versus-control BPB delta, so single-seed architectural conclusions remain especially tentative.

## One recommended next experiment (not implemented)

Run a bounded M reproducibility replay before changing memory capacity: two fresh-process repeats for each historical captured source, seed 42, identical initialization and the first 32 optimizer updates of the existing tape/schedule. Record exact per-step parameter/gradient hashes and selected kernel/environment/determinism settings, with deterministic algorithms enforced and unsupported operations treated as a recorded failure, not silently bypassed. This four-short-run study would test within-source repeatability and locate cross-source divergence near update 17. It would not retroactively prove the original cause. No such replay has been started.

## Provenance, repair and preservation

The first DG probe failed enabled reproduction by 0.002079311 BPB because the diagnostic constructor omitted the captured adapter's WINDOW_PATTERN='L' override. That failed probe, its source snapshot, log and partial measurements are preserved and excluded from conclusions. DG-retry1 restores the original attention setting; no checkpoint, trained model code, data or tolerance was changed.

Each successful process used Python isolated mode with captured project/upstream modules, verified source/config hashes, checkpoint hashes, initialization hashes and sealed data. Diagnostic scripts are separately captured and hashed; the new publishing commit is not claimed as the source used for historical training. Detailed source comparisons and checkpoint identities are in analysis.json and the child receipts.

Fresh preservation verification: 869 pre-existing artifact files match their starting SHA-256 hashes. Local checkpoints, datasets, logs and large artifacts remain outside normal Git. Only compact reports and diagnostic code are published. Measurement overhead is outside training, and no throughput or VRAM improvement is claimed.

## Probe receipts

- [DG](../ngram-diagnostics-20260915--DG-f6ea0955/README.md)
- [DG-retry1](../ngram-diagnostics-20260915--DG-retry1-c4727406/README.md)
- [MG](../ngram-diagnostics-20260915--MG-efab1ca5/README.md)
- [M-old](../ngram-diagnostics-20260915--M-old-1da8b977/README.md)
- [M-new](../ngram-diagnostics-20260915--M-new-e0167c45/README.md)
- [M-new-old-source](../ngram-diagnostics-20260915--M-new-old-source-0f2d2923/README.md)
- [M-old-new-source](../ngram-diagnostics-20260915--M-old-new-source-fc1ec60a/README.md)
