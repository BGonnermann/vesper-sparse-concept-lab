# ncp-20260915

Outcome: **running**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
audit-recovered.log:
SyntaxError: '(' was never closed

campaign.log:
2026-09-15T16:56:40.286744+00:00 Conditional depth12 transfer not entered: D6 primary replication rule failed
2026-09-15T17:07:05.699998+00:00 PREDECLARED conditional module-seed diagnosis in docs/ncp-campaign.md commit13d0225 before final crossed NCP outcome: require init45 less favorable than42 in both order columns and mean initializer contrast>0.002; then two module/backbone seed swaps at fixedorder42 only if full timeout forecast fits before90-minute reserve. No candidate reselection.

controller-red.log:
Traceback (most recent call last):
AssertionError: False is not true : Missing campaign controller
Traceback (most recent call last):
AssertionError: False is not true : Missing campaign controller
Ran 2 tests in 0.001s
FAILED (failures=2)

cpu-suite1.log:
Ran 60 tests in 3.949s
OK

cuda-suite1.log:
Ran 60 tests in 6.630s
OK

dimension-checks-after-implementation.log:
Ran 3 tests in 0.917s
OK

dimension-checks-before-implementation.log:
    raise ValueError("Fixed comparison requires seed 42, 43 or 44.")
ValueError: Fixed comparison requires seed 42, 43 or 44.
ERROR: test_width_override_is_bounded (__main__.DimensionChecks.test_width_override_is_bounded)
Traceback (most recent call last):
    raise ValueError("Candidate must select dense or moe with exactly its supported settings.")
ValueError: Candidate must select dense or moe with exactly its supported settings.
Ran 3 tests in 0.002s
FAILED (errors=3)

dimension-checks-portable-before-implementation.log:
    raise ValueError("Fixed comparison requires seed 42, 43 or 44.")
ValueError: Fixed comparison requires seed 42, 43 or 44.
ERROR: test_width_override_is_bounded (__main__.DimensionChecks.test_width_override_is_bounded)
Traceback (most recent call last):
    raise ValueError("Candidate must select dense or moe with exactly its supported settings.")
ValueError: Candidate must select dense or moe with exactly its supported settings.
Ran 3 tests in 0.001s
FAILED (errors=3)

early-metadata-red.log:
FAIL: test_loss_failure_preserves_partial_state_and_accounting (test_fixed_updates.FixedTests.test_loss_failure_preserves_partial_state_and_accounting)
Traceback (most recent call last):
  File "<repo>\tests\test_fixed_updates.py", line 79, in test_loss_failure_preserves_partial_state_and_accounting
    self.assertTrue((path/'model.json').exists(),'Missing parameter count for failed training')
AssertionError: False is not true : Missing parameter count for failed training
Ran 5 tests in 1.375s
FAILED (failures=1)

failure-receipt-red.log:
FAIL: test_loss_failure_preserves_partial_state_and_accounting (test_fixed_updates.FixedTests.test_loss_failure_preserves_partial_state_and_accounting)
Traceback (most recent call last):
  File "<repo>\tests\test_fixed_updates.py", line 60, in test_loss_failure_preserves_partial_state_and_accounting
    self.assertTrue((path/'training-failure.json').exists(),'Missing partial failure receipt')
AssertionError: False is not true : Missing partial failure receipt
Ran 4 tests in 0.990s
FAILED (failures=1)

initial-queue.log:
2026-09-15T12:00:33.731574+00:00 FAILED trial-0002-NCP-s42 RuntimeError('Training child exit 1')
Traceback (most recent call last):
    if result.returncode: raise RuntimeError('Preserved failed '+label+' trial; diagnose before more trials')
RuntimeError: Preserved failed NCP trial; diagnose before more trials

launch-AUX-s42.log:
2026-09-15T12:03:24.714710+00:00 FAILED trial-0003-AUX-s42 RuntimeError('Training child exit 1')

mechanism-queue.log:
2026-09-15T14:14:52.246985+00:00 Publication deferred; local mechanism result retained: CalledProcessError(128, ['git', 'add', '--', 'reports/README.md', 'reports/index.json', 'reports\\experiments\\ncp-20260915--trial-0001-D6-s42-b009c334', 'reports\\experiments\\ncp-20260915--trial-0002-NCP-s42-22537f23', 'reports\\experiments\\ncp-20260915--trial-0003-AUX-s42-6ac0782c', 'reports\\experiments\\ncp-20260915--trial-0004-N-prediction_weight0.1-s42-d0ad5ff9', 'reports\\experiments\\ncp-20260915--t

normalization-green-cpu.log:
Ran 8 tests in 1.461s
OK

normalization-red.log:
Traceback (most recent call last):
    raise error_metas[0].to_error(msg)
AssertionError: Tensor-likes are not close!
Ran 8 tests in 1.503s
FAILED (failures=1)

order-identity-red.log:
    raise ValueError("Protocol fields are missing or unknown.")
ValueError: Protocol fields are missing or unknown.
Traceback (most recent call last):
AssertionError: [102, 486, 438, 2, 458, 172, 372, 221, 246, 123, 100, 462, 222, 290, 299, 101, 51, 6, 111, 57, 321, 474, 505, 397, 467, 346, 361, 236, 251, 1, 190, 147, 267, 367, 181, 21, 243, 188, 504, 114, 378, 461, 399, 412, 227, 247, 279, 456, 509, 226, 338, 446, 162, 395, 328, 408, 159, 76, 454, 59, 174, 481, 352, 331, 230, 241, 349, 362, 255, 41, 14, 89, 136, 192, 180, 443, 403, 39, 434, 288, 374, 382, 54, 302, 430, 234, 357, 265, 91, 495, 7, 115, 317, 30, 275, 496, 425, 371, 427, 343, 286
Ran 2 tests in 1.818s
FAILED (failures=1, errors=1)

order-preflight.log:
Traceback (most recent call last):
    if result.returncode: raise RuntimeError('Correctness failed: '+str(path))
RuntimeError: Correctness failed: <repo>\runs\autoresearch\ncp-20260915\preflight-cpu-1789491292.log

order-preview-green.log:
Ran 2 tests in 1.776s
OK

preflight-cpu-1789473248.log:
ok
ok
ok
ok
ok
ok
Ran 63 tests in 3.917s
OK

preflight-cpu-1789473331.log:
ok
ok
ok
ok
ok
ok
Ran 64 tests in 3.911s
OK

preflight-cpu-1789473887.log:
ok
ok
ok
ok
ok
ok
Ran 65 tests in 3.871s
OK

preflight-cpu-1789474401.log:
ok
ok
ok
ok
ok
ok
Ran 66 tests in 4.144s
OK

preflight-cpu-1789475877.log:
ok
ok
ok
ok
ok
ok
Ran 67 tests in 4.140s
OK

preflight-cpu-1789480557.log:
ok
ok
ok
ok
ok
ok
Ran 71 tests in 4.117s
OK

preflight-cpu-1789481853.log:
ok
ok
ok
ok
ok
ok
Ran 72 tests in 4.348s
OK

preflight-cpu-1789485871.log:
ok
ok
ok
ok
ok
ok
Ran 75 tests in 4.331s
OK

preflight-cpu-1789487177.log:
ok
ok
ok
ok
ok
ok
Ran 75 tests in 4.309s
OK

preflight-cpu-1789491292.log:
ok
ok
ok
ERROR: test_budget_and_schedule_fingerprints_are_distinct (test_experiment_reports.ReportTests.test_budget_and_schedule_fingerprints_are_distinct)
Traceback (most recent call last):
KeyError: 'seed'
Ran 77 tests in 4.926s
FAILED (errors=1)

preflight-cpu-1789491332.log:
ok
ok
ok
ok
ok
ok
Ran 77 tests in 4.937s
OK

preflight-cuda-1789473254.log:
ok
ok
ok
ok
ok
ok
Ran 63 tests in 6.129s
OK

preflight-cuda-1789473337.log:
ok
ok
ok
ok
ok
ok
Ran 64 tests in 5.956s
OK

preflight-cuda-1789473893.log:
ok
ok
ok
ok
ok
ok
Ran 65 tests in 5.766s
OK

preflight-cuda-1789474407.log:
ok
ok
ok
ok
ok
ok
Ran 66 tests in 6.033s
OK

preflight-cuda-1789475883.log:
ok
ok
ok
ok
ok
ok
Ran 67 tests in 6.128s
OK

preflight-cuda-1789480563.log:
ok
ok
ok
ok
ok
ok
Ran 71 tests in 6.262s
OK

preflight-cuda-1789481860.log:
ok
ok
ok
ok
ok
ok
Ran 72 tests in 6.644s
OK

preflight-cuda-1789485878.log:
ok
ok
ok
ok
ok
ok
Ran 75 tests in 6.621s
OK

preflight-cuda-1789487184.log:
ok
ok
ok
ok
ok
ok
Ran 75 tests in 6.673s
OK

preflight-cuda-1789491339.log:
ok
ok
ok
ok
ok
ok
Ran 77 tests in 7.249s
OK

report-red.log:
ERROR: test_ncp_health_and_prospective_hypothesis_are_published (test_experiment_reports.ReportTests.test_ncp_health_and_prospective_hypothesis_are_published)
Traceback (most recent call last):
KeyError: 'collapsed'
Ran 8 tests in 0.229s
FAILED (errors=1)

test-ncp-green1.log:
ok
Ran 6 tests in 1.153s
OK

test-ncp-red.log:
Traceback (most recent call last):
AssertionError: unexpectedly None : Missing concept mechanism
Traceback (most recent call last):
AssertionError: unexpectedly None : Missing concept mechanism
Traceback (most recent call last):
AssertionError: unexpectedly None : Missing concept mechanism
Ran 6 tests in 1.272s
FAILED (failures=5, errors=1)

token-gate-full-cpu.log:
Ran 67 tests in 4.938s
OK

token-gate-ncp-cpu.log:
Ran 8 tests in 1.419s
OK

token-gate-red.log:
FAIL: test_scale_dependent_auxiliary_loss_is_not_a_token_ce_failure (test_fixed_updates.FixedTests.test_scale_dependent_auxiliary_loss_is_not_a_token_ce_failure)
Traceback (most recent call last):
  File "<repo>\tests\test_fixed_updates.py", line 45, in test_scale_dependent_auxiliary_loss_is_not_a_token_ce_failure
AssertionError: False is not true : Missing separate token-CE gate
Ran 5 tests in 1.335s
FAILED (failures=1)

transfer-entry.log:
2026-09-15T16:56:40.286744+00:00 Conditional depth12 transfer not entered: D6 primary replication rule failed
```

## Retained narrative: findings.md

# Campaign findings

Frozen NCP did not pass the predeclared primary two-seed consistency rule.

The four-seed mean NCP-minus-dense difference is -0.002320 BPB; lower is better. 2 of four paired seeds favor NCP. These are equal-token validation results, not held-out or equal-time results.

## Frozen confirmation and ablations

| Seed | Role | Dense BPB | NCP BPB | NCP minus dense |
|---:|---|---:|---:|---:|
| 45 | primary | 0.637922 | 0.640041 | +0.002119 |
| 46 | primary | 0.639401 | 0.635421 | -0.003980 |
| 43 | sensitivity | 0.637774 | 0.639233 | +0.001459 |
| 44 | sensitivity | 0.643075 | 0.634198 | -0.008877 |

| Comparison | Mean NCP minus control BPB | Seeds favoring NCP |
|---|---:|---:|
| D6 | -0.002320 | 2/4 |
| FROZEN-AUX | -0.001990 | 2/4 |
| FROZEN-CAP | +0.000729 | 2/4 |
| FROZEN-NOPRED | -0.001667 | 3/4 |

AUX keeps the selected concept objectives and parameters but removes predicted feedback. CAP replaces NCP with an exactly parameter-matched token-rate residual MLP. NOPRED removes prediction MSE and concept CE while retaining feedback, token CE and VQ learning. It is not a no-future-label or all-supervision-off condition.

The seed42 search selected 0.631877 BPB versus its fresh dense control at0.634848. That selection gain is reported separately from all four fresh-seed outcomes; the candidate was not replaced after confirmation.

## Measured cost across the four confirmation seeds

| Condition | Total / active parameters | Mean BPB | Mean update s | Mean trial s | Mean timed tok/s | Max allocated / reserved MiB | Max sampled board MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| I2-37a8ad42c0 | 30,001,452 / 30,001,452 | 0.637223 | 218.6 | 224.3 | 38453 | 653.3 / 688.0 | 2059.0 |
| D6 | 26,345,772 / 26,345,772 | 0.639543 | 164.4 | 170.9 | 51131 | 581.7 / 622.0 | 2430.0 |
| FROZEN-AUX | 30,001,452 / 30,001,452 | 0.639213 | 215.5 | 221.3 | 38974 | 653.3 / 690.0 | 2063.0 |
| FROZEN-CAP | 30,001,452 / 30,001,452 | 0.636494 | 174.8 | 180.1 | 48029 | 686.5 / 716.0 | 2087.0 |
| FROZEN-NOPRED | 30,001,452 / 30,001,452 | 0.638890 | 219.0 | 224.7 | 38396 | 653.3 / 688.0 | 2059.0 |

NCP adds 3,655,680 parameters (13.88%) and its mean all-update time is 1.33 times dense. Structural active counts are not amortized FLOPs: concepts run once per chunk; CAP runs at every token. The selected codebook has98,304 trainable transform parameters,24,576 frozen-basis bytes and24,576 effective-code bytes. No n-gram memory table is present.

Timed throughput excludes11 warmup updates. All-update time includes them; trial time additionally includes preparation, evaluation and verification, excluding publication. Sampled board memory includes desktop/driver use and may miss peaks. These instrumented wall-time measurements do not establish equal-time quality or a measured FLOPs advantage.

## Interpretation boundaries

This is a small ConceptLM-inspired mechanism test: complete-chunk pooling, causal prediction over segmented codebooks, detached future targets and delayed predicted feedback. It is not a faithful reproduction of the paper or evidence that the learned entries represent human-interpretable concepts. Official training code is unreleased; the inspected implementation and paper also differ in feedback weighting. Native SDPA, initialization, normalization, positional features, model size and TinyStories training differ here.

The mandatory native D6/D12 comparison changes both depth and width. The separate2x2 grid tests depth at fixed width and width at fixed depth under the same optimizer policy; width-dependent embedding/unembedding LR scaling remains part of that pinned policy. Matrix LR stays0.04. No optimum depth, width or training budget is established.

All per-trial configurations, hashes, failures, utilization diagnostics and source corrections are in [the full campaign report](CAMPAIGN.md). The initial finite total-loss guard failures do not establish divergence; the corrected guard and exact retries are disclosed there. Collapsed candidates remain reported and excluded from selection.


## Retained narrative: method-notes.md

# Method notes and prospective decisions

- Prototype correctness/fit completed at11:56 UTC, about15 minutes into the eight-hour campaign, within the two-hour implementation allocation.
- Unit-weight NCP feedback failed at optimizer update169; same-weight auxiliary-only failed at213. Neither has a valid fixed-budget BPB or final checkpoint. Raw logs, captured source/configuration and whole-board samples are retained.
- These are configuration instability outcomes. Causal, target-isolation and source-execution gates passed. Lowering prediction-loss weight is inside the predeclared search.
- Before trial4, failure handling was extended to retain partial model state, consumed-token accounting and NCP loss-component means on subsequent loss-limit failures. This changes failure reporting only; all successful training math and evaluation remain unchanged. Source hashes distinguish versions.
- Primary and newer source reviews are separate JSON receipts. Softmax mixing is supported explicitly by ArchPreview Eq8, but released official inference code still multiplies raw logits by codebooks. This discrepancy remains a reproduction limitation.
- Synthetic fit checkpoints are test artifacts, not training scores. Repeated preflight fit checks used the same named synthetic checkpoint files; all full research trial directories are unique and retained.

##12:29 UTC diagnosis correction

Dense saved-checkpoint probe on the same first32 seeded microbatches gives block0 hidden RMS12.5442 and raw chunk4 RMS10.2389. Source and checkpoint bytes were verified; probe uses CPU BF16 SDPA, so it is diagnostic rather than a GPU quality score. Thus high encoder-state scale is not NCP-specific. The first two failed runs crossed an inherited total-loss100 guard, which combines token CE with scale-dependent MSE. This does not establish divergence. Planned correction: keep nonfinite-total rejection and apply the100 bound to actual token CE, with separate auxiliary diagnostics. Retry only the two incomplete configurations after CPU/CUDA checks. Completed runs remain valid because no stopping gate fired in them.


## Retained narrative: plan.md

# NCP and independent depth campaign

## Contract, written before training

Start 2026-09-15 11:41:32 UTC; hard deadline 19:41:32 UTC. Final 90 minutes
begin at 18:11:32 UTC. Prototype implementation/check allocation ends at
13:41:32 UTC at the latest. No trial-count limit. One GPU child at a time,
900-second full-run timeout; launch only when its timeout fits before the
final 20-minute report-only interval. Enter confirmation early if needed to
fit all frozen pairs. Routine repairs are authorized; preserve failed attempts.
Stop only for shared correctness/provenance failure, insufficient resources,
or deadline. No cloud, upgrades, paid services, deletion, or new credentials.

Use the existing pinned runtime, cache seals, captured-source runner and compact
GitHub reporter. Commit only this campaign's changes and generated reports.
Publication failure must preserve local work. Campaign log is
`runs/autoresearch/ncp-20260915/campaign.log`.

## Evidence and source identity

Read the previous `runs/autoresearch/overnight-20260915/conclusions.md`.
Its 87 trials found large dense depth/width gains, but independent seeds tested
LR .03 versus .04 at depth12, with mixed signs. Depth itself lacks that check.
Identical seed42 repeats are therefore not a priority here.

Primary design source: ConceptLM, https://arxiv.org/html/2602.08984v1,
sections 2 and 3, accessed 2026-09-15. Official code repository
https://github.com/LUMIA-Group/ConceptLM pinned at
`a0ab281286f5c0337c35de3181cc992c562eacaa`, inspected locally without execution.
Relevant files: `lm_eval/lm_eval/model_arc/ConceptLM_Pythiamodel.py` and
`ConceptLM_GPT2model.py`. Record their SHA-256 in the source receipt.
No official implementation is copied or imported. The repository is an
architecture/evaluation reference, not our training dependency.

## Hypothesis and counter-hypothesis

Predicting the next multi-token latent chunk, then feeding the predicted
representation to the token decoder, may improve BPB at equal training tokens.
It may instead collapse, distort token learning, or buy only added capacity.
An auxiliary-only ablation separates supervision from feedback. A non-NCP
capacity control, if feasible, separates the added parameter budget.

## Simplified prototype, not a paper reproduction

Start with existing dense depth6/width384 and matrix LR .04. After block0,
mean-pool nonoverlapping complete k=4 chunks of hidden states. Process those
continuous chunks with two small causal Transformer blocks. Segment the output
into S=6 distributions over N=64 entries each, using independent linear heads.
Softmax probabilities weight learned codebook entries; concatenate segment
expectations. This is the predicted next concept, not a future lookup.

For chunk j covering input positions jk through (j+1)k-1, its prediction may
first affect logits at (j+1)k-1, which predict input token (j+1)k. Broadcast it
for k positions starting there. Earlier k-1 positions receive zeros. Only
complete chunks are pooled; partial-prefix inference recomputes known chunks.
Causal attention over chunks prevents later chunks from affecting earlier ones.
No KV-cache performance claim is made.

The target for prediction j is detached pooled encoder chunk j+1. Each segment
is quantized by nearest codebook entry for usage/accuracy diagnostics and VQ
training. Codebooks use fixed random bases and learned two-layer transforms,
inspired by the paper's SimVQ collapse mitigation, without that dependency.
VQ MSE trains codebook transforms toward detached pooled states. NCP MSE trains
the predicted weighted representation toward the detached next pooled state.
Token CE trains the causal predictive path, including feedback and codebooks.
No future-derived target or nearest-code assignment enters predictive logits.
No codebook update occurs inside forward; optimizer steps perform all learning.

Objective: token CE + alpha*NCP MSE + beta*VQ MSE. Initial alpha=beta=1.
Optional discrete concept CE has coefficient zero initially and is a separately
labeled later ablation only. BPB always uses token CE alone. Inference requires
only prefix tokens and stored model state; neither VQ assignment nor labels.

Differences: small existing backbone, native SDPA chunk blocks, our initialization,
normalization/positional policy and codebook transform implementation, TinyStories,
short training, and no official training recipe/checkpoint. Paper/code discrepancies
will be documented. This is a mechanism test, not semantic-concept proof.

## Gates before full trials

Test causal prefix invariance at every position including incomplete chunks;
explicit scalar broadcast alignment; perturb labels while comparing logits;
target stop-gradient; VQ gradient isolation; nonzero finite task/concept/codebook
gradients; complete optimizer coverage; shared backbone initialization; CPU/CUDA
save/load; evaluation state immutability; token-CE-only unreduced evaluation;
captured source closure execution including the new module. Run existing suite.
Use synthetic full-context fit and a small codebook-learning check. Preserve
initial failing tests and failed gates. A collapsed configuration is invalid for
promotion, not a reason to stop other valid experiments.

## Fixed protocol and bounded adaptive search

All full trials: 512 updates, 16,384 tokens/update, 8,388,608 tokens including
warmup, context512, microbatch2, BF16, checkpointing off, full attention,
65,536 validation tokens. Reuse the exact sealed batch tape, seed permutation,
step schedule, tokenizer, splits and evaluation order. Compare batch hash chains,
schedule updates and shared parameter initialization. Do not copy the tape.

Initial sequence: fresh D6 seed42, NCP seed42, auxiliary-only seed42. After gates,
prioritize NCP alone. Search only k={2,4,8}, concept layers={1,2}, codebook
entries={16,32,64,128}, insertion={0,1,2}, alpha={.03,.1,.3,1}, beta={.1,1},
module AdamW LR={.0003,.001,.003}, feedback scale={.1,.3,1}, and optional
concept-CE weight={0,.1,1}. S equals backbone heads. Start one factor at a time;
combine settings only after recorded individual evidence with a prospective
hypothesis. No MoE/memory combination is planned unless NCP and the other
component each pass and a concrete hypothesis justifies the cost.

Every trial records hypothesis, matching control, reason and current evidence
before launch. Favor diagnosing collapse or strongest marginal effects. Avoid
identical repeats after stable results. If the space becomes uninformative,
use fresh paired seeds and frozen ablations instead of recycling seed42.
Full seeds43/44 are reserved for depth confirmation. For NCP use at least two
fresh seeds not used to select its frozen configuration, initially 43/44; later
selection must not reuse inspected confirmation seeds as independent evidence.

Success screening: finite, valid, noncollapsed NCP with BPB at least .001 below
matching D6. Freeze the strongest qualifying candidate before confirmation.
Require same-sign improvement on both new seeds to call it promising replicated
evidence, still preliminary and validation-selected. Report all signs regardless.
Codebook collapse flag: any segment uses fewer than 4 entries or target assignment
perplexity <2 on the diagnostic sample. Report predictive entropy and usage too;
uniform probabilities are not proof of useful concepts.

Independent depth confirmation is mandatory: fresh D6/D12 seeds43 and44, LR .04.
Depth also changes width under the existing rule; report counts and runtime.
Equal-token results do not establish equal-time or equal-compute gains.
Repeated validation search remains exploratory, not held-out generalization.

## Resources, retention and report

Initial C: free 254,449,881,088 bytes, about237 GiB. Prior campaign checkpoint
logical storage was20.03 GiB. Estimate15-40 GiB new storage, contingent on trials.
Check free space before every launch, keep a20 GiB floor. Retain one checkpoint
per new control/ablation/confirmation and all failures; no duplicate checkpoint
copies and no deletion. Record actual campaign storage at completion.

Record total/active parameters, codebook parameters and bytes, memory table size,
timed throughput, all-update and child wall time, allocator peaks and sampled
board VRAM separately. Unknown measurements remain unknown. Final report covers
every attempt, failure, source differences, ablations, fresh-seed NCP/depth pairs,
capacity limitations and next experiment. Save and verify reports/index, commit
and push relevant code/tests/reports, verify remote HEAD. Do not start a new campaign.
