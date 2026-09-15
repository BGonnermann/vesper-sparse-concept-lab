# ncp-20260915

Outcome: **running**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
audit-recovered.log:
SyntaxError: '(' was never closed

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
```

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


## Retained narrative: summary.md

# NCP campaign progress

49 completed of 51 attempted full trials. Budget: 11:41:32 to 19:41:32 UTC, 2026-09-15.
512 updates and 8,388,608 tokens per full trial. Seed42 screens are exploratory. Lower BPB is better.

Strongest eligible selection-seed NCP: I2-37a8ad42c0, 0.631877 BPB, delta -0.002971 versus D6. Adds 3,655,680 parameters; measured update-time ratio 1.23. This is a search result; independent confirmation is reported separately.

## Implementation

Dense encoder pools complete multi-token chunks; causal chunk Transformers predict segmented discrete-codebook weights. Only predicted concepts feed the token decoder, delayed by k-1 positions. Detached future chunks supervise NCP MSE; VQ MSE fits a transformed frozen random codebook basis. Token BPB excludes both auxiliary losses.
This is a simplified ConceptLM-inspired prototype, not a paper reproduction. Initial softmax feedback differs from the official GPT2/Pythia raw-logit multiplication; raw-logit variants are separately labeled. Native SDPA, initialization, positional features, codebook transforms and the small TinyStories fixed-token experiment also differ. Official revision: a0ab281286f5c0337c35de3181cc992c562eacaa.
See [campaign plan](../../../docs/ncp-campaign.md) and the published source receipt for exact references.

## Every attempted trial

| Trial | Seed | BPB | Update s | Trial s | Timed tok/s | Total / active params | Alloc / reserved MiB | Target-code collapse |
|---|---:|---:|---:|---:|---:|---|---|---|
| [trial-0001-D6-s42](../ncp-20260915--trial-0001-D6-s42-b009c334/README.md) | 42 | 0.634848 | 177.1 | 184.0 | 47526 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0002-NCP-s42](../ncp-20260915--trial-0002-NCP-s42-22537f23/README.md) | 42 | failed | unavailable | 80.0 | unavailable | 30,056,748 / 30,056,748 * | 654.4 / 690.0 | unavailable |
| [trial-0003-AUX-s42](../ncp-20260915--trial-0003-AUX-s42-6ac0782c/README.md) | 42 | failed | unavailable | 82.4 | unavailable | 30,056,748 / 30,056,748 * | 654.4 / 690.0 | unavailable |
| [trial-0004-N-prediction_weight0.1-s42](../ncp-20260915--trial-0004-N-prediction_weight0.1-s42-d0ad5ff9/README.md) | 42 | 0.635524 | 202.6 | 210.3 | 41362 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0005-N-prediction_weight0.03-s42](../ncp-20260915--trial-0005-N-prediction_weight0.03-s42-cd708cd3/README.md) | 42 | 0.635888 | 203.3 | 211.0 | 41265 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0006-N-RMS-s42](../ncp-20260915--trial-0006-N-RMS-s42-d3652f3a/README.md) | 42 | 0.635927 | 214.2 | 221.8 | 39314 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0007-N-RMS-AUX-s42](../ncp-20260915--trial-0007-N-RMS-AUX-s42-d5def34d/README.md) | 42 | 0.635215 | 204.9 | 212.4 | 41139 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0008-D6-s43](../ncp-20260915--trial-0008-D6-s43-6ae7cb75/README.md) | 43 | 0.637774 | 163.7 | 170.5 | 51373 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0009-D12-s43](../ncp-20260915--trial-0009-D12-s43-3e98bde0/README.md) | 43 | 0.594495 | 308.5 | 316.9 | 27244 | 135,267,480 / 135,267,480 | 2277.2 / 2408.0 | n/a |
| [trial-0010-D6-s44](../ncp-20260915--trial-0010-D6-s44-8211e6c7/README.md) | 44 | 0.643075 | 158.5 | 165.5 | 52909 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0011-D12-s44](../ncp-20260915--trial-0011-D12-s44-afbc92ee/README.md) | 44 | 0.592813 | 323.0 | 331.3 | 26022 | 135,267,480 / 135,267,480 | 2277.2 / 2408.0 | n/a |
| [trial-0012-NCP-s42](../ncp-20260915--trial-0012-NCP-s42-bf446fb5/README.md) | 42 | 0.649647 | 220.4 | 227.8 | 38159 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0013-AUX-s42](../ncp-20260915--trial-0013-AUX-s42-bb8c5a76/README.md) | 42 | 0.652928 | 217.4 | 224.8 | 38680 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0014-R-prediction_weight0.1-s42](../ncp-20260915--trial-0014-R-prediction_weight0.1-s42-06b4af48/README.md) | 42 | 0.636121 | 220.2 | 227.7 | 38211 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0015-R-feedback_scale0.1-s42](../ncp-20260915--trial-0015-R-feedback_scale0.1-s42-23081d16/README.md) | 42 | 0.633732 | 220.0 | 226.0 | 38226 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0016-R-feedback_scale0.3-s42](../ncp-20260915--trial-0016-R-feedback_scale0.3-s42-866e46eb/README.md) | 42 | 0.634247 | 222.0 | 228.0 | 37882 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0017-R-layers1-s42](../ncp-20260915--trial-0017-R-layers1-s42-341c3f13/README.md) | 42 | 0.638393 | 207.8 | 214.1 | 40394 | 28,287,276 / 28,287,276 | 625.4 / 664.0 | False |
| [trial-0018-R-after_layer1-s42](../ncp-20260915--trial-0018-R-after_layer1-s42-93561fc2/README.md) | 42 | 0.634774 | 221.0 | 226.9 | 38057 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0019-R-after_layer2-s42](../ncp-20260915--trial-0019-R-after_layer2-s42-bbb25dff/README.md) | 42 | 0.635594 | 217.0 | 223.7 | 38695 | 30,056,748 / 30,056,748 | 655.8 / 688.0 | False |
| [trial-0020-R-chunk_size2-s42](../ncp-20260915--trial-0020-R-chunk_size2-s42-94b10b74/README.md) | 42 | 0.634780 | 219.5 | 226.8 | 38242 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0021-R-chunk_size8-s42](../ncp-20260915--trial-0021-R-chunk_size8-s42-4e711e4f/README.md) | 42 | 0.637063 | 218.2 | 224.3 | 38464 | 30,056,748 / 30,056,748 | 652.1 / 686.0 | False |
| [trial-0022-R-entries16-s42](../ncp-20260915--trial-0022-R-entries16-s42-2bda239a/README.md) | 42 | 0.633742 | 218.3 | 227.1 | 38544 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0023-R-entries32-s42](../ncp-20260915--trial-0023-R-entries32-s42-0c3fc7ea/README.md) | 42 | 0.634643 | 220.8 | 226.9 | 38057 | 30,019,884 / 30,019,884 | 653.8 / 690.0 | False |
| [trial-0024-R-entries128-s42](../ncp-20260915--trial-0024-R-entries128-s42-c922a0b0/README.md) | 42 | 0.634150 | 220.0 | 226.4 | 38188 | 30,130,476 / 30,130,476 | 656.7 / 694.0 | False |
| [trial-0025-R-lr0.0003-s42](../ncp-20260915--trial-0025-R-lr0.0003-s42-df807d5a/README.md) | 42 | 0.634800 | 221.2 | 227.3 | 37996 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0026-R-lr0.003-s42](../ncp-20260915--trial-0026-R-lr0.003-s42-45bd7ff8/README.md) | 42 | 0.635624 | 222.5 | 228.8 | 37721 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0027-R-vq_weight0.1-s42](../ncp-20260915--trial-0027-R-vq_weight0.1-s42-0864ad97/README.md) | 42 | 0.635949 | 220.5 | 226.4 | 38137 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0028-R-ce_weight0.1-s42](../ncp-20260915--trial-0028-R-ce_weight0.1-s42-5fc605e8/README.md) | 42 | 0.633991 | 220.2 | 225.9 | 38171 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0029-R-ce_weight1.0-s42](../ncp-20260915--trial-0029-R-ce_weight1.0-s42-cca8c2df/README.md) | 42 | 0.642264 | 218.4 | 224.0 | 38489 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0030-R-prediction_weight0.03-s42](../ncp-20260915--trial-0030-R-prediction_weight0.03-s42-40596925/README.md) | 42 | 0.635307 | 220.7 | 226.6 | 38110 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0031-R-prediction_weight0.3-s42](../ncp-20260915--trial-0031-R-prediction_weight0.3-s42-15ed2fdd/README.md) | 42 | 0.636698 | 217.8 | 223.6 | 38601 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0032-R-raw-s42](../ncp-20260915--trial-0032-R-raw-s42-d6f49d44/README.md) | 42 | 0.638488 | 218.4 | 225.6 | 38487 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0033-R-no_prediction-s42](../ncp-20260915--trial-0033-R-no_prediction-s42-6bdf165e/README.md) | 42 | 0.636180 | 218.6 | 224.3 | 38442 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0034-R-gain4-s42](../ncp-20260915--trial-0034-R-gain4-s42-4dc85cf9/README.md) | 42 | 0.637500 | 219.4 | 225.2 | 38248 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0035-R-gain8-s42](../ncp-20260915--trial-0035-R-gain8-s42-694ab183/README.md) | 42 | 0.635120 | 219.6 | 225.4 | 38221 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0036-CAP-L2-K64-s42](../ncp-20260915--trial-0036-CAP-L2-K64-s42-0d439659/README.md) | 42 | 0.635008 | 176.4 | 181.7 | 47592 | 30,056,748 / 30,056,748 | 683.0 / 716.0 | n/a |
| [trial-0037-I2-169ef2cc79-s42](../ncp-20260915--trial-0037-I2-169ef2cc79-s42-bd04fe9b/README.md) | 42 | 0.636130 | 214.6 | 221.8 | 39202 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0038-I2-0f670cafec-s42](../ncp-20260915--trial-0038-I2-0f670cafec-s42-9cd421a4/README.md) | 42 | 0.639392 | 218.3 | 224.0 | 38534 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0039-I2-37a8ad42c0-s42](../ncp-20260915--trial-0039-I2-37a8ad42c0-s42-ee3d34bd/README.md) | 42 | 0.631877 | 218.0 | 223.7 | 38576 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0040-I2-1c8a278808-s42](../ncp-20260915--trial-0040-I2-1c8a278808-s42-4e5a0c79/README.md) | 42 | 0.632987 | 220.7 | 226.5 | 38086 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0041-I2-253cd0d9c9-s42](../ncp-20260915--trial-0041-I2-253cd0d9c9-s42-811588a4/README.md) | 42 | 0.633320 | 218.3 | 224.0 | 38509 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0042-I2-491f8344d3-s42](../ncp-20260915--trial-0042-I2-491f8344d3-s42-0a3774ac/README.md) | 42 | 0.635269 | 218.4 | 224.1 | 38492 | 30,001,452 / 30,001,452 | 656.3 / 688.0 | False |
| [trial-0043-I2-2fd4eaacd2-s42](../ncp-20260915--trial-0043-I2-2fd4eaacd2-s42-3aab367f/README.md) | 42 | 0.634902 | 220.1 | 225.9 | 38201 | 30,001,452 / 30,001,452 | 670.2 / 694.0 | False |
| [trial-0044-I2-ae91a9e332-s42](../ncp-20260915--trial-0044-I2-ae91a9e332-s42-7f8617c8/README.md) | 42 | 0.632972 | 218.7 | 224.5 | 38454 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0045-I2-e4d0b728f2-s42](../ncp-20260915--trial-0045-I2-e4d0b728f2-s42-bf33b8c4/README.md) | 42 | 0.634788 | 214.4 | 220.2 | 39242 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | True |
| [trial-0046-I2-4919c50f43-s42](../ncp-20260915--trial-0046-I2-4919c50f43-s42-1f63b17c/README.md) | 42 | 0.632179 | 219.5 | 225.3 | 38285 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0047-I2-1736a5fa37-s42](../ncp-20260915--trial-0047-I2-1736a5fa37-s42-a4b9cecf/README.md) | 42 | 0.633942 | 218.8 | 224.6 | 38356 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0048-I2-b248268b0f-s42](../ncp-20260915--trial-0048-I2-b248268b0f-s42-df31e309/README.md) | 42 | 0.633824 | 219.4 | 225.2 | 38336 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0049-I2-9178a17b2b-s42](../ncp-20260915--trial-0049-I2-9178a17b2b-s42-a90c14b8/README.md) | 42 | 0.634048 | 218.5 | 224.2 | 38467 | 30,056,748 / 30,056,748 | 674.8 / 712.0 | False |
| [trial-0050-I2-8e935c3aca-s42](../ncp-20260915--trial-0050-I2-8e935c3aca-s42-560d02c3/README.md) | 42 | 0.635248 | 218.6 | 224.3 | 38455 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0051-I2-2ceac4edb4-s42](../ncp-20260915--trial-0051-I2-2ceac4edb4-s42-1ccc71c7/README.md) | 42 | 0.635224 | 218.4 | 224.1 | 38497 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |

* Early-failure parameter counts were reconstructed exactly on a meta device from captured source and logged model configuration. No missing performance measurement was reconstructed.

Depth6 uses width 384 and 26,345,772 parameters; depth12 uses width 768 and 135,267,480 parameters (5.13 times as many). This comparison changes both depth and width.

## Depth6 versus depth12, reference LR .04

| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |
|---:|---|---|---:|---:|---:|---:|
| 43 | D12 | D6 | 0.594495 | 0.637774 | -0.043279 | 1.88 |
| 44 | D12 | D6 | 0.592813 | 0.643075 | -0.050262 | 2.04 |

Mean paired delta: -0.046770 BPB. Mean update-time ratio: 1.96.

## Frozen NCP confirmation

| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |
|---:|---|---|---:|---:|---:|---:|
No completed pair yet.

## Feedback versus auxiliary-only ablations

Negative delta favors predicted-concept feedback. These selection-seed comparisons are exploratory.

| Feedback configuration | Auxiliary-only configuration | Seed | Feedback BPB | Auxiliary BPB | Delta |
|---|---|---:|---:|---:|---:|
| NCP | AUX | 42 | 0.649647 | 0.652928 | -0.003281 |
| N-RMS | N-RMS-AUX | 42 | 0.635927 | 0.635215 | +0.000712 |

## Attempts, decisions and failures

- trial-0001-D6-s42: Fresh depth6 matching control before NCP trials
- trial-0002-NCP-s42: Source-inspired discrete chunk prediction plus causal predicted feedback may improve BPB at equal tokens Failure: RuntimeError('Training child exit 1')
- trial-0003-AUX-s42: Initial feedback trial exceeded loss100 at update169; same-weight auxiliary-only condition tests whether feedback caused instability Failure: RuntimeError('Training child exit 1')
- trial-0004-N-prediction_weight0.1-s42: Both unit-weight variants diverged; reduce NCP prediction MSE coefficient tenfold while keeping codebook fitting and feedback unchanged
- trial-0005-N-prediction_weight0.03-s42: Alpha0.1 completed without codebook collapse but was worse than D6 by0.000676 BPB; reduce alpha to0.03 to limit auxiliary interference
- trial-0006-N-RMS-s42: After raw-latent scale growth, normalize pooled concept states at original alpha=beta=1 to test stability and token quality
- trial-0007-N-RMS-AUX-s42: Normalized-state auxiliary-only ablation isolates concept supervision from predicted feedback
- trial-0008-D6-s43: Fresh seed43 control for mandatory independent depth comparison at reference LR0.04
- trial-0009-D12-s43: Independent seed43 depth12 versus depth6; fixed tokens and LR0.04, report extra width/parameters/time
- trial-0010-D6-s44: Fresh seed44 control for mandatory independent depth comparison at reference LR0.04
- trial-0011-D12-s44: Independent seed44 depth12 versus depth6; fixed tokens and LR0.04, report extra width/parameters/time
- trial-0012-NCP-s42: Retry original unit-weight NCP after distinguishing finite auxiliary MSE from token CE in the stopping guard; same training objective and update budget
- trial-0013-AUX-s42: Retry original unit-weight AUX after distinguishing finite auxiliary MSE from token CE in the stopping guard; same training objective and update budget
- trial-0014-R-prediction_weight0.1-s42: Change only prediction_weight to 0.1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0015-R-feedback_scale0.1-s42: Change only feedback_scale to 0.1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0016-R-feedback_scale0.3-s42: Change only feedback_scale to 0.3 on normalized NCP to test quality versus its stable unit-weight control
- trial-0017-R-layers1-s42: Change only layers to 1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0018-R-after_layer1-s42: Change only after_layer to 1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0019-R-after_layer2-s42: Change only after_layer to 2 on normalized NCP to test quality versus its stable unit-weight control
- trial-0020-R-chunk_size2-s42: Change only chunk_size to 2 on normalized NCP to test quality versus its stable unit-weight control
- trial-0021-R-chunk_size8-s42: Change only chunk_size to 8 on normalized NCP to test quality versus its stable unit-weight control
- trial-0022-R-entries16-s42: Change only entries to 16 on normalized NCP to test quality versus its stable unit-weight control
- trial-0023-R-entries32-s42: Change only entries to 32 on normalized NCP to test quality versus its stable unit-weight control
- trial-0024-R-entries128-s42: Change only entries to 128 on normalized NCP to test quality versus its stable unit-weight control
- trial-0025-R-lr0.0003-s42: Change only lr to 0.0003 on normalized NCP to test quality versus its stable unit-weight control
- trial-0026-R-lr0.003-s42: Change only lr to 0.003 on normalized NCP to test quality versus its stable unit-weight control
- trial-0027-R-vq_weight0.1-s42: Change only vq_weight to 0.1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0028-R-ce_weight0.1-s42: Change only ce_weight to 0.1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0029-R-ce_weight1.0-s42: Change only ce_weight to 1.0 on normalized NCP to test quality versus its stable unit-weight control
- trial-0030-R-prediction_weight0.03-s42: Change only prediction_weight to 0.03 on normalized NCP to test quality versus its stable unit-weight control
- trial-0031-R-prediction_weight0.3-s42: Change only prediction_weight to 0.3 on normalized NCP to test quality versus its stable unit-weight control
- trial-0032-R-raw-s42: Test the pinned official implementation raw-logit codebook reconstruction against paper-inspired softmax; all other N-RMS settings fixed
- trial-0033-R-no_prediction-s42: Remove next-concept MSE while retaining VQ fitting and predicted latent feedback; test whether next-concept supervision contributes beyond the added latent path
- trial-0034-R-gain4-s42: Test stronger normalized feedback, gain4, because anchor predicted RMS is only6.2% of hidden RMS; stronger feedback remains an unproven hypothesis
- trial-0035-R-gain8-s42: Test the upper bounded feedback gain8; compare with gain1 anchor and gain4 to assess amplitude sensitivity
- trial-0036-CAP-L2-K64-s42: Parameter-matched token-level residual MLP adds exactly 3710976 parameters at the same insertion and AdamW LR as N-RMS; test added-capacity effects without concept prediction; compute is not matched
- trial-0037-I2-169ef2cc79-s42: Test interaction of entries=16 and feedback_scale=0.1; individual BPB 0.633742 and 0.633732 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0038-I2-0f670cafec-s42: Test interaction of ce_weight=0.1 and feedback_scale=0.1; individual BPB 0.633991 and 0.633732 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0039-I2-37a8ad42c0-s42: Test interaction of ce_weight=0.1 and entries=16; individual BPB 0.633991 and 0.633742 versus normalized anchor 0.635927; compare against stronger individual R-entries16
- trial-0040-I2-1c8a278808-s42: Test interaction of after_layer=1 and feedback_scale=0.1; individual BPB 0.634774 and 0.633732 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0041-I2-253cd0d9c9-s42: Test interaction of chunk_size=2 and feedback_scale=0.1; individual BPB 0.634780 and 0.633732 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0042-I2-491f8344d3-s42: Test interaction of after_layer=1 and entries=16; individual BPB 0.634774 and 0.633742 versus normalized anchor 0.635927; compare against stronger individual R-entries16
- trial-0043-I2-2fd4eaacd2-s42: Test interaction of chunk_size=2 and entries=16; individual BPB 0.634780 and 0.633742 versus normalized anchor 0.635927; compare against stronger individual R-entries16
- trial-0044-I2-ae91a9e332-s42: Test interaction of feedback_scale=0.1 and lr=0.0003; individual BPB 0.633732 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0045-I2-e4d0b728f2-s42: Test interaction of entries=16 and lr=0.0003; individual BPB 0.633742 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-entries16
- trial-0046-I2-4919c50f43-s42: Test interaction of after_layer=1 and ce_weight=0.1; individual BPB 0.634774 and 0.633991 versus normalized anchor 0.635927; compare against stronger individual R-ce_weight0.1
- trial-0047-I2-1736a5fa37-s42: Test interaction of ce_weight=0.1 and chunk_size=2; individual BPB 0.633991 and 0.634780 versus normalized anchor 0.635927; compare against stronger individual R-ce_weight0.1
- trial-0048-I2-b248268b0f-s42: Test interaction of ce_weight=0.1 and lr=0.0003; individual BPB 0.633991 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-ce_weight0.1
- trial-0049-I2-9178a17b2b-s42: Test interaction of after_layer=1 and chunk_size=2; individual BPB 0.634774 and 0.634780 versus normalized anchor 0.635927; compare against stronger individual R-after_layer1
- trial-0050-I2-8e935c3aca-s42: Test interaction of after_layer=1 and lr=0.0003; individual BPB 0.634774 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-after_layer1
- trial-0051-I2-2ceac4edb4-s42: Test interaction of chunk_size=2 and lr=0.0003; individual BPB 0.634780 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-chunk_size2

## Correctness and diagnosis

The first two unit-weight attempts hit an inherited finite total-loss100 guard. A source-verified CPU checkpoint probe found dense hidden RMS12.54 too, so those stops do not establish NCP-specific divergence. The corrected guard checks token CE separately and still rejects nonfinite total loss. Initial failed attempts remain preserved, and their historical hypotheses using the word divergence are superseded by this diagnosis. Completed BPB runs never hit that gate.
Trial25 copied a newer controller file while its long-running parent retained an earlier imported controller. Both versions and a correction receipt are retained. Their trial, preflight, candidate and health function bodies are identical; the difference is a GPU lock wrapper. The loaded-controller reference is reconstructed from the same-process import history, not direct process-memory inspection. Captured training-child sources and data are independently verified. The next controller archives immutable startup source bytes to prevent recurrence.
CPU/CUDA tests cover prefix causality, future-label isolation, VQ/encoder gradients, optimizer coverage, save/load, codebook learning and evaluation immutability. Every completed training child executes captured sources; the final evidence audit also verifies saved checkpoints and committed source-archive bytes.

## Measurement limits

Equal-token quality comparisons; measured runtime is a separate cost axis. No equal-time quality claim. Depth changes width too. Timed throughput excludes the first11 updates; all-update time includes them. Trial wall time includes preparation, child execution and verification, excluding reporting/publication. Allocator peaks exclude driver/desktop use. Whole-board sampled VRAM, dictionary bytes, exact configurations, source/data/checkpoint hashes, auxiliary losses and utilization are in the JSON receipts. Active counts describe structural training participation, not amortized per-token compute. NCP runs at chunk rate; the capacity-control MLP runs at token rate, and auxiliary-only concepts do not feed token logits. Diagnostic feedback_rms is the unscaled prediction; injected_feedback_rms applies the configured gain and is zero for auxiliary-only runs. For raw-logit mixing, reported entropy describes softmax classification probabilities, not the signed reconstruction weights. Codebook assignments do not prove semantic concepts. Repeated validation selection is exploratory, not held-out generalization.
All artifacts are retained locally. No cloud, dependency upgrades, paid services or deletion.
