# ncp-20260915

Outcome: **running**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
campaign.log:
2026-09-15T12:00:33.731574+00:00 FAILED trial-0002-NCP-s42 RuntimeError('Training child exit 1')
2026-09-15T12:02:13.093679+00:00 DIAGNOSIS trial-0002: finite but growing total loss crossed existing100 threshold at update169. Treat alpha=beta=1 feedback configuration as invalid; no quality score. Run same-weight AUX, then lower alpha within predeclared search. No shared causal/provenance failure identified.

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

initial-queue.log:
2026-09-15T12:00:33.731574+00:00 FAILED trial-0002-NCP-s42 RuntimeError('Training child exit 1')
Traceback (most recent call last):
    if result.returncode: raise RuntimeError('Preserved failed '+label+' trial; diagnose before more trials')
RuntimeError: Preserved failed NCP trial; diagnose before more trials

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
```

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

1 completed of 3 attempted full trials. Budget: 11:41:32 to 19:41:32 UTC, 2026-09-15.
512 updates and 8,388,608 tokens per full trial. Seed42 screens are exploratory. Lower BPB is better.

## Implementation

Dense encoder pools complete multi-token chunks; causal chunk Transformers predict segmented discrete-codebook weights. Only predicted concepts feed the token decoder, delayed by k-1 positions. Detached future chunks supervise NCP MSE; VQ MSE fits a transformed frozen random codebook basis. Token BPB excludes both auxiliary losses.
This is a simplified ConceptLM-inspired prototype, not a paper reproduction. Softmax feedback differs from the official GPT2/Pythia raw-logit multiplication. Native SDPA, initialization, positional features, codebook transforms and the small TinyStories fixed-token experiment also differ. Official revision: a0ab281286f5c0337c35de3181cc992c562eacaa.
See [campaign plan](../../../docs/ncp-campaign.md) and the published source receipt for exact references.

## Every attempted trial

| Trial | Seed | BPB | Update s | Child s | Timed tok/s | Total / active params | Alloc / reserved MiB | NCP collapse |
|---|---:|---:|---:|---:|---:|---|---|---|
| [trial-0001-D6-s42](../ncp-20260915--trial-0001-D6-s42-b009c334/README.md) | 42 | 0.634848 | 177.1 | 184.0 | 47526 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0002-NCP-s42](../ncp-20260915--trial-0002-NCP-s42-22537f23/README.md) | 42 | failed | | | | | | |
| [trial-0003-AUX-s42](../ncp-20260915--trial-0003-AUX-s42-6ac0782c/README.md) | 42 | running | | | | | | |

## Depth6 versus depth12, reference LR .04

| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |
|---:|---|---|---:|---:|---:|---:|
No completed pair yet.

## Frozen NCP confirmation

| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |
|---:|---|---|---:|---:|---:|---:|
No completed pair yet.

## Attempts, decisions and failures

- trial-0001-D6-s42: Fresh depth6 matching control before NCP trials
- trial-0002-NCP-s42: Source-inspired discrete chunk prediction plus causal predicted feedback may improve BPB at equal tokens Failure: RuntimeError('Training child exit 1')
- trial-0003-AUX-s42: Initial feedback trial exceeded loss100 at update169; same-weight auxiliary-only condition tests whether feedback caused instability

## Measurement limits

Equal-token quality comparisons; measured runtime is a separate cost axis. No equal-time quality claim. Depth changes width too. Timed throughput excludes the first11 updates; all-update time includes them. Allocator peaks exclude driver/desktop use. Whole-board sampled VRAM, dictionary bytes, exact configurations, source/data/checkpoint hashes, auxiliary losses and utilization are in the JSON receipts. Codebook assignments do not prove semantic concepts. Repeated validation selection is exploratory, not held-out generalization.
All artifacts are retained locally. No cloud, dependency upgrades, paid services or deletion.
