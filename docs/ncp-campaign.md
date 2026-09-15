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

Source discrepancies found during review: GPT2/Pythia heads return raw logits
and multiply them directly by codebook entries, without softmax. This prototype
uses normalized weights, consistent with the paper's distribution description
but different from that code. Pythia explicitly configures a two-layer ReLU
codebook transform; GPT2 uses unspecified SimVQ defaults. Llama detaches feedback.
The training driver is unreleased and SimVQ is unpinned. Pythia computes concept
CE but returns NCP MSE as its prediction loss. We follow detached VQ inputs and
detached next-pool MSE targets, not the ambiguous printed commitment formula.

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
into S=backbone-head-count distributions over N=64 entries each, using independent linear heads.
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

## Prospective scale-control amendment, 12:10 UTC

Unit-weight feedback and auxiliary-only trials exceeded the existing loss100
limit at updates169 and213. Alpha .1 completed with BPB .635524, worse than
D6 .634848; diagnostic hidden RMS11.38 and NCP MSE24.56 suggest changing latent
scale is a confound. Alpha .03 is tested first within the original space.
Add exactly one normalization axis, pooled concept RMS normalization {none,rms},
applied identically to predictor inputs and detached labels. It does not alter
the ordinary token hidden path or use future labels in feedback. Test RMS at
the original alpha=beta=1 first, then controlled existing coefficients if justified.
This is another explicit difference from the source design, not a silent repair
of earlier results. New causality/scale checks must pass before this variant runs.

The runtime uses3 attention heads at width384, hence S=3 segments of128 values.
The initial prose incorrectly said S=6; executable code always used the pinned
backbone's head count. This correction changes no prior trial.

## Loss-gate correction, before unit-weight retries

A captured-source CPU BF16 checkpoint probe on the same samples found dense
block0 RMS12.54 and chunk4 RMS10.24. Large hidden-state scale is therefore not
NCP-specific. The initial two runs crossed a finite total-loss100 guard; their
divergence was not established. Historical hypotheses using that word are kept
as the original decision record, with this correction.

Apply the100 bound to measured token CE, while rejecting nonfinite total or
token loss. Record auxiliary MSE separately rather than interpreting its scale
as token loss. This changes termination/diagnostics, not training objectives or
updates. Retry the two incomplete configurations after new CPU/CUDA gates.
Completed trials are unaffected by a guard that never fired in them.
New failure receipts also retain initial parameter/optimizer reports, partial
checkpoint, consumed-token count and loss-component means. Synthetic save/load
checks now use in-memory checkpoints and immutable fit receipts, preserving
the existing on-disk artifacts without adding redundant checkpoint copies.

## Prospective mechanism ablations, 12:53 UTC

Prepare these isolated changes while the initial normalized one-factor screen
runs; integrate and pass fresh CPU/CUDA gates before any new variant trains.
Use N-RMS as the fixed anchor for each new axis, not a moving winner.

- Mixing `{softmax,raw_logits}`: the pinned official GPT2/Pythia code multiplies
  raw concept-head logits by the codebook. Test that source discrepancy directly.
  Raw weights can be negative and need not sum to one; both reconstructed-concept
  MSE and feedback use that reconstruction. This does not make our remaining
  architecture or training recipe a faithful reproduction.
- Feedback gain `{4,8}` in addition to the original range: N-RMS unscaled
  prediction RMS .857 versus hidden RMS13.829 gives only6.2% injected amplitude.
  Gains4/8 would initially target roughly25/50%, but retraining can change those
  ratios. Its weaker result than AUX does not establish that more gain will help.
- Prediction objective off: alpha0, concept-CE0, beta1, feedback enabled.
  This isolates supervised next-concept prediction from the added latent path
  learned through token CE and codebook fitting. Label it an ablation, not NCP
  evidence in its own right.

A feasible non-NCP capacity control is a pointwise residual MLP at the same
insertion, RMS-normalized input, two bias-free matrices and squared ReLU.
Width384 with hidden4832 adds exactly3,710,976 parameters, matching N-RMS.
Use the same added-module AdamW LR and backbone initialization. This matches
parameter capacity, not architecture or compute: the MLP runs at token rate,
while the concept Transformer runs at chunk rate. Derive the hidden size from
the selected NCP parameter count if a differently sized candidate is frozen.
Gate causality, optimizer coverage, save/load, finite gradients, and exact counts
before using this control. Record every actual hypothesis before its trial.

For the next interaction stage, combine two distinct one-factor settings only
when each improves fixed N-RMS by at least .001 BPB on seed42 and passes the
utilization gate. Retain the strongest measured setting per axis, rank pairs
by summed individual effects, and compare each pair with its stronger measured
constituent. This ranking is a hypothesis about additivity, not an estimated
result. Exclude confirmation seeds, objective-off ablations and all previously
attempted configurations. Exhaustion of this bounded space returns control for
a new evidence-backed decision; it does not end the eight-hour campaign.

## Controller archive correction, 13:30 UTC

Trial25 copied a just-edited controller file while its running parent retained
the earlier imported functions. Preserve both versions and the correction
receipt. AST checks show unchanged trial, preflight, candidate and health bodies;
the added wrapper only provides cross-process GPU exclusion. The loaded-version
reference is reconstructed from the same-process import history, not a process
memory measurement. Training-child project/upstream/configuration/data capture
is unaffected and remains independently checked. Restore the live file for the
remaining old-process trials; integrate the new controller only after that
process exits. The new controller captures its own source bytes at import and
archives those bytes, preventing later file edits from changing its receipts.

## Prospective confirmation allocation, 14:27 UTC

Before any NCP confirmation outcome is available, allocate four paired seeds:
43,44,45,46. Seeds45/46 are the primary fresh pairs; neither condition has run
on them. Seeds43/44 add sensitivity evidence using this campaign's already
completed matching dense controls. Report all four, without reselection or
extending the seed count in response to their signs. Preserve the original
same-sign rule on the two primary pairs for preliminary replication; the full
four-seed result and any mixed signs must appear alongside that conclusion.

After the declared interaction search and any separately justified bounded
extension, freeze one eligible seed42 configuration and its exact hash, source
hashes and selection-trial identity. Compare frozen feedback, AUX changing only
mode, and an exact parameter-matched residual MLP on every allocated seed.
Derive MLP hidden size from actual NCP parameter count divided by twice backbone
width; retain insertion and added-module LR. This is one capacity control, not
compute matching or universal exclusion of capacity effects. Bind report pairs
to exact trial identities and configuration/protocol hashes, not label averages.

The additional seed values require a narrow protocol-validator extension and
fresh CPU/CUDA gates after the active interaction driver exits. Do not edit
its imported training/controller sources while it runs. Confirmation has
priority over optional further exploration if time becomes constrained.

Also include a frozen prediction-objective-off condition on these same four
seeds: change only prediction_weight and ce_weight to zero, retaining feedback
and VQ fitting. This tests whether future-concept supervision is needed beyond
the added latent path. It is an ablation, never an eligible NCP candidate.

## Optional fixed-width depth decomposition, after NCP confirmation preparation

The required D6/D12 seeds43/44 already confirm a joint depth/width gain. If the
remaining budget supports it without displacing frozen NCP confirmation,
complete the dense depth{6,12} by width{384,768} grid on those same two seeds.
Existing D6-width384 and D12-width768 are reused; only D12-width384 and
D6-width768 are new. This separates depth changes at fixed width from width
changes at fixed depth. Keep matrix LR .04, batch tape/order,512updates,token
budget and evaluation fixed. No architecture selection uses these scores.

Implement only a bounded explicit-width override for these two depths and
widths, using the pinned backbone's existing configuration builder. Verify
actual width/head shapes, parameter counts, optimizer coverage and captured
execution. Pairing/audit keys must include width. Do not apply this harness
extension until the current interaction process exits and fresh gates pass.

## Prospective third-factor rule, 15:01 UTC

After the declared pair screen finishes, permit one bounded third-factor stage
only if a healthy pair improves its stronger individual constituent by at
least .001 BPB. Freeze the strongest such pair using seed42 alone. Add each
distinct remaining single-factor setting that independently improved fixed
N-RMS by at least .001, excluding tried configurations. Compare every result
with the frozen pair. Do not recursively move the anchor or add a fourth factor.
The measured gain of entries16 plus concept-CE .1 currently motivates this
conditional rule; later pair results may determine the frozen anchor instead.
Any new collapse remains disqualifying. Previously collapsed related pairs
remain negative evidence; the gate tests the actual new configuration. Confirmation
seeds are excluded from all selection decisions.

## Conditional frozen transfer, declared before NCP confirmation outcomes

If the frozen D6 NCP passes its predeclared primary same-sign rule on45/46,
and the remaining measured run forecasts fit before the report reserve,
test the same mechanism at default dense depth12/width768 without tuning.
Change only backbone depth; retain absolute insertion index, chunk size,
concept layers, entries, normalization, losses, feedback gain and module LR.
Segment count follows backbone heads, preserving segment dimension128.

Use seeds45/46 for depth12 dense, transferred NCP, mode-only AUX, prediction-
objective-off and a newly derived exact parameter-matched MLP. Verify full-
context fit and actual counts before training. These reuse primary seed
identities and the validation corpus: this is a conditional second-backbone
test, not additional independent seeds or held-out generalization. Do not
search depth12 hyperparameters or reselect the frozen D6 mechanism from these
scores. Skip this stage if its entry criterion or budget forecast fails;
retain the already planned fixed-width depth grid and final reporting.

Transfer scheduling is paired by condition: dense45/46, NCP45/46, CAP45/46,
then AUX45/46 and NOPRED45/46. Require conservative fit forecasts for the first
three conditions to fit before the90-minute reserve. The last two are optional
paired controls admitted by remaining time alone, never by transfer scores.
All D6 ablations remain mandatory. Report any transfer controls omitted for time.

## Frozen-checkpoint feedback interventions

After the four-seed confirmation, evaluate each frozen NCP checkpoint with
its original settings, feedback gain zero, and a deterministic one-row rotation
of each segment's codebook basis. These are inference interventions on an
in-memory copy, not retrained candidates. They test reliance on feedback and
the learned head-to-code mapping; they do not establish semantic concepts.

Reproduce the original BPB first, using captured source, the same tokenizer,
validation order and65536-token accounting. Record input/target batch hashes,
target bytes, checkpoint/source hashes and zero training updates. Keep
trainable weights fixed, check state immutability during each evaluation,
record the intentional buffer/settings change before it, and restore/verify
the original state afterward. Check prefix causality for every intervention.
Run under the same sequential GPU lock and per-child timeout. These results
cannot change the frozen configuration or the predeclared confirmation rule.
## Prospective initializer versus batch-order diagnosis

Planned after the seed45 sign reversal, before any crossed-order training. The frozen NCP is unchanged and cannot be reselected. Current runs couple parameter initialization and batch permutation under one seed. Cross initialization seeds42/45 with order seeds42/45, reuse the two existing diagonal dense/NCP pairs, and train only four off-diagonal runs: dense and frozen NCP at init42/order45 and init45/order42. Each consumes the same complete tape once, 512 updates and 8,388,608 tokens, with the existing schedule and validation accounting.

Add an optional explicit batch-order seed only after the active four-seed confirmation driver exits. Omission must preserve historical behavior. Validate that order changes leave every initial parameter hash unchanged, initializer changes leave a fixed order unchanged, and both permutations cover every microbatch exactly once. Freeze new source/config/protocol/data identities and reused diagonal trial/result/checkpoint/source hashes before training. Run fresh CPU/CUDA gates. Keep these runs outside selection and independent-seed confirmation tables.

Report NCP-minus-dense BPB in the 2x2 table, then descriptive initialization, order, and interaction contrasts. Two deliberately chosen seed levels diagnose this observed reversal; they are not four independent replications or held-out evidence. Diagnostic utilization samples follow the order seed and therefore are not identical across order levels. Launch only if all four forecast runs fit before the final90-minute reserve; preserve failures without automatic reselection.
## Conditional module-initialization diagnosis

Predeclared after the init42/order45 pair and before the init45/order42 pair completes. Enter only if initialization45 gives a less favorable NCP-minus-dense effect than initialization42 under both tested orders, and the mean initialization contrast exceeds0.002 BPB. This would point to initialization sensitivity without distinguishing the backbone from the NCP module.

If that condition holds and both900-second child timeouts plus preparation fit before the final90-minute reserve, add exactly the two missing module/backbone initializer cells at fixed order42: backbone42 with NCP module seed45, and backbone45 with NCP module seed42. Reuse the existing backbone42/module42 and backbone45/module45 NCP checkpoints and their matching dense controls. The complete NCP architecture, objectives, coefficients and optimizer policy remain frozen. This is initialization diagnosis, not candidate or seed selection.

A narrow optional protocol override may reinitialize only NCP before optimizer construction, using the existing initializer at12600+module_seed. It must preserve all non-NCP initial tensors and the data order; NCP initial parameters and frozen basis must match the corresponding module-seed anchor. Preserve default behavior and receipts, pin new source/data/protocol identities, and run fresh CPU/CUDA gates. No implementation enters the active training process. Report all four initializer cells and descriptive contrasts, with no independent-replication or semantic-concept claim. If the entry rule fails, record that result and skip this stage.

## Final future-supervision ablation, declared 17:33 UTC

The frozen NCP failed its primary consistency rule and did not beat the exactly parameter-matched residual MLP on mean quality or runtime. NOPRED still trains the codebook with future-derived VQ targets, so it does not isolate a latent path trained only by token prediction. Add one final diagnostic condition: keep the frozen architecture and predicted feedback, but set prediction_weight, ce_weight and vq_weight to zero. This is a token-objective-only latent-feedback control, not an NCP candidate or evidence of learned semantic concepts. Future-target diagnostics may still be computed but cannot contribute a training gradient.

Use exactly seeds45/46 and reuse their frozen NCP, NOPRED, capacity and dense controls. Compare the no-future-loss condition with NOPRED to isolate VQ supervision, and with NCP/dense/CAP to describe the complete latent path. No coefficient search or reselection follows. These reuse confirmation seeds and validation data and are not additional independent replications. Report target/prediction utilization descriptively; target-code collapse is not a validity condition for a model whose objective does not use target codes.

Freeze configuration, exact current captured sources, data/protocol identities and reused run/checkpoint/source hashes before launch. Verify matched initial backbone/NCP parameters and frozen basis, tape order, schedule, parameter count and evaluation. Add a meaningful zero-supervision gradient isolation check using the existing model mechanism without changing it. Preserve failures and do not automatically rerun. Admit both900-second timeouts plus preparation only if they fit before a fixed75-minute final reserve (18:26:32 UTC), within the requested60–90-minute range. This is the last training stage; remaining time is for the predeclared checkpoint interventions, confirmation audits and reporting.
