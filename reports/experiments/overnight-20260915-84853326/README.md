# overnight-20260915

Outcome: **running**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
preflight.log:
ok
ok
ok
ok
ok
Ran 52 tests in 6.444s
OK
cuda correctness exit: 0
```

## Retained narrative: campaign-report.md

# Overnight adaptive campaign

Status: running. Completed full trials: 24. No candidate-count cap.

All rows use 512 updates / 8,388,608 tokens, checkpointing off, context512, microbatch2, fixed validation and the same step schedule. These are equal-token comparisons; time is a separate cost axis, not an equal-time leaderboard.

| Run / label | Phase / seed | BPB | Timed tok/s | GPU update s | Child wall s | Alloc / reserved MiB | Total / active params |
|---|---|---:|---:|---:|---:|---|---|
| trial-0001-D-s42 | screen / 42 | 0.634848 | 50706.4 | 181.4 | 188.3 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0002-M-s42 | screen / 42 | 0.637217 | 27487.7 | 306.0 | 312.6 | 902.2 / 956.0 | 47,588,652 / 26,354,988 |
| trial-0003-DG-s42 | screen / 42 | 0.634178 | 44050.3 | 190.9 | 197.4 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0004-DG-layer3-s42 | screen / 42 | 0.634633 | 43597.1 | 192.9 | 199.5 | 602.2 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0005-DG-layer0-s42 | screen / 42 | 0.634020 | 43803.4 | 192.0 | 198.6 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0006-DG-layer2-s42 | screen / 42 | 0.634624 | 43845.9 | 191.8 | 198.3 | 600.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0007-D-lr0.02-s42 | screen / 42 | 0.635645 | 48507.6 | 173.2 | 179.3 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0008-M-experts2-s42 | screen / 42 | 0.634345 | 31489.2 | 267.2 | 273.6 | 703.8 / 756.0 | 33,428,268 / 26,350,380 |
| trial-0009-M-router_lr0.0003-s42 | screen / 42 | 0.638516 | 26640.3 | 315.7 | 322.3 | 900.4 / 954.0 | 47,588,652 / 26,354,988 |
| trial-0010-D-lr0.06-s42 | screen / 42 | 0.640004 | 48117.8 | 174.6 | 180.8 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0011-M-router_lr0.003-s42 | screen / 42 | 0.633858 | 26756.5 | 314.4 | 320.8 | 901.6 / 956.0 | 47,588,652 / 26,354,988 |
| trial-0012-M-aux_loss_weight0.003-s42 | screen / 42 | 0.633895 | 26673.5 | 315.4 | 321.9 | 902.0 / 952.0 | 47,588,652 / 26,354,988 |
| trial-0013-M-aux_loss_weight0.03-s42 | screen / 42 | 0.642550 | 26780.4 | 314.0 | 320.7 | 900.9 / 958.0 | 47,588,652 / 26,354,988 |
| trial-0014-D-lr0.01-s42 | screen / 42 | 0.644417 | 52354.8 | 160.5 | 167.1 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0015-DG-layer4-s42 | screen / 42 | 0.635105 | 47639.6 | 176.5 | 183.4 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0016-D-lr0.03-s42 | screen / 42 | 0.634627 | 48740.4 | 172.4 | 178.6 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0017-D-depth4-s42 | screen / 42 | 0.671612 | 68345.5 | 123.0 | 129.0 | 330.6 / 350.0 | 11,534,472 / 11,534,472 |
| trial-0018-DG-layer5-s42 | screen / 42 | 0.635327 | 43671.9 | 192.4 | 199.0 | 603.1 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0019-D-lr0.05-s42 | screen / 42 | 0.637371 | 48768.5 | 172.3 | 178.5 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0020-D-lr0.08-s42 | screen / 42 | 0.643495 | 48374.3 | 173.7 | 179.8 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0021-D-depth8-s42 | screen / 42 | 0.617737 | 37777.0 | 222.3 | 228.5 | 943.1 / 964.0 | 50,332,176 / 50,332,176 |
| trial-0022-D-depth7-s42 | screen / 42 | 0.625398 | 41734.3 | 201.4 | 207.5 | 873.1 / 904.0 | 47,186,446 / 47,186,446 |
| trial-0023-D-depth5-s42 | screen / 42 | 0.643525 | 56091.6 | 149.8 | 155.8 | 531.8 / 578.0 | 24,576,298 / 24,576,298 |
| trial-0024-D-depth2-s42 | screen / 42 | 0.786799 | 123867.0 | 68.0 | 73.7 | 200.0 / 204.0 | 3,538,980 / 3,538,980 |

Timed throughput excludes the first11 updates; all-update GPU time includes them. Child wall includes startup, diagnostics, checkpoint saving and evaluation. PyTorch allocator memory excludes other processes/driver allocations. Exact widths, memory-table bytes, expert utilization and per-trial selection reasons are in compact JSON and child receipts.

## Confirmation

Confirmation pending or no frozen comparison.

Averaging repeats within a seed does not create new independent seeds. Three paired seeds, when complete, remain preliminary; repeated validation-guided selection is not an untouched test.

## Boundaries and provenance

No NCP, architecture combinations, dependency changes, cloud jobs, paid services or deletion. Each training child executes captured source/configuration; publishing commits are not retrospective execution provenance. Failed/invalid trials are retained and excluded from quality rankings. Hypotheses were saved before process launch.

## Failures

None.

## Preservation

Full preservation verification reserved for final report.


## Retained narrative: frozen-plan-v2.md

# Bounded overnight campaign

Authorized 2026-09-15. This replaces the old three-candidate/dense-only program
for this invocation, not for unrelated future work.

## Contract

- Conservative session start: 2026-09-15 03:30 UTC. Hard end: 11:30 UTC (8 hours).
  Research, implementation, tests, diagnosis, reporting and publication count.
- User amendment: no candidate-count limit. Continue adaptive screening and
  replication until the eight-hour budget is nearly exhausted, reserving 90
  minutes for confirmation trials and reporting. Completed search-space coverage
  is not a stopping condition: use remaining screening time for informative
  repetitions of promising candidates and their controls.
- Each training child has a 900-second hard timeout. Reserve 30 minutes for
  reporting; launch only if the child timeout fits before that reserve.
- Native Windows, existing RTX 5070 Ti/runtime and TinyStories seals. No cloud,
  new dependencies, upgrades, paid services, downloads or deletion.
- Stop on unexplained runtime/correctness failure; preserve failure evidence.
  At most three evidence-backed implementation repair iterations. Never weaken
  causal, gradient, evaluation or provenance checks to pass a candidate.
- Every trial creates its log and records the hypothesis/selection reason before
  launch. The assistant announces its exact PowerShell tail command, then opens
  the start gate without waiting for the user.

## Evidence and design

Inspected docs/research.md, architecture.md, experiments.md, autoresearch.md,
research-publication.md, the executed n-gram diagnostic report and current code.
Local results favor inexpensive dense training; four-expert routing incurred
large dispatch/optimizer overhead. Memory gates saturate but residual RMS is
under 1%, and its small BPB effects are below the unexplained historical M shift.

Prioritize matrix LR, depth/width, fewer experts and memory placement. These
change one factor against a fresh matching control. Keep attention, tokenization,
batching, precision, optimizer policy except the named LR, and evaluation fixed.
The two-expert variant retains dropless packed top-1 dispatch and its existing
auxiliary loss. The memory variant changes only insertion location, or only its
AdamW LR. Neither is a faithful new Engram reproduction.

NCP is deferred, not implemented: the project's sources leave target alignment,
codebook training and loss policy unresolved. Inventing those simultaneously is
less justified than cheap controlled changes with this campaign's confirmation
budget. No category is included merely to fill a checklist.

## Frozen search space

| ID | Change | Matching control | Hypothesis |
|---|---|---|---|
| D | Dense depth 6, matrix LR .04 | fresh control | Current fast backbone reference |
| M | Four experts, top-1, depth 6 | fresh control | Sparse reference and overhead |
| DG | D plus original memory after block 1 | D | Fresh memory reference |
| D-lr02 | Matrix LR .02 | D | Gentler matrix updates improve short-run quality |
| D-lr06 | Matrix LR .06 | D | Faster optimization improves fixed-token quality |
| D-depth4 | Depth 4, coupled width | D | Less compute gives a better cost/quality frontier |
| D-depth8 | Depth 8, coupled width | D | More capacity buys enough quality to justify cost |
| M-experts2 | Two experts, same top-1 | M | Less optimizer/dispatch cost without major quality loss |
| DG-layer3 | Memory after block 3 instead of 1 | DG | Later context makes the small residual more useful |
| DG-lr003 | Memory AdamW LR .003 instead of .001 | DG | Faster memory adaptation improves its contribution |

No combinations in this campaign. Each condition varies one named factor.
Depth also changes width through the existing upstream shape rule; exact counts
and actual width are recorded, not inferred from active parameters.

After the initial conditions, the documented one-factor space also permits
dense matrix LR {.01,.03,.05,.08}, dense depths {2,3,5,7}, four-expert router LR
{.0003,.003}, four-expert auxiliary weight {.003,.03}, dense-memory insertion
blocks {0,2,4,5}, and dense-memory LR {.0001,.0003,.01}. Every alternative starts
from its D, M or DG control, never from a combination of discovered winners.
After all eligible choices are screened, repeat the strongest candidate/control
tradeoffs with seed 42 to estimate execution variability; these repetitions are
not independent seeds. Freeze one candidate before final seeds 43/44.

## Protocol and adaptive selection

All screening and confirmation runs use 512 optimizer updates including warmup,
16,384 tokens/update (8,388,608 total), checkpointing off, context 512, microbatch
2, BF16 AMP, full attention, fixed 65,536-token validation, and the existing
step-based schedule. Replay the same complete training tape with seed-specific
permutations. Seed controls initialization and ordering. Seed 42 screens; seeds
43 and 44 confirm one selected candidate and its matching control (three paired
seeds total). Preserve all original artifacts and identify actual executed code.

Run fresh controls first. Then prioritize unscreened families by the observed
control BPB/elapsed-time frontier: dense LR and shape first when sparse/memory
controls do not improve quality enough to justify their cost; otherwise prioritize
the respective sparse/memory alternative. Within dense LR options, inspect the
first outcome before choosing the other direction or shape. The next invocation
records actual completed evidence and a concrete reason in selection.json before
creating its child process. No candidate is selected from outside this table.

Confirmation selects the strongest measured candidate/control tradeoff: prefer
BPB improvement without more than 10% extra all-update GPU time; otherwise
consider at least 10% time saving with no more than .001 BPB degradation. If none
qualifies, report that no candidate qualified; do not invent a winner. Confirmed
means measured over three seeds, not statistically established superiority.

These are **equal-token** comparisons with measured time as a second axis. Do
not divide BPB by seconds, rank different training budgets on one leaderboard,
or claim an equal-time quality gain. No equal-time runs are planned. Selection
uses this validation slice repeatedly, so validation overfitting and winner's
curse remain limitations; no new untouched test result is claimed.

## Gates and reporting

Before trials run CPU/CUDA unit suites, including new-option validation, causal
prefix invariance, task/router/memory gradients, optimizer coverage, state-dict
roundtrip, checkpoint recomputation, routing counts/no dropping and CE-only eval.
Fixed-update tests remain the bounded smoke substitute used by the existing
equal-token harness; they are not full trial scores.

Verify each training run's captured execution, token count, complete batch order,
step schedule, parameter reports and finite metrics before selecting another.
Publish compact results/index using experiment_reports.py, retaining failed runs,
configuration hashes and original source snapshots. Exclude weights/data/logs
and traces from Git. Publishing commits never replace executed-source provenance.


## Retained narrative: frozen-plan.md

# Bounded overnight campaign

Authorized 2026-09-15. This replaces the old three-candidate/dense-only program
for this invocation, not for unrelated future work.

## Contract

- Conservative session start: 2026-09-15 03:30 UTC. Hard end: 11:30 UTC (8 hours).
  Research, implementation, tests, diagnosis, reporting and publication count.
- At most 20 full training trials, including controls and confirmation repeats.
  The documented search contains 10 screening conditions, followed by at most
  four confirmation trials. Eight hours is a ceiling, not a utilization target.
- Each training child has a 900-second hard timeout. Reserve 30 minutes for
  reporting; launch only if the child timeout fits before that reserve.
- Native Windows, existing RTX 5070 Ti/runtime and TinyStories seals. No cloud,
  new dependencies, upgrades, paid services, downloads or deletion.
- Stop on unexplained runtime/correctness failure; preserve failure evidence.
  At most three evidence-backed implementation repair iterations. Never weaken
  causal, gradient, evaluation or provenance checks to pass a candidate.
- Every trial creates its log and records the hypothesis/selection reason before
  launch. The assistant announces its exact PowerShell tail command, then opens
  the start gate without waiting for the user.

## Evidence and design

Inspected docs/research.md, architecture.md, experiments.md, autoresearch.md,
research-publication.md, the executed n-gram diagnostic report and current code.
Local results favor inexpensive dense training; four-expert routing incurred
large dispatch/optimizer overhead. Memory gates saturate but residual RMS is
under 1%, and its small BPB effects are below the unexplained historical M shift.

Prioritize matrix LR, depth/width, fewer experts and memory placement. These
change one factor against a fresh matching control. Keep attention, tokenization,
batching, precision, optimizer policy except the named LR, and evaluation fixed.
The two-expert variant retains dropless packed top-1 dispatch and its existing
auxiliary loss. The memory variant changes only insertion location, or only its
AdamW LR. Neither is a faithful new Engram reproduction.

NCP is deferred, not implemented: the project's sources leave target alignment,
codebook training and loss policy unresolved. Inventing those simultaneously is
less justified than cheap controlled changes with this campaign's confirmation
budget. No category is included merely to fill a checklist.

## Frozen search space

| ID | Change | Matching control | Hypothesis |
|---|---|---|---|
| D | Dense depth 6, matrix LR .04 | fresh control | Current fast backbone reference |
| M | Four experts, top-1, depth 6 | fresh control | Sparse reference and overhead |
| DG | D plus original memory after block 1 | D | Fresh memory reference |
| D-lr02 | Matrix LR .02 | D | Gentler matrix updates improve short-run quality |
| D-lr06 | Matrix LR .06 | D | Faster optimization improves fixed-token quality |
| D-depth4 | Depth 4, coupled width | D | Less compute gives a better cost/quality frontier |
| D-depth8 | Depth 8, coupled width | D | More capacity buys enough quality to justify cost |
| M-experts2 | Two experts, same top-1 | M | Less optimizer/dispatch cost without major quality loss |
| DG-layer3 | Memory after block 3 instead of 1 | DG | Later context makes the small residual more useful |
| DG-lr003 | Memory AdamW LR .003 instead of .001 | DG | Faster memory adaptation improves its contribution |

No combinations in this campaign. Each condition varies one named factor.
Depth also changes width through the existing upstream shape rule; exact counts
and actual width are recorded, not inferred from active parameters.

## Protocol and adaptive selection

All screening and confirmation runs use 512 optimizer updates including warmup,
16,384 tokens/update (8,388,608 total), checkpointing off, context 512, microbatch
2, BF16 AMP, full attention, fixed 65,536-token validation, and the existing
step-based schedule. Replay the same complete training tape with seed-specific
permutations. Seed controls initialization and ordering. Seed 42 screens; seeds
43 and 44 confirm one selected candidate and its matching control (three paired
seeds total). Preserve all original artifacts and identify actual executed code.

Run fresh controls first. Then prioritize unscreened families by the observed
control BPB/elapsed-time frontier: dense LR and shape first when sparse/memory
controls do not improve quality enough to justify their cost; otherwise prioritize
the respective sparse/memory alternative. Within dense LR options, inspect the
first outcome before choosing the other direction or shape. The next invocation
records actual completed evidence and a concrete reason in selection.json before
creating its child process. No candidate is selected from outside this table.

Confirmation selects the strongest measured candidate/control tradeoff: prefer
BPB improvement without more than 10% extra all-update GPU time; otherwise
consider at least 10% time saving with no more than .001 BPB degradation. If none
qualifies, report that no candidate qualified; do not invent a winner. Confirmed
means measured over three seeds, not statistically established superiority.

These are **equal-token** comparisons with measured time as a second axis. Do
not divide BPB by seconds, rank different training budgets on one leaderboard,
or claim an equal-time quality gain. No equal-time runs are planned. Selection
uses this validation slice repeatedly, so validation overfitting and winner's
curse remain limitations; no new untouched test result is claimed.

## Gates and reporting

Before trials run CPU/CUDA unit suites, including new-option validation, causal
prefix invariance, task/router/memory gradients, optimizer coverage, state-dict
roundtrip, checkpoint recomputation, routing counts/no dropping and CE-only eval.
Fixed-update tests remain the bounded smoke substitute used by the existing
equal-token harness; they are not full trial scores.

Verify each training run's captured execution, token count, complete batch order,
step schedule, parameter reports and finite metrics before selecting another.
Publish compact results/index using experiment_reports.py, retaining failed runs,
configuration hashes and original source snapshots. Exclude weights/data/logs
and traces from Git. Publishing commits never replace executed-source provenance.
