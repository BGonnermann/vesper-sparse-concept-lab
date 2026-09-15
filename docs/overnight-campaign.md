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

The launch scheduler enters confirmation at 105 minutes remaining: the 90-minute
reserve plus one 15-minute child-timeout buffer. If confirmations finish early,
it repeats the frozen comparison until the final reporting reserve, without
changing the selected winner. Generated labels include decimal points (for
example D-lr0.02 and DG-lr0.003); shortened table labels below are descriptive.

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

Outcome-driven refinement after trial 6: continue an improving family; after a
candidate fails to improve matched-control BPB, prioritize the less-tested other
family. This avoids exhausting memory placements after repeated negative results.
The candidate space and training/evaluation protocol remain unchanged. Earlier
controller revisions are preserved locally and archived by the reporter.

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
