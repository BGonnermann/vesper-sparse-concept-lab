# Bounded equal-token comparison

The September 15 local comparison is recorded under
`runs/autoresearch/equal-token-20260915/`. Its `contract.json`, `preflight.json`,
`comparison.json`, and `report.md` define the authorization, correctness evidence,
measurements, and interpretation. Artifacts are local and intentionally ignored.

This experiment uses the dedicated captured-source `driver.py` in that directory,
not the legacy time-budget CLI, which still records seed 42. No additional smoke
training is performed: CPU/CUDA correctness and fixed-loop tests are the explicit
pre-training gate for the four authorized trials.

Each captured protocol specifies `stopping_rule: optimizer_updates`, 512 updates,
seed 42 or 43, and an explicit step schedule. The retained `training_seconds: 300`
field is legacy schema compatibility only and is **not a stopping condition** in
this mode. The external 900-second deadline remains a failure limit.

The LR multiplier is `min(1, 2 * (1 - i / 512))` at zero-based update `i`.
There is no LR warmup, matching the existing optimizer policy. The first 11
updates warm up timing and are included in the 512-update budget. Muon momentum
warms from .85 to .95 over 300 updates; its weight decay is `.2 * (1 - i / 512)`.
Every applied schedule entry and initial optimizer group is saved.

One shared CPU batch tape is produced by the existing tokenizer and best-fit
packer. An independent CPU generator seeded with the experiment seed permutes
the 8,192 microbatches. Both variants within a seed replay identical tensors;
actual consumed indices and hash chains are recorded. Tape preparation is not
part of timed training throughput. Validation uses the unchanged upstream data
loader and byte-normalized cross-entropy, without routing auxiliary loss.

The experiment seed initializes shared weights. MoE routers and extra experts
use a separate seed `4200 + experiment_seed`, preserving the previous seed-42
initialization while allowing genuinely different seed-43 initializations.
Zero-initialized projections remain zero by design.

`tests/test_fixed_updates.py` exercises the actual fixed training loop with a
small model and checks exactly 512 optimizer calls, schedule endpoints and
complete independent seeded permutations. Existing CPU/CUDA causal, routing,
gradient, checkpoint and snapshot-isolation tests remain required.

Equal training tokens do not equalize total parameter budgets or training
compute. Results from two paired seeds remain preliminary.
