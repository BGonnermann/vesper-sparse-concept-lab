# Frozen causal n-gram implementation

The user authorized the previously proposed plan without research-design changes.
Before implementation, the plan was frozen under
`runs/autoresearch/ngram-20260915/frozen-plan.md`, with its revision in
`plan-revision.json`: base commit `7f2ceab77d69d2e4be70969a4cae4c34d82790a9`,
plan SHA256 `e8d7872c21fc7b5d5df561863ce0f7724730fb0da643724ce8053cf14e25a71a`.
The original plan document remains unchanged as the accepted specification.

## Implementation and gates

`scripts/autoresearch_memory.py` implements the two frozen8192x64 FP32 tables,
polynomial-mod-v1 causal suffix hash,128-to384 projection and scalar contextual
gate. Invalid addresses use a -1 diagnostic sentinel; the gather clamps them to0
and masks their vectors and gradients to zero. This does not reserve a trainable
padding row or change valid suffix addresses.

`scripts/autoresearch_model.py` inserts the memory residual after block1, preserves
the memory-off backbone, and initializes added parameters under seed8400+seed.
All added parameters use the specified independent AdamW group. The sealed
upstream optimizer's fixed backbone partition is constructed with the independent
memory module temporarily unregistered, then restored and explicitly covered.
No upstream source or attention/tokenizer behavior was changed.

`tests/test_ngram_memory.py` covers independent scalar keys, collisions, empty
histories, BOS and row isolation, matching-precision output/gradient references,
future suffix interventions, future-position activation gradients, task/router
gradients, selected/unselected table gradients, optimizer membership/updates,
checkpoint recomputation, initialization pairing, parameter counts, read-only
validation and lookup state/reset/reorder. A CPU Adam/RNG/sampler fixture tests
exact small-state resume; it is not a whole-model training-resume implementation.

The full preflight passed46 tests on CPU and46 on CUDA, with original numerical
checks unchanged. It additionally verified all full-depth parameter counts,
paired initialization hashes, unchanged backbone optimizer groups, the complete
8192-microbatch tape/order and BOS/shifted-target alignment. The existing captured
packer prepends BOS to every document, including cropped documents; it emits no
separate boundary metadata. The new module belongs to the captured import closure
and its live-edit isolation regression.

## Execution and measurement

`scripts/ngram_experiment.py` enforces the four-condition D,DG,M,MG order, seed42,
512 updates including warmup, exact8,388,608 training tokens and900-second child
timeout. It opens each log before its start-marker gate, captures sources/config,
validates receipts before the next condition and emits compact reports on success
or failure. Preflight hashes must match the sources used for training.

The fixed-update learning-rate schedule, data tape/order and evaluation are copied
from the accepted equal-token reference. Publication does not replace historical
execution provenance with a later commit hash.

Expensive complete-tape occupancy/collision analysis runs in preflight. Gradient
and parameter-delta norms are sampled at the final optimizer update, outside the
timed segments. Gate diagnostics use32 inference-only training-tape microbatches
after training, without targets or optimizer updates. These samples are not an
all-training gate distribution. BPB remains the unmodified held-out CE/byte metric.
Peak allocated/reserved VRAM covers the complete process, including diagnostics;
timed throughput excludes those diagnostics. Run wall time includes them.

`scripts/ngram_report.py` checks four completed receipts and all708 previously
recorded artifact hashes, computes DG-D and MG-M, and writes the final compact
comparison. One seed, fixed order and unequal added parameter budgets limit any
causal-memory or efficiency claim. No NCP, extra seeds or tuning is part of this run.
