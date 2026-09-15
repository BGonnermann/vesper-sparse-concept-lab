# Longer-training depth campaign progress

Recommendation under the frozen validation rule: **inconclusive**. Complete paired seeds: [].

TinyStories only. Equal-token comparisons are not equal-compute comparisons. Test results never drive selection.

[Frozen plan](PLAN.md) · [Exact measurements](measurements.csv) · [Reproducibility receipts](report.json)

## Every completed run

| Seed | Depth | Budget | Validation BPB | Actual tokens | Update s | Trial s | Timed tok/s | Parameters | Allocated / reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Equal-token paired differences

| Seed | Budget | D12 minus D6 BPB | D12 / D6 update time |
|---:|---:|---:|---:|

## Approximately time-matched comparisons

| Seed | D6 budget | D12 budget | D6 minus D12 BPB | D6 / D12 update time |
|---:|---:|---:|---:|---:|

These are actual unequal-time endpoints. The frozen approximate-match tolerance is0.85–1.15 per pair; no exact equal-time quality is inferred. Both contrast means must improve by at least0.001 BPB, with every seed agreeing, for a directional recommendation.

## Protocol and interpretation

Each point starts from initialization. LR and Muon weight decay follow progress i/N; momentum warms over300 absolute updates. This is a family of independently annealed budget endpoints. The final applied LR is2/N of base. Matrix LR0.04, BF16, context512, batch16,384 tokens, microbatch2 and checkpointing off are fixed.

Update time includes all updates; timed throughput excludes the first11 updates. Trial wall time also includes setup, validation and artifact verification, excluding publication. Allocator peaks exclude driver and desktop allocations; whole-board samples remain in each trial receipt. All parameters are active; no memory table or auxiliary architecture is used.


## Training stream

Tape: 32,768 microbatches, 33,554,432 tokens, epochs [1]. Exact duplicate microbatches: 0. 8,192 microbatches also occur in the previous campaign tape. All32,768 rows independently reconstructed. Each run consumes distinct indices; separate models/budgets reuse data by design. Near-duplicates and repeated phrases are not excluded.

## Failures and retained evidence

No training failures recorded.

## Final test stage

Pending. Exact candidate/reference checkpoint identities will be frozen before test evaluation; no subsequent training or reselection.

## Reproducibility

Frozen matrix SHA256: ca3df2755225acb142e1ac176c2cb3908447d4220ca298c2794f848fb99ba5a4. Every trial retains its checkpoint, source snapshot, data/config hashes, consumed-order hash chain, optimizer coverage, schedule and immutable-evaluation receipt. See the progress index for individual trial reports. Existing artifacts and production baseline remain unchanged.
