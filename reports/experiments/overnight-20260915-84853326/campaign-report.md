# Overnight adaptive campaign

Status: running. Completed full trials: 3. No candidate-count cap.

All rows use 512 updates / 8,388,608 tokens, checkpointing off, context512, microbatch2, fixed validation and the same step schedule. These are equal-token comparisons; time is a separate cost axis, not an equal-time leaderboard.

| Run / label | Phase / seed | BPB | Timed tok/s | GPU update s | Child wall s | Alloc / reserved MiB | Total / active params |
|---|---|---:|---:|---:|---:|---|---|
| trial-0001-D-s42 | screen / 42 | 0.634848 | 50706.4 | 181.4 | 188.3 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0002-M-s42 | screen / 42 | 0.637217 | 27487.7 | 306.0 | 312.6 | 902.2 / 956.0 | 47,588,652 / 26,354,988 |
| trial-0003-DG-s42 | screen / 42 | 0.634178 | 44050.3 | 190.9 | 197.4 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |

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
