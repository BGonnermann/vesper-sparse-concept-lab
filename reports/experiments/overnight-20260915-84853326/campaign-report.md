# Overnight adaptive campaign

Status: running. Completed full trials: 39. No candidate-count cap.

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
| trial-0025-D-depth3-s42 | screen / 42 | 0.686539 | 87045.3 | 96.6 | 102.5 | 307.6 / 316.0 | 10,748,038 / 10,748,038 |
| trial-0026-DG-lr0.003-s42 | screen / 42 | 0.634195 | 43748.3 | 192.1 | 198.7 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0027-DG-lr0.0001-s42 | screen / 42 | 0.635100 | 44113.3 | 190.6 | 197.2 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0028-DG-lr0.0003-s42 | screen / 42 | 0.635031 | 45520.0 | 184.9 | 191.4 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0029-DG-lr0.01-s42 | screen / 42 | 0.634524 | 44118.0 | 190.4 | 196.8 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0030-DG-s42 | replication / 42 | 0.634178 | 44653.1 | 188.2 | 194.6 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |
| trial-0031-D-depth8-s42 | replication / 42 | 0.617737 | 37752.4 | 222.6 | 228.8 | 943.1 / 964.0 | 50,332,176 / 50,332,176 |
| trial-0032-D-depth10-s42 | screen / 42 | 0.609964 | 31103.6 | 270.3 | 276.8 | 1511.2 / 1602.0 | 85,852,980 / 85,852,980 |
| trial-0033-D-depth12-s42 | screen / 42 | 0.590324 | 26799.0 | 313.4 | 320.1 | 2277.2 / 2408.0 | 135,267,480 / 135,267,480 |
| trial-0034-D-depth12-s42 | replication / 42 | 0.590240 | 26842.4 | 313.0 | 319.7 | 2277.2 / 2408.0 | 135,267,480 / 135,267,480 |
| trial-0035-D-depth10-s42 | replication / 42 | 0.610179 | 30800.1 | 272.7 | 279.2 | 1511.2 / 1602.0 | 85,852,980 / 85,852,980 |
| trial-0036-D-depth12-s42 | replication / 42 | 0.590862 | 26935.0 | 312.0 | 318.7 | 2277.2 / 2408.0 | 135,267,480 / 135,267,480 |
| trial-0037-D-depth10-s42 | replication / 42 | 0.610642 | 30826.4 | 272.5 | 279.0 | 1511.2 / 1602.0 | 85,852,980 / 85,852,980 |
| trial-0038-D-s42 | replication / 42 | 0.634848 | 48748.3 | 172.4 | 178.4 | 581.7 / 622.0 | 26,345,772 / 26,345,772 |
| trial-0039-DG-s42 | replication / 42 | 0.634178 | 44092.4 | 190.7 | 197.3 | 599.7 / 624.0 | 27,443,885 / 26,395,437 |

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
