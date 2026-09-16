# Final held-out test: all15pilot models

One frozen test stage; same32selected documents, exact UTF-8 byte accounting, context512. No training or reselection followed test opening. All rows have135,267,480total/active parameters,8,388,608training tokens and checkpointing off.

| Run | Test aggregate BPB | General BPB | Technical BPB | Eval tok/s | Eval allocated MiB |
|---|---:|---:|---:|---:|---:|
| pilot-control-s201 | 1.487855 | 1.544599 | 1.414218 | 30164 | 714.6 |
| pilot-mixture-s201 | 1.425510 | 1.592771 | 1.208456 | 30270 | 714.6 |
| pilot-tokenizer-s201 | 1.367550 | 1.473375 | 1.230220 | 27528 | 706.7 |
| pilot-control-s202 | 1.455895 | 1.523435 | 1.368249 | 29543 | 714.6 |
| pilot-mixture-s202 | 1.416285 | 1.575615 | 1.209522 | 30070 | 714.6 |
| pilot-tokenizer-s202 | 1.355917 | 1.470844 | 1.206775 | 27292 | 706.7 |
| pilot-control-s203 | 1.462492 | 1.524254 | 1.382344 | 29532 | 714.6 |
| pilot-mixture-s203 | 1.418838 | 1.582792 | 1.206074 | 30563 | 714.6 |
| pilot-tokenizer-s203 | 1.365267 | 1.479468 | 1.217067 | 26321 | 706.7 |
| data-old-s211 | 1.366929 | 1.475140 | 1.226503 | 26937 | 706.7 |
| data-expanded-s211 | 1.330591 | 1.431370 | 1.199811 | 25707 | 706.7 |
| data-old-s212 | 1.354404 | 1.468148 | 1.206798 | 27128 | 706.7 |
| data-expanded-s212 | 1.330746 | 1.431396 | 1.200132 | 27169 | 706.7 |
| data-old-s213 | 1.362256 | 1.477066 | 1.213266 | 26888 | 706.7 |
| data-expanded-s213 | 1.344401 | 1.447575 | 1.210512 | 28346 | 706.7 |

Exact per-document scores, nats, bytes, token counts, memory, checkpoint/source identities and paired gate calculations: [final-test-results.json](final-test-results.json). Evaluation-only memory excludes optimizer state and differs from training peak.
