# Dense foundation campaign: verified pilot results

**Keep the frozen dense architecture and original mainline tokenizer.** The new data/evaluation foundation is operational. The comparisons below are small-corpus pilots, not evidence of general reasoning or coding capability. No architecture, mixture or tokenizer is automatically promoted.

## Verified facts

- Frozen dense D12/width768, LR.04, BF16, context512, microbatch2, batch16,384, checkpointing off. Total/active parameters135,267,480; memory table0.
- Pinned WikiText-2 raw and CPython documentation:37,263,470downloaded bytes;1,049accepted documents; corpus fingerprint `debe7de4f51d9487e316aedf1a7dc828350d7a77f20d99a913a0143e456b121f`.
- Current tokenizer SHA256 `4d1991faca1391dbc13ba13ef7ed19a3ae77dde50d4a6927fd068090b80cda5c`; candidate `736bb1a35ce3b2a3143095bfc1390858d67dd7f606fda25137b83095bace0102`. Both8192tokens; candidate retraining was byte-identical.
- Every paired run passed parameter, initialization, token-budget, schedule, data, evaluator-input and checkpoint checks; every validation checkpoint was independently replayed. Exact code/runtime/hardware/source hashes and per-domain losses are in [results.json](results.json).

## Baseline reproduction

Seed101,8,388,608tokens: legacy TinyStories BPB **0.589531**, reference.589157, absolute difference0.000374. Timed throughput25798tok/s; wall334.9s; peak allocated/reserved2277.2/2408.0MiB. The wrapper's missing-metadata failure is retained separately from the successful artifact verification.

## Controlled results

Each row:512updates,8,388,608tokens, independent initialization. Same-token comparisons are not same-byte exposure or equal compute. Primary metric is aggregate exact-byte BPB; lower is better. Allocator memory excludes desktop/driver use.

| Arm | Seed | Aggregate BPB | General BPB | Technical BPB | Timed tok/s | Update s | Wall s | Alloc/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 201 | 1.429069 | 1.496914 | 1.358454 | 25407 | 330.2 | 344.3 | 2277.2/2408.0 |
| mixture | 201 | 1.367413 | 1.564995 | 1.161759 | 26033 | 322.3 | 336.2 | 2277.2/2408.0 |
| tokenizer | 201 | 1.305620 | 1.422883 | 1.183566 | 25412 | 330.1 | 341.8 | 2277.2/2408.0 |
| control | 202 | 1.397499 | 1.478392 | 1.313302 | 25474 | 329.3 | 342.7 | 2277.2/2408.0 |
| mixture | 202 | 1.356937 | 1.547788 | 1.158289 | 25619 | 327.5 | 341.2 | 2277.2/2408.0 |
| tokenizer | 202 | 1.295521 | 1.416601 | 1.169495 | 25552 | 328.4 | 339.7 | 2277.2/2408.0 |
| control | 203 | 1.412444 | 1.482412 | 1.339619 | 25499 | 329.2 | 342.6 | 2277.2/2408.0 |
| mixture | 203 | 1.363257 | 1.551891 | 1.166917 | 25341 | 331.0 | 344.3 | 2277.2/2408.0 |
| tokenizer | 203 | 1.301785 | 1.423551 | 1.175046 | 25371 | 330.9 | 342.3 | 2277.2/2408.0 |

Control=current tokenizer80/20general/technical source token mass. Mixture=current50/50. Tokenizer=candidate80/20. Actual sampled masses and repeated corpus passes are in each stream receipt.

**mixture vs control:** mean paired BPB delta-0.050469, paired SD0.010605; deltas[-0.061656, -0.040562, -0.049188]. Domain mean deltas{'general': 0.0689857591554613, 'technical': -0.17480312030128653}. Predeclared screen passed: **False**. No promotion.

**tokenizer vs control:** mean paired BPB delta-0.112029, paired SD0.010801; deltas[-0.12345, -0.101978, -0.110659]. Domain mean deltas{'general': -0.06489414212987255, 'technical': -0.1610889409268085}. Predeclared screen passed: **True**. No promotion.

## Inferences and limitations

- Data ingestion trains cleanly. There is no fair TinyStories-only versus new-corpus training comparison, so these results do **not** establish that more data helped.
- Tokenizer compression, raw-byte exposure, context in bytes and repeated passes differ. Only the recorded paired results establish improvement or regression on this held-out pilot; no wider capability claim follows.
- The legacy per-token decoded Unicode byte table overcounts split UTF-8 pieces. New evaluation uses exact document bytes. Historical BPB is preserved, not pooled with the new series.
- RST/Wikipedia markup, small topic coverage, coarse language filtering and lexical rather than semantic dedup remain limitations. Test corpus model scores remain unopened.

## Failures

- Baseline wrapper omitted condition metadata after successful training; original failed result retained, independent correction receipt verifies the artifacts.
- First new-path smoke hit noncontiguous target tensors in the pinned model before completing an update; contiguous copies fixed it; a new smoke completed.
- Initial Git commit failed because author identity was unset. Command-local campaign assistant identity was used; no global Git configuration changed.

## Decisions made

- Dense mainline; experimental NCP/MoE/n-grams preserved and excluded.
- Preserve original tokenizer and baseline; use paired pilot evidence to decide the next confirmation, not automatic promotion.
- No cloud, paid services, dependency upgrades, destructive cleanup or push. Unrelated user files remain untouched.

## Unresolved / not completed

- No broad science, mathematics or standalone code corpus; no general-capability benchmark claims.
- No depth/width search or checkpointing re-sweep on the new data.
- No optimizer/RNG-resumable training checkpoints (retained weights are evaluation checkpoints).
- No untouched final-test promotion stage, external benchmark contamination certification, or distributed/off-machine reproduction.

## Next three highest-value actions

1. Review per-domain paired results and fixed samples; freeze one follow-up hypothesis before opening any test scores.
2. Add a small explicitly licensed science/math/code source with the same provenance, dedup and held-out controls; improve markup handling independently.
3. Run fresh-seed confirmation at a larger unique-data/token budget, then one frozen test stage before considering tokenizer or mixture promotion.

## Reproduce / continue

See [campaign protocol](../../docs/foundation-campaign.md) and [tokenizer audit](../../docs/tokenizer-foundation-v1.md). Example validation replay (native PowerShell):

```powershell
$py = '.autoresearch/upstream/.venv/Scripts/python.exe'
& $py scripts/evaluate_foundation.py runs/foundation_campaign/pilot-control-s201 --output runs/foundation_campaign/manual-replay.json
& $py scripts/report_foundation.py
```

Local artifacts, logs, weights, candidate tokenizer and evaluation text remain under `runs/foundation_campaign/`; corpus under `data/foundation-v1/`. Compact manifests and receipts only are published here.
