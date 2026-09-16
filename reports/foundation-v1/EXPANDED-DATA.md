# Bounded unique-general-data comparison

Expanded-minus-original aggregate BPB: **-0.024891** (paired SD0.005830). Predeclared validation screen passed: **True**. No promotion from validation alone; see the final report for the frozen test gate.

Three fresh paired seeds211/212/213;8,388,608tokens per run; candidate tokenizer fixed; D12/width768,135,267,480total/active parameters, memory table0, checkpointing off. Same optimizer, schedule, actual domain token mass, technical stream and exact held-out text.

| Training pool | Seed | Aggregate BPB | General BPB | Technical BPB | Timed tok/s | Update s | Wall s | Alloc/reserved MiB | General stream passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old | 211 | 1.304569 | 1.420287 | 1.184124 | 25455 | 329.7 | 341.2 | 2277.2/2408.0 | 2.273 |
| expanded | 211 | 1.272947 | 1.381632 | 1.159822 | 25261 | 332.0 | 346.2 | 2277.2/2408.0 | 0.671 |
| old | 212 | 1.293470 | 1.413504 | 1.168533 | 25205 | 332.9 | 344.3 | 2277.2/2408.0 | 2.287 |
| expanded | 212 | 1.272054 | 1.379377 | 1.160346 | 25341 | 331.5 | 346.1 | 2277.2/2408.0 | 0.675 |
| old | 213 | 1.308188 | 1.422952 | 1.188735 | 25345 | 331.1 | 342.2 | 2277.2/2408.0 | 2.286 |
| expanded | 213 | 1.286553 | 1.384559 | 1.184544 | 25408 | 330.3 | 345.1 | 2277.2/2408.0 | 0.675 |

Paired aggregate deltas: [-0.031622, -0.021416, -0.021634]. Domain mean deltas: {'general': -0.037058135838570504, 'technical': -0.012226136911561936}.

## Interpretation

The original general pool contains609documents/10,890,601normalized bytes; the expanded pool contains1,999documents/36,367,702bytes. They share only33exact normalized documents. **This is pool replacement, not a nested data-size-only ablation:** article/topic coverage and repeated exposure change together. Do not attribute the result solely to having more bytes.

Both pools use the same source repository revision, lexical/quality filters and source license. The expanded pool contains9,993,911candidate-tokenizer text tokens (before BOS), versus the smaller original pool recorded in its source audit. Technical training remains264documents/5,230,783bytes. Validation/test files are byte-identical across corpora; test scores were not used in this comparison; the final test stage is reported separately.

Source payload downloaded for this extension:314,076,578bytes; total campaign source payload351,340,048bytes. Expanded fingerprint: `f783433992d354c5887f5130f3f9f7892f171916b4e9ce5294fda2ed87a5b6eb`. Candidate tokenizer SHA256 remains `736bb1a35ce3b2a3143095bfc1390858d67dd7f606fda25137b83095bace0102`. Exact code/runtime/checkpoint identities and immutable replay receipts are in [expanded-results.json](expanded-results.json).

This result is conditional on the short fixed-token budget and selected held-out target. It neither proves that larger datasets always help nor that diversity is intrinsically harmful. This validation-stage report makes no promotion; see the final combined validation/test gate.

## Reproduce

```powershell
$py = '.autoresearch/upstream/.venv/Scripts/python.exe'
& $py scripts/foundation_expand.py
& $py scripts/expanded_data_campaign.py
& $py scripts/evaluate_foundation.py runs/foundation_campaign/data-expanded-s211 --data data/foundation-expanded-v1 --output runs/foundation_campaign/manual-expanded-replay.json
& $py scripts/report_expanded_data.py
```

The controller only admits jobs within the original campaign deadline. For later studies, freeze a new protocol/deadline and new output names rather than modifying earlier receipts.
