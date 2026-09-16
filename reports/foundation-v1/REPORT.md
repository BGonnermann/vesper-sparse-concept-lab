# Dense foundation campaign: final report

**Use the pilot BPE candidate and expanded general-data pool for the next local general/technical phase.** Dense architecture and the original TinyStories regression baseline are preserved. This is a foundation-profile decision, not a useful-assistant or production-weight claim.

## Verified facts

- Native Windows execution on the observed RTX5070Ti. Frozen D12/width768,135,267,480total and structurally active parameters, memory table0, BF16 AMP, context512, microbatch2,16,384tokens/update, checkpointing off.
- Fifteen full pilot runs, six fresh seeds (201–203 and211–213),8,388,608tokens/run;125,829,120pilot training tokens. A separate seed101baseline reproduction and two successful two-update smokes are recorded separately.
- Original corpus:1,049documents. Expanded corpus:2,439documents, with unchanged held-out files and unchanged264-document technical training pool. Total retained source-download payload:351,340,048bytes; no cloud, paid services or dependency upgrades.
- V1 dataset fingerprint: `debe7de4f51d9487e316aedf1a7dc828350d7a77f20d99a913a0143e456b121f`; expanded: `f783433992d354c5887f5130f3f9f7892f171916b4e9ce5294fda2ed87a5b6eb`.
- Both tokenizers have8192entries and50,331,648vocabulary-dependent parameters including the existing six value-embedding tables. Original SHA256: `4d1991faca1391dbc13ba13ef7ed19a3ae77dde50d4a6927fd068090b80cda5c`; candidate: `736bb1a35ce3b2a3143095bfc1390858d67dd7f606fda25137b83095bace0102`.
- Every pilot checkpoint reproduced its validation score within1e-6 before one frozen test stage. All15test outcomes and unchanged-weight checks are reported. Exact run/source/runtime identities are in the linked JSON receipts.

## Experimental results

Negative deltas favor the candidate. Each comparison changes one factor; do not treat equal tokens as equal raw-byte exposure.

| Change vs its matched control | Validation BPB delta (paired SD) | Test BPB delta (paired SD) | Both conservative gates |
|---|---:|---:|---|
| mixture | -0.050469 (0.010605) | -0.048537 (0.012128) | False |
| tokenizer | -0.112029 (0.010801) | -0.105836 (0.012606) | True |
| expanded_data | -0.024891 (0.005830) | -0.025950 (0.009452) | True |

Per-domain paired mean BPB deltas (positive means regression):

| Change | Validation general | Validation technical | Test general | Test technical |
|---|---:|---:|---:|---:|
| mixture | +0.068986 | -0.174803 | +0.052963 | -0.180253 |
| tokenizer | -0.064894 | -0.161089 | -0.056200 | -0.170249 |
| expanded_data | -0.037058 | -0.012226 | -0.036671 | -0.012037 |

The50/50mixture remains rejected: its validation general-domain regression failed the predeclared guard. The additional test gate also requires improvement >2paired SD, all3paired aggregate signs favorable, mean gain >=.01BPB, domain mean regressions <=.02BPB, and acceptable costs.

All per-domain and individual results: [initial9-run pilot](PILOT.md), [fresh-seed data-only comparison](EXPANDED-DATA.md), [all15test models](TEST.md). Aggregate results must not hide the domain tables.

Measured training throughput: 25,205–26,033tok/s; full per-run wall time: 336.2–346.2s. Peak training allocator memory: 2277.2MiB allocated / 2408.0MiB reserved. Whole-board samples and full cost breakdowns are retained separately.

Baseline reproduction: seed101,8,388,608tokens, legacy TinyStories BPB.589531 versus recorded.589157;25,798timed tok/s,334.9s wall,2,277.2/2,408.0MiB allocator peak. Its wrapper metadata failure and separate successful verification are both retained.

## Inferences

- The tokenizer improvement is measured model quality, not compression alone, and is conditional on these general/technical corpora and short-context protocol.
- The expanded pool comparison changes article/topic coverage and repeated exposure together: only33general documents overlap exactly. It is **not** a nested data-size-only ablation. Report the measured pool-replacement result, not a universal claim that more data helps.
- Exact-byte evaluation fixes a Unicode accounting problem in the legacy evaluator. Historical packed TinyStories BPB is preserved and must not be pooled with the new document-aligned series.

## Generated-sample inspection

Fixed prompts/seed20260915/top-k40/temperature.8/64tokens expose major limitations. The candidate on the original corpus emits constant returns for `count_words` and inconsistent arithmetic. The expanded-pool seed211model emits `x + y` repetitions for word counting and does not explain why the sky is blue. These are base-LM continuations, not an instruction-following test, but they provide no support for useful coding, mathematics, reasoning or factual-reliability claims. All fixed samples are retained; none were used to select a lucky seed.

## Decisions made

- Dense remains mainline. Frozen NCP is a negative result under its tested conditions; MoE/n-grams remain experimental and untouched.
- Tokenizer eligible for the versioned general/technical profile: **True**. Expanded pool eligible with that tokenizer: **True**. Preserve regression-v1 and every earlier artifact.
- No production checkpoint replacement, weight upload, automatic push or history rewrite. No further training after the final-test freeze.

## Failures

- Baseline training succeeded but its wrapper omitted condition metadata; original failure receipt retained, independent verification repaired only the metadata interpretation.
- First new-path smoke hit noncontiguous targets before completing an update. Contiguous copies fixed it; new smoke attempts passed.
- Initial commit failed because Git author identity was unset; command-local campaign assistant identity was used without global configuration changes.

## Unresolved questions / work not completed

- No frontier/general-reasoning claim; no broad science, math or standalone code corpus; no production-quality assistant.
- No new depth/width search, checkpointing re-sweep, optimizer/RNG-resumable checkpoints or8GB-device deployment measurement.
- Lexical dedup is not semantic or external-benchmark decontamination. The now-opened test split is a regression set for future work, not an untouched selection set.
- This is a bounded pilot campaign within the8-hour ceiling, not an8-hour-duration training run. Actual clock, test counts, health, disk, file and commit inventory are in [closeout](CLOSEOUT.md).

## Next three highest-value actions

1. Use the gated foundation profile in a new predeclared study with longer learning curves and fresh held-out data; do not retune against this now-opened test.
2. Add bounded, explicitly licensed science/math/code sources; independently ablate markup cleanup and data quality rather than architecture mechanisms.
3. Add small task-level evaluations and broaden tokenizer/domain audits before making any useful-capability claim.

## Reproduce / continue

See [main protocol](../../docs/foundation-campaign.md), [data extension](../../docs/foundation-data-extension.md), [test freeze](../../docs/foundation-final-test.md), and [tokenizer audit](../../docs/tokenizer-foundation-v1.md). Native Windows commands and artifact locations are in [closeout](CLOSEOUT.md). All weights/corpora/tokenizer artifacts remain outside Git.
