# Tokenizer audit: foundation v1

Measured on the complete pilot **validation** splits. Both vocabularies have8192tokens, byte fallback, zero unknown tokens on this corpus, and exact round trips. No test model scores were consulted. Full per-document token lengths, vocabulary utilization and encoding timings are retained in `runs/foundation_campaign/tokenizer/audit.json`.

| Tokenizer | General bytes/token | Technical bytes/token |
|---|---:|---:|
| Sealed TinyStories BPE | 2.7926 | 2.1499 |
| Training-only pilot BPE candidate | 3.6896 | 3.7736 |

Rounded values should be checked against the machine-readable audit before reuse. The candidate uses the same pretokenization regex and vocabulary budget, trained on a deterministic8million-character sample (4million/domain), not held-out data. Repeating training produced a byte-identical candidate file. It does not introduce a tokenizer dependency or download external weights.

Original tokenizer SHA256: `4d1991faca1391dbc13ba13ef7ed19a3ae77dde50d4a6927fd068090b80cda5c`.
Candidate JSON SHA256: `736bb1a35ce3b2a3143095bfc1390858d67dd7f606fda25137b83095bace0102`.

## Important byte-accounting distinction

The53-byte Unicode probe round-trips correctly through both tokenizers. Summing the UTF-8 lengths of **individually decoded** current-tokenizer tokens yields135bytes because incomplete UTF-8 byte pieces decode to replacement characters. Raw token-byte lengths sum to the correct53bytes. The candidate similarly yields131legacy-decoded bytes versus53actual bytes.

This is a defect in using legacy token-length accounting for arbitrary Unicode, **not** a tokenizer round-trip failure. Historical TinyStories results retain their old evaluator and are explicitly labeled legacy BPB. We have not recomputed every historical score or quantified its historical effect. New quality comparisons use identical selected documents and exact UTF-8 bytes, and independently verify raw-byte token sums. Never merge the two BPB series.

## Parameter and throughput costs

At width768, input embedding and untied output projection each contain6,291,456parameters. Their12,582,912combined count is not the entire vocabulary-dependent cost: the existing frozen dense backbone also has six value-embedding tables. Total vocabulary-dependent parameters are50,331,648, unchanged between these tokenizers. Total model parameters remain135,267,480, all structurally active; memory table0.

Encoding throughput is CPU throughput, not GPU training throughput. Fewer tokens per byte allow more raw bytes per fixed-token training budget but do not prove better model quality. They also change repeated corpus exposure and effective byte-context length. Training receipts report source stream lengths, repeated passes, actual source token mass, GPU throughput and wall time separately.

## Recommendation before model comparison

Keep the original tokenizer as mainline while testing the candidate. Compression is clearly better matched to this pilot, especially Python RST documentation, but the pilot is small and not a general web/science/math/code distribution. Any model-based recommendation must cite the paired fixed-token results, per-domain regressions, parameter equality and untouched confirmation status. Code/math/Unicode/whitespace/long-identifier probes are diagnostics, not capability benchmarks.
