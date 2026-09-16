# Fixed-prompt sample progression

**These samples do not show a useful assistant.** Inspect the complete recorded outputs, not selected successful snippets:

- [Early: 8,388,608 tokens](samples-000512.json)
- [Middle: 234,881,024 tokens](samples-014336.json)
- [Final: 469,762,048 tokens](samples-028672.json)

All six prompts, generation seed 20260915, temperature 0.8, top-k 40 and 64 new tokens were fixed. This is a base model without instruction tuning. Short continuations can be truncated; these are qualitative diagnostics, not capability benchmarks.

## Observations

| Probe | Early | Middle | Final |
|---|---|---|---|
| Public-library prose | Drifts into platform/exception documentation; repeated phrases | Mixes a library sentence with `os.environ` documentation and malformed terms | Still changes into invented API documentation; “commercience” and “datetach” are not coherent explanations |
| Why the sky looks blue | Unrelated theater/biographical-looking prose | Invented entities and unrelated geographic-looking prose | More invented words and unrelated proper-name collage; no explanation of the sky |
| `count_words` completion | Unclosed string with a long `adv` repetition loop | Invalid body followed by RST-like explanatory text | Invalid `retrieval ... pass` line, then markup/prose; not a word-counting implementation |
| `x + 3 = 7` continuation | Repeated dotted numbers | Changes into class/version documentation | Changes into a code-block-like passage; does not give `x = 4` |
| Careful-experiment prose | Long repeated `ly` suffix loop | More varied words but malformed, off-topic music/art-like prose | Syntactic fragments with unrelated biological/business terms |
| One-sentence dictionary explanation | Film-like prose, no dictionary definition | Headers/coroutine references, no requested explanation | More invented terms and unrelated text, still no definition |

The conspicuous early substring loops are less dominant later, but lexical variety is **not** evidence of semantic improvement. Topic drift, invented terms, malformed code and lack of task completion persist. There is no coherent basis here for a coding, mathematics, reasoning, instruction-following or factual-accuracy claim.

All three code continuations fail Python AST parsing as complete snippets. This is a syntax observation only: samples are length-limited, and no generated code was executed. The final snippet already contains invalid syntax near the beginning, before its truncation boundary.

The final checkpoint's six outputs were reproduced byte-identically in a fresh process. This verifies the diagnostic reproduction path, not their quality. See `independent-audit.json` and `sample-syntax-audit.json`.

## Interpretation limits

The held-out BPB regression and falling training loss support overfitting under this locked setup. The sample observations are consistent with continued poor generalization, but cannot isolate data repetition, optimizer/schedule effects or sampling effects as the cause. No factual-looking assertion in these samples has been treated as verified information.
