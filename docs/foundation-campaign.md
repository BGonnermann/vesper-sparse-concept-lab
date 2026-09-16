# Dense foundation campaign v1

## Research decision (2026-09-15)

Dense is the mainline architecture. Frozen NCP is archived as a negative result **under the tested conditions**: the published NCP campaign reports 2/4 fresh paired seeds favoring NCP and no advantage over exactly parameter-matched residual MLP. MoE remains experimental, not adopted. N-grams and any other mechanism require independent controlled evaluation. Existing implementations and evidence remain unchanged. No mechanism enters mainline without reproducible improvement and acceptable costs.

Freeze `experiments/mainline/dense-v1.json`: D12, width768, dense, matrix LR .04. The existing optimizer, full causal attention, context512, BF16, microbatch2, 16,384 tokens/update and checkpointing **off** are retained. All 135,267,480 parameters are active; memory table size is zero. This is the best validated *fixed-token quality* baseline, not a claim of best wall-time efficiency. D6-width768 is cheaper and has not been conclusively displaced on a compute frontier.

Evidence inspected: overnight 87-trial report, NCP final report and findings, equal-token report, checkpointing report, and the longer-depth campaign. D12 LR .03 reversed sign on seed44; keep .04. The longer-depth campaign has only two completed published trials and is not a completed multi-seed result. Reproduce D12-width768 seed101 at512 updates against the retained .589157 legacy BPB result; predeclare absolute tolerance .005 BPB for this nondeterministic runtime. Reproduction is an infrastructure gate, not new independent-seed evidence.

## Honest next-stage target

Learn English expository syntax, short-context topical continuity, and Python documentation vocabulary from a small licensed corpus. Improve held-out document-aligned language modeling over a reproducible dense reference without hiding technical-domain regression. This is a base LM, not an assistant or a safe tool user. General reasoning, reliable factual recall, mathematics, production coding and frontier-scale behavior are outside demonstrated scope. TinyStories remains a regression dataset only.

Primary new metric: byte-weighted aggregate UTF-8 bits per byte on immutable, document-aligned validation inputs; also report each domain, macro-domain BPB, document loss variation, fixed-token cost, throughput, total/active parameters and allocated/reserved VRAM. Do not compare perplexity across tokenizers. Do not compare new document-aligned BPB numerically with legacy packed TinyStories BPB. A legacy Unicode byte-table issue is retained rather than retroactively changing old scores.

Consumer constraints: native Windows/5070 Ti runtime observed by tools; sequential GPU jobs, no cloud, no dependency upgrades. Hard experiment cutoff04:57 EDT and stop05:57 EDT on September16. Reserve >=30GiB disk; fixed download inventory is ~36MiB (well below20GB). Stop on any shared correctness/provenance issue, non-finite loss, corrupted cache, or three consecutive10-second GPU samples >=85C. At most two CUDA-failed jobs before halt; no blind retry.

## Pilot sources and pipeline

`foundation_data.py` pins WikiText-2 raw to Hugging Face revision `b08601e04326c79dfdd32d625aee71d232d685c3` and CPython3.11.9 to commit `de54cf5be371a6f5e2e9f208c38def5f81d3ef02`. Sources were chosen for manageable size, explicit licensing, article/file boundaries, and complementary general/technical language, not scale. Retained source cards/licenses and download receipts live under `data/foundation-v1/raw/`. The generated manifest records bytes, hashes, split counts, source provenance and known limitations; tokenizer audit adds per-source token counts.

WikiText: CC-BY-SA3.0/GFDL per source card; preserve attribution/share-alike obligations before redistribution. CPython documentation: PSF license and included historical notices; examples additionally0BSD. Read retained license files before redistribution. No corpus is checked into Git.

Reconstruct WikiText articles at top-level headings, keep official splits. Split CPython by SHA256 of whole file path (80/10/10 buckets); only tutorial/library RST. Normalize NFC and newlines without flattening indentation. Filter malformed lengths, replacement characters, NUL and strongly non-Latin scripts. These English editions supply language provenance; script filtering is not robust language identification. Exact SHA256 dedup, inverted5-word-shingle Jaccard >=.8 near dedup, and exact shared paragraphs >=200characters operate globally, held-out first. Never sample before separation. Filtering may remove legitimate repeated documentation; the manifest reports removals. This is not semantic dedup or benchmark-wide decontamination.

Downloads are atomic, byte-capped, content-sealed and reusable. Completed preprocessing is hash-verified on resume; interruption before the final manifest can rerun deterministically from sealed downloads. Mid-document CPU state is not checkpointed. Corrupted caches fail rather than silently rebuild. A changed pipeline revision requires a new output directory. Tests exercise leakage, malformed records, determinism, cache corruption, byte accounting and source sampling.

Science and mathematics sources are deliberately deferred. Technical documentation is not equivalent to a broad code corpus. Raw RST/Wikipedia markup and limited topical coverage make this an engineering pilot, not the final general pretraining mixture.

## Tokenizer and evaluation protocol

Audit original8192-token tokenizer and one8192-token BPE candidate trained only on a deterministic training sample capped at4million characters/domain. Preserve both. Same vocabulary makes embedding/parameter comparison exact; changed segmentation changes raw-byte exposure at fixed tokens and must be reported. Candidate compression is not sufficient for adoption.

Validation inputs: first16 documents/domain ordered by SHA256(document id), truncated to8192Unicode characters each *before tokenization*. Evaluate every target token exactly once; BOS predicts the first token, subsequent disjoint512-token target windows retain the preceding token. Byte denominator is the exact UTF-8 byte count of the selected text, independently checked against raw token bytes. No held-out mixing into training. This finite-token-context policy changes byte-context length across tokenizers; disclose it. Full split hashes plus selected input text/hashes are saved locally with every run. Test split stays unopened for model scoring during pilot selection.

Generated samples use fixed prompts, seed20260915, top-k40, temperature.8 and64new tokens. Samples diagnose failures; they do not prove capability.

## Predeclared controlled pilot

1. Two-update full-path smoke (seed201, current tokenizer,80/20 source token mass).
2. Three paired fresh seeds201/202/203, each512updates (8,388,608tokens), D12-width768 and unchanged optimizer/batch/evaluation. Control: current tokenizer80/20. Mixture arm: current tokenizer50/50. Tokenizer arm: candidate tokenizer80/20. Compare each arm only against its control, not against each other as a one-factor contrast. Alternate arm order by seed. Bernoulli microbatch source selection determines actual weights; record counts and repeated passes.
3. No depth search or new mechanisms. Checkpointing remains off based on prior evidence; only revisit if new memory observations require it.

Primary selection metric is byte-weighted aggregate validation BPB, with both domain BPBs visible. Require all three paired seeds to improve, mean improvement >=.01 BPB, no per-domain mean regression >.02 BPB, and no >20% throughput regression or >16GiB peak device use. A candidate must also survive an untouched final evaluation and review before mainline promotion. This pilot alone does not promote anything, even if the screen passes. Reject invalid provenance, non-finite loss or token mismatch. Stop admission when a job cannot finish before the experiment cutoff. No score-based extra seeds, no test-driven reselection.

The new loop deliberately makes data construction explicit (cyclic packed domain streams, BOS separators, source sampling per microbatch); historical TinyStories runs are not a controlled data-only comparator. Improvement claims are limited to paired arms on this new fixed pipeline. All-update timing includes transfers and synchronization; tokenization/preparation and evaluation are separate. Timed throughput excludes11updates; full wall time is also retained.

## Reproduce / continue (native PowerShell)

```powershell
$py = '.autoresearch/upstream/.venv/Scripts/python.exe'
& $py -m unittest discover -s tests -v
$env:MOE_TEST_DEVICE = 'cuda'; & $py -m unittest discover -s tests -v
& $py scripts/reproduce_dense_baseline.py --output runs/foundation-baseline-repeat-01
& $py scripts/foundation_data.py build
& $py scripts/foundation_data.py verify
& $py scripts/foundation_tokenizer.py --output runs/foundation_campaign/tokenizer-new
& $py scripts/foundation_train.py --output runs/foundation_campaign/manual-smoke --updates 2 --seed 201
```

For the predeclared sequential pilot queue (only within its recorded deadline):

```powershell
& $py scripts/foundation_campaign.py run
```

It resumes only completed attempts, verifies the frozen plan, and refuses to overwrite failed attempts. For independent checkpoint verification:

```powershell
& $py scripts/evaluate_foundation.py runs/foundation_campaign/pilot-control-s201 --output runs/foundation_campaign/manual-replay.json
```

Replay also evaluates the same first16 sealed TinyStories validation documents with each tokenizer using exact document bytes, separately from legacy packed regression BPB. Source/runtime/data identities and unchanged model weights are checked. No pilot test split model scores are opened.

Output directories are immutable per attempt; choose new names when repeating. Use the monitored campaign supervisor for longer jobs, not the unmonitored train child directly. Existing untracked `scripts/depth_split_audit.py` was present at startup and must remain untouched. Local commits only: initial tree was dirty, so automatic push is not authorized by this campaign's safety rule.
