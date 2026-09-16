# Predeclared unique-data extension (2026-09-15, after the nine-run pilot)

The first pilot's candidate tokenizer improved both domains on all three seeds. The50/50mixture failed the general-domain guard. No change was promoted and no test model scores were opened.

There is still ample campaign budget. Next decision: does more **unique general-domain training text**, rather than repeated passes through WikiText-2, improve the same immutable held-out target? This is a controlled data comparison, not an assumption that more data helps.

## Data contract

Use the same pinned Salesforce/wikitext repository revision `b08601e04326c79dfdd32d625aee71d232d685c3`, but ingest the two `wikitext-103-raw-v1` training shards. These are curated Wikipedia text with the already recorded CC-BY-SA3.0/GFDL license, article boundaries and accessible pinned parquet, not an arbitrary large web scrape. Download at most200MiB/shard, with >=30GiB free disk. Campaign download total stays well below20GB.

Retain at most2,000articles selected by lowest SHA256(document identifier) using a bounded heap while streaming parquet, capped at64MiB of selected UTF-8 training text. Reuse exactly the existing normalization, quality, script and global lexical-dedup rules. Preserve the original pilot's validation and test files **byte-for-byte**, and preserve the complete technical training domain. Held-out documents have priority during dedup. The new ingest code, parent manifest, source revisions, download bytes/hashes and final split hashes are all recorded. The expanded corpus is a separate version, never an overwrite of v1.

This deliberately changes only the general training pool. The hypothesis is that reduced repeated exposure and broader curated topics may improve held-out BPB; less repetition could also hurt short-budget optimization. Results are unknown at declaration.

## Six-run comparison

Fresh seeds211,212,213. Every seed trains two models from initialization: original pilot corpus and expanded corpus. Both use the **same existing pilot BPE candidate** (not retrained), frozen dense D12/width768, same optimizer, checkpointing off, context512, microbatch2,16,384tokens/update,512updates/8,388,608tokens,80/20general/technical source token mass. Alternate arm order by seed. Training tensor source identity and repeated passes differ by design; architecture, tokenization, initial tensors, schedules, technical stream, actual per-domain token counts and evaluation text must match within pairs.

Primary metric: the original pilot's exact-byte aggregate validation BPB. Report both domains, all paired seeds, update time, total wall time, throughput and memory. Require all three paired deltas negative, mean <=-.01BPB, no per-domain mean regression >.02BPB, and no >20% throughput regression. Do not promote automatically; the test stage remains unopened. No extra seeds or budget extension selected from these outcomes.

Stop on any shared provenance, leakage, non-finite loss or CUDA/thermal problem. Each job has a20-minute maximum and must fit before04:57EDT. Preserve every attempt. The initial nine-run pilot remains independently valid regardless of this extension's outcome.
