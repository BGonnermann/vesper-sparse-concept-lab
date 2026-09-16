# Campaign closeout

**Completed and published:** [publication receipt](publication.json), [changed-file inventory](CHANGED-FILES.txt), [implementation commits](IMPLEMENTATION-COMMITS.txt). Publication was verified after **6.012 hours**; the eight-hour ceiling was not filled with extra training.

As of **2026-09-16T12:12:35.261266-04:00**, verification is complete; publication follows. Elapsed campaign time: **5.934 hours**, not eight elapsed hours.

Started 2026-09-16T06:16:33.205990-04:00. Training finished **2026-09-16T11:47:07.969701-04:00**, before the 2026-09-16T12:46:33.205990-04:00 cutoff. The eight-hour ceiling was 2026-09-16T14:16:33.205990-04:00; 90 minutes were reserved after the cutoff. The calibrated fixed budget finished earlier, leaving additional reporting headroom. No budget was changed for quality and no extra training was performed to consume the remaining time.

## Verification

- **17 focused CPU tests and 8 focused CUDA tests passed.** Earlier test logs remain locally. The full 260-test legacy suite was not rerun; shared training/evaluation sources were unchanged.
- Exact restored model/optimizer/RNG/sampler state verified before continued full-model preflight updates. Native BF16 future trajectories are not bitwise deterministic; the preserved initial strict failure is not hidden.
- All 28,672 update records and independent source counts/RNG/offsets checked; optimizer group/initial-LR coverage receipt matches the accepted pilot byte-for-byte.
- All 34 validation/replay/test aggregates recomputed from nats and actual UTF-8 bytes. Final fresh-process validation difference was zero. Test evaluated once on the fixed final endpoint; no test-based checkpoint selection.
- Final generation reproduced byte-identically in fresh processes, including the documented inference command. Weight/checkpoint hashes remained unchanged.
- All seven charts inspected; all 14 PNG/SVG files reproduced byte-identically from exported data in a fresh process. JSON, image dimensions/DPI, SVG parsing and Markdown links checked.
- No owned training/evaluation processes remain. Free disk at verification: **194.21 GiB**. No new data downloads, cloud training, dependency upgrades or experimental mechanisms.

## Evidence and retained artifacts

- [Independent accounting audit](independent-audit.json), [chart reproduction](chart-reproduction.json), [chart package/font runtime](chart-runtime.json), [inference verification](inference-verification.json), [runtime identity](runtime-identity.json).
- [CPU tests](tests-final-cpu.log.txt), [CUDA tests](tests-closeout-cuda.log.txt), [full-model preflight/calibration accounting](ancillary-work.json).
- [Initial strict resume failure](initial-resume-failure.log.txt) preserved. A closeout orchestration command also initially supplied an unsupported `env` argument; it failed before tests ran. [Repair record](closeout-command-repair.json); no model/evaluator change was needed.
- Large checkpoints, original/pruned checkpoint hash receipts, raw updates, failed attempts, logs, raw corpora and caches stay outside Git. [Checkpoint manifest](checkpoint-manifest.json) gives local paths and hashes.
- Captured training source bytes are under `runs/foundation_long_20260916/reference/source`; final reporting sources under `runs/foundation_long_20260916/reporting-source`. Preserve these archives: Git newline conversion can change raw hashes, including an existing mixed-newline model source. An isolated replication checkout must restore the recorded bytes and runtime/data artifacts; do not weaken provenance checks or overwrite active user files.
- Unrelated untracked monitor/depth-audit files were not edited, staged or deleted. No original pilot/TinyStories/model artifacts were replaced.
- Commits are published by normal pushes on `research/foundation-long-20260916`; see Git history and the publication inventory. No force push.

## Continuation boundary

**This campaign is closed to further training.** The supervised recovery wrapper rejects a completed run or an opened test marker. Do not restart its queue or extend its budget. Future work needs a separate frozen configuration/run directory and fresh held-out evidence. The old regression test is not an untouched benchmark.

Use the [report](REPORT.md) for the historical inference/chart commands and the exact final, early-frontier and accepted-pilot comparison bars. Merely beating this degraded endpoint is not sufficient evidence of progress.

The report's `manual-generation-01.json` output was created during verification. The evaluator correctly refuses to overwrite it. For another inference-only check, choose a fresh filename, for example:

```powershell
$py = '.autoresearch/upstream/.venv/Scripts/python.exe'
& $py scripts/evaluate_long_baseline.py --run runs/foundation_long_20260916/reference --mode generate --output runs/foundation_long_20260916/manual-generation-02.json
```

This loads the final checkpoint, generates the fixed prompts and verifies unchanged weights; it does not train or evaluate the test set.
