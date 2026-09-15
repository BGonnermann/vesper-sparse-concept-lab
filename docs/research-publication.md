# Research reporting and publication

The accepted scope is compact reporting/backfill, tests, reviewed Git publication,
disk accounting and a proposed causal memory experiment. No training, cloud jobs,
artifact deletion or NCP implementation is authorized here. Bound routine repair
to three iterations; preserve failed attempts and diagnose unresolved failures.

## Offline receipts

Normal runner trials automatically write a compact receipt after their local
`result.json` is finalized, on both success and failure. Reporting failure leaves
that result intact and records a retry instruction. Historical/custom stage
drivers are backfilled explicitly; they are not silently rewritten.

```powershell
python scripts/experiment_reports.py backfill
python scripts/experiment_reports.py one runs/autoresearch/EXPERIMENT
```

Each stable artifact-relative ID has `reports/experiments/<id>/report.json` and
`README.md`. The generated progress index is `reports/README.md` and `index.json`.
Repeated reporting of unchanged artifacts produces identical bytes, not duplicate
IDs or timestamps. Run one reporting writer at a time. Writes are atomic per file;
a retry repairs any interrupted report/index update.

Receipts contain captured configuration, seeds, measured metrics, outcome,
diagnostics, source/data hashes and local evidence locations. Large arrays and
traces remain local with file hashes; unknown historical fields remain unknown.
Small captured Python sources are deduplicated by SHA-256 under
`reports/source-snapshots/`. Their filenames identify the exact original bytes.
Historical execution receipts are retained as evidence, not treated as a fresh
re-execution. A publishing commit never replaces a historical training commit.

Budget/schedule/source fingerprints are compatibility filters, not proof of a
controlled comparison. Do not pool time-budget runs, fixed-token runs, smoke tests
or profiler-instrumented timings. Refer to each retained comparison narrative.
Opposite signs across two seeds do not prove equivalence.

## Explicit publication and retry

Review and stage only relevant source, tests, configurations and reports; commit
them before publishing. Publication is never an automatic training side effect.

```powershell
python scripts/experiment_reports.py publish
```

This pushes the existing HEAD to the current branch on `origin`, without force,
and verifies the remote branch hash. It does not create commits, delete local
results, retrain, or request new credentials. After a network/upload failure,
retry the same command. A non-fast-forward response requires reconciliation, not
a force push. No GitHub Pages/site/cloud deployment is involved.

## Retention proposal: no deletion performed

- Keep compact reports, code/config snapshots, hashes, tests and small failure
  diagnostics indefinitely in Git.
- Keep raw run logs and metadata indefinitely locally; preserve all failed-run
  evidence until its diagnosis is documented and independently backed up.
- Keep representative dense/MoE checkpoints for reproducibility and future
  inference checks. Before removing redundant smoke/checkpoint copies, verify
  content hashes and an approved backup; obtain separate deletion approval.
- Keep the sealed TinyStories corpus/tokenizer and pinned runtime while they are
  active controls. Re-downloading from a mutable source is not an equivalent backup.
- Large profiler traces and replay batch tapes are candidates for a verified
  local archive after the experiment is closed, not normal Git. Never silently
  remove the sole copy of a replay tape needed to reproduce paired batches.
- Propose a 30-day review for large redundant artifacts, not automatic expiry.
  Report size, reproducibility value and backup status before any deletion.

Disk figures in `reports/disk-usage.json` are logical file sizes measured locally,
not unique physical allocation (hardlinks and filesystem compression can differ).
