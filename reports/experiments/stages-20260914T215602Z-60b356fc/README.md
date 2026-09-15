# stages-20260914T215602Z

Outcome: **stage_evidence_only**. Budget family: **stage_or_unknown**.

Full compact configuration, metrics, seeds, hashes and diagnostics: [report.json](report.json).

Provenance: historical recorded commits/hashes only. This publication does not retroactively identify training source.
Unknown fields remain null. Do not pool different budgets/schedules. Two seeds are not proof of equivalence.

## Diagnostics
```text
setup.log:
error: Failed to initialize cache at `C:\Users\bgonn\AppData\Local\uv\cache`
  Caused by: failed to open file `C:\Users\bgonn\AppData\Local\uv\cache\sdists-v9\.git`: Access is denied. (os error 5)
```

## Retained narrative: summary.md

# First baseline receipt

Executed setup, doctor, prepare, smoke, and baseline in order on native Windows with the NVIDIA GeForce RTX 5070 Ti. Checked each stage before proceeding. Stopped after the first successful baseline; no autonomous experiments or candidate/protocol edits.

- Setup: passed on retry. Initial attempt could not access the existing uv cache from the sandbox. Retried with approved execution access. Both logs retained.
- Doctor: passed CUDA matrix multiplication, backward, and optimizer update. Python 3.11.15, PyTorch 2.9.1+cu128, CUDA build 12.8, BF16 supported.
- Prepare: passed tokenizer sanity check, vocabulary 8192; dataset and tokenizer hashes recorded in .autoresearch/data-seal.json and run results.
- Smoke: completed, exit 0, 3 updates, validation BPB 2.237952, wall time 4.985 seconds.
- Baseline: completed, exit 0, 804 updates, validation BPB 0.630201 over the configured 65536 validation tokens.
- Baseline wall time: 309.422 seconds, including process startup and evaluation.
- Baseline reported training time: 300.3 seconds, excluding the first 11 updates.
- Peak allocated GPU memory: 285164032 bytes, approximately 272 MiB.
- Peak reserved GPU memory: 306184192 bytes, 292 MiB.
- Memory scope: PyTorch allocator only; excludes other processes and driver allocations. Host RAM peak was not measured.
- Total parameters: 11534472, dense depth-4 model.
- Active parameters: not separately instrumented; do not interpret total parameters as a measured per-token active count.
- Sparse memory-table size: not applicable; no n-gram memory component in this baseline.
- Derived training throughput: approximately 43249 tokens/second, using 793 post-warmup logged updates times 16384 tokens divided by 300.409 seconds summed from rounded per-step durations. Excludes startup, warmup, checkpoint saving, and evaluation.
- Nonfatal launcher warning: uv ignored a dangling temporary pip directory in C:/Users/bgonn/.venv. All requested stages completed; that unrelated directory was left unchanged.

## Retained evidence

- Stage logs: setup.log, setup-retry1.log, doctor.log, prepare.log, smoke.log, baseline.log in this directory.
- Smoke artifacts: ../20260914T220045Z-0c8b28f2/
- Baseline artifacts: ../20260914T220112Z-d2949ab5/
- Each training directory retains result.json, run.log, memory.json, environment.json, candidate.json, protocol.json, and the runtime-reported checkpoint_pre_eval.pt.
- Baseline source commit: 1919da4362fbd20d781e72f34bf0d0e2d9ccea5f; runner recorded a clean worktree.
- Pinned upstream revision: a4123c6e5c6287f90be04026642ba20b94e424df. Runtime and data hashes are retained in result.json.

This is a single seeded TinyStories baseline, not evidence of broader task capability or multi-seed quality.
