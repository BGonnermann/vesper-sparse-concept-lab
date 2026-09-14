# Vesper local autoresearch program

Purpose: use an external coding agent to screen small dense-model candidates on the user's GPU. This is the first stage of the MoE/NCP/n-gram project. Read `AGENTS.md` and `docs/autoresearch.md` first.

## Prerequisites

The user must have completed setup, doctor, prepare, smoke, and the initial baseline. Inspect the actual baseline's `result.json` and `run.log`. If they are absent or failed, diagnose that run before searching. Work on a dedicated Git branch and retain every candidate commit. Preserve unrelated user changes.

## Scope and budget

- Edit only `experiments/autoresearch/candidate.json` during this first search.
- Change either depth or matrix learning rate in each candidate. Record a short hypothesis before running it.
- Run at most **three candidates per invocation**, one at a time. Stop after 60 minutes of elapsed session time; do not start a candidate unless its smoke and baseline deadlines fit within the remaining time. Stop immediately on user interruption.
- Keep protocol, upstream code, dependency environment, tokenizer, dataset, runner, adapter, and evaluation fixed. Do not run setup/prepare again inside the search.
- Do not launch cloud jobs or download alternative datasets as part of this loop.

## Loop

1. Read the current best valid baseline and its configuration. Make one hypothesis-driven candidate edit, validate its JSON, and commit it.
2. Run `uv run --no-project --python 3.11 scripts/autoresearch.py smoke`.
3. If smoke succeeds, run `uv run --no-project --python 3.11 scripts/autoresearch.py baseline`.
4. Read the saved records and logs. A failed, timed-out, invalid, or interrupted run is not a score. Preserve it and diagnose; stop the campaign on an unexplained runtime failure.
5. Compare only completed baseline runs with the same protocol, data/tokenizer hashes, upstream/runtime hashes, adapter, and hardware. Smoke scores are not baseline scores. Note that every run uses seed 42.
6. Record BPB, total elapsed seconds, memory, candidate commit, and run directory in `experiments/autoresearch/notes.md` (create this human-readable record when the first real experiment exists). Retain local raw logs and weights. Do not commit weights or corpus files.
7. Treat lower BPB as a provisional candidate improvement, subject to memory and elapsed-cost tradeoffs. Restore the best candidate JSON with a new commit if needed; never erase failed candidate history or unrelated work using a hard reset.

Report tested candidates, failures, best provisional configuration, and remaining uncertainty at the budget limit. Do not claim MoE/NCP/n-gram gains from this dense-only search. Do not promote to a larger training run solely because a five-minute score improved.
