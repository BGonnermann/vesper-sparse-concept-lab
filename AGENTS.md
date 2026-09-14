# Project context

- Target initial training: RTX 5070 Ti, native Windows. Do not assume WSL.
- Eventual deployment target: 8–16GB VRAM. Cloud is a later scaling option.
- Keep published evidence, proposed designs, estimates, and measured results clearly distinct.
- Record total parameters, active parameters, memory-table size, peak memory, and actual throughput separately.
- Use primary sources. Pin revisions before reusing third-party code, models, or datasets.
- Prefer small causal correctness checks and controlled ablations before larger training runs.
- Preserve failed runs and their logs. Never populate unknown measurements with plausible numbers.
- Keep weights, corpora, credentials, and large run artifacts out of Git. Track their identities and locations in run records.
- The assistant environment is not the user's desktop; do not report its hardware as the user's hardware.
