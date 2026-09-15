# Run the first local training tests

This integration runs dense or four-expert top-1 MoE TinyStories models through a pinned Windows autoresearch fork and a local model adapter. It does not implement NCP or n-gram memory. The first MoE design and correctness contract are in [moe-feasibility.md](moe-feasibility.md).

## Windows quick start

Use native PowerShell, Git, and uv. If uv is missing, install it with `winget install --id=astral-sh.uv -e`, then reopen PowerShell. [Official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Clone this repository if you have not already. If you have a normal GitHub clone, update it with `git pull --ff-only`. The earlier ZIP has separate Git history: use a fresh clone alongside it and preserve any local work.

```powershell
git clone https://github.com/VesperEngineering/vesper-sparse-concept-lab.git
cd vesper-sparse-concept-lab
```

Run each command separately and stop if one fails:

```powershell
uv run --no-project --python 3.11 scripts/autoresearch.py setup
uv run --no-project --python 3.11 scripts/autoresearch.py doctor
uv run --no-project --python 3.11 scripts/autoresearch.py prepare
uv run --no-project --python 3.11 scripts/autoresearch.py smoke
uv run --no-project --python 3.11 scripts/autoresearch.py baseline
```

| Command | What it does |
| --- | --- |
| `setup` | Fetches the pinned public fork into `.autoresearch/upstream`, installs its dependencies into an isolated Python 3.11 environment, and records the resolved lock hash. Downloads can be substantial; allow disk space for Python, CUDA/PyTorch wheels, data, and checkpoints. |
| `doctor` | Reports the actual GPU/runtime, then performs a small CUDA matrix multiplication, backward pass, and optimizer update. |
| `prepare` | Downloads the fork's full TinyStories parquet and trains its tokenizer. Saves content hashes of the prepared files. The source URL uses `main`; the resulting content hashes identify the data actually used. |
| `smoke` | Runs up to three optimizer updates and a small validation pass. Hard process deadline: 180 seconds, including startup and evaluation. |
| `baseline` | Runs a five-minute timed training phase plus warmup and evaluation. Hard process deadline: 900 seconds. Requires a successful smoke test with the same candidate, protocol, runtime, data, and adapter. |

Python 3.11 is explicit because the launcher uses standard-library features introduced in 3.11. No WSL, FlashAttention package, or Triton installation is required by this integration.

## What to send back

Each smoke/baseline command prints a unique folder under `runs/autoresearch/`. Attach **`result.json` and `run.log`** from that folder. If available, also include `environment.json` and `memory.json`. These local files are ignored by Git, so GitHub access alone will not let an assistant read them.

A nonzero exit, timeout, missing summary, or nonfinite score is recorded as failure/invalid, never as an improvement. Every run keeps its own log and configuration. The fork also attempts to save `checkpoint_pre_eval.pt` in that run's folder. That file contains weights only; it is **not a resumable optimizer/RNG checkpoint**. Repeated runs can consume disk space.

## Fixed initial comparison

The profile uses context 512, 16,384 tokens per optimizer update, microbatch 2, full causal attention, activation checkpointing, and no autotuning. Evaluation batch is fixed at 2, with 8,192 smoke tokens and 65,536 baseline tokens. The candidate starts at depth 4 and matrix learning rate 0.04; the runtime prints the actual parameter count.

Candidate files select `feedforward: dense` or `feedforward: moe`, depth 2–8 and matrix learning rate in `(0, 0.1]`. MoE additionally requires `num_experts: 4`, `top_k: 1`, a positive `aux_loss_weight`, and a positive `router_lr`. `protocol.json` defines the comparison; changing it starts a new protocol and invalidates the smoke gate. Variant, candidate, data, runtime, seed, runner, adapter and local model hashes all participate in the gate. Old dense smoke runs cannot authorize trials with the new adapter.

The fixed comparison candidates are `experiments/autoresearch/dense-depth6.json` and `experiments/autoresearch/moe-depth6.json`. Select one explicitly for both smoke and baseline, for example:

```powershell
uv run --no-project --python 3.11 scripts/autoresearch.py smoke --candidate experiments/autoresearch/moe-depth6.json
uv run --no-project --python 3.11 scripts/autoresearch.py baseline --candidate experiments/autoresearch/moe-depth6.json
```

Version-2 records also require `model.json`, `optimizer.json`, `routing.json`, and `training.json`, and preserve local runner/adapter/model source snapshots. They record exact total/structural active parameters, optimizer membership, exact processed tokens, router gradients, zero dropped tokens, auxiliary loss and per-layer train/eval utilization. BPB excludes the routing loss. The displayed training `loss` includes the configured auxiliary term. These candidates differ in parameter budget and cannot establish an efficiency gain.

The upstream code resets seed 42 internally. `baseline --repeat 3` measures repeated execution of the same seeded candidate, not three independent seeds. We must add independent seed handling before making multi-seed research claims.

The fork excludes the first 11 optimizer updates from its training timer. Smoke runs therefore report zero `training_seconds`. Use the runner's `wall_seconds` for total elapsed cost. Do not divide all training tokens by the upstream timer to infer throughput. Hardware MFU is suppressed because the fork's approximate hardware constants and step counting are unsuitable for our measurements. Allocated/reserved memory covers the PyTorch allocator, not all device users.

## Let Pi or another coding agent run bounded experiments

After one successful baseline, point the agent at [program.md](../program.md). The file defines a maximum of three candidate experiments, immutable evaluation/data, smoke gates, log retention, and a final report. The launcher itself does not call an LLM or generate candidate edits. Your coding agent supplies the research loop; the launcher executes and records each test.

TinyStories measures basic training behavior. A lower score here does not establish better Python, security, quant, tool use, or NCP performance. The next implementation stages are MoE, NCP, and memory individually, followed by the controlled combinations in [experiments.md](experiments.md). They need architecture-specific correctness tests and new candidate interfaces.

## Reproducibility and limitations

Checkpointing is an explicit boolean `activation_checkpointing` in each captured
protocol. The default `protocol.json` keeps it on. Use
`--protocol experiments/autoresearch/protocol-no-checkpoint.json` to select off
without changing microbatch, context or any other budget. Pass the same protocol
to smoke and baseline. Protocol hashes isolate smoke gates, and model/training
receipts must confirm the captured checkpoint setting actually executed.


The fork is pinned to [`a4123c6`](https://github.com/jsegov/autoresearch-win-rtx/tree/a4123c6e5c6287f90be04026642ba20b94e424df). Its README lists native Windows/5070 Ti support, but its reported hardware test is an RTX 3080. Our GPU tests still need to run on the user's PC. The original idea and research loop come from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

The fork's checked-in lock does not match its dependency file, so setup resolves it once rather than falsely claiming `--locked` reproducibility. Subsequent trials use that environment directly and verify file hashes. Different installations can resolve different dependency versions: compare their setup manifests and actual lockfiles before pooling results. Do not edit the venv between experiments.

Prepared data and tokenizer hashes are verified before each command. This detects accidental changes; it is not a sandbox against a malicious coding agent. The dataset's split/evaluation implementation remains upstream's, including its token-byte accounting. Freeze a separate untouched final evaluation before larger research claims.

The integration tests exercise result validation, actual subprocess execution, deadlines, log retention, smoke gates, and file integrity without GPU downloads. GitHub Actions runs those checks on Windows and Ubuntu. Passing them does not establish GPU compatibility or model quality. Hardware tests are the commands above.

If a trial times out, inspect its retained log to see whether it was in training or validation. Do not disable the deadline or immediately scale up. If CUDA initialization fails, send the `doctor` output. Setup and data preparation are separately bounded and can be retried; existing edited upstream code is left untouched.
