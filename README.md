# Vesper Sparse Concept Lab

Research started: 2026-09-14. Current mainline: a validated dense Transformer, with native Windows training and a pinned pilot data/tokenizer/evaluation foundation. Experimental MoE/NCP/n-gram implementations and their results are preserved separately from mainline selection.

Goal: build a skilled model for coding, Python, research, and tool workflows that can eventually run on GPUs with 8–16GB VRAM. Start architecture experiments on an RTX 5070 Ti; use cloud GPUs when measured resource needs justify them.

Dense is the mainline architecture. Frozen NCP failed its tested confirmation rule; MoE has not earned adoption. N-grams and other mechanisms must be evaluated independently. No experimental mechanism enters mainline without a controlled, reproducible improvement. Earlier architecture documents remain historical proposals, not the current development plan.

## Start here

- **[Dense foundation protocol and research decision](docs/foundation-campaign.md)** — frozen baseline, capability target, data pipeline, evaluation and reproduction commands.
- **[Verified foundation pilot report](reports/foundation-v1/REPORT.md)** — paired results, costs, limitations and next actions.
- **[469.8M-token dense reference](reports/foundation-long-v1/REPORT.md)** — fixed-budget native-Windows run, seven reproducible charts, resumable checkpoints and inference commands. It strongly overfit the repeated small pool; it is a measured reference, not a stronger replacement for the accepted pilot.
- [Tokenizer audit](docs/tokenizer-foundation-v1.md)
- [Gated general/technical profile v2](experiments/mainline/foundation-general-v2.json) — same dense architecture; tokenizer and data-pool improvements passed three-seed validation and frozen test gates. TinyStories regression v1 is preserved.
- **[Run the Windows training tests](docs/autoresearch.md)** — setup, GPU probe, smoke test, and dense baseline.
- [Bounded instructions for your local coding agent](program.md)
- [Research findings and primary sources](docs/research.md)
- [Architecture options and proposed prototype](docs/architecture.md)
- [5070 Ti memory budget and hardware check](docs/hardware.md)
- [Controlled experiments and workflow evaluation](docs/experiments.md)
- [Local-to-cloud roadmap](docs/roadmap.md)
- [Source catalog](references/sources.json)
- [Run record template](experiments/run-template.json)

Clone the repository and open the folder in your editor. Run these commands in PowerShell with Git installed and access to the private repository:

```powershell
git clone https://github.com/VesperEngineering/vesper-sparse-concept-lab.git
cd vesper-sparse-concept-lab
git status
git log -1 --oneline
```

This repository holds the research documents, experiment plans, and a pinned Windows autoresearch integration. The runner has CPU and CUDA correctness tests; hardware compatibility and training outcomes remain specific to the recorded runtime. GitHub access allows collaborative changes and review, but does not provide access to the desktop GPU.

## First milestone

The frozen quality baseline is dense D12/width768 with135,267,480total/structurally active parameters and activation checkpointing off. TinyStories is an infrastructure regression dataset, not a useful-general-model capability claim. The next milestone is reproducible improvement on clean general and technical held-out data, with per-domain quality, exact-byte BPB, wall time and memory reported separately. No new architectural mechanism is needed for that milestone.

Architecture training and deployment are different budgets: few active experts reduce some computation, while stored weights, optimizer states, caches, and memory tables still occupy space. Fitting a quantized model for inference does not establish that it can be trained from scratch on the same GPU.

Project license is undecided. Linked papers, repositories, datasets, and weights retain their own terms; no upstream source code or weights are bundled.
