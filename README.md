# Vesper Sparse Concept Lab

Research started: 2026-09-14. Status: research and experiment design; no model implementation or training results yet.

Goal: build a skilled model for coding, Python, research, and tool workflows that can eventually run on GPUs with 8–16GB VRAM. Start architecture experiments on an RTX 5070 Ti; use cloud GPUs when measured resource needs justify them.

The working hypothesis is that sparse experts, next-concept prediction, and conditional n-gram memory can improve useful capability per unit of memory and compute. Their combination is unproven here. Each component must earn its cost against a simpler baseline.

## Start here

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

This repository holds the research documents and experiment plans. No GPU jobs have been launched. Dependency versions will be pinned after a native Windows/CUDA compatibility check; there is no validated training install command yet. GitHub access allows collaborative changes and review, but does not provide access to the desktop GPU.

## First milestone

Measure the actual desktop environment, then implement a roughly 100–300M-total-parameter mechanics prototype. This size is a starting proposal, not a demonstrated fit or a useful assistant. Establish correctness, memory use, and training throughput before committing to a larger training budget.

Architecture training and deployment are different budgets: few active experts reduce some computation, while stored weights, optimizer states, caches, and memory tables still occupy space. Fitting a quantized model for inference does not establish that it can be trained from scratch on the same GPU.

Project license is undecided. Linked papers, repositories, datasets, and weights retain their own terms; no upstream source code or weights are bundled.
