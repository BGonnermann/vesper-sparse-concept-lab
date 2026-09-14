# Vesper Sparse Concept Lab

Research started: 2026-09-14. Status: research plus a runnable dense autoresearch integration; custom MoE/NCP/n-gram modules and local GPU results are still pending.

Goal: build a skilled model for coding, Python, research, and tool workflows that can eventually run on GPUs with 8–16GB VRAM. Start architecture experiments on an RTX 5070 Ti; use cloud GPUs when measured resource needs justify them.

The working hypothesis is that sparse experts, next-concept prediction, and conditional n-gram memory can improve useful capability per unit of memory and compute. Their combination is unproven here. Each component must earn its cost against a simpler baseline.

## Start here

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

This repository holds the research documents, experiment plans, and a pinned Windows autoresearch integration. The runner has CPU-only tests; GPU compatibility and training outcomes must be measured on your PC. GitHub access allows collaborative changes and review, but does not provide access to the desktop GPU.

## First milestone

Measure the actual desktop environment and run the smaller dense TinyStories profile in the autoresearch integration. The earlier 100–300M custom prototype remains a later proposal. Establish correctness, memory use, and training throughput before adding the research modules or committing to a larger training budget.

Architecture training and deployment are different budgets: few active experts reduce some computation, while stored weights, optimizer states, caches, and memory tables still occupy space. Fitting a quantized model for inference does not establish that it can be trained from scratch on the same GPU.

Project license is undecided. Linked papers, repositories, datasets, and weights retain their own terms; no upstream source code or weights are bundled.
