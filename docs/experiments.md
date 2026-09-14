# First experiments

All runs below are planned and unexecuted. Numerical caps are proposed experiment budgets, not performance claims.

## Hypothesis and conditions

Hypothesis: NCP and n-gram memory improve held-out quality for a small MoE enough to offset their measured memory and wall-clock costs.

| ID | Feedforward | NCP | N-gram memory | Purpose |
| --- | --- | --- | --- | --- |
| D | Dense | Off | Off | Establish simple baseline |
| M | MoE | Off | Off | Measure sparse backbone cost/benefit |
| MC | MoE | On | Off | Isolate concept prediction |
| MG | MoE | Off | On | Isolate conditional memory |
| MCG | MoE | On | On | Measure interaction |

First use the same MoE backbone for M/MC/MG/MCG to debug mechanisms. Adding components changes parameter and compute budgets, so those runs cannot establish efficiency by themselves. For confirmation, resize conditions to match total parameters and estimate training FLOPs, then also compare held-out quality at equal measured GPU time. Report mismatches explicitly; matching steps or tokens is not matching compute.

## Stages and gates

1. **Correctness:** check the architecture contracts, finite losses/gradients, tiny-batch overfitting, and checkpoint resume. Inspect validation separately; tiny-batch overfitting establishes mechanics only. Stop at any causal leak or invalid resumption.
2. **Profile:** after kernel initialization, use 100 warmup and 200 measured steps. Record synchronized elapsed time, processed non-padding tokens, allocated/reserved peak VRAM, device-level memory, host RAM, and checkpoint/evaluation overhead. Compare context 512/1,024 and memory settings. If this exceeds the one-hour smoke cap, record the shorter sample and its uncertainty.
3. **Pilot:** set a common data-token budget using measured throughput that fits a proposed maximum of eight local GPU-hours per condition. Keep tokenizer, splits, order, seed, effective tokens/update, evaluation cadence, and optimizer policy controlled. Save raw per-step logs and failures.
4. **Confirmation:** use at least three seeds for the promising comparisons. Freeze the model choices and held-out suite before looking at final outcomes. Report dispersion and per-task results rather than only an average.
5. **Scale decision:** proceed only if an improvement survives budget controls and exceeds run-to-run noise. If the combination loses, keep the simpler winner. Do not scale merely because every module trains without crashing.

The pilot is a mechanism screen. Limited tokens and tiny models cannot establish that a future assistant is highly skilled.

## Data plan

Start with a small, licensed public corpus covering code and explanatory text. Choose exact datasets in the next implementation phase. Record dataset revision, license, source hashes, token counts, filtering, and mixture proportions. Split by repository/document before chunking; deduplicate across splits. Freeze a held-out evaluation set. Exclude benchmark solutions and user-private material from initial training.

Use a fixed existing tokenizer first. Measure tokens per byte separately for code and prose. A custom tokenizer would need its own experiment and could require retraining embeddings; it is not necessary to test NCP.

## Workflow evaluation

The user's likely priorities are Python/coding, model and quant research, and tool-driven workflows. Confirm relative weights with real examples before selecting a production winner. Proposed small suite:

| Task family | Initial sample | Primary measurement |
| --- | --- | --- |
| Python bug fixes | 10 synthetic or permitted tasks | Held-out tests passed |
| Structured tool use | 10 mocked tool tasks | Valid arguments and correct completed action |
| Research synthesis | 10 source-bounded questions | Supported claims and correct citations |
| Quant/data reasoning | 10 toy-data problems | Numeric correctness and absence of future-data leakage |

Use sandboxed execution for generated code and mocked tools. This suite is development feedback, not evidence of broad benchmark leadership. Keep a separate final set; do not tune on final answers. Small scratch models may score near zero, so early comparison should also use held-out token loss and bits per byte.

## Metrics and records

Track loss curves, domain loss, quality scores, training tokens/s, total GPU-hours, inference time to first token, decode tokens/s, peak VRAM, host RAM, and context length. Track router balance, overflow, concept code usage, and memory collisions. Measure latency at batch 1 for local interaction and separately at higher concurrency for deployment density.

Store resolved configurations and raw measurements next to the [run record](../experiments/run-template.json). Unknown values remain null. Each result must identify Git commit, dirty state, hardware, runtime, dataset/tokenizer revisions, seed, precision, and checkpoint identity. Record OOMs and discarded configurations with their reasons.
