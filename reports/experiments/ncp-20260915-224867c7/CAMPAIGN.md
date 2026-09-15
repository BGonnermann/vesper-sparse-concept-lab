# NCP campaign progress

5 completed of 7 attempted full trials. Budget: 11:41:32 to 19:41:32 UTC, 2026-09-15.
512 updates and 8,388,608 tokens per full trial. Seed42 screens are exploratory. Lower BPB is better.

## Implementation

Dense encoder pools complete multi-token chunks; causal chunk Transformers predict segmented discrete-codebook weights. Only predicted concepts feed the token decoder, delayed by k-1 positions. Detached future chunks supervise NCP MSE; VQ MSE fits a transformed frozen random codebook basis. Token BPB excludes both auxiliary losses.
This is a simplified ConceptLM-inspired prototype, not a paper reproduction. Softmax feedback differs from the official GPT2/Pythia raw-logit multiplication. Native SDPA, initialization, positional features, codebook transforms and the small TinyStories fixed-token experiment also differ. Official revision: a0ab281286f5c0337c35de3181cc992c562eacaa.
See [campaign plan](../../../docs/ncp-campaign.md) and the published source receipt for exact references.

## Every attempted trial

| Trial | Seed | BPB | Update s | Child s | Timed tok/s | Total / active params | Alloc / reserved MiB | NCP collapse |
|---|---:|---:|---:|---:|---:|---|---|---|
| [trial-0001-D6-s42](../ncp-20260915--trial-0001-D6-s42-b009c334/README.md) | 42 | 0.634848 | 177.1 | 184.0 | 47526 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0002-NCP-s42](../ncp-20260915--trial-0002-NCP-s42-22537f23/README.md) | 42 | failed | unavailable | 80.0 | unavailable | unavailable | 654.4 / 690.0 | unavailable |
| [trial-0003-AUX-s42](../ncp-20260915--trial-0003-AUX-s42-6ac0782c/README.md) | 42 | failed | unavailable | 82.4 | unavailable | unavailable | 654.4 / 690.0 | unavailable |
| [trial-0004-N-prediction_weight0.1-s42](../ncp-20260915--trial-0004-N-prediction_weight0.1-s42-d0ad5ff9/README.md) | 42 | 0.635524 | 202.6 | 210.3 | 41362 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0005-N-prediction_weight0.03-s42](../ncp-20260915--trial-0005-N-prediction_weight0.03-s42-cd708cd3/README.md) | 42 | 0.635888 | 203.3 | 211.0 | 41265 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0006-N-RMS-s42](../ncp-20260915--trial-0006-N-RMS-s42-d3652f3a/README.md) | 42 | 0.635927 | 214.2 | 221.8 | 39314 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0007-N-RMS-AUX-s42](../ncp-20260915--trial-0007-N-RMS-AUX-s42-d5def34d/README.md) | 42 | 0.635215 | 204.9 | 212.4 | 41139 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |

## Depth6 versus depth12, reference LR .04

| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |
|---:|---|---|---:|---:|---:|---:|
No completed pair yet.

## Frozen NCP confirmation

| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |
|---:|---|---|---:|---:|---:|---:|
No completed pair yet.

## Attempts, decisions and failures

- trial-0001-D6-s42: Fresh depth6 matching control before NCP trials
- trial-0002-NCP-s42: Source-inspired discrete chunk prediction plus causal predicted feedback may improve BPB at equal tokens Failure: RuntimeError('Training child exit 1')
- trial-0003-AUX-s42: Initial feedback trial exceeded loss100 at update169; same-weight auxiliary-only condition tests whether feedback caused instability Failure: RuntimeError('Training child exit 1')
- trial-0004-N-prediction_weight0.1-s42: Both unit-weight variants diverged; reduce NCP prediction MSE coefficient tenfold while keeping codebook fitting and feedback unchanged
- trial-0005-N-prediction_weight0.03-s42: Alpha0.1 completed without codebook collapse but was worse than D6 by0.000676 BPB; reduce alpha to0.03 to limit auxiliary interference
- trial-0006-N-RMS-s42: After raw-latent scale growth, normalize pooled concept states at original alpha=beta=1 to test stability and token quality
- trial-0007-N-RMS-AUX-s42: Normalized-state auxiliary-only ablation isolates concept supervision from predicted feedback

## Measurement limits

Equal-token quality comparisons; measured runtime is a separate cost axis. No equal-time quality claim. Depth changes width too. Timed throughput excludes the first11 updates; all-update time includes them. Allocator peaks exclude driver/desktop use. Whole-board sampled VRAM, dictionary bytes, exact configurations, source/data/checkpoint hashes, auxiliary losses and utilization are in the JSON receipts. Codebook assignments do not prove semantic concepts. Repeated validation selection is exploratory, not held-out generalization.
All artifacts are retained locally. No cloud, dependency upgrades, paid services or deletion.
