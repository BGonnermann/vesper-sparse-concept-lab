# NCP campaign progress

16 completed of 18 attempted full trials. Budget: 11:41:32 to 19:41:32 UTC, 2026-09-15.
512 updates and 8,388,608 tokens per full trial. Seed42 screens are exploratory. Lower BPB is better.

## Implementation

Dense encoder pools complete multi-token chunks; causal chunk Transformers predict segmented discrete-codebook weights. Only predicted concepts feed the token decoder, delayed by k-1 positions. Detached future chunks supervise NCP MSE; VQ MSE fits a transformed frozen random codebook basis. Token BPB excludes both auxiliary losses.
This is a simplified ConceptLM-inspired prototype, not a paper reproduction. Softmax feedback differs from the official GPT2/Pythia raw-logit multiplication. Native SDPA, initialization, positional features, codebook transforms and the small TinyStories fixed-token experiment also differ. Official revision: a0ab281286f5c0337c35de3181cc992c562eacaa.
See [campaign plan](../../../docs/ncp-campaign.md) and the published source receipt for exact references.

## Every attempted trial

| Trial | Seed | BPB | Update s | Trial s | Timed tok/s | Total / active params | Alloc / reserved MiB | NCP collapse |
|---|---:|---:|---:|---:|---:|---|---|---|
| [trial-0001-D6-s42](../ncp-20260915--trial-0001-D6-s42-b009c334/README.md) | 42 | 0.634848 | 177.1 | 184.0 | 47526 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0002-NCP-s42](../ncp-20260915--trial-0002-NCP-s42-22537f23/README.md) | 42 | failed | unavailable | 80.0 | unavailable | unavailable | 654.4 / 690.0 | unavailable |
| [trial-0003-AUX-s42](../ncp-20260915--trial-0003-AUX-s42-6ac0782c/README.md) | 42 | failed | unavailable | 82.4 | unavailable | unavailable | 654.4 / 690.0 | unavailable |
| [trial-0004-N-prediction_weight0.1-s42](../ncp-20260915--trial-0004-N-prediction_weight0.1-s42-d0ad5ff9/README.md) | 42 | 0.635524 | 202.6 | 210.3 | 41362 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0005-N-prediction_weight0.03-s42](../ncp-20260915--trial-0005-N-prediction_weight0.03-s42-cd708cd3/README.md) | 42 | 0.635888 | 203.3 | 211.0 | 41265 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0006-N-RMS-s42](../ncp-20260915--trial-0006-N-RMS-s42-d3652f3a/README.md) | 42 | 0.635927 | 214.2 | 221.8 | 39314 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0007-N-RMS-AUX-s42](../ncp-20260915--trial-0007-N-RMS-AUX-s42-d5def34d/README.md) | 42 | 0.635215 | 204.9 | 212.4 | 41139 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0008-D6-s43](../ncp-20260915--trial-0008-D6-s43-6ae7cb75/README.md) | 43 | 0.637774 | 163.7 | 170.5 | 51373 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0009-D12-s43](../ncp-20260915--trial-0009-D12-s43-3e98bde0/README.md) | 43 | 0.594495 | 308.5 | 316.9 | 27244 | 135,267,480 / 135,267,480 | 2277.2 / 2408.0 | n/a |
| [trial-0010-D6-s44](../ncp-20260915--trial-0010-D6-s44-8211e6c7/README.md) | 44 | 0.643075 | 158.5 | 165.5 | 52909 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0011-D12-s44](../ncp-20260915--trial-0011-D12-s44-afbc92ee/README.md) | 44 | 0.592813 | 323.0 | 331.3 | 26022 | 135,267,480 / 135,267,480 | 2277.2 / 2408.0 | n/a |
| [trial-0012-NCP-s42](../ncp-20260915--trial-0012-NCP-s42-bf446fb5/README.md) | 42 | 0.649647 | 220.4 | 227.8 | 38159 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0013-AUX-s42](../ncp-20260915--trial-0013-AUX-s42-bb8c5a76/README.md) | 42 | 0.652928 | 217.4 | 224.8 | 38680 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0014-R-prediction_weight0.1-s42](../ncp-20260915--trial-0014-R-prediction_weight0.1-s42-06b4af48/README.md) | 42 | 0.636121 | 220.2 | 227.7 | 38211 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0015-R-feedback_scale0.1-s42](../ncp-20260915--trial-0015-R-feedback_scale0.1-s42-23081d16/README.md) | 42 | 0.633732 | 220.0 | 226.0 | 38226 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0016-R-feedback_scale0.3-s42](../ncp-20260915--trial-0016-R-feedback_scale0.3-s42-866e46eb/README.md) | 42 | 0.634247 | 222.0 | 228.0 | 37882 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0017-R-layers1-s42](../ncp-20260915--trial-0017-R-layers1-s42-341c3f13/README.md) | 42 | 0.638393 | 207.8 | 214.1 | 40394 | 28,287,276 / 28,287,276 | 625.4 / 664.0 | False |
| [trial-0018-R-after_layer1-s42](../ncp-20260915--trial-0018-R-after_layer1-s42-93561fc2/README.md) | 42 | 0.634774 | 221.0 | 226.9 | 38057 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |

## Depth6 versus depth12, reference LR .04

| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |
|---:|---|---|---:|---:|---:|---:|
| 43 | D12 | D6 | 0.594495 | 0.637774 | -0.043279 | 1.88 |
| 44 | D12 | D6 | 0.592813 | 0.643075 | -0.050262 | 2.04 |

Mean paired delta: -0.046770 BPB. Mean update-time ratio: 1.96.

## Frozen NCP confirmation

| Seed | Candidate | Control | Candidate BPB | Control BPB | Delta BPB | Update-time ratio |
|---:|---|---|---:|---:|---:|---:|
No completed pair yet.

## Feedback versus auxiliary-only ablations

Negative delta favors predicted-concept feedback. These selection-seed comparisons are exploratory.

| Feedback configuration | Auxiliary-only configuration | Seed | Feedback BPB | Auxiliary BPB | Delta |
|---|---|---:|---:|---:|---:|
| NCP | AUX | 42 | 0.649647 | 0.652928 | -0.003281 |
| N-RMS | N-RMS-AUX | 42 | 0.635927 | 0.635215 | +0.000712 |

## Attempts, decisions and failures

- trial-0001-D6-s42: Fresh depth6 matching control before NCP trials
- trial-0002-NCP-s42: Source-inspired discrete chunk prediction plus causal predicted feedback may improve BPB at equal tokens Failure: RuntimeError('Training child exit 1')
- trial-0003-AUX-s42: Initial feedback trial exceeded loss100 at update169; same-weight auxiliary-only condition tests whether feedback caused instability Failure: RuntimeError('Training child exit 1')
- trial-0004-N-prediction_weight0.1-s42: Both unit-weight variants diverged; reduce NCP prediction MSE coefficient tenfold while keeping codebook fitting and feedback unchanged
- trial-0005-N-prediction_weight0.03-s42: Alpha0.1 completed without codebook collapse but was worse than D6 by0.000676 BPB; reduce alpha to0.03 to limit auxiliary interference
- trial-0006-N-RMS-s42: After raw-latent scale growth, normalize pooled concept states at original alpha=beta=1 to test stability and token quality
- trial-0007-N-RMS-AUX-s42: Normalized-state auxiliary-only ablation isolates concept supervision from predicted feedback
- trial-0008-D6-s43: Fresh seed43 control for mandatory independent depth comparison at reference LR0.04
- trial-0009-D12-s43: Independent seed43 depth12 versus depth6; fixed tokens and LR0.04, report extra width/parameters/time
- trial-0010-D6-s44: Fresh seed44 control for mandatory independent depth comparison at reference LR0.04
- trial-0011-D12-s44: Independent seed44 depth12 versus depth6; fixed tokens and LR0.04, report extra width/parameters/time
- trial-0012-NCP-s42: Retry original unit-weight NCP after distinguishing finite auxiliary MSE from token CE in the stopping guard; same training objective and update budget
- trial-0013-AUX-s42: Retry original unit-weight AUX after distinguishing finite auxiliary MSE from token CE in the stopping guard; same training objective and update budget
- trial-0014-R-prediction_weight0.1-s42: Change only prediction_weight to 0.1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0015-R-feedback_scale0.1-s42: Change only feedback_scale to 0.1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0016-R-feedback_scale0.3-s42: Change only feedback_scale to 0.3 on normalized NCP to test quality versus its stable unit-weight control
- trial-0017-R-layers1-s42: Change only layers to 1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0018-R-after_layer1-s42: Change only after_layer to 1 on normalized NCP to test quality versus its stable unit-weight control

## Correctness and diagnosis

The first two unit-weight attempts hit an inherited finite total-loss100 guard. A source-verified CPU checkpoint probe found dense hidden RMS12.54 too, so those stops do not establish NCP-specific divergence. The corrected guard checks token CE separately and still rejects nonfinite total loss. Initial failed attempts remain preserved, and their historical hypotheses using the word divergence are superseded by this diagnosis. Completed BPB runs never hit that gate.
CPU/CUDA tests cover prefix causality, future-label isolation, VQ/encoder gradients, optimizer coverage, save/load, codebook learning and evaluation immutability. Every completed training child executes captured sources; the final evidence audit also verifies saved checkpoints and committed source-archive bytes.

## Measurement limits

Equal-token quality comparisons; measured runtime is a separate cost axis. No equal-time quality claim. Depth changes width too. Timed throughput excludes the first11 updates; all-update time includes them. Trial wall time includes preparation, child execution and verification, excluding reporting/publication. Allocator peaks exclude driver/desktop use. Whole-board sampled VRAM, dictionary bytes, exact configurations, source/data/checkpoint hashes, auxiliary losses and utilization are in the JSON receipts. Diagnostic feedback_rms is the unscaled prediction; injected_feedback_rms applies the configured gain and is zero for auxiliary-only runs. Codebook assignments do not prove semantic concepts. Repeated validation selection is exploratory, not held-out generalization.
All artifacts are retained locally. No cloud, dependency upgrades, paid services or deletion.
