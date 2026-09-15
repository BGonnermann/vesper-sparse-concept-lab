# NCP campaign progress

43 completed of 45 attempted full trials. Budget: 11:41:32 to 19:41:32 UTC, 2026-09-15.
512 updates and 8,388,608 tokens per full trial. Seed42 screens are exploratory. Lower BPB is better.

Strongest eligible selection-seed NCP: I2-37a8ad42c0, 0.631877 BPB, delta -0.002971 versus D6. Adds 3,655,680 parameters; measured update-time ratio 1.23. This is a search result; independent confirmation is reported separately.

## Implementation

Dense encoder pools complete multi-token chunks; causal chunk Transformers predict segmented discrete-codebook weights. Only predicted concepts feed the token decoder, delayed by k-1 positions. Detached future chunks supervise NCP MSE; VQ MSE fits a transformed frozen random codebook basis. Token BPB excludes both auxiliary losses.
This is a simplified ConceptLM-inspired prototype, not a paper reproduction. Initial softmax feedback differs from the official GPT2/Pythia raw-logit multiplication; raw-logit variants are separately labeled. Native SDPA, initialization, positional features, codebook transforms and the small TinyStories fixed-token experiment also differ. Official revision: a0ab281286f5c0337c35de3181cc992c562eacaa.
See [campaign plan](../../../docs/ncp-campaign.md) and the published source receipt for exact references.

## Every attempted trial

| Trial | Seed | BPB | Update s | Trial s | Timed tok/s | Total / active params | Alloc / reserved MiB | Target-code collapse |
|---|---:|---:|---:|---:|---:|---|---|---|
| [trial-0001-D6-s42](../ncp-20260915--trial-0001-D6-s42-b009c334/README.md) | 42 | 0.634848 | 177.1 | 184.0 | 47526 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0002-NCP-s42](../ncp-20260915--trial-0002-NCP-s42-22537f23/README.md) | 42 | failed | unavailable | 80.0 | unavailable | 30,056,748 / 30,056,748 * | 654.4 / 690.0 | unavailable |
| [trial-0003-AUX-s42](../ncp-20260915--trial-0003-AUX-s42-6ac0782c/README.md) | 42 | failed | unavailable | 82.4 | unavailable | 30,056,748 / 30,056,748 * | 654.4 / 690.0 | unavailable |
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
| [trial-0019-R-after_layer2-s42](../ncp-20260915--trial-0019-R-after_layer2-s42-bbb25dff/README.md) | 42 | 0.635594 | 217.0 | 223.7 | 38695 | 30,056,748 / 30,056,748 | 655.8 / 688.0 | False |
| [trial-0020-R-chunk_size2-s42](../ncp-20260915--trial-0020-R-chunk_size2-s42-94b10b74/README.md) | 42 | 0.634780 | 219.5 | 226.8 | 38242 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0021-R-chunk_size8-s42](../ncp-20260915--trial-0021-R-chunk_size8-s42-4e711e4f/README.md) | 42 | 0.637063 | 218.2 | 224.3 | 38464 | 30,056,748 / 30,056,748 | 652.1 / 686.0 | False |
| [trial-0022-R-entries16-s42](../ncp-20260915--trial-0022-R-entries16-s42-2bda239a/README.md) | 42 | 0.633742 | 218.3 | 227.1 | 38544 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0023-R-entries32-s42](../ncp-20260915--trial-0023-R-entries32-s42-0c3fc7ea/README.md) | 42 | 0.634643 | 220.8 | 226.9 | 38057 | 30,019,884 / 30,019,884 | 653.8 / 690.0 | False |
| [trial-0024-R-entries128-s42](../ncp-20260915--trial-0024-R-entries128-s42-c922a0b0/README.md) | 42 | 0.634150 | 220.0 | 226.4 | 38188 | 30,130,476 / 30,130,476 | 656.7 / 694.0 | False |
| [trial-0025-R-lr0.0003-s42](../ncp-20260915--trial-0025-R-lr0.0003-s42-df807d5a/README.md) | 42 | 0.634800 | 221.2 | 227.3 | 37996 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0026-R-lr0.003-s42](../ncp-20260915--trial-0026-R-lr0.003-s42-45bd7ff8/README.md) | 42 | 0.635624 | 222.5 | 228.8 | 37721 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0027-R-vq_weight0.1-s42](../ncp-20260915--trial-0027-R-vq_weight0.1-s42-0864ad97/README.md) | 42 | 0.635949 | 220.5 | 226.4 | 38137 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0028-R-ce_weight0.1-s42](../ncp-20260915--trial-0028-R-ce_weight0.1-s42-5fc605e8/README.md) | 42 | 0.633991 | 220.2 | 225.9 | 38171 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0029-R-ce_weight1.0-s42](../ncp-20260915--trial-0029-R-ce_weight1.0-s42-cca8c2df/README.md) | 42 | 0.642264 | 218.4 | 224.0 | 38489 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0030-R-prediction_weight0.03-s42](../ncp-20260915--trial-0030-R-prediction_weight0.03-s42-40596925/README.md) | 42 | 0.635307 | 220.7 | 226.6 | 38110 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0031-R-prediction_weight0.3-s42](../ncp-20260915--trial-0031-R-prediction_weight0.3-s42-15ed2fdd/README.md) | 42 | 0.636698 | 217.8 | 223.6 | 38601 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0032-R-raw-s42](../ncp-20260915--trial-0032-R-raw-s42-d6f49d44/README.md) | 42 | 0.638488 | 218.4 | 225.6 | 38487 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0033-R-no_prediction-s42](../ncp-20260915--trial-0033-R-no_prediction-s42-6bdf165e/README.md) | 42 | 0.636180 | 218.6 | 224.3 | 38442 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0034-R-gain4-s42](../ncp-20260915--trial-0034-R-gain4-s42-4dc85cf9/README.md) | 42 | 0.637500 | 219.4 | 225.2 | 38248 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0035-R-gain8-s42](../ncp-20260915--trial-0035-R-gain8-s42-694ab183/README.md) | 42 | 0.635120 | 219.6 | 225.4 | 38221 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0036-CAP-L2-K64-s42](../ncp-20260915--trial-0036-CAP-L2-K64-s42-0d439659/README.md) | 42 | 0.635008 | 176.4 | 181.7 | 47592 | 30,056,748 / 30,056,748 | 683.0 / 716.0 | n/a |
| [trial-0037-I2-169ef2cc79-s42](../ncp-20260915--trial-0037-I2-169ef2cc79-s42-bd04fe9b/README.md) | 42 | 0.636130 | 214.6 | 221.8 | 39202 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0038-I2-0f670cafec-s42](../ncp-20260915--trial-0038-I2-0f670cafec-s42-9cd421a4/README.md) | 42 | 0.639392 | 218.3 | 224.0 | 38534 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0039-I2-37a8ad42c0-s42](../ncp-20260915--trial-0039-I2-37a8ad42c0-s42-ee3d34bd/README.md) | 42 | 0.631877 | 218.0 | 223.7 | 38576 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0040-I2-1c8a278808-s42](../ncp-20260915--trial-0040-I2-1c8a278808-s42-4e5a0c79/README.md) | 42 | 0.632987 | 220.7 | 226.5 | 38086 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0041-I2-253cd0d9c9-s42](../ncp-20260915--trial-0041-I2-253cd0d9c9-s42-811588a4/README.md) | 42 | 0.633320 | 218.3 | 224.0 | 38509 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0042-I2-491f8344d3-s42](../ncp-20260915--trial-0042-I2-491f8344d3-s42-0a3774ac/README.md) | 42 | 0.635269 | 218.4 | 224.1 | 38492 | 30,001,452 / 30,001,452 | 656.3 / 688.0 | False |
| [trial-0043-I2-2fd4eaacd2-s42](../ncp-20260915--trial-0043-I2-2fd4eaacd2-s42-3aab367f/README.md) | 42 | 0.634902 | 220.1 | 225.9 | 38201 | 30,001,452 / 30,001,452 | 670.2 / 694.0 | False |
| [trial-0044-I2-ae91a9e332-s42](../ncp-20260915--trial-0044-I2-ae91a9e332-s42-7f8617c8/README.md) | 42 | 0.632972 | 218.7 | 224.5 | 38454 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0045-I2-e4d0b728f2-s42](../ncp-20260915--trial-0045-I2-e4d0b728f2-s42-bf33b8c4/README.md) | 42 | 0.634788 | 214.4 | 220.2 | 39242 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | True |

* Early-failure parameter counts were reconstructed exactly on a meta device from captured source and logged model configuration. No missing performance measurement was reconstructed.

Depth6 uses width 384 and 26,345,772 parameters; depth12 uses width 768 and 135,267,480 parameters (5.13 times as many). This comparison changes both depth and width.

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
- trial-0019-R-after_layer2-s42: Change only after_layer to 2 on normalized NCP to test quality versus its stable unit-weight control
- trial-0020-R-chunk_size2-s42: Change only chunk_size to 2 on normalized NCP to test quality versus its stable unit-weight control
- trial-0021-R-chunk_size8-s42: Change only chunk_size to 8 on normalized NCP to test quality versus its stable unit-weight control
- trial-0022-R-entries16-s42: Change only entries to 16 on normalized NCP to test quality versus its stable unit-weight control
- trial-0023-R-entries32-s42: Change only entries to 32 on normalized NCP to test quality versus its stable unit-weight control
- trial-0024-R-entries128-s42: Change only entries to 128 on normalized NCP to test quality versus its stable unit-weight control
- trial-0025-R-lr0.0003-s42: Change only lr to 0.0003 on normalized NCP to test quality versus its stable unit-weight control
- trial-0026-R-lr0.003-s42: Change only lr to 0.003 on normalized NCP to test quality versus its stable unit-weight control
- trial-0027-R-vq_weight0.1-s42: Change only vq_weight to 0.1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0028-R-ce_weight0.1-s42: Change only ce_weight to 0.1 on normalized NCP to test quality versus its stable unit-weight control
- trial-0029-R-ce_weight1.0-s42: Change only ce_weight to 1.0 on normalized NCP to test quality versus its stable unit-weight control
- trial-0030-R-prediction_weight0.03-s42: Change only prediction_weight to 0.03 on normalized NCP to test quality versus its stable unit-weight control
- trial-0031-R-prediction_weight0.3-s42: Change only prediction_weight to 0.3 on normalized NCP to test quality versus its stable unit-weight control
- trial-0032-R-raw-s42: Test the pinned official implementation raw-logit codebook reconstruction against paper-inspired softmax; all other N-RMS settings fixed
- trial-0033-R-no_prediction-s42: Remove next-concept MSE while retaining VQ fitting and predicted latent feedback; test whether next-concept supervision contributes beyond the added latent path
- trial-0034-R-gain4-s42: Test stronger normalized feedback, gain4, because anchor predicted RMS is only6.2% of hidden RMS; stronger feedback remains an unproven hypothesis
- trial-0035-R-gain8-s42: Test the upper bounded feedback gain8; compare with gain1 anchor and gain4 to assess amplitude sensitivity
- trial-0036-CAP-L2-K64-s42: Parameter-matched token-level residual MLP adds exactly 3710976 parameters at the same insertion and AdamW LR as N-RMS; test added-capacity effects without concept prediction; compute is not matched
- trial-0037-I2-169ef2cc79-s42: Test interaction of entries=16 and feedback_scale=0.1; individual BPB 0.633742 and 0.633732 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0038-I2-0f670cafec-s42: Test interaction of ce_weight=0.1 and feedback_scale=0.1; individual BPB 0.633991 and 0.633732 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0039-I2-37a8ad42c0-s42: Test interaction of ce_weight=0.1 and entries=16; individual BPB 0.633991 and 0.633742 versus normalized anchor 0.635927; compare against stronger individual R-entries16
- trial-0040-I2-1c8a278808-s42: Test interaction of after_layer=1 and feedback_scale=0.1; individual BPB 0.634774 and 0.633732 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0041-I2-253cd0d9c9-s42: Test interaction of chunk_size=2 and feedback_scale=0.1; individual BPB 0.634780 and 0.633732 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0042-I2-491f8344d3-s42: Test interaction of after_layer=1 and entries=16; individual BPB 0.634774 and 0.633742 versus normalized anchor 0.635927; compare against stronger individual R-entries16
- trial-0043-I2-2fd4eaacd2-s42: Test interaction of chunk_size=2 and entries=16; individual BPB 0.634780 and 0.633742 versus normalized anchor 0.635927; compare against stronger individual R-entries16
- trial-0044-I2-ae91a9e332-s42: Test interaction of feedback_scale=0.1 and lr=0.0003; individual BPB 0.633732 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-feedback_scale0.1
- trial-0045-I2-e4d0b728f2-s42: Test interaction of entries=16 and lr=0.0003; individual BPB 0.633742 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-entries16

## Correctness and diagnosis

The first two unit-weight attempts hit an inherited finite total-loss100 guard. A source-verified CPU checkpoint probe found dense hidden RMS12.54 too, so those stops do not establish NCP-specific divergence. The corrected guard checks token CE separately and still rejects nonfinite total loss. Initial failed attempts remain preserved, and their historical hypotheses using the word divergence are superseded by this diagnosis. Completed BPB runs never hit that gate.
Trial25 copied a newer controller file while its long-running parent retained an earlier imported controller. Both versions and a correction receipt are retained. Their trial, preflight, candidate and health function bodies are identical; the difference is a GPU lock wrapper. The loaded-controller reference is reconstructed from the same-process import history, not direct process-memory inspection. Captured training-child sources and data are independently verified. The next controller archives immutable startup source bytes to prevent recurrence.
CPU/CUDA tests cover prefix causality, future-label isolation, VQ/encoder gradients, optimizer coverage, save/load, codebook learning and evaluation immutability. Every completed training child executes captured sources; the final evidence audit also verifies saved checkpoints and committed source-archive bytes.

## Measurement limits

Equal-token quality comparisons; measured runtime is a separate cost axis. No equal-time quality claim. Depth changes width too. Timed throughput excludes the first11 updates; all-update time includes them. Trial wall time includes preparation, child execution and verification, excluding reporting/publication. Allocator peaks exclude driver/desktop use. Whole-board sampled VRAM, dictionary bytes, exact configurations, source/data/checkpoint hashes, auxiliary losses and utilization are in the JSON receipts. Active counts describe structural training participation, not amortized per-token compute. NCP runs at chunk rate; the capacity-control MLP runs at token rate, and auxiliary-only concepts do not feed token logits. Diagnostic feedback_rms is the unscaled prediction; injected_feedback_rms applies the configured gain and is zero for auxiliary-only runs. For raw-logit mixing, reported entropy describes softmax classification probabilities, not the signed reconstruction weights. Codebook assignments do not prove semantic concepts. Repeated validation selection is exploratory, not held-out generalization.
All artifacts are retained locally. No cloud, dependency upgrades, paid services or deletion.
