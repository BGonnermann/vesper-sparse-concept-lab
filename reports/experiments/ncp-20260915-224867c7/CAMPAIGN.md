# NCP campaign progress

76 completed of 78 attempted full trials. Budget: 11:41:32 to 19:41:32 UTC, 2026-09-15.
512 updates and 8,388,608 tokens per full trial. Seed42 screens are exploratory. Lower BPB is better.

Strongest eligible selection-seed NCP: I2-37a8ad42c0, 0.631877 BPB, delta -0.002971 versus D6. Adds 3,655,680 parameters; measured update-time ratio 1.23. This is a search result; independent confirmation is reported separately.

## Implementation

Dense encoder pools complete multi-token chunks; causal chunk Transformers predict segmented discrete-codebook weights. Only predicted concepts feed the token decoder, delayed by k-1 positions. Detached future chunks supervise NCP MSE; VQ MSE fits a transformed frozen random codebook basis. Token BPB excludes both auxiliary losses.
This is a simplified ConceptLM-inspired prototype, not a paper reproduction. Initial softmax feedback differs from the official GPT2/Pythia raw-logit multiplication; raw-logit variants are separately labeled. Native SDPA, initialization, positional features, codebook transforms and the small TinyStories fixed-token experiment also differ. Official revision: a0ab281286f5c0337c35de3181cc992c562eacaa.
See [campaign plan](../../../docs/ncp-campaign.md) and the published source receipt for exact references.

## Every attempted trial

| Trial | Init / order seed | BPB | Update s | Trial s | Timed tok/s | Total / active params | Alloc / reserved MiB | Target-code collapse |
|---|---:|---:|---:|---:|---:|---|---|---|
| [trial-0001-D6-s42](../ncp-20260915--trial-0001-D6-s42-b009c334/README.md) | 42 / 42 | 0.634848 | 177.1 | 184.0 | 47526 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0002-NCP-s42](../ncp-20260915--trial-0002-NCP-s42-22537f23/README.md) | 42 / 42 | failed | unavailable | 80.0 | unavailable | 30,056,748 / 30,056,748 * | 654.4 / 690.0 | unavailable |
| [trial-0003-AUX-s42](../ncp-20260915--trial-0003-AUX-s42-6ac0782c/README.md) | 42 / 42 | failed | unavailable | 82.4 | unavailable | 30,056,748 / 30,056,748 * | 654.4 / 690.0 | unavailable |
| [trial-0004-N-prediction_weight0.1-s42](../ncp-20260915--trial-0004-N-prediction_weight0.1-s42-d0ad5ff9/README.md) | 42 / 42 | 0.635524 | 202.6 | 210.3 | 41362 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0005-N-prediction_weight0.03-s42](../ncp-20260915--trial-0005-N-prediction_weight0.03-s42-cd708cd3/README.md) | 42 / 42 | 0.635888 | 203.3 | 211.0 | 41265 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0006-N-RMS-s42](../ncp-20260915--trial-0006-N-RMS-s42-d3652f3a/README.md) | 42 / 42 | 0.635927 | 214.2 | 221.8 | 39314 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0007-N-RMS-AUX-s42](../ncp-20260915--trial-0007-N-RMS-AUX-s42-d5def34d/README.md) | 42 / 42 | 0.635215 | 204.9 | 212.4 | 41139 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0008-D6-s43](../ncp-20260915--trial-0008-D6-s43-6ae7cb75/README.md) | 43 / 43 | 0.637774 | 163.7 | 170.5 | 51373 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0009-D12-s43](../ncp-20260915--trial-0009-D12-s43-3e98bde0/README.md) | 43 / 43 | 0.594495 | 308.5 | 316.9 | 27244 | 135,267,480 / 135,267,480 | 2277.2 / 2408.0 | n/a |
| [trial-0010-D6-s44](../ncp-20260915--trial-0010-D6-s44-8211e6c7/README.md) | 44 / 44 | 0.643075 | 158.5 | 165.5 | 52909 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0011-D12-s44](../ncp-20260915--trial-0011-D12-s44-afbc92ee/README.md) | 44 / 44 | 0.592813 | 323.0 | 331.3 | 26022 | 135,267,480 / 135,267,480 | 2277.2 / 2408.0 | n/a |
| [trial-0012-NCP-s42](../ncp-20260915--trial-0012-NCP-s42-bf446fb5/README.md) | 42 / 42 | 0.649647 | 220.4 | 227.8 | 38159 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0013-AUX-s42](../ncp-20260915--trial-0013-AUX-s42-bb8c5a76/README.md) | 42 / 42 | 0.652928 | 217.4 | 224.8 | 38680 | 30,056,748 / 30,056,748 | 654.4 / 690.0 | False |
| [trial-0014-R-prediction_weight0.1-s42](../ncp-20260915--trial-0014-R-prediction_weight0.1-s42-06b4af48/README.md) | 42 / 42 | 0.636121 | 220.2 | 227.7 | 38211 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0015-R-feedback_scale0.1-s42](../ncp-20260915--trial-0015-R-feedback_scale0.1-s42-23081d16/README.md) | 42 / 42 | 0.633732 | 220.0 | 226.0 | 38226 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0016-R-feedback_scale0.3-s42](../ncp-20260915--trial-0016-R-feedback_scale0.3-s42-866e46eb/README.md) | 42 / 42 | 0.634247 | 222.0 | 228.0 | 37882 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0017-R-layers1-s42](../ncp-20260915--trial-0017-R-layers1-s42-341c3f13/README.md) | 42 / 42 | 0.638393 | 207.8 | 214.1 | 40394 | 28,287,276 / 28,287,276 | 625.4 / 664.0 | False |
| [trial-0018-R-after_layer1-s42](../ncp-20260915--trial-0018-R-after_layer1-s42-93561fc2/README.md) | 42 / 42 | 0.634774 | 221.0 | 226.9 | 38057 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0019-R-after_layer2-s42](../ncp-20260915--trial-0019-R-after_layer2-s42-bbb25dff/README.md) | 42 / 42 | 0.635594 | 217.0 | 223.7 | 38695 | 30,056,748 / 30,056,748 | 655.8 / 688.0 | False |
| [trial-0020-R-chunk_size2-s42](../ncp-20260915--trial-0020-R-chunk_size2-s42-94b10b74/README.md) | 42 / 42 | 0.634780 | 219.5 | 226.8 | 38242 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0021-R-chunk_size8-s42](../ncp-20260915--trial-0021-R-chunk_size8-s42-4e711e4f/README.md) | 42 / 42 | 0.637063 | 218.2 | 224.3 | 38464 | 30,056,748 / 30,056,748 | 652.1 / 686.0 | False |
| [trial-0022-R-entries16-s42](../ncp-20260915--trial-0022-R-entries16-s42-2bda239a/README.md) | 42 / 42 | 0.633742 | 218.3 | 227.1 | 38544 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0023-R-entries32-s42](../ncp-20260915--trial-0023-R-entries32-s42-0c3fc7ea/README.md) | 42 / 42 | 0.634643 | 220.8 | 226.9 | 38057 | 30,019,884 / 30,019,884 | 653.8 / 690.0 | False |
| [trial-0024-R-entries128-s42](../ncp-20260915--trial-0024-R-entries128-s42-c922a0b0/README.md) | 42 / 42 | 0.634150 | 220.0 | 226.4 | 38188 | 30,130,476 / 30,130,476 | 656.7 / 694.0 | False |
| [trial-0025-R-lr0.0003-s42](../ncp-20260915--trial-0025-R-lr0.0003-s42-df807d5a/README.md) | 42 / 42 | 0.634800 | 221.2 | 227.3 | 37996 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0026-R-lr0.003-s42](../ncp-20260915--trial-0026-R-lr0.003-s42-45bd7ff8/README.md) | 42 / 42 | 0.635624 | 222.5 | 228.8 | 37721 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0027-R-vq_weight0.1-s42](../ncp-20260915--trial-0027-R-vq_weight0.1-s42-0864ad97/README.md) | 42 / 42 | 0.635949 | 220.5 | 226.4 | 38137 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0028-R-ce_weight0.1-s42](../ncp-20260915--trial-0028-R-ce_weight0.1-s42-5fc605e8/README.md) | 42 / 42 | 0.633991 | 220.2 | 225.9 | 38171 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0029-R-ce_weight1.0-s42](../ncp-20260915--trial-0029-R-ce_weight1.0-s42-cca8c2df/README.md) | 42 / 42 | 0.642264 | 218.4 | 224.0 | 38489 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0030-R-prediction_weight0.03-s42](../ncp-20260915--trial-0030-R-prediction_weight0.03-s42-40596925/README.md) | 42 / 42 | 0.635307 | 220.7 | 226.6 | 38110 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0031-R-prediction_weight0.3-s42](../ncp-20260915--trial-0031-R-prediction_weight0.3-s42-15ed2fdd/README.md) | 42 / 42 | 0.636698 | 217.8 | 223.6 | 38601 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0032-R-raw-s42](../ncp-20260915--trial-0032-R-raw-s42-d6f49d44/README.md) | 42 / 42 | 0.638488 | 218.4 | 225.6 | 38487 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0033-R-no_prediction-s42](../ncp-20260915--trial-0033-R-no_prediction-s42-6bdf165e/README.md) | 42 / 42 | 0.636180 | 218.6 | 224.3 | 38442 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0034-R-gain4-s42](../ncp-20260915--trial-0034-R-gain4-s42-4dc85cf9/README.md) | 42 / 42 | 0.637500 | 219.4 | 225.2 | 38248 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0035-R-gain8-s42](../ncp-20260915--trial-0035-R-gain8-s42-694ab183/README.md) | 42 / 42 | 0.635120 | 219.6 | 225.4 | 38221 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0036-CAP-L2-K64-s42](../ncp-20260915--trial-0036-CAP-L2-K64-s42-0d439659/README.md) | 42 / 42 | 0.635008 | 176.4 | 181.7 | 47592 | 30,056,748 / 30,056,748 | 683.0 / 716.0 | n/a |
| [trial-0037-I2-169ef2cc79-s42](../ncp-20260915--trial-0037-I2-169ef2cc79-s42-bd04fe9b/README.md) | 42 / 42 | 0.636130 | 214.6 | 221.8 | 39202 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0038-I2-0f670cafec-s42](../ncp-20260915--trial-0038-I2-0f670cafec-s42-9cd421a4/README.md) | 42 / 42 | 0.639392 | 218.3 | 224.0 | 38534 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0039-I2-37a8ad42c0-s42](../ncp-20260915--trial-0039-I2-37a8ad42c0-s42-ee3d34bd/README.md) | 42 / 42 | 0.631877 | 218.0 | 223.7 | 38576 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0040-I2-1c8a278808-s42](../ncp-20260915--trial-0040-I2-1c8a278808-s42-4e5a0c79/README.md) | 42 / 42 | 0.632987 | 220.7 | 226.5 | 38086 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0041-I2-253cd0d9c9-s42](../ncp-20260915--trial-0041-I2-253cd0d9c9-s42-811588a4/README.md) | 42 / 42 | 0.633320 | 218.3 | 224.0 | 38509 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0042-I2-491f8344d3-s42](../ncp-20260915--trial-0042-I2-491f8344d3-s42-0a3774ac/README.md) | 42 / 42 | 0.635269 | 218.4 | 224.1 | 38492 | 30,001,452 / 30,001,452 | 656.3 / 688.0 | False |
| [trial-0043-I2-2fd4eaacd2-s42](../ncp-20260915--trial-0043-I2-2fd4eaacd2-s42-3aab367f/README.md) | 42 / 42 | 0.634902 | 220.1 | 225.9 | 38201 | 30,001,452 / 30,001,452 | 670.2 / 694.0 | False |
| [trial-0044-I2-ae91a9e332-s42](../ncp-20260915--trial-0044-I2-ae91a9e332-s42-7f8617c8/README.md) | 42 / 42 | 0.632972 | 218.7 | 224.5 | 38454 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0045-I2-e4d0b728f2-s42](../ncp-20260915--trial-0045-I2-e4d0b728f2-s42-bf33b8c4/README.md) | 42 / 42 | 0.634788 | 214.4 | 220.2 | 39242 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | True |
| [trial-0046-I2-4919c50f43-s42](../ncp-20260915--trial-0046-I2-4919c50f43-s42-1f63b17c/README.md) | 42 / 42 | 0.632179 | 219.5 | 225.3 | 38285 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0047-I2-1736a5fa37-s42](../ncp-20260915--trial-0047-I2-1736a5fa37-s42-a4b9cecf/README.md) | 42 / 42 | 0.633942 | 218.8 | 224.6 | 38356 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0048-I2-b248268b0f-s42](../ncp-20260915--trial-0048-I2-b248268b0f-s42-df31e309/README.md) | 42 / 42 | 0.633824 | 219.4 | 225.2 | 38336 | 30,056,748 / 30,056,748 | 654.8 / 690.0 | False |
| [trial-0049-I2-9178a17b2b-s42](../ncp-20260915--trial-0049-I2-9178a17b2b-s42-a90c14b8/README.md) | 42 / 42 | 0.634048 | 218.5 | 224.2 | 38467 | 30,056,748 / 30,056,748 | 674.8 / 712.0 | False |
| [trial-0050-I2-8e935c3aca-s42](../ncp-20260915--trial-0050-I2-8e935c3aca-s42-560d02c3/README.md) | 42 / 42 | 0.635248 | 218.6 | 224.3 | 38455 | 30,056,748 / 30,056,748 | 657.8 / 690.0 | False |
| [trial-0051-I2-2ceac4edb4-s42](../ncp-20260915--trial-0051-I2-2ceac4edb4-s42-1ccc71c7/README.md) | 42 / 42 | 0.635224 | 218.4 | 224.1 | 38497 | 30,056,748 / 30,056,748 | 671.9 / 698.0 | False |
| [trial-0052-I3-36eb3a7c14-s42](../ncp-20260915--trial-0052-I3-36eb3a7c14-s42-dd7b51fd/README.md) | 42 / 42 | 0.635110 | 218.6 | 225.9 | 38446 | 30,001,452 / 30,001,452 | 656.3 / 688.0 | False |
| [trial-0053-I3-81772591c9-s42](../ncp-20260915--trial-0053-I3-81772591c9-s42-c8449081/README.md) | 42 / 42 | 0.632797 | 210.8 | 216.6 | 39918 | 30,001,452 / 30,001,452 | 670.2 / 694.0 | False |
| [trial-0054-I3-8fc5ce9693-s42](../ncp-20260915--trial-0054-I3-8fc5ce9693-s42-a8e4c2bf/README.md) | 42 / 42 | 0.633706 | 214.8 | 220.4 | 39141 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0055-I3-da7015e3e7-s42](../ncp-20260915--trial-0055-I3-da7015e3e7-s42-802a8bdb/README.md) | 42 / 42 | 0.633584 | 219.1 | 224.9 | 38376 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | True |
| [trial-0056-D6-s45](../ncp-20260915--trial-0056-D6-s45-b6df3f4a/README.md) | 45 / 45 | 0.637922 | 169.7 | 176.5 | 49521 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0057-I2-37a8ad42c0-s45](../ncp-20260915--trial-0057-I2-37a8ad42c0-s45-142a64df/README.md) | 45 / 45 | 0.640041 | 217.8 | 223.4 | 38577 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0058-FROZEN-AUX-s45](../ncp-20260915--trial-0058-FROZEN-AUX-s45-b6e43569/README.md) | 45 / 45 | 0.638363 | 216.1 | 221.9 | 38905 | 30,001,452 / 30,001,452 | 653.3 / 690.0 | False |
| [trial-0059-FROZEN-CAP-s45](../ncp-20260915--trial-0059-FROZEN-CAP-s45-3d706092/README.md) | 45 / 45 | 0.638105 | 176.9 | 182.2 | 47499 | 30,001,452 / 30,001,452 | 686.5 / 716.0 | n/a |
| [trial-0060-FROZEN-NOPRED-s45](../ncp-20260915--trial-0060-FROZEN-NOPRED-s45-45761a66/README.md) | 45 / 45 | 0.639167 | 218.1 | 223.8 | 38565 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0061-D6-s46](../ncp-20260915--trial-0061-D6-s46-444edc46/README.md) | 46 / 46 | 0.639401 | 165.8 | 171.1 | 50721 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0062-I2-37a8ad42c0-s46](../ncp-20260915--trial-0062-I2-37a8ad42c0-s46-94257b4a/README.md) | 46 / 46 | 0.635421 | 216.1 | 221.9 | 38863 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0063-FROZEN-AUX-s46](../ncp-20260915--trial-0063-FROZEN-AUX-s46-8ca2da5b/README.md) | 46 / 46 | 0.637895 | 216.1 | 221.8 | 38871 | 30,001,452 / 30,001,452 | 653.3 / 690.0 | False |
| [trial-0064-FROZEN-CAP-s46](../ncp-20260915--trial-0064-FROZEN-CAP-s46-a94324a7/README.md) | 46 / 46 | 0.635740 | 171.9 | 177.1 | 48866 | 30,001,452 / 30,001,452 | 686.5 / 716.0 | n/a |
| [trial-0065-FROZEN-NOPRED-s46](../ncp-20260915--trial-0065-FROZEN-NOPRED-s46-9e4a889f/README.md) | 46 / 46 | 0.638454 | 218.1 | 223.8 | 38553 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0066-I2-37a8ad42c0-s43](../ncp-20260915--trial-0066-I2-37a8ad42c0-s43-83d15adb/README.md) | 43 / 43 | 0.639233 | 222.5 | 228.2 | 37793 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0067-FROZEN-AUX-s43](../ncp-20260915--trial-0067-FROZEN-AUX-s43-0c6af3a5/README.md) | 43 / 43 | 0.637646 | 212.9 | 218.6 | 39421 | 30,001,452 / 30,001,452 | 653.3 / 690.0 | False |
| [trial-0068-FROZEN-CAP-s43](../ncp-20260915--trial-0068-FROZEN-CAP-s43-4d62a0b4/README.md) | 43 / 43 | 0.637069 | 175.5 | 180.8 | 47794 | 30,001,452 / 30,001,452 | 686.5 / 716.0 | n/a |
| [trial-0069-FROZEN-NOPRED-s43](../ncp-20260915--trial-0069-FROZEN-NOPRED-s43-c32cec4d/README.md) | 43 / 43 | 0.640659 | 219.7 | 225.5 | 38242 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0070-I2-37a8ad42c0-s44](../ncp-20260915--trial-0070-I2-37a8ad42c0-s44-f0eb8e58/README.md) | 44 / 44 | 0.634198 | 217.9 | 223.6 | 38578 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0071-FROZEN-AUX-s44](../ncp-20260915--trial-0071-FROZEN-AUX-s44-c5dc09aa/README.md) | 44 / 44 | 0.642949 | 217.0 | 222.7 | 38699 | 30,001,452 / 30,001,452 | 653.3 / 690.0 | True |
| [trial-0072-FROZEN-CAP-s44](../ncp-20260915--trial-0072-FROZEN-CAP-s44-90db58b4/README.md) | 44 / 44 | 0.635063 | 175.0 | 180.2 | 47959 | 30,001,452 / 30,001,452 | 686.5 / 716.0 | n/a |
| [trial-0073-FROZEN-NOPRED-s44](../ncp-20260915--trial-0073-FROZEN-NOPRED-s44-bdbd4373/README.md) | 44 / 44 | 0.637281 | 219.9 | 225.7 | 38224 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0074-CROSS-D6-s42](../ncp-20260915--trial-0074-CROSS-D6-s42-4f6c1484/README.md) | 42 / 45 | 0.640354 | 168.1 | 174.8 | 49995 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0075-CROSS-NCP-s42](../ncp-20260915--trial-0075-CROSS-NCP-s42-bfea33ae/README.md) | 42 / 45 | 0.635876 | 218.3 | 224.1 | 38516 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0076-CROSS-D6-s45](../ncp-20260915--trial-0076-CROSS-D6-s45-13536966/README.md) | 45 / 42 | 0.635365 | 169.8 | 175.1 | 49514 | 26,345,772 / 26,345,772 | 581.7 / 622.0 | n/a |
| [trial-0077-CROSS-NCP-s45](../ncp-20260915--trial-0077-CROSS-NCP-s45-695f2d13/README.md) | 45 / 42 | 0.637095 | 218.9 | 224.7 | 38390 | 30,001,452 / 30,001,452 | 653.3 / 688.0 | False |
| [trial-0078-D12-W384-s43](../ncp-20260915--trial-0078-D12-W384-s43-d2aa9dbf/README.md) | 43 / 43 | 0.622336 | 317.5 | 324.6 | 26471 | 46,400,088 / 46,400,088 | 942.8 / 992.0 | n/a |

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
| 45 | I2-37a8ad42c0 | D6 | 0.640041 | 0.637922 | +0.002119 | 1.28 |
| 46 | I2-37a8ad42c0 | D6 | 0.635421 | 0.639401 | -0.003980 | 1.30 |
| 43 | I2-37a8ad42c0 | D6 | 0.639233 | 0.637774 | +0.001459 | 1.36 |
| 44 | I2-37a8ad42c0 | D6 | 0.634198 | 0.643075 | -0.008877 | 1.37 |

Mean paired delta: -0.002320 BPB. Mean update-time ratio: 1.33.

## Feedback versus auxiliary-only ablations

Negative delta favors predicted-concept feedback. These selection-seed comparisons are exploratory.

| Feedback configuration | Auxiliary-only configuration | Seed | Feedback BPB | Auxiliary BPB | Delta |
|---|---|---:|---:|---:|---:|
| NCP | AUX | 42 | 0.649647 | 0.652928 | -0.003281 |
| N-RMS | N-RMS-AUX | 42 | 0.635927 | 0.635215 | +0.000712 |

## Frozen mechanism ablations

Seeds45/46 are the predeclared primary pairs;43/44 are sensitivity pairs. Each row binds exact trial IDs, configurations, executed sources, protocol, batch order, schedule, optimizer settings and shared initialization. AUX changes only mode; NOPRED removes both prediction objectives; CAP matches the selected added parameter count. Negative delta favors frozen NCP. No confirmation outcome selects a replacement.

| Seed | Role | Control | NCP BPB | Control BPB | Delta |
|---:|---|---|---:|---:|---:|
| 45 | primary | FROZEN-AUX | 0.640041 | 0.638363 | +0.001678 |
| 45 | primary | FROZEN-CAP | 0.640041 | 0.638105 | +0.001936 |
| 45 | primary | FROZEN-NOPRED | 0.640041 | 0.639167 | +0.000874 |
| 46 | primary | FROZEN-AUX | 0.635421 | 0.637895 | -0.002474 |
| 46 | primary | FROZEN-CAP | 0.635421 | 0.635740 | -0.000319 |
| 46 | primary | FROZEN-NOPRED | 0.635421 | 0.638454 | -0.003033 |
| 43 | sensitivity | FROZEN-AUX | 0.639233 | 0.637646 | +0.001587 |
| 43 | sensitivity | FROZEN-CAP | 0.639233 | 0.637069 | +0.002164 |
| 43 | sensitivity | FROZEN-NOPRED | 0.639233 | 0.640659 | -0.001426 |
| 44 | sensitivity | FROZEN-AUX | 0.634198 | 0.642949 | -0.008751 |
| 44 | sensitivity | FROZEN-CAP | 0.634198 | 0.635063 | -0.000865 |
| 44 | sensitivity | FROZEN-NOPRED | 0.634198 | 0.637281 | -0.003083 |

D6: 4/4 pairs; mean delta -0.002320; 2 negative signs. Primary same-sign improvement: False.

FROZEN-AUX: 4/4 pairs; mean delta -0.001990; 2 negative signs. Primary same-sign improvement: False.

FROZEN-CAP: 4/4 pairs; mean delta +0.000729; 2 negative signs. Primary same-sign improvement: False.

FROZEN-NOPRED: 4/4 pairs; mean delta -0.001667; 3 negative signs. Primary same-sign improvement: False.

## Fixed-width depth decomposition

Predeclared depth6/12 by width384/768, seeds43/44, matrixLR .04, fixed tokens; no configuration selection
5/8 conditions completed.

| Comparison | Seed | Candidate BPB | Control BPB | Delta | Parameter ratio | Update-time ratio |
|---|---:|---:|---:|---:|---:|---:|
| depth_at_width384 | 43 | 0.622336 | 0.637774 | -0.015438 | 1.76 | 1.94 |
| width_at_depth12 | 43 | 0.594495 | 0.622336 | -0.027841 | 2.92 | 0.97 |

Transfer entry status: not_entered. Frozen D6 mechanism did not pass the predeclared primary same-sign rule

## Initializer versus batch-order diagnosis

Two deliberately chosen seed levels diagnose the observed reversal; not four independent replications or held-out evidence. Utilization samples follow order seed.

| Initialization seed | Batch-order seed | Dense BPB | NCP BPB | NCP minus dense | Evidence |
|---:|---:|---:|---:|---:|---|
| 42 | 42 | 0.634848 | 0.631877 | -0.002971 | Reused diagonal |
| 42 | 45 | 0.640354 | 0.635876 | -0.004478 | New off-diagonal pair |
| 45 | 42 | 0.635365 | 0.637095 | +0.001730 | New off-diagonal pair |
| 45 | 45 | 0.637922 | 0.640041 | +0.002119 | Reused diagonal |

Contrasts operate on NCP-minus-dense BPB; positive means a less favorable NCP effect.

| Descriptive contrast | BPB |
|---|---:|
| initialization 45 minus 42 | +0.005649 |
| order 45 minus 42 | -0.000559 |
| interaction difference of differences | +0.001896 |

The optional order-seed field changes only the tape permutation; the architecture and objectives stay frozen. Actual-loop tests verify default compatibility, unchanged initialization when order changes, unchanged order when initialization changes, and complete tape coverage. New runs are excluded from selection and independent-seed confirmation summaries.

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
- trial-0046-I2-4919c50f43-s42: Test interaction of after_layer=1 and ce_weight=0.1; individual BPB 0.634774 and 0.633991 versus normalized anchor 0.635927; compare against stronger individual R-ce_weight0.1
- trial-0047-I2-1736a5fa37-s42: Test interaction of ce_weight=0.1 and chunk_size=2; individual BPB 0.633991 and 0.634780 versus normalized anchor 0.635927; compare against stronger individual R-ce_weight0.1
- trial-0048-I2-b248268b0f-s42: Test interaction of ce_weight=0.1 and lr=0.0003; individual BPB 0.633991 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-ce_weight0.1
- trial-0049-I2-9178a17b2b-s42: Test interaction of after_layer=1 and chunk_size=2; individual BPB 0.634774 and 0.634780 versus normalized anchor 0.635927; compare against stronger individual R-after_layer1
- trial-0050-I2-8e935c3aca-s42: Test interaction of after_layer=1 and lr=0.0003; individual BPB 0.634774 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-after_layer1
- trial-0051-I2-2ceac4edb4-s42: Test interaction of chunk_size=2 and lr=0.0003; individual BPB 0.634780 and 0.634800 versus normalized anchor 0.635927; compare against stronger individual R-chunk_size2
- trial-0052-I3-36eb3a7c14-s42: Frozen third-factor stage: add after_layer=1 from healthy individual R-after_layer1 (0.634774 BPB) to qualifying pair I2-37a8ad42c0 (0.631877); test incremental benefit, not assumed additivity; enforce utilization gate
- trial-0053-I3-81772591c9-s42: Frozen third-factor stage: add chunk_size=2 from healthy individual R-chunk_size2 (0.634780 BPB) to qualifying pair I2-37a8ad42c0 (0.631877); test incremental benefit, not assumed additivity; enforce utilization gate
- trial-0054-I3-8fc5ce9693-s42: Frozen third-factor stage: add feedback_scale=0.1 from healthy individual R-feedback_scale0.1 (0.633732 BPB) to qualifying pair I2-37a8ad42c0 (0.631877); test incremental benefit, not assumed additivity; enforce utilization gate
- trial-0055-I3-da7015e3e7-s42: Frozen third-factor stage: add lr=0.0003 from healthy individual R-lr0.0003 (0.634800 BPB) to qualifying pair I2-37a8ad42c0 (0.631877); test incremental benefit, not assumed additivity; enforce utilization gate
- trial-0056-D6-s45: Frozen four-seed confirmation: D6, seed45; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0057-I2-37a8ad42c0-s45: Frozen four-seed confirmation: I2-37a8ad42c0, seed45; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0058-FROZEN-AUX-s45: Frozen four-seed confirmation: FROZEN-AUX, seed45; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0059-FROZEN-CAP-s45: Frozen four-seed confirmation: FROZEN-CAP, seed45; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0060-FROZEN-NOPRED-s45: Frozen four-seed confirmation: FROZEN-NOPRED, seed45; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0061-D6-s46: Frozen four-seed confirmation: D6, seed46; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0062-I2-37a8ad42c0-s46: Frozen four-seed confirmation: I2-37a8ad42c0, seed46; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0063-FROZEN-AUX-s46: Frozen four-seed confirmation: FROZEN-AUX, seed46; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0064-FROZEN-CAP-s46: Frozen four-seed confirmation: FROZEN-CAP, seed46; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0065-FROZEN-NOPRED-s46: Frozen four-seed confirmation: FROZEN-NOPRED, seed46; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0066-I2-37a8ad42c0-s43: Frozen four-seed confirmation: I2-37a8ad42c0, seed43; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0067-FROZEN-AUX-s43: Frozen four-seed confirmation: FROZEN-AUX, seed43; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0068-FROZEN-CAP-s43: Frozen four-seed confirmation: FROZEN-CAP, seed43; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0069-FROZEN-NOPRED-s43: Frozen four-seed confirmation: FROZEN-NOPRED, seed43; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0070-I2-37a8ad42c0-s44: Frozen four-seed confirmation: I2-37a8ad42c0, seed44; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0071-FROZEN-AUX-s44: Frozen four-seed confirmation: FROZEN-AUX, seed44; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0072-FROZEN-CAP-s44: Frozen four-seed confirmation: FROZEN-CAP, seed44; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0073-FROZEN-NOPRED-s44: Frozen four-seed confirmation: FROZEN-NOPRED, seed44; selection trial-0039-I2-37a8ad42c0-s42; compare token BPB with dense, mode-only AUX, exact parameter-matched MLP and prediction-objective-off; no reselection
- trial-0074-CROSS-D6-s42: Separate initialization and training-order sensitivity of the already frozen NCP-minus-dense difference; no reselection; initializer 42, batch order 45
- trial-0075-CROSS-NCP-s42: Separate initialization and training-order sensitivity of the already frozen NCP-minus-dense difference; no reselection; initializer 42, batch order 45
- trial-0076-CROSS-D6-s45: Separate initialization and training-order sensitivity of the already frozen NCP-minus-dense difference; no reselection; initializer 45, batch order 42
- trial-0077-CROSS-NCP-s45: Separate initialization and training-order sensitivity of the already frozen NCP-minus-dense difference; no reselection; initializer 45, batch order 42
- trial-0078-D12-W384-s43: Separate the already confirmed joint depth/width gain into fixed-width depth and fixed-depth width comparisons; no tuning

## Correctness and diagnosis

The first two unit-weight attempts hit an inherited finite total-loss100 guard. A source-verified CPU checkpoint probe found dense hidden RMS12.54 too, so those stops do not establish NCP-specific divergence. The corrected guard checks token CE separately and still rejects nonfinite total loss. Initial failed attempts remain preserved, and their historical hypotheses using the word divergence are superseded by this diagnosis. Completed BPB runs never hit that gate.
Trials1-3 initially retained controller hashes without controller files. The exact bytes were later recovered from Git revision26d471ba734dcce5aebd843aacb7b95f73170a61 and match every original recorded SHA256. The recovery receipt distinguishes these recovered files from trial-time archives; training-child sources were originally captured.
Trial25 copied a newer controller file while its long-running parent retained an earlier imported controller. Both versions and a correction receipt are retained. Their trial, preflight, candidate and health function bodies are identical; the difference is a GPU lock wrapper. The loaded-controller reference is reconstructed from the same-process import history, not direct process-memory inspection. Captured training-child sources and data are independently verified. The next controller archives immutable startup source bytes to prevent recurrence.
CPU/CUDA tests cover prefix causality, future-label isolation, VQ/encoder gradients, optimizer coverage, save/load, codebook learning and evaluation immutability. Every completed training child executes captured sources; the final evidence audit also verifies saved checkpoints and committed source-archive bytes.

## Measurement limits

Equal-token quality comparisons; measured runtime is a separate cost axis. No equal-time quality claim. Depth changes width too. Timed throughput excludes the first11 updates; all-update time includes them. Trial wall time includes preparation, child execution and verification, excluding reporting/publication. Allocator peaks exclude driver/desktop use. Whole-board sampled VRAM, dictionary bytes, exact configurations, source/data/checkpoint hashes, auxiliary losses and utilization are in the JSON receipts. Active counts describe structural training participation, not amortized per-token compute. NCP runs at chunk rate; the capacity-control MLP runs at token rate, and auxiliary-only concepts do not feed token logits. Diagnostic feedback_rms is the unscaled prediction; injected_feedback_rms applies the configured gain and is zero for auxiliary-only runs. For raw-logit mixing, reported entropy describes softmax classification probabilities, not the signed reconstruction weights. Codebook assignments do not prove semantic concepts. Repeated validation selection is exploratory, not held-out generalization.
All artifacts are retained locally. No cloud, dependency upgrades, paid services or deletion.
