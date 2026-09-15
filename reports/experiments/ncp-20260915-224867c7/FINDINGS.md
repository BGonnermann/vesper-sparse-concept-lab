# Campaign findings

Frozen NCP did not pass the predeclared primary two-seed consistency rule.

The four-seed mean NCP-minus-dense difference is -0.002320 BPB; lower is better. 2 of four paired seeds favor NCP. These are equal-token validation results, not held-out or equal-time results.

The exactly parameter-matched residual MLP averaged 0.636494 BPB versus NCP at 0.637223, with 174.8 versus 218.6 seconds of training updates. NCP did not show an advantage over this added-capacity control.

## Frozen confirmation and ablations

| Seed | Role | Dense BPB | NCP BPB | NCP minus dense |
|---:|---|---:|---:|---:|
| 45 | primary | 0.637922 | 0.640041 | +0.002119 |
| 46 | primary | 0.639401 | 0.635421 | -0.003980 |
| 43 | sensitivity | 0.637774 | 0.639233 | +0.001459 |
| 44 | sensitivity | 0.643075 | 0.634198 | -0.008877 |

| Comparison | Mean NCP minus control BPB | Seeds favoring NCP |
|---|---:|---:|
| D6 | -0.002320 | 2/4 |
| FROZEN-AUX | -0.001990 | 2/4 |
| FROZEN-CAP | +0.000729 | 2/4 |
| FROZEN-NOPRED | -0.001667 | 3/4 |

AUX keeps the selected concept objectives and parameters but removes predicted feedback. CAP replaces NCP with an exactly parameter-matched token-rate residual MLP. NOPRED removes prediction MSE and concept CE while retaining feedback, token CE and VQ learning. It is not a no-future-label or all-supervision-off condition.

The seed42 search selected 0.631877 BPB versus its fresh dense control at0.634848. That selection gain is reported separately from all four fresh-seed outcomes; the candidate was not replaced after confirmation.

## Measured cost across the four confirmation seeds

| Condition | Total / active parameters | Mean BPB | Mean update s | Mean trial s | Mean timed tok/s | Max allocated / reserved MiB | Max sampled board MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| I2-37a8ad42c0 | 30,001,452 / 30,001,452 | 0.637223 | 218.6 | 224.3 | 38453 | 653.3 / 688.0 | 2059.0 |
| D6 | 26,345,772 / 26,345,772 | 0.639543 | 164.4 | 170.9 | 51131 | 581.7 / 622.0 | 2430.0 |
| FROZEN-AUX | 30,001,452 / 30,001,452 | 0.639213 | 215.5 | 221.3 | 38974 | 653.3 / 690.0 | 2063.0 |
| FROZEN-CAP | 30,001,452 / 30,001,452 | 0.636494 | 174.8 | 180.1 | 48029 | 686.5 / 716.0 | 2087.0 |
| FROZEN-NOPRED | 30,001,452 / 30,001,452 | 0.638890 | 219.0 | 224.7 | 38396 | 653.3 / 688.0 | 2059.0 |

NCP adds 3,655,680 parameters (13.88%) and its mean all-update time is 1.33 times dense. Structural active counts are not amortized FLOPs: concepts run once per chunk; CAP runs at every token. The selected codebook has98,304 trainable transform parameters,24,576 frozen-basis bytes and24,576 effective-code bytes. No n-gram memory table is present.

Timed throughput excludes11 warmup updates. All-update time includes them; trial time additionally includes preparation, evaluation and verification, excluding publication. Sampled board memory includes desktop/driver use and may miss peaks. These instrumented wall-time measurements do not establish equal-time quality or a measured FLOPs advantage.

## Interpretation boundaries

This is a small ConceptLM-inspired mechanism test: complete-chunk pooling, causal prediction over segmented codebooks, detached future targets and delayed predicted feedback. It is not a faithful reproduction of the paper or evidence that the learned entries represent human-interpretable concepts. Official training code is unreleased; the inspected implementation and paper also differ in feedback weighting. Native SDPA, initialization, normalization, positional features, model size and TinyStories training differ here.

The mandatory native D6/D12 comparison changes both depth and width. The separate2x2 grid tests depth at fixed width and width at fixed depth under the same optimizer policy; width-dependent embedding/unembedding LR scaling remains part of that pinned policy. Matrix LR stays0.04. No optimum depth, width or training budget is established.

All per-trial configurations, hashes, failures, utilization diagnostics and source corrections are in [the full campaign report](CAMPAIGN.md). The initial finite total-loss guard failures do not establish divergence; the corrected guard and exact retries are disclosed there. Collapsed candidates remain reported and excluded from selection.

## Depth and width confirmation

| Change | Mean BPB delta | Mean update-time ratio |
|---|---:|---:|
| depth at width384 | -0.016547 | 1.98 |
| depth at width768 | -0.014130 | 1.92 |
| width at depth12 | -0.030223 | 0.99 |
| width at depth6 | -0.032640 | 1.02 |

Both seeds favor increasing either axis. The original native-depth comparison improves by0.046771 BPB on average, with5.13 times the parameters; it changes both depth and width. Similar measured runtime across widths is specific to this runtime and microbatch setup, not a FLOPs equivalence claim.

## Mechanism diagnosis

Crossed initializer/order runs traced most of the observed reversal to initialization. At fixed order42, the descriptive module-initializer contrast is +0.002696 BPB and the backbone contrast is +0.002004. Only two deliberately chosen initializer levels were tested; this is not a general variance estimate.

Removing all future-target objectives, including VQ, changes mean BPB by -0.000192 versus NOPRED on reused seeds45/46. It does not reveal a substantial VQ-supervision benefit in this pair. Token-only feedback still trains the latent path; it is an ablation, not an NCP candidate.

Every original checkpoint score reproduced within1e-6 BPB. Zeroing feedback worsens BPB by 0.000174 to 0.000440; code-identity rotations have smaller effects. Reliance on feedback does not establish an advantage over a separately trained dense or capacity control.

On the same validation batches, learned next-code accuracy is 55.7%–64.8%, versus 22.5%–45.9% for repeating the current code and 14.4%–32.6% for a training-derived majority code. Learned predictions beat both baselines on every checkpoint. This supports code-label prediction learning, not semantic concepts or a reliable token-quality benefit.

All 83 completed checkpoints independently replayed their original BPB within1e-6 (largest difference 4.85e-07), using captured source and unchanged saved weights. This confirms reproducibility of these scores, not generalization.

## Next best experiment

Compare dense D6-width768 and D12-width768 at longer fixed-token budgets on new paired seeds, with one final evaluation on an untouched test split. Measure whether the depth gain persists and warrants roughly twice the update time. Keep quality-versus-time analysis separate. Do not combine NCP with other mechanisms on the strength of these mixed results. No next campaign is launched automatically.
