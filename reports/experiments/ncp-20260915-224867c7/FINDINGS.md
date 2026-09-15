# Campaign findings

Frozen NCP did not pass the predeclared primary two-seed consistency rule.

The four-seed mean NCP-minus-dense difference is -0.002320 BPB; lower is better. 2 of four paired seeds favor NCP. These are equal-token validation results, not held-out or equal-time results.

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
