# Campaign conclusions

87 completed full trials across 35 distinct configurations. No candidate-count cap. Every full trial used 512 updates and 8,388,608 tokens. Synthetic gates were separate and are not research scores.

## Independent confirmation

Frozen candidate: D12-lr0.03; control: D-depth12. Both have 135,267,480 total/active parameters and width768. Only matrix LR differs.

| Seed | Candidate BPB mean | Control BPB mean | Difference | Candidate/control repeats | First-pair difference |
|---|---:|---:|---:|---|---:|
| 43 | 0.591444 | 0.594301 | -0.002857 | 3/3 | -0.002433 |
| 44 | 0.594267 | 0.593229 | +0.001038 | 2/3 | +0.001486 |

Mean difference across the two independent seeds: -0.000909 BPB. First-pair-only sensitivity: -0.000474; balanced-repeat sensitivity: -0.000967.

The seeds disagree on direction. Do not promote LR0.03 as a reliable improvement. Seed42 was used for selection and is not a third independent confirmation. Repeats estimate execution variability, not additional seeds. Unequal repeat counts are disclosed and sensitivity calculations do not treat repeats as independent samples.

## Seed42 screen and repeat means

| Label | Repeats | BPB mean [min,max] | Timed tok/s mean | Update wall s | Child wall s | Alloc/reserved MiB peak | Total/active parameters | Width |
|---|---:|---|---:|---:|---:|---|---|---:|
| D | 8 | 0.634848 [0.634848,0.634848] | 49099.3 | 173.1 | 179.3 | 581.7/622.0 | 26,345,772/26,345,772 | 384 |
| M | 1 | 0.637217 [0.637217,0.637217] | 27487.7 | 306.0 | 312.6 | 902.2/956.0 | 47,588,652/26,354,988 | 384 |
| DG | 9 | 0.634178 [0.634178,0.634178] | 44144.9 | 190.5 | 197.0 | 599.7/624.0 | 27,443,885/26,395,437 | 384 |
| M-experts2 | 1 | 0.634345 [0.634345,0.634345] | 31489.2 | 267.2 | 273.6 | 703.8/756.0 | 33,428,268/26,350,380 | 384 |
| M-router_lr0.003 | 1 | 0.633858 [0.633858,0.633858] | 26756.5 | 314.4 | 320.8 | 901.6/956.0 | 47,588,652/26,354,988 | 384 |
| DG-layer0 | 1 | 0.634020 [0.634020,0.634020] | 43803.4 | 192.0 | 198.6 | 599.7/624.0 | 27,443,885/26,395,437 | 384 |
| D-depth8 | 8 | 0.617737 [0.617737,0.617737] | 37728.7 | 222.7 | 228.9 | 943.1/964.0 | 50,332,176/50,332,176 | 512 |
| D-depth10 | 4 | 0.609912 [0.608865,0.610642] | 30905.4 | 271.8 | 278.4 | 1511.2/1602.0 | 85,852,980/85,852,980 | 640 |
| D-depth12 | 9 | 0.590304 [0.589884,0.591172] | 26834.0 | 313.1 | 319.8 | 2277.2/2408.0 | 135,267,480/135,267,480 | 768 |
| D12-lr0.03 | 9 | 0.588723 [0.588123,0.589460] | 26935.1 | 311.9 | 318.6 | 2277.2/2408.0 | 135,267,480/135,267,480 | 768 |

Larger dense depth/width produced the largest observed quality improvement, with more parameters and slower updates. This is equal-token evidence, not an equal-time or equal-compute efficiency claim. No independent depth8/depth12 confirmation pair was run: the reserved independent comparison tested LR within depth12. MoE/router/memory findings remain selection-seed screens. Expert utilization and every configuration are preserved in the per-run reports and row receipts.

Timed throughput excludes the first11 updates; all training tokens include them. Synchronized update wall time includes CPU dispatch and optimizer work, not just GPU kernel-busy time. Allocator peaks do not measure total board usage.

## Correctness, diagnostics and provenance

Initial CPU/CUDA suites passed52 tests each; expanded-depth suites passed54 each. Full-context depth10/12 synthetic gates verified causality, outputs, gradients, optimizer coverage, save/load and comfortable GPU fit. Previously gated LR settings were combined only after separate shape and LR screens; the inherited gate receipt explicitly says no rerun was performed for that configuration-only amendment.

All training trials completed. A reporting-only identity assertion failed because schedule.json also contains intentionally different optimizer groups. The diagnostic was preserved; the repaired check compares exact schedule definitions/updates, protocol, batch identities and data seals. Full optimizer groups remain captured. Historical selection boilerplate saying no combined winners was stale for the four depth12/LR trials; their prospective hypotheses and matching controls explicitly identify the combination. The controller wording was corrected after training selection without rewriting those historical receipts.

Repeated larger-depth runs vary despite identical recorded initialization and source hashes; no specific numerical kernel is established as the cause. The historical M-control discrepancy remains unresolved. Do not claim deterministic training, equivalence, or statistical proof.

Prior artifacts are hash-verified separately. Exact executed sources are archived by hash; publication commits are never retroactive run provenance. NCP was deferred rather than added to fill a category. No dependency upgrades, cloud jobs, paid services or deletion.

## Retention

Campaign artifacts: 20.11 GiB; checkpoints: 20.03 GiB. Preserve all artifacts now. Proposed later policy, requiring separate deletion approval: keep compact reports, hashes, source snapshots, controls, independent-seed checkpoints and failures indefinitely; archive redundant same-seed checkpoints and large traces after a30-day review. Never remove datasets or provenance needed to reproduce retained results.
