# Two-Stage SBERT-FT → LEGR-V3 Scaling Results

## Outcome

The same factorized architecture was evaluated at 15, 30, and 45 tools. Each tier uses its own SBERT-FT checkpoint and its own LEGR-V3 checkpoint. SBERT-FT selects an exact tool set; a frozen V3 encoder plus a newly trained 264,705-parameter residual pair head ranks only DAGs with that tool set.

No dataset, split, existing model implementation, or pre-existing checkpoint was modified. The 30- and 45-tool V3 checkpoints and all reranker outputs were written under new `artifacts/legr_model_search/` paths.

## Twin-filled galleries

| Tools | Train DAGs | Expanded-dev DAGs | Dev twin queries | Held-out gold DAGs | Candidate-only DAGs | Final gallery | Test queries |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 248 | 297 | 150 | 50 | 272 | 322 | 300 |
| 30 | 346 | 415 | 222 | 70 | 385 | 455 | 420 |
| 45 | 498 | 597 | 342 | 100 | 550 | 650 | 600 |

All held-out queries are twin eligible.

## Main results

| Tools | Model | R@1 | R@3 | R@5 | MRR@5 | Tool F1 | Twin R@1 | Pair accuracy |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 15 | SBERT-FT | 0.2133 | 0.6100 | 0.8533 | 0.4359 | 0.9689 | 0.2333 | 0.6687 |
| 15 | LEGR-V3 | 0.2100 | 0.5667 | 0.8133 | 0.4239 | 0.9508 | 0.2367 | 0.6960 |
| 15 | Two-stage, 3-seed mean | **0.2756** | **0.6689** | **0.8789** | **0.4986** | **0.9689** | **0.2756** | 0.6830 |
| 30 | SBERT-FT | 0.1762 | 0.6000 | 0.9119 | 0.4198 | 0.9747 | 0.1762 | 0.6175 |
| 30 | LEGR-V3 | 0.2310 | 0.7524 | 0.8976 | 0.4969 | 0.9737 | 0.2357 | 0.7464 |
| 30 | Two-stage, 3-seed mean | **0.3246** | **0.8198** | **0.9524** | **0.5714** | **0.9747** | **0.3246** | **0.7569** |
| 45 | SBERT-FT | 0.1700 | 0.6500 | 0.8883 | 0.4276 | 0.9747 | 0.1767 | 0.6522 |
| 45 | LEGR-V3 | **0.4183** | 0.8167 | 0.9400 | **0.6239** | 0.9711 | **0.4333** | **0.8186** |
| 45 | Two-stage, 3-seed mean | 0.3611 | **0.8167** | **0.9467** | 0.5934 | **0.9747** | 0.3611 | 0.7650 |

The two-stage system beats SBERT-FT in exact R@1 at all three tiers while preserving SBERT-FT tool-set F1 exactly. Its advantage grows with scale: +0.0622, +0.1484, and +0.1911 at 15, 30, and 45 tools. At 45 tools, standalone V3 is the best exact-plan model, reaching 0.4183 R@1, but its tool F1 is 0.0036 below SBERT-FT. The two-stage constraint trades some of V3's structural recall for exact preservation of semantic tool selection.

## Reranker seeds

| Tools | Seed | Selected epoch | R@1 | R@3 | R@5 | MRR@5 | Tool F1 | Twin R@1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 42 | 30 | 0.2767 | 0.6833 | 0.8800 | 0.5023 | 0.9689 | 0.2767 |
| 15 | 123 | 60 | 0.2633 | 0.6333 | 0.8767 | 0.4865 | 0.9689 | 0.2633 |
| 15 | 2026 | 25 | 0.2867 | 0.6900 | 0.8800 | 0.5069 | 0.9689 | 0.2867 |
| 30 | 42 | 10 | 0.3095 | 0.8286 | 0.9548 | 0.5676 | 0.9747 | 0.3095 |
| 30 | 123 | 5 | 0.3262 | 0.8119 | 0.9500 | 0.5696 | 0.9747 | 0.3262 |
| 30 | 2026 | 5 | 0.3381 | 0.8190 | 0.9524 | 0.5770 | 0.9747 | 0.3381 |
| 45 | 42 | 15 | 0.3867 | 0.8267 | 0.9483 | 0.6080 | 0.9747 | 0.3867 |
| 45 | 123 | 5 | 0.3217 | 0.8033 | 0.9450 | 0.5711 | 0.9747 | 0.3217 |
| 45 | 2026 | 15 | 0.3750 | 0.8200 | 0.9467 | 0.6011 | 0.9747 | 0.3750 |

## Paired clustered uncertainty versus SBERT-FT

Bootstrap resampling uses canonical held-out DAG as the cluster, retaining all six paraphrases together.

| Tools | R@1 change | R@1 95% CI | Twin-R@1 change | Twin 95% CI | Tool-F1 change |
|---:|---:|---:|---:|---:|---:|
| 15 | +0.0622 | [-0.0356, 0.1644] | +0.0422 | [-0.0589, 0.1467] | 0.0000 |
| 30 | +0.1484 | [0.0563, 0.2405] | +0.1472 | [0.0536, 0.2377] | 0.0000 |
| 45 | +0.1911 | [0.1244, 0.2589] | +0.1828 | [0.1183, 0.2483] | 0.0000 |

The 30- and 45-tool improvements are statistically supported under the predeclared DAG-clustered bootstrap. The 15-tool direction is positive across all three reranker seeds but is not conclusive.

## Evidence chronology

- Tier 15 is post-hoc exploratory because the 322-DAG gallery had been inspected before the two-stage architecture was proposed.
- The architecture was fixed from the 15-tool study before running the 30- and 45-tool held-out scaling evaluations.
- Each scaling tier used only its development queries for checkpoint selection; there was no test-time model or epoch selection.
- One seed-42 base V3 checkpoint was trained per tier, matching the inherited-checkpoint convention at 15 tools. The three reported seeds vary only the residual reranker initialization and pair order.

The 30-tool V3 checkpoint selected epoch 49 with validation loss 2.1935885. The 45-tool checkpoint selected epoch 57 with validation loss 2.1997080. Both use the 75-epoch cap, patience 15, batch size 128, shared MiniLM backbone, directed SetGNN, and `lambda_ged=0` configuration of V3-15.

## Artifact locations

- 30-tool V3: `artifacts/legr_model_search/v3_scale/legr_setgnn_tied_no_ged_30t_s42/`
- 45-tool V3: `artifacts/legr_model_search/v3_scale/legr_setgnn_tied_no_ged_45t_s42/`
- 30-tool final evaluation: `artifacts/legr_model_search/two_stage_v3_reranker_30t_final/`
- 45-tool final evaluation: `artifacts/legr_model_search/two_stage_v3_reranker_45t_final/`

Both new evaluation integrity reports contain empty `changed`, `missing`, and `added` lists.
