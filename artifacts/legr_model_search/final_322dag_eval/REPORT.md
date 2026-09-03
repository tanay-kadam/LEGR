# Locked 322-DAG Final Evaluation

This report evaluates the three preselected `confirm_r1` LEGR-GPS checkpoints on the unchanged Campaign-v4 15-tool topology-held-out queries and the combined 322-DAG gallery. No training or model selection used these results.

## Gallery

- Queries: **300**
- Gold DAGs: **50**
- Candidate-only DAGs: **272**
- Combined gallery: **322**
- Unique tool sets: **49**
- Tie tolerance: `1e-05`

## Results

| Model | Seed | R@1 | R@3 | R@5 | Tool F1 | True-twin R@1 | Tie-expected twin R@1 | Mean GED | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SBERT-FT | 42 checkpoint | 0.2133 | 0.6100 | 0.8533 | 0.9689 | 0.2333 | 0.2333 | 3.0400 | 8.025 |
| LEGR V3 | 42 | 0.2100 | 0.5667 | 0.8133 | 0.9508 | 0.2367 | 0.2367 | 3.0967 | --- |
| LEGR-GPS | 42 | 0.1467 | 0.5800 | 0.7900 | 0.9655 | 0.1600 | 0.1600 | 3.0600 | 15.902 |
| LEGR-GPS | 123 | 0.1033 | 0.5900 | 0.8033 | 0.9727 | 0.1133 | 0.1133 | 3.1800 | 18.605 |
| LEGR-GPS | 2026 | 0.1500 | 0.5400 | 0.7833 | 0.9623 | 0.1600 | 0.1600 | 3.0467 | 21.194 |
| **LEGR-GPS mean** | 3 seeds | **0.1333** | 0.5700 | 0.7922 | **0.9668** | **0.1444** | 0.1444 | 3.0956 | 18.567 |

## Paired DAG-clustered bootstrap: LEGR-GPS minus SBERT-FT

- exact_recall@1: delta **-0.0800**, 95% CI **[-0.1378, -0.0211]** (50 gold-DAG clusters; 10000 resamples).
- tool_set_f1: delta **-0.0021**, 95% CI **[-0.0169, +0.0123]** (50 gold-DAG clusters; 10000 resamples).
- tie_expected_true_twin_recall@1: delta **-0.0889**, 95% CI **[-0.1500, -0.0278]** (50 gold-DAG clusters; 10000 resamples).

## Interpretation rule

The predefined final-test success criterion was not fully met. The paper must report the measured trade-off and must not claim universal superiority over SBERT-FT.

Candidate embeddings were cached once per model. Latency is synchronized batch-size-one query encoding plus scoring over all 322 cached candidates; candidate-cache construction is reported separately in `summary.json`.
