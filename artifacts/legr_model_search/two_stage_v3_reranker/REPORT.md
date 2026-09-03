# Two-stage SBERT-FT → V3 Reranker

The 322-DAG gallery was inspected before this architecture was proposed; results are exploratory, not pristine confirmatory test evidence.

| Model | Seed | R@1 | R@3 | R@5 | Tool F1 | True-twin R@1 |
|---|---:|---:|---:|---:|---:|---:|
| SBERT-FT | fixed | 0.2133 | 0.6100 | 0.8533 | 0.9689 | 0.2333 |
| Two-stage | 42 | 0.2767 | 0.6833 | 0.8800 | 0.9689 | 0.2767 |
| Two-stage | 123 | 0.2633 | 0.6333 | 0.8767 | 0.9689 | 0.2633 |
| Two-stage | 2026 | 0.2867 | 0.6900 | 0.8800 | 0.9689 | 0.2867 |

## Paired bootstrap versus SBERT-FT

- recall@1: +0.0622, 95% CI [-0.0356, +0.1644]
- tool_set_f1: +0.0000, 95% CI [+0.0000, +0.0000]
- true_twin_recall@1: +0.0422, 95% CI [-0.0589, +0.1467]
