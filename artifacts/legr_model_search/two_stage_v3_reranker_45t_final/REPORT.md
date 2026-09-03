# Two-stage SBERT-FT → V3 Reranker (45 tools)

The two-stage architecture was fixed on the 15-tool study before this tier's held-out scaling evaluation; no test-time checkpoint selection was performed.

| Model | Seed | R@1 | R@3 | R@5 | Tool F1 | True-twin R@1 |
|---|---:|---:|---:|---:|---:|---:|
| SBERT-FT | fixed | 0.1700 | 0.6500 | 0.8883 | 0.9747 | 0.1767 |
| LEGR-V3 | fixed | 0.4183 | 0.8167 | 0.9400 | 0.9711 | 0.4333 |
| Two-stage | 42 | 0.3867 | 0.8267 | 0.9483 | 0.9747 | 0.3867 |
| Two-stage | 123 | 0.3217 | 0.8033 | 0.9450 | 0.9747 | 0.3217 |
| Two-stage | 2026 | 0.3750 | 0.8200 | 0.9467 | 0.9747 | 0.3750 |

## Paired bootstrap versus SBERT-FT

- recall@1: +0.1911, 95% CI [+0.1244, +0.2589]
- tool_set_f1: +0.0000, 95% CI [+0.0000, +0.0000]
- true_twin_recall@1: +0.1828, 95% CI [+0.1183, +0.2483]
