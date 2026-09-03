# Two-stage SBERT-FT → V3 Reranker (30 tools)

The two-stage architecture was fixed on the 15-tool study before this tier's held-out scaling evaluation; no test-time checkpoint selection was performed.

| Model | Seed | R@1 | R@3 | R@5 | Tool F1 | True-twin R@1 |
|---|---:|---:|---:|---:|---:|---:|
| SBERT-FT | fixed | 0.1762 | 0.6000 | 0.9119 | 0.9747 | 0.1762 |
| Two-stage | 42 | 0.3095 | 0.8286 | 0.9548 | 0.9747 | 0.3095 |
| Two-stage | 123 | 0.3262 | 0.8119 | 0.9500 | 0.9747 | 0.3262 |
| Two-stage | 2026 | 0.3381 | 0.8190 | 0.9524 | 0.9747 | 0.3381 |

## Paired bootstrap versus SBERT-FT

- recall@1: +0.1484, 95% CI [+0.0563, +0.2405]
- tool_set_f1: +0.0000, 95% CI [+0.0000, +0.0000]
- true_twin_recall@1: +0.1472, 95% CI [+0.0536, +0.2377]
