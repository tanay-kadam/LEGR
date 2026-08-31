# Paper-ready results (independently verified only)

Seed 42. Device `cuda:0`. Git `fae6f498512f442218366b8fb264ff35c3834f1c`.
Run root: `experiment_runs/20260831_1218`.

Exact floats: `all_metrics.json`. Retrieval tables use 4 decimals; atomic accuracies use 1 decimal.

---

## Table 2 — compositional retrieval, Dataset A `upgraded` 30-tool (n=332)

| Model | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Tool-Set F1 | Mean GED | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sentence-BERT (frozen, eval.py) | 0.5512 | 0.8042 | 0.8795 | 0.6833 | 0.7111 | 2.2470 | VERIFIED |
| BM25 | 0.0120 | 0.0361 | 0.0602 | 0.0275 | 0.0873 | 6.3976 | VERIFIED |
| Sentence-BERT FT (untied, λ_GED=0) | 0.9217 | 1.0000 | 1.0000 | 0.9603 | 0.9721 | 0.2078 | VERIFIED |
| Sentence-BERT FT (untied, λ_GED=0.30) | 0.9217 | 1.0000 | 1.0000 | 0.9603 | 0.9721 | 0.2078 | VERIFIED |
| Sentence-BERT FT (tied, λ_GED=0) | 0.9458 | 0.9970 | 1.0000 | 0.9721 | 0.9810 | 0.1807 | VERIFIED |
| LEGR GCN (published default) | 0.8524 | 0.9759 | 0.9910 | 0.9086 | 0.9057 | 0.4910 | VERIFIED |
| DirGNN directed | 0.6747 | 0.8825 | 0.9639 | 0.7848 | 0.8080 | 1.0693 | VERIFIED |
| DirGNN tied W_in=W_out | 0.7892 | 0.9849 | 0.9970 | 0.8787 | 0.8989 | 0.5964 | VERIFIED |

GED0 and GED030: identical test metrics, different SHA256.

Dataset B Table 2 is in `FINAL_REPORT.md` §4 / `experiment_comparison.csv`. Main paper should use Dataset A unless the hard-negative and n_test differences are stated.

---

## Table 1 — zero-shot atomic (routing_15tools queries)

| Encoder | Corpus | Standard | Lexical | Confusable | Paraphrase | Status |
| --- | --- | --- | --- | --- | --- | --- |
| LEGR 15-tool GCN (Dataset A) | 45 | 84.0 | 50.7 | 45.1 | 81.4 | VERIFIED — **main Table 1 row** |
| LEGR 15-tool GCN (Dataset B) | 53 | 92.8 | 62.1 | 59.1 | 93.0 | VERIFIED — appendix |
| LEGR 30-tool GCN (Dataset A) | 45 | 63.8 | 31.5 | 36.4 | 65.3 | VERIFIED — ablation |
| LEGR 30-tool GCN (Dataset B) | 69 | 91.2 | 59.0 | 51.6 | 89.9 | VERIFIED — ablation |

Protocol: 15 one-node LEGR tools + unique compositional DAGs from the matching graph test CSV. Queries always `upgraded_data/routing_15tools`. **Not** `routing_30tools` (those labels are OOV for the DAG encoder).

Unification (≥70% on all four conditions): **not met**. Outcome B. Do not add unification language.

---

## Ablation — DirGNN directed vs tied-in/out (Dataset A)

| Variant | Recall@1 | Tool-Set F1 | Mean GED | Status |
| --- | --- | --- | --- | --- |
| DIRGNN_DIRECTED | 0.6747 | 0.8080 | 1.0693 | VERIFIED |
| DIRGNN_TIED_IN_OUT | 0.7892 | 0.8989 | 0.5964 | VERIFIED |
| LEGR GCN (context; different layer) | 0.8524 | 0.9057 | 0.4910 | VERIFIED |

`DIRGNN_TIED_IN_OUT` ≠ `LEGR_DEFAULT_GCN`.

---

## Action-type latent space

Both datasets: `REAL_CHECKPOINT_EMBEDDINGS`, evidence **NO SUPPORT**. **Omit figure.**
