# Verified paper tables (do not overwrite `table_stubs.md`)

Copied from `experiment_runs/20260831_1218/final_analysis/paper_ready_results.md`.
Fill the PDF from these cells only. Seed 42, CUDA, independently verified.

## Table 2 candidate rows (compositional retrieval, Dataset A `upgraded`)

| Model | Recall@1 | Recall@3 | Recall@5 | MRR@5 | Tool-Set F1 | Mean GED |
| --- | --- | --- | --- | --- | --- | --- |
| Sentence-BERT (frozen) | 0.5512 | 0.8042 | 0.8795 | 0.6833 | 0.7111 | 2.2470 |
| Sentence-BERT FT (untied, λ_GED=0) | 0.9217 | 1.0000 | 1.0000 | 0.9603 | 0.9721 | 0.2078 |
| Sentence-BERT FT (untied, λ_GED=0.30) | 0.9217 | 1.0000 | 1.0000 | 0.9603 | 0.9721 | 0.2078 |
| Sentence-BERT FT (tied, λ_GED=0) | 0.9458 | 0.9970 | 1.0000 | 0.9721 | 0.9810 | 0.1807 |
| LEGR (default GCN) | 0.8524 | 0.9759 | 0.9910 | 0.9086 | 0.9057 | 0.4910 |

There is no separate “LEGR (no GED)” trained row in this campaign.

## Table 1 candidate row (atomic, 15-tool encoder, unified 45-candidate corpus)

| Model | Standard | Lexical | Confusable | Paraphrase |
| --- | --- | --- | --- | --- |
| LEGR (zero-shot, unified corpus) | 84.0 | 50.7 | 45.1 | 81.4 |

Outcome B: do not add unification language (Lexical and Confusable < 70%).

## Ablation candidate

| Variant | Recall@1 | Tool-Set F1 | Mean GED |
| --- | --- | --- | --- |
| LEGR GCN (published, undirected edges) | 0.8524 | 0.9057 | 0.4910 |
| DirGNN directed | 0.6747 | 0.8080 | 1.0693 |
| DirGNN tied W_in=W_out | 0.7892 | 0.8989 | 0.5964 |

Do not claim the GCN vs DirGNN gap is caused by directionality alone.
