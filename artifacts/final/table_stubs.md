# Table stubs (empty — do not typeset as results)

## Table 2 candidate rows (compositional retrieval)

Fill only from `eval_metrics.json` / `src/eval.py` CSVs after a real run.

| Model | Recall@1 | Recall@3 | Recall@5 | MRR | Tool-Set F1 | Mean GED |
| --- | --- | --- | --- | --- | --- | --- |
| Sentence-BERT (frozen) |  |  |  |  |  |  |
| Sentence-BERT FT (untied, λ_GED=0) |  |  |  |  |  |  |
| Sentence-BERT FT (untied, λ_GED=0.30) |  |  |  |  |  |  |
| Sentence-BERT FT (tied, λ_GED=0) |  |  |  |  |  |  |
| LEGR (no GED) |  |  |  |  |  |  |
| LEGR |  |  |  |  |  |  |

## Table 1 candidate row (atomic, 15-tool, unified corpus)

| Model | Standard | Lexical | Confusable | Paraphrase |
| --- | --- | --- | --- | --- |
| LEGR (zero-shot, unified corpus) |  |  |  |  |

Do not add unification language until OUTCOME A and manager review of the measured margins.

## Ablation candidate

| Variant | Recall@1 | Tool-Set F1 | Mean GED |
| --- | --- | --- | --- |
| LEGR GCN (published, undirected edges) |  |  |  |
| DirGNN directed |  |  |  |
| DirGNN tied W_in=W_out |  |  |  |
