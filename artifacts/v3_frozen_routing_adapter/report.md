# Frozen V3 + supervised routing adapter

## Experiment

A residual routing adapter was trained on top of the byte-identical Campaign V4
V3 no-GED 15-tool checkpoint. V3 MiniLM, set-branch, directed GNN, and fusion
weights received no gradients and were never rewritten. Only the adapter
(256→128→256 residual query MLP plus 15 residual tool prototypes) was optimized.

The training corpus contains 900 independent routing utterances (60 per tool)
and 150 held-out validation utterances (10 per tool). Exact normalized queries
were asserted to have zero overlap with Standard, Lexical, Confusable, and
Paraphrase evaluation files. Candidates remain one-node, zero-edge graphs.

Source checkpoint: `artifacts/campaign_v4/results/legr_setgnn_tied_no_ged_15t_s42/best_model.pt`
Source SHA256 before/after: `6027a4df479bd30137f97e6eb26278b05bab9adbb61ad4e57b5354bb8e7f727a` / `6027a4df479bd30137f97e6eb26278b05bab9adbb61ad4e57b5354bb8e7f727a`
Original model modified: `False`

## Accuracy mean ± std across seeds

| Stage | Standard | Lexical | Confusable | Paraphrase |
|---|---:|---:|---:|---:|
| Frozen V3 | 52.44 | 28.66 | 40.22 | 51.24 |
| Adapter | 75.12 ± 0.53 | 48.72 ± 0.75 | 61.56 ± 3.20 | 73.49 ± 0.80 |

## Interpretation

This measures supervised atomic-routing adaptation of a frozen Campaign-pretrained
encoder, not zero-shot graph transfer. Because every candidate has one node and
zero edges, the directed GNN remains inactive; the adapter learns residual
routing boundaries in the shared 256-d embedding space.
