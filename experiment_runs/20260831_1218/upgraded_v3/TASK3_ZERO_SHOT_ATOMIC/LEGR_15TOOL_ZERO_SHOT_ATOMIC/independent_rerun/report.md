# Zero-shot atomic LEGR

**Status:** COMPUTED against `C:\Users\tkadam\LEGR\experiment_runs\20260831_1218\upgraded_v3\TASK4_DIRECTION_ABLATION\LEGR_DEFAULT_GCN_15TOOL\best_model.pt`
**Classification:** OUTCOME B — ZERO-SHOT FAILURE

Do not insert a unified-framework claim automatically.

## Per-condition metrics

{
  "Standard": {
    "n": 1005,
    "correct": 933,
    "accuracy_pct": 92.8,
    "recall@1": 0.9283582089552239,
    "recall@3": 0.9830845771144279,
    "recall@5": 0.9920398009950249,
    "mean_rank": 1.1492537313432836,
    "multi_node_steal_rate": 0.0029850746268656717
  },
  "Lexical": {
    "n": 1005,
    "correct": 624,
    "accuracy_pct": 62.1,
    "recall@1": 0.6208955223880597,
    "recall@3": 0.8557213930348259,
    "recall@5": 0.9711442786069652,
    "mean_rank": 1.8945273631840795,
    "multi_node_steal_rate": 0.005970149253731343
  },
  "Confusable": {
    "n": 450,
    "correct": 266,
    "accuracy_pct": 59.1,
    "recall@1": 0.5911111111111111,
    "recall@3": 0.9022222222222223,
    "recall@5": 0.9533333333333334,
    "mean_rank": 1.9644444444444444,
    "multi_node_steal_rate": 0.028888888888888888
  },
  "Paraphrase": {
    "n": 1255,
    "correct": 1167,
    "accuracy_pct": 93.0,
    "recall@1": 0.9298804780876494,
    "recall@3": 0.9840637450199203,
    "recall@5": 0.9888446215139443,
    "mean_rank": 1.1474103585657371,
    "multi_node_steal_rate": 0.002390438247011952
  }
}

## Aggregate

{
  "n": 3715,
  "correct": 2990,
  "accuracy_pct": 80.5
}

## One-node GNN notes

- Empty `edge_index` of shape (2, 0); no artificial edges.
- GCNConv may add self-loops internally; DirectedGraphEncoder uses W_self only.
- Frozen checkpoint embedding table is not expanded.
