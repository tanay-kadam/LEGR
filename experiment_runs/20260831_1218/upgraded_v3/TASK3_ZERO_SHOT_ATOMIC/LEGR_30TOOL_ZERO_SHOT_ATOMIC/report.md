# Zero-shot atomic LEGR

**Status:** COMPUTED against `C:\Users\tkadam\LEGR\experiment_runs\20260831_1218\upgraded_v3\TASK4_DIRECTION_ABLATION\LEGR_DEFAULT_GCN_30TOOL\best_model.pt`
**Classification:** OUTCOME B — ZERO-SHOT FAILURE

Do not insert a unified-framework claim automatically.

## Per-condition metrics

{
  "Standard": {
    "n": 1005,
    "correct": 917,
    "accuracy_pct": 91.2,
    "recall@1": 0.9124378109452737,
    "recall@3": 0.972139303482587,
    "recall@5": 0.9800995024875622,
    "mean_rank": 1.2845771144278606,
    "multi_node_steal_rate": 0.010945273631840797
  },
  "Lexical": {
    "n": 1005,
    "correct": 593,
    "accuracy_pct": 59.0,
    "recall@1": 0.5900497512437811,
    "recall@3": 0.8537313432835821,
    "recall@5": 0.9233830845771144,
    "mean_rank": 2.162189054726368,
    "multi_node_steal_rate": 0.003980099502487562
  },
  "Confusable": {
    "n": 450,
    "correct": 232,
    "accuracy_pct": 51.6,
    "recall@1": 0.5155555555555555,
    "recall@3": 0.8222222222222222,
    "recall@5": 0.9422222222222222,
    "mean_rank": 2.348888888888889,
    "multi_node_steal_rate": 0.04666666666666667
  },
  "Paraphrase": {
    "n": 1255,
    "correct": 1128,
    "accuracy_pct": 89.9,
    "recall@1": 0.8988047808764941,
    "recall@3": 0.9729083665338646,
    "recall@5": 0.9864541832669322,
    "mean_rank": 1.248605577689243,
    "multi_node_steal_rate": 0.00796812749003984
  }
}

## Aggregate

{
  "n": 3715,
  "correct": 2870,
  "accuracy_pct": 77.3
}

## One-node GNN notes

- Empty `edge_index` of shape (2, 0); no artificial edges.
- GCNConv may add self-loops internally; DirectedGraphEncoder uses W_self only.
- Frozen checkpoint embedding table is not expanded.
