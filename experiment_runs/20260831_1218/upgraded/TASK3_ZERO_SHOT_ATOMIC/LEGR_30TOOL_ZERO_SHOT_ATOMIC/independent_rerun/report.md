# Zero-shot atomic LEGR

**Status:** COMPUTED against `C:\Users\tkadam\LEGR\experiment_runs\20260831_1218\upgraded\TASK4_DIRECTION_ABLATION\LEGR_DEFAULT_GCN_30TOOL\best_model.pt`
**Classification:** OUTCOME B — ZERO-SHOT FAILURE

Do not insert a unified-framework claim automatically.

## Per-condition metrics

{
  "Standard": {
    "n": 1005,
    "correct": 641,
    "accuracy_pct": 63.8,
    "recall@1": 0.6378109452736318,
    "recall@3": 0.8139303482587065,
    "recall@5": 0.8427860696517413,
    "mean_rank": 3.000995024875622,
    "multi_node_steal_rate": 0.29850746268656714
  },
  "Lexical": {
    "n": 1005,
    "correct": 317,
    "accuracy_pct": 31.5,
    "recall@1": 0.3154228855721393,
    "recall@3": 0.5651741293532339,
    "recall@5": 0.6985074626865672,
    "mean_rank": 4.939303482587064,
    "multi_node_steal_rate": 0.4696517412935323
  },
  "Confusable": {
    "n": 450,
    "correct": 164,
    "accuracy_pct": 36.4,
    "recall@1": 0.36444444444444446,
    "recall@3": 0.6422222222222222,
    "recall@5": 0.74,
    "mean_rank": 4.1066666666666665,
    "multi_node_steal_rate": 0.44
  },
  "Paraphrase": {
    "n": 1255,
    "correct": 820,
    "accuracy_pct": 65.3,
    "recall@1": 0.6533864541832669,
    "recall@3": 0.8199203187250996,
    "recall@5": 0.8661354581673307,
    "mean_rank": 2.594422310756972,
    "multi_node_steal_rate": 0.2597609561752988
  }
}

## Aggregate

{
  "n": 3715,
  "correct": 1942,
  "accuracy_pct": 52.3
}

## One-node GNN notes

- Empty `edge_index` of shape (2, 0); no artificial edges.
- GCNConv may add self-loops internally; DirectedGraphEncoder uses W_self only.
- Frozen checkpoint embedding table is not expanded.
