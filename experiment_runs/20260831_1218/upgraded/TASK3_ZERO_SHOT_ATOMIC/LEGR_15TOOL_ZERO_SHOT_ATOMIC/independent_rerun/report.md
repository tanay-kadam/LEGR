# Zero-shot atomic LEGR

**Status:** COMPUTED against `C:\Users\tkadam\LEGR\experiment_runs\20260831_1218\upgraded\TASK4_DIRECTION_ABLATION\LEGR_DEFAULT_GCN_15TOOL\best_model.pt`
**Classification:** OUTCOME B — ZERO-SHOT FAILURE

Do not insert a unified-framework claim automatically.

## Per-condition metrics

{
  "Standard": {
    "n": 1005,
    "correct": 844,
    "accuracy_pct": 84.0,
    "recall@1": 0.8398009950248756,
    "recall@3": 0.945273631840796,
    "recall@5": 0.9791044776119403,
    "mean_rank": 1.3920398009950248,
    "multi_node_steal_rate": 0.11741293532338308
  },
  "Lexical": {
    "n": 1005,
    "correct": 510,
    "accuracy_pct": 50.7,
    "recall@1": 0.5074626865671642,
    "recall@3": 0.7472636815920398,
    "recall@5": 0.8616915422885573,
    "mean_rank": 2.6965174129353233,
    "multi_node_steal_rate": 0.2099502487562189
  },
  "Confusable": {
    "n": 450,
    "correct": 203,
    "accuracy_pct": 45.1,
    "recall@1": 0.45111111111111113,
    "recall@3": 0.7644444444444445,
    "recall@5": 0.9244444444444444,
    "mean_rank": 2.631111111111111,
    "multi_node_steal_rate": 0.16444444444444445
  },
  "Paraphrase": {
    "n": 1255,
    "correct": 1022,
    "accuracy_pct": 81.4,
    "recall@1": 0.8143426294820717,
    "recall@3": 0.9378486055776892,
    "recall@5": 0.9721115537848606,
    "mean_rank": 1.4478087649402391,
    "multi_node_steal_rate": 0.14661354581673305
  }
}

## Aggregate

{
  "n": 3715,
  "correct": 2579,
  "accuracy_pct": 69.4
}

## One-node GNN notes

- Empty `edge_index` of shape (2, 0); no artificial edges.
- GCNConv may add self-loops internally; DirectedGraphEncoder uses W_self only.
- Frozen checkpoint embedding table is not expanded.
