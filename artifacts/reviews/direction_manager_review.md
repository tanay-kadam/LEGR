# Manager review — Workstream D (directed GNN)

**Status: PASS** (code and tests). Ablation not trained.

## Scientific validity

Current LEGR is already undirected (`dag_to_pyg` bidirectional + `GCNConv`). The new `DirectedGraphEncoder` implements `W_self` / `W_in` / `W_out`. `tied_in_out` shares `W_in` and `W_out`. Default `graph_direction=gcn_undirected` leaves GCN checkpoints loadable.

Eval encodes with `bidirectional` taken from checkpoint config so directed models are not silently undirected at test time.

## Issues found and fixed

| Severity | Location | Issue | Fix |
|----------|----------|-------|-----|
| Blocker (design) | original GCN | No `W_in`/`W_out` to tie | Implemented DirGNN instead of a fake ablation |
| Major | `eval.encode_all_dags` | Always used bidirectional pyg | `bidirectional` argument from config |

**Verdict:** PASS for implementation. Directionality conclusion pending train+eval.
