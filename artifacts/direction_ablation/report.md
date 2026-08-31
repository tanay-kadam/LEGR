# Directed vs undirected GNN

**Status:** IMPLEMENTED, NOT TRAINED.

## Code fact

Published LEGR is already undirected: `dag_to_pyg` bidirectionalizes edges and `GCNConv` is symmetric. Order is `topo_pos`. There were no `W_in`/`W_out` to tie.

## What was added

- `DirectedGraphEncoder` in `src/encoders.py`
- `TrainConfig.graph_direction`: `gcn_undirected` (default) | `directed` | `tied_in_out`
- `dag_to_pyg(..., bidirectional=True)` default unchanged so existing GCN checkpoints still load

## Controlled ablation (when trained)

1. `graph_direction=directed` — independent `W_in`, `W_out`, original directed edges
2. `graph_direction=tied_in_out` — `W_in is W_out`, still directed neighborhoods
3. Secondary reference: existing GCN checkpoint (confounded by GCNConv vs DirGNN)

## Conclusion (pending numbers)

DIRECTIONALITY IMPORTANT / MODESTLY HELPFUL / NO MEANINGFUL DIFFERENCE / UNDIRECTED BETTER — not assigned.
