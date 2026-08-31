# QA report — Workstream D (direction)

**Status: PASS**

## Independent tests

`tests/test_directed_encoder.py`: default settings remain bidirectional GCN; directed/tied config mapping; tied `W_in is W_out`; reverse edges change DirGNN embeddings and do **not** change bidirectional GCN (topo zeroed).

`tests/test_one_node_gnn.py`: directed isolated-node `W_self` path.

## Regression

Existing GCN default path still used by `TrainConfig.graph_direction=gcn_undirected`. 138 passed.

## Metrics

No trained ablation numbers. Do not fill the ablation table.
