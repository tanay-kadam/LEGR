# Code / config diff (direction ablation)

## Default path (must stay bit-compatible with old checkpoints)

- `TrainConfig.graph_direction = "gcn_undirected"`
- `dag_to_pyg(G)` still bidirectional
- `LEGRDualEncoder(graph_encoder_type="gcn")` still `GCNGraphEncoder`

## Additive

- `encoders.DirectedGraphEncoder`
- `encoders.resolve_graph_encoder_settings`
- `dag_to_pyg(..., bidirectional=False)` for directed modes
- `eval.encode_all_dags(..., bidirectional=...)` reads the checkpoint config
