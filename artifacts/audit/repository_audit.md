# LEGR repository audit — additional experiments

Date: 2026-08-31  
Scope: code-first workstreams A–D. No experimental numbers in this document.

## Repository layout (relevant)

| Area | Path | Role |
|------|------|------|
| Taxonomy routing (Table 1) | `src/main.py`, `src/evaluator.py`, `src/routers.py`, `src/taxonomies.py` | Atomic-query LLM router |
| LEGR train | `src/train.py` (`TrainConfig`, `main`, `CSVTrainDataset`, `make_collate_fn`) | Dual-encoder training |
| LEGR eval | `src/eval.py` (`CSVEvalDataset`, `compute_metrics`, `_sbert_baseline`, `encode_all_dags`) | Retrieval metrics + frozen SBERT |
| Encoders | `src/encoders.py` (`TextEncoder`, `GCNGraphEncoder`, `GATGraphEncoder`, `LEGRDualEncoder`) | Towers |
| Loss | `src/loss.py` (`GraphAwareContrastiveLoss`) | Bidirectional InfoNCE + GED term |
| DAG I/O | `src/data_synth.py` (`dag_to_pyg`, `dag_to_text`, `build_dag`, `TOOL_VOCAB`) | Graphs and strings |
| Topology labels | `src/utils/graph_utils.py` (`classify_topology`) | Family names for viz |
| Tool-count bootstrap | `src/legr_tool_count.py`, `src/vocab_config.py` | 15 / 30 / 45 |
| Routing vocab | `src/routing_tiers.py` (`EXPLICIT_ROUTING_TOOL_NAMES_15`) | Table 1 tool names |
| Paper LEGR splits | `upgraded/upgraded_{15,30,45}tools/` | train / dev / test_topology_heldout |
| Routing stress CSVs | `upgraded_data/routing_{15,30,45}tools/` | Standard / Lexical / Confusable / Paraphrase |
| Tests | `tests/test_*.py` | Structural regression; no LLM |

`paper.tex` is gitignored and is not in the working tree. Checkpoints (`*.pt`) are gitignored; none are present. There is no existing t-SNE/UMAP script.

## Training / evaluation entry points

- Train: `python src/train.py --tool_count 30 --checkpoint_dir checkpoints_30tools`
- Eval: `python src/eval.py --tool_count 30 --checkpoint … --dataset_csv upgraded/upgraded_30tools/test_topology_heldout.csv`
- Frozen SBERT: `eval._sbert_baseline` encodes queries and `dag_to_text` strings with `sentence-transformers/all-MiniLM-L6-v2`
- Table 2 metrics: `eval.compute_metrics` → `recall@{1,3,5}`, `mrr@{1,3,5}`, `tool_set_f1`, `mean_ged_error`
- Table 1 metrics: `evaluator._compute_aggregate_metrics` → `accuracy_pct` (1 decimal)

## Checkpoint payload (`train._build_checkpoint_payload`)

Keys: `epoch`, `model_state`, `criterion_state`, `config` (`vars(TrainConfig)`), `tool_count`, optional optimizer/scheduler/`val_loss`.

`eval._load_model_and_tokenizer` rebuilds `LEGRDualEncoder` from `config` and `load_state_dict(..., strict=True)`.

## Default TrainConfig (must be reused, not retyped)

Copied from `src/train.py` `TrainConfig`: text `sentence-transformers/all-MiniLM-L6-v2`, `embed_dim=256`, `gcn_layers=3`, `node_embed_dim=64`, `gcn_hidden=256`, `num_frozen_layers=4`, `max_topo_pos=16`, `temperature_init=0.05`, `lambda_ged=0.30`, `ged_scale=2.5`, `ged_margin=0.05`, `lr=2e-4`, `text_backbone_lr=2e-5`, `weight_decay=1e-4`, `max_grad_norm=1.0`, `epochs=100`, `warmup_epochs=3`, `batch_size=128`, `max_length=128`, `patience=15`, `seed=42`. Train loader uses `drop_last=True`.

## Workstream reuse / modify map

### A — Fine-tuned Sentence-BERT 2×2

Reuse: `TextEncoder`, `GraphAwareContrastiveLoss`, `TrainConfig`, `_build_csv_train_val_datasets`, `_ged_submatrix`, `dag_to_text`, `CSVEvalDataset`, `compute_metrics`.

New: `src/sbert_ft_baseline.py`, `artifacts/sbert_finetuned/`.

Do not retrain Cells 3–4 (LEGR no-GED / full LEGR).

### B — Action-type latent space

Reuse: `eval._load_model_and_tokenizer`, `encode_all_dags`, `CSVEvalDataset`, `classify_topology`, LEGR `TOOL_VOCAB`.

New: `src/action_type_mapping.py`, `scripts/analyze_action_latent_space.py`, `artifacts/action_latent_space/`.

Taxonomy branches in `taxonomies.py` use routing names (`query_database`); mapping must use LEGR names (`db_read`).

### C — Zero-shot atomic

Reuse: frozen `LEGRDualEncoder.encode_text` / `encode_graph`, `dag_to_pyg`, `dag_canonical_hash`, routing CSVs, `compute_metrics` analogue.

New: `src/atomic_zero_shot.py`, `scripts/eval_zero_shot_atomic.py`, `artifacts/zero_shot_atomic/`.

15-tool alias only: `query_database→db_read`, `update_database→db_write`. Do not `register_tools` OOV names into a frozen embedding table.

### D — Directed GNN

Reuse: training loop, loss, splits, text tower, dims.

Modify (additive): `dag_to_pyg(..., bidirectional=True)` default; `DirectedGraphEncoder`; `TrainConfig.graph_direction`; eval encoding must honor directed `edge_index`.

Default `graph_encoder_type="gcn"` and bidirectional edges stay unchanged so existing checkpoints still load.

## Critical findings (block paper claims until measured)

1. **Directed GNN was never implemented.** `dag_to_pyg` mirrors every edge; `GCNConv` is symmetric. Order is `topo_pos` only. Ablating `W_in=W_out` on the current model is invalid.

2. **Routing vocab ≠ LEGR vocab.** 15-tool: 13/15 identical; two aliases above. 30-tool: ~3/30 name overlap — out of scope.

3. **No checkpoints in tree.** B/C (and A/D eval vs LEGR) cannot produce paper numbers until `best_model.pt` is supplied.

4. **30-tool labelled-DAG leakage exists** in `upgraded/` (documented by `scripts/audit_split_leakage.py`). Do not silently change splits.

5. **No t-SNE convention in repo.** New script; cluster metrics in original embedding space.

6. **`paper.tex` absent.** Integration = artifact stubs only.
