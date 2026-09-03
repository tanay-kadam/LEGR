# Campaign v4 notes: tool-name results, GNN input, SBERT-FT

## Question 1. Where are the tool-name run results? I only see the integer-ID ones.

**Answer.** They were trained and saved; they are not in the integer-ID CSVs (`legr_legacy_*`, `legr_directed_*`). Only **15-tool** tool-name models were trained. 30-tool and 45-tool tool-name runs do not exist.

Files:

- `artifacts/campaign_v4/results/legr_directed_toolname_15t_eval_75ep.csv`
- `artifacts/campaign_v4/results/legr_setgnn_tied_15t_eval.csv`

Protocol for the table below: unique DAGs in the 15-tool `test_topology_heldout.csv` (50 DAGs, unique toolsets, **no twins** in the ranking gallery). Frozen SBERT / BM25 in the same CSVs. SBERT-FT GED is from `sbert_ft_ged_15t_s42/eval_metrics.json`.

### 15-tool held-out unique-DAG gallery

| Model | Node features | R@1 | R@3 | R@5 | MRR@5 | Tool F1 | Mean GED err |
|-------|---------------|-----|-----|-----|-------|---------|--------------|
| Integer-ID directed + GED, 75ep | `nn.Embedding(id)` | 0.0800 | 0.1533 | 0.2267 | 0.1268 | 0.3779 | 4.25 |
| Integer-ID directed, no GED, 75ep | `nn.Embedding(id)` | 0.0533 | 0.1433 | 0.1833 | 0.0985 | 0.3388 | 4.48 |
| Frozen SBERT (same eval) | dag_text string | 0.4333 | 0.7033 | 0.7567 | 0.5681 | 0.7116 | 2.56 |
| BM25 | bag of words | 0.0200 | 0.0600 | 0.1000 | 0.0457 | 0.3351 | 4.82 |
| **LEGR toolname, no GED** | MiniLM(tool name) | **0.6067** | 0.8600 | 0.9200 | 0.7353 | 0.8138 | 1.75 |
| **LEGR toolname + GED** | MiniLM(tool name) | **0.6200** | 0.8300 | 0.9033 | 0.7309 | 0.8241 | 1.69 |
| **LEGR SetGNN tied, no GED** | tied MiniLM + set pool | **0.8267** | 0.9700 | 0.9800 | 0.8990 | 0.9464 | 0.71 |
| **LEGR SetGNN tied + GED** | tied MiniLM + set pool | **0.8133** | 0.9700 | 0.9800 | 0.8907 | 0.9402 | 0.76 |
| SBERT-FT GED | serialized dag_text | 0.9267 | 0.9933 | 0.9967 | 0.9596 | 0.9811 | 0.29 |
| SBERT-FT no GED | serialized dag_text | 0.9200 | 0.9933 | 0.9933 | 0.9561 | 0.9781 | 0.31 |

SetGNN is the other text-node variant: same MiniLM node names, plus a tool-set attention pool concatenated with the GNN readout.

### Fair gallery with twins (322 DAGs)

300 held-out queries ranked against test unique ∪ corpus unique. Chance same-toolset R@1 = 0.1538. Sources: `full_gallery_15t.json`, `sbert_text_only_proof_15t.json`.

| Model | Full R@1 | Full R@3 | Full R@5 | Same-toolset R@1 | Twin cosine |
|-------|----------|----------|----------|------------------|-------------|
| LEGR toolname + GED | 0.2000 | 0.4000 | 0.5033 | 0.2733 | 0.43 |
| LEGR SetGNN tied, no GED | 0.2100 | 0.5667 | 0.8133 | 0.2367 | 0.63 |
| SBERT-FT dag_text | 0.2133 | 0.6100 | 0.8533 | 0.2333 | 0.95 |
| SBERT-FT tools-only | 0.6467 | 0.6500 | 0.6567 | 0.6900* | 1.00 |

\*Tools-only same-toolset R@1 is an argmax-on-tie artifact. Random tie-break is 0.1467 (chance). Twin cosine 1.00 means SBERT tools-only cannot separate `A→B` from `B→A`.

Checkpoints (weights gitignored via `*.pt`):

- `legr_directed_toolname_no_ged_15t_s42`
- `legr_directed_toolname_ged_15t_s42`
- `legr_setgnn_tied_no_ged_15t_s42`
- `legr_setgnn_tied_ged_15t_s42`

---

## Question 2. How is the GNN input prepared? Is it MiniLM node names + adjacency + InfoNCE?

Expected form:

```
z_q = Normalize( MLP_q( MiniLM(query) ) )
z_g = Normalize( MLP_g( Pool( GNN(X, A) ) ) )
loss = InfoNCE( z_q · z_g_positive , z_q · z_g_negatives )
```

`X` = embeddings of node names from an S-BERT encoder; `A` = connections between nodes.

**Answer.** Yes. That is `LEGRDualEncoderV2` (`legr_directed_toolname_*`) in `src/encoders_v2.py`. The integer-ID runs (`legr_legacy_*`, `legr_directed_*`) are **not** that model: their `X` is `nn.Embedding(tool_id)`, randomly initialized, which is why they sat near chance (R@1 0.08).

### How `X` is built

1. CSV: `tools = read_user_profile;write_database_record`, `edges = 0->1`
2. NetworkX: node `i` has attribute `tool = "read_user_profile"`
3. `dag_to_pyg` stores **integer indices** in `Data.x` (a lookup table only) and directed `edge_index` (`bidirectional=False` for toolname)
4. At encode time those IDs are mapped back to names, then:

```
name  →  "read user profile"          # underscores → spaces
      →  frozen MiniLM (all-MiniLM-L6-v2)
      →  mean-pool last hidden         # 384-d  ← node_vec
      →  Linear(384 → 64)             # trainable, like a small MLP on X
```

So `X = [node1_vec, node2_vec, …]` is MiniLM of the node names. A 64-d learned topological-rank embedding is concatenated before the GNN; it is not part of MiniLM.

### How `A` is built

Not a dense adjacency matrix. PyG COO `edge_index`: column `k` is `(src, dst)` for a directed dependency. Isolated nodes still exist; they only get `W_self`.

### GNN, pool, query tower

3 directed layers:

```
h ← ReLU( LN( W_self h + W_in Σ_in + W_out Σ_out ) )
```

then `global_mean_pool`, then `Linear(128 → 256)` (`MLP_g`), then L2-normalize.

Query tower: MiniLM (first 4 layers frozen) → Linear 384→256 (`MLP_q`) → L2-normalize.

### Loss

In-batch bidirectional InfoNCE on `z_q @ z_g.T / τ` (batch 128). Negatives are the other graphs in the batch. GED runs add `λ = 0.10` times a term that down-weights easy (high-GED) negatives.

Two MiniLM copies in V2: query tower is partly trainable; node MiniLM is fully frozen and cached. SetGNN V3 ties them into one backbone and concatenates a node-set attention pool with the GNN readout.

---

## SBERT-FT: GED vs no GED

Campaign v4 fine-tuned SBERT dual encoder (`all-MiniLM-L6-v2`).
Protocol: rank unique DAGs in `test_topology_heldout.csv` (50 / 70 / 100 unique DAGs).
This gallery has unique toolsets and **no structural twins**.

Source files:

- `artifacts/campaign_v4/results/sbert_ft_no_ged_{15,30,45}t_s42/eval_metrics.json`
- `artifacts/campaign_v4/results/sbert_ft_ged_{15,30,45}t_s42/eval_metrics.json`

### No GED (`λ_ged = 0`)

| Tier | R@1 | R@3 | R@5 | MRR@1 | MRR@3 | MRR@5 | Tool F1 | Mean GED err | Hard-neg rank acc | Hard-neg FPR |
|------|-----|-----|-----|-------|-------|-------|---------|--------------|-------------------|--------------|
| 15 | 0.9200 | 0.9933 | 0.9933 | 0.9200 | 0.9561 | 0.9561 | 0.9781 | 0.310 | 1.0000 | 0.0000 |
| 30 | 0.9571 | 0.9929 | 0.9976 | 0.9571 | 0.9730 | 0.9742 | 0.9724 | 0.217 | 1.0000 | 0.0000 |
| 45 | 0.9467 | 0.9867 | 0.9933 | 0.9467 | 0.9658 | 0.9673 | 0.9671 | 0.243 | 1.0000 | 0.0000 |

Hard-neg pairs evaluated: 312 on every run.

### GED (`λ_ged > 0`)

| Tier | R@1 | R@3 | R@5 | MRR@1 | MRR@3 | MRR@5 | Tool F1 | Mean GED err | Hard-neg rank acc | Hard-neg FPR |
|------|-----|-----|-----|-------|-------|-------|---------|--------------|-------------------|--------------|
| 15 | 0.9267 | 0.9933 | 0.9967 | 0.9267 | 0.9589 | 0.9596 | 0.9811 | 0.290 | 0.9936 | 0.0064 |
| 30 | 0.9571 | 0.9929 | 0.9976 | 0.9571 | 0.9730 | 0.9742 | 0.9724 | 0.217 | 1.0000 | 0.0000 |
| 45 | 0.9467 | 0.9867 | 0.9933 | 0.9467 | 0.9658 | 0.9673 | 0.9671 | 0.243 | 1.0000 | 0.0000 |

Hard-neg pairs evaluated: 312 on every run.

### Side-by-side (R@1 / Tool F1)

| Tier | No GED R@1 | GED R@1 | No GED F1 | GED F1 |
|------|------------|---------|-----------|--------|
| 15 | 0.9200 | 0.9267 | 0.9781 | 0.9811 |
| 30 | 0.9571 | 0.9571 | 0.9724 | 0.9724 |
| 45 | 0.9467 | 0.9467 | 0.9671 | 0.9671 |

GED moves 15-tool R@1 by +0.0067. The 30-tool and 45-tool GED and no-GED `eval_metrics.json` files are identical.

The 322-DAG twin gallery (`full_gallery_15t.json`) was scored with the **15-tool GED** checkpoint, not a separate no-GED pass: SBERT-FT dag_text R@1 0.2133 / R@3 0.6100 / R@5 0.8533.
