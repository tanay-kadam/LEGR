# LEGR Campaign v4 — Architecture Audit

**Date:** 2026-08-31
**Role:** RESEARCH ARCHITECT
**Status:** PHASE 0 COMPLETE — All 15 audit questions answered

---

## 1. Repository Dependency Map

```
src/
├── main.py                  — CLI entry point for taxonomy routing experiments (Semantic vs Tool-Bound)
├── train.py                 — LEGR dual-encoder training loop (GCN/GAT/Directed + GACL)
├── eval.py                  — LEGR evaluation: retrieval metrics, SBERT baseline, BM25 baseline, hard-neg eval
├── sbert_ft_baseline.py     — Fine-tuned Sentence-BERT 2-tower baseline (query tower + doc tower)
├── encoders.py              — TextEncoder, GCNGraphEncoder, GATGraphEncoder, DirectedGraphEncoder, LEGRDualEncoder
├── loss.py                  — GraphAwareContrastiveLoss (InfoNCE + GED weighting)
├── data_synth.py            — Tool vocabulary (45 tools), workflow templates, DAG→PyG, GED matrix, dataset builder
├── dag_extract.py           — LLM-based DAG extraction with cycle repair
├── llm_dag_baseline.py      — LLM generative DAG baseline evaluation
├── llm_backends.py          — AzureOpenAIBackend, OllamaBackend, Gemini, provider factory
├── routers.py               — Two-step hierarchical LLM routing (branch → tool)
├── evaluator.py             — Routing experiment runner (semantic vs tool-bound taxonomy comparison)
├── taxonomies.py            — Semantic/Tool-Bound taxonomy definitions (15/30/45 tool tiers)
├── vocab_config.py          — ACTIVE_TOOL_COUNT global (15, 30, or 45)
├── legr_tool_count.py       — CLI helpers for --tool_count override
├── dataset.py               — Single-tool dataset builder
├── utils/
│   └── graph_utils.py       — DAG parsing, topology classification, topology generators, hard-negative generators
│
scripts/
├── run_routing_experiments.py      — Multi-model/multi-tool routing experiment runner
├── probe_baselines_on_failures.py  — Baseline failure analysis
├── eval_zero_shot_atomic.py        — Zero-shot atomic routing evaluation
├── prepare_legr_30tool_dataset.py  — 30-tool dataset preparation

configs/
├── llm_providers.json       — Azure OpenAI + Ollama provider profiles
├── pipeline_config.json     — Dataset generation config (graph families, split ratios)

upgraded/
├── upgraded_15tools/        — 15-tool train/dev/test CSVs
├── upgraded_30tools/        — 30-tool train/dev/test CSVs + hard_negatives
├── upgraded_45tools/        — 45-tool train/dev/test CSVs
```

---

## 2. Audit Question Answers

### Q1: What are LEGR node features currently?

**ANSWER: INTEGER TOOL IDs — METHODOLOGICAL DEFECT**

In `encoders.py` (line 150):
```python
self.tool_embedding = nn.Embedding(num_tools + 1, tool_embed_dim, padding_idx=0)
```

In `data_synth.py` (line 1589-1590):
```python
tool_indices = [TOOL_TO_IDX[G.nodes[n]["tool"]] for n in nodes]
x = torch.tensor(tool_indices, dtype=torch.long).unsqueeze(-1)
```

Node features are **arbitrary integer indices** into a learned `nn.Embedding` table. The integer 0 maps to `"db_read"`, 1 to `"db_write"`, etc. These integers carry no semantic information from the tool name itself.

**DEFECT D-001:** The graph encoder does not use textual tool-name embeddings. The tool name's semantic content (e.g., "read_user_profile" vs "edit_username") is encoded only by a randomly-initialized embedding row, not by the text encoder. This means the graph tower starts with random feature initialization for each tool, while the text tower has MiniLM pretrained knowledge about tool name semantics.

### Q2: Are they textual tool-name embeddings or arbitrary IDs?

**Arbitrary IDs.** See Q1. The `GCNGraphEncoder`, `GATGraphEncoder`, and `DirectedGraphEncoder` all use:
```python
tool_emb = self.tool_embedding(x.squeeze(-1))
```
where `x` contains integer tool indices.

There is a `use_text_node_features` flag in the encoder constructors, but:
- It defaults to `False`
- When enabled, it adds a `text_proj` linear layer but fills the input with **zeros** at forward time:
  ```python
  text_feat = torch.zeros(h.size(0), self.text_proj.in_features, device=h.device)
  ```
- No code path ever provides actual text embeddings as node features

**DEFECT D-002:** The `use_text_node_features` path is a dead code stub that always feeds zeros.

### Q3: Is edge direction preserved?

**DEFAULT: NO — DEFECT**

In `data_synth.py` `dag_to_pyg()` (lines 1578-1605):
```python
def dag_to_pyg(G: nx.DiGraph, bidirectional: bool = True) -> Data:
    ...
    for u, v in G.edges():
        src.append(u)
        dst.append(v)
        if bidirectional:
            src.append(v)
            dst.append(u)
```

The **default** is `bidirectional=True`, which adds reverse edges for every forward edge. This makes the graph effectively undirected for GCN message passing.

In `train.py` `resolve_graph_encoder_settings()`, the default `graph_direction` is `"gcn_undirected"`, which returns `bidirectional=True`.

**DEFECT D-003:** Default LEGR training uses bidirectional (undirected) edges. Topology information (which node depends on which) is partially destroyed.

### Q4: Is the GNN directed?

**DEFAULT: NO.** The default graph encoder is `GCNConv` (undirected GCN). A `DirectedGraphEncoder` exists with separate `W_self`, `W_in`, `W_out` transforms:
```python
h_new = w_self(h)
in_agg.index_add_(0, dst, w_in(h)[src])   # incoming neighbor messages
out_agg.index_add_(0, src, w_out(h)[dst])  # outgoing neighbor messages
h_new = h_new + in_agg + out_agg
```

This is the correct directed architecture, but:
- It must be explicitly selected via `--graph_direction directed`
- Previous results show `DIRGNN_DIRECTED` R@1=0.6747 (upgraded), which is **significantly lower** than default GCN (R@1=0.8524)
- The `DIRGNN_TIED_IN_OUT` variant (shared W_in/W_out) reached R@1=0.7892

**FINDING:** The DirectedGraphEncoder exists and is correct, but it underperforms. This could be because:
1. It was trained with integer-ID features (not text-name embeddings) — the directed model has less information to compensate with
2. It was trained without the bidirectional edge hack that helps GCN

### Q5: What pooling is used?

**Global mean pooling** (`global_mean_pool` from PyG):
```python
graph_emb = global_mean_pool(h, batch)  # (num_graphs, hidden)
return self.proj(graph_emb)             # (num_graphs, embed_dim)
```
This is applied in all three encoder variants (GCN, GAT, Directed).

### Q6: What text encoder is used?

**`sentence-transformers/all-MiniLM-L6-v2`** (384-dim hidden, 6 layers, 22M params):
```python
class TextEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", embed_dim=256):
        self.backbone = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(hidden, embed_dim)
```
Output: mean pooling → linear projection → L2 normalization → 256-dim embedding.

### Q7: Which MiniLM layers are frozen?

Default: `num_frozen_layers=4` (of 6 total layers).

```python
_freeze_transformer_backbone_layers(self.backbone, num_frozen_layers)
```
This freezes the first 4 transformer encoder layers, leaving layers 4-5 (0-indexed) trainable along with the projection head. The embedding layer remains trainable.

### Q8: How is GED used?

GED is used in the `GraphAwareContrastiveLoss` as an auxiliary weighting term:

```python
loss = InfoNCE(z_text, z_graph)  # standard symmetric cross-entropy

if lambda_ged > 0:
    ged_weights = clamp(ged_sub / ged_scale - ged_margin, min=0) / max_ged
    inv_ged = 1.0 - ged_weights
    weighted_neg_sim = sim * neg_mask * inv_ged
    # Structurally-similar negatives get HIGHER repulsion
    ged_loss = -log(exp(pos) / (exp(pos) + weighted_neg_sim.sum()))
    loss += lambda_ged * ged_loss
```

Defaults: `lambda_ged=0.30`, `ged_scale=2.5`, `ged_margin=0.05`.

GED is **pre-computed** over all unique DAGs (exact `nx.graph_edit_distance` for small DAGs, fast surrogate for CSV-backed training).

### Q9: How are negatives selected?

**In-batch negatives** only. The contrastive loss uses all non-diagonal elements in the (B, B) similarity matrix as negatives. There is no explicit hard-negative mining during training.

Hard negatives are generated separately in `graph_utils.py` for evaluation only:
- `hard_neg_swap_edges`: same tools, different edges
- `hard_neg_swap_tools`: same topology, different tools
- `hard_neg_remove_edge`: missing dependency
- `hard_neg_add_edge`: extra dependency
- `hard_neg_extra_node`: distractor tool

### Q10: What exact code implements fine-tuned SBERT?

`src/sbert_ft_baseline.py` — `SBERTFineTuneDualEncoder`:
- **Query tower:** `TextEncoder(all-MiniLM-L6-v2)` → 256-dim embedding
- **Document tower:** `TextEncoder(all-MiniLM-L6-v2)` → 256-dim embedding (separate or tied)
- **Document input:** `dag_to_text(G)` — canonical text like `"db_read -> process_refund, db_read -> create_ticket"`
- **Loss:** Same `GraphAwareContrastiveLoss` as LEGR
- **Variants:** untied (lambda_ged=0), untied (lambda_ged=0.30), tied weights

### Q11: Are SBERT and LEGR evaluated against identical candidate corpora?

**YES, when using `eval.py`.** Both `evaluate()` and `evaluate_ablation_two()` load the same `CSVEvalDataset` and compute metrics against the same set of unique DAGs. The `_sbert_baseline()` function uses the same dataset's `get_dag_text()` for document encoding.

However: the fine-tuned SBERT in `sbert_ft_baseline.py` has its own evaluation path that uses `evaluate_sbert_ft()`, which also computes against the same unique DAGs.

### Q12: Are retrieval metrics computed identically?

**YES.** Both use `compute_metrics()` from `eval.py`:
- Recall@{1,3,5}
- MRR@{1,3,5}
- Tool-set F1 (top-1 prediction)
- Mean GED error (top-1 prediction)

The fine-tuned SBERT evaluation in `sbert_ft_baseline.py` calls the same `compute_metrics()`.

### Q13: Are query/DAG splits actually leakage-free?

**PARTIALLY.** For the built-in synthetic dataset (`data_synth.py`):
- `split_mode="template"` splits by unique DAG IDs, so the same DAG structure doesn't appear in train and test
- BUT: the same tool combinations can appear across splits (tools are shared vocabulary)

For CSV-backed datasets (`upgraded/`):
- The `pipeline_config.json` specifies `train_topology_families` and `test_topology_families`
- Train families: `chain_short`, `chain_medium`, `single_node`, `fanout`, `fanin`
- Test families: `diamond`, `fork_join`, `complex_mixed`, `complex_deep`, `wide_fanout`, `wide_fanin`, `parallel_paths`
- **This is a topology-family-level OOD split** — good design

**CONCERN:** No automated leakage check runs in the current pipeline. It relies on the split builder to correctly separate families.

### Q14: Is the existing latency code CUDA-correct?

**NO — DEFECT D-004.**

In `eval.py` `benchmark_latency()`:
```python
t0 = time.perf_counter()
# ... encode ...
latencies.append(time.perf_counter() - t0)
```

Missing:
- No `torch.cuda.synchronize()` before starting timer
- No `torch.cuda.synchronize()` after inference
- No warm-up iterations
- CUDA operations are asynchronous — `time.perf_counter()` measures CPU time, not actual GPU time

### Q15: Do current results use the same architecture described in the paper?

**NO — SIGNIFICANT DISCREPANCIES.**

| Intended (Paper) | Actual Implementation |
|---|---|
| Tool-name text embeddings as node features | Integer ID nn.Embedding |
| Directed GNN (separate in/out transforms) | Default GCN with bidirectional edges (undirected) |
| Tool name = semantic description | Tool names are short abbreviations (db_read, etc.) |
| Structural generalization evaluation | Partial — only one held-out topology split |

The previous experiment results confirm:
- **SBERT FT R@1 = 0.922-0.945** across configurations
- **LEGR Default GCN R@1 = 0.852-0.918** (lower than SBERT)
- **LEGR Directed R@1 = 0.675-0.789** (even lower)

---

## 3. Identified Defects

### D-001: Integer-ID Node Features (CRITICAL)
- **Severity:** CRITICAL
- **Evidence:** `encoders.py:150` — `nn.Embedding(num_tools + 1, tool_embed_dim)`
- **Cause:** Graph nodes use learned integer embeddings, not text-name embeddings
- **Impact:** Graph tower has no pretrained semantic knowledge of tool names; must learn tool semantics from scratch during contrastive training
- **Owner:** IMPLEMENTATION ENGINEER
- **Fix:** Encode tool names through the MiniLM text backbone (frozen or cached) to produce initial node features. Replace `nn.Embedding` with text-derived features.

### D-002: Dead Text Node Feature Code Path
- **Severity:** MEDIUM
- **Evidence:** `encoders.py:186-188` — fills `text_feat` with zeros
- **Cause:** `use_text_node_features` stub never receives actual embeddings
- **Owner:** IMPLEMENTATION ENGINEER
- **Fix:** Implement proper text-name feature injection via this path

### D-003: Default Bidirectional (Undirected) Edges (CRITICAL)
- **Severity:** CRITICAL
- **Evidence:** `data_synth.py:1601-1603` — `bidirectional=True` by default
- **Cause:** Adding reverse edges destroys dependency direction information
- **Impact:** GCN cannot distinguish A→B from B→A
- **Owner:** IMPLEMENTATION ENGINEER
- **Fix:** For the new campaign, use `bidirectional=False` with `DirectedGraphEncoder`

### D-004: CUDA Latency Measurement (MEDIUM)
- **Severity:** MEDIUM
- **Evidence:** `eval.py:710-720` — no `cuda.synchronize()`, no warm-up
- **Cause:** CUDA async ops make wall-clock timing inaccurate
- **Owner:** IMPLEMENTATION ENGINEER
- **Fix:** Add synchronize barriers and warm-up passes

### D-005: No Structural-Twin Distractors (CRITICAL for campaign_v4)
- **Severity:** CRITICAL
- **Evidence:** Previous datasets had mostly unique tool sets per DAG
- **Cause:** Templates and random generation rarely produce same-toolset/different-edge variants
- **Impact:** SBERT can solve the benchmark via tool-set matching alone
- **Owner:** RESEARCH ARCHITECT + ENGINEER
- **Fix:** Campaign v4 must generate structural twins — same tool multiset, different directed edges

### D-006: No Automated Leakage Check
- **Severity:** HIGH
- **Evidence:** No test validates that held-out topology families don't appear in training
- **Owner:** TEST ENGINEER
- **Fix:** Add programmatic leakage detection in dataset validation

---

## 4. Current Dataset Pipeline

```
data_synth.py WORKFLOW_TEMPLATES (hardcoded + programmatic generation)
    → LEGRDataset (entity-variant multiplication)
    → build_splits() (template-level train/val/test)
    → dag_to_pyg() (integer features, bidirectional edges)
    → build_ged_matrix() (exact GED for small DAGs)

graph_utils.py
    → generate_dags() (topology generators: diamond, fork_join, etc.)
    → generate_hard_negatives() (edge-swap, tool-swap, etc.)

scripts/prepare_legr_30tool_dataset.py
    → Builds upgraded/ CSV splits from the full pipeline
```

Queries are generated **locally** using programmatic template synthesis (`_synthesize_queries()`), NOT Azure OpenAI. The current pipeline does not use Azure for query generation.

---

## 5. Current LEGR Architecture Summary

```
QUERY TOWER:
  query text → all-MiniLM-L6-v2 (4 of 6 layers frozen)
  → mean pool → Linear(384, 256) → L2-normalize → z_text ∈ R^256

GRAPH TOWER (default):
  tool integer ID → nn.Embedding(46, 64) → tool_emb
  topological rank → nn.Embedding(17, 64) → topo_emb
  [tool_emb; topo_emb] → 3-layer GCNConv(128→128→128) + LayerNorm + ReLU
  → global_mean_pool → Linear(128, 256) → L2-normalize → z_graph ∈ R^256

LOSS:
  InfoNCE(z_text, z_graph) + 0.30 * GED_weighted_loss
```

---

## 6. Current SBERT Architecture

```
QUERY TOWER:
  query text → all-MiniLM-L6-v2 → mean pool → Linear(384, 256) → L2-normalize

DOCUMENT TOWER:
  dag_to_text(G) → all-MiniLM-L6-v2 → mean pool → Linear(384, 256) → L2-normalize

  dag_to_text example: "db_read -> process_refund, db_read -> create_ticket"
```

**Critical insight:** SBERT's document tower sees the full edge structure as text (`"A -> B, A -> C"`), which contains BOTH tool semantics AND dependency direction. The pretrained MiniLM can parse this semi-structured text with positional understanding. LEGR's graph tower, by contrast, starts with integer IDs and must learn everything from scratch.

---

## 7. Current LLM Baseline Pipeline

```
llm_dag_baseline.py:
  For each query:
    → Send query + tool vocabulary to LLM (Ollama/Azure)
    → LLM outputs {"tools": [...], "edges": [[src, dst], ...]}
    → Parse JSON, validate tools, check DAG acyclicity
    → Compute tool-set F1, GED vs ground truth
    → Track: parse failures, cyclic outputs, latency

  Supported providers: ollama_llama (llama3.2:3b), ollama_gpt_oss (gpt-oss:120b), azure_openai
```

---

## 8. Files Requiring Modification for Campaign v4

### Must Modify:
1. **`src/encoders.py`** — Add text-name-based node feature encoder
2. **`src/data_synth.py`** — New tool vocabulary, dag_to_pyg with text features
3. **`src/train.py`** — Support new `legr_directed_toolname` config
4. **`src/eval.py`** — Add structural-twin metrics, CUDA-correct latency
5. **`src/sbert_ft_baseline.py`** — Ensure compatibility with new dataset schema
6. **`src/utils/graph_utils.py`** — Add new topology generators and structural-twin builder

### Must Create:
1. **`src/data/tool_registry.py`** — New 15/30/45 nested tool library (ACTION_FIRST naming)
2. **`src/data/topology_templates.py`** — Extended topology family definitions
3. **`src/data/dag_generator.py`** — Programmatic DAG + structural-twin generation
4. **`src/data/azure_query_generator.py`** — Azure OpenAI query synthesis (with caching)
5. **`src/data/dataset_validator.py`** — 20-point validation suite
6. **`src/data/split_builder.py`** — Topology-aware OOD split builder
7. **`src/data/build_campaign_v4.py`** — Campaign entry point

### Must NOT Modify (Preserve):
1. Existing `configs/llm_providers.json` — Azure/Ollama config
2. Existing `upgraded/` datasets — keep as legacy reference
3. Existing checkpoint format — maintain backward compatibility

---

## 9. Previous Campaign Results (for comparison)

| Experiment | Upgraded R@1 | Upgraded_v3 R@1 |
|---|---|---|
| SBERT FT (GED=0) | 0.9217 | 0.9273 |
| SBERT FT (GED=0.30) | 0.9217 | 0.9273 |
| SBERT FT (tied) | 0.9458 | 0.9364 |
| LEGR Default GCN (30-tool) | 0.8524 | 0.9182 |
| LEGR Default GCN (15-tool) | 0.9561 | 0.9674 |
| DirGNN Directed | 0.6747 | 0.7509 |
| DirGNN Tied-In-Out | 0.7892 | 0.8255 |

**Key finding:** SBERT consistently matches or exceeds LEGR, likely because:
1. SBERT sees tool names + edge text ("A→B") via a pretrained text encoder
2. LEGR uses random integer embeddings for tools
3. Unique tool sets per DAG allow SBERT to solve via tool-bag matching

---

## 10. Architect's Verdict

The current LEGR implementation has **three critical methodological defects** that likely explain SBERT's dominance:

1. **No semantic tool-name features** — LEGR's graph tower starts blind
2. **Undirected edges by default** — topology direction is destroyed
3. **No same-toolset distractors** — the benchmark doesn't require structural reasoning

Campaign v4 must fix all three simultaneously to create a fair benchmark that actually measures structural generalization capability.
