# LEGR GPS-Degree Model Analysis

## Result Status

The model described here outperformed the frozen fine-tuned SBERT control on the **unchanged Campaign-v4 development protocol**. It has not yet been evaluated on the final 322-DAG held-out twin gallery, so this document does not claim final test-set superiority.

Development protocol:

- Tier: 15 tools
- Queries: 294 development queries
- Gallery: 49 unique development DAGs
- Seeds used for confirmation: 42, 123, and 2026
- Candidate order: deterministically randomized
- Selection metric: Recall@1, followed by same-toolset Recall@1 and tool-set F1
- Maximum permitted p95 latency: 100 ms

## Main Comparison

The SBERT control is the existing `sbert_ft_ged_15t_s42` fine-tuned dual-text encoder used alone. It embeds the query and serialized `dag_text`, then ranks by cosine similarity. It contains no GNN or explicit structural reasoning.

| Metric | Frozen SBERT control | LEGR winner, three-seed mean | Absolute difference |
|---|---:|---:|---:|
| Recall@1 | 87.415% | **91.383%** | **+3.968 pp** |
| Recall@3 | **99.660%** | 98.980% | -0.680 pp |
| Recall@5 | **100.000%** | 99.773% | -0.227 pp |
| Tool-set F1 | 98.090% | **98.832%** | **+0.742 pp** |
| Same-toolset Recall@1 | 93.197% | **94.898%** | **+1.701 pp** |
| p95 latency | 15.14 ms | 16.85 ms | +1.71 ms |

The primary objectives both improved: exact-plan Recall@1 increased by 3.97 percentage points and tool-set F1 increased by 0.74 percentage points. Same-toolset Recall@1 also improved, indicating that the gain was not exclusively tool-bag matching. Recall@3 and Recall@5 decreased slightly but remained near saturation.

## Three-Seed Confirmation

| Seed | Recall@1 | Recall@3 | Recall@5 | Tool F1 | Same-toolset R@1 | p95 latency | Epochs completed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | **92.517%** | 98.980% | 99.660% | **98.915%** | **95.578%** | 14.80 ms | 23 |
| 123 | 90.816% | 98.980% | 100.000% | 98.818% | 94.558% | 21.85 ms | 23 |
| 2026 | 90.816% | 98.980% | 99.660% | 98.764% | 94.558% | 13.88 ms | 23 |
| **Mean** | **91.383%** | **98.980%** | **99.773%** | **98.832%** | **94.898%** | **16.85 ms** | **23** |

The improvement was not confined to seed 42. Recall@1 remained above the SBERT control for all three seeds, ranging from 90.82% to 92.52%. Tool F1 remained between 98.76% and 98.91%.

## Winning Model Configuration

The winning configuration is the `confirm_r1` model:

- Text model: `sentence-transformers/all-MiniLM-L6-v2`
- Existing graph foundation: `LEGRDualEncoderV3`, loaded without changing its implementation
- Semantic expert: existing fine-tuned SBERT, loaded and frozen
- New graph adapter: GPS-style hybrid
- Local graph component: directed message passing with separate incoming and outgoing transformations
- Global graph component: relation-biased multi-head graph attention
- Structural node features: normalized in-degree, normalized out-degree, source flag, and sink flag
- Readout: dual attention pooling
- Fusion: query-conditioned gated fusion
- Graph layers: 4
- Hidden and output dimensions: 256
- Attention heads: 8
- Query backbone unfreezing: final two transformer layers
- Batch size: 64
- Maximum epochs: 75
- Early stopping: epoch 22 or 23 during confirmation
- Reranker: disabled in the winner
- Training curriculum: enabled

## Architectural Changes

### 1. Non-destructive composition

The new model wraps the existing V3 LEGR and SBERT implementations. Existing source files, checkpoints, results, and datasets were not edited. The inherited models are loaded as components inside a new research model.

### 2. SBERT retained as a semantic expert

The SBERT score remains available to the model instead of forcing the graph branch to relearn language and tool semantics. This is important because graph-only fusion scored only 87.075% Recall@1 and 97.076% tool F1.

### 3. GPS directed graph encoder

The new graph adapter combines:

- sparse directed local message passing;
- separate transformations for incoming and outgoing dependencies;
- global graph attention;
- attention biases for direct-forward, direct-reverse, ancestor, descendant, self, and unrelated node pairs.

This gives the graph branch both local edge sensitivity and a global receptive field.

### 4. Permutation-invariant structural features

The existing total topological-sort rank was not used. A topological sort arbitrarily orders parallel nodes and can teach false sequencing. The winning model instead uses degree/source/sink features that are unchanged under node permutation.

The architecture sweep showed:

| Architecture | Recall@1 | Tool F1 | Same-toolset R@1 | p95 |
|---|---:|---:|---:|---:|
| GPS + degree/source/sink | **91.497%** | **98.920%** | **94.898%** | 15.09 ms |
| GPS + depth | 91.156% | 98.783% | 94.898% | 15.62 ms |
| GPS + combined features | 91.156% | 98.783% | 94.898% | 13.64 ms |
| Graphormer, no node-position features | 91.156% | 98.731% | 94.898% | 13.63 ms |
| PNA + combined features | 89.796% | 98.523% | 94.898% | 1.49 ms |
| Gated directed GNN | 89.796% | 97.962% | 94.898% | approximately 1.4 ms |

These are single-seed screening numbers. GPS-degree was promoted to multi-seed confirmation.

### 5. Query-conditioned gated fusion

Five scores are available to the fusion layer:

1. frozen SBERT semantic score;
2. inherited V3 graph score;
3. new structural graph score;
4. explicit tool-compatibility score;
5. explicit pair-relation score.

The query encoder produces nonnegative mixture weights, allowing structural language to receive more graph weight while ordinary tool-selection language retains more semantic weight.

Fusion ablation:

| Fusion | Recall@1 | Tool F1 | Same-toolset R@1 |
|---|---:|---:|---:|
| Graph only | 87.075% | 97.076% | 94.218% |
| SBERT semantic only | 87.415% | 98.090% | 93.197% |
| Fixed mean | 72.109% | 86.641% | 94.558% |
| Learned global scalar | 72.109% | 86.618% | 94.558% |
| Query-conditioned gate | **91.497%** | **98.920%** | **94.898%** |

Fixed score averaging failed because expert score distributions have different scales. Query-conditioned fusion learned when each expert was reliable.

## Mathematical Changes

The winning objective was:

\[
L = L_{\text{listwise}}
  + 0.5L_{\text{twin}}
  + 0.25L_{\text{tool}}
  + 0.5L_{\text{relation}}
  + 0.1L_{\text{multi-positive}}
  + 0.1L_{\text{distance}}.
\]

### Multi-positive contrastive learning

All six existing paraphrases sharing a `dag_id` are treated as positives. The earlier diagonal-only InfoNCE formulation could treat another correct paraphrase/DAG pairing as a negative.

### Group-aware batching

Each batch uses one existing paraphrase per DAG and deliberately colocates existing same-toolset twins. This changes only training order; it creates no examples and changes no dataset files.

### Explicit tool objective

A class-balanced binary objective predicts the tools required by the query. This protects tool F1 while the structural branch learns graph differences.

### Explicit relation objective

For each active tool pair, the model predicts one of five relations:

1. parallel or incomparable;
2. direct forward dependency;
3. indirect forward precedence;
4. direct reverse dependency;
5. indirect reverse precedence.

### Structural-distance regularization

Graph-embedding distance is weakly aligned with normalized disagreement between pair-relation signatures. This regularizer uses existing edges in memory and does not add persistent labels.

### Reduced auxiliary weights

Full-strength auxiliary losses were not optimal. Halving twin, tool, and relation weights increased screening Recall@1 to 90.476% while retaining 98.669% F1 before the architecture search.

### Curriculum

Twin-ranking pressure increases during the first half of training instead of being applied at full strength from epoch one. The confirmation winner used this curriculum.

## Why the Model Improved

The improvement comes from complementary specialization:

- SBERT preserves strong query-to-tool and serialized-language matching.
- The GPS branch can distinguish directed graph relationships.
- Explicit tool supervision prevents graph learning from sacrificing tool identity.
- Relation supervision makes ordering and parallelism directly learnable.
- Query-conditioned fusion avoids destructive averaging between differently calibrated experts.
- Group-aware, multi-positive training removes false negatives and exposes structural twins consistently.

## What Did Not Help

- Graph-only scoring lost semantic accuracy.
- Fixed and global-scalar fusion collapsed Recall and F1.
- Random row batching was slower and did not improve the promoted metrics.
- Stronger auxiliary-loss weights reduced Recall.
- Gated sparse GNN and PNA did not beat GPS or Graphormer.
- Path bias without degree features did not beat GPS-degree.
- Candidate reranking was not necessary for the best promoted model.
- MPNet, E5, and BGE were not locally cached and were therefore recorded as unavailable rather than downloaded or silently substituted.

## Required Final Evaluation

Before claiming that LEGR universally beats SBERT, the confirmed checkpoint must still be evaluated on:

1. the unchanged 50-DAG held-out gallery, where the historical SBERT result is 92.667% Recall@1 and 98.107% tool F1;
2. the unchanged 322-DAG twin-filled gallery;
3. paired-bootstrap confidence intervals for Recall@1 and tool-F1 differences.

The development results support promotion to final evaluation, not a final test-set claim.

## Artifacts

- Complete experiment manifest: `artifacts/legr_model_search/search_manifest.json`
- Integrity report: `artifacts/legr_model_search/integrity_report.json`
- Winner checkpoints: directories beginning with `confirm_r1_15t_`
- Implementation package: `src/legr_experiments/`
- Campaign runner: `scripts/run_legr_model_search.py`

