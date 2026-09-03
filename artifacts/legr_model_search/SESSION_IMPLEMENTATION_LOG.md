# LEGR Model-Research Implementation Log

## Purpose

This document records the work completed in this chat to design, implement, test, and run a model-only LEGR research campaign.

The user constraints were:

- do not change Campaign-v4 datasets or splits;
- do not directly change existing LEGR or SBERT implementations;
- new models may build on existing models through wrapping, subclassing, checkpoint initialization, and composition;
- mathematical objectives and any model-related components were fair game;
- run a broad search within 24 hours of RTX 6000 Ada compute;
- optimize exact-plan Recall and tool-set F1;
- preserve existing checkpoints and results.

## Initial Diagnosis

The Campaign-v4 session log and source were inspected before implementation. The important findings were:

1. Integer-ID LEGR was not a meaningful final architecture because arbitrary tool IDs did not share pretrained semantics with the query tower.
2. V2 tool-name features improved held-out retrieval substantially.
3. V3 tied MiniLM and split set/GNN readout improved the 50-DAG results to approximately 82.7% Recall@1 and 94.6% tool F1.
4. The 50-DAG held-out gallery contained no same-toolset structural twins, making SBERT's 92.7% Recall@1 largely a tool-set retrieval result.
5. The 322-DAG fair gallery exposed SBERT's structural weakness, but existing LEGR same-toolset accuracy was only slightly above chance.
6. The existing diagonal InfoNCE objective could treat additional paraphrases of the same DAG as negatives.
7. With six paraphrases per DAG and batch size 128, the old random row batching was expected to create roughly 27 off-diagonal same-DAG false-negative pairs per batch.
8. The existing GED auxiliary was mathematically questionable because it summed raw similarity logits in its denominator and used a fast graph distance incapable of distinguishing some equal-degree structural twins.
9. Total topological-sort rank arbitrarily orders parallel nodes and is therefore not a permutation-invariant structural position.
10. A single cosine between two 256-dimensional vectors forced tool identity and graph structure through one bottleneck.

## Repository Safety

Before implementation:

- the worktree was inspected;
- two pre-existing modified result files were found;
- two unrelated untracked files were found;
- those files were not edited or removed;
- CUDA availability was confirmed;
- the GPU was an NVIDIA RTX 6000 Ada Generation with 49,140 MiB memory;
- MiniLM was the only requested language backbone present in the local Hugging Face cache.

The implementation created new files and artifact directories only. Existing datasets, model files, training entrypoints, evaluation entrypoints, checkpoints, and results were hashed before and after the campaign.

Final integrity result:

```json
{
  "changed": [],
  "missing": [],
  "added": []
}
```

This integrity report applies to the declared immutable paths. New files under `src/legr_experiments`, the new runner, tests, and `artifacts/legr_model_search` were intentionally added.

## New Implementation

### Package and entrypoint

Created:

- `src/legr_experiments/__init__.py`
- `src/legr_experiments/config.py`
- `src/legr_experiments/data.py`
- `src/legr_experiments/structures.py`
- `src/legr_experiments/samplers.py`
- `src/legr_experiments/losses.py`
- `src/legr_experiments/graph_encoders.py`
- `src/legr_experiments/model.py`
- `src/legr_experiments/metrics.py`
- `src/legr_experiments/evaluation.py`
- `src/legr_experiments/integrity.py`
- `src/legr_experiments/training.py`
- `src/legr_experiments/search.py`
- `scripts/run_legr_model_search.py`
- `tests/test_legr_experiments.py`

### Data handling

The new loader reads the existing CSV files without rewriting them. At runtime it derives:

- unique labelled-DAG keys;
- tool-membership targets;
- direct adjacency;
- transitive reachability;
- five-class pair relations;
- invariant structural node features;
- structural-twin group indices.

None of these derived tensors are written back into Campaign-v4 data.

### Sampling

A `GroupAwareBatchSampler` was implemented. It:

- selects one existing paraphrase per DAG per epoch;
- rotates the selected paraphrase across epochs;
- keeps existing same-toolset twin DAGs together;
- avoids same-DAG false negatives;
- reduces redundant computation relative to processing every paraphrase in every epoch.

### Losses

Implemented:

- multi-positive InfoNCE;
- listwise exact-DAG ranking;
- same-toolset twin listwise ranking;
- pairwise softplus twin ranking with margins 0.1, 0.2, and 0.4;
- class-balanced tool-membership loss;
- five-class pair-relation cross-entropy;
- relation-signature embedding-distance regularization;
- online hard-negative loss;
- curriculum scaling of twin pressure.

The previous GED objective remained untouched in its original file and was not used by the winning model.

### Graph architectures

Implemented as new adapters around existing V3 tool-name node features:

- V3-style/residual directed message passing;
- directed gated message passing;
- directed PNA-style aggregation;
- relation-biased DAG Graphormer;
- GPS local directed message passing plus global attention.

Every new directed block maintains separate incoming and outgoing transformations.

### Structural encodings

Implemented runtime variants:

- none;
- source and sink depth;
- in/out degree plus source/sink flags;
- directed relation/path bias;
- combined invariant features.

No new model uses arbitrary total topological-sort rank.

### Readout variants

Implemented:

- mean/V3-style pooling;
- dual tool and structure attention;
- virtual-node attention;
- Set2Set;
- concatenated mean, max, and normalized-add pooling.

### Fusion

The research model composes:

- the unchanged existing `LEGRDualEncoderV3`;
- the unchanged existing `SBERTFineTuneDualEncoder`;
- a new graph adapter;
- explicit tool and relation heads;
- optional query-graph reranking.

Fusion variants tested:

- graph only;
- SBERT semantic only;
- fixed mean;
- learned global scalar;
- query-conditioned gated fusion.

### Reranker

A candidate-aware cross-attention reranker was implemented with:

- top-K settings 10, 20, and 40;
- one or two cross-attention layers;
- residual score addition.

The final promoted model did not require the reranker.

### Optimization

Implemented and screened:

- frozen inherited backbone;
- final-two-layer unfreezing;
- full structural-query unfreezing;
- cosine schedule;
- plateau schedule;
- exponential moving average;
- stochastic weight averaging;
- online hard-negative mining;
- structural curriculum.

## Verification

### Static verification

- All new modules passed `compileall`.
- The isolated package imported successfully.

### Unit tests

Seven new tests passed, covering:

- direct, indirect, reverse, and parallel relation labels;
- equal structural features for parallel nodes;
- permutation equivariance of invariant features;
- correct multi-positive behavior;
- finite composite loss and backward pass;
- one paraphrase per DAG in group-aware batches;
- permutation-invariant graph readout.

Fifteen relevant existing tests also passed, covering SBERT helpers and one-node/directed-GNN behavior.

Total targeted result: 22 passed.

### CUDA smoke

CUDA forward tests passed for:

- residual directed adapter;
- gated directed adapter;
- PNA adapter;
- Graphormer;
- GPS;
- cross-attention reranker.

### End-to-end smoke

A real Campaign-v4 smoke run:

- loaded both V3 and SBERT checkpoints with zero missing or unexpected keys;
- trained one CUDA epoch;
- produced a checkpoint and metrics;
- achieved 84.694% dev Recall@1 and 97.354% tool F1 for the control configuration;
- measured 3.75 ms p95 in that smoke configuration;
- passed the immutable-file comparison.

## Search Campaign

The complete campaign ran 65 configurations with zero failures.

| Stage | Completed runs |
|---|---:|
| Reproduced baseline | 1 |
| Mathematical screening | 12 |
| Architecture/position/readout screening | 20 |
| Backbone/fusion screening | 5 |
| Reranker/optimization screening | 15 |
| Three-seed confirmation | 12 |
| **Total** | **65** |

Language-backbone availability:

| Backbone | Status |
|---|---|
| all-MiniLM-L6-v2 | Available and tested |
| all-mpnet-base-v2 | Not present in local cache; recorded as skipped |
| e5-base-v2 | Not present in local cache; recorded as skipped |
| bge-base-en-v1.5 | Not present in local cache; recorded as skipped |

No network download or silent replacement was performed.

## Search Findings

### Mathematical stage

- The reproduced SBERT dev control scored 87.415% Recall@1 and 98.090% tool F1.
- The first hybrid configuration immediately improved Recall@1 to 89.796% and F1 to 98.658%.
- Pairwise margins 0.1 and 0.2 reached 90.136% Recall@1.
- Halved auxiliary weights were best, reaching 90.476% Recall@1 and 98.669% F1.
- Random row batching was slower and scored only 88.776% Recall@1 and 97.802% F1.

### Architecture stage

- Removing arbitrary topological-rank features preserved Recall and increased F1.
- Residual sparse message passing tied V3-style performance.
- Gating and PNA did not improve the joint objective.
- Graphormer reached 91.156% Recall@1.
- GPS with combined or depth features reached 91.156% Recall@1.
- GPS with degree/source/sink features led the stage at 91.497% Recall@1 and 98.920% F1.

### Fusion stage

- Graph only: 87.075% Recall@1, 97.076% F1.
- Semantic only: 87.415% Recall@1, 98.090% F1.
- Fixed mean and scalar fusion: approximately 72.109% Recall@1 and 86.6% F1.
- Query-conditioned gated fusion: 91.497% Recall@1 and 98.920% F1.

### Confirmation

The winning GPS-degree, query-gated, curriculum configuration was trained with seeds 42, 123, and 2026.

| Metric | SBERT control | Confirmed LEGR mean | Difference |
|---|---:|---:|---:|
| Recall@1 | 87.415% | **91.383%** | **+3.968 pp** |
| Tool F1 | 98.090% | **98.832%** | **+0.742 pp** |
| Same-toolset R@1 | 93.197% | **94.898%** | **+1.701 pp** |
| Recall@3 | **99.660%** | 98.980% | -0.680 pp |
| Recall@5 | **100.000%** | 99.773% | -0.227 pp |

## Interpretation Given During the Chat

The results were described as promising but not final because:

- the comparison above is on the development protocol;
- SBERT's historical 92.667% Recall@1 is from a different 50-DAG held-out protocol;
- the final 322-DAG twin gallery has not yet been scored with the confirmed model;
- paired-bootstrap confidence intervals have not yet been calculated on final test predictions.

No claim of universal test-set superiority was made.

## Artifacts Produced

Primary output directory: `artifacts/legr_model_search/`

Important files:

- `search_manifest.json`: all 65 runs, resolved configurations, metrics, failures, timings, and checkpoint-load reports;
- `immutable_before.json`: hashes before the campaign;
- `immutable_after.json`: hashes after the campaign;
- `integrity_report.json`: zero immutable drift;
- `skipped_backbones.json`: unavailable model backbones and reasons;
- one isolated directory per run containing its resolved configuration, history, summary, and checkpoint;
- this session log;
- `LEGR_WINNER_ANALYSIS.md`: focused winning-model analysis.

## Remaining Work

The implementation and development search are complete. The remaining evidence required for a final paper claim is:

1. load each confirmed winner checkpoint;
2. evaluate on the unchanged 50-DAG held-out gallery;
3. evaluate on the unchanged 322-DAG twin-filled gallery;
4. run the same exact evaluation for the SBERT control;
5. compute paired-bootstrap confidence intervals;
6. report whether LEGR wins Recall@1 and F1 on final test data;
7. avoid changing the dataset if the final result is negative.

