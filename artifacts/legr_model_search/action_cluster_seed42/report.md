# LEGR Functional Clustering Audit — Seed 42

**Result: CORROBORATED**

The seed-42 winning graph embedding passes the predeclared corroboration rule: its cross-validated macro-F1 and distance separation exceed shuffled-label performance, and the macro-F1 bootstrap lower bound is above 1/3 chance.

## Population and labels

The audit deduplicated **651** Campaign-v4 15-tool DAGs by canonical graph identity. The primary three-class analysis contains **487** graphs; **164** exact-plurality ties are retained in `graph_labels.csv` but excluded from metrics and plots.

| Label | Graphs |
|---|---:|
| read | 232 |
| edit | 223 |
| orchestrate | 32 |
| mixed (excluded) | 164 |

Labels were computed without an LLM: Campaign-v4 registry category `DATA_RETRIEVAL` maps to `read`, `STATE_MODIFICATION` to `edit`, and `ORCHESTRATION` to `orchestrate`. A graph receives the unique most frequent label among its tools; exact ties are `mixed`.

## Original-space diagnostics

| Representation | Cosine silhouette | Silhouette p | 5-NN macro-F1 (95% CI) | F1 p | Balanced purity | Distance gap | Gap p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Winning GPS adapter | 0.1550 | 0.0010 | 0.9462 [0.9102, 0.9735] | 0.0010 | 0.8931 | 0.2304 | 0.0010 |
| Inherited V3 graph | 0.0873 | 0.0010 | 0.9291 [0.8923, 0.9624] | 0.0010 | 0.8868 | 0.1888 | 0.0010 |
| SBERT document control | 0.1163 | 0.0010 | 0.9359 [0.9006, 0.9658] | 0.0010 | 0.8763 | 0.1619 | 0.0010 |

Permutation p-values use shuffled action labels. The distance gap is mean between-class minus mean within-class cosine distance; positive values indicate separation. The predeclared criterion requires GPS-adapter 5-NN macro-F1 p < 0.05, distance-gap p < 0.05, and a macro-F1 bootstrap 95% lower bound above the three-class chance value of 1/3.

## Representation definitions

- **Winning GPS adapter:** the trained degree-encoded, directed GPS graph adapter with dual-attention readout.
- **Inherited V3 graph:** the V3 graph-tower output inside the same composite checkpoint.
- **SBERT document control:** the frozen SBERT expert's embedding of the serialized DAG text. This is the semantic control, not a graph encoder.

## Integrity and interpretation

Protected input integrity: **PASS** (27 files checked before and after).

PCA and t-SNE are visual summaries only. All evidence decisions use the original 256-dimensional cosine space. Because the population includes training graphs, this audit concerns learned representation geometry rather than zero-shot generalization. The orchestrate class is smaller than read/edit, so macro-F1, class-balanced purity, stratified folds, and stratified bootstrap intervals are reported. Finally, these labels are derived from the same tool identities available to the encoders; clustering supports functional organization but does not by itself prove causal or independently learned functional reasoning.

## Run metadata

- Checkpoint: `C:/Users/tkadam/LEGR/artifacts/legr_model_search/confirm_r1_15t_s42_30795749e6/best_model.pt`
- Device: `cuda`
- Seed: 42
- Embedding extraction: 6.77 seconds
- Total runtime: 86.18 seconds
- Permutations/bootstrap samples: 1000/2000
