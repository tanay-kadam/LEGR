# FINAL REPORT

## 1. Executive Summary

Timestamped run root: `C:\Users\tkadam\LEGR\experiment_runs\20260831_1218`.
Git commit: `fae6f498512f442218366b8fb264ff35c3834f1c` branch `main` dirty=True.

Statuses:

- `upgraded__dep_legr__legr_gcn__gcn_undirected__seed_42`: VERIFIED
- `upgraded__dep_legr__legr_gcn__gcn_undirected_15__seed_42`: VERIFIED
- `upgraded__task1_sbert__sbert_ft__ged_0__seed_42`: VERIFIED
- `upgraded__task1_sbert__sbert_ft__ged_030__seed_42`: VERIFIED
- `upgraded__task1_sbert__sbert_ft__tied_weights__seed_42`: VERIFIED
- `upgraded__task4_dirgnn__dirgnn__directed__seed_42`: VERIFIED
- `upgraded__task4_dirgnn__dirgnn__tied_in_out__seed_42`: VERIFIED
- `upgraded__task2_latent__legr__action_type_analysis__seed_42`: VERIFIED
- `upgraded__task3_atomic__legr_15tool__zero_shot_atomic__seed_42`: VERIFIED
- `upgraded__task3_atomic__legr_30tool__zero_shot_atomic__seed_42`: VERIFIED
- `upgraded_v3__dep_legr__legr_gcn__gcn_undirected__seed_42`: VERIFIED
- `upgraded_v3__dep_legr__legr_gcn__gcn_undirected_15__seed_42`: VERIFIED
- `upgraded_v3__task1_sbert__sbert_ft__ged_0__seed_42`: VERIFIED
- `upgraded_v3__task1_sbert__sbert_ft__ged_030__seed_42`: VERIFIED
- `upgraded_v3__task1_sbert__sbert_ft__tied_weights__seed_42`: VERIFIED
- `upgraded_v3__task4_dirgnn__dirgnn__directed__seed_42`: VERIFIED
- `upgraded_v3__task4_dirgnn__dirgnn__tied_in_out__seed_42`: VERIFIED
- `upgraded_v3__task2_latent__legr__action_type_analysis__seed_42`: VERIFIED
- `upgraded_v3__task3_atomic__legr_15tool__zero_shot_atomic__seed_42`: VERIFIED
- `upgraded_v3__task3_atomic__legr_30tool__zero_shot_atomic__seed_42`: VERIFIED

Verified runs: 20/20. Failed: 0 (archived retries documented in `failures_and_retries.md`).

30-tool atomic is now measured: 30-tool frozen GCN + routing_15tools queries (not the routing_30 benchmark; those labels remain OOV). Dataset A 45-candidate corpus. Outcome B on both datasets.

## 2. Repository / Environment

- Python: 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)]
- PyTorch: 2.11.0+cu128 CUDA runtime 12.8
- cuda_available: True
- GPU: NVIDIA RTX 6000 Ada Generation index 0 mem 47.99 GB
- selected_device: cuda:0
- nvidia-smi: 0, NVIDIA RTX 6000 Ada Generation, 0 %, 1217 MiB, 49140 MiB, 38

## 3. Dataset Summary

Dataset A `upgraded` and Dataset B `upgraded_v3` are distinct SHA256 trees (see `dataset_validation.json`).
30-tool train sizes: upgraded=1396, upgraded_v3=1692. Test: 332 vs 550.
15-tool train sizes: upgraded=2814, upgraded_v3=2922. Test: 592 vs 737.
upgraded_v3 has no local hard_negatives.csv; Dataset B eval skipped packaged upgraded hard negatives to avoid mixing.

## 4. Task 1 Results

R@1=0.9217; F1=0.9721; GED=0.2078

- SBERT_FT_GED0: R@1=0.9217; F1=0.9721; GED=0.2078 vs R@1=0.9273; F1=0.9654; GED=0.1564
- SBERT_FT_GED030: R@1=0.9217; F1=0.9721; GED=0.2078 vs R@1=0.9273; F1=0.9654; GED=0.1564
- SBERT_FT_TIED: R@1=0.9458; F1=0.9810; GED=0.1807 vs R@1=0.9364; F1=0.9595; GED=0.1509
- LEGR_DEFAULT_GCN_30: R@1=0.8524; F1=0.9057; GED=0.4910 vs R@1=0.9182; F1=0.9573; GED=0.2109

## 5. Task 2 Results

- upgraded: NO SUPPORT sil=-0.011417719535529613
- upgraded_v3: NO SUPPORT sil=-0.013153651729226112

## 6. Task 3 Results

Same atomic queries (`routing_15tools`) and 15 one-node LEGR tools. Encoder and compositional pool vary. `routing_30tools` was not used.

- 15-tool encoder, upgraded (45 candidates): Standard=84.0; Lexical=50.7; Confusable=45.1; Paraphrase=81.4; agg=69.4
- 15-tool encoder, upgraded_v3 (53 candidates): Standard=92.8; Lexical=62.1; Confusable=59.1; Paraphrase=93.0; agg=80.5
- 30-tool encoder, upgraded (45 candidates): Standard=63.8; Lexical=31.5; Confusable=36.4; Paraphrase=65.3; agg=52.3
- 30-tool encoder, upgraded_v3 (69 candidates): Standard=91.2; Lexical=59.0; Confusable=51.6; Paraphrase=89.9; agg=77.3

All four are Outcome B (Lexical and Confusable < 70%). The 30-tool encoder does **not** improve atomic retrieval on Dataset A versus the 15-tool encoder. Do not typeset these as a routing_30 result.

## 7. Task 4 Results

- DIRGNN_DIRECTED: R@1=0.6747; F1=0.8080; GED=1.0693 vs R@1=0.7509; F1=0.8683; GED=0.8982
- DIRGNN_TIED_IN_OUT: R@1=0.7892; F1=0.8989; GED=0.5964 vs R@1=0.8255; F1=0.9073; GED=0.6291
Comparison of DirGNN vs default GCN is confounded by layer type (DirectedGraphEncoder vs GCNConv).

## 8. Cross-Dataset Comparison

See `dataset_comparison.csv`. Do not treat a higher metric as automatically better data.

## 9. Verification Summary

Independent reload+eval used `verify_one.py` for trained models. Atomic eval was rerun into `independent_rerun/`.

## 10. Failures and Fixes

See `failures_and_retries.md`. Integrity patches applied before training:
- CUDA hard-fail in `train.py` / `sbert_ft_baseline.py`
- Reload best SBERT checkpoint before its built-in eval
- Do not attach default upgraded hard negatives when `--dataset_csv` points elsewhere
- Save real embeddings + embedding_kind for action-type analysis

## 11. Paper Readiness

- Fine-tuned SBERT Table 2 row: **READY**
- Table 2 LEGR GCN / DirGNN: **READY**
- Table 1 15-tool zero-shot: **READY** as Outcome B (no unification language)
- 30-tool encoder + routing_15 queries: **READY** as a protocol-labeled ablation, not as routing_30
- routing_30tools zero-shot: **NOT_SUPPORTED** (OOV vs frozen DAG vocab)
- Action-type figure: **NOT_SUPPORTED** for main paper (evidence NO SUPPORT)
- DirGNN vs published GCN as “directionality”: **NOT_SUPPORTED** (layer confound)

## 12. Recommended Paper Changes

Populate Table 2 and Table 1 from `paper_ready_results.md`. Table 1 main row = **15-tool encoder**, Dataset A (45-candidate protocol). Optional 30-tool-encoder row must state routing_15 queries. Omit action-type figure. DirGNN ablation = directed vs tied-in/out only.

## 13. Remaining Risks

- Single seed (42).
- DirGNN vs GCN is a different layer, not a pure W_in=W_out ablation against published GCN.
- Atomic stress CSVs come from `upgraded_data/routing_15tools`; compositional candidates come from each graph dataset.
- Dataset B 30-tool atomic corpus is 69 unique DAGs (one-nodes already present), not 45.
- CSV GED used in training is the structural surrogate in `train.py`, not exact graph_edit_distance.

## Results matrix

```
Experiment                         upgraded                              upgraded_v3                           Verified
------------------------------------------------------------------------------------------------------------------------
SBERT_FT_GED0                      R@1=0.9217 F1=0.9721 GED=0.2078       R@1=0.9273 F1=0.9654 GED=0.1564       YES
SBERT_FT_GED030                    R@1=0.9217 F1=0.9721 GED=0.2078       R@1=0.9273 F1=0.9654 GED=0.1564       YES
SBERT_FT_TIED                      R@1=0.9458 F1=0.9810 GED=0.1807       R@1=0.9364 F1=0.9595 GED=0.1509       YES
LEGR_ACTION_LATENT                 NO SUPPORT sil=-0.0114                NO SUPPORT sil=-0.0132                YES
LEGR_15TOOL_ZERO_SHOT_ATOMIC       Std=84.0 Lex=50.7 Con=45.1 Par=81.4   Std=92.8 Lex=62.1 Con=59.1 Par=93.0   YES
LEGR_30TOOL_ZERO_SHOT_ATOMIC       Std=63.8 Lex=31.5 Con=36.4 Par=65.3   Std=91.2 Lex=59.0 Con=51.6 Par=89.9   YES
DIRGNN_DIRECTED                    R@1=0.6747 F1=0.8080 GED=1.0693       R@1=0.7509 F1=0.8683 GED=0.8982       YES
DIRGNN_TIED_IN_OUT                 R@1=0.7892 F1=0.8989 GED=0.5964       R@1=0.8255 F1=0.9073 GED=0.6291       YES
LEGR_DEFAULT_GCN_30                R@1=0.8524 F1=0.9057 GED=0.4910       R@1=0.9182 F1=0.9573 GED=0.2109       YES
LEGR_DEFAULT_GCN_15                R@1=0.9561 F1=0.9937 GED=0.1318       R@1=0.9674 F1=0.9889 GED=0.1791       YES
```
