# Fresh SBERT FT — `upgraded` only

Train: `upgraded/upgraded_{N}tools/train.csv`
Val: `dev.csv` (early stopping)
Test: `test_topology_heldout.csv` (independent reload of best_model.pt)

| Tools | Variant | Recall@1 | Tool-Set F1 | Mean GED | n_eval | unique DAGs | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | ged_0 | 1.0000 | 1.0000 | 0.0000 | 592 | 30 | VERIFIED |
| 15 | ged_030 | 1.0000 | 1.0000 | 0.0000 | 592 | 30 | VERIFIED |
| 15 | tied_weights | 1.0000 | 1.0000 | 0.0000 | 592 | 30 | VERIFIED |
| 30 | ged_0 | 0.9217 | 0.9721 | 0.2078 | 332 | 30 | VERIFIED |
| 30 | ged_030 | 0.9217 | 0.9721 | 0.2078 | 332 | 30 | VERIFIED |
| 30 | tied_weights | 0.9458 | 0.9810 | 0.1807 | 332 | 30 | VERIFIED |
| 45 | ged_0 | 0.9964 | 0.9989 | 0.0073 | 1100 | 78 | VERIFIED |
| 45 | ged_030 | 0.9964 | 0.9989 | 0.0073 | 1100 | 78 | VERIFIED |
| 45 | tied_weights | 0.9955 | 0.9984 | 0.0118 | 1100 | 78 | VERIFIED |
