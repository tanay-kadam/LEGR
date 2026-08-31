# Implementation notes — SBERT-FT

- Entry: `src/sbert_ft_baseline.py`
- Reuses `train.TrainConfig`, `_build_csv_train_val_datasets`, `_ged_submatrix`, `GraphAwareContrastiveLoss`, `eval.compute_metrics`
- Does not fork `train.main`
- Cells 3–4 must be filled from existing LEGR eval artifacts when available; they are not retrained here
