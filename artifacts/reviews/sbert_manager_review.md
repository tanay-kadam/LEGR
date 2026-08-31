# Manager review — Workstream A (fine-tuned SBERT)

**Status: PASS** (code and tests). Experiments not executed.

## Scientific validity

The 2×2 (untied text towers × lambda_ged ∈ {0, 0.30}) plus a tied-weights variant answers the attribution question. Cells 3–4 are not retrained. Metrics import `eval.compute_metrics`. Splits are `upgraded/upgraded_*tools/`.

Unavoidable differences are logged (text document tower, DAG-string collate, no GCN params).

## Engineering

- Reuses `TrainConfig`, CSV datasets, GED submatrix, `GraphAwareContrastiveLoss`.
- `encoder_cls` injection keeps tests off MiniLM downloads.
- Checkpoints go under `artifacts/sbert_finetuned/checkpoints/` only.

## Issues found and fixed

| Severity | Location | Issue | Fix |
|----------|----------|-------|-----|
| Major | `tests/test_sbert_ft_baseline.py` | Assumed 30-tool labelled-DAG leakage; current packaged files have **zero** labelled overlap | Test now asserts the measured overlap (`set()`) |
| Minor | `main` eval_after condition | Operator-precedence bug (`and`/`or`) | Rewritten as nested `if eval_after` |

## Integrity

No hard-coded table numbers. `report.md` metrics are empty pending a train run.

**Verdict:** PASS for implementation. Do not add a Table 2 row until Cell 1/2 `eval_metrics.json` exists.
