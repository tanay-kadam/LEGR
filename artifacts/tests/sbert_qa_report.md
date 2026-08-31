# QA report — Workstream A (SBERT-FT)

**Status: PASS**

## Independent tests

`tests/test_sbert_ft_baseline.py`: collate DAG strings, tied vs untied param counts, freeze policy, dummy InfoNCE batch, `drop_last` incomplete batch, config copy from fake checkpoint, 15/45 labelled-hash disjointness, 30-tool overlap measured as empty.

## Suite

`py -3 -m pytest -q` → **138 passed, 1 xfailed** (unrelated dataset-builder count, isolated to tmp_path).

## Leakage

TRAIN ∩ TEST labelled-DAG hashes: 15-tool ∅, 45-tool ∅, 30-tool ∅ on current `upgraded/` files.

## Metrics

No experiment metrics to recompute. Dummy loss is finite.

## Reproducibility

Commands in `artifacts/sbert_finetuned/repro.md`. Not executed (code-first).
