# FINAL REPORT — LEGR additional experiments (code-first)

## Executive Summary

Implemented four isolated experiment workstreams **without training or publishing numbers**. Default GCN LEGR training/eval paths remain compatible with existing checkpoints. pytest: **138 passed, 1 xfailed**.

## Experiment Results

No trained/evaluated LEGR or SBERT-FT metrics. Negative placeholder: synthetic action-space diagnostics on **random** embeddings report NO SUPPORT (not a LEGR finding).

| Experiment | Engineering | Manager | QA | Result | Paper Action |
| --- | --- | --- | --- | --- | --- |
| Fine-tuned SBERT 2×2 | done | PASS | PASS | not trained | stub Table 2 row only |
| Action-type latent space | done | PASS | PASS | pending checkpoint | omit until STRONG SUPPORT |
| Zero-shot atomic | done | PASS | PASS | pending checkpoint | stub Table 1; no unification claim |
| Directed vs undirected | done | PASS | PASS | not trained | stub ablation; default GCN unchanged |

## Fine-tuned Sentence-BERT

Two-tower `TextEncoder` over `dag_to_text`, same `TrainConfig` recipe and `GraphAwareContrastiveLoss`. Cells: λ_GED=0, λ_GED=0.30, optional `--tied`. Entry: `src/sbert_ft_baseline.py`. Interpretation deferred.

## Action-Type Latent Structure

Mapping in `src/action_type_mapping.py`. Cluster metrics in original space (`src/latent_space_metrics.py`). **Recommendation:** omit from the main paper until a real checkpoint run yields STRONG SUPPORT.

## Zero-Shot Atomic LEGR

15-tool unified corpus (compositional unique DAGs + 15 one-node graphs). Aliases documented. Dry-run corpus written. Stress-condition metrics **not measured**.

## Directed vs Undirected

Published code is bidirectional GCN. New `DirectedGraphEncoder` + `--graph_direction {directed,tied_in_out}`. Conclusion deferred.

## Engineering Validation

pytest 138 passed, 1 xfailed (`prepare_legr_30tool_dataset` train unique-DAG 137 vs coded 138; builder isolated to tmp_path so it cannot overwrite `upgraded/`). Verification also fixed zero-shot stress-CSV column mapping (`transformed_query`/`label`).

## Manager Reviews

PASS after fixing: leakage test vs actual 30-tool files; DirGNN instead of fake W_in=W_out on GCN; eval bidirectional flag; atomic id helper; SBERT eval_after condition.

## QA Validation

Independent unit tests for collate, mapping, one-node graphs, directed reverse edges, metric rounding, dummy contrastive loss.

## Reproducibility

See `artifacts/final/reproducibility_report.md` and per-workstream `repro.md`.

## Paper Changes

None applied (`paper.tex` not in repo). Recommended stubs: `artifacts/final/table_stubs.md`.

## Negative Results

- Synthetic (random) embeddings: NO SUPPORT for action-type clusters — expected, not a LEGR claim.
- Current packaged 30-tool labelled train/test overlap is **empty** (older 1200-row leakage notes do not match these files).

## Remaining Limitations

- No checkpoints, no train/eval numbers
- No 15-tool LEGR checkpoint for zero-shot
- matplotlib is installed; a synthetic t-SNE figure can be generated. Official `artifacts/action_latent_space/` still has no figure because that run used **random** embeddings, not a LEGR checkpoint.
- 30-tool routing vocab still incomparable
- DirGNN vs published GCN is confounded unless the tied DirGNN run is also trained

## Final Status

**NOT READY FOR PAPER**

Remaining:

1. Train SBERT Cells 1–2 (and optional tied) and evaluate Table 2 metrics
2. Train DirGNN `directed` and `tied_in_out`; evaluate ablation
3. Provide 15- and 30-tool `best_model.pt`
4. Run action-type analysis and zero-shot atomic on frozen encoders
5. Fill table stubs only from those artifacts
6. Resolve adversarial BLOCKERs (numbers, checkpoints, paper source)
