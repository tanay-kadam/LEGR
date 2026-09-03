# Final LEGR NeurIPS Manuscript Handoff

## Completed work

- Trained a new 30-tool LEGR-V3 checkpoint with the same V3 configuration as the established 15-tool model.
- Trained a new 45-tool LEGR-V3 checkpoint with the same configuration.
- Ran the SBERT-FT → V3 two-stage reranker at 30 and 45 tools with seeds 42, 123, and 2026.
- Evaluated SBERT-FT, standalone V3, and the two-stage system on each tier's complete twin-filled held-out gallery.
- Computed deterministic, tie-aware, same-tool-set, hard-pair, and DAG-clustered paired-bootstrap statistics.
- Verified that protected datasets, existing checkpoints, and existing source files did not change during each evaluation.
- Rewrote the manuscript as an anonymous NeurIPS paper and included the requested functional-cluster scatterplot.
- Added corrected GPT-OSS 120B and Llama 3.2 DAG-generation results for 15, 30, and 45 tools.
- Packaged the complete LaTeX directory for Overleaf.

## Central scaling result

| Tools | Gallery | Queries | SBERT-FT R@1 | V3 R@1 | Two-stage R@1 | Two-stage − SBERT | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 322 | 300 | 0.2133 | 0.2100 | 0.2756 | +0.0622 | [-0.0356, 0.1644] |
| 30 | 455 | 420 | 0.1762 | 0.2310 | 0.3246 | +0.1484 | [0.0563, 0.2405] |
| 45 | 650 | 600 | 0.1700 | **0.4183** | 0.3611 | +0.1911 | [0.1244, 0.2589] |

The two-stage system preserves SBERT-FT tool-set F1 exactly at every tier: 0.9689, 0.9747, and 0.9747. Its gain over SBERT-FT is statistically supported at 30 and 45 tools. At 45 tools, standalone V3 is the strongest exact-plan retriever, while two-stage retains the strongest semantic guarantee.

## Corrected LLM DAG-generation results

| Model | Tools | Tool F1 | Exact match | Mean GED | Parse failures | Cyclic | Mean / p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-OSS 120B | 15 | 0.9450 | 0.76 | 0.44 | 0 | 0 | 3.047 / 4.468 s |
| GPT-OSS 120B | 30 | 0.9517 | 0.80 | 0.62 | 0 | 0 | 3.268 / 4.777 s |
| GPT-OSS 120B | 45 | 0.9486 | 0.76 | 0.80 | 0 | 0 | 3.106 / 4.609 s |
| Llama 3.2 3B | 15 | 0.7911 | 0.08 | 6.11 | 0 | 6 | 4.485 / 15.301 s |
| Llama 3.2 3B | 30 | 0.7769 | 0.06 | 4.80 | 2 | 4 | 4.610 / 15.038 s |
| Llama 3.2 3B | 45 | 0.6844 | 0.04 | 4.58 | 5 | 2 | 5.080 / 15.685 s |

The paper keeps generative exact match separate from retrieval Recall@1 because these are different output protocols.

## Files to use

- Manuscript source: `latex/main_new.tex`
- Overleaf package: `artifacts/legr_model_search/LEGR_NEURIPS_UPDATED_LATEX.zip`
- Scaling analysis: `artifacts/legr_model_search/TWO_STAGE_SCALING_RESULTS.md`
- 30-tool evaluation: `artifacts/legr_model_search/two_stage_v3_reranker_30t_final/summary.json`
- 45-tool evaluation: `artifacts/legr_model_search/two_stage_v3_reranker_45t_final/summary.json`
- 30-tool V3 checkpoint: `artifacts/legr_model_search/v3_scale/legr_setgnn_tied_no_ged_30t_s42/best_model.pt`
- 45-tool V3 checkpoint: `artifacts/legr_model_search/v3_scale/legr_setgnn_tied_no_ged_45t_s42/best_model.pt`

## Validation status

- Evaluator/reranker tests: 8 passed.
- 30-tool integrity report: no changed, missing, or added protected files.
- 45-tool integrity report: no changed, missing, or added protected files.
- LaTeX citations: all citation keys resolved.
- LaTeX environments and braces: balanced.
- Referenced figure files: present.
- Anonymous NeurIPS style: enabled.

No local TeX compiler is installed, so final rendering and page-count/overfull-box inspection must be performed in Overleaf. Set `main_new.tex` as the main document.
