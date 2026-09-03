# Revised NeurIPS manuscript

Upload this directory to Overleaf and set `main_new.tex` as the main document.

Required files for the revised paper:

- `main_new.tex`
- `neurips_2026.sty`
- `references.bib`
- `references_new.bib`
- `winning_gps_clusters_scatter.png`

The remaining PNG and TeX files are retained from the original bundle for provenance but are not referenced by `main_new.tex`.

The manuscript is configured for anonymous submission. Do not add the `final` option to `neurips_2026` until camera-ready preparation.

## Result sources

- 15-tool locked/fair evaluation: `../artifacts/legr_model_search/final_322dag_eval/summary.json`
- 15-tool two-stage evaluation: `../artifacts/legr_model_search/two_stage_v3_reranker/summary.json`
- 30-tool two-stage evaluation: `../artifacts/legr_model_search/two_stage_v3_reranker_30t_final/summary.json`
- 45-tool two-stage evaluation: `../artifacts/legr_model_search/two_stage_v3_reranker_45t_final/summary.json`
- Scaling analysis: `../artifacts/legr_model_search/TWO_STAGE_SCALING_RESULTS.md`
- Functional audit: `../artifacts/legr_model_search/action_cluster_seed42/FUNCTIONAL_CLUSTERING_ANALYSIS.md`
- Comprehensive revision dossier: `../artifacts/legr_model_search/NEURIPS_PAPER_REVISION_REFERENCE.md`

The local environment did not contain `pdflatex`, `latexmk`, or `tectonic`. Static validation confirmed balanced LaTeX environments/braces, complete citation keys, and present figure assets. Overleaf should be used for the final rendered-page and overfull-box check.
