# Action-type structure in LEGR latent space

**Status:** COMPUTED
**Evidence class:** NO SUPPORT
**Paper recommendation:** omit from main paper (appendix only if at all)

## Mapping

See `src/action_type_mapping.py`. 15-tool Tool-Bound branches are the
source of truth; remaining tools follow 45-tool Tool-Bound with the
documented lifecycle/access rules. Unmapped tools abort the run.

## Counts

{
  "mostly-write": 27,
  "mixed": 22,
  "mostly-orchestrate": 13,
  "mostly-read": 7
}

## Diagnostics (original embedding space, not t-SNE)

{
  "n": 69,
  "majority_frac": 0.391304347826087,
  "silhouette": -0.013153651729226112,
  "neighborhood_purity_k": 5,
  "neighborhood_purity": 0.3623188405797101,
  "ami_vs_1nn": 0.00788235636451138,
  "evidence": "NO SUPPORT",
  "embedding_space": "original",
  "tsne_not_used_for_metrics": true,
  "source": "C:\\Users\\tkadam\\LEGR\\experiment_runs\\20260831_1218\\upgraded_v3\\TASK4_DIRECTION_ABLATION\\LEGR_DEFAULT_GCN_30TOOL\\best_model.pt",
  "embedding_kind": "REAL_CHECKPOINT_EMBEDDINGS",
  "device": "cuda"
}

t-SNE random_state = 42.

Do not infer clusters from the plot alone.
