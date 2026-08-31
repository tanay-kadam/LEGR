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
  "mixed": 16,
  "mostly-write": 11,
  "mostly-orchestrate": 2,
  "mostly-read": 1
}

## Diagnostics (original embedding space, not t-SNE)

{
  "n": 30,
  "majority_frac": 0.5333333333333333,
  "silhouette": -0.011417719535529613,
  "neighborhood_purity_k": 5,
  "neighborhood_purity": 0.46,
  "ami_vs_1nn": -0.005613492656602941,
  "evidence": "NO SUPPORT",
  "embedding_space": "original",
  "tsne_not_used_for_metrics": true,
  "source": "C:\\Users\\tkadam\\LEGR\\experiment_runs\\20260831_1218\\upgraded\\TASK4_DIRECTION_ABLATION\\LEGR_DEFAULT_GCN_30TOOL\\best_model.pt",
  "embedding_kind": "REAL_CHECKPOINT_EMBEDDINGS",
  "device": "cuda"
}

t-SNE random_state = 42.

Do not infer clusters from the plot alone.
