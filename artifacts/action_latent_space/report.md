# Action-type structure in LEGR latent space

**Status:** PENDING_CHECKPOINT (synthetic embeddings)
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
  "silhouette": -0.010086598806083202,
  "neighborhood_purity_k": 5,
  "neighborhood_purity": 0.39333333333333337,
  "ami_vs_1nn": -0.07258381543543473,
  "evidence": "NO SUPPORT",
  "embedding_space": "original",
  "tsne_not_used_for_metrics": true,
  "source": "synthetic_random"
}

t-SNE random_state = 42.

Do not infer clusters from the plot alone.

## Note

This run used random embeddings because no checkpoint was provided.
The evidence class is therefore not a scientific result about LEGR.
