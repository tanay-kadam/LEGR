# Adversarial review (code-first)

Persona: skeptical ML conference reviewer. Scope: implementation and methodology, not unpublished numbers.

## BLOCKER

1. **No experimental numbers.** Fine-tuned SBERT, directed GNN, zero-shot atomic retrieval, and LEGR action-space diagnostics were not trained/evaluated on a real checkpoint. Any table cell filled now would be fabrication.
2. **No LEGR checkpoint in the tree** (`*.pt` gitignored). Workstreams B and C cannot produce paper figures/metrics until a matching `--tool_count` `best_model.pt` is supplied.
3. **`paper.tex` is absent** (gitignored). Paper integration cannot be compiled or cross-checked.

## MAJOR

1. **Paper claims a directed GNN; code was (and the default still is) bidirectional GCN + topo-rank features.** The new DirGNN is an additive experiment, not a retroactive description of published runs.
2. **Atomic Table 1 vs LEGR vocab.** 15-tool comparison requires the two-name alias map. 30-tool routing names are not comparable without a human-validated 27-entry map (out of scope).
3. **30-tool test size.** Packaged `upgraded/upgraded_30tools/test_topology_heldout.csv` is **332 rows / 30 DAGs**, not the 1200-row file mentioned in older notes. Stale tests assumed 1200. Report which file is used.
4. **`prepare_legr_30tool_dataset` unique-DAG count** is 137 vs coded 138 (xfail). Do not overwrite `upgraded/` to chase the old count.

## MINOR

1. Synthetic action-type t-SNE plotting works (matplotlib 3.11.1). Official `artifacts/action_latent_space/` has no figure because those embeddings are random, not a LEGR checkpoint.
2. Hard-negative SBERT protocol uses the same 0.5 cosine threshold on text documents — analogous but not identical to GNN scoring.
3. DirGNN vs published GCN is confounded by layer type; the controlled pair is `directed` vs `tied_in_out`.

## NIT

1. `TrainConfig` CLI still uses `store_true` for bools (pre-existing).
2. W&B remains a hard import in `train.py` (pre-existing).

## Disposition

BLOCKER items are inherent to the agreed **code-first** pass. They are not waived: they block `READY FOR PAPER`.
