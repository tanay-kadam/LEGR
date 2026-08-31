# QA report — Workstream C (zero-shot atomic)

**Status: PASS**

## Independent tests

`tests/test_zero_shot_atomic.py`: 15-tool alias coverage, OOV rejection, empty-edge one-node pyg, corpus hash dedupe, Table-1 `accuracy_pct` rounding, **stress-CSV column canonicalisation** (`query`/`ground_truth` vs `transformed_query`/`label`).

`tests/test_one_node_gnn.py`: GCN and DirGNN isolated nodes, mixed batches, two-node empty-edge graphs.

## Dry run

`python scripts/eval_zero_shot_atomic.py --tool_count 15 --dry_run` wrote `candidate_corpus.csv` and definition JSON. No encoder metrics (no checkpoint).

## Leakage / OOV

Script refuses to `register_tools` routing-only names into a frozen embedding table.
