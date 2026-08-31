# upgraded_v2 — regenerated query corpus

`upgraded_v2/` is a drop-in replacement for `upgraded/` in which **only the
`query` column differs**. Every `dag_id`, `tools`, `edges`, `dag_text`,
`topo_family`, `source` and `split` value is byte-identical, in the same row
order, across all three tiers.

That makes v2 a controlled A/B on phrasing alone: an already-trained LEGR
checkpoint can be scored on it with no retraining, and any change in Recall@k is
attributable to the query text and nothing else.

## Structure

Mirrors `upgraded/` exactly — 3 directories, 9 files, no extras:

```
upgraded_v2/
  upgraded_15tools/{train,dev,test_topology_heldout}.csv
  upgraded_30tools/{train,dev,test_topology_heldout}.csv
  upgraded_45tools/{train,dev,test_topology_heldout}.csv
```

Reports and provenance are kept in this directory (`upgraded_v2_reports/`) rather
than inside `upgraded_v2/`, to preserve that parity.

## What was wrong with the v1 queries

Measured, not assumed. Numbers reproduce via `scripts/verify_queries_v2.py`.

| Defect | Evidence in v1 |
|---|---|
| Rows were entity re-fills of a few base sentences | 30-tool train: 1396 rows collapse to **601** distinct phrasings once entity names are normalised — 4.39 per DAG. 15-tool train 28.6% redundant, 45-tool train 26.4%. |
| Literal tool names leaked into queries | 203 rows at 45-tool train, 76 at 45-tool test, 49 at 30-tool train — a free lexical match for BM25 and S-BERT. |
| Entity names were shared across splits | Every held-out split reused 29–47 of train's entity values, so entity identity was memorisable. |

Reported dataset size therefore overstated the real linguistic variety,
substantially so at 30 tools.

## What v2 changes

1. **Sampling happens in skeleton space.** Queries are built with `{entity}`
   placeholders still unresolved, deduplicated *there*, and only then filled. A
   name swap can no longer count as a new example. Uniqueness is enforced across
   all splits of a tier, so train/held-out phrasing overlap is zero by
   construction.
2. **Deterministic leakage gate.** Any candidate containing a tool name or its
   de-underscored form is rejected. The gate runs on the filled query as well as
   the skeleton, because substitution can manufacture a tool name that the
   skeleton did not contain — a server named `cache-*` turns
   `invalidate {server}'s cache` into the literal tool `invalidate_cache`.
3. **Disjoint entity pools per split.** Pools were enlarged (36 users, 24
   servers, 18 departments, 24 tickets, 24 orders) and split into
   non-overlapping train/dev/test slices.
4. **Wider surface inventory.** 44 openers, 18 sequential / 15 parallel / 13
   merge / 11 final connectors, 18 optional closers, 14 single-node patterns.
5. **Grammar fixes.** Possessives collapse correctly for values ending in "s"
   (`Sales' log`, not `Sales's log`). Connectors requiring a gerund were dropped
   rather than carried over — v1's `". End with "` and `", followed by "`
   produced "End with apply the tag".

Topology wording is unchanged in kind: v2 reuses v1's longest-path layering, so
sequential edges still read as dependent steps, sibling branches as parallel, and
fan-in nodes as merges. A v2 query describes the same structure as its v1
counterpart; only the surface form differs.

## Result

| Metric | v1 | v2 |
|---|---|---|
| Redundant rows (entity-normalised) | 26.4–58.7% | **0.0%** all 9 files |
| Literal tool-name leaks | up to 203 per file | **0** all 9 files |
| Entity values shared with train | 29–47 per held-out split | **0** |
| Phrasing shared with train | 0 | 0 |
| Rows / DAGs / splits | — | unchanged |

## Reproducing

```powershell
python scripts/regenerate_queries_v2.py          # writes upgraded_v2/
python scripts/verify_queries_v2.py              # parity + redundancy report
```

Deterministic given `--seed` (default 42); the RNG is keyed per
(seed, tier, split file, dag_id). Requires only `pandas` — the phrase library is
read out of `data_synth.py` with `ast` specifically so regeneration does not pull
in torch.

## Caveats

- The phrase inventory is still hand-written, so vocabulary and scenario space
  remain bounded by the template library. v2 removes *redundancy*, not the fact
  that queries are synthetic.
- Tool-to-topology assignment is untouched: `gen_diamond` still drops four
  uniformly-sampled tools onto a fixed diamond, so topology remains semantically
  arbitrary. That is the separate, larger fix (type-constrained edges) and it
  would require retraining every model at every tier.
- `hard_negatives.csv` and `test_topology_heldout_1200.csv` are not present in
  the current `upgraded/` tree, so they are not generated. The regeneration
  script still handles `hard_negatives.csv` (remapping each positive DAG's query)
  if it is restored to the source.
