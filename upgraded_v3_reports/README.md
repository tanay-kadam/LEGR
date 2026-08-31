# upgraded_v3 — split fixes (requires retraining)

`upgraded_v3/` builds on the v2 query regeneration and additionally changes the
**splits**. That invalidates `checkpoints_{15,30,45}tools/`, so v3 requires
retraining LEGR and re-running every baseline.

Use `upgraded_v2/` if you want the query-quality fix with no retraining: it is
row-for-row identical to `upgraded/` except for the `query` column.

| Tree | Queries | Splits | Retraining |
|---|---|---|---|
| `upgraded/` | v1 | v1 | — (current results) |
| `upgraded_v2/` | regenerated | identical to v1 | not needed |
| `upgraded_v3/` | regenerated | **changed** | **required** |

## Change 1 — diamond is now genuinely held out at 15 and 30 tools

Previously only the 45-tool split held the diamond family out. At 15 and 30 tools
the diamond family appeared in **both** train and test, so "held-out topology" was
not true there:

| Tier | v1: diamond in train | v3: diamond in train |
|---|---|---|
| 15 tools | yes — 48 rows | **no** |
| 30 tools | yes — 52 rows | **no** |
| 45 tools | no | no |

Every row whose whole-graph shape is the canonical 4-node diamond was moved out
of train and dev into test: 70 rows at 15 tools, 68 at 30 tools. Verified three
ways (`scripts/audit_topology_overlap.py --root upgraded_v3 --exclude-single-node`):
the `diamond` family label is absent from train, the diamond *shape* is absent
from train under any family label, and multi-node labelled-DAG overlap between
train and test is 0.

Test diamond content after the move: 118 rows / 10 unique DAGs at 15 tools,
84 / 13 at 30 tools, 300 / 24 at 45 tools.

**Still shared:** only the diamond family was held out, so other families remain
in both splits — 13 of 14 test shapes at 15 tools, 6 of 7 at 30 tools, still 0 of 5
at 45 tools. The 45-tool split remains the only one where *no* test topology
appears in training. If the paper needs that property at every tier, more
families have to be held out; say so and it is a one-line change to the builder.

## Change 2 — single-node graphs across the full tool vocabulary

Single-node graphs already existed in `upgraded/`, but too sparsely to evaluate
and entirely absent from test:

| Tier | v1 coverage | v1 in test | v3 coverage | v3 in test |
|---|---|---|---|---|
| 15 tools | 2 / 15 tools, 58 rows | **0** | 15 / 15 | 75 rows |
| 30 tools | 1 / 30 (+1 in dev), 20 rows | **0** | 30 / 30 | 150 rows |
| 45 tools | 8 / 45 tools, 120 rows | **0** | 45 / 45 | 225 rows |

v3 gives every tool in the tier a single-node DAG with rows in train (12/tool),
dev (3/tool) and test (5/tool), counting rows that already existed. Tools that
already had a single-node DAG keep its original `dag_id` rather than getting a
duplicate under a new id. New rows carry `source=single_node_v3` and
`topo_family=single_node`.

### Why train and test share these DAGs on purpose

For single-node DAGs the same labelled DAG appears in train and test. That is
deliberate, and it is not the leakage the audit flags elsewhere. In a retrieval
comparison against the taxonomy router, the candidate corpus is *supposed* to
contain one entry per tool, and the router likewise knows every tool up front.
What must be held out is the query phrasing, and the v2 generator guarantees
train/test phrasings are disjoint (verified: 0 shared phrasings, 0 shared entity
values).

Because these DAGs would otherwise mask the multi-node figures, the topology
audit takes `--exclude-single-node` for the topology claim.

## Blocker for the taxonomy comparison: the vocabularies disagree at 30 tools

The point of change 2 is to compare functional-taxonomy routing against LEGR
single-node retrieval. That comparison is only meaningful on tools whose names
line up between the routing benchmark and the LEGR vocabulary:

| Tier | Names matching | Gap |
|---|---|---|
| 15 tools | 13 / 15 | `query_database`/`update_database` vs `db_read`/`db_write` |
| 45 tools | 43 / 45 | same two |
| **30 tools** | **3 / 30** | routing uses verbose names (`check_service_status`, `create_support_ticket`) where LEGR uses short ones (`check_status`, `create_ticket`) |

At 15 and 45 tools this is two renames. At 30 tools the routing tier uses a
different naming convention throughout; the tools look semantically equivalent
(`approve_access_request` ≈ `approve_access`, `log_compliance_event` ≈
`log_audit_event`) but establishing that requires a hand-checked 27-entry alias
map, and a few are genuinely ambiguous (`inspect_security_alerts` could map to
`scan_malware` or `acknowledge_alert`). That mapping is **not** included here — it
needs a human decision, and guessing it would put unverifiable claims in a table.

The tightest version of the experiment would score the taxonomy router and LEGR
single-node retrieval on *the same queries* by importing
`upgraded_data/routing_{15,45}tools/base_cleaned.csv` into the corpus as
single-node DAGs. v3's generated single-node queries are a looser comparison
(same tools, different phrasings). Both are viable; the shared-query version is
stronger and is the natural next step at 15 and 45 tools.

## Reproducing

```powershell
python scripts/build_upgraded_v3.py
python scripts/regenerate_queries_v2.py --source-root upgraded_v3 `
    --out-root upgraded_v3 --report-dir upgraded_v3_reports
python scripts/audit_topology_overlap.py --root upgraded_v3 --exclude-single-node
python scripts/audit_single_node_coverage.py --roots upgraded upgraded_v3
python scripts/verify_queries_v2.py --v2-root upgraded_v3 `
    --report-dir upgraded_v3_reports --skip-parity
```

Row counts: 15 tools 2922/587/737, 30 tools 1692/403/550, 45 tools 4316/585/1325
(train/dev/test). Query quality matches v2 — 0% entity-normalised redundancy,
0 tool-name leaks, 0 phrasing or entity overlap with train, in all nine files.
