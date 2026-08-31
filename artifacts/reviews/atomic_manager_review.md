# Manager review — Workstream C (zero-shot atomic)

**Status: PASS** (code and tests). Frozen eval not executed (no checkpoint).

## Scientific validity

15-tool routing only. Aliases: `query_database→db_read`, `update_database→db_write`. Unmapped names raise (no 30-tool map, no `register_tools` OOV). One-node graphs have empty edge lists. Unified corpus = compositional unique DAGs + 15 one-nodes, hash-deduped.

## One-node GNN

Empty `edge_index` is valid. GCNConv may add self-loops internally (documented). Directed encoder uses `W_self` only.

## Issues found and fixed

| Severity | Location | Issue | Fix |
|----------|----------|-------|-----|
| Major | `one_node_id_by_tool` | Duplicate/conflicting return paths | Simplified to a single dict |
| Minor | `--tool_count` ≠ 15 | Would silently mis-align vocabs | Script raises |
| Major | `eval_zero_shot_atomic.py` stress CSVs | Lexical/Confusable/Paraphrase use `transformed_query`/`label`; script only read `query`/`ground_truth` | `canonicalise_routing_columns` (same candidates as Table 1 `main.py`) |

**Verdict:** PASS for implementation. No Table 1 row until per-condition `accuracy_pct` is measured.
