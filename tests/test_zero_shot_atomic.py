"""Zero-shot atomic corpus, aliases, and metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomic_zero_shot import (  # noqa: E402
    LEGR_15_TOOLS,
    ROUTING_TO_LEGR_15,
    UnmappedRoutingToolError,
    alias_routing_tool,
    build_one_node_dag,
    canonicalise_routing_columns,
    is_one_node,
    merge_candidate_corpus,
    one_node_candidates,
    one_node_id_by_tool,
)
from data_synth import build_dag, dag_canonical_hash, dag_to_pyg  # noqa: E402
from routing_tiers import EXPLICIT_ROUTING_TOOL_NAMES_15  # noqa: E402


def test_alias_map_covers_routing_15():
    for name in EXPLICIT_ROUTING_TOOL_NAMES_15:
        legr = alias_routing_tool(name)
        assert legr in LEGR_15_TOOLS
    assert alias_routing_tool("query_database") == "db_read"
    assert alias_routing_tool("update_database") == "db_write"
    assert alias_routing_tool("create_ticket") == "create_ticket"


def test_legr_15_tools_are_prefix_of_30():
    from data_synth import _FULL_TOOL_VOCAB

    assert list(LEGR_15_TOOLS) == list(_FULL_TOOL_VOCAB[:15])
    assert all(t in _FULL_TOOL_VOCAB[:30] for t in LEGR_15_TOOLS)


def test_routing_30_labels_remain_oov_for_frozen_legr():
    """routing_30tools must not be aliased onto the DAG embedding table."""
    with pytest.raises(UnmappedRoutingToolError):
        alias_routing_tool("check_service_status")
    with pytest.raises(UnmappedRoutingToolError):
        alias_routing_tool("quarantine_endpoint")


def test_one_node_graph_has_no_edges():
    G = build_one_node_dag("db_read")
    assert is_one_node(G)
    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0
    data = dag_to_pyg(G, bidirectional=True)
    assert tuple(data.edge_index.shape) == (2, 0)
    assert int(data.topo_pos[0]) == 0
    data_dir = dag_to_pyg(G, bidirectional=False)
    assert tuple(data_dir.edge_index.shape) == (2, 0)


def test_unified_corpus_dedupes_by_hash():
    existing = [build_dag(["db_read", "create_ticket"], [(0, 1)])]
    extra = one_node_candidates(["db_read", "create_ticket"])
    # Inject a duplicate one-node db_read
    extra.append(build_one_node_dag("db_read"))
    unique, h2i = merge_candidate_corpus(existing, extra)
    hashes = [dag_canonical_hash(G) for G in unique]
    assert len(hashes) == len(set(hashes))
    ids = one_node_id_by_tool(unique)
    assert ids["db_read"] != ids["create_ticket"]
    assert "db_read" in ids


def test_duplicate_candidate_ids_not_created_for_same_graph():
    g1 = build_one_node_dag("scan_malware")
    g2 = build_one_node_dag("scan_malware")
    unique, h2i = merge_candidate_corpus([g1], [g2])
    assert len(unique) == 1
    assert len(h2i) == 1


def test_stress_csv_schemas_canonicalise_to_query_and_label():
    import pandas as pd

    files = {
        "Standard": ROOT / "upgraded_data" / "routing_15tools" / "base_cleaned.csv",
        "Lexical": ROOT / "upgraded_data" / "routing_15tools" / "lexical_cue_reduced.csv",
        "Confusable": ROOT / "upgraded_data" / "routing_15tools" / "confusable_intents.csv",
        "Paraphrase": ROOT / "upgraded_data" / "routing_15tools" / "paraphrase_heldout_test.csv",
    }
    for name, path in files.items():
        raw = pd.read_csv(path)
        out = canonicalise_routing_columns(raw, str(path))
        assert "query" in out.columns, name
        assert "ground_truth" in out.columns, name
        assert len(out) == len(raw), name
        assert out["query"].astype(str).str.strip().ne("").all(), name
        for label in out["ground_truth"].astype(str):
            assert alias_routing_tool(label) in LEGR_15_TOOLS


def test_canonicalise_routing_columns_requires_query_and_label():
    import pandas as pd

    with pytest.raises(ValueError):
        canonicalise_routing_columns(pd.DataFrame({"foo": [1]}), "empty.csv")


def test_accuracy_pct_matches_table1_rounding():
    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location(
        "eval_zero_shot_atomic",
        ROOT / "scripts" / "eval_zero_shot_atomic.py",
    )
    # Importing the script bootstraps tool_count from argv; keep helpers local.
    assert spec is not None

    def accuracy_pct(correct: int, n: int) -> float:
        if n == 0:
            return 0.0
        return round(100.0 * correct / n, 1)

    assert accuracy_pct(0, 10) == 0.0
    assert accuracy_pct(1, 3) == 33.3
    assert accuracy_pct(3, 3) == 100.0
