"""Tests for src/dag_extract.py cycle repair and structural utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dag_extract import (
    break_cycles_min_confidence,
    check_structural_validity,
    detect_cycles,
    enforce_dag,
    parse_extraction_response,
    results_to_corpus_row,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

TOOLS_3 = ["db_read", "db_write", "send_notification"]


def _make_edges(pairs, confidences=None):
    """Helper to build edge dicts from (source, target) pairs."""
    if confidences is None:
        confidences = [50] * len(pairs)
    return [
        {"source": s, "target": t, "confidence": c}
        for (s, t), c in zip(pairs, confidences)
    ]


# ── 3-cycle drops lowest-confidence edge ──────────────────────────────────

class TestBreakCyclesMinConfidence:
    def test_3_cycle_drops_lowest_confidence(self):
        """A→B(90), B→C(80), C→A(30) should drop C→A."""
        edges = _make_edges(
            [(0, 1), (1, 2), (2, 0)],
            [90, 80, 30],
        )
        clean, removed = break_cycles_min_confidence(TOOLS_3, edges)

        assert len(removed) == 1
        assert removed[0]["source"] == 2
        assert removed[0]["target"] == 0
        assert removed[0]["confidence"] == 30

        assert len(clean) == 2
        assert not detect_cycles(TOOLS_3, clean)

    def test_3_cycle_drops_lowest_when_tied_picks_first(self):
        """All edges confidence=50: should still produce a DAG."""
        edges = _make_edges(
            [(0, 1), (1, 2), (2, 0)],
            [50, 50, 50],
        )
        clean, removed = break_cycles_min_confidence(TOOLS_3, edges)

        assert len(removed) >= 1
        assert not detect_cycles(TOOLS_3, clean)

    def test_acyclic_graph_unchanged(self):
        """An already-acyclic chain should return unchanged."""
        edges = _make_edges(
            [(0, 1), (1, 2)],
            [90, 85],
        )
        clean, removed = break_cycles_min_confidence(TOOLS_3, edges)

        assert len(removed) == 0
        assert len(clean) == 2
        assert clean[0]["source"] == 0 and clean[0]["target"] == 1
        assert clean[1]["source"] == 1 and clean[1]["target"] == 2

    def test_empty_graph(self):
        clean, removed = break_cycles_min_confidence(TOOLS_3, [])
        assert len(clean) == 0
        assert len(removed) == 0


# ── enforce_dag ───────────────────────────────────────────────────────────

class TestEnforceDAG:
    def test_cyclic_returns_had_cycle_true(self):
        edges = _make_edges([(0, 1), (1, 2), (2, 0)], [90, 80, 30])
        clean, removed, had_cycle = enforce_dag(TOOLS_3, edges)
        assert had_cycle is True
        assert len(removed) == 1

    def test_acyclic_returns_had_cycle_false(self):
        edges = _make_edges([(0, 1), (1, 2)], [90, 85])
        clean, removed, had_cycle = enforce_dag(TOOLS_3, edges)
        assert had_cycle is False
        assert len(removed) == 0


# ── check_structural_validity ─────────────────────────────────────────────

class TestCheckStructuralValidity:
    def test_valid_chain(self):
        result = check_structural_validity(TOOLS_3, [(0, 1), (1, 2)])
        assert result["is_dag"] is True
        assert result["has_cycle"] is False
        assert result["is_connected"] is True

    def test_cycle_detected(self):
        result = check_structural_validity(TOOLS_3, [(0, 1), (1, 2), (2, 0)])
        assert result["is_dag"] is False
        assert result["has_cycle"] is True

    def test_disconnected(self):
        result = check_structural_validity(TOOLS_3, [(0, 1)])
        assert result["is_dag"] is True
        assert result["is_connected"] is False


# ── parse_extraction_response ─────────────────────────────────────────────

class TestParseExtractionResponse:
    def test_parse_with_confidence(self):
        raw = '{"tools": ["db_read", "db_write"], "edges": [{"source": 0, "target": 1, "confidence": 85}]}'
        tools, edges = parse_extraction_response(raw)
        assert tools == ["db_read", "db_write"]
        assert len(edges) == 1
        assert edges[0]["confidence"] == 85

    def test_parse_fallback_list_edges(self):
        raw = '{"tools": ["db_read", "db_write"], "edges": [[0, 1]]}'
        tools, edges = parse_extraction_response(raw)
        assert tools == ["db_read", "db_write"]
        assert len(edges) == 1
        assert edges[0]["confidence"] == 50

    def test_parse_invalid_json(self):
        tools, edges = parse_extraction_response("this is not json")
        assert tools == []
        assert edges == []

    def test_parse_filters_invalid_tools(self):
        raw = '{"tools": ["db_read", "FAKE_TOOL"], "edges": []}'
        tools, edges = parse_extraction_response(raw)
        assert tools == ["db_read"]

    def test_parse_with_code_fences(self):
        raw = '```json\n{"tools": ["db_read", "db_write"], "edges": [{"source": 0, "target": 1, "confidence": 75}]}\n```'
        tools, edges = parse_extraction_response(raw)
        assert tools == ["db_read", "db_write"]
        assert edges[0]["confidence"] == 75


# ── Schema round-trip through build_dag ───────────────────────────────────

class TestSchemaRoundTrip:
    def test_corpus_row_schema(self):
        result = {
            "tools": ["db_read", "db_write", "send_notification"],
            "edges": [(0, 1), (1, 2)],
            "confidence_per_edge": [90, 85],
            "had_cycle": False,
            "removed_edges": [],
            "latency_s": 1.5,
            "parse_failure": False,
            "topo_family": "chain_short",
        }
        row = results_to_corpus_row(result, query="test query", dag_id=42)

        assert row["query"] == "test query"
        assert row["dag_id"] == 42
        assert row["source"] == "extracted"
        assert row["tools"] == "db_read;db_write;send_notification"
        assert row["edges"] == "0->1;1->2"
        assert row["dag_text"] != ""

    def test_corpus_row_roundtrip_via_build_dag(self):
        """Edges survive the corpus CSV schema and can rebuild a valid DAG."""
        from data_synth import build_dag

        result = {
            "tools": ["db_read", "db_write"],
            "edges": [(0, 1)],
            "confidence_per_edge": [90],
            "had_cycle": False,
            "removed_edges": [],
            "latency_s": 0.5,
            "parse_failure": False,
            "topo_family": "chain_short",
        }
        row = results_to_corpus_row(result, query="q", dag_id=1)

        tools = row["tools"].split(";")
        import re
        edge_pairs = []
        for part in row["edges"].split(";"):
            m = re.match(r"(\d+)->(\d+)", part)
            if m:
                edge_pairs.append((int(m.group(1)), int(m.group(2))))

        G = build_dag(tools, edge_pairs)
        import networkx as nx
        assert nx.is_directed_acyclic_graph(G)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1
