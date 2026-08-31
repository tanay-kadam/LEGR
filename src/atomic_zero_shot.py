"""
atomic_zero_shot.py — Frozen LEGR one-node candidates for atomic queries.

Pure helpers (no checkpoint I/O). Routing labels are aliased onto the LEGR
15-tool vocabulary. Unmapped labels raise ``KeyError`` rather than expanding
the frozen embedding table.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx
import pandas as pd

from data_synth import _FULL_TOOL_VOCAB, dag_canonical_hash, dag_to_pyg
from routing_tiers import EXPLICIT_ROUTING_TOOL_NAMES_15

# Same candidate order as ``src/main.py`` ``_canonicalise_dataset_columns``.
QUERY_COLUMN_CANDIDATES = ("query", "transformed_query", "text", "utterance")
LABEL_COLUMN_CANDIDATES = ("ground_truth", "label", "tool", "target")

# 15-tool routing → LEGR. Identity for the 13 shared names.
ROUTING_TO_LEGR_15: dict[str, str] = {
    "query_database": "db_read",
    "update_database": "db_write",
}
for _name in EXPLICIT_ROUTING_TOOL_NAMES_15:
    ROUTING_TO_LEGR_15.setdefault(_name, _name)

LEGR_15_TOOLS: tuple[str, ...] = tuple(_FULL_TOOL_VOCAB[:15])


class UnmappedRoutingToolError(KeyError):
    """Raised when a routing ground-truth name has no LEGR alias."""


def alias_routing_tool(name: str, alias_map: dict[str, str] | None = None) -> str:
    """Map a routing-benchmark tool name to a LEGR tool name."""
    table = alias_map if alias_map is not None else ROUTING_TO_LEGR_15
    key = str(name).strip()
    if key not in table:
        raise UnmappedRoutingToolError(
            f"No LEGR alias for routing tool {key!r}. "
            "Refusing to register an OOV name into a frozen checkpoint."
        )
    return table[key]


def build_one_node_dag(tool: str) -> nx.DiGraph:
    """Trivial one-node execution graph: no edges, no self-loops added here."""
    G = nx.DiGraph()
    G.add_node(0, tool=tool)
    return G


def one_node_candidates(tools: Sequence[str] | None = None) -> list[nx.DiGraph]:
    """One isolated node per LEGR tool (default: first 15)."""
    names = list(tools) if tools is not None else list(LEGR_15_TOOLS)
    return [build_one_node_dag(t) for t in names]


def dag_tools(G: nx.DiGraph) -> list[str]:
    return [G.nodes[n]["tool"] for n in sorted(G.nodes())]


def is_one_node(G: nx.DiGraph) -> bool:
    return G.number_of_nodes() == 1 and G.number_of_edges() == 0


def merge_candidate_corpus(
    existing: Iterable[nx.DiGraph],
    extra: Iterable[nx.DiGraph],
) -> tuple[list[nx.DiGraph], dict[str, int]]:
    """Deduplicate DAGs by labelled canonical hash. Extra graphs are appended.

    Returns (unique_dags, hash_to_id).
    """
    unique: list[nx.DiGraph] = []
    hash_to_id: dict[str, int] = {}
    for G in list(existing) + list(extra):
        h = dag_canonical_hash(G)
        if h not in hash_to_id:
            hash_to_id[h] = len(unique)
            unique.append(G)
    return unique, hash_to_id


def one_node_id_by_tool(unique_dags: Sequence[nx.DiGraph]) -> dict[str, int]:
    """Map LEGR tool name → candidate id for one-node graphs only."""
    out: dict[str, int] = {}
    for i, G in enumerate(unique_dags):
        if is_one_node(G):
            out[G.nodes[0]["tool"]] = i
    return out


def pyg_one_node(tool: str, bidirectional: bool = True):
    """PyG Data for an isolated tool node (empty ``edge_index``)."""
    return dag_to_pyg(build_one_node_dag(tool), bidirectional=bidirectional)


def canonicalise_routing_columns(df: pd.DataFrame, dataset_path: str = "") -> pd.DataFrame:
    """Map routing CSVs onto canonical ``query`` / ``ground_truth`` columns.

    Standard ``base_cleaned.csv`` uses those names. Lexical / Confusable /
    Paraphrase files use ``transformed_query`` / ``label`` (same as Table 1).
    """
    query_col = next((c for c in QUERY_COLUMN_CANDIDATES if c in df.columns), None)
    label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns), None)
    if query_col is None or label_col is None:
        raise ValueError(
            f"Could not infer query/label columns for {dataset_path or 'dataframe'}. "
            f"Found columns: {list(df.columns)}"
        )
    out = df.copy()
    if query_col != "query":
        out = out.rename(columns={query_col: "query"})
    if label_col != "ground_truth":
        out = out.rename(columns={label_col: "ground_truth"})
    return out.dropna(subset=["query", "ground_truth"]).reset_index(drop=True)
