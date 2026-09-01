"""
topology_templates.py — Campaign v4 Topology Family Definitions
================================================================

Defines all topology families used in Campaign v4, including:
  - Training families (seen during training)
  - Held-out test families (diamond, asymmetric_fork_join)
  - Edge generators for each family
  - Topology classification from graph structure
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# ═══════════════════════════════════════════════════════════════════════════
#  Family definitions and edge generators
# ═══════════════════════════════════════════════════════════════════════════

def _edges_single_node() -> List[Tuple[int, int]]:
    return []


def _edges_chain(n: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _edges_fanout(n_children: int) -> List[Tuple[int, int]]:
    return [(0, i + 1) for i in range(n_children)]


def _edges_fanin(n_parents: int) -> List[Tuple[int, int]]:
    sink = n_parents
    return [(i, sink) for i in range(n_parents)]


def _edges_diamond() -> List[Tuple[int, int]]:
    """0→1, 0→2, 1→3, 2→3"""
    return [(0, 1), (0, 2), (1, 3), (2, 3)]


def _edges_asymmetric_fork_join() -> List[Tuple[int, int]]:
    """Diamond + tail: 0→1, 0→2, 1→3, 2→3, 3→4"""
    return [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]


def _edges_hourglass() -> List[Tuple[int, int]]:
    """Fan-in then fan-out: 0→2, 1→2, 2→3, 2→4"""
    return [(0, 2), (1, 2), (2, 3), (2, 4)]


def _edges_y_shape() -> List[Tuple[int, int]]:
    """Two sources merge, then chain: 0→2, 1→2, 2→3"""
    return [(0, 2), (1, 2), (2, 3)]


def _edges_inverted_y() -> List[Tuple[int, int]]:
    """Chain then fan-out: 0→1, 1→2, 1→3"""
    return [(0, 1), (1, 2), (1, 3)]


def _edges_fork_join_generic(n_branches: int = 2) -> List[Tuple[int, int]]:
    """Generic: root fans out to N branches, all merge at sink."""
    edges = [(0, i + 1) for i in range(n_branches)]
    sink = n_branches + 1
    edges += [(i + 1, sink) for i in range(n_branches)]
    return edges


def _edges_w_shape() -> List[Tuple[int, int]]:
    """Two parallel chains merging: 0→1, 2→3, 1→4, 3→4"""
    return [(0, 1), (2, 3), (1, 4), (3, 4)]


def _edges_multi_branch_independent() -> List[Tuple[int, int]]:
    """Root → 3 children, one child has sub-chain: 0→1, 0→2, 0→3, 3→4"""
    return [(0, 1), (0, 2), (0, 3), (3, 4)]


def _edges_wide_fanout() -> List[Tuple[int, int]]:
    """Root → 4 children: 0→1, 0→2, 0→3, 0→4"""
    return [(0, 1), (0, 2), (0, 3), (0, 4)]


def _edges_wide_fanout_deep() -> List[Tuple[int, int]]:
    """Root → 3 children, last child has sub-chain: 0→1, 0→2, 0→3, 3→4, 4→5"""
    return [(0, 1), (0, 2), (0, 3), (3, 4), (4, 5)]


def _edges_long_chain_branched() -> List[Tuple[int, int]]:
    """6-node chain with skip: 0→1→2→3→4→5, 2→5"""
    return [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 5)]


def _edges_deep_asymmetric_merge() -> List[Tuple[int, int]]:
    """Deep chain with late side-input: 0→1→2→3, 4→3"""
    return [(0, 1), (1, 2), (2, 3), (4, 3)]


def _edges_double_diamond() -> List[Tuple[int, int]]:
    """Two diamonds chained: 0→1,0→2,1→3,2→3,3→4,3→5,4→6,5→6"""
    return [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]


# ═══════════════════════════════════════════════════════════════════════════
#  Family metadata
# ═══════════════════════════════════════════════════════════════════════════

class TopologyFamily:
    def __init__(
        self,
        name: str,
        num_nodes: int,
        edge_fn,
        is_heldout: bool = False,
        description: str = "",
    ):
        self.name = name
        self.num_nodes = num_nodes
        self.edge_fn = edge_fn
        self.is_heldout = is_heldout
        self.description = description

    def generate_edges(self) -> List[Tuple[int, int]]:
        return self.edge_fn()


TRAINING_FAMILIES: List[TopologyFamily] = [
    TopologyFamily("single_node", 1, _edges_single_node,
                   description="Single tool, no edges"),
    TopologyFamily("chain_short", 2, lambda: _edges_chain(2),
                   description="A→B linear chain"),
    TopologyFamily("chain_medium_3", 3, lambda: _edges_chain(3),
                   description="A→B→C 3-node chain"),
    TopologyFamily("chain_medium_4", 4, lambda: _edges_chain(4),
                   description="A→B→C→D 4-node chain"),
    TopologyFamily("chain_long_5", 5, lambda: _edges_chain(5),
                   description="5-node linear chain"),
    TopologyFamily("chain_long_6", 6, lambda: _edges_chain(6),
                   description="6-node linear chain"),
    TopologyFamily("fanout_2", 3, lambda: _edges_fanout(2),
                   description="Root → 2 children"),
    TopologyFamily("fanout_3", 4, lambda: _edges_fanout(3),
                   description="Root → 3 children"),
    TopologyFamily("fanin_2", 3, lambda: _edges_fanin(2),
                   description="2 parents → sink"),
    TopologyFamily("fanin_3", 4, lambda: _edges_fanin(3),
                   description="3 parents → sink"),
    TopologyFamily("hourglass", 5, _edges_hourglass,
                   description="Fan-in → bottleneck → fan-out"),
    TopologyFamily("y_shape", 4, _edges_y_shape,
                   description="Two sources merge, then chain"),
    TopologyFamily("inverted_y", 4, _edges_inverted_y,
                   description="Chain then fan-out"),
    TopologyFamily("fork_join_2", 4, lambda: _edges_fork_join_generic(2),
                   description="Root → 2 branches → merge"),
    TopologyFamily("w_shape", 5, _edges_w_shape,
                   description="Two parallel chains merging at end"),
    TopologyFamily("multi_branch_independent", 5, _edges_multi_branch_independent,
                   description="Root → 3 branches, one extended"),
    TopologyFamily("wide_fanout", 5, _edges_wide_fanout,
                   description="Root → 4 children"),
    TopologyFamily("wide_fanout_deep", 6, _edges_wide_fanout_deep,
                   description="Root → 3 children, last has sub-chain"),
    TopologyFamily("long_chain_branched", 6, _edges_long_chain_branched,
                   description="6-node chain with skip edge"),
    TopologyFamily("deep_asymmetric_merge", 5, _edges_deep_asymmetric_merge,
                   description="Deep chain with late side-input"),
]

HELDOUT_FAMILIES: List[TopologyFamily] = [
    TopologyFamily("diamond", 4, _edges_diamond,
                   is_heldout=True,
                   description="Fork at root, merge at sink (4 nodes, 4 edges)"),
    TopologyFamily("asymmetric_fork_join", 5, _edges_asymmetric_fork_join,
                   is_heldout=True,
                   description="Diamond + tail (5 nodes, 5 edges)"),
]

OPTIONAL_CHALLENGE_FAMILIES: List[TopologyFamily] = [
    TopologyFamily("double_diamond", 7, _edges_double_diamond,
                   is_heldout=True,
                   description="Two diamonds chained (7 nodes, 8 edges)"),
]

ALL_FAMILIES = TRAINING_FAMILIES + HELDOUT_FAMILIES + OPTIONAL_CHALLENGE_FAMILIES

FAMILY_BY_NAME: Dict[str, TopologyFamily] = {f.name: f for f in ALL_FAMILIES}

TRAINING_FAMILY_NAMES: Set[str] = {f.name for f in TRAINING_FAMILIES}
HELDOUT_FAMILY_NAMES: Set[str] = {f.name for f in HELDOUT_FAMILIES}


# ═══════════════════════════════════════════════════════════════════════════
#  Topology classification from graph structure
# ═══════════════════════════════════════════════════════════════════════════

def classify_topology(G: nx.DiGraph) -> str:
    """Classify a DAG into a topology family based on structural properties."""
    n = G.number_of_nodes()
    e = G.number_of_edges()

    if n == 0:
        return "empty"
    if n == 1:
        return "single_node"
    if e == 0:
        return "disconnected"

    max_in = max(G.in_degree(v) for v in G.nodes())
    max_out = max(G.out_degree(v) for v in G.nodes())
    roots = [v for v in G.nodes() if G.in_degree(v) == 0]
    sinks = [v for v in G.nodes() if G.out_degree(v) == 0]

    is_chain = (max_in <= 1 and max_out <= 1)
    has_fork = max_out > 1
    has_join = max_in > 1

    if is_chain:
        if n == 2:
            return "chain_short"
        elif n <= 4:
            return f"chain_medium_{n}"
        else:
            return f"chain_long_{n}"

    # Diamond: exactly 1 fork node, 1 join node, 4 nodes, 4 edges
    if has_fork and has_join:
        fork_nodes = [v for v in G.nodes() if G.out_degree(v) > 1]
        join_nodes = [v for v in G.nodes() if G.in_degree(v) > 1]

        if len(fork_nodes) == 1 and len(join_nodes) == 1:
            fn, jn = fork_nodes[0], join_nodes[0]
            if nx.has_path(G, fn, jn):
                if n == 4 and e == 4:
                    return "diamond"
                if n == 5 and e == 5:
                    tail_nodes = [v for v in G.nodes()
                                  if v != fn and v not in G.predecessors(jn) and v != jn]
                    if tail_nodes:
                        return "asymmetric_fork_join"
                if n == 4 and e == 3:
                    if len(roots) == 2 and len(sinks) == 1:
                        return "y_shape"
                    if len(roots) == 1 and len(sinks) == 2:
                        return "inverted_y"
                return "fork_join_2"

        if len(fork_nodes) >= 2 and len(join_nodes) >= 2:
            if n == 7 and e == 8:
                return "double_diamond"

        if len(roots) >= 2 and len(sinks) >= 2:
            if n == 5 and e == 4 and len(roots) == 2:
                return "w_shape"

        if n == 5:
            if len(roots) == 2 and len(sinks) == 2:
                return "hourglass"

        return "complex_mixed"

    if has_fork and not has_join:
        out_counts = sorted((G.out_degree(v) for v in G.nodes()), reverse=True)
        if out_counts[0] >= 4:
            if n >= 6:
                return "wide_fanout_deep"
            return "wide_fanout"
        if out_counts[0] >= 3:
            if any(G.out_degree(v) > 0 for v in G.nodes() if G.in_degree(v) > 0 and v not in roots):
                return "multi_branch_independent"
            return "fanout_3"
        return "fanout_2"

    if has_join and not has_fork:
        if max_in >= 3:
            return "fanin_3"
        return "fanin_2"

    return "other"


def canonical_structure_hash(edges: List[Tuple[int, int]], num_nodes: int) -> str:
    """Structure-only hash (tool-label-invariant)."""
    canon = sorted(edges)
    payload = f"{num_nodes}|{canon}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def canonical_labeled_hash(
    tools: List[str],
    edges: List[Tuple[int, int]],
) -> str:
    """Hash including tool labels (for labeled DAG dedup)."""
    node_labels = tuple(sorted(tools))
    edge_labels = tuple(sorted((tools[u], tools[v]) for u, v in edges))
    payload = f"{node_labels}|{edge_labels}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def canonical_toolset_hash(tools: List[str]) -> str:
    """Hash of the sorted tool multiset (for structural-twin grouping)."""
    payload = ";".join(sorted(tools)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def validate_dag(
    tools: List[str],
    edges: List[Tuple[int, int]],
) -> Dict[str, object]:
    """Validate a DAG specification."""
    result = {"valid": True, "errors": []}
    n = len(tools)

    for s, d in edges:
        if s < 0 or s >= n or d < 0 or d >= n:
            result["valid"] = False
            result["errors"].append(f"Edge ({s},{d}) out of range [0,{n-1}]")
        if s == d:
            result["valid"] = False
            result["errors"].append(f"Self-loop at node {s}")

    if len(edges) != len(set(edges)):
        result["valid"] = False
        result["errors"].append("Duplicate edges")

    if result["valid"] and edges:
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(G):
            result["valid"] = False
            result["errors"].append("Graph contains a cycle")

    return result
