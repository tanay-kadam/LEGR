"""
graph_utils.py -- Graph / DAG utilities for the graph dataset upgrade
=====================================================================

Provides:
  - CSV row -> NetworkX DiGraph parsing & validation
  - Structure-only topology hashing (isomorphism-invariant)
  - Named topology-family classification
  - New topology family generators (diamond, asymmetric fork-join, etc.)
  - Controlled DAG generator
  - Hard-negative DAG generators
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from vocab_config import ACTIVE_TOOL_COUNT


# ─────────────────────────────────────────────────────────────────────────────
#  Tool vocabulary (mirrored for standalone use; authoritative source is
#  data_synth.TOOL_VOCAB)
# ─────────────────────────────────────────────────────────────────────────────

_FULL_TOOL_VOCAB: List[str] = [
    "db_read", "db_write", "reset_password", "create_ticket",
    "send_notification", "quarantine_system", "scan_malware",
    "generate_report", "process_refund", "update_subscription",
    "provision_vm", "restart_service", "check_status",
    "escalate_to_human", "log_audit_event",
    # Tier 2
    "deploy_container", "rollback_deployment", "rotate_api_key",
    "backup_database", "restore_backup", "scale_service",
    "run_pipeline", "approve_access", "revoke_access",
    "transfer_ownership", "schedule_maintenance", "archive_data",
    "enable_feature_flag", "disable_feature_flag", "invalidate_cache",
    # Tier 3
    "create_dns_record", "renew_certificate", "block_ip_address",
    "unblock_ip_address", "assign_role", "remove_role",
    "trigger_failover", "run_load_test", "snapshot_vm",
    "migrate_database", "tag_resource", "create_alert_rule",
    "acknowledge_alert", "merge_accounts", "export_data",
]

TOOL_VOCAB: List[str] = _FULL_TOOL_VOCAB[:ACTIVE_TOOL_COUNT]


# ─────────────────────────────────────────────────────────────────────────────
#  Parsing CSV rows into NetworkX DAGs
# ─────────────────────────────────────────────────────────────────────────────

def parse_tools(tools_str: str) -> List[str]:
    """Parse semicolon-separated tool string -> list."""
    if not tools_str or (isinstance(tools_str, float)):
        return []
    return [t.strip() for t in str(tools_str).split(";") if t.strip()]


def parse_edges(edges_str: str) -> List[Tuple[int, int]]:
    """Parse '0->1;1->2' edge string -> list of (src, dst) tuples."""
    if not edges_str or (isinstance(edges_str, float)):
        return []
    edges_str = str(edges_str).strip()
    if not edges_str:
        return []
    pairs = []
    for part in edges_str.split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\d+)\s*->\s*(\d+)", part)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def build_dag_from_row(
    tools: List[str],
    edges: List[Tuple[int, int]],
) -> nx.DiGraph:
    """Build a NetworkX DiGraph from parsed tools and edges."""
    G = nx.DiGraph()
    for i, tool in enumerate(tools):
        G.add_node(i, tool=tool)
    G.add_edges_from(edges)
    return G


def validate_dag(
    tools: List[str],
    edges: List[Tuple[int, int]],
) -> Dict[str, object]:
    """Validate a DAG specification. Returns a dict of validation results."""
    result = {
        "valid": True,
        "is_dag": True,
        "invalid_node_refs": False,
        "tool_count_matches": True,
        "errors": [],
    }
    n = len(tools)

    for src, dst in edges:
        if src >= n or dst >= n or src < 0 or dst < 0:
            result["valid"] = False
            result["invalid_node_refs"] = True
            result["errors"].append(f"Edge ({src},{dst}) references node outside [0,{n-1}]")

    if result["valid"]:
        G = build_dag_from_row(tools, edges)
        if not nx.is_directed_acyclic_graph(G):
            result["valid"] = False
            result["is_dag"] = False
            result["errors"].append("Graph contains a cycle")

        actual_nodes = set(G.nodes())
        expected_nodes = set(range(n))
        if actual_nodes != expected_nodes:
            result["tool_count_matches"] = False
            result["errors"].append(
                f"Node set mismatch: expected {expected_nodes}, got {actual_nodes}"
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Topology hashing (structure-only, tool-label-invariant)
# ─────────────────────────────────────────────────────────────────────────────

def topology_hash(edges: List[Tuple[int, int]], num_nodes: int) -> str:
    """Compute a structure-only hash invariant to node renaming.

    Uses Weisfeiler-Lehman graph hash with degree labels only.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    node_labels = {}
    for n in G.nodes():
        in_d = G.in_degree(n)
        out_d = G.out_degree(n)
        node_labels[n] = f"i{in_d}o{out_d}"

    try:
        h = nx.weisfeiler_lehman_graph_hash(G, node_attr="wl_label",
                                             iterations=3)
    except Exception:
        # Fallback: canonical sorted edge representation
        nx.set_node_attributes(G, node_labels, "wl_label")
        canon = sorted((node_labels[s], node_labels[d]) for s, d in edges)
        payload = f"{num_nodes}|{canon}".encode()
        h = hashlib.sha256(payload).hexdigest()[:16]
        return h

    nx.set_node_attributes(G, node_labels, "wl_label")
    h = nx.weisfeiler_lehman_graph_hash(G, node_attr="wl_label", iterations=3)
    return h


def labeled_dag_hash(G: nx.DiGraph) -> str:
    """Hash including tool labels (for deduplication of tool-specific DAGs)."""
    node_labels = tuple(sorted(G.nodes[n]["tool"] for n in G.nodes()))
    edge_labels = tuple(sorted(
        (G.nodes[u]["tool"], G.nodes[v]["tool"]) for u, v in G.edges()
    ))
    payload = f"{node_labels}|{edge_labels}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
#  Topology family classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_topology(
    edges: List[Tuple[int, int]],
    num_nodes: int,
) -> str:
    """Classify the topology family of a DAG by structural properties."""
    if num_nodes == 0:
        return "empty"
    if num_nodes == 1:
        return "single_node"
    if not edges:
        return "disconnected"

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    max_in = max(G.in_degree(n) for n in G.nodes())
    max_out = max(G.out_degree(n) for n in G.nodes())
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    depth = nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else 0

    is_chain = (max_in <= 1 and max_out <= 1)
    has_fork = max_out > 1
    has_join = max_in > 1

    if is_chain:
        if num_nodes <= 2:
            return "chain_short"
        elif num_nodes <= 4:
            return "chain_medium"
        else:
            return "chain_long"

    if has_fork and has_join:
        fork_nodes = [n for n in G.nodes() if G.out_degree(n) > 1]
        join_nodes = [n for n in G.nodes() if G.in_degree(n) > 1]
        if len(fork_nodes) == 1 and len(join_nodes) == 1:
            fn, jn = fork_nodes[0], join_nodes[0]
            if nx.has_path(G, fn, jn):
                if num_nodes == 4 and len(edges) == 4:
                    return "diamond"
                return "fork_join"

        if len(roots) >= 2 and len(sinks) >= 2:
            return "parallel_paths"

        if depth >= 4:
            return "complex_deep"
        return "complex_mixed"

    if has_fork and not has_join:
        if max_out >= 3:
            return "wide_fanout"
        return "fanout"

    if has_join and not has_fork:
        if max_in >= 3:
            return "wide_fanin"
        return "fanin"

    return "other"


# ─────────────────────────────────────────────────────────────────────────────
#  New topology family generators
# ─────────────────────────────────────────────────────────────────────────────

def _sample_tools(rng: random.Random, n: int, vocab: List[str],
                  allow_repeats: bool = False) -> List[str]:
    """Sample *n* tools from vocabulary."""
    if allow_repeats or n > len(vocab):
        return [rng.choice(vocab) for _ in range(n)]
    return rng.sample(vocab, n)


def gen_diamond(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    tools = _sample_tools(rng, 4, vocab)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    return tools, edges


def gen_asymmetric_fork_join(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Diamond + tail: 0->1, 0->2, 1->3, 2->3, 3->4"""
    tools = _sample_tools(rng, 5, vocab)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]
    return tools, edges


def gen_deep_asymmetric_merge(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Deep chain with late side-input: 0->1->2->3, 4->3"""
    tools = _sample_tools(rng, 5, vocab)
    edges = [(0, 1), (1, 2), (2, 3), (4, 3)]
    return tools, edges


def gen_multi_branch_independent(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Root fans out to three branches, one has a sub-chain: 0->1, 0->2, 0->3, 3->4"""
    tools = _sample_tools(rng, 5, vocab)
    edges = [(0, 1), (0, 2), (0, 3), (3, 4)]
    return tools, edges


def gen_repeated_tool(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Chain where one tool appears twice: 0->1->2->3, tools[0]==tools[3]"""
    base = _sample_tools(rng, 3, vocab)
    tools = base + [base[0]]  # repeat first tool at end
    edges = [(0, 1), (1, 2), (2, 3)]
    return tools, edges


def gen_hourglass(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Fan-in then fan-out: 0->2, 1->2, 2->3, 2->4"""
    tools = _sample_tools(rng, 5, vocab)
    edges = [(0, 2), (1, 2), (2, 3), (2, 4)]
    return tools, edges


def gen_w_shape(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Two parallel chains merging: 0->1, 2->3, 1->4, 3->4"""
    tools = _sample_tools(rng, 5, vocab)
    edges = [(0, 1), (2, 3), (1, 4), (3, 4)]
    return tools, edges


def gen_long_chain_branched(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """6-node chain with a branch: 0->1->2->3->4->5, 2->5"""
    tools = _sample_tools(rng, 6, vocab, allow_repeats=True)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 5)]
    return tools, edges


def gen_double_diamond(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Two diamonds chained: 0->1,0->2,1->3,2->3,3->4,3->5,4->6,5->6"""
    tools = _sample_tools(rng, 7, vocab, allow_repeats=True)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]
    return tools, edges


def gen_wide_fanout_deep(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Root -> 4 children, one child has a sub-chain: 0->1,0->2,0->3,0->4,4->5,5->6"""
    tools = _sample_tools(rng, 7, vocab, allow_repeats=True)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6)]
    return tools, edges


def gen_y_shape(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Y-merge: 0->2, 1->2, 2->3"""
    tools = _sample_tools(rng, 4, vocab)
    edges = [(0, 2), (1, 2), (2, 3)]
    return tools, edges


def gen_inverted_y(rng: random.Random, vocab: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Chain then fan-out: 0->1, 1->2, 1->3"""
    tools = _sample_tools(rng, 4, vocab)
    edges = [(0, 1), (1, 2), (1, 3)]
    return tools, edges


TOPOLOGY_GENERATORS = {
    "diamond": gen_diamond,
    "asymmetric_fork_join": gen_asymmetric_fork_join,
    "deep_asymmetric_merge": gen_deep_asymmetric_merge,
    "multi_branch_independent": gen_multi_branch_independent,
    "repeated_tool": gen_repeated_tool,
    "hourglass": gen_hourglass,
    "w_shape": gen_w_shape,
    "long_chain_branched": gen_long_chain_branched,
    "double_diamond": gen_double_diamond,
    "wide_fanout_deep": gen_wide_fanout_deep,
    "y_shape": gen_y_shape,
    "inverted_y": gen_inverted_y,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Controlled DAG generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dags(
    n_dags: int,
    families: Optional[List[str]] = None,
    vocab: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict]:
    """Generate *n_dags* unique DAGs across specified topology families.

    Returns list of dicts with keys: tools, edges, dag_text, family, dag_hash.
    """
    rng = random.Random(seed)
    if vocab is None:
        vocab = TOOL_VOCAB
    if families is None:
        families = list(TOPOLOGY_GENERATORS.keys())

    seen_hashes: Set[str] = set()
    results: List[Dict] = []
    attempts = 0
    max_attempts = n_dags * 20

    while len(results) < n_dags and attempts < max_attempts:
        attempts += 1
        family = rng.choice(families)
        gen_fn = TOPOLOGY_GENERATORS[family]
        tools, edges = gen_fn(rng, vocab)

        G = build_dag_from_row(tools, edges)
        if not nx.is_directed_acyclic_graph(G):
            continue

        h = labeled_dag_hash(G)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        dag_text = _dag_to_text(G)
        results.append({
            "tools": tools,
            "edges": edges,
            "dag_text": dag_text,
            "family": family,
            "dag_hash": h,
        })

    return results


def _dag_to_text(G: nx.DiGraph) -> str:
    """Canonical text description of a DAG."""
    if G.number_of_edges() == 0:
        tools = [G.nodes[n]["tool"] for n in sorted(G.nodes())]
        return ", ".join(tools)
    edges_str = sorted(
        f"{G.nodes[u]['tool']} -> {G.nodes[v]['tool']}" for u, v in G.edges()
    )
    return ", ".join(edges_str)


def tools_to_str(tools: List[str]) -> str:
    return ";".join(tools)


def edges_to_str(edges: List[Tuple[int, int]]) -> str:
    if not edges:
        return ""
    return ";".join(f"{s}->{d}" for s, d in edges)


# ─────────────────────────────────────────────────────────────────────────────
#  Hard-negative DAG generators
# ─────────────────────────────────────────────────────────────────────────────

def hard_neg_swap_edges(
    tools: List[str],
    edges: List[Tuple[int, int]],
    rng: random.Random,
) -> Optional[Tuple[List[str], List[Tuple[int, int]]]]:
    """Same tools, different (valid DAG) edge wiring."""
    n = len(tools)
    if n < 3 or len(edges) < 2:
        return None

    for _ in range(20):
        new_edges = []
        nodes = list(range(n))
        rng.shuffle(nodes)
        topo = {v: i for i, v in enumerate(nodes)}

        available_pairs = [
            (a, b) for a in range(n) for b in range(n)
            if a != b and topo[a] < topo[b]
        ]
        if len(available_pairs) < len(edges):
            continue
        new_edges = rng.sample(available_pairs, len(edges))

        G = build_dag_from_row(tools, new_edges)
        if nx.is_directed_acyclic_graph(G) and set(new_edges) != set(edges):
            return tools, new_edges
    return None


def hard_neg_swap_tools(
    tools: List[str],
    edges: List[Tuple[int, int]],
    rng: random.Random,
    vocab: Optional[List[str]] = None,
) -> Optional[Tuple[List[str], List[Tuple[int, int]]]]:
    """Same topology, different tool assignment."""
    if vocab is None:
        vocab = TOOL_VOCAB
    for _ in range(20):
        new_tools = _sample_tools(rng, len(tools), vocab)
        if new_tools != tools:
            return new_tools, edges
    return None


def hard_neg_remove_edge(
    tools: List[str],
    edges: List[Tuple[int, int]],
    rng: random.Random,
) -> Optional[Tuple[List[str], List[Tuple[int, int]]]]:
    """Remove one edge (missing dependency)."""
    if len(edges) < 2:
        return None
    idx = rng.randint(0, len(edges) - 1)
    new_edges = edges[:idx] + edges[idx + 1:]
    return tools, new_edges


def hard_neg_add_edge(
    tools: List[str],
    edges: List[Tuple[int, int]],
    rng: random.Random,
) -> Optional[Tuple[List[str], List[Tuple[int, int]]]]:
    """Add one edge while maintaining DAG property."""
    n = len(tools)
    G = build_dag_from_row(tools, edges)
    existing = set(edges)

    candidates = [
        (a, b) for a in range(n) for b in range(n)
        if a != b and (a, b) not in existing
    ]
    rng.shuffle(candidates)

    for a, b in candidates[:30]:
        trial = list(edges) + [(a, b)]
        Gt = build_dag_from_row(tools, trial)
        if nx.is_directed_acyclic_graph(Gt):
            return tools, trial
    return None


def hard_neg_extra_node(
    tools: List[str],
    edges: List[Tuple[int, int]],
    rng: random.Random,
    vocab: Optional[List[str]] = None,
) -> Optional[Tuple[List[str], List[Tuple[int, int]]]]:
    """Add one distractor node with a plausible edge."""
    if vocab is None:
        vocab = TOOL_VOCAB
    new_tool = rng.choice(vocab)
    new_idx = len(tools)
    new_tools = tools + [new_tool]

    attach = rng.randint(0, len(tools) - 1)
    if rng.random() < 0.5:
        new_edge = (attach, new_idx)
    else:
        new_edge = (new_idx, attach)

    trial_edges = list(edges) + [new_edge]
    G = build_dag_from_row(new_tools, trial_edges)
    if nx.is_directed_acyclic_graph(G):
        return new_tools, trial_edges

    new_edge = (new_idx, attach) if new_edge == (attach, new_idx) else (attach, new_idx)
    trial_edges = list(edges) + [new_edge]
    G = build_dag_from_row(new_tools, trial_edges)
    if nx.is_directed_acyclic_graph(G):
        return new_tools, trial_edges

    return new_tools, edges


def generate_hard_negatives(
    tools: List[str],
    edges: List[Tuple[int, int]],
    rng: random.Random,
    vocab: Optional[List[str]] = None,
) -> List[Dict]:
    """Generate up to 5 hard-negative variants for a given DAG."""
    negatives = []
    generators = [
        ("swap_edges", hard_neg_swap_edges),
        ("swap_tools", hard_neg_swap_tools),
        ("remove_edge", hard_neg_remove_edge),
        ("add_edge", hard_neg_add_edge),
        ("extra_node", hard_neg_extra_node),
    ]

    for neg_type, fn in generators:
        if neg_type in ("swap_tools", "extra_node"):
            result = fn(tools, edges, rng, vocab)
        else:
            result = fn(tools, edges, rng)
        if result is not None:
            neg_tools, neg_edges = result
            G = build_dag_from_row(neg_tools, neg_edges)
            negatives.append({
                "negative_type": neg_type,
                "neg_tools": neg_tools,
                "neg_edges": neg_edges,
                "neg_dag_text": _dag_to_text(G),
            })

    return negatives
