"""
dag_generator.py — Campaign v4 Programmatic DAG Generator
============================================================

Generates DAGs for all three tool tiers with:
  - Structural twins (same tool multiset, different edge structures)
  - Topology family diversity
  - Strict held-out topology enforcement
  - Acyclicity and validity guarantees
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from src.data.tool_registry import (
    TOOL_TO_CATEGORY,
    FunctionalCategory,
    get_tools,
)
from src.data.topology_templates import (
    FAMILY_BY_NAME,
    HELDOUT_FAMILY_NAMES,
    TRAINING_FAMILIES,
    TRAINING_FAMILY_NAMES,
    TopologyFamily,
    canonical_labeled_hash,
    canonical_toolset_hash,
    classify_topology,
    validate_dag,
)


def _select_tools_for_family(
    family: TopologyFamily,
    vocab: List[str],
    rng: random.Random,
    category_map: Dict[str, FunctionalCategory],
) -> List[str]:
    """Select tools matching the family's node count from the vocabulary."""
    n = family.num_nodes
    if n > len(vocab):
        return []
    return rng.sample(vocab, n)


def _build_dag(tools: List[str], edges: List[Tuple[int, int]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for i, tool in enumerate(tools):
        G.add_node(i, tool=tool)
    G.add_edges_from(edges)
    return G


def _is_valid_dag(tools: List[str], edges: List[Tuple[int, int]]) -> bool:
    v = validate_dag(tools, edges)
    return v["valid"]


# ═══════════════════════════════════════════════════════════════════════════
#  Structural Twin Generation
# ═══════════════════════════════════════════════════════════════════════════

# Families grouped by node count for twin generation
_FAMILIES_BY_NODE_COUNT: Dict[int, List[TopologyFamily]] = defaultdict(list)
for _f in TRAINING_FAMILIES:
    _FAMILIES_BY_NODE_COUNT[_f.num_nodes].append(_f)


def _generate_all_topologies_for_toolset(
    tools: List[str],
    allowed_families: Optional[Set[str]] = None,
    include_heldout: bool = False,
) -> List[Dict]:
    """Generate all valid topology variants for a given tool set.

    Returns list of dicts with: tools, edges, family, labeled_hash, toolset_hash.
    """
    n = len(tools)
    results = []
    seen_edge_sets: Set[str] = set()

    families_for_n = _FAMILIES_BY_NODE_COUNT.get(n, [])
    if include_heldout:
        for fname in HELDOUT_FAMILY_NAMES:
            f = FAMILY_BY_NAME[fname]
            if f.num_nodes == n:
                families_for_n = list(families_for_n) + [f]

    for family in families_for_n:
        if allowed_families is not None and family.name not in allowed_families:
            continue
        edges = family.generate_edges()
        if not _is_valid_dag(tools, edges):
            continue

        edge_key = str(sorted(edges))
        if edge_key in seen_edge_sets:
            continue
        seen_edge_sets.add(edge_key)

        results.append({
            "tools": list(tools),
            "edges": edges,
            "family": family.name,
            "labeled_hash": canonical_labeled_hash(tools, edges),
            "toolset_hash": canonical_toolset_hash(tools),
            "num_nodes": n,
            "num_edges": len(edges),
        })

    return results


def generate_structural_twins(
    vocab: List[str],
    n_twin_groups: int = 50,
    min_variants_per_group: int = 2,
    max_variants_per_group: int = 5,
    node_counts: Optional[List[int]] = None,
    allowed_families: Optional[Set[str]] = None,
    include_heldout: bool = False,
    seed: int = 42,
) -> List[Dict]:
    """Generate structural twin groups: same tool multiset, different edges.

    Returns a flat list of DAG dicts, each tagged with structural_twin_group.
    """
    rng = random.Random(seed)
    if node_counts is None:
        node_counts = [3, 4, 5]

    all_dags = []
    twin_groups_found = 0
    attempts = 0
    max_attempts = n_twin_groups * 50
    seen_labeled_hashes: Set[str] = set()

    while twin_groups_found < n_twin_groups and attempts < max_attempts:
        attempts += 1
        n = rng.choice(node_counts)
        if n > len(vocab):
            continue

        tools = rng.sample(vocab, n)
        variants = _generate_all_topologies_for_toolset(
            tools,
            allowed_families=allowed_families,
            include_heldout=include_heldout,
        )

        new_variants = [v for v in variants if v["labeled_hash"] not in seen_labeled_hashes]
        if len(new_variants) < min_variants_per_group:
            continue

        selected = new_variants[:max_variants_per_group]
        for v in selected:
            seen_labeled_hashes.add(v["labeled_hash"])
            v["structural_twin_group"] = canonical_toolset_hash(tools)
            all_dags.append(v)

        twin_groups_found += 1

    return all_dags


# ═══════════════════════════════════════════════════════════════════════════
#  Main DAG generation pipeline
# ═══════════════════════════════════════════════════════════════════════════

def generate_campaign_dags(
    tier: int,
    target_unique_dags: int = 300,
    twin_fraction: float = 0.4,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Generate DAGs for a single tool tier.

    Returns dict with keys: train_dags, val_dags, test_indomain_dags,
    test_heldout_dags, all_dags.
    """
    rng = random.Random(seed)
    vocab = get_tools(tier)

    n_twin_target = int(target_unique_dags * twin_fraction)
    n_regular_target = target_unique_dags - n_twin_target

    all_dags: List[Dict] = []
    seen_hashes: Set[str] = set()

    # Phase A: Generate structural twins (training families only)
    twins = generate_structural_twins(
        vocab=vocab,
        n_twin_groups=n_twin_target // 2,
        min_variants_per_group=2,
        max_variants_per_group=4,
        node_counts=[3, 4, 5],
        allowed_families=TRAINING_FAMILY_NAMES,
        include_heldout=False,
        seed=seed,
    )

    for dag in twins:
        if dag["labeled_hash"] not in seen_hashes:
            seen_hashes.add(dag["labeled_hash"])
            dag["split"] = "train"
            dag["is_twin"] = True
            all_dags.append(dag)

    # Phase B: Generate regular training DAGs (diverse topologies)
    attempts = 0
    max_attempts = n_regular_target * 30
    while len([d for d in all_dags if d["split"] == "train"]) < n_regular_target + len(twins) and attempts < max_attempts:
        attempts += 1
        family = rng.choice(TRAINING_FAMILIES)
        tools = _select_tools_for_family(family, vocab, rng, TOOL_TO_CATEGORY)
        if not tools:
            continue
        edges = family.generate_edges()
        if not _is_valid_dag(tools, edges):
            continue

        h = canonical_labeled_hash(tools, edges)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        all_dags.append({
            "tools": tools,
            "edges": edges,
            "family": family.name,
            "labeled_hash": h,
            "toolset_hash": canonical_toolset_hash(tools),
            "num_nodes": len(tools),
            "num_edges": len(edges),
            "split": "train",
            "is_twin": False,
            "structural_twin_group": canonical_toolset_hash(tools),
        })

    # Phase C: Generate held-out topology DAGs
    heldout_families = [FAMILY_BY_NAME["diamond"], FAMILY_BY_NAME["asymmetric_fork_join"]]
    n_heldout_per_family = max(20, target_unique_dags // 10)
    heldout_dags = []

    for family in heldout_families:
        count = 0
        h_attempts = 0
        while count < n_heldout_per_family and h_attempts < n_heldout_per_family * 20:
            h_attempts += 1
            tools = _select_tools_for_family(family, vocab, rng, TOOL_TO_CATEGORY)
            if not tools:
                continue
            edges = family.generate_edges()
            if not _is_valid_dag(tools, edges):
                continue

            h = canonical_labeled_hash(tools, edges)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            dag = {
                "tools": tools,
                "edges": edges,
                "family": family.name,
                "labeled_hash": h,
                "toolset_hash": canonical_toolset_hash(tools),
                "num_nodes": len(tools),
                "num_edges": len(edges),
                "split": "test_topology_heldout",
                "is_twin": False,
                "structural_twin_group": canonical_toolset_hash(tools),
            }
            heldout_dags.append(dag)
            all_dags.append(dag)
            count += 1

    # Phase D: Generate structural-twin distractors for held-out DAGs
    for dag in list(heldout_dags):
        twin_variants = _generate_all_topologies_for_toolset(
            dag["tools"],
            allowed_families=TRAINING_FAMILY_NAMES,
            include_heldout=True,
        )
        for variant in twin_variants:
            if variant["labeled_hash"] != dag["labeled_hash"] and variant["labeled_hash"] not in seen_hashes:
                seen_hashes.add(variant["labeled_hash"])
                variant["split"] = "candidate_only"
                variant["is_twin"] = True
                variant["structural_twin_group"] = dag["structural_twin_group"]
                all_dags.append(variant)

    # Phase E: Split training DAGs into train / val / test_indomain
    train_dags_raw = [d for d in all_dags if d["split"] == "train"]
    rng.shuffle(train_dags_raw)

    n_total_train = len(train_dags_raw)
    n_val = max(1, int(n_total_train * 0.15))
    n_test_id = max(1, int(n_total_train * 0.10))

    for d in train_dags_raw[:n_val]:
        d["split"] = "val"
    for d in train_dags_raw[n_val:n_val + n_test_id]:
        d["split"] = "test_indomain"

    return {
        "train_dags": [d for d in all_dags if d["split"] == "train"],
        "val_dags": [d for d in all_dags if d["split"] == "val"],
        "test_indomain_dags": [d for d in all_dags if d["split"] == "test_indomain"],
        "test_heldout_dags": [d for d in all_dags if d["split"] == "test_topology_heldout"],
        "candidate_dags": [d for d in all_dags if d["split"] == "candidate_only"],
        "all_dags": all_dags,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Serialization helpers
# ═══════════════════════════════════════════════════════════════════════════

def dag_to_text(tools: List[str], edges: List[Tuple[int, int]]) -> str:
    if not edges:
        return ", ".join(tools)
    edge_strs = sorted(f"{tools[u]} -> {tools[v]}" for u, v in edges)
    return ", ".join(edge_strs)


def tools_to_str(tools: List[str]) -> str:
    return ";".join(tools)


def edges_to_str(edges: List[Tuple[int, int]]) -> str:
    if not edges:
        return ""
    return ";".join(f"{s}->{d}" for s, d in edges)


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_dag_statistics(dags: List[Dict]) -> Dict:
    """Compute summary statistics for a DAG collection."""
    twin_groups: Dict[str, List[str]] = defaultdict(list)
    family_counts: Dict[str, int] = defaultdict(int)
    split_counts: Dict[str, int] = defaultdict(int)
    tool_usage: Dict[str, int] = defaultdict(int)

    for dag in dags:
        twin_groups[dag["toolset_hash"]].append(dag["labeled_hash"])
        family_counts[dag["family"]] += 1
        split_counts[dag["split"]] += 1
        for t in dag["tools"]:
            tool_usage[t] += 1

    groups_with_twins = {k: v for k, v in twin_groups.items() if len(v) >= 2}
    mean_per_group = (
        sum(len(v) for v in groups_with_twins.values()) / len(groups_with_twins)
        if groups_with_twins else 0
    )

    return {
        "total_unique_dags": len(dags),
        "split_counts": dict(split_counts),
        "family_counts": dict(family_counts),
        "twin_groups_total": len(twin_groups),
        "twin_groups_with_2plus": len(groups_with_twins),
        "mean_dags_per_twin_group": round(mean_per_group, 2),
        "tool_coverage": len(tool_usage),
        "tools_unused": [t for t in get_tools(45) if t not in tool_usage],
    }
