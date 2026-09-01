"""
dataset_assembler.py — Campaign v4 Dataset Assembly & CSV Export
================================================================

Assembles generated DAGs into backward-compatible CSV datasets.
Generates placeholder queries (to be replaced by Azure OpenAI) and
exports the full campaign CSVs + manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.data.dag_generator import (
    compute_dag_statistics,
    dag_to_text,
    edges_to_str,
    generate_campaign_dags,
    tools_to_str,
)
from src.data.tool_registry import (
    TOOL_TO_CATEGORY,
    get_category_counts,
    get_tools,
    validate_registry,
)
from src.data.topology_templates import (
    HELDOUT_FAMILY_NAMES,
    TRAINING_FAMILY_NAMES,
    canonical_labeled_hash,
    canonical_toolset_hash,
)


SCHEMA_COLUMNS = [
    "query",
    "dag_id",
    "dag_text",
    "tools",
    "edges",
    "topo_family",
    "source",
    "split",
    "strict_fix_applied",
    "had_duplicate_node_labels",
    "original_tools",
]

EXTENDED_COLUMNS = [
    "tool_count",
    "canonical_dag_hash",
    "canonical_toolset_hash",
    "num_nodes",
    "num_edges",
    "query_condition",
    "structural_twin_group",
    "heldout_topology",
    "dataset_version",
]


def _generate_local_queries(
    dag: Dict,
    n_queries: int = 4,
) -> List[Dict]:
    """Generate placeholder local queries for a DAG.

    These are template-based queries good enough for pipeline testing.
    Azure OpenAI will generate the real queries in Phase 5-7.
    """
    tools = dag["tools"]
    edges = dag["edges"]
    family = dag["family"]
    n = len(tools)

    queries = []

    if n == 1:
        templates = [
            f"Execute {tools[0]}",
            f"Run the {tools[0].replace('_', ' ')} operation",
            f"I need to {tools[0].replace('_', ' ')}",
            f"Please perform {tools[0].replace('_', ' ')}",
        ]
    elif not edges:
        tool_list = " and ".join(t.replace("_", " ") for t in tools)
        templates = [
            f"Run {tool_list} in sequence",
            f"Execute these steps: {tool_list}",
            f"I need to {tool_list}",
            f"Perform {tool_list} operations",
        ]
    else:
        dag_desc = dag_to_text(tools, edges)
        readable_tools = [t.replace("_", " ") for t in tools]

        if family.startswith("chain"):
            chain_desc = " then ".join(readable_tools)
            templates = [
                f"First {readable_tools[0]}, then {', then '.join(readable_tools[1:])}",
                f"Execute in sequence: {chain_desc}",
                f"Run a pipeline: {chain_desc}",
                f"Step by step: {chain_desc}",
            ]
        elif family.startswith("fanout"):
            root = readable_tools[0]
            children = readable_tools[1:]
            child_list = " and ".join(children)
            templates = [
                f"After {root}, run {child_list} in parallel",
                f"Start with {root}, then simultaneously do {child_list}",
                f"{root} first, then fan out to {child_list}",
                f"Execute {root} and then dispatch {child_list} concurrently",
            ]
        elif family.startswith("fanin"):
            parents = readable_tools[:-1]
            sink = readable_tools[-1]
            parent_list = " and ".join(parents)
            templates = [
                f"Run {parent_list} in parallel, then {sink}",
                f"After {parent_list} all complete, execute {sink}",
                f"Gather results from {parent_list} and feed into {sink}",
                f"Do {parent_list} first, merge results, then {sink}",
            ]
        elif "diamond" in family:
            templates = [
                f"Start with {readable_tools[0]}, fork into {readable_tools[1]} and {readable_tools[2]}, then merge at {readable_tools[3]}",
                f"Execute {readable_tools[0]}, then do {readable_tools[1]} and {readable_tools[2]} in parallel, finally {readable_tools[3]}",
                f"First {readable_tools[0]}, split: {readable_tools[1]} independently from {readable_tools[2]}, join at {readable_tools[3]}",
                f"Run {readable_tools[0]}, branch to parallel {readable_tools[1]} and {readable_tools[2]}, converge at {readable_tools[3]}",
            ]
        elif "fork_join" in family or "asymmetric" in family:
            root = readable_tools[0]
            mid = readable_tools[1:-1]
            end = readable_tools[-1]
            mid_str = " and ".join(mid)
            templates = [
                f"Start with {root}, do {mid_str} in parallel, then {end}",
                f"{root} first, fork to {mid_str}, join at {end}",
                f"Execute {root}, branch into {mid_str}, merge results at {end}",
                f"Begin {root}, parallel paths: {mid_str}, converge at {end}",
            ]
        else:
            templates = [
                f"Execute workflow: {dag_desc}",
                f"Run the following dependent tasks: {dag_desc}",
                f"I need a workflow where {dag_desc}",
                f"Perform this operation graph: {dag_desc}",
            ]

    for i, query in enumerate(templates[:n_queries]):
        condition = "standard" if i == 0 else ("paraphrase" if i == 1 else "structural_clear")
        queries.append({
            "query": query,
            "query_condition": condition,
            "source": "local_template",
        })

    return queries


def assemble_tier_dataset(
    tier: int,
    target_dags: int = 300,
    queries_per_dag: int = 4,
    seed: int = 42,
) -> Tuple[List[Dict], Dict]:
    """Assemble full dataset rows for a single tier.

    Returns: (rows, stats) where rows are CSV-ready dicts and stats is metadata.
    """
    result = generate_campaign_dags(tier=tier, target_unique_dags=target_dags, seed=seed)
    all_dags = result["all_dags"]

    rows = []
    dag_id_counter = 0
    dag_hash_to_id: Dict[str, int] = {}

    for dag in all_dags:
        h = dag["labeled_hash"]
        if h not in dag_hash_to_id:
            dag_hash_to_id[h] = dag_id_counter
            dag_id_counter += 1
        dag_id = dag_hash_to_id[h]

        local_queries = _generate_local_queries(dag, n_queries=queries_per_dag)

        for q in local_queries:
            row = {
                "query": q["query"],
                "dag_id": str(dag_id),
                "dag_text": dag_to_text(dag["tools"], dag["edges"]),
                "tools": tools_to_str(dag["tools"]),
                "edges": edges_to_str(dag["edges"]),
                "topo_family": dag["family"],
                "source": q["source"],
                "split": dag["split"],
                "strict_fix_applied": "FALSE",
                "had_duplicate_node_labels": "FALSE",
                "original_tools": tools_to_str(dag["tools"]),
                # Extended columns
                "tool_count": str(tier),
                "canonical_dag_hash": h,
                "canonical_toolset_hash": dag["toolset_hash"],
                "num_nodes": str(dag["num_nodes"]),
                "num_edges": str(dag["num_edges"]),
                "query_condition": q["query_condition"],
                "structural_twin_group": dag.get("structural_twin_group", ""),
                "heldout_topology": str(dag["family"] in HELDOUT_FAMILY_NAMES),
                "dataset_version": "campaign_v4",
            }
            rows.append(row)

    stats = compute_dag_statistics(all_dags)
    stats["tier"] = tier
    stats["total_rows"] = len(rows)
    stats["unique_queries"] = len(set(r["query"] for r in rows))
    stats["queries_per_dag"] = queries_per_dag

    return rows, stats


def export_tier_csvs(
    rows: List[Dict],
    tier: int,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Export rows to per-split CSVs compatible with existing loaders."""
    output_dir = Path(output_dir)
    tier_dir = output_dir / f"campaign_v4_{tier}tools"
    tier_dir.mkdir(parents=True, exist_ok=True)

    all_columns = SCHEMA_COLUMNS + EXTENDED_COLUMNS
    splits = defaultdict(list)
    for row in rows:
        splits[row["split"]].append(row)

    split_remap = {
        "train": "train",
        "val": "dev",
        "test_indomain": "test_indomain",
        "test_topology_heldout": "test_topology_heldout",
        "candidate_only": "candidate_corpus",
    }

    paths = {}
    for split_name, split_rows in splits.items():
        fname = split_remap.get(split_name, split_name) + ".csv"
        path = tier_dir / fname
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(split_rows)
        paths[split_name] = path

    return paths


def generate_manifest(
    all_stats: Dict[int, Dict],
    output_dir: str | Path,
) -> Path:
    """Generate campaign manifest JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "campaign": "campaign_v4",
        "generated_at": datetime.now().isoformat(),
        "tiers": {},
        "design_principles": {
            "nested_tool_sets": "tools_15 ⊂ tools_30 ⊂ tools_45",
            "structural_twins": "Same tool multiset, different directed edge structure",
            "heldout_topologies": ["diamond", "asymmetric_fork_join"],
            "query_source": "local_template (pending Azure OpenAI upgrade)",
        },
    }

    for tier, stats in all_stats.items():
        manifest["tiers"][str(tier)] = stats

    path = output_dir / "campaign_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Campaign entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_local_campaign(
    output_dir: str | Path = "data/campaign_v4",
    target_dags: Optional[Dict[int, int]] = None,
    queries_per_dag: int = 4,
    seed: int = 42,
    pilot: bool = False,
) -> Dict:
    """Run the complete local (non-Azure) dataset generation campaign.

    Args:
        output_dir: Root output directory.
        target_dags: Per-tier DAG targets, e.g. {15: 250, 30: 350, 45: 500}.
        queries_per_dag: Queries per DAG (local templates).
        seed: Random seed.
        pilot: If True, use minimal targets (10 DAGs/tier).

    Returns:
        Dict with paths and statistics for all tiers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pilot:
        target_dags = {15: 10, 30: 10, 45: 10}
        queries_per_dag = 2
    elif target_dags is None:
        target_dags = {15: 250, 30: 350, 45: 500}

    all_stats: Dict[int, Dict] = {}
    all_paths: Dict[int, Dict] = {}

    for tier in [15, 30, 45]:
        target = target_dags.get(tier, 300)
        print(f"\n{'='*60}")
        print(f"  Generating Tier {tier} ({target} target DAGs)...")
        print(f"{'='*60}")

        rows, stats = assemble_tier_dataset(
            tier=tier,
            target_dags=target,
            queries_per_dag=queries_per_dag,
            seed=seed,
        )
        paths = export_tier_csvs(rows, tier, output_dir)

        all_stats[tier] = stats
        all_paths[tier] = {k: str(v) for k, v in paths.items()}

        print(f"  Total rows: {stats['total_rows']}")
        print(f"  Unique DAGs: {stats['total_unique_dags']}")
        print(f"  Twin groups (2+): {stats['twin_groups_with_2plus']}")
        print(f"  Split counts: {stats['split_counts']}")
        for split, path in paths.items():
            print(f"    {split}: {path}")

    manifest_path = generate_manifest(all_stats, output_dir)
    print(f"\nManifest: {manifest_path}")

    return {
        "manifest": str(manifest_path),
        "stats": all_stats,
        "paths": all_paths,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Campaign v4 local dataset generation")
    parser.add_argument("--output-dir", default="data/campaign_v4")
    parser.add_argument("--pilot", action="store_true", help="Minimal pilot run (10 DAGs/tier)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries-per-dag", type=int, default=4)
    args = parser.parse_args()

    result = run_local_campaign(
        output_dir=args.output_dir,
        queries_per_dag=args.queries_per_dag,
        seed=args.seed,
        pilot=args.pilot,
    )
    print("\nDone.")
