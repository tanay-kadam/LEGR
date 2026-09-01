"""
run_azure_generation.py — Full Azure query generation for Campaign v4
======================================================================

Reads the locally-generated CSVs, replaces template queries with
Azure OpenAI generated queries, and writes upgraded CSVs.

Features:
  - Resumable (skips DAGs already cached)
  - Rate-limit aware with exponential backoff
  - Budget tracking
  - Per-DAG caching to JSON
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, ".")

import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

from src.data.azure_query_gen import (
    AzureBudgetTracker,
    generate_queries_for_dag,
    test_azure_connectivity,
)
from src.data.dag_generator import (
    dag_to_text,
    edges_to_str,
    tools_to_str,
)
from src.data.dataset_assembler import SCHEMA_COLUMNS, EXTENDED_COLUMNS
from src.data.topology_templates import HELDOUT_FAMILY_NAMES


CAMPAIGN_DIR = Path("data/campaign_v4")
CACHE_DIR = CAMPAIGN_DIR / "azure_cache"
TIERS = [15, 30, 45]
QUERIES_PER_DAG = 6
CONDITIONS = [
    "standard",
    "paraphrase",
    "structural_clear",
    "structural_paraphrase",
    "lexical",
    "confusable",
]


def _load_unique_dags_from_tier(tier: int) -> List[Dict]:
    """Load unique DAGs from all CSVs for a tier."""
    tier_dir = CAMPAIGN_DIR / f"campaign_v4_{tier}tools"
    seen_hashes = set()
    dags = []

    for csv_path in sorted(tier_dir.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                h = row.get("canonical_dag_hash", "")
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                tools = row["tools"].split(";")
                edge_str = row["edges"].strip()
                edges = []
                if edge_str:
                    for e in edge_str.split(";"):
                        parts = e.split("->")
                        edges.append((int(parts[0].strip()), int(parts[1].strip())))

                dags.append({
                    "labeled_hash": h,
                    "tools": tools,
                    "edges": edges,
                    "family": row["topo_family"],
                    "num_nodes": len(tools),
                    "num_edges": len(edges),
                    "split": row["split"],
                    "toolset_hash": row.get("canonical_toolset_hash", ""),
                    "structural_twin_group": row.get("structural_twin_group", ""),
                })

    return dags


def _load_cache(tier: int) -> Dict[str, List[Dict]]:
    """Load cached Azure query results for a tier."""
    cache_file = CACHE_DIR / f"tier_{tier}_queries.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return data
    return {}


def _save_cache(tier: int, cache: Dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"tier_{tier}_queries.json"
    cache_file.write_text(json.dumps(cache, indent=2, default=str), encoding="utf-8")


def generate_all_queries_for_tier(
    tier: int,
    budget: AzureBudgetTracker,
    max_cost_usd: float = 10.0,
) -> Dict[str, List[Dict]]:
    """Generate Azure queries for all DAGs in a tier."""
    dags = _load_unique_dags_from_tier(tier)
    cache = _load_cache(tier)

    print(f"\n  Tier {tier}: {len(dags)} unique DAGs, {len(cache)} already cached")

    for i, dag in enumerate(dags):
        h = dag["labeled_hash"]
        if h in cache:
            continue

        if budget.estimated_cost > max_cost_usd:
            print(f"  Budget limit reached (${budget.estimated_cost:.2f} > ${max_cost_usd})")
            break

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(dags)}] cost=${budget.estimated_cost:.3f}")
            _save_cache(tier, cache)

        result = generate_queries_for_dag(
            dag,
            n_queries=QUERIES_PER_DAG,
            conditions=CONDITIONS,
        )
        budget.record(result)

        if result.queries:
            cache[h] = result.queries
        else:
            cache[h] = [
                {"query": f"Execute workflow: {dag_to_text(dag['tools'], dag['edges'])}",
                 "condition": "standard"}
            ]
            print(f"    WARNING: Failed for {h} ({dag['family']}): {result.error}")

        time.sleep(0.2)

    _save_cache(tier, cache)
    return cache


def rebuild_csvs_with_azure_queries(tier: int, cache: Dict[str, List[Dict]]):
    """Rebuild CSVs replacing local templates with Azure queries."""
    tier_dir = CAMPAIGN_DIR / f"campaign_v4_{tier}tools"
    all_columns = SCHEMA_COLUMNS + EXTENDED_COLUMNS

    for csv_path in sorted(tier_dir.glob("*.csv")):
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            original_rows_by_hash = defaultdict(list)
            for row in reader:
                h = row.get("canonical_dag_hash", "")
                original_rows_by_hash[h].append(row)

        new_rows = []
        for h, orig_rows in original_rows_by_hash.items():
            azure_queries = cache.get(h, [])
            if not azure_queries:
                new_rows.extend(orig_rows)
                continue

            base_row = orig_rows[0]
            for i, aq in enumerate(azure_queries):
                row = dict(base_row)
                row["query"] = aq["query"]
                row["query_condition"] = aq.get("condition", "standard")
                row["source"] = "azure_gpt4o"
                new_rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(new_rows)

        print(f"    {csv_path.name}: {len(new_rows)} rows")


def main():
    print("=== Campaign v4: Full Azure Query Generation ===\n")

    conn = test_azure_connectivity()
    if not conn["success"]:
        print(f"  Azure connectivity FAILED: {conn['error']}")
        print("  Keeping local template queries.")
        return

    print(f"  Azure connected: {conn['deployment']} @ {conn['endpoint_hostname']}")

    budget = AzureBudgetTracker()

    for tier in TIERS:
        print(f"\n{'='*60}")
        print(f"  TIER {tier}")
        print(f"{'='*60}")

        cache = generate_all_queries_for_tier(tier, budget, max_cost_usd=10.0)
        rebuild_csvs_with_azure_queries(tier, cache)

        print(f"  Cost so far: ${budget.estimated_cost:.3f}")

    final_report = {
        "total_budget": budget.to_dict(),
        "tiers": {},
    }
    for tier in TIERS:
        cache = _load_cache(tier)
        total_queries = sum(len(v) for v in cache.values())
        final_report["tiers"][str(tier)] = {
            "unique_dags": len(cache),
            "total_queries_generated": total_queries,
        }

    report_path = CAMPAIGN_DIR / "azure_generation_report.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"\n  Final report: {report_path}")
    print(f"  Total cost: ${budget.estimated_cost:.3f}")


if __name__ == "__main__":
    main()
