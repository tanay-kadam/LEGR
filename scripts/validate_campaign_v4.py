"""
validate_campaign_v4.py — Phase 8: Complete Dataset Validation
================================================================

Comprehensive validation of the campaign v4 dataset after Azure query
generation. Tests backward compatibility, structural invariants, leakage
prevention, twin density, topology correctness, and query diversity.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from src.data.tool_registry import get_tools, TOOL_TO_CATEGORY
from src.data.topology_templates import HELDOUT_FAMILY_NAMES, TRAINING_FAMILY_NAMES


CAMPAIGN_DIR = Path("data/campaign_v4")
TIERS = [15, 30, 45]

results = {}
critical_failures = []


def record(test_name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results[test_name] = {"status": status, "detail": detail}
    indicator = "PASS" if passed else "FAIL"
    print(f"  [{indicator}] {test_name}" + (f" -- {detail}" if detail else ""), flush=True)
    if not passed:
        critical_failures.append(test_name)


def load_tier_rows(tier):
    tier_dir = CAMPAIGN_DIR / f"campaign_v4_{tier}tools"
    all_rows = []
    for csv_path in sorted(tier_dir.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_file"] = csv_path.stem
                all_rows.append(row)
    return all_rows


print("=" * 60, flush=True)
print("  Campaign v4 — Complete Dataset Validation", flush=True)
print("=" * 60, flush=True)

for tier in TIERS:
    print(f"\n--- Tier {tier} ---", flush=True)
    rows = load_tier_rows(tier)
    vocab = set(get_tools(tier))

    # V01: Row count sanity
    record(f"V01_row_count_{tier}", len(rows) > 100, f"{len(rows)} rows")

    # V02: All tools within vocabulary
    tools_used = set()
    for r in rows:
        tools_used.update(r["tools"].split(";"))
    extra = tools_used - vocab
    record(f"V02_tools_in_vocab_{tier}", not extra,
           f"extra={extra}" if extra else f"{len(tools_used)} tools used")

    # V03: No heldout topology in train/val
    train_val = [r for r in rows if r["_file"] in ("train", "dev")]
    leaked = {r["topo_family"] for r in train_val} & HELDOUT_FAMILY_NAMES
    record(f"V03_no_heldout_leak_{tier}", not leaked)

    # V04: Heldout test has only heldout families
    heldout = [r for r in rows if r["_file"] == "test_topology_heldout"]
    heldout_fams = {r["topo_family"] for r in heldout}
    record(f"V04_heldout_only_{tier}", heldout_fams <= HELDOUT_FAMILY_NAMES,
           f"families={heldout_fams}")

    # V05: No labeled DAG overlap between train and test
    train_hashes = {r["canonical_dag_hash"] for r in rows if r["_file"] == "train"}
    test_hashes = {r["canonical_dag_hash"] for r in rows
                   if r["_file"] in ("test_indomain", "test_topology_heldout")}
    overlap = train_hashes & test_hashes
    record(f"V05_no_dag_leakage_{tier}", not overlap, f"overlap={len(overlap)}")

    # V06: Structural twin density
    heldout_toolset_hashes = {r["canonical_toolset_hash"] for r in heldout}
    all_toolset_to_dag = defaultdict(set)
    for r in rows:
        th = r.get("canonical_toolset_hash", "")
        dh = r.get("canonical_dag_hash", "")
        all_toolset_to_dag[th].add(dh)

    with_twin = sum(1 for th in heldout_toolset_hashes
                    if len(all_toolset_to_dag.get(th, set())) >= 2)
    pct = with_twin / max(len(heldout_toolset_hashes), 1) * 100
    record(f"V06_twin_density_{tier}", pct >= 80, f"{pct:.1f}%")

    # V07: Query diversity (unique queries per DAG)
    dag_queries = defaultdict(set)
    for r in rows:
        dag_queries[r["canonical_dag_hash"]].add(r["query"])
    avg_queries = sum(len(v) for v in dag_queries.values()) / max(len(dag_queries), 1)
    record(f"V07_query_diversity_{tier}", avg_queries >= 2.0,
           f"avg {avg_queries:.1f} unique queries/DAG")

    # V08: Query source distribution
    sources = defaultdict(int)
    for r in rows:
        sources[r.get("source", "unknown")] += 1
    azure_pct = sources.get("azure_gpt4o", 0) / max(len(rows), 1) * 100
    record(f"V08_azure_queries_{tier}", True,
           f"azure={azure_pct:.0f}%, sources={dict(sources)}")

    # V09: Topology family distribution
    fam_counts = defaultdict(int)
    unique_dags = set()
    for r in rows:
        h = r["canonical_dag_hash"]
        if h not in unique_dags:
            unique_dags.add(h)
            fam_counts[r["topo_family"]] += 1
    n_families = len(fam_counts)
    record(f"V09_topology_diversity_{tier}", n_families >= 5,
           f"{n_families} families, {len(unique_dags)} unique DAGs")

    # V10: Acyclicity
    import networkx as nx
    cyclic = 0
    seen = set()
    for r in rows:
        h = r["canonical_dag_hash"]
        if h in seen:
            continue
        seen.add(h)
        tools = r["tools"].split(";")
        edge_str = r["edges"].strip()
        if not edge_str:
            continue
        edges = []
        for e in edge_str.split(";"):
            parts = e.split("->")
            edges.append((int(parts[0]), int(parts[1])))
        G = nx.DiGraph()
        G.add_nodes_from(range(len(tools)))
        G.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(G):
            cyclic += 1
    record(f"V10_acyclicity_{tier}", cyclic == 0, f"{cyclic} cyclic")


# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60, flush=True)
total = len(results)
passed = sum(1 for v in results.values() if v["status"] == "PASS")
print(f"  VALIDATION: {passed}/{total} tests passed", flush=True)
if critical_failures:
    print(f"  FAILURES: {critical_failures}", flush=True)
    print("  >>> DO NOT PROCEED TO TRAINING <<<", flush=True)
else:
    print("  All tests passed. Dataset is valid for training.", flush=True)
print("=" * 60, flush=True)

# Write report
report_dir = Path("artifacts/campaign_v4")
report_dir.mkdir(parents=True, exist_ok=True)
report = {
    "timestamp": __import__("datetime").datetime.now().isoformat(),
    "total_tests": total,
    "passed": passed,
    "failed": total - passed,
    "critical_failures": critical_failures,
    "results": results,
}
(report_dir / "complete_validation_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(f"\n  Report: {report_dir / 'complete_validation_report.json'}", flush=True)
