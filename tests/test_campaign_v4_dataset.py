"""Phase 4: Dataset structural tests for Campaign v4."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

from src.data.tool_registry import get_tools, validate_registry, TOOL_TO_CATEGORY
from src.data.topology_templates import HELDOUT_FAMILY_NAMES, TRAINING_FAMILY_NAMES


CAMPAIGN_DIR = Path("data/campaign_v4")
TIERS = [15, 30, 45]

results = {}
critical_failures = []


def record(test_name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results[test_name] = {"status": status, "detail": detail}
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))
    if not passed:
        critical_failures.append(test_name)


# ─── T01: Tool registry invariants ───
reg = validate_registry()
record("T01_registry_invariants", reg["all_passed"])

# ─── T02: JSON files exist and are valid ───
for tier in TIERS:
    p = CAMPAIGN_DIR / f"tools_{tier}.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        record(f"T02_json_{tier}", data["tool_count"] == tier)
    except Exception as e:
        record(f"T02_json_{tier}", False, str(e))

# ─── T03: CSV files exist per tier ───
expected_csvs = {"train.csv", "dev.csv", "test_indomain.csv", "test_topology_heldout.csv", "candidate_corpus.csv"}
for tier in TIERS:
    tier_dir = CAMPAIGN_DIR / f"campaign_v4_{tier}tools"
    found = {f.name for f in tier_dir.glob("*.csv")}
    record(f"T03_csvs_exist_{tier}", expected_csvs <= found, f"found={found}")

# ─── T04: Schema compatibility ───
required_cols = {"query", "dag_id", "dag_text", "tools", "edges", "topo_family", "source", "split"}
for tier in TIERS:
    train_csv = CAMPAIGN_DIR / f"campaign_v4_{tier}tools" / "train.csv"
    with open(train_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
    record(f"T04_schema_compat_{tier}", required_cols <= cols, f"found={sorted(cols)[:8]}...")

# ─── T05: Nested tool sets ───
for tier in TIERS:
    tier_dir = CAMPAIGN_DIR / f"campaign_v4_{tier}tools"
    all_tools_used = set()
    for csv_file in tier_dir.glob("*.csv"):
        with open(csv_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_tools_used.update(row["tools"].split(";"))
    vocab = set(get_tools(tier))
    record(f"T05_tools_within_vocab_{tier}", all_tools_used <= vocab,
           f"extra={all_tools_used - vocab}" if all_tools_used - vocab else "all in vocab")

# ─── T06: No heldout topology in train/val ───
for tier in TIERS:
    for split_name in ["train.csv", "dev.csv"]:
        csv_path = CAMPAIGN_DIR / f"campaign_v4_{tier}tools" / split_name
        with open(csv_path, "r", encoding="utf-8") as f:
            families = {row["topo_family"] for row in csv.DictReader(f)}
        leaked = families & HELDOUT_FAMILY_NAMES
        record(f"T06_no_heldout_leak_{tier}_{split_name}", not leaked,
               f"leaked={leaked}" if leaked else "")

# ─── T07: Heldout test only has heldout families ───
for tier in TIERS:
    csv_path = CAMPAIGN_DIR / f"campaign_v4_{tier}tools" / "test_topology_heldout.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        families = {row["topo_family"] for row in csv.DictReader(f)}
    record(f"T07_heldout_only_heldout_{tier}", families <= HELDOUT_FAMILY_NAMES,
           f"found={families}")

# ─── T08: No duplicate labeled DAGs across train/test ───
for tier in TIERS:
    tier_dir = CAMPAIGN_DIR / f"campaign_v4_{tier}tools"
    split_hashes = defaultdict(set)
    for csv_file in tier_dir.glob("*.csv"):
        with open(csv_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "canonical_dag_hash" in row:
                    split_name = csv_file.stem
                    split_hashes[split_name].add(row["canonical_dag_hash"])

    train_hashes = split_hashes.get("train", set())
    test_hashes = split_hashes.get("test_topology_heldout", set()) | split_hashes.get("test_indomain", set())
    overlap = train_hashes & test_hashes
    record(f"T08_no_dag_leakage_{tier}", not overlap,
           f"overlap={len(overlap)}" if overlap else "")

# ─── T09: Structural twins exist in heldout ───
for tier in TIERS:
    csv_path = CAMPAIGN_DIR / f"campaign_v4_{tier}tools" / "test_topology_heldout.csv"
    all_csvs = list((CAMPAIGN_DIR / f"campaign_v4_{tier}tools").glob("*.csv"))

    heldout_toolset_hashes = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            heldout_toolset_hashes.add(row.get("canonical_toolset_hash", ""))

    all_toolset_to_dag_hashes = defaultdict(set)
    for csv_file in all_csvs:
        with open(csv_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                th = row.get("canonical_toolset_hash", "")
                dh = row.get("canonical_dag_hash", "")
                all_toolset_to_dag_hashes[th].add(dh)

    heldout_with_twin = sum(
        1 for th in heldout_toolset_hashes
        if len(all_toolset_to_dag_hashes.get(th, set())) >= 2
    )
    pct = heldout_with_twin / max(len(heldout_toolset_hashes), 1) * 100
    record(f"T09_twin_density_{tier}", pct >= 80,
           f"{pct:.1f}% (target >= 80%)")

# ─── T10: DAG acyclicity validation ───
import networkx as nx

acyclic_failures = 0
total_dags = 0
for tier in TIERS:
    tier_dir = CAMPAIGN_DIR / f"campaign_v4_{tier}tools"
    seen = set()
    for csv_file in tier_dir.glob("*.csv"):
        with open(csv_file, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dag_hash = row.get("canonical_dag_hash", row.get("dag_id", ""))
                if dag_hash in seen:
                    continue
                seen.add(dag_hash)
                total_dags += 1

                tools = row["tools"].split(";")
                edge_str = row["edges"]
                if not edge_str.strip():
                    continue
                edges = []
                for e in edge_str.split(";"):
                    parts = e.split("->")
                    edges.append((int(parts[0]), int(parts[1])))

                G = nx.DiGraph()
                G.add_nodes_from(range(len(tools)))
                G.add_edges_from(edges)
                if not nx.is_directed_acyclic_graph(G):
                    acyclic_failures += 1

record("T10_acyclicity", acyclic_failures == 0,
       f"{acyclic_failures}/{total_dags} cyclic DAGs found")

# ─── T11: Backward-compatible load test ───
import pandas as pd

def parse_tools(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [x.strip() for x in str(cell).split(";") if x.strip()]

def parse_edges(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    text = str(cell).strip()
    if not text:
        return []
    out = []
    for part in text.split(";"):
        part = part.strip()
        if "->" in part:
            s, d = part.split("->")
            out.append((int(s.strip()), int(d.strip())))
    return out

load_errors = 0
for tier in TIERS:
    train_csv = CAMPAIGN_DIR / f"campaign_v4_{tier}tools" / "train.csv"
    df = pd.read_csv(train_csv)
    for _, row in df.iterrows():
        try:
            tools = parse_tools(row["tools"])
            edges = parse_edges(row["edges"])
            assert len(tools) > 0
            for s, d in edges:
                assert 0 <= s < len(tools) and 0 <= d < len(tools)
        except Exception:
            load_errors += 1

record("T11_backward_compat_load", load_errors == 0,
       f"{load_errors} load errors")

# ─── T12: Category balance ───
for tier in TIERS:
    tools = get_tools(tier)
    cats = defaultdict(int)
    for t in tools:
        cats[TOOL_TO_CATEGORY[t].value] += 1
    expected = {
        15: {"DATA_RETRIEVAL": 6, "STATE_MODIFICATION": 6, "ORCHESTRATION": 3},
        30: {"DATA_RETRIEVAL": 12, "STATE_MODIFICATION": 12, "ORCHESTRATION": 6},
        45: {"DATA_RETRIEVAL": 18, "STATE_MODIFICATION": 18, "ORCHESTRATION": 9},
    }[tier]
    record(f"T12_category_balance_{tier}", dict(cats) == expected, f"got={dict(cats)}")

# ─── T13: Manifest exists and is valid ───
manifest_path = CAMPAIGN_DIR / "campaign_manifest.json"
try:
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    record("T13_manifest", "campaign" in m and "tiers" in m)
except Exception as e:
    record("T13_manifest", False, str(e))

# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
total = len(results)
passed = sum(1 for v in results.values() if v["status"] == "PASS")
print(f"  RESULTS: {passed}/{total} tests passed")
if critical_failures:
    print(f"  CRITICAL FAILURES: {critical_failures}")
    print("  >>> DO NOT PROCEED TO TRAINING <<<")
else:
    print("  All tests passed. Safe to proceed.")
print("="*60)

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
(report_dir / "dataset_test_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

with open(report_dir / "dataset_test_report.md", "w", encoding="utf-8") as f:
    f.write("# Campaign v4 Dataset Test Report\n\n")
    f.write(f"**Date:** {report['timestamp']}\n\n")
    f.write(f"**Result:** {passed}/{total} tests passed\n\n")
    f.write("| Test | Status | Detail |\n|------|--------|--------|\n")
    for name, info in sorted(results.items()):
        f.write(f"| {name} | {info['status']} | {info['detail']} |\n")
    if critical_failures:
        f.write(f"\n**CRITICAL FAILURES:** {', '.join(critical_failures)}\n")
    else:
        f.write("\nAll tests passed. Safe to proceed to training.\n")

print(f"\n  Report: {report_dir / 'dataset_test_report.md'}")
