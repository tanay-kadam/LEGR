"""
upgrade_graph.py -- Part B: Upgrade the DAG / graph retrieval dataset
=====================================================================

Reads test_corpus_for_all_models.csv and produces:

  1. base_validated.csv          -- original data with validation flags
  2. generated_augmented.csv     -- new DAGs from diverse topology families
  3. hard_negatives.csv          -- hard-negative DAG variants
  4. train.csv / dev.csv / test_topology_heldout.csv -- topology-aware splits
  5. topology_report.json / .md  -- topology analysis

Uses utils/graph_utils.py for all DAG operations and reuses
data_synth._synthesize_queries / _fill for query generation.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.graph_utils import (
    TOOL_VOCAB,
    TOPOLOGY_GENERATORS,
    build_dag_from_row,
    classify_topology,
    edges_to_str,
    generate_dags,
    generate_hard_negatives,
    labeled_dag_hash,
    parse_edges,
    parse_tools,
    tools_to_str,
    topology_hash,
    validate_dag,
    _dag_to_text,
)

# Try to import the project's query synthesiser for richer query generation
try:
    from data_synth import _synthesize_queries, _fill
    HAS_DATA_SYNTH = True
except ImportError:
    HAS_DATA_SYNTH = False


def load_config() -> Dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Entity-fill for standalone query generation
# ─────────────────────────────────────────────────────────────────────────────

_ENTITY_POOLS = {
    "user":   ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"],
    "order":  ["#10234", "#20891", "#31450", "#42017", "#53698", "#60122"],
    "server": ["prod-web-01", "staging-db-02", "payment-api-03",
               "auth-svc-04", "ml-infer-05", "cdn-edge-06"],
    "dept":   ["Engineering", "Finance", "Marketing", "Legal", "HR"],
    "ticket": ["INC-4021", "INC-7733", "INC-1198", "INC-5560", "INC-8842"],
}


def _standalone_fill(template: str, rng: random.Random) -> str:
    result = template
    for key, pool in _ENTITY_POOLS.items():
        tag = "{" + key + "}"
        while tag in result:
            result = result.replace(tag, rng.choice(pool), 1)
    return result


def generate_queries_for_dag(
    tools: List[str],
    edges: List[Tuple[int, int]],
    n: int,
    seed_offset: int,
) -> List[str]:
    """Generate natural-language queries for a DAG using data_synth if available."""
    rng = random.Random(42 + seed_offset)

    if HAS_DATA_SYNTH:
        raw = _synthesize_queries(tools, edges, seed_offset=seed_offset, n=n)
        return [_fill(q, rng) for q in raw]

    # Fallback: simple template-based generation
    queries = []
    for i in range(n):
        if len(tools) == 1:
            q = _standalone_fill(f"Handle the {tools[0]} task for {{user}}.", rng)
        elif not edges:
            tool_list = " and ".join(tools)
            q = _standalone_fill(f"Run {tool_list} for {{dept}}.", rng)
        else:
            parts = []
            for j, t in enumerate(tools):
                parts.append(t.replace("_", " "))
            q = _standalone_fill(
                f"Execute: {', then '.join(parts)} for {{user}}.", rng
            )
        queries.append(q)
    return queries


# ─────────────────────────────────────────────────────────────────────────────
#  Load and validate existing data
# ─────────────────────────────────────────────────────────────────────────────

def load_and_validate(cfg: Dict) -> pd.DataFrame:
    """Load the graph dataset, parse, and validate every row."""
    root = Path(__file__).resolve().parent.parent
    gcfg = cfg["graph"]
    path = root / gcfg["input_path"]
    df = pd.read_csv(path)

    print(f"  Graph dataset: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    tools_col = gcfg["tools_col"]
    edges_col = gcfg["edges_col"]

    parsed_tools = df[tools_col].apply(parse_tools)
    parsed_edges = df[edges_col].apply(parse_edges)

    valid_flags = []
    topo_hashes = []
    topo_families = []
    dag_hashes = []

    for i in range(len(df)):
        tools = parsed_tools.iloc[i]
        edges = parsed_edges.iloc[i]
        vr = validate_dag(tools, edges)
        valid_flags.append(vr["valid"])

        if vr["valid"]:
            th = topology_hash(edges, len(tools))
            G = build_dag_from_row(tools, edges)
            dh = labeled_dag_hash(G)
            fam = classify_topology(edges, len(tools))
        else:
            th = ""
            dh = ""
            fam = "invalid"

        topo_hashes.append(th)
        dag_hashes.append(dh)
        topo_families.append(fam)

    df["_valid"] = valid_flags
    df["_topo_hash"] = topo_hashes
    df["_dag_hash"] = dag_hashes
    df["_topo_family"] = topo_families

    n_valid = sum(valid_flags)
    n_invalid = len(df) - n_valid
    print(f"  Valid DAGs  : {n_valid}")
    print(f"  Invalid DAGs: {n_invalid}")
    print(f"  Unique topology hashes: {len(set(topo_hashes) - {''})}")
    print(f"  Unique labeled DAG hashes: {len(set(dag_hashes) - {''})}")
    print(f"  Topology families: {dict(Counter(topo_families))}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Generate augmented DAGs
# ─────────────────────────────────────────────────────────────────────────────

def generate_augmented_data(
    existing_hashes: set,
    cfg: Dict,
) -> pd.DataFrame:
    """Generate new DAGs from diverse topology families."""
    gcfg = cfg["graph"]
    seed = cfg["seed"]
    n_dags = gcfg["n_new_dags"]
    families = gcfg["new_dag_families"]
    queries_per = gcfg["queries_per_dag"]

    print(f"\n  Generating {n_dags} new DAGs across {len(families)} families...")

    new_dags = generate_dags(
        n_dags=n_dags,
        families=families,
        vocab=TOOL_VOCAB,
        seed=seed,
    )

    # Filter out any that collide with existing
    filtered = [d for d in new_dags if d["dag_hash"] not in existing_hashes]
    print(f"  Generated {len(new_dags)} DAGs, {len(filtered)} are novel")

    rows = []
    dag_id_offset = 10000

    for i, dag_info in enumerate(filtered):
        tools = dag_info["tools"]
        edges = dag_info["edges"]
        dag_text = dag_info["dag_text"]
        family = dag_info["family"]
        dag_id = dag_id_offset + i

        queries = generate_queries_for_dag(
            tools, edges, n=queries_per, seed_offset=seed + i * 7
        )

        for q in queries:
            rows.append({
                "query": q,
                "dag_id": dag_id,
                "dag_text": dag_text,
                "tools": tools_to_str(tools),
                "edges": edges_to_str(edges),
                "source": "generated",
                "topo_family": family,
                "dag_hash": dag_info["dag_hash"],
            })

    result = pd.DataFrame(rows)
    print(f"  Augmented dataset: {len(result)} rows, "
          f"{result['dag_id'].nunique()} unique DAGs")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Hard negatives
# ─────────────────────────────────────────────────────────────────────────────

def build_hard_negatives(
    df: pd.DataFrame,
    cfg: Dict,
) -> pd.DataFrame:
    """Generate hard-negative DAGs for retrieval evaluation."""
    seed = cfg["seed"]
    rng = random.Random(seed + 100)
    gcfg = cfg["graph"]

    tools_col = gcfg["tools_col"]
    edges_col = gcfg["edges_col"]

    # Only process unique valid DAGs
    valid_df = df[df.get("_valid", True) == True] if "_valid" in df.columns else df

    # Get unique DAGs by dag_id
    seen_dag_ids = set()
    unique_rows = []
    for _, row in valid_df.iterrows():
        did = row.get("dag_id", row.name)
        if did in seen_dag_ids:
            continue
        seen_dag_ids.add(did)
        unique_rows.append(row)

    print(f"\n  Generating hard negatives for {len(unique_rows)} unique DAGs...")

    neg_rows = []
    neg_dag_counter = 0

    for row in unique_rows:
        tools = parse_tools(row[tools_col])
        edges = parse_edges(row[edges_col])
        dag_id = row.get("dag_id", row.name)
        query = row.get("query", "")

        negatives = generate_hard_negatives(tools, edges, rng, TOOL_VOCAB)

        for neg in negatives:
            neg_rows.append({
                "query": query,
                "positive_dag_id": dag_id,
                "negative_dag_id": f"neg_{neg_dag_counter}",
                "negative_type": neg["negative_type"],
                "neg_tools": tools_to_str(neg["neg_tools"]),
                "neg_edges": edges_to_str(neg["neg_edges"]),
                "neg_dag_text": neg["neg_dag_text"],
            })
            neg_dag_counter += 1

    result = pd.DataFrame(neg_rows)
    print(f"  Hard negatives: {len(result)} rows")
    if len(result) > 0:
        type_counts = dict(Counter(result["negative_type"]))
        print(f"  By type: {type_counts}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Topology-held-out splits
# ─────────────────────────────────────────────────────────────────────────────

def build_topology_splits(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/dev/test splits where some topology families are held out."""
    gcfg = cfg["graph"]
    seed = cfg["seed"]
    rng = random.Random(seed + 200)

    train_families = set(gcfg["train_topology_families"])
    test_families = set(gcfg["test_topology_families"])

    # Combine base and augmented
    shared_cols = ["query", "dag_id", "dag_text", "tools", "edges"]

    base_rows = []
    for _, row in base_df.iterrows():
        if "_valid" in base_df.columns and not row["_valid"]:
            continue
        family = row.get("_topo_family", "unknown")
        base_rows.append({
            "query": row["query"],
            "dag_id": row["dag_id"],
            "dag_text": row["dag_text"],
            "tools": row["tools"],
            "edges": row["edges"],
            "topo_family": family,
            "source": "original",
        })

    aug_rows = []
    for _, row in aug_df.iterrows():
        aug_rows.append({
            "query": row["query"],
            "dag_id": row["dag_id"],
            "dag_text": row["dag_text"],
            "tools": row["tools"],
            "edges": row["edges"],
            "topo_family": row.get("topo_family", "unknown"),
            "source": "generated",
        })

    all_rows = base_rows + aug_rows
    all_df = pd.DataFrame(all_rows)

    if len(all_df) == 0:
        empty = pd.DataFrame(columns=["query", "dag_id", "dag_text", "tools",
                                       "edges", "topo_family", "source", "split"])
        return empty, empty.copy(), empty.copy()

    # Assign splits
    # Priority: test_families -> test, train_families -> train, rest -> split proportionally
    all_df["split"] = "unassigned"

    for idx, row in all_df.iterrows():
        fam = row["topo_family"]
        if fam in test_families:
            all_df.at[idx, "split"] = "test"
        elif fam in train_families:
            all_df.at[idx, "split"] = "train"

    # Unassigned families get split proportionally
    unassigned = all_df[all_df["split"] == "unassigned"]
    if len(unassigned) > 0:
        unassigned_families = list(unassigned["topo_family"].unique())
        rng.shuffle(unassigned_families)

        n_un = len(unassigned_families)
        n_dev = max(1, int(n_un * gcfg["dev_frac"]))
        n_test_extra = max(1, int(n_un * gcfg["test_frac"]))

        dev_families = set(unassigned_families[:n_dev])
        test_extra = set(unassigned_families[n_dev:n_dev + n_test_extra])
        train_extra = set(unassigned_families[n_dev + n_test_extra:])

        for idx, row in all_df.iterrows():
            if all_df.at[idx, "split"] != "unassigned":
                continue
            fam = row["topo_family"]
            if fam in dev_families:
                all_df.at[idx, "split"] = "dev"
            elif fam in test_extra:
                all_df.at[idx, "split"] = "test"
            else:
                all_df.at[idx, "split"] = "train"

    # Any remaining unassigned -> train
    all_df.loc[all_df["split"] == "unassigned", "split"] = "train"

    train = all_df[all_df["split"] == "train"].reset_index(drop=True)
    dev = all_df[all_df["split"] == "dev"].reset_index(drop=True)
    test = all_df[all_df["split"] == "test"].reset_index(drop=True)

    print(f"\n  Topology-held-out splits:")
    print(f"    Train: {len(train)} rows, {train['topo_family'].nunique()} families")
    print(f"    Dev  : {len(dev)} rows, {dev['topo_family'].nunique()} families")
    print(f"    Test : {len(test)} rows, {test['topo_family'].nunique()} families")

    return train, dev, test


# ─────────────────────────────────────────────────────────────────────────────
#  Topology report
# ─────────────────────────────────────────────────────────────────────────────

def build_topology_report(
    base_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict,
) -> Dict:
    """Build comprehensive topology analysis report."""
    gcfg = cfg["graph"]

    def count_stats(df: pd.DataFrame) -> Dict:
        if len(df) == 0:
            return {"total_rows": 0}

        tools_col = "tools"
        queries = df["query"].tolist() if "query" in df.columns else []
        node_counts = []
        for _, row in df.iterrows():
            tools = parse_tools(row.get(tools_col, ""))
            node_counts.append(len(tools))

        return {
            "total_rows": len(df),
            "unique_dag_ids": int(df["dag_id"].nunique()) if "dag_id" in df.columns else 0,
            "avg_query_length": round(
                sum(len(q) for q in queries) / max(len(queries), 1), 1
            ) if queries else 0,
            "node_count_distribution": dict(Counter(node_counts)),
        }

    # Base topology analysis
    base_families = dict(Counter(
        base_df["_topo_family"].tolist()
    )) if "_topo_family" in base_df.columns else {}

    base_topo_hashes = set(
        base_df["_topo_hash"].tolist()
    ) if "_topo_hash" in base_df.columns else set()
    base_topo_hashes.discard("")

    # Augmented topology analysis
    aug_families = dict(Counter(
        aug_df["topo_family"].tolist()
    )) if "topo_family" in aug_df.columns and len(aug_df) > 0 else {}

    # Combined
    all_families = Counter()
    all_families.update(base_families)
    all_families.update(aug_families)

    # Train/test overlap
    train_fams = set(train_df["topo_family"].unique()) if len(train_df) > 0 else set()
    dev_fams = set(dev_df["topo_family"].unique()) if len(dev_df) > 0 else set()
    test_fams = set(test_df["topo_family"].unique()) if len(test_df) > 0 else set()

    report = {
        "base_stats": count_stats(base_df),
        "augmented_stats": count_stats(aug_df),
        "base_unique_topo_hashes": len(base_topo_hashes),
        "base_topo_families": base_families,
        "augmented_topo_families": aug_families,
        "combined_topo_families": dict(all_families),
        "hard_negative_count": len(neg_df),
        "hard_negative_types": dict(Counter(
            neg_df["negative_type"].tolist()
        )) if len(neg_df) > 0 and "negative_type" in neg_df.columns else {},
        "train_stats": count_stats(train_df),
        "dev_stats": count_stats(dev_df),
        "test_stats": count_stats(test_df),
        "train_families": sorted(train_fams),
        "dev_families": sorted(dev_fams),
        "test_families": sorted(test_fams),
        "train_test_family_overlap": sorted(train_fams & test_fams),
        "topology_diversity": {
            "before": len(base_topo_hashes),
            "after_base_plus_aug": len(base_topo_hashes) + len(
                set(aug_df["dag_hash"].tolist()) if "dag_hash" in aug_df.columns and len(aug_df) > 0 else set()
            ),
        },
    }

    # Duplicate analysis
    if "query" in base_df.columns:
        base_queries = base_df["query"].tolist()
        report["base_exact_dup_queries"] = len(base_queries) - len(set(base_queries))
        norm = [q.lower().strip() for q in base_queries]
        report["base_normalized_dup_queries"] = len(norm) - len(set(norm))

    return report


def write_topology_report_md(report: Dict, path: Path) -> None:
    """Write Markdown topology report."""
    lines = ["# Graph Dataset Topology Report\n"]

    bs = report.get("base_stats", {})
    lines.append("## Original Dataset\n")
    lines.append(f"- Total rows: {bs.get('total_rows', 'N/A')}")
    lines.append(f"- Unique DAG IDs: {bs.get('unique_dag_ids', 'N/A')}")
    lines.append(f"- Unique topology hashes: {report.get('base_unique_topo_hashes', 'N/A')}")
    lines.append(f"- Avg query length: {bs.get('avg_query_length', 'N/A')}")
    nc = bs.get("node_count_distribution", {})
    if nc:
        lines.append(f"- Node count distribution: {nc}")
    lines.append("")

    bf = report.get("base_topo_families", {})
    if bf:
        lines.append("### Original Topology Families\n")
        for fam, count in sorted(bf.items(), key=lambda x: -x[1]):
            lines.append(f"- {fam}: {count}")
        lines.append("")

    aus = report.get("augmented_stats", {})
    lines.append("## Augmented Dataset\n")
    lines.append(f"- Total rows: {aus.get('total_rows', 'N/A')}")
    lines.append(f"- Unique DAG IDs: {aus.get('unique_dag_ids', 'N/A')}")
    af = report.get("augmented_topo_families", {})
    if af:
        lines.append("### Augmented Topology Families\n")
        for fam, count in sorted(af.items(), key=lambda x: -x[1]):
            lines.append(f"- {fam}: {count}")
        lines.append("")

    lines.append("## Hard Negatives\n")
    lines.append(f"- Total: {report.get('hard_negative_count', 0)}")
    ht = report.get("hard_negative_types", {})
    if ht:
        for t, c in sorted(ht.items()):
            lines.append(f"- {t}: {c}")
    lines.append("")

    lines.append("## Topology-Held-Out Splits\n")
    for split_name in ["train", "dev", "test"]:
        ss = report.get(f"{split_name}_stats", {})
        fams = report.get(f"{split_name}_families", [])
        lines.append(f"### {split_name.capitalize()}\n")
        lines.append(f"- Rows: {ss.get('total_rows', 0)}")
        lines.append(f"- Unique DAGs: {ss.get('unique_dag_ids', 0)}")
        lines.append(f"- Families: {', '.join(fams) if fams else 'none'}")
        lines.append("")

    overlap = report.get("train_test_family_overlap", [])
    lines.append(f"### Train-Test Topology Overlap: {len(overlap)} families\n")
    if overlap:
        lines.append(f"Overlapping: {', '.join(overlap)}")
    lines.append("")

    td = report.get("topology_diversity", {})
    lines.append("## Topology Diversity\n")
    lines.append(f"- Before: {td.get('before', 'N/A')} unique topologies")
    lines.append(f"- After (base + augmented): {td.get('after_base_plus_aug', 'N/A')} unique topologies")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(output_dir_override: str | None = None) -> None:
    print("\n" + "=" * 60)
    print("  Part B: Upgrade Graph Dataset")
    print("=" * 60 + "\n")

    cfg = load_config()
    gcfg = cfg["graph"]

    graph_out = output_dir_override or gcfg["output_dir"]
    out_dir = Path(__file__).resolve().parent.parent / graph_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate
    base_df = load_and_validate(cfg)
    base_path = out_dir / "base_validated.csv"
    base_df.to_csv(base_path, index=False)
    print(f"  Saved base_validated.csv")

    # 2. Collect existing hashes for dedup
    existing_hashes = set(base_df["_dag_hash"].tolist())
    existing_hashes.discard("")

    # 3. Generate augmented DAGs
    aug_df = generate_augmented_data(existing_hashes, cfg)
    aug_df.to_csv(out_dir / "generated_augmented.csv", index=False)
    print(f"  Saved generated_augmented.csv")

    # 4. Hard negatives
    neg_df = build_hard_negatives(base_df, cfg)
    neg_df.to_csv(out_dir / "hard_negatives.csv", index=False)
    print(f"  Saved hard_negatives.csv")

    # 5. Topology-held-out splits
    train_df, dev_df, test_df = build_topology_splits(base_df, aug_df, cfg)
    train_df.to_csv(out_dir / "train.csv", index=False)
    dev_df.to_csv(out_dir / "dev.csv", index=False)
    test_df.to_csv(out_dir / "test_topology_heldout.csv", index=False)
    print(f"  Saved train.csv, dev.csv, test_topology_heldout.csv")

    # 6. Topology report
    report = build_topology_report(
        base_df, aug_df, neg_df, train_df, dev_df, test_df, cfg
    )
    with open(out_dir / "topology_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_topology_report_md(report, out_dir / "topology_report.md")
    print(f"  Saved topology_report.json / .md")
    print("  Done.\n")


if __name__ == "__main__":
    main()
