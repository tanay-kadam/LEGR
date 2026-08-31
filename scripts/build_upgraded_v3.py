"""
build_upgraded_v3.py — split fixes on top of the v2 query regeneration
======================================================================

Produces ``upgraded_v3/``, which differs from ``upgraded/`` in two ways that
``upgraded_v2/`` deliberately did *not* touch, because both change the splits and
therefore invalidate existing checkpoints:

1.  **Diamond genuinely held out at 15 and 30 tools.** Today the diamond family
    appears in *both* train and test at those tiers (48 and 52 train rows), so the
    "topology held out" claim only holds at 45 tools. Every row whose whole-graph
    shape is the canonical 4-node diamond is moved out of train/dev into test.
2.  **Single-node graphs across the full tool vocabulary.** They exist today but
    cover only 2/15, 1/30 and 8/45 tools, and appear in **no** test split, so
    single-node retrieval cannot currently be evaluated at all. v3 gives every
    tool in the tier a single-node DAG with rows in train, dev and test.

On the single-node DAGs, train and test intentionally share the same labelled
DAG. That is not leakage: in a retrieval comparison against the taxonomy router,
the candidate corpus is supposed to contain one entry per tool and the router
likewise knows every tool. What is held out is the query phrasing, which the v2
generator guarantees is disjoint across splits.

Queries are placeholders here; run ``scripts/regenerate_queries_v2.py`` against
``upgraded_v3`` afterwards to fill them.

Usage
-----
    python scripts/build_upgraded_v3.py
    python scripts/regenerate_queries_v2.py --source-root upgraded_v3 \
        --out-root upgraded_v3 --report-dir upgraded_v3_reports
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import query_library as QL  # noqa: E402

warnings.filterwarnings("ignore", message=".*hashes produced.*")

TIERS = ("15", "30", "45")
SPLITS = ("train", "dev", "test_topology_heldout")
DIAMOND_EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]


def parse_tools(cell: object) -> List[str]:
    return [t.strip() for t in str(cell).split(";") if t.strip()]


def parse_edges(cell: object) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in str(cell).split(";"):
        part = part.strip()
        if "->" in part:
            src, dst = part.split("->", 1)
            out.append((int(src), int(dst)))
    return out


def structure_hash(num_nodes: int, edges: List[Tuple[int, int]]) -> str:
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from([(s, d) for s, d in edges if s < num_nodes and d < num_nodes])
    return nx.weisfeiler_lehman_graph_hash(G, iterations=3)


DIAMOND = structure_hash(4, DIAMOND_EDGES)


def build_tier(
    tier: str,
    source_root: Path,
    out_root: Path,
    holdout_diamond: bool,
    per_tool: Dict[str, int],
) -> List[dict]:
    src_dir = source_root / f"upgraded_{tier}tools"
    out_dir = out_root / f"upgraded_{tier}tools"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        df = pd.read_csv(src_dir / f"{split}.csv", keep_default_na=False, dtype=str)
        frames[split] = df
    columns = list(frames["train"].columns)

    # Value used in the 'split' column by each destination file.
    split_label = {
        s: (frames[s]["split"].mode().iat[0] if len(frames[s]) else s) for s in SPLITS
    }

    stats: List[dict] = []
    before = {s: len(frames[s]) for s in SPLITS}

    # ── 1. Diamond holdout ────────────────────────────────────────────────
    moved = 0
    if holdout_diamond:
        for source_split in ("train", "dev"):
            df = frames[source_split]
            shapes = [
                structure_hash(len(parse_tools(t)), parse_edges(e))
                for t, e in zip(df["tools"], df["edges"])
            ]
            is_diamond = pd.Series(shapes, index=df.index) == DIAMOND
            if not is_diamond.any():
                continue
            movers = df[is_diamond].copy()
            movers["split"] = split_label["test_topology_heldout"]
            frames[source_split] = df[~is_diamond].reset_index(drop=True)
            frames["test_topology_heldout"] = pd.concat(
                [frames["test_topology_heldout"], movers], ignore_index=True
            )
            moved += len(movers)

    # ── 2. Single-node coverage over the full vocabulary ──────────────────
    vocab = QL.TOOL_VOCAB[: int(tier)]
    all_rows = pd.concat(frames.values(), ignore_index=True)
    next_dag_id = all_rows["dag_id"].astype(int).max() + 1

    # Reuse an existing single-node dag_id per tool so one labelled DAG keeps
    # one id rather than being duplicated under a fresh one.
    existing_id: Dict[str, str] = {}
    for _, row in all_rows.iterrows():
        tools = parse_tools(row["tools"])
        if len(tools) == 1:
            existing_id.setdefault(tools[0], row["dag_id"])

    existing_counts: Dict[Tuple[str, str], int] = {}
    for split in SPLITS:
        df = frames[split]
        for _, row in df.iterrows():
            tools = parse_tools(row["tools"])
            if len(tools) == 1:
                key = (tools[0], split)
                existing_counts[key] = existing_counts.get(key, 0) + 1

    added = 0
    for tool in vocab:
        if tool in existing_id:
            dag_id = existing_id[tool]
        else:
            dag_id = str(next_dag_id)
            next_dag_id += 1
            existing_id[tool] = dag_id

        for split in SPLITS:
            target = per_tool[split]
            have = existing_counts.get((tool, split), 0)
            need = max(0, target - have)
            if not need:
                continue
            new_rows = []
            for _ in range(need):
                record = {
                    "query": "",
                    "dag_id": dag_id,
                    "dag_text": tool,
                    "tools": tool,
                    "edges": "",
                    "topo_family": "single_node",
                    "source": "single_node_v3",
                    "split": split_label[split],
                }
                if "strict_fix_applied" in columns:
                    record["strict_fix_applied"] = "FALSE"
                    record["had_duplicate_node_labels"] = "FALSE"
                    record["original_tools"] = tool
                new_rows.append(record)
            frames[split] = pd.concat(
                [frames[split], pd.DataFrame(new_rows, columns=columns)],
                ignore_index=True,
            )
            added += need

    for split in SPLITS:
        out = frames[split][columns]
        out.to_csv(out_dir / f"{split}.csv", index=False)
        single = sum(
            1 for t in out["tools"] if len(parse_tools(t)) == 1
        )
        stats.append(
            {
                "tier": tier,
                "split": split,
                "rows_before": before[split],
                "rows_after": len(out),
                "dags": out["dag_id"].nunique(),
                "single_node_rows": single,
                "single_node_tools": len({parse_tools(t)[0] for t in out["tools"] if len(parse_tools(t)) == 1}),
            }
        )

    stats.append(
        {
            "tier": tier,
            "split": "(actions)",
            "rows_before": moved,
            "rows_after": added,
            "dags": "",
            "single_node_rows": "",
            "single_node_tools": "",
        }
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", default="upgraded")
    ap.add_argument("--out-root", default="upgraded_v3")
    ap.add_argument("--report-dir", default="upgraded_v3_reports")
    ap.add_argument(
        "--holdout-diamond-tiers",
        nargs="*",
        default=["15", "30"],
        help="45 tools already holds diamond out, so it is excluded by default.",
    )
    ap.add_argument("--single-node-train", type=int, default=12)
    ap.add_argument("--single-node-dev", type=int, default=3)
    ap.add_argument("--single-node-test", type=int, default=5)
    args = ap.parse_args()

    per_tool = {
        "train": args.single_node_train,
        "dev": args.single_node_dev,
        "test_topology_heldout": args.single_node_test,
    }

    source_root = ROOT / args.source_root
    out_root = ROOT / args.out_root
    report_dir = ROOT / args.report_dir

    all_stats: List[dict] = []
    for tier in TIERS:
        all_stats.extend(
            build_tier(
                tier,
                source_root,
                out_root,
                holdout_diamond=tier in args.holdout_diamond_tiers,
                per_tool=per_tool,
            )
        )

    summary = pd.DataFrame(all_stats)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(report_dir / "v3_build_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(
        "\nRows shown under '(actions)': rows_before = diamond rows moved to test, "
        "rows_after = single-node rows added."
    )
    print("\nNext: python scripts/regenerate_queries_v2.py --source-root upgraded_v3 "
          "--out-root upgraded_v3 --report-dir upgraded_v3_reports")


if __name__ == "__main__":
    main()
