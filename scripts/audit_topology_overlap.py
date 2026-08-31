"""
audit_topology_overlap.py — is the held-out test topology actually held out?
============================================================================

Answers three separate questions per tier, which the phrase "topology held out"
tends to conflate:

1.  **Family label** — does the ``topo_family`` string in test also appear in
    train? This is the weakest check and the one the paper's split claim rests on.
2.  **Unlabelled structure** — does the test DAG's *shape* appear in train under
    any family label? A row labelled ``fork_join`` or ``double_diamond`` can
    contain the same 4-node/4-edge diamond shape, so a clean family split does
    not imply an unseen structure.
3.  **Labelled DAG** — does the exact (tool multiset, tool-edge set) appear in
    train? This is leakage in the strongest sense.

The canonical diamond shape is checked explicitly, since that family is the one
the paper reserves for evaluation.

Usage
-----
    python scripts/audit_topology_overlap.py
    python scripts/audit_topology_overlap.py --root upgraded_v2
"""

from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path
from typing import List, Set, Tuple

import networkx as nx
import pandas as pd

# networkx >= 3.5 changed WL hash values. Only train-vs-test sets computed in
# this same process are ever compared, so the change is immaterial here; the
# hashes are not comparable across networkx versions.
warnings.filterwarnings("ignore", message=".*hashes produced.*")

ROOT = Path(__file__).resolve().parents[1]
TIERS = ("15", "30", "45")

# Shape used by graph_utils.gen_diamond.
DIAMOND_EDGES: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 3), (2, 3)]
DIAMOND_NODES = 4


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


def labeled_hash(tools: List[str], edges: List[Tuple[int, int]]) -> str:
    """Tool-aware hash: same definition as scripts/audit_split_leakage.py."""
    nodes = tuple(sorted(tools))
    e = tuple(
        sorted((tools[s], tools[d]) for s, d in edges if s < len(tools) and d < len(tools))
    )
    return hashlib.sha256(f"{nodes}|{e}".encode()).hexdigest()[:16]


def structure_hash(num_nodes: int, edges: List[Tuple[int, int]]) -> str:
    """Tool-blind shape hash (Weisfeiler-Lehman over node indices)."""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from([(s, d) for s, d in edges if s < num_nodes and d < num_nodes])
    return nx.weisfeiler_lehman_graph_hash(G, iterations=3)


DIAMOND_STRUCTURE = structure_hash(DIAMOND_NODES, DIAMOND_EDGES)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    df["_tools"] = df["tools"].map(parse_tools)
    df["_edges"] = df["edges"].map(parse_edges)
    df = df[df["_tools"].map(len) > 0].copy()
    df["_lab"] = [labeled_hash(t, e) for t, e in zip(df["_tools"], df["_edges"])]
    df["_str"] = [structure_hash(len(t), e) for t, e in zip(df["_tools"], df["_edges"])]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="upgraded")
    ap.add_argument(
        "--exclude-single-node",
        action="store_true",
        help=(
            "Report multi-node graphs only. Single-node DAGs are shared between "
            "train and test by design in upgraded_v3 (the retrieval corpus needs "
            "one entry per tool), so they otherwise mask the multi-node figures."
        ),
    )
    args = ap.parse_args()
    root = ROOT / args.root

    print(f"Auditing {args.root}/   (diamond structure hash = {DIAMOND_STRUCTURE[:12]}…)")

    for tier in TIERS:
        tier_dir = root / f"upgraded_{tier}tools"
        train_p = tier_dir / "train.csv"
        test_p = tier_dir / "test_topology_heldout.csv"
        if not (train_p.exists() and test_p.exists()):
            print(f"\n=== {tier} tools: files missing, skipped")
            continue

        train = load(train_p)
        test = load(test_p)
        if args.exclude_single_node:
            train = train[train["_tools"].map(len) > 1]
            test = test[test["_tools"].map(len) > 1]

        tr_fam: Set[str] = set(train["topo_family"])
        te_fam: Set[str] = set(test["topo_family"])
        tr_lab, te_lab = set(train["_lab"]), set(test["_lab"])
        tr_str, te_str = set(train["_str"]), set(test["_str"])

        print(f"\n=== {tier} tools ".ljust(70, "="))
        print(f"  train: {len(train)} rows, {len(tr_lab)} labelled DAGs, {len(tr_str)} shapes")
        print(f"  test : {len(test)} rows, {len(te_lab)} labelled DAGs, {len(te_str)} shapes")
        print(f"  train families ({len(tr_fam)}): {', '.join(sorted(tr_fam))}")
        print(f"  test  families ({len(te_fam)}): {', '.join(sorted(te_fam))}")

        fam_shared = sorted(tr_fam & te_fam)
        print(f"  (1) family-label overlap    : {len(fam_shared)} -> {fam_shared if fam_shared else 'none'}")
        print(f"  (2) shape overlap           : {len(tr_str & te_str)} of {len(te_str)} test shapes seen in train")
        print(f"  (3) labelled-DAG overlap    : {len(tr_lab & te_lab)} of {len(te_lab)} test DAGs seen in train")
        leaked_rows = int(test["_lab"].isin(tr_lab).sum())
        print(f"      -> affected test rows   : {leaked_rows} of {len(test)}")

        # Diamond specifically.
        print("  diamond:")
        print(f"      'diamond' family in train: {'diamond' in tr_fam}")
        print(f"      'diamond' family in test : {'diamond' in te_fam}")
        print(f"      diamond SHAPE in train   : {DIAMOND_STRUCTURE in tr_str}")
        print(f"      diamond SHAPE in test    : {DIAMOND_STRUCTURE in te_str}")

        te_diamond = test[test["_str"] == DIAMOND_STRUCTURE]
        if len(te_diamond):
            seen = int(te_diamond["_lab"].isin(tr_lab).sum())
            print(
                f"      test diamond rows        : {len(te_diamond)} "
                f"({te_diamond['_lab'].nunique()} unique DAGs); "
                f"{seen} rows whose exact DAG is in train"
            )
            if DIAMOND_STRUCTURE in tr_str:
                tr_diamond = train[train["_str"] == DIAMOND_STRUCTURE]
                labels = sorted(set(tr_diamond["topo_family"]))
                print(
                    f"      train diamond rows       : {len(tr_diamond)} "
                    f"labelled {labels}"
                )


if __name__ == "__main__":
    main()
