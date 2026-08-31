"""
audit_single_node_coverage.py — can single-node retrieval be evaluated?
=======================================================================

Reports, per tier and split, how many tools have a single-node DAG. This is the
precondition for comparing the functional-taxonomy router (Experiment 1) against
LEGR single-node retrieval: the router chooses among all N tools, so a
like-for-like retrieval comparison needs a single-node DAG per tool in the
corpus, with held-out query phrasings in test.

Also reports how far the routing benchmark's label vocabulary agrees with LEGR's
tool vocabulary, since the two experiments can only be compared on tools whose
names line up.

Usage
-----
    python scripts/audit_single_node_coverage.py --roots upgraded upgraded_v3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import query_library as QL  # noqa: E402

TIERS = ("15", "30", "45")
SPLITS = ("train", "dev", "test_topology_heldout")


def tools_of(cell: object) -> List[str]:
    return [t.strip() for t in str(cell).split(";") if t.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=["upgraded", "upgraded_v3"])
    args = ap.parse_args()

    rows: List[dict] = []
    for root in args.roots:
        for tier in TIERS:
            n = int(tier)
            for split in SPLITS:
                path = ROOT / root / f"upgraded_{tier}tools" / f"{split}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path, keep_default_na=False, dtype=str)
                parsed = df["tools"].map(tools_of)
                single = parsed[parsed.map(len) == 1]
                covered = {t[0] for t in single}
                rows.append(
                    {
                        "root": root,
                        "tier": tier,
                        "split": split,
                        "rows": len(df),
                        "single_node_rows": len(single),
                        "tools_covered": len(covered),
                        "of": n,
                        "evaluable": "yes" if len(covered) == n else "no",
                    }
                )

    table = pd.DataFrame(rows)
    print("SINGLE-NODE COVERAGE")
    print("=" * 78)
    print(table.to_string(index=False))

    print()
    print("ROUTING vs LEGR VOCABULARY (name agreement)")
    print("=" * 78)
    for tier in TIERS:
        n = int(tier)
        legr: Set[str] = set(QL.TOOL_VOCAB[:n])
        rpath = ROOT / "upgraded_data" / f"routing_{tier}tools" / "base_cleaned.csv"
        if not rpath.exists():
            print(f"  {tier} tools: routing benchmark missing")
            continue
        routing = set(pd.read_csv(rpath, keep_default_na=False)["ground_truth"])
        shared = legr & routing
        print(
            f"  {tier} tools: {len(shared)}/{n} names match; "
            f"{len(routing - legr)} routing-only, {len(legr - routing)} LEGR-only"
        )
        if routing - legr:
            print(f"      routing-only: {sorted(routing - legr)}")
            print(f"      LEGR-only   : {sorted(legr - routing)}")


if __name__ == "__main__":
    main()
