"""
verify_queries_v2.py — structural parity and redundancy check for upgraded_v2/
==============================================================================

Two jobs:

1.  Parity. ``upgraded_v2/`` must mirror ``upgraded/`` exactly: same tier
    directories, same filenames, same columns, same row order, and every column
    except ``query`` byte-identical. This is what makes v2 a controlled A/B on
    phrasing and lets an already-trained checkpoint be scored on it directly.
2.  Redundancy. Reports the v1-vs-v2 metrics that motivated the regeneration:
    entity-normalised distinct phrasings, literal tool-name leakage,
    train/test phrasing overlap, and train/test entity disjointness.

Usage
-----
    python scripts/verify_queries_v2.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import query_library as QL  # noqa: E402

TIERS = ("15", "30", "45")

_TOOL_FORMS: Tuple[str, ...] = tuple(
    {t.lower() for t in QL.TOOL_VOCAB} | {t.replace("_", " ").lower() for t in QL.TOOL_VOCAB}
)

_ALL_ENTITIES: Dict[str, List[str]] = QL.ENTITY_POOLS


def skeletonise(query: str) -> str:
    """Strip entity values so only the phrasing pattern remains."""
    text = query
    text = re.sub(r"#\d+", "{order}", text)
    text = re.sub(r"\bINC-\d+\b", "{ticket}", text)
    text = re.sub(r"\b[a-z]+-[a-z]+-\d+\b", "{server}", text)
    for name in _ALL_ENTITIES["user"]:
        text = re.sub(rf"\b{re.escape(name)}\b", "{user}", text)
    for name in _ALL_ENTITIES["dept"]:
        text = re.sub(rf"\b{re.escape(name)}\b", "{dept}", text)
    return text


def entities_in(query: str) -> Set[str]:
    found: Set[str] = set()
    for key in ("user", "dept", "server", "ticket", "order"):
        for value in _ALL_ENTITIES[key]:
            if re.search(rf"(?<![\w-]){re.escape(value)}(?![\w-])", query):
                found.add(f"{key}:{value}")
    return found


def leak_count(queries: pd.Series) -> int:
    total = 0
    for q in queries:
        low = q.lower()
        if any(form in low for form in _TOOL_FORMS):
            total += 1
    return total


def check_parity(src_root: Path, v2_root: Path) -> List[str]:
    problems: List[str] = []
    for tier in TIERS:
        src_dir = src_root / f"upgraded_{tier}tools"
        v2_dir = v2_root / f"upgraded_{tier}tools"
        if not v2_dir.is_dir():
            problems.append(f"tier {tier}: {v2_dir} missing")
            continue

        src_files = {p.name for p in src_dir.glob("*.csv")}
        v2_files = {p.name for p in v2_dir.glob("*.csv")}
        if src_files != v2_files:
            missing = sorted(src_files - v2_files)
            extra = sorted(v2_files - src_files)
            problems.append(f"tier {tier}: filename mismatch missing={missing} extra={extra}")

        for name in sorted(src_files & v2_files):
            a = pd.read_csv(src_dir / name, keep_default_na=False, dtype=str)
            b = pd.read_csv(v2_dir / name, keep_default_na=False, dtype=str)
            if list(a.columns) != list(b.columns):
                problems.append(f"tier {tier}/{name}: column mismatch")
                continue
            if len(a) != len(b):
                problems.append(f"tier {tier}/{name}: row count {len(a)} vs {len(b)}")
                continue
            for col in a.columns:
                if col == "query":
                    continue
                if not a[col].equals(b[col]):
                    diffs = int((a[col] != b[col]).sum())
                    problems.append(f"tier {tier}/{name}: column {col!r} differs in {diffs} rows")
            if (a["query"] == b["query"]).any():
                same = int((a["query"] == b["query"]).sum())
                problems.append(f"tier {tier}/{name}: {same} queries unchanged from v1")
    return problems


def redundancy_table(root: Path, label: str) -> pd.DataFrame:
    rows: List[dict] = []
    for tier in TIERS:
        tier_dir = root / f"upgraded_{tier}tools"
        for path in sorted(tier_dir.glob("*.csv")):
            df = pd.read_csv(path, keep_default_na=False, dtype=str)
            if "query" not in df.columns:
                continue
            queries = df["query"]
            skeletons = queries.map(skeletonise)
            rows.append(
                {
                    "version": label,
                    "tier": tier,
                    "file": path.stem,
                    "rows": len(df),
                    "unique_queries": queries.nunique(),
                    "unique_phrasings": skeletons.nunique(),
                    "redundant_pct": round(100.0 * (len(df) - skeletons.nunique()) / len(df), 1),
                    "toolname_leaks": leak_count(queries),
                }
            )
    return pd.DataFrame(rows)


def overlap_table(root: Path, label: str) -> pd.DataFrame:
    rows: List[dict] = []
    for tier in TIERS:
        tier_dir = root / f"upgraded_{tier}tools"
        train_path = tier_dir / "train.csv"
        if not train_path.exists():
            continue
        train = pd.read_csv(train_path, keep_default_na=False, dtype=str)
        train_skel = set(train["query"].map(skeletonise))
        train_ents: Set[str] = set()
        for q in train["query"]:
            train_ents |= entities_in(q)

        for path in sorted(tier_dir.glob("*.csv")):
            if path.stem.startswith("train"):
                continue
            df = pd.read_csv(path, keep_default_na=False, dtype=str)
            skels = set(df["query"].map(skeletonise))
            ents: Set[str] = set()
            for q in df["query"]:
                ents |= entities_in(q)
            rows.append(
                {
                    "version": label,
                    "tier": tier,
                    "file": path.stem,
                    "phrasings_shared_with_train": len(skels & train_skel),
                    "phrasing_overlap_pct": round(100.0 * len(skels & train_skel) / max(1, len(skels)), 1),
                    "entities_shared_with_train": len(ents & train_ents),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", default="upgraded")
    ap.add_argument("--v2-root", default="upgraded_v2")
    # Kept outside --v2-root so upgraded_v2/ mirrors upgraded/ exactly.
    ap.add_argument("--report-dir", default="upgraded_v2_reports")
    ap.add_argument(
        "--skip-parity",
        action="store_true",
        help="For trees that change the splits on purpose (upgraded_v3), where "
             "row-for-row parity with the source does not hold.",
    )
    args = ap.parse_args()

    src_root = ROOT / args.source_root
    v2_root = ROOT / args.v2_root
    report_dir = ROOT / args.report_dir

    problems: List[str] = []
    if args.skip_parity:
        print("=" * 78)
        print(f"PARITY: skipped ({args.v2_root} changes the splits by design)")
        print("=" * 78)
    else:
        print("=" * 78)
        print(f"PARITY: {args.v2_root}/ vs {args.source_root}/")
        print("=" * 78)
        problems = check_parity(src_root, v2_root)
        if problems:
            for p in problems:
                print(f"  FAIL  {p}")
        else:
            print("  OK — same files, columns and row order; only 'query' differs, on every row.")

    label = Path(args.v2_root).name
    v1 = redundancy_table(src_root, "v1")
    v2 = redundancy_table(v2_root, label)
    combined = pd.concat([v1, v2]).sort_values(["tier", "file", "version"])

    print()
    print("=" * 78)
    print("REDUNDANCY AND LEAKAGE")
    print("=" * 78)
    print(combined.to_string(index=False))

    o1 = overlap_table(src_root, "v1")
    o2 = overlap_table(v2_root, label)
    print()
    print("=" * 78)
    print("TRAIN / HELD-OUT OVERLAP")
    print("=" * 78)
    print(pd.concat([o1, o2]).sort_values(["tier", "file", "version"]).to_string(index=False))

    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / "verification_report.csv"
    combined.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(ROOT)}")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
