"""
regenerate_queries_v2.py — rebuild the natural-language query column
=====================================================================

Writes ``upgraded_v2/upgraded_{15,30,45}tools/`` from ``upgraded/``, replacing
only the ``query`` column. Every DAG, ``dag_id``, edge list, topology family and
split assignment carries through untouched, so v2 is a controlled A/B against v1
on phrasing alone and an already-trained checkpoint can be scored on it directly.

Three defects in the v1 queries are fixed:

1.  Redundancy. v1 reached its row counts by re-filling a handful of base
    sentences with different entity names: at 30 tools, 1396 rows collapse to 601
    distinct phrasings (4.39 per DAG). v2 samples in *skeleton* space (entity
    placeholders still unresolved) and dedups there before filling entities, so
    every row is a genuinely distinct sentence by construction.
2.  Tool-name leakage. 7.8% of the 30-tool test queries and 6.9% of the 45-tool
    ones contain a literal tool name, handing BM25 and S-BERT a free lexical
    match. A deterministic gate rejects any such skeleton.
3.  Entity memorisation. v1 drew every split from the same small pools. v2 gives
    train/dev/test disjoint slices of enlarged pools.

Skeleton uniqueness is enforced across all splits of a tier, so train/test
phrasing overlap is zero by construction.

Usage
-----
    python scripts/regenerate_queries_v2.py                  # all three tiers
    python scripts/regenerate_queries_v2.py --tiers 30
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import query_library as QL  # noqa: E402

HARD_NEG_FILE = "hard_negatives.csv"

# Literal tool names and their de-underscored forms, for the leakage gate.
_TOOL_FORMS: Tuple[str, ...] = tuple(
    sorted(
        {t.lower() for t in QL.TOOL_VOCAB}
        | {t.replace("_", " ").lower() for t in QL.TOOL_VOCAB},
        key=len,
        reverse=True,
    )
)


# ─────────────────────────────────────────────────────────────────────────────
#  Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_tools(value: object) -> List[str]:
    return [t.strip() for t in str(value).split(";") if t.strip()]


def parse_edges(value: object) -> List[Tuple[int, int]]:
    text = str(value).strip()
    if not text:
        return []
    edges: List[Tuple[int, int]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part or "->" not in part:
            continue
        src, dst = part.split("->", 1)
        edges.append((int(src.strip()), int(dst.strip())))
    return edges


def split_kind(filename_stem: str) -> str:
    if filename_stem.startswith("train"):
        return "train"
    if filename_stem.startswith("dev"):
        return "dev"
    return "test"


# ─────────────────────────────────────────────────────────────────────────────
#  Composition
# ─────────────────────────────────────────────────────────────────────────────

def layer_nodes(
    num_nodes: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[List[List[int]], Dict[int, List[int]]]:
    """Longest-path layering, matching data_synth._synthesize_queries.

    Reusing v1's layering keeps the connector *class* per layer identical, so a
    v2 query describes the same topology as its v1 counterpart; only the surface
    form changes.
    """
    parents: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for src, dst in edges:
        parents[dst].append(src)

    layers: Dict[int, int] = {i: 0 for i in range(num_nodes) if not parents[i]}
    changed = True
    while changed:
        changed = False
        for src, dst in edges:
            if src in layers:
                candidate = layers[src] + 1
                if dst not in layers or candidate > layers[dst]:
                    layers[dst] = candidate
                    changed = True

    max_layer = max(layers.values()) if layers else 0
    grouped: List[List[int]] = [[] for _ in range(max_layer + 1)]
    for node in range(num_nodes):
        if node in layers:
            grouped[layers[node]].append(node)
    return grouped, parents


def _join_layer(phrases: Dict[int, str], layer: Sequence[int]) -> str:
    if len(layer) == 1:
        return phrases[layer[0]]
    if len(layer) == 2:
        return f"{phrases[layer[0]]} and {phrases[layer[1]]}"
    head = ", ".join(phrases[nd] for nd in layer[:-1])
    return f"{head}, and {phrases[layer[-1]]}"


def compose_skeleton(
    tools: Sequence[str], edges: Sequence[Tuple[int, int]], rng: random.Random
) -> Tuple[str, int]:
    """Build one query with ``{entity}`` placeholders intact.

    Returns the skeleton and the index of the opener used (-1 if none), which
    labels the phrasing family for the held-out-paraphrase split.
    """
    num_nodes = len(tools)

    if num_nodes == 1:
        pattern = rng.choice(QL.SINGLE_NODE_PATTERNS)
        opener_idx = rng.randrange(len(QL.OPENERS))
        opener = QL.OPENERS[opener_idx] or "Please"
        text = pattern.format(p0=rng.choice(QL.TOOL_PHRASES[tools[0]]), opener=opener)
        text = text[0].upper() + text[1:]
        return text + rng.choice(QL.CLOSERS), opener_idx

    grouped, parents = layer_nodes(num_nodes, edges)
    phrases = {i: rng.choice(QL.TOOL_PHRASES[tools[i]]) for i in range(num_nodes)}

    parts: List[str] = []
    opener_idx = -1
    for idx, layer in enumerate(grouped):
        if not layer:
            continue
        phrase = _join_layer(phrases, layer)
        is_fan_in = any(len(parents[nd]) > 1 for nd in layer)
        is_parallel = len(layer) > 1

        if idx == 0:
            opener_idx = rng.randrange(len(QL.OPENERS))
            opener = QL.OPENERS[opener_idx]
            parts.append(f"{opener} {phrase}" if opener else phrase[0].upper() + phrase[1:])
        elif is_fan_in and not is_parallel:
            parts.append(rng.choice(QL.MERGE_CONN) + phrase)
        elif is_parallel:
            parts.append(rng.choice(QL.PAR_CONN) + phrase)
        elif idx == len(grouped) - 1 and idx > 1:
            parts.append(rng.choice(QL.FINAL_CONN) + phrase)
        else:
            parts.append(rng.choice(QL.SEQ_CONN) + phrase)

    text = "".join(parts)
    if not text.endswith("."):
        text += "."
    text = text.replace("..", ".").replace(". .", ".")
    return text + rng.choice(QL.CLOSERS), opener_idx


def leaks_tool_name(skeleton: str) -> bool:
    low = skeleton.lower()
    return any(form in low for form in _TOOL_FORMS)


def generate_for_dag(
    tools: Sequence[str],
    edges: Sequence[Tuple[int, int]],
    count: int,
    rng: random.Random,
    used_skeletons: set,
    pools: Dict[str, List[str]],
) -> List[Tuple[str, int, str]]:
    """Return *count* distinct, leak-free (skeleton, opener, filled query) triples.

    The gate runs on the filled query as well as the skeleton, because entity
    substitution can manufacture a tool name that the skeleton did not contain
    (``invalidate {server}'s cache`` + a server called ``cache-*`` reads as the
    literal tool ``invalidate_cache``).
    """
    produced: List[Tuple[str, int, str]] = []
    attempts = 0
    budget = max(2000, count * 400)
    while len(produced) < count and attempts < budget:
        attempts += 1
        skeleton, opener_idx = compose_skeleton(tools, edges, rng)
        if skeleton in used_skeletons or leaks_tool_name(skeleton):
            continue
        filled = None
        for _ in range(8):
            candidate = QL.fill_entities(skeleton, pools, rng)
            if not leaks_tool_name(candidate):
                filled = candidate
                break
        if filled is None:
            continue
        used_skeletons.add(skeleton)
        produced.append((skeleton, opener_idx, filled))
    return produced


# ─────────────────────────────────────────────────────────────────────────────
#  Tier regeneration
# ─────────────────────────────────────────────────────────────────────────────

def regenerate_tier(
    tier: str, source_root: Path, out_root: Path, seed: int
) -> Tuple[List[dict], List[dict]]:
    src_dir = source_root / f"upgraded_{tier}tools"
    out_dir = out_root / f"upgraded_{tier}tools"
    out_dir.mkdir(parents=True, exist_ok=True)

    used_skeletons: set = set()
    stats: List[dict] = []
    provenance: List[dict] = []
    # dag_id -> filled queries, for rewriting hard_negatives.csv
    dag_queries: Dict[int, List[str]] = {}

    csv_paths = sorted(p for p in src_dir.glob("*.csv") if p.name != HARD_NEG_FILE)
    # Process train first so held-out splits are never starved of phrasings by it.
    csv_paths.sort(key=lambda p: 0 if p.stem.startswith("train") else 1)

    for path in csv_paths:
        df = pd.read_csv(path, keep_default_na=False, dtype=str)
        kind = split_kind(path.stem)
        pools = QL.partition_entity_pools(kind)

        new_queries: List[str] = [""] * len(df)
        shortfalls = 0

        for dag_id, group in df.groupby("dag_id", sort=True):
            positions = list(group.index)
            tools = parse_tools(group.iloc[0]["tools"])
            edges = parse_edges(group.iloc[0]["edges"])
            rng = random.Random(f"{seed}|{tier}|{path.stem}|{dag_id}")

            produced = generate_for_dag(
                tools, edges, len(positions), rng, used_skeletons, pools
            )
            if len(produced) < len(positions):
                shortfalls += len(positions) - len(produced)
                if not produced:
                    raise RuntimeError(
                        f"tier {tier} dag {dag_id}: no leak-free skeleton could be built"
                    )
                while len(produced) < len(positions):
                    produced.append(produced[len(produced) % len(produced)])

            for pos, (skeleton, opener_idx, filled) in zip(positions, produced):
                new_queries[pos] = filled
                provenance.append(
                    {
                        "tier": tier,
                        "split_file": path.stem,
                        "dag_id": dag_id,
                        "opener_id": opener_idx,
                        "skeleton": skeleton,
                        "query": filled,
                    }
                )
                dag_queries.setdefault(int(dag_id), []).append(filled)

        out_df = df.copy()
        out_df["query"] = new_queries
        out_df.to_csv(out_dir / path.name, index=False)

        stats.append(
            {
                "tier": tier,
                "file": path.name,
                "rows": len(out_df),
                "dags": out_df["dag_id"].nunique(),
                "unique_queries": out_df["query"].nunique(),
                "shortfalls": shortfalls,
            }
        )

    hard_neg_src = src_dir / HARD_NEG_FILE
    if hard_neg_src.exists():
        hn = pd.read_csv(hard_neg_src, keep_default_na=False, dtype=str)
        # One original query may be repeated across a positive's negative types;
        # map each distinct original query to a distinct v2 query for that DAG.
        remap: Dict[Tuple[str, str], str] = {}
        for pos_id, group in hn.groupby("positive_dag_id", sort=True):
            available = dag_queries.get(int(pos_id), [])
            for i, original in enumerate(dict.fromkeys(group["query"])):
                if available:
                    remap[(pos_id, original)] = available[i % len(available)]
        hn["query"] = [
            remap.get((r.positive_dag_id, r.query), r.query)
            for r in hn.itertuples(index=False)
        ]
        hn.to_csv(out_dir / HARD_NEG_FILE, index=False)
        stats.append(
            {
                "tier": tier,
                "file": HARD_NEG_FILE,
                "rows": len(hn),
                "dags": hn["positive_dag_id"].nunique(),
                "unique_queries": hn["query"].nunique(),
                "shortfalls": 0,
            }
        )

    return stats, provenance


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiers", nargs="+", default=["15", "30", "45"])
    ap.add_argument("--source-root", default="upgraded")
    ap.add_argument("--out-root", default="upgraded_v2")
    # Reports live outside --out-root so upgraded_v2/ mirrors upgraded/ exactly.
    ap.add_argument("--report-dir", default="upgraded_v2_reports")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source_root = (ROOT / args.source_root).resolve()
    out_root = (ROOT / args.out_root).resolve()
    report_dir = (ROOT / args.report_dir).resolve()

    all_stats: List[dict] = []
    all_prov: List[dict] = []
    for tier in args.tiers:
        stats, prov = regenerate_tier(tier, source_root, out_root, args.seed)
        all_stats.extend(stats)
        all_prov.extend(prov)

    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_prov).to_csv(report_dir / "query_provenance.csv", index=False)
    summary = pd.DataFrame(all_stats)
    summary.to_csv(report_dir / "regeneration_summary.csv", index=False)

    print(summary.to_string(index=False))
    total_short = int(summary["shortfalls"].sum())
    if total_short:
        print(f"\nWARNING: {total_short} rows could not get a unique skeleton.")
    else:
        print("\nAll rows received a distinct, leak-free skeleton.")


if __name__ == "__main__":
    main()
