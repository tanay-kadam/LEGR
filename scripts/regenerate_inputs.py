"""
regenerate_inputs.py — Rebuild pipeline input files from the active vocabulary
===============================================================================

Generates both input CSVs required by run_pipeline.py from scratch, using the
current ACTIVE_TOOL_COUNT in vocab_config.py.

Step 1 — Routing input (results/single_tool_dataset_scaled.csv)
    Synthesises ACTIVE_TOOL_COUNT × 67 single-tool queries via
    dataset.build_scaled_single_tool_dataset(n_per_tool=67).
    confusable_per_label=30 in pipeline_config.json stays valid at 45 tools
    because 30 confusable pairs per label is conservative (~67% of n_per_tool).

Step 2 — Graph input (outputs/test_corpus_for_all_models.csv)
    Exports the LEGR test split as JSONL via
    data_synth.export_llm_routing_corpus_jsonl, then converts it to the CSV
    format that upgrade_graph.py expects (semicolon-delimited tools / edges).

Usage:
    python scripts/regenerate_inputs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legr_tool_count import (
    add_tool_count_argument,
    apply_tool_count_override,
    get_active_tool_count,
)

ROUTING_CSV = ROOT / "results" / "single_tool_dataset_scaled.csv"
GRAPH_JSONL = ROOT / "outputs" / "legr_llm_test.jsonl"
GRAPH_CSV = ROOT / "outputs" / "test_corpus_for_all_models.csv"


def step1_routing() -> int:
    """Build scaled single-tool routing dataset and save to CSV."""
    from dataset import build_scaled_single_tool_dataset

    print("\n[Step 1] Building routing input ...")
    df = build_scaled_single_tool_dataset(n_per_tool=67, seed=42)
    ROUTING_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ROUTING_CSV, index=True)
    n = len(df)
    tools = df["ground_truth"].nunique()
    print(f"  Wrote {n} rows x {tools} tools -> {ROUTING_CSV.relative_to(ROOT)}")
    return n


def step2_graph() -> int:
    """Export LEGR test split as JSONL, then convert to upgrade_graph CSV."""
    import data_synth

    print("\n[Step 2] Exporting graph test corpus ...")
    GRAPH_JSONL.parent.mkdir(parents=True, exist_ok=True)

    data_synth.export_llm_routing_corpus_jsonl(
        path=GRAPH_JSONL,
        split="test",
        entity_variants=4,
        seed=42,
    )

    rows = []
    with GRAPH_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            tools_str = ";".join(ex["tools"])
            edges_str = ";".join(f"{u}->{v}" for u, v in ex["edges"])
            rows.append({
                "query": ex["query"],
                "dag_id": ex["dag_id"],
                "dag_text": ex["dag_text"],
                "tools": tools_str,
                "edges": edges_str,
            })

    import pandas as pd
    df = pd.DataFrame(rows, columns=["query", "dag_id", "dag_text", "tools", "edges"])
    GRAPH_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GRAPH_CSV, index=False)
    n = len(df)
    print(f"  Wrote {n} rows -> {GRAPH_CSV.relative_to(ROOT)}")
    return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Regenerate routing and LEGR graph inputs for the active tool tier.",
    )
    add_tool_count_argument(p)
    args = p.parse_args(argv)
    apply_tool_count_override(args.tool_count)
    active_tool_count = get_active_tool_count()

    print("=" * 60)
    print(f"  Regenerate Pipeline Inputs  (ACTIVE_TOOL_COUNT={active_tool_count})")
    print("=" * 60)

    n_routing = step1_routing()
    n_graph = step2_graph()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Routing CSV : {n_routing} rows  ({ROUTING_CSV.relative_to(ROOT)})")
    print(f"  Graph CSV   : {n_graph} rows  ({GRAPH_CSV.relative_to(ROOT)})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
