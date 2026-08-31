"""
audit_dag_validity.py — Structural Validity Audit for LLM Baseline Predictions
================================================================================

Reads progress JSONL files produced by ``llm_dag_baseline.py`` (with the
extended ``pred_tools``, ``pred_edges``, ``had_cycle``, ``structurally_valid``
fields) and reports:

    - Parse-failure rate
    - Cyclic-graph rate (among successfully parsed predictions)
    - Structural-validity rate (DAG + weakly connected)
    - Exact-match rate
    - Mean GED (before and after cycle repair)

Usage
-----
    $ python scripts/audit_dag_validity.py \\
          --progress new_results/llm_dag_gpt-oss_30tools.progress.jsonl \\
                     new_results/llm_dag_llama3.2_30tools.progress.jsonl \\
          --labels   "GPT-OSS 30T" "Llama3.2 30T"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dag_extract import break_cycles_min_confidence, DEFAULT_CONFIDENCE


def _load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _audit_one(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Compute validity metrics for one set of progress records."""
    from data_synth import build_dag, compute_ged

    total = len(records)
    parse_failures = sum(1 for r in records if r.get("parse_failure"))
    parsed = [r for r in records if not r.get("parse_failure")]
    n_parsed = len(parsed)

    cyclic = sum(1 for r in parsed if r.get("had_cycle"))
    valid = sum(1 for r in parsed if r.get("structurally_valid"))
    exact = sum(1 for r in records if r.get("exact_match"))

    ged_errors = [
        float(r["ged_error"]) for r in records
        if r.get("ged_error") is not None
    ]

    ged_after_repair = []
    for r in parsed:
        pred_tools = r.get("pred_tools", [])
        pred_edges = r.get("pred_edges", [])
        if not pred_tools or not pred_edges:
            continue

        conf_edges = [
            {"source": int(e[0]), "target": int(e[1]),
             "confidence": DEFAULT_CONFIDENCE}
            for e in pred_edges
            if isinstance(e, (list, tuple)) and len(e) >= 2
        ]
        clean, _ = break_cycles_min_confidence(pred_tools, conf_edges)
        clean_pairs = [(e["source"], e["target"]) for e in clean]

        gt_tools = r.get("gt_tools")
        gt_edges = r.get("gt_edges")
        if gt_tools is None or gt_edges is None:
            continue

        try:
            gt_G = build_dag(gt_tools, [tuple(e) for e in gt_edges])
            repaired_G = build_dag(pred_tools, clean_pairs)
            ged_after_repair.append(compute_ged(gt_G, repaired_G))
        except Exception:
            pass

    return {
        "label": label,
        "total": total,
        "parse_failures": parse_failures,
        "parse_failure_rate": round(parse_failures / max(total, 1), 4),
        "parsed": n_parsed,
        "cyclic": cyclic,
        "cyclic_rate": round(cyclic / max(n_parsed, 1), 4),
        "structurally_valid": valid,
        "validity_rate": round(valid / max(n_parsed, 1), 4),
        "exact_match": exact,
        "exact_match_rate": round(exact / max(total, 1), 4),
        "mean_ged": round(np.mean(ged_errors), 4) if ged_errors else None,
        "mean_ged_after_repair": (
            round(np.mean(ged_after_repair), 4) if ged_after_repair else None
        ),
    }


def main():
    p = argparse.ArgumentParser(
        description="Audit structural validity of LLM DAG baseline predictions"
    )
    p.add_argument(
        "--progress", nargs="+", required=True,
        help="Progress JSONL file(s) from llm_dag_baseline.py",
    )
    p.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels for each progress file (same order)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Save audit results as CSV",
    )
    args = p.parse_args()

    labels = args.labels or [Path(p).stem for p in args.progress]
    if len(labels) != len(args.progress):
        labels = [Path(p).stem for p in args.progress]

    results = []
    for path, label in zip(args.progress, labels):
        print(f"\n  Auditing: {label} ({path})")
        records = _load_records(path)
        if not records:
            print(f"    WARNING: No records found in {path}")
            continue
        result = _audit_one(records, label)
        results.append(result)

    if not results:
        print("\n  No results to report.")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 72)
    print("  STRUCTURAL VALIDITY AUDIT")
    print("=" * 72)
    for r in results:
        print(f"\n  --- {r['label']} ---")
        print(f"    Total examples:        {r['total']}")
        print(f"    Parse failures:        {r['parse_failures']} "
              f"({r['parse_failure_rate']:.1%})")
        print(f"    Parsed predictions:    {r['parsed']}")
        print(f"    Cyclic graphs:         {r['cyclic']} "
              f"({r['cyclic_rate']:.1%})")
        print(f"    Structurally valid:    {r['structurally_valid']} "
              f"({r['validity_rate']:.1%})")
        print(f"    Exact match:           {r['exact_match']} "
              f"({r['exact_match_rate']:.1%})")
        if r['mean_ged'] is not None:
            print(f"    Mean GED (raw):        {r['mean_ged']:.4f}")
        if r['mean_ged_after_repair'] is not None:
            print(f"    Mean GED (repaired):   {r['mean_ged_after_repair']:.4f}")
    print("\n" + "=" * 72)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n  Saved audit results to {args.output}")


if __name__ == "__main__":
    main()
