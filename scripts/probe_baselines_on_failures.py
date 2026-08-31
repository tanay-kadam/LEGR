"""
probe_baselines_on_failures.py -- Run the generative baselines on exactly the
queries where LEGR misretrieves.

Consumes ``legr_failures.json`` produced by ``analyze_legr_failures.py`` and
re-queries each configured model with the same system prompt, parser and metrics
that ``llm_dag_baseline.py`` uses, so the resulting numbers are directly
comparable to the aggregate baseline tables.

Usage
-----
    python scripts/probe_baselines_on_failures.py \
        --tool_count 30 \
        --failures new_results/failures_30tools_1200/legr_failures.json \
        --model llama3.2 --model gpt-oss:120b-cloud
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from legr_tool_count import add_tool_count_argument, bootstrap_tool_count_from_argv

_TOOL_COUNT_OVERRIDE = bootstrap_tool_count_from_argv(sys.argv)

from dag_extract import check_structural_validity  # noqa: E402
from data_synth import build_dag, compute_ged, dag_to_text  # noqa: E402
from llm_dag_baseline import (  # noqa: E402
    _build_system_prompt,
    _compute_tool_f1,
    _parse_llm_response,
)
from llm_backends import OllamaBackend  # noqa: E402


def dag_text_from_prediction(tools: list[str], edges: list[list[int]]) -> str:
    if not tools:
        return "(no tools)"
    if not edges:
        return ", ".join(tools)
    return ", ".join(
        f"{tools[s]} -> {tools[d]}"
        for s, d in edges
        if s < len(tools) and d < len(tools)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_tool_count_argument(p, default=_TOOL_COUNT_OVERRIDE)
    p.add_argument("--failures", required=True)
    p.add_argument("--model", action="append", default=[])
    p.add_argument("--timeout_s", type=float, default=180.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    payload = json.loads(Path(args.failures).read_text(encoding="utf-8"))
    failures = payload["failures"]
    system_prompt = _build_system_prompt()
    print(f"  {len(failures)} LEGR failure queries from {args.failures}")

    results = [
        {
            "csv_row": f["csv_row"],
            "query": f["query"],
            "topo_family": f["topo_family"],
            "gt_dag": f["gt_dag"],
            "legr_top1_dag": f["legr_top1_dag"],
            "legr_gt_rank": f["legr_gt_rank"],
            "legr_ged_error": f["legr_ged_error"],
            "models": {},
        }
        for f in failures
    ]

    # Ground-truth graphs are reconstructed from the split so GED is exact.
    import pandas as pd

    df = pd.read_csv(payload["dataset_csv"])
    for entry in results:
        row = df.iloc[entry["csv_row"]]
        tools = [t.strip() for t in str(row["tools"]).split(";") if t.strip()]
        edges = []
        for part in str(row["edges"]).split(";"):
            part = part.strip()
            if "->" in part:
                s, d = part.split("->", 1)
                edges.append((int(s), int(d)))
        entry["_gt_graph"] = build_dag(tools, edges)
        entry["_gt_tools"] = tools

    for model_name in args.model:
        print(f"\n  === {model_name} ===")
        backend = OllamaBackend(model_name=model_name, timeout_s=args.timeout_s)
        for entry in results:
            t0 = time.perf_counter()
            try:
                raw = backend.call(system_prompt, entry["query"]).text
                pred_tools, pred_edges = _parse_llm_response(raw)
                err = None
            except Exception as exc:  # network / timeout / auth
                raw, pred_tools, pred_edges, err = "", [], [], str(exc)
            latency = time.perf_counter() - t0

            record = {
                "latency_s": round(latency, 3),
                "pred_tools": pred_tools,
                "pred_edges": pred_edges,
                "pred_dag": dag_text_from_prediction(pred_tools, pred_edges),
                "parse_failure": not pred_tools,
                "error": err,
            }
            if pred_tools:
                record["tool_f1"] = round(
                    _compute_tool_f1(entry["_gt_tools"], pred_tools), 4)
                validity = check_structural_validity(pred_tools, pred_edges)
                record["had_cycle"] = validity["has_cycle"]
                record["structurally_valid"] = (
                    validity["is_dag"] and validity["is_connected"])
                try:
                    pred_G = build_dag(pred_tools, [tuple(e) for e in pred_edges])
                    record["ged_error"] = compute_ged(entry["_gt_graph"], pred_G)
                    record["exact_match"] = int(record["ged_error"] == 0)
                except Exception:
                    record["ged_error"] = None
                    record["exact_match"] = 0
            else:
                record.update({"tool_f1": 0.0, "ged_error": None,
                               "exact_match": 0, "had_cycle": False,
                               "structurally_valid": False})

            entry["models"][model_name] = record
            print(f"    row {entry['csv_row']:>4}  F1={record['tool_f1']}  "
                  f"GED={record['ged_error']}  {record['pred_dag'][:90]}")

    for entry in results:
        entry.pop("_gt_graph", None)
        entry.pop("_gt_tools", None)

    out_path = Path(args.out or Path(args.failures).with_name(
        "failures_with_baselines.json"))
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
