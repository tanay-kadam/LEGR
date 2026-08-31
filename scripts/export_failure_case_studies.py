"""
export_failure_case_studies.py -- Render LEGR misretrieval case studies as Markdown.

Complements ``case_studies/case_studies.md`` (where LEGR wins) with the opposite
direction: the queries LEGR gets wrong and what the generative baselines produced
for the same query.

Usage
-----
    python scripts/export_failure_case_studies.py \
        --input new_results/failures_30tools_1200/failures_with_baselines.json \
        --out case_studies/legr_failure_cases.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_baseline(name: str, d: dict) -> list[str]:
    if not d["pred_tools"]:
        detail = f"**{name}:** parse failure — no usable JSON plan"
        if d.get("error"):
            detail += f" ({d['error']})"
        return [detail + f" · {d['latency_s']}s", ""]

    flags = []
    if d.get("had_cycle"):
        flags.append("**cyclic — not executable**")
    elif not d.get("structurally_valid", True):
        flags.append("structurally invalid (disconnected)")
    if d.get("exact_match"):
        flags.append("**exact match**")

    ged = "undefined (cyclic)" if d["ged_error"] is None else d["ged_error"]
    line = (f"**{name}:** `{d['pred_dag']}`  \n"
            f"Tool-Set F1 {d['tool_f1']} · GED {ged} · {d['latency_s']}s")
    if flags:
        line += " · " + ", ".join(flags)
    return [line, ""]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    recs = json.loads(Path(args.input).read_text(encoding="utf-8"))

    lines = [
        "# LEGR Failure Cases vs. Generative Baselines",
        "",
        "Every query on the held-out split where LEGR's top-1 retrieved DAG is not "
        "the ground-truth DAG, paired with what each generative baseline produced "
        "for the identical query and the same system prompt.",
        "",
    ]

    for i, r in enumerate(recs, 1):
        lines += [
            f"## Case {i} — row {r['csv_row']} ({r['topo_family']})",
            "",
            f"**Query:** {r['query']}",
            "",
            f"**Ground truth:** `{r['gt_dag']}`",
            "",
            f"**LEGR top-1 (wrong):** `{r['legr_top1_dag']}`  \n"
            f"Ground truth recovered at rank {r['legr_gt_rank']} · "
            f"GED {r['legr_ged_error']}",
            "",
        ]
        for name, d in r["models"].items():
            lines += format_baseline(name, d)
        lines.append("---")
        lines.append("")

    n = len(recs)
    for name in (recs[0]["models"] if recs else {}):
        em = sum(1 for r in recs if r["models"][name].get("exact_match"))
        cyc = sum(1 for r in recs if r["models"][name].get("had_cycle"))
        pf = sum(1 for r in recs if not r["models"][name]["pred_tools"])
        lines += [f"**{name} on these {n} LEGR-failure queries:** "
                  f"{em} exact match, {cyc} cyclic, {pf} parse failure.", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out} ({n} cases)")


if __name__ == "__main__":
    main()
