"""
analyze_legr_failures.py -- Locate LEGR misretrievals and align them with the
per-query predictions of the generative baselines.

For every query in a held-out split this script records the LEGR top-5 ranking,
flags the cases where the top-1 DAG is not the ground-truth DAG, and (when a
matching ``*.progress.jsonl`` from ``llm_dag_baseline.py`` is supplied) attaches
what GPT-OSS / Llama produced for the very same query.

The JSONL progress records are keyed by ``example_index``, which is the row index
of the source CSV, so the split file must be the same one the baseline was run on.

Usage
-----
    python scripts/analyze_legr_failures.py \
        --tool_count 45 \
        --checkpoint checkpoints_45tools/best_model.pt \
        --dataset_csv upgraded/upgraded_45tools/test_topology_heldout.csv \
        --llm_log "Llama 3.2=new_results/llm_dag_llama3.2_45tools.progress.jsonl" \
        --out_dir new_results/failures_45tools
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import networkx as nx
import pandas as pd
import torch

from legr_tool_count import add_tool_count_argument, bootstrap_tool_count_from_argv

_TOOL_COUNT_OVERRIDE = bootstrap_tool_count_from_argv(sys.argv)

from data_synth import (  # noqa: E402
    build_dag,
    compute_ged,
    dag_canonical_hash,
    dag_to_pyg,
    dag_to_text,
    register_tools,
)
from eval import _load_model_and_tokenizer, _parse_edges, _parse_tools  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402


def build_eval_index(df: pd.DataFrame):
    """Return (samples, unique_dags) keeping the source CSV row index per sample."""
    all_tools = set()
    for cell in df["tools"]:
        if isinstance(cell, str):
            all_tools.update(t.strip() for t in cell.split(";") if t.strip())
    register_tools(sorted(all_tools))

    unique_dags: list[nx.DiGraph] = []
    hash_to_id: dict[str, int] = {}
    samples: list[dict] = []

    for row_idx, row in df.iterrows():
        tools = _parse_tools(row["tools"])
        edges = _parse_edges(row["edges"])
        if not tools:
            continue
        try:
            G = build_dag(tools, edges)
        except (AssertionError, nx.NetworkXError):
            continue

        h = dag_canonical_hash(G)
        if h not in hash_to_id:
            hash_to_id[h] = len(unique_dags)
            unique_dags.append(G)

        samples.append({
            "csv_row": int(row_idx),
            "query": str(row.get("query", "")),
            "dag_id": hash_to_id[h],
            "topo_family": row.get("topo_family", ""),
        })

    return samples, unique_dags


@torch.no_grad()
def rank_all(model, tokenizer, samples, unique_dags, device, batch_size: int = 64):
    q_embs = []
    for i in range(0, len(samples), batch_size):
        chunk = [s["query"] for s in samples[i:i + batch_size]]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=128,
                        return_tensors="pt")
        q_embs.append(model.encode_text(enc["input_ids"].to(device),
                                        enc["attention_mask"].to(device)).cpu())
    q_embs = torch.cat(q_embs, dim=0)

    d_embs = []
    for i in range(0, len(unique_dags), batch_size):
        batch = Batch.from_data_list([dag_to_pyg(G)
                                      for G in unique_dags[i:i + batch_size]])
        topo = getattr(batch, "topo_pos", None)
        d_embs.append(model.encode_graph(
            batch.x.to(device),
            batch.edge_index.to(device),
            batch.batch.to(device),
            topo_pos=topo.to(device) if topo is not None else None,
        ).cpu())
    d_embs = torch.cat(d_embs, dim=0)

    sim = torch.mm(q_embs, d_embs.t())
    k = min(5, sim.size(1))
    top = sim.topk(k=k, dim=1)
    return top.indices, top.values


def load_llm_logs(specs: list[str]) -> dict[str, dict[int, dict]]:
    """Parse ``LABEL=path`` specs into {label: {example_index: record}}."""
    logs: dict[str, dict[int, dict]] = {}
    for spec in specs or []:
        label, _, path = spec.partition("=")
        records: dict[int, dict] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = rec.get("example_index")
                if isinstance(idx, int):
                    # Later records win: reruns append to the same file.
                    records[idx] = rec
        logs[label] = records
        print(f"  {label}: {len(records)} per-query records from {path}")
    return logs


def describe_llm_record(rec: dict) -> str:
    if rec is None:
        return "no logged prediction"
    if rec.get("parse_failure"):
        return f"parse failure ({rec.get('error', 'unparseable output')})"
    if "pred_tools" not in rec:
        return "predicted DAG not logged (scalar metrics only)"
    tools = rec.get("pred_tools", [])
    edges = rec.get("pred_edges", [])
    if not edges:
        body = ", ".join(tools) if tools else "(empty)"
    else:
        body = ", ".join(f"{tools[s]} -> {tools[d]}" for s, d in edges
                         if s < len(tools) and d < len(tools))
    flags = []
    if rec.get("had_cycle"):
        flags.append("CYCLIC")
    if not rec.get("structurally_valid", True):
        flags.append("structurally invalid")
    suffix = f"  [{'; '.join(flags)}]" if flags else ""
    return f"{body}{suffix}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_tool_count_argument(p, default=_TOOL_COUNT_OVERRIDE)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--llm_log", action="append", default=[],
                   metavar="LABEL=PATH",
                   help="Per-query progress JSONL from llm_dag_baseline.py")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    df = pd.read_csv(args.dataset_csv)
    samples, unique_dags = build_eval_index(df)
    print(f"  {len(df)} CSV rows -> {len(samples)} eval samples, "
          f"{len(unique_dags)} unique DAGs")

    model, cfg, tokenizer = _load_model_and_tokenizer(args.checkpoint, device)
    print(f"  Loaded {args.checkpoint} (lambda_ged={cfg.lambda_ged}, "
          f"ged_scale={cfg.ged_scale}, ged_margin={cfg.ged_margin})")

    topk, topv = rank_all(model, tokenizer, samples, unique_dags, device)
    llm_logs = load_llm_logs(args.llm_log)

    hits = {1: 0, 3: 0, 5: 0}
    failures = []
    for i, s in enumerate(samples):
        gt = s["dag_id"]
        ranked = topk[i].tolist()
        for k in (1, 3, 5):
            if gt in ranked[:k]:
                hits[k] += 1
        if ranked[0] == gt:
            continue

        pred = ranked[0]
        rank_of_gt = ranked.index(gt) + 1 if gt in ranked else None
        failures.append({
            "csv_row": s["csv_row"],
            "query": s["query"],
            "topo_family": s["topo_family"],
            "gt_dag": dag_to_text(unique_dags[gt]),
            "legr_top1_dag": dag_to_text(unique_dags[pred]),
            "legr_top1_score": round(float(topv[i][0]), 4),
            "legr_gt_score": (round(float(topv[i][rank_of_gt - 1]), 4)
                              if rank_of_gt else None),
            "legr_gt_rank": rank_of_gt,
            "legr_ged_error": compute_ged(unique_dags[gt], unique_dags[pred]),
            "legr_topk_dags": [dag_to_text(unique_dags[j]) for j in ranked],
            "baselines": {
                label: {
                    "prediction": describe_llm_record(recs.get(s["csv_row"])),
                    "tool_f1": (recs.get(s["csv_row"]) or {}).get("tool_f1"),
                    "ged_error": (recs.get(s["csv_row"]) or {}).get("ged_error"),
                    "exact_match": (recs.get(s["csv_row"]) or {}).get("exact_match"),
                    "latency_s": (recs.get(s["csv_row"]) or {}).get("latency_s"),
                }
                for label, recs in llm_logs.items()
            },
        })

    n = len(samples)
    print(f"\n  LEGR recall@1={hits[1]/n:.4f}  recall@3={hits[3]/n:.4f}  "
          f"recall@5={hits[5]/n:.4f}   ({n - hits[1]} misretrievals / {n})")

    for f in failures:
        print("\n" + "-" * 72)
        print(f"  row {f['csv_row']} [{f['topo_family']}]  GT rank: {f['legr_gt_rank']}")
        print(f"  query      : {f['query']}")
        print(f"  ground truth: {f['gt_dag']}")
        print(f"  LEGR top-1  : {f['legr_top1_dag']}   "
              f"(cos {f['legr_top1_score']} vs GT {f['legr_gt_score']}, "
              f"GED {f['legr_ged_error']})")
        for label, b in f["baselines"].items():
            print(f"  {label:<12}: {b['prediction']}   "
                  f"(F1 {b['tool_f1']}, GED {b['ged_error']})")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint,
            "dataset_csv": args.dataset_csv,
            "n_samples": n,
            "n_unique_dags": len(unique_dags),
            "recall@1": hits[1] / n,
            "recall@3": hits[3] / n,
            "recall@5": hits[5] / n,
            "failures": failures,
        }
        (out / "legr_failures.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Wrote {out / 'legr_failures.json'}")


if __name__ == "__main__":
    main()
