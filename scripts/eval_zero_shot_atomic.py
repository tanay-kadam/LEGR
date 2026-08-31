"""
eval_zero_shot_atomic.py — Frozen LEGR on atomic routing queries.

Adds one-node graphs for the 15 LEGR tools to the compositional candidate
corpus and measures whether the correct one-node graph is retrieved.

Does not retrain. Does not register OOV routing names.

Usage::

    python scripts/eval_zero_shot_atomic.py --tool_count 15 \\
        --checkpoint checkpoints_15tools/best_model.pt \\
        --output artifacts/zero_shot_atomic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legr_tool_count import (  # noqa: E402
    add_tool_count_argument,
    apply_tool_count_override,
    bootstrap_tool_count_from_argv,
)

bootstrap_tool_count_from_argv(sys.argv)

from atomic_zero_shot import (  # noqa: E402
    LEGR_15_TOOLS,
    UnmappedRoutingToolError,
    alias_routing_tool,
    canonicalise_routing_columns,
    is_one_node,
    merge_candidate_corpus,
    one_node_candidates,
    one_node_id_by_tool,
)
from data_synth import dag_canonical_hash, dag_to_pyg, dag_to_text  # noqa: E402
from encoders import resolve_graph_encoder_settings  # noqa: E402
from eval import CSVEvalDataset, _load_model_and_tokenizer  # noqa: E402
from utils import read_datafile  # noqa: E402

STRESS_CSVS = {
    "Standard": ROOT / "upgraded_data" / "routing_15tools" / "base_cleaned.csv",
    "Lexical": ROOT / "upgraded_data" / "routing_15tools" / "lexical_cue_reduced.csv",
    "Confusable": ROOT / "upgraded_data" / "routing_15tools" / "confusable_intents.csv",
    "Paraphrase": ROOT / "upgraded_data" / "routing_15tools" / "paraphrase_heldout_test.csv",
}

COMPOSITIONAL_CSV = ROOT / "upgraded" / "upgraded_15tools" / "test_topology_heldout.csv"


def _repo_rel(path: str | Path) -> str:
    """Store repository-relative paths in artifacts when possible."""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


@torch.no_grad()
def encode_graphs(model, graphs, device, bidirectional: bool, batch_size: int = 64):
    embs = []
    for i in range(0, len(graphs), batch_size):
        pyg = [dag_to_pyg(G, bidirectional=bidirectional) for G in graphs[i:i + batch_size]]
        batch = Batch.from_data_list(pyg)
        tp = getattr(batch, "topo_pos", None)
        if tp is not None:
            tp = tp.to(device)
        z = model.encode_graph(
            batch.x.to(device),
            batch.edge_index.to(device),
            batch.batch.to(device),
            topo_pos=tp,
        )
        embs.append(z.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_queries(model, queries, tokenizer, device, batch_size: int = 64):
    out = []
    for i in range(0, len(queries), batch_size):
        enc = tokenizer(
            queries[i:i + batch_size],
            padding=True, truncation=True, max_length=128, return_tensors="pt",
        )
        out.append(model.encode_text(enc["input_ids"].to(device), enc["attention_mask"].to(device)).cpu())
    return torch.cat(out, dim=0)


def accuracy_pct(correct: int, n: int) -> float:
    if n == 0:
        return 0.0
    return round(100.0 * correct / n, 1)


def recall_at_k(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return float(np.mean([1.0 if r <= k else 0.0 for r in ranks]))


def evaluate_condition(
    model,
    tokenizer,
    device,
    queries: list[str],
    gt_tools: list[str],
    candidate_embs: torch.Tensor,
    tool_to_id: dict[str, int],
    unique_dags,
    bidirectional: bool,
) -> dict:
    q_embs = encode_queries(model, queries, tokenizer, device)
    sim = torch.mm(q_embs, candidate_embs.t())
    order = torch.argsort(sim, dim=1, descending=True)
    n = len(queries)
    top1_correct = 0
    stolen = 0
    ranks: list[int] = []
    failures = []
    for i in range(n):
        gt_tool = gt_tools[i]
        gt_id = tool_to_id[gt_tool]
        ranking = order[i].tolist()
        rank = ranking.index(gt_id) + 1
        ranks.append(rank)
        pred_id = ranking[0]
        pred_G = unique_dags[pred_id]
        if pred_id == gt_id:
            top1_correct += 1
        else:
            if (not is_one_node(pred_G)) and rank > 1:
                stolen += 1
            elif not is_one_node(pred_G):
                stolen += 1
            failures.append({
                "query": queries[i],
                "gt_tool": gt_tool,
                "rank_of_one_node": rank,
                "top1_is_one_node": is_one_node(pred_G),
                "top1_dag_text": dag_to_text(pred_G),
                "top1_cosine": float(sim[i, pred_id]),
                "gt_cosine": float(sim[i, gt_id]),
            })
    return {
        "n": n,
        "correct": top1_correct,
        "accuracy_pct": accuracy_pct(top1_correct, n),
        "recall@1": recall_at_k(ranks, 1),
        "recall@3": recall_at_k(ranks, 3),
        "recall@5": recall_at_k(ranks, 5),
        "mean_rank": float(np.mean(ranks)) if ranks else None,
        "multi_node_steal_rate": stolen / n if n else 0.0,
        "failures": failures,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_tool_count_argument(p, default=15)
    p.add_argument("--checkpoint", required=False, default=None)
    p.add_argument(
        "--compositional_csv",
        default="upgraded/upgraded_15tools/test_topology_heldout.csv",
    )
    p.add_argument("--output", default="artifacts/zero_shot_atomic")
    p.add_argument("--dry_run", action="store_true", help="Build corpus only; no encoder")
    return p.parse_args()


def main():
    args = parse_args()
    if args.tool_count != 15:
        raise ValueError(
            "Zero-shot atomic eval is defined for the 15-tool routing benchmark "
            "plus the two-name alias map. 30-tool is out of scope (vocab mismatch)."
        )
    apply_tool_count_override(15)
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_csv = Path(args.compositional_csv)
    if not comp_csv.is_absolute():
        comp_csv = ROOT / comp_csv
    comp_df = read_datafile(str(comp_csv))
    comp_ds = CSVEvalDataset(comp_df)
    extra = one_node_candidates(LEGR_15_TOOLS)
    unique, _h2i = merge_candidate_corpus(comp_ds._unique_dags, extra)
    tool_to_id = one_node_id_by_tool(unique)
    missing_tools = [t for t in LEGR_15_TOOLS if t not in tool_to_id]
    if missing_tools:
        raise RuntimeError(f"One-node candidates missing for {missing_tools}")

    corpus_rows = []
    for i, G in enumerate(unique):
        corpus_rows.append({
            "candidate_id": i,
            "dag_hash": dag_canonical_hash(G),
            "dag_text": dag_to_text(G),
            "is_one_node": is_one_node(G),
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
        })
    pd.DataFrame(corpus_rows).to_csv(out_dir / "candidate_corpus.csv", index=False)
    definition = {
        "compositional_csv": _repo_rel(comp_csv),
        "n_compositional_unique": comp_ds.num_unique_dags,
        "n_one_node_injected": len(extra),
        "n_unified": len(unique),
        "aliases": {"query_database": "db_read", "update_database": "db_write"},
        "legr_15_tools": list(LEGR_15_TOOLS),
        "self_loops": "Not added by us. GCNConv may add self-loops internally.",
        "fake_edges": False,
    }
    stress_preview = {}
    for name, path in STRESS_CSVS.items():
        sdf = canonicalise_routing_columns(pd.read_csv(path), str(path))
        labels = [alias_routing_tool(x) for x in sdf["ground_truth"].astype(str)]
        stress_preview[name] = {
            "path": _repo_rel(path),
            "n": int(len(sdf)),
            "n_aliased_labels": int(len(set(labels))),
        }
    definition["stress_conditions"] = stress_preview
    (out_dir / "candidate_corpus_definition.json").write_text(
        json.dumps(definition, indent=2), encoding="utf-8",
    )

    if args.dry_run or not args.checkpoint:
        report = [
            "# Zero-shot atomic LEGR",
            "",
            "**Status:** PENDING_CHECKPOINT",
            "",
            f"Unified corpus size: {len(unique)} "
            f"({comp_ds.num_unique_dags} compositional unique + 15 one-node, deduped).",
            "",
            "Stress CSVs canonicalised to query/ground_truth. No encoder was run.",
            "",
        ]
        (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
        print("Dry run / no checkpoint: corpus written.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tokenizer = _load_model_and_tokenizer(args.checkpoint, device)
    model.eval()
    _, _, bidirectional = resolve_graph_encoder_settings(cfg)
    cand_embs = encode_graphs(model, unique, device, bidirectional)
    torch.save(cand_embs, out_dir / "candidate_embeddings.pt")

    per_condition = {}
    all_failures = {}
    for name, path in STRESS_CSVS.items():
        df = canonicalise_routing_columns(pd.read_csv(path), str(path))
        queries = df["query"].astype(str).tolist()
        try:
            gt_tools = [alias_routing_tool(x) for x in df["ground_truth"].astype(str)]
        except UnmappedRoutingToolError:
            raise
        metrics = evaluate_condition(
            model, tokenizer, device, queries, gt_tools,
            cand_embs, tool_to_id, unique, bidirectional,
        )
        failures = metrics.pop("failures")
        per_condition[name] = metrics
        all_failures[name] = failures
        pd.DataFrame(failures).to_csv(out_dir / f"failures_{name.lower()}.csv", index=False)

    n_total = sum(m["n"] for m in per_condition.values())
    c_total = sum(m["correct"] for m in per_condition.values())
    aggregate = {
        "n": n_total,
        "correct": c_total,
        "accuracy_pct": accuracy_pct(c_total, n_total),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps({"per_condition": per_condition, "aggregate": aggregate}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "repro.md").write_text(
        f"python scripts/eval_zero_shot_atomic.py --tool_count 15 "
        f"--checkpoint {_repo_rel(args.checkpoint) if args.checkpoint else 'path/to/best_model.pt'} "
        f"--output {_repo_rel(out_dir)}\n",
        encoding="utf-8",
    )

    outcome = "OUTCOME B — ZERO-SHOT FAILURE"
    # Threshold is documentary only after a real run; do not claim success here
    # unless accuracy is competitive with Table 1 functional taxonomy (~80%+).
    if all(m["accuracy_pct"] >= 70.0 for m in per_condition.values()):
        outcome = "OUTCOME A — ZERO-SHOT SUCCESS (candidate Table 1 row; unification language still requires review)"

    lines = [
        "# Zero-shot atomic LEGR",
        "",
        f"**Status:** COMPUTED against `{args.checkpoint}`",
        f"**Classification:** {outcome}",
        "",
        "Do not insert a unified-framework claim automatically.",
        "",
        "## Per-condition metrics",
        "",
        json.dumps(per_condition, indent=2),
        "",
        "## Aggregate",
        "",
        json.dumps(aggregate, indent=2),
        "",
        "## One-node GNN notes",
        "",
        "- Empty `edge_index` of shape (2, 0); no artificial edges.",
        "- GCNConv may add self-loops internally; DirectedGraphEncoder uses W_self only.",
        "- Frozen checkpoint embedding table is not expanded.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"outcome": outcome, "per_condition": per_condition, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
