"""Evaluate frozen Campaign V4 V3 checkpoints on all 15-tool routing conditions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.data import Batch

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from prepaper_common import (
    ROOT,
    checkpoint_manifest,
    create_output_dir,
    environment_snapshot,
    repo_relative,
    sha256_file,
    validate_checkpoint_metadata,
    write_json,
)

sys.path.insert(0, str(ROOT / "src"))


ROUTING_FILES = {
    "Standard": ROOT / "upgraded_data" / "routing_15tools" / "base_cleaned.csv",
    "Lexical": ROOT / "upgraded_data" / "routing_15tools" / "lexical_cue_reduced.csv",
    "Confusable": ROOT / "upgraded_data" / "routing_15tools" / "confusable_intents.csv",
    "Paraphrase": ROOT / "upgraded_data" / "routing_15tools" / "paraphrase_heldout_test.csv",
}

EXPECTED_ROWS = {"Standard": 1005, "Lexical": 1005, "Confusable": 450, "Paraphrase": 1255}

LLM_ROUTING_RESULTS = [
    {"model": "GPT-OSS Topic-based", "router_type": "two-stage taxonomy", "Standard": 78.3, "Lexical": 68.1, "Confusable": 62.7, "Paraphrase": 78.0},
    {"model": "GPT-OSS Functional", "router_type": "two-stage taxonomy", "Standard": 93.3, "Lexical": 75.6, "Confusable": 68.0, "Paraphrase": 90.6},
    {"model": "Llama 3.2 Topic-based", "router_type": "two-stage taxonomy", "Standard": 70.0, "Lexical": 43.3, "Confusable": 43.3, "Paraphrase": 64.9},
    {"model": "Llama 3.2 Functional", "router_type": "two-stage taxonomy", "Standard": 83.8, "Lexical": 61.8, "Confusable": 55.3, "Paraphrase": 79.8},
]


@torch.no_grad()
def encode_candidates(model, graphs, device: torch.device, bidirectional: bool) -> torch.Tensor:
    from data_synth import dag_to_pyg

    batch = Batch.from_data_list([dag_to_pyg(graph, bidirectional=bidirectional) for graph in graphs])
    topo_pos = getattr(batch, "topo_pos", None)
    if topo_pos is not None:
        topo_pos = topo_pos.to(device)
    return model.encode_graph(
        batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device),
        topo_pos=topo_pos,
    )


@torch.no_grad()
def score_queries(model, tokenizer, queries: list[str], candidates: torch.Tensor,
                  device: torch.device, batch_size: int) -> torch.Tensor:
    all_scores = []
    for start in range(0, len(queries), batch_size):
        encoded = tokenizer(
            queries[start:start + batch_size], padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        query_embeddings = model.encode_text(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
        )
        all_scores.append((query_embeddings @ candidates.T).cpu())
    return torch.cat(all_scores, dim=0)


def condition_metrics(
    frame: pd.DataFrame,
    scores: torch.Tensor,
    candidate_tools: list[str],
    aliased_truth: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame, np.ndarray]:
    tool_to_index = {tool: index for index, tool in enumerate(candidate_tools)}
    order = torch.argsort(scores, dim=1, descending=True).numpy()
    truth_ids = np.asarray([tool_to_index[tool] for tool in aliased_truth], dtype=np.int64)
    pred_ids = order[:, 0]
    ranks = np.asarray([
        int(np.where(order[index] == truth_ids[index])[0][0]) + 1
        for index in range(len(frame))
    ])
    correct = pred_ids == truth_ids
    precision, recall, f1, support = precision_recall_fscore_support(
        truth_ids, pred_ids, labels=np.arange(len(candidate_tools)), zero_division=0
    )
    macro_f1 = float(f1.mean())
    metrics = {
        "n": int(len(frame)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "accuracy_pct": float(100.0 * correct.mean()),
        "recall@1": float(np.mean(ranks <= 1)),
        "recall@3": float(np.mean(ranks <= 3)),
        "recall@5": float(np.mean(ranks <= 5)),
        "mrr@5": float(np.mean(np.where(ranks <= 5, 1.0 / ranks, 0.0))),
        "mean_rank": float(ranks.mean()),
        "macro_f1": macro_f1,
    }
    per_tool = pd.DataFrame({
        "tool": candidate_tools,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    rows = []
    for index, row in frame.reset_index(drop=True).iterrows():
        top5_ids = order[index, :5].tolist()
        record = {
            "row_index": index,
            "query": str(row["query"]),
            "ground_truth_raw": str(row["ground_truth"]),
            "ground_truth_tool": aliased_truth[index],
            "predicted_tool": candidate_tools[pred_ids[index]],
            "correct": bool(correct[index]),
            "rank": int(ranks[index]),
            "top1_cosine": float(scores[index, pred_ids[index]]),
            "ground_truth_cosine": float(scores[index, truth_ids[index]]),
            "top5_tools": ";".join(candidate_tools[value] for value in top5_ids),
            "top5_cosines": ";".join(f"{float(scores[index, value]):.8f}" for value in top5_ids),
        }
        for optional in ("source_row_id", "generation_type", "confusable_with"):
            if optional in row.index:
                record[optional] = row[optional]
        rows.append(record)
    per_query = pd.DataFrame(rows)
    matrix = confusion_matrix(
        truth_ids, pred_ids, labels=np.arange(len(candidate_tools))
    )
    return metrics, per_tool, per_query, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/prepaper_v3_atomic_routing_s42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from legr_tool_count import apply_tool_count_override

    apply_tool_count_override(15)
    from atomic_zero_shot import (
        LEGR_15_TOOLS,
        alias_routing_tool,
        canonicalise_routing_columns,
        is_one_node,
        one_node_candidates,
        one_node_id_by_tool,
    )
    from data_synth import dag_canonical_hash
    from encoders import resolve_graph_encoder_settings
    from eval import _load_model_and_tokenizer

    output = create_output_dir(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    candidate_tools = list(LEGR_15_TOOLS)
    candidates = one_node_candidates(candidate_tools)
    if len(candidates) != 15 or len(set(candidate_tools)) != 15:
        raise AssertionError("Routing gallery must contain exactly 15 distinct tools")
    if not all(is_one_node(graph) for graph in candidates):
        raise AssertionError("Every routing candidate must have one node and zero edges")
    if set(one_node_id_by_tool(candidates)) != set(candidate_tools):
        raise AssertionError("One-node candidate map does not cover all tools")
    candidate_frame = pd.DataFrame([
        {
            "candidate_id": index,
            "tool": candidate_tools[index],
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "canonical_dag_hash": dag_canonical_hash(graph),
        }
        for index, graph in enumerate(candidates)
    ])
    candidate_frame.to_csv(output / "candidate_gallery.csv", index=False)

    datasets = {}
    dataset_manifest = {}
    for condition, path in ROUTING_FILES.items():
        frame = canonicalise_routing_columns(pd.read_csv(path), str(path))
        if len(frame) != EXPECTED_ROWS[condition]:
            raise AssertionError(f"{condition}: expected {EXPECTED_ROWS[condition]}, found {len(frame)}")
        aliased = [alias_routing_tool(value) for value in frame["ground_truth"].astype(str)]
        if set(aliased) != set(candidate_tools):
            raise AssertionError(f"{condition}: labels do not cover the 15-candidate gallery exactly")
        datasets[condition] = (frame, aliased)
        dataset_manifest[condition] = {
            "path": repo_relative(path),
            "sha256": sha256_file(path),
            "rows": len(frame),
            "distinct_labels": len(set(aliased)),
        }

    selected_entries = [checkpoint_manifest()[15][2], checkpoint_manifest()[15][3]]
    model_results = {}
    summary_rows = []
    comparison_rows = list(LLM_ROUTING_RESULTS)
    checkpoint_audit = {}
    for entry in selected_entries:
        checkpoint_path = Path(entry["checkpoint"])
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = validate_checkpoint_metadata(entry, checkpoint_payload, 15)
        checkpoint_audit[entry["model_id"]] = {
            "path": repo_relative(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            **metadata,
        }
        model, cfg, tokenizer = _load_model_and_tokenizer(str(checkpoint_path), device)
        model.eval()
        _, _, bidirectional = resolve_graph_encoder_settings(cfg)
        candidate_embeddings = encode_candidates(model, candidates, device, bidirectional)
        if candidate_embeddings.shape != (15, 256) or not torch.isfinite(candidate_embeddings).all():
            raise AssertionError(f"Unexpected candidate embeddings: {candidate_embeddings.shape}")

        condition_results = {}
        comparison = {
            "model": f"LEGR {entry['model_id']}",
            "router_type": "direct frozen one-node retrieval",
        }
        for condition, (frame, aliased_truth) in datasets.items():
            queries = frame["query"].astype(str).tolist()
            scores = score_queries(
                model, tokenizer, queries, candidate_embeddings, device, args.batch_size
            )
            metrics, per_tool, per_query, matrix = condition_metrics(
                frame, scores, candidate_tools, aliased_truth
            )
            condition_results[condition] = metrics
            comparison[condition] = metrics["accuracy_pct"]
            summary_rows.append({
                "model_id": entry["model_id"],
                "architecture": entry["architecture"],
                "objective": entry["objective"],
                "condition": condition,
                **metrics,
            })
            per_tool.insert(0, "condition", condition)
            per_tool.to_csv(
                output / f"per_tool_{entry['model_id']}_{condition.lower()}.csv", index=False
            )
            per_query.insert(0, "condition", condition)
            per_query.to_csv(
                output / f"per_query_{entry['model_id']}_{condition.lower()}.csv", index=False
            )
            pd.DataFrame(matrix, index=candidate_tools, columns=candidate_tools).to_csv(
                output / f"confusion_{entry['model_id']}_{condition.lower()}.csv"
            )
        total = sum(value["n"] for value in condition_results.values())
        correct = sum(value["correct"] for value in condition_results.values())
        model_results[entry["model_id"]] = {
            "architecture": entry["architecture"],
            "objective": entry["objective"],
            "conditions": condition_results,
            "micro_aggregate": {
                "n": total,
                "correct": correct,
                "accuracy": correct / total,
                "accuracy_pct": 100.0 * correct / total,
            },
        }
        comparison_rows.append(comparison)
        del model, candidate_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(summary_rows).to_csv(output / "routing_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "routing_comparison.csv", index=False)
    payload = {
        "experiment": "Frozen Campaign V4 V3 cross-dataset atomic routing transfer",
        "not_an_unseen_topology_claim": True,
        "training_or_finetuning_performed": False,
        "candidate_protocol": "15 single-node, zero-edge routing tools only",
        "aliases": {"query_database": "db_read", "update_database": "db_write"},
        "datasets": dataset_manifest,
        "checkpoints": checkpoint_audit,
        "models": model_results,
        "environment": environment_snapshot(device),
    }
    write_json(output / "routing_results.json", payload)

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Objective & Standard & Lexical & Confusable & Paraphrase \\",
        r"\midrule",
    ]
    for model_id, result in model_results.items():
        values = result["conditions"]
        lines.append(
            f"LEGR V3 & {result['objective'].replace('InfoNCE+GED', 'InfoNCE+GED')} & "
            f"{values['Standard']['accuracy_pct']:.1f} & {values['Lexical']['accuracy_pct']:.1f} & "
            f"{values['Confusable']['accuracy_pct']:.1f} & {values['Paraphrase']['accuracy_pct']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "routing_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "report.md").write_text(
        "# Frozen V3 cross-dataset atomic routing\n\n"
        "The Campaign V4 V3 checkpoints are frozen. Each query is ranked directly against "
        "15 one-node, zero-edge tool graphs; neither routing taxonomy is used by V3. This is "
        "a cross-dataset transfer result, not an unseen-topology claim and not causal evidence "
        "for Functional Categorization.\n\n"
        "```json\n" + json.dumps(model_results, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    (output / "reproduce.txt").write_text(
        f"{sys.executable} scripts/evaluate_v3_atomic_routing.py --device {args.device} "
        f"--batch-size {args.batch_size} --output {repo_relative(output)}\n",
        encoding="utf-8",
    )
    print(json.dumps(model_results, indent=2))


if __name__ == "__main__":
    main()
