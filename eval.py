"""
eval.py — LEGR Evaluation & Baseline Comparison
==================================================

Evaluates the LEGR dual-encoder against S-BERT and BM25 baselines on
execution-graph retrieval.

Metrics:
    - tool_set_f1: F1 score on predicted vs ground-truth tool sets
    - mean_ged_error: Mean Graph Edit Distance between predicted and GT DAGs
    - recall@k (k=1,3,5): Fraction of queries with GT DAG in top-k
    - mrr@k: Mean Reciprocal Rank at k
    - Hard-negative ranking accuracy (optional)

Usage
-----
    $ python eval.py --checkpoint checkpoints/best_model.pt
    $ python eval.py --checkpoint checkpoints/best_model.pt \\
                     --dataset_csv upgraded_data/graph/test_topology_heldout.csv \\
                     --hard_negative_csv upgraded_data/graph/hard_negatives.csv \\
                     --save_results results/legr_topology_heldout_metrics.csv
    $ python eval.py --export_case_studies case_studies --max_case_studies 10

    GED ablation (same dataset, two LEGR checkpoints; baselines run once)::

        $ python eval.py --checkpoint ckpt_with_ged/best_model.pt ckpt_no_ged/best_model.pt \\
            --dataset_csv upgraded_data/graph/test_topology_heldout.csv \\
            --checkpoint_labels LEGR_WITH_GED LEGR_NO_GED \\
            --save_results results/legr_ged_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Batch

from data_synth import (
    TOOL_VOCAB,
    TOOL_TO_IDX,
    NUM_TOOLS,
    LEGRDataset,
    build_splits,
    build_dag,
    dag_to_pyg,
    dag_canonical_hash,
    dag_to_text,
    compute_ged,
    build_ged_matrix,
    register_tools,
)
from encoders import LEGRDualEncoder
from train import TrainConfig, _parse_tools, _parse_edges
from utils import read_datafile


def _text_model_name(cfg: TrainConfig) -> str:
    """Checkpoints from ``train.py`` use ``text_model``; older code used ``text_model_name``."""
    return getattr(cfg, "text_model_name", None) or getattr(
        cfg, "text_model", "sentence-transformers/all-MiniLM-L6-v2"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  CSV-backed evaluation dataset
# ═══════════════════════════════════════════════════════════════════════════

class CSVEvalDataset(torch.utils.data.Dataset):
    """Evaluation dataset loaded from CSV with query/tools/edges columns."""

    def __init__(self, df: pd.DataFrame, tools_col="tools", edges_col="edges"):
        self.samples = []
        self._unique_dags = []
        self._dag_texts = []
        hash_to_id: Dict[str, int] = {}

        all_tools = set()
        for tools_str in df[tools_col]:
            if isinstance(tools_str, str):
                all_tools.update(t.strip() for t in tools_str.split(";") if t.strip())
        register_tools(all_tools)

        for _, row in df.iterrows():
            tools = _parse_tools(row[tools_col])
            edges = _parse_edges(row[edges_col])
            query = row.get("query", "")

            if not tools:
                continue

            try:
                G = build_dag(tools, edges)
            except (AssertionError, nx.NetworkXError):
                continue

            h = dag_canonical_hash(G)
            if h not in hash_to_id:
                hash_to_id[h] = len(self._unique_dags)
                self._unique_dags.append(G)
                dag_text = row.get("dag_text", "")
                if not dag_text or (isinstance(dag_text, float) and pd.isna(dag_text)):
                    dag_text = dag_to_text(G)
                self._dag_texts.append(dag_text)

            dag_id = hash_to_id[h]
            pyg_data = dag_to_pyg(G)

            self.samples.append({
                "query": query,
                "graph": pyg_data,
                "dag_id": dag_id,
                "dag_nx": G,
            })

        self.ged_matrix = build_ged_matrix(self._unique_dags)
        self.num_unique_dags = len(self._unique_dags)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "query": s["query"],
            "graph": s["graph"],
            "dag_id": s["dag_id"],
        }

    def get_unique_dag(self, dag_id: int) -> nx.DiGraph:
        return self._unique_dags[dag_id]

    def get_dag_text(self, dag_id: int) -> str:
        return self._dag_texts[dag_id]

    def get_ged(self, id_a: int, id_b: int) -> float:
        return float(self.ged_matrix[id_a, id_b])


# ═══════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_model_and_tokenizer(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[LEGRDualEncoder, TrainConfig, object]:
    """Load a trained LEGR model from checkpoint."""
    if not checkpoint_path:
        raise ValueError(
            "checkpoint_path is missing. If you use run_multi_seed.train+eval, "
            "ensure train.main() returns the saved checkpoint path."
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = ckpt.get("config", {})
    cfg = TrainConfig(**{k: v for k, v in config_dict.items()
                         if k in TrainConfig.__dataclass_fields__})

    use_text = getattr(cfg, "use_text_node_features", False)
    text_feat_dim = getattr(cfg, "text_feature_dim", 0)
    tm = _text_model_name(cfg)

    # Architecture must match ``train.py`` / checkpoint (gcn_layers, node_embed_dim, …).
    model = LEGRDualEncoder(
        num_tools=NUM_TOOLS,
        text_model_name=tm,
        embed_dim=cfg.embed_dim,
        gcn_layers=cfg.gcn_layers,
        node_embed_dim=cfg.node_embed_dim,
        gcn_hidden=cfg.gcn_hidden,
        freeze_text=cfg.freeze_text,
        num_frozen_layers=cfg.num_frozen_layers,
        max_topo_pos=cfg.max_topo_pos,
        graph_encoder_type=getattr(cfg, "graph_encoder_type", "gcn"),
        use_text_node_features=use_text,
        text_feature_dim=text_feat_dim,
    ).to(device)

    model_state = ckpt.get("model_state") or ckpt.get("model_state_dict")
    if model_state is None:
        raise KeyError(
            f"Checkpoint {checkpoint_path} has no 'model_state' or 'model_state_dict'."
        )
    model.load_state_dict(model_state, strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tm)
    return model, cfg, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  Embedding computation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_all_queries(
    model: LEGRDualEncoder,
    dataset,
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode all queries in the dataset to text embeddings."""
    all_embs = []
    queries = [dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
               else dataset.samples[i].query for i in range(len(dataset))]

    for i in range(0, len(queries), batch_size):
        batch_q = queries[i:i + batch_size]
        encoded = tokenizer(
            batch_q, padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        z = model.encode_text(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )
        all_embs.append(z.cpu())

    return torch.cat(all_embs, dim=0)


@torch.no_grad()
def encode_all_dags(
    model: LEGRDualEncoder,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode all unique DAGs in the dataset to graph embeddings."""
    all_embs = []

    for i in range(0, dataset.num_unique_dags, batch_size):
        graphs = []
        for j in range(i, min(i + batch_size, dataset.num_unique_dags)):
            G = dataset.get_unique_dag(j)
            graphs.append(dag_to_pyg(G))

        batch = Batch.from_data_list(graphs)
        tp = getattr(batch, "topo_pos", None)
        if tp is not None:
            tp = tp.to(device)
        z = model.encode_graph(
            batch.x.to(device),
            batch.edge_index.to(device),
            batch.batch.to(device),
            topo_pos=tp,
        )
        all_embs.append(z.cpu())

    return torch.cat(all_embs, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    topk_ids: torch.Tensor,
    gt_dag_ids: torch.Tensor,
    dataset,
    k: int = 5,
) -> Dict[str, float]:
    """Compute retrieval metrics: tool_set_f1, GED error, recall@k, mrr@k."""
    n = len(gt_dag_ids)
    tool_f1s = []
    ged_errors = []
    recalls = {f"recall@{kk}": 0 for kk in [1, 3, 5]}
    mrrs = {f"mrr@{kk}": 0.0 for kk in [1, 3, 5]}

    for i in range(n):
        gt_id = gt_dag_ids[i].item()
        gt_dag = dataset.get_unique_dag(gt_id)
        gt_tools = set(gt_dag.nodes[n]["tool"] for n in gt_dag.nodes())

        pred_id = topk_ids[i, 0].item()
        pred_dag = dataset.get_unique_dag(pred_id)
        pred_tools = set(pred_dag.nodes[n]["tool"] for n in pred_dag.nodes())

        # Tool set F1
        if gt_tools or pred_tools:
            intersection = gt_tools & pred_tools
            precision = len(intersection) / len(pred_tools) if pred_tools else 0
            recall = len(intersection) / len(gt_tools) if gt_tools else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            tool_f1s.append(f1)
        else:
            tool_f1s.append(1.0)

        # GED error
        ged = dataset.get_ged(gt_id, pred_id)
        ged_errors.append(ged)

        # Recall@k and MRR@k
        topk_list = topk_ids[i].tolist()
        for kk in [1, 3, 5]:
            if gt_id in topk_list[:kk]:
                recalls[f"recall@{kk}"] += 1
                rank = topk_list[:kk].index(gt_id) + 1
                mrrs[f"mrr@{kk}"] += 1.0 / rank

    mean_tool_f1 = np.mean(tool_f1s)
    mean_ged_error = np.mean(ged_errors)
    for key in recalls:
        recalls[key] /= n
    for key in mrrs:
        mrrs[key] /= n

    return {
        "tool_set_f1": mean_tool_f1,
        "mean_ged_error": mean_ged_error,
        **recalls,
        **mrrs,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Baselines: S-BERT and BM25
# ═══════════════════════════════════════════════════════════════════════════

def _sbert_baseline(
    dataset,
    device: torch.device,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> Dict[str, float]:
    """Run S-BERT retrieval baseline."""
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(model_name, device=str(device))

    queries = [dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
               else dataset.samples[i].query for i in range(len(dataset))]
    gt_dag_ids = torch.tensor([
        dataset.samples[i]["dag_id"] if isinstance(dataset.samples[i], dict)
        else dataset.samples[i].dag_id for i in range(len(dataset))
    ])

    dag_texts = [dataset.get_dag_text(j) for j in range(dataset.num_unique_dags)]

    q_embs = torch.tensor(sbert.encode(queries, batch_size=batch_size, show_progress_bar=False))
    d_embs = torch.tensor(sbert.encode(dag_texts, batch_size=batch_size, show_progress_bar=False))

    q_embs = F.normalize(q_embs, p=2, dim=-1)
    d_embs = F.normalize(d_embs, p=2, dim=-1)

    sim = torch.mm(q_embs, d_embs.t())
    topk_ids = sim.topk(k=5, dim=1).indices

    return compute_metrics(topk_ids, gt_dag_ids, dataset)


def _bm25_baseline(dataset) -> Dict[str, float]:
    """Run BM25 retrieval baseline."""
    from rank_bm25 import BM25Okapi

    dag_texts = [dataset.get_dag_text(j) for j in range(dataset.num_unique_dags)]
    tokenized_corpus = [doc.lower().split() for doc in dag_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    queries = [dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
               else dataset.samples[i].query for i in range(len(dataset))]
    gt_dag_ids = torch.tensor([
        dataset.samples[i]["dag_id"] if isinstance(dataset.samples[i], dict)
        else dataset.samples[i].dag_id for i in range(len(dataset))
    ])

    topk_list = []
    for q in queries:
        scores = bm25.get_scores(q.lower().split())
        top_indices = np.argsort(scores)[::-1][:5]
        topk_list.append(top_indices.tolist())

    topk_ids = torch.tensor(topk_list)
    return compute_metrics(topk_ids, gt_dag_ids, dataset)


# ═══════════════════════════════════════════════════════════════════════════
#  Hard-negative evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_hard_negatives(
    model: LEGRDualEncoder,
    dataset,
    hard_neg_df: pd.DataFrame,
    tokenizer,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate hard-negative ranking accuracy."""
    correct = 0
    total = 0
    false_positives = 0

    for _, row in hard_neg_df.iterrows():
        query = row.get("query", "")
        if not query:
            continue

        neg_tools = _parse_tools(row.get("neg_tools", ""))
        neg_edges = _parse_edges(row.get("neg_edges", ""))
        if not neg_tools:
            continue

        try:
            neg_G = build_dag(neg_tools, neg_edges)
            neg_pyg = dag_to_pyg(neg_G)
        except Exception:
            continue

        encoded = tokenizer(
            [query], padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        z_text = model.encode_text(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )

        neg_batch = Batch.from_data_list([neg_pyg])
        neg_tp = getattr(neg_batch, "topo_pos", None)
        if neg_tp is not None:
            neg_tp = neg_tp.to(device)
        z_neg = model.encode_graph(
            neg_batch.x.to(device),
            neg_batch.edge_index.to(device),
            neg_batch.batch.to(device),
            topo_pos=neg_tp,
        )

        sim = F.cosine_similarity(z_text, z_neg).item()
        if sim < 0.5:
            correct += 1
        else:
            false_positives += 1
        total += 1

    accuracy = correct / max(total, 1)
    fp_rate = false_positives / max(total, 1)

    return {
        "hardneg_pairs_evaluated": total,
        "hardneg_ranking_accuracy": round(accuracy, 4),
        "hardneg_false_positive_rate": round(fp_rate, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Case study export
# ═══════════════════════════════════════════════════════════════════════════

def export_case_studies(
    dataset,
    legr_topk: torch.Tensor,
    sbert_topk: torch.Tensor,
    bm25_topk: torch.Tensor,
    gt_dag_ids: torch.Tensor,
    output_dir: str,
    max_cases: int = 10,
) -> None:
    """Export cases where LEGR is correct but baselines fail."""
    cases = []
    n = len(gt_dag_ids)

    for i in range(n):
        gt_id = gt_dag_ids[i].item()
        legr_id = legr_topk[i, 0].item()
        sbert_id = sbert_topk[i, 0].item()
        bm25_id = bm25_topk[i, 0].item()

        legr_correct = legr_id == gt_id
        sbert_correct = sbert_id == gt_id
        bm25_correct = bm25_id == gt_id

        if legr_correct and (not sbert_correct or not bm25_correct):
            query = (dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
                     else dataset.samples[i].query)
            gt_tools = [dataset.get_unique_dag(gt_id).nodes[n]["tool"]
                        for n in dataset.get_unique_dag(gt_id).nodes()]

            explanation_parts = ["LEGR retrieves the correct DAG (structure + tools)."]
            if not sbert_correct:
                explanation_parts.append(
                    "S-BERT retrieves a structurally/semantically wrong DAG."
                )
            if not bm25_correct:
                explanation_parts.append(
                    "BM25 retrieves a keyword-similar but wrong DAG."
                )

            cases.append({
                "query": query,
                "ground_truth_dag_text": dataset.get_dag_text(gt_id),
                "ground_truth_tools": gt_tools,
                "legr_top1_dag_text": dataset.get_dag_text(legr_id),
                "sbert_top1_dag_text": dataset.get_dag_text(sbert_id),
                "bm25_top1_dag_text": dataset.get_dag_text(bm25_id),
                "sbert_correct": sbert_correct,
                "bm25_correct": bm25_correct,
                "explanation": " ".join(explanation_parts),
            })

            if len(cases) >= max_cases:
                break

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "case_studies.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    lines = ["# Case Studies: LEGR vs Baselines\n"]
    for i, c in enumerate(cases, 1):
        lines.append(f"## Case {i}\n")
        lines.append(f"**Query:** {c['query']}\n")
        lines.append(f"**Ground Truth:** {c['ground_truth_dag_text']}\n")
        lines.append(f"**LEGR Top-1:** {c['legr_top1_dag_text']}\n")
        lines.append(f"**S-BERT Top-1:** {c['sbert_top1_dag_text']} "
                      f"({'✓' if c['sbert_correct'] else '✗'})\n")
        lines.append(f"**BM25 Top-1:** {c['bm25_top1_dag_text']} "
                      f"({'✓' if c['bm25_correct'] else '✗'})\n")
        lines.append(f"**Explanation:** {c['explanation']}\n")
        lines.append("---\n")

    (out_dir / "case_studies.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Exported {len(cases)} case studies to {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  Results CSV
# ═══════════════════════════════════════════════════════════════════════════

def save_results_csv(results: Dict[str, Dict[str, float]], path: str) -> None:
    """Write evaluation results to CSV (one row per model, columns per metric)."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    all_keys = set()
    for m in model_names:
        all_keys.update(results[m].keys())
    metric_keys = sorted(all_keys)

    with open(path_obj, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + metric_keys)
        for model in model_names:
            row = [model] + [results[model].get(k, "") for k in metric_keys]
            writer.writerow(row)

    print(f"  Results saved to {path_obj}")


# ═══════════════════════════════════════════════════════════════════════════
#  Latency benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_latency(
    model: LEGRDualEncoder,
    dataset,
    tokenizer,
    device: torch.device,
    n_samples: int = 100,
) -> Dict[str, float]:
    """Measure inference latency for LEGR retrieval."""
    latencies = []
    n = min(n_samples, len(dataset))

    for i in range(n):
        sample = dataset[i]
        query = sample["query"]

        t0 = time.perf_counter()

        encoded = tokenizer(
            [query], padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        z_text = model.encode_text(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )

        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    p95_idx = max(0, int(len(latencies) * 0.95) - 1)

    return {
        "num_samples": n,
        "mean_latency_s": round(np.mean(latencies), 6),
        "median_latency_s": round(np.median(latencies), 6),
        "p95_latency_s": round(latencies[p95_idx], 6),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_ablation_two(
    checkpoint_paths: List[str],
    labels: List[str],
    dataset_csv: Optional[str] = None,
    hard_negative_csv: Optional[str] = None,
    save_results_path: Optional[str] = None,
    export_case_studies_dir: Optional[str] = None,
    max_case_studies: int = 10,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Evaluate two LEGR checkpoints on the same dataset; S-BERT / BM25 once."""
    if len(checkpoint_paths) != 2 or len(labels) != 2:
        raise ValueError("Ablation requires exactly two checkpoints and two labels.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print("  Mode: two-checkpoint LEGR ablation (same dataset, shared baselines)")

    # Dataset once
    if dataset_csv:
        print(f"  Loading eval dataset: {dataset_csv}")
        df = read_datafile(dataset_csv)
        dataset = CSVEvalDataset(df)
    else:
        print("  Using built-in test split")
        _, _, dataset = build_splits(seed=seed)

    print(f"  Eval samples: {len(dataset)}, Unique DAGs: {dataset.num_unique_dags}")

    gt_dag_ids = torch.tensor([
        dataset.samples[i]["dag_id"] if isinstance(dataset.samples[i], dict)
        else dataset.samples[i].dag_id for i in range(len(dataset))
    ])

    results: Dict[str, Dict[str, float]] = {}
    legr_topk_first: Optional[torch.Tensor] = None
    cfg_first: Optional[TrainConfig] = None

    for idx, (ckpt_path, label) in enumerate(zip(checkpoint_paths, labels)):
        print(f"\n  --- LEGR run: {label} ({ckpt_path}) ---")
        model, cfg, tokenizer = _load_model_and_tokenizer(ckpt_path, device)

        q_embs = encode_all_queries(model, dataset, tokenizer, device)
        d_embs = encode_all_dags(model, dataset, device)
        sim = torch.mm(q_embs, d_embs.t())
        legr_topk = sim.topk(k=5, dim=1).indices

        legr_metrics = compute_metrics(legr_topk, gt_dag_ids, dataset)
        print(f"  {label}: {legr_metrics}")

        if hard_negative_csv:
            print(f"\n  Hard negatives ({label}): {hard_negative_csv}")
            hard_neg_df = read_datafile(hard_negative_csv)
            hn_metrics = evaluate_hard_negatives(
                model, dataset, hard_neg_df, tokenizer, device,
            )
            legr_metrics.update(hn_metrics)
            print(f"  {label} hard negatives: {hn_metrics}")

        lat = benchmark_latency(model, dataset, tokenizer, device)
        legr_metrics.update(lat)
        print(f"  {label} latency: {lat}")

        results[label] = legr_metrics

        if idx == 0:
            legr_topk_first = legr_topk
            cfg_first = cfg

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert cfg_first is not None and legr_topk_first is not None

    print("\n  Running S-BERT baseline (once)...")
    sbert_metrics = _sbert_baseline(
        dataset, device, model_name=_text_model_name(cfg_first),
    )
    print(f"  S-BERT: {sbert_metrics}")

    print("  Running BM25 baseline (once)...")
    bm25_metrics = _bm25_baseline(dataset)
    print(f"  BM25: {bm25_metrics}")

    results["S-BERT"] = sbert_metrics
    results["BM25"] = bm25_metrics

    if save_results_path:
        save_results_csv(results, save_results_path)

    if export_case_studies_dir:
        tm = _text_model_name(cfg_first)
        sbert_sim = torch.mm(
            F.normalize(torch.tensor(
                __import__("sentence_transformers").SentenceTransformer(
                    tm
                ).encode(
                    [dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
                     else dataset.samples[i].query for i in range(len(dataset))]
                )
            ), p=2, dim=-1),
            F.normalize(torch.tensor(
                __import__("sentence_transformers").SentenceTransformer(
                    tm
                ).encode(
                    [dataset.get_dag_text(j) for j in range(dataset.num_unique_dags)]
                )
            ), p=2, dim=-1).t(),
        )
        sbert_topk = sbert_sim.topk(k=5, dim=1).indices

        queries_for_bm25 = [
            dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
            else dataset.samples[i].query for i in range(len(dataset))
        ]
        from rank_bm25 import BM25Okapi
        dag_texts_bm25 = [dataset.get_dag_text(j) for j in range(dataset.num_unique_dags)]
        bm25_obj = BM25Okapi([d.lower().split() for d in dag_texts_bm25])
        bm25_topk_list = []
        for q in queries_for_bm25:
            scores = bm25_obj.get_scores(q.lower().split())
            bm25_topk_list.append(np.argsort(scores)[::-1][:5].tolist())
        bm25_topk = torch.tensor(bm25_topk_list)

        export_case_studies(
            dataset, legr_topk_first, sbert_topk, bm25_topk, gt_dag_ids,
            export_case_studies_dir, max_cases=max_case_studies,
        )

    return results


def evaluate(
    checkpoint_path: str,
    dataset_csv: Optional[str] = None,
    hard_negative_csv: Optional[str] = None,
    save_results_path: Optional[str] = None,
    export_case_studies_dir: Optional[str] = None,
    max_case_studies: int = 10,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Run full evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model, cfg, tokenizer = _load_model_and_tokenizer(checkpoint_path, device)
    print(f"  Model loaded from {checkpoint_path}")

    # Load dataset
    if dataset_csv:
        print(f"  Loading eval dataset: {dataset_csv}")
        df = read_datafile(dataset_csv)
        dataset = CSVEvalDataset(df)
    else:
        print("  Using built-in test split")
        _, _, dataset = build_splits(seed=seed)

    print(f"  Eval samples: {len(dataset)}, Unique DAGs: {dataset.num_unique_dags}")

    # LEGR retrieval
    print("\n  Running LEGR retrieval...")
    q_embs = encode_all_queries(model, dataset, tokenizer, device)
    d_embs = encode_all_dags(model, dataset, device)

    sim = torch.mm(q_embs, d_embs.t())
    legr_topk = sim.topk(k=5, dim=1).indices

    gt_dag_ids = torch.tensor([
        dataset.samples[i]["dag_id"] if isinstance(dataset.samples[i], dict)
        else dataset.samples[i].dag_id for i in range(len(dataset))
    ])

    legr_metrics = compute_metrics(legr_topk, gt_dag_ids, dataset)
    print(f"  LEGR: {legr_metrics}")

    # S-BERT baseline (same backbone name as training for a fair comparison)
    print("  Running S-BERT baseline...")
    sbert_metrics = _sbert_baseline(dataset, device, model_name=_text_model_name(cfg))
    print(f"  S-BERT: {sbert_metrics}")

    # BM25 baseline
    print("  Running BM25 baseline...")
    bm25_metrics = _bm25_baseline(dataset)
    print(f"  BM25: {bm25_metrics}")

    results = {
        "LEGR": legr_metrics,
        "S-BERT": sbert_metrics,
        "BM25": bm25_metrics,
    }

    # Hard negatives
    if hard_negative_csv:
        print(f"\n  Evaluating hard negatives: {hard_negative_csv}")
        hard_neg_df = read_datafile(hard_negative_csv)
        hn_metrics = evaluate_hard_negatives(
            model, dataset, hard_neg_df, tokenizer, device,
        )
        legr_metrics.update(hn_metrics)
        print(f"  Hard negatives: {hn_metrics}")

    # Latency
    lat = benchmark_latency(model, dataset, tokenizer, device)
    print(f"  Latency: {lat}")

    # Save results
    if save_results_path:
        save_results_csv(results, save_results_path)

    # Case studies
    if export_case_studies_dir:
        tm = _text_model_name(cfg)
        sbert_sim = torch.mm(
            F.normalize(torch.tensor(
                __import__("sentence_transformers").SentenceTransformer(
                    tm
                ).encode(
                    [dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
                     else dataset.samples[i].query for i in range(len(dataset))]
                )
            ), p=2, dim=-1),
            F.normalize(torch.tensor(
                __import__("sentence_transformers").SentenceTransformer(
                    tm
                ).encode(
                    [dataset.get_dag_text(j) for j in range(dataset.num_unique_dags)]
                )
            ), p=2, dim=-1).t(),
        )
        sbert_topk = sbert_sim.topk(k=5, dim=1).indices

        queries_for_bm25 = [
            dataset.samples[i]["query"] if isinstance(dataset.samples[i], dict)
            else dataset.samples[i].query for i in range(len(dataset))
        ]
        from rank_bm25 import BM25Okapi
        dag_texts_bm25 = [dataset.get_dag_text(j) for j in range(dataset.num_unique_dags)]
        bm25_obj = BM25Okapi([d.lower().split() for d in dag_texts_bm25])
        bm25_topk_list = []
        for q in queries_for_bm25:
            scores = bm25_obj.get_scores(q.lower().split())
            bm25_topk_list.append(np.argsort(scores)[::-1][:5].tolist())
        bm25_topk = torch.tensor(bm25_topk_list)

        export_case_studies(
            dataset, legr_topk, sbert_topk, bm25_topk, gt_dag_ids,
            export_case_studies_dir, max_cases=max_case_studies,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Evaluate LEGR dual-encoder (one or two checkpoints for GED ablation)",
    )
    p.add_argument(
        "--checkpoint",
        dest="checkpoint_groups",
        action="append",
        nargs="+",
        metavar="PATH",
        required=True,
        help="Path(s) to .pt checkpoint(s). Use once with two paths, or repeat the flag: "
        "--checkpoint WITH_GED/best_model.pt NO_GED/best_model.pt "
        "or --checkpoint a.pt --checkpoint b.pt. Baselines run once in two-checkpoint mode.",
    )
    p.add_argument(
        "--checkpoint_labels",
        nargs=2,
        metavar=("LABEL1", "LABEL2"),
        default=None,
        help="Labels for two-checkpoint mode (order matches --checkpoint). "
        "Default: LEGR_WITH_GED LEGR_NO_GED",
    )
    p.add_argument("--dataset_csv", type=str, default=None,
                    help="CSV or Parquet eval dataset (query, tools, edges)")
    p.add_argument("--hard_negative_csv", type=str, default=None,
                    help="Hard-negative CSV for ranking evaluation")
    p.add_argument("--save_results", type=str, default=None,
                    help="Save metrics to CSV at this path")
    p.add_argument("--export_case_studies", type=str, default=None,
                    help="Directory to export case studies")
    p.add_argument("--max_case_studies", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Flatten: supports `--checkpoint a b` and `--checkpoint a --checkpoint b`
    checkpoint_paths = [p for g in args.checkpoint_groups for p in g]
    n_ckpt = len(checkpoint_paths)
    if n_ckpt not in (1, 2):
        p.error("Provide one or two checkpoint paths in total.")

    if n_ckpt == 1:
        evaluate(
            checkpoint_path=checkpoint_paths[0],
            dataset_csv=args.dataset_csv,
            hard_negative_csv=args.hard_negative_csv,
            save_results_path=args.save_results,
            export_case_studies_dir=args.export_case_studies,
            max_case_studies=args.max_case_studies,
            seed=args.seed,
        )
    else:
        labels = list(args.checkpoint_labels) if args.checkpoint_labels else [
            "LEGR_WITH_GED",
            "LEGR_NO_GED",
        ]
        evaluate_ablation_two(
            checkpoint_paths=checkpoint_paths,
            labels=labels,
            dataset_csv=args.dataset_csv,
            hard_negative_csv=args.hard_negative_csv,
            save_results_path=args.save_results,
            export_case_studies_dir=args.export_case_studies,
            max_case_studies=args.max_case_studies,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
