"""
run_bm25_baseline.py — BM25 Baseline Evaluation for Campaign v4
================================================================

Runs BM25 retrieval against the campaign_v4 test sets (indomain + heldout).
No model training needed — BM25 indexes dag_text tokens and ranks by query.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _load_candidate_corpus(tier_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tier_dir / "candidate_corpus.csv")


def _load_test_set(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _compute_bm25_metrics(
    test_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    k_values=(1, 3, 5),
) -> dict:
    from rank_bm25 import BM25Okapi

    dag_texts = corpus_df["dag_text"].tolist()
    dag_ids_corpus = corpus_df["dag_id"].tolist()
    tokenized = [doc.lower().replace("->", " ").replace(";", " ").split() for doc in dag_texts]
    bm25 = BM25Okapi(tokenized)

    dag_id_to_corpus_idx = {}
    for idx, did in enumerate(dag_ids_corpus):
        if did not in dag_id_to_corpus_idx:
            dag_id_to_corpus_idx[did] = idx

    queries = test_df["query"].tolist()
    gt_dag_ids = test_df["dag_id"].tolist()

    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    rr_sum = {k: 0.0 for k in k_values}
    n = len(queries)

    for i, q in enumerate(queries):
        scores = bm25.get_scores(q.lower().split())
        top_indices = np.argsort(scores)[::-1][:max_k]
        top_dag_ids = [dag_ids_corpus[j] for j in top_indices]

        gt = gt_dag_ids[i]
        for k in k_values:
            topk = top_dag_ids[:k]
            if gt in topk:
                hits[k] += 1
                rank = topk.index(gt) + 1
                rr_sum[k] += 1.0 / rank

    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = round(hits[k] / n, 4) if n > 0 else 0
        metrics[f"mrr@{k}"] = round(rr_sum[k] / n, 4) if n > 0 else 0
    metrics["num_queries"] = n
    metrics["corpus_size"] = len(dag_texts)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="BM25 baseline for campaign v4")
    parser.add_argument("--tier", type=int, default=15, choices=[15, 30, 45])
    parser.add_argument("--output-dir", default="artifacts/campaign_v4/results")
    args = parser.parse_args()

    tier_dir = Path(f"data/campaign_v4/campaign_v4_{args.tier}tools")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_df = _load_candidate_corpus(tier_dir)
    print(f"BM25 — Tier {args.tier}: {len(corpus_df)} candidates in corpus")

    results = {"tier": args.tier, "model": "BM25"}

    for split_name, csv_name in [
        ("test_indomain", "test_indomain.csv"),
        ("test_topology_heldout", "test_topology_heldout.csv"),
    ]:
        csv_path = tier_dir / csv_name
        if not csv_path.exists():
            print(f"  SKIP {split_name}: {csv_path} not found")
            continue

        test_df = _load_test_set(csv_path)
        print(f"  {split_name}: {len(test_df)} queries")
        metrics = _compute_bm25_metrics(test_df, corpus_df)
        results[split_name] = metrics
        print(f"    R@1={metrics['recall@1']:.4f}  R@3={metrics['recall@3']:.4f}  "
              f"R@5={metrics['recall@5']:.4f}  MRR@5={metrics['mrr@5']:.4f}")

    out_path = out_dir / f"bm25_{args.tier}t.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
