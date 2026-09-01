"""
run_frozen_sbert_baseline.py — Frozen SBERT Baseline for Campaign v4
====================================================================

Evaluates frozen Sentence-BERT (no fine-tuning) as a retrieval baseline.
Encodes queries and dag_text using MiniLM, retrieves by cosine similarity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _encode_batch(texts: list[str], tokenizer, model, device: str, batch_size: int = 64) -> torch.Tensor:
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=-1)
        all_embeds.append(emb.cpu())
    return torch.cat(all_embeds, dim=0)


def _compute_metrics(
    test_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    k_values=(1, 3, 5),
) -> dict:
    corpus_texts = corpus_df["dag_text"].tolist()
    corpus_dag_ids = corpus_df["dag_id"].tolist()

    query_texts = test_df["query"].tolist()
    gt_dag_ids = test_df["dag_id"].tolist()

    corpus_emb = _encode_batch(corpus_texts, tokenizer, model, device)
    query_emb = _encode_batch(query_texts, tokenizer, model, device)

    sim = query_emb @ corpus_emb.t()

    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    rr_sum = {k: 0.0 for k in k_values}
    n = len(query_texts)

    for i in range(n):
        top_indices = sim[i].topk(max_k).indices.tolist()
        top_dag_ids = [corpus_dag_ids[j] for j in top_indices]
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
    metrics["corpus_size"] = len(corpus_texts)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Frozen SBERT baseline for campaign v4")
    parser.add_argument("--tier", type=int, default=15, choices=[15, 30, 45])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/campaign_v4/results")
    args = parser.parse_args()

    tier_dir = Path(f"data/campaign_v4/campaign_v4_{args.tier}tools")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(args.device).eval()

    corpus_df = pd.read_csv(tier_dir / "candidate_corpus.csv")
    print(f"Frozen SBERT — Tier {args.tier}: {len(corpus_df)} candidates")

    results = {"tier": args.tier, "model": "Frozen_SBERT (all-MiniLM-L6-v2)"}

    for split_name, csv_name in [
        ("test_indomain", "test_indomain.csv"),
        ("test_topology_heldout", "test_topology_heldout.csv"),
    ]:
        csv_path = tier_dir / csv_name
        if not csv_path.exists():
            print(f"  SKIP {split_name}")
            continue

        test_df = pd.read_csv(csv_path)
        print(f"  {split_name}: {len(test_df)} queries")
        metrics = _compute_metrics(test_df, corpus_df, tokenizer, model, args.device)
        results[split_name] = metrics
        print(f"    R@1={metrics['recall@1']:.4f}  R@3={metrics['recall@3']:.4f}  "
              f"R@5={metrics['recall@5']:.4f}  MRR@5={metrics['mrr@5']:.4f}")

    out_path = out_dir / f"frozen_sbert_{args.tier}t.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
