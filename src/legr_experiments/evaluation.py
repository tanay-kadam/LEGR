from __future__ import annotations

import random
import time

import torch
from torch.utils.data import DataLoader

from .data import ResearchDataset, UniqueGraphDataset, make_collate
from .metrics import retrieval_metrics


@torch.no_grad()
def evaluate_gallery(
    model,
    query_dataset: ResearchDataset,
    gallery_dataset: ResearchDataset,
    tokenizer,
    batch_size: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[dict[str, float], dict]:
    model.eval()
    collate = make_collate(tokenizer)
    unique_gallery = UniqueGraphDataset(gallery_dataset)
    rng = random.Random(seed)
    rng.shuffle(unique_gallery.samples)
    gallery_loader = DataLoader(unique_gallery, batch_size=len(unique_gallery), collate_fn=collate)
    candidate_batch = next(iter(gallery_loader))
    candidate_keys = [sample.signature.dag_key for sample in unique_gallery.samples]
    key_to_index = {key: index for index, key in enumerate(candidate_keys)}
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    all_scores = []
    gold = []
    query_tools = []
    latencies = []
    for query_batch in query_loader:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = model.score_batches(query_batch, candidate_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        count = len(query_batch["queries"])
        latencies.extend([elapsed / count] * count)
        all_scores.append(output["scores"].detach().cpu())
        query_tools.append(query_batch["tool_targets"])
        for dag_id in query_batch["dag_ids"].tolist():
            key = query_dataset.signatures[dag_id].dag_key
            gold.append(key_to_index[key])
    scores = torch.cat(all_scores)
    gold_tensor = torch.tensor(gold)
    query_tool_tensor = torch.cat(query_tools)
    candidate_tools = torch.stack([sample.signature.tool_target for sample in unique_gallery.samples])
    metrics = retrieval_metrics(scores, gold_tensor, query_tool_tensor, candidate_tools)
    latency_values = sorted(latencies)
    metrics["mean_latency_ms"] = 1000 * sum(latency_values) / max(1, len(latency_values))
    metrics["p95_latency_ms"] = 1000 * latency_values[min(len(latency_values) - 1, int(0.95 * len(latency_values)))]
    details = {
        "scores": scores,
        "gold_indices": gold_tensor,
        "candidate_keys": candidate_keys,
    }
    return metrics, details
