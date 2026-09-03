from __future__ import annotations

import math
from typing import Sequence

import torch


def _tool_f1(predicted: torch.Tensor, gold: torch.Tensor) -> float:
    predicted = predicted.bool()
    gold = gold.bool()
    tp = (predicted & gold).sum().item()
    fp = (predicted & ~gold).sum().item()
    fn = (~predicted & gold).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(1e-12, precision + recall)


def retrieval_metrics(
    scores: torch.Tensor,
    gold_indices: torch.Tensor,
    query_tools: torch.Tensor,
    candidate_tools: torch.Tensor,
) -> dict[str, float]:
    scores = scores.float().cpu()
    gold_indices = gold_indices.long().cpu()
    query_tools = query_tools.cpu()
    candidate_tools = candidate_tools.cpu()
    order = scores.argsort(dim=1, descending=True)
    ranks = []
    for row, gold in enumerate(gold_indices.tolist()):
        position = torch.where(order[row] == gold)[0]
        ranks.append(int(position[0]) + 1 if len(position) else scores.size(1) + 1)
    metrics = {}
    for k in (1, 3, 5):
        metrics[f"recall@{k}"] = sum(rank <= k for rank in ranks) / max(1, len(ranks))
        metrics[f"mrr@{k}"] = sum((1 / rank) if rank <= k else 0 for rank in ranks) / max(1, len(ranks))
    predicted = order[:, 0]
    metrics["tool_set_f1"] = sum(
        _tool_f1(candidate_tools[predicted[row]], query_tools[row])
        for row in range(scores.size(0))
    ) / max(1, scores.size(0))

    correct = 0
    eligible_count = 0
    for row, gold in enumerate(gold_indices.tolist()):
        same = (candidate_tools == query_tools[row]).all(dim=1)
        if not same[gold]:
            continue
        same_indices = torch.where(same)[0]
        local_best = same_indices[scores[row, same_indices].argmax()].item()
        correct += int(local_best == gold)
        eligible_count += 1
    metrics["same_toolset_recall@1"] = correct / max(1, eligible_count)
    metrics["n_queries"] = float(scores.size(0))
    metrics["gallery_size"] = float(scores.size(1))
    return metrics


def paired_bootstrap_delta(
    left_correct: Sequence[float], right_correct: Sequence[float], seed: int = 42,
    samples: int = 5000,
) -> tuple[float, float, float]:
    left = torch.tensor(left_correct, dtype=torch.float32)
    right = torch.tensor(right_correct, dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    deltas = []
    for _ in range(samples):
        indices = torch.randint(len(left), (len(left),), generator=generator)
        deltas.append(float((left[indices] - right[indices]).mean()))
    values = torch.tensor(deltas).sort().values
    return float(values.mean()), float(values[int(0.025 * samples)]), float(values[int(0.975 * samples)])
