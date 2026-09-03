"""Locked, read-only evaluation utilities for the Campaign-v4 fair gallery.

The functions in this module never train models or modify Campaign-v4 inputs.
They provide deterministic gallery construction, cached candidate scoring,
tie-aware retrieval metrics, and DAG-clustered paired bootstrap intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import ResearchDataset, ResearchSample, UniqueGraphDataset, make_collate
from .model import LEGRResearchModel
from .structures import REL_IGNORE


@dataclass
class CandidateCache:
    gps: torch.Tensor
    v3: torch.Tensor
    semantic: torch.Tensor
    tools: torch.Tensor
    relations: torch.Tensor
    keys: list[str]
    samples: list[ResearchSample]
    build_seconds: float


class SampleListDataset(Dataset):
    def __init__(self, samples: Sequence[ResearchSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ResearchSample:
        return self.samples[index]


def deterministic_gallery(dataset: ResearchDataset, seed: int = 42) -> SampleListDataset:
    """Deduplicate, sort by canonical key, then shuffle independently of CSV order."""
    samples = sorted(UniqueGraphDataset(dataset).samples, key=lambda sample: sample.signature.dag_key)
    random.Random(seed).shuffle(samples)
    keys = [sample.signature.dag_key for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Gallery still contains duplicate canonical DAG keys")
    return SampleListDataset(samples)


@torch.no_grad()
def build_candidate_cache(
    model: LEGRResearchModel,
    gallery: SampleListDataset,
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> CandidateCache:
    model.eval()
    loader = DataLoader(
        gallery,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_length=max_length),
        num_workers=0,
    )
    collected = {"gps": [], "v3": [], "semantic": [], "tools": [], "relations": []}
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for batch in loader:
        graph_x = batch["graph_x"].to(device)
        edge_index = batch["graph_edge_index"].to(device)
        graph_batch = batch["graph_batch"].to(device)
        structural = batch["graph_struct_x"].to(device)
        node_features = model.base_legr._node_features_from_tool_ids(graph_x)
        gps, _ = model.graph_adapter(node_features, structural, edge_index, graph_batch)
        v3 = model.base_legr.encode_graph(node_features, edge_index, graph_batch, topo_pos=None)
        semantic = model.semantic_expert.encode_document(
            batch["doc_input_ids"].to(device), batch["doc_attention_mask"].to(device)
        )
        collected["gps"].append(gps)
        collected["v3"].append(v3)
        collected["semantic"].append(semantic)
        collected["tools"].append(batch["tool_targets"].to(device))
        collected["relations"].append(batch["relation_targets"].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tensors = {name: torch.cat(values, dim=0) for name, values in collected.items()}
    count = len(gallery)
    if any(value.size(0) != count for value in tensors.values()):
        raise ValueError("Candidate cache has inconsistent lengths")
    for name in ("gps", "v3", "semantic"):
        if not torch.isfinite(tensors[name]).all():
            raise ValueError(f"Non-finite candidate embeddings in {name}")
    return CandidateCache(
        gps=tensors["gps"],
        v3=tensors["v3"],
        semantic=tensors["semantic"],
        tools=tensors["tools"],
        relations=tensors["relations"],
        keys=[sample.signature.dag_key for sample in gallery.samples],
        samples=list(gallery.samples),
        build_seconds=elapsed,
    )


def vectorized_relation_scores(
    relation_logits: torch.Tensor, candidate_targets: torch.Tensor
) -> torch.Tensor:
    """Mean active-pair log likelihood for every query/candidate pair."""
    log_probs = F.log_softmax(relation_logits, dim=-1)
    active = candidate_targets.ne(REL_IGNORE)
    safe_targets = candidate_targets.masked_fill(~active, 0)
    batch, vocab, _, classes = log_probs.shape
    candidates = candidate_targets.size(0)
    expanded_probs = log_probs[:, None].expand(batch, candidates, vocab, vocab, classes)
    gather_index = safe_targets[None, ..., None].expand(batch, candidates, vocab, vocab, 1)
    selected = expanded_probs.gather(-1, gather_index).squeeze(-1)
    weights = active[None].to(selected.dtype)
    return (selected * weights).sum(dim=(-1, -2)) / weights.sum(dim=(-1, -2)).clamp(min=1)


@torch.no_grad()
def score_cached(
    model: LEGRResearchModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: CandidateCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fused and five-expert scores without re-encoding candidates."""
    tokens, pooled, z_query, tool_logits, relation_logits = model._query_outputs(
        input_ids, attention_mask
    )
    del tokens
    semantic_query = model.semantic_expert.encode_query(input_ids, attention_mask)
    v3_query = F.normalize(model.base_legr.text_encoder.proj(pooled), dim=-1)
    semantic_score = semantic_query @ cache.semantic.t()
    v3_score = v3_query @ cache.v3.t()
    gps_score = z_query @ cache.gps.t()
    tool_score = F.normalize(torch.sigmoid(tool_logits), dim=-1) @ F.normalize(cache.tools, dim=-1).t()
    relation_score = vectorized_relation_scores(relation_logits, cache.relations)
    experts = torch.stack(
        [semantic_score, v3_score, gps_score, tool_score, relation_score], dim=-1
    )
    return model._fuse(pooled, experts), experts


@torch.no_grad()
def score_query_dataset(
    model: LEGRResearchModel,
    query_dataset: ResearchDataset,
    tokenizer,
    cache: CandidateCache,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
    all_scores, all_experts = [], []
    gold_indices, gold_keys = [], []
    key_to_index = {key: index for index, key in enumerate(cache.keys)}
    for start in range(0, len(query_dataset), batch_size):
        samples = query_dataset.samples[start : start + batch_size]
        encoded = tokenizer(
            [sample.query for sample in samples],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        scores, experts = score_cached(
            model,
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
            cache,
        )
        all_scores.append(scores.cpu())
        all_experts.append(experts.cpu())
        for sample in samples:
            key = sample.signature.dag_key
            if key not in key_to_index:
                raise ValueError(f"Gold DAG {key} is absent from the combined gallery")
            gold_keys.append(key)
            gold_indices.append(key_to_index[key])
    return torch.cat(all_scores), torch.cat(all_experts), gold_indices, gold_keys


def tool_f1(predicted: np.ndarray, gold: np.ndarray) -> float:
    predicted = predicted.astype(bool)
    gold = gold.astype(bool)
    tp = int(np.logical_and(predicted, gold).sum())
    fp = int(np.logical_and(predicted, ~gold).sum())
    fn = int(np.logical_and(~predicted, gold).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return 2 * precision * recall / max(1e-12, precision + recall)


def _tie_rank_probabilities(row: np.ndarray, gold: int, ks: Sequence[int], tolerance: float):
    gold_score = row[gold]
    if not np.isfinite(gold_score):
        zeros = {k: 0.0 for k in ks}
        finite = int(np.isfinite(row).sum())
        return finite + 1, len(row), zeros, zeros.copy()
    better = int((row > gold_score + tolerance).sum())
    tied = int((np.abs(row - gold_score) <= tolerance).sum())
    recall, mrr = {}, {}
    for k in ks:
        slots = max(0, min(tied, k - better))
        recall[k] = slots / max(1, tied)
        mrr[k] = sum(
            (1.0 / (better + offset)) if better + offset <= k else 0.0
            for offset in range(1, tied + 1)
        ) / max(1, tied)
    return better + 1, better + tied, recall, mrr


def retrieval_diagnostics(
    scores: torch.Tensor | np.ndarray,
    gold_indices: Sequence[int],
    query_tools: torch.Tensor | np.ndarray,
    candidate_tools: torch.Tensor | np.ndarray,
    tolerance: float = 1e-5,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute deterministic and random-tie-expected metrics.

    The candidate order must already be deterministically randomized. Stable
    sorting then provides a reproducible point estimate, while the expected
    metrics remove any remaining dependence on order for numerical ties.
    """
    values = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.asarray(scores)
    qtools = query_tools.detach().cpu().numpy() if isinstance(query_tools, torch.Tensor) else np.asarray(query_tools)
    ctools = candidate_tools.detach().cpu().numpy() if isinstance(candidate_tools, torch.Tensor) else np.asarray(candidate_tools)
    gold = np.asarray(gold_indices, dtype=np.int64)
    n = len(gold)
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    best_ranks = np.empty(n, dtype=np.int64)
    worst_ranks = np.empty(n, dtype=np.int64)
    predicted = order[:, 0]
    per_tool_f1 = np.empty(n, dtype=np.float64)
    expected_top_tool_f1 = np.empty(n, dtype=np.float64)
    twin_eligible = np.zeros(n, dtype=bool)
    twin_det_correct = np.zeros(n, dtype=np.float64)
    twin_expected_correct = np.zeros(n, dtype=np.float64)
    twin_chance = np.zeros(n, dtype=np.float64)
    hard_pair_accuracy = np.full(n, np.nan, dtype=np.float64)
    tie_recall = {k: np.empty(n, dtype=np.float64) for k in (1, 3, 5)}
    tie_mrr = {k: np.empty(n, dtype=np.float64) for k in (1, 3, 5)}

    for row in range(n):
        ranks[row] = int(np.where(order[row] == gold[row])[0][0]) + 1
        best, worst, recalls, mrrs = _tie_rank_probabilities(
            values[row], int(gold[row]), (1, 3, 5), tolerance
        )
        best_ranks[row], worst_ranks[row] = best, worst
        for k in (1, 3, 5):
            tie_recall[k][row] = recalls[k]
            tie_mrr[k][row] = mrrs[k]
        per_tool_f1[row] = tool_f1(ctools[predicted[row]], qtools[row])
        top_tied = np.flatnonzero(values[row] >= values[row].max() - tolerance)
        expected_top_tool_f1[row] = np.mean(
            [tool_f1(ctools[index], qtools[row]) for index in top_tied]
        )

        same = np.flatnonzero(np.all(ctools == qtools[row], axis=1))
        if gold[row] in same and len(same) >= 2:
            twin_eligible[row] = True
            if not np.isfinite(values[row, gold[row]]):
                twin_det_correct[row] = 0.0
                twin_expected_correct[row] = 0.0
                twin_chance[row] = 1.0 / len(same)
                hard_pair_accuracy[row] = 0.0
                continue
            local_order = same[np.argsort(-values[row, same], kind="stable")]
            twin_det_correct[row] = float(local_order[0] == gold[row])
            local_scores = values[row, same]
            local_gold = int(np.where(same == gold[row])[0][0])
            _, _, local_recall, _ = _tie_rank_probabilities(
                local_scores, local_gold, (1,), tolerance
            )
            twin_expected_correct[row] = local_recall[1]
            twin_chance[row] = 1.0 / len(same)
            negative_scores = values[row, same[same != gold[row]]]
            wins = (values[row, gold[row]] > negative_scores + tolerance).astype(float)
            ties = (np.abs(values[row, gold[row]] - negative_scores) <= tolerance).astype(float)
            hard_pair_accuracy[row] = float(np.mean(wins + 0.5 * ties))

    metrics: dict[str, float] = {
        "n_queries": float(n),
        "gallery_size": float(values.shape[1]),
        "tool_set_f1": float(per_tool_f1.mean()),
        "tie_expected_tool_set_f1": float(expected_top_tool_f1.mean()),
        "true_twin_queries": float(twin_eligible.sum()),
        "true_twin_recall@1": float(twin_det_correct[twin_eligible].mean()),
        "tie_expected_true_twin_recall@1": float(twin_expected_correct[twin_eligible].mean()),
        "true_twin_random_chance": float(twin_chance[twin_eligible].mean()),
        "hard_negative_pair_accuracy": float(np.nanmean(hard_pair_accuracy)),
    }
    for k in (1, 3, 5):
        metrics[f"recall@{k}"] = float((ranks <= k).mean())
        metrics[f"mrr@{k}"] = float(np.where(ranks <= k, 1.0 / ranks, 0.0).mean())
        metrics[f"tie_expected_recall@{k}"] = float(tie_recall[k].mean())
        metrics[f"tie_expected_mrr@{k}"] = float(tie_mrr[k].mean())
    details = {
        "predicted_indices": predicted,
        "ranks": ranks,
        "best_tie_ranks": best_ranks,
        "worst_tie_ranks": worst_ranks,
        "exact_correct": (ranks == 1).astype(np.float64),
        "tie_expected_exact_correct": tie_recall[1],
        "tool_f1": per_tool_f1,
        "tie_expected_tool_f1": expected_top_tool_f1,
        "twin_eligible": twin_eligible,
        "twin_exact_correct": twin_det_correct,
        "twin_expected_correct": twin_expected_correct,
        "hard_negative_pair_accuracy": hard_pair_accuracy,
    }
    return metrics, details


def dag_clustered_paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    dag_keys: Sequence[str],
    samples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap paired deltas over DAGs, retaining paraphrases as a cluster."""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    keys = np.asarray(dag_keys)
    if not (len(left_values) == len(right_values) == len(keys)):
        raise ValueError("Bootstrap inputs have different lengths")
    unique = np.unique(keys)
    cluster_delta = np.asarray([
        (left_values[keys == key] - right_values[keys == key]).mean() for key in unique
    ])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(samples, len(unique)))
    values = cluster_delta[draws].mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    observed = float(cluster_delta.mean())
    return {
        "delta": observed,
        "bootstrap_mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "clusters": int(len(unique)),
        "samples": int(samples),
        "seed": int(seed),
    }


@torch.no_grad()
def synchronized_cached_latency(
    model: LEGRResearchModel,
    queries: Sequence[str],
    tokenizer,
    cache: CandidateCache,
    device: torch.device,
    warmup: int = 20,
    max_length: int = 128,
) -> dict[str, float]:
    """Time batch-size-one query encoding and scoring with candidates cached."""
    model.eval()
    if not queries:
        raise ValueError("No latency queries")

    def run_one(query: str) -> None:
        encoded = tokenizer(
            [query], truncation=True, max_length=max_length, return_tensors="pt"
        )
        score_cached(
            model,
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
            cache,
        )

    for index in range(warmup):
        run_one(queries[index % len(queries)])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = []
    for query in queries:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        run_one(query)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - started) * 1000)
    values = np.asarray(elapsed, dtype=np.float64)
    return {
        "trials": int(len(values)),
        "warmup": int(warmup),
        "batch_size": 1,
        "candidate_cache_size": int(len(cache.keys)),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "candidate_cache_build_seconds": float(cache.build_seconds),
    }
