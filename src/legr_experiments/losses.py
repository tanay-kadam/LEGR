from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig
from .structures import REL_IGNORE, relation_distance


def _positive_logprob(logits: torch.Tensor, positives: torch.Tensor, eligible=None) -> torch.Tensor:
    if eligible is None:
        eligible = torch.ones_like(positives, dtype=torch.bool)
    eligible = eligible | positives
    masked = logits.masked_fill(~eligible, float("-inf"))
    numerator = torch.logsumexp(logits.masked_fill(~positives, float("-inf")), dim=1)
    denominator = torch.logsumexp(masked, dim=1)
    valid = positives.any(dim=1) & eligible.any(dim=1)
    return -(numerator[valid] - denominator[valid]).mean() if valid.any() else logits.sum() * 0


def multi_positive_info_nce(
    query_embeddings: torch.Tensor,
    graph_embeddings: torch.Tensor,
    dag_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = query_embeddings @ graph_embeddings.t() / max(temperature, 1e-6)
    positives = dag_ids[:, None].eq(dag_ids[None, :])
    return 0.5 * (
        _positive_logprob(logits, positives)
        + _positive_logprob(logits.t(), positives.t())
    )


def twin_listwise_loss(
    scores: torch.Tensor,
    dag_ids: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    positives = dag_ids[:, None].eq(dag_ids[None, :])
    eligible = group_ids[:, None].eq(group_ids[None, :])
    has_negative = (eligible & ~positives).any(dim=1)
    if not has_negative.any():
        return scores.sum() * 0
    row_loss = []
    for row in torch.where(has_negative)[0].tolist():
        row_loss.append(_positive_logprob(
            scores[row : row + 1],
            positives[row : row + 1],
            eligible[row : row + 1],
        ))
    return torch.stack(row_loss).mean()


def pairwise_twin_loss(
    scores: torch.Tensor,
    dag_ids: torch.Tensor,
    group_ids: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    values = []
    for row in range(scores.size(0)):
        pos = scores[row][dag_ids == dag_ids[row]].mean()
        neg_mask = (group_ids == group_ids[row]) & (dag_ids != dag_ids[row])
        if neg_mask.any():
            values.append(F.softplus(margin - pos + scores[row][neg_mask]).mean())
    return torch.stack(values).mean() if values else scores.sum() * 0


def balanced_tool_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positives = target.sum().clamp(min=1)
    negatives = (1 - target).sum().clamp(min=1)
    pos_weight = (negatives / positives).detach()
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def relation_classification_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=REL_IGNORE)


def embedding_distance_loss(graph_embeddings: torch.Tensor, relation_targets: torch.Tensor) -> torch.Tensor:
    distances = []
    predicted = []
    for i in range(graph_embeddings.size(0)):
        for j in range(i + 1, graph_embeddings.size(0)):
            mask = (relation_targets[i] != REL_IGNORE) & (relation_targets[j] != REL_IGNORE)
            if mask.any():
                distances.append(relation_distance(relation_targets[i], relation_targets[j]))
                predicted.append(1 - F.cosine_similarity(graph_embeddings[i], graph_embeddings[j], dim=0))
    if not distances:
        return graph_embeddings.sum() * 0
    target = torch.stack(distances).to(graph_embeddings.device)
    pred = torch.stack(predicted)
    return F.smooth_l1_loss(pred, target)


@dataclass
class LossOutput:
    total: torch.Tensor
    parts: dict[str, torch.Tensor]

    def detached(self) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.parts.items()}


class CompositeRetrievalLoss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config

    def forward(self, output: dict, batch: dict) -> LossOutput:
        cfg = self.config
        scores = output["scores"]
        dag_ids = batch["dag_ids"].to(scores.device)
        group_ids = batch["group_ids"].to(scores.device)
        positives = dag_ids[:, None].eq(dag_ids[None, :])
        listwise = _positive_logprob(scores, positives)
        if cfg.use_twin:
            if cfg.rank_kind == "pairwise":
                twin = pairwise_twin_loss(scores, dag_ids, group_ids, cfg.margin)
            else:
                twin = twin_listwise_loss(scores, dag_ids, group_ids)
        else:
            twin = scores.sum() * 0
        tool = balanced_tool_loss(
            output["tool_logits"], batch["tool_targets"].to(scores.device),
        ) if cfg.use_tool else scores.sum() * 0
        relation = relation_classification_loss(
            output["relation_logits"], batch["relation_targets"].to(scores.device),
        ) if cfg.use_relation else scores.sum() * 0
        multi = multi_positive_info_nce(
            output["query_embedding"], output["graph_embedding"], dag_ids, cfg.temperature,
        ) if cfg.use_multi_positive else scores.sum() * 0
        distance = embedding_distance_loss(
            output["graph_embedding"], batch["relation_targets"].to(scores.device),
        ) if cfg.use_distance else scores.sum() * 0
        parts = {
            "listwise": listwise,
            "twin": twin,
            "tool": tool,
            "relation": relation,
            "multi_positive": multi,
            "distance": distance,
        }
        total = (
            cfg.listwise_weight * listwise
            + cfg.twin_weight * twin
            + cfg.tool_weight * tool
            + cfg.relation_weight * relation
            + cfg.multi_positive_weight * multi
            + cfg.distance_weight * distance
        )
        parts["total"] = total
        return LossOutput(total=total, parts=parts)
