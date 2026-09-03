"""Two-stage SBERT-FT tool selection and V3 structural reranking.

The wrapper keeps both inherited encoders frozen.  SBERT-FT selects one tool
set; a new two-layer residual head ranks only candidates with that exact set.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class V3PairReranker(nn.Module):
    """A zero-initialized residual MLP over frozen query/graph embeddings."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim * 4)
        self.hidden = nn.Linear(embed_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        features = torch.cat([query, graph, torch.abs(query - graph), query * graph], dim=-1)
        residual = self.output(self.dropout(F.gelu(self.hidden(self.norm(features))))).squeeze(-1)
        return (query * graph).sum(-1) + residual

    def score_matrix(
        self, query_embeddings: torch.Tensor, graph_embeddings: torch.Tensor,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        rows = []
        for start in range(0, len(query_embeddings), chunk_size):
            query = query_embeddings[start : start + chunk_size]
            batch, candidates = len(query), len(graph_embeddings)
            q = query[:, None, :].expand(batch, candidates, -1).reshape(-1, query.size(-1))
            g = graph_embeddings[None, :, :].expand(batch, candidates, -1).reshape(-1, graph_embeddings.size(-1))
            rows.append(self(q, g).reshape(batch, candidates))
        return torch.cat(rows, dim=0)


def same_toolset_pair_indices(
    query_gold_indices: list[int] | np.ndarray,
    candidate_tools: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return query, positive, and all same-tool-set negative indices."""
    tools = candidate_tools.bool().cpu()
    query_rows, positives, negatives = [], [], []
    for row, gold in enumerate(np.asarray(query_gold_indices, dtype=np.int64)):
        same = torch.where((tools == tools[int(gold)]).all(dim=1))[0]
        for negative in same[same != int(gold)].tolist():
            query_rows.append(row)
            positives.append(int(gold))
            negatives.append(negative)
    if not query_rows:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty, empty
    return (
        torch.tensor(query_rows, dtype=torch.long),
        torch.tensor(positives, dtype=torch.long),
        torch.tensor(negatives, dtype=torch.long),
    )


def hierarchical_scores(
    sbert_scores: torch.Tensor,
    structural_scores: torch.Tensor,
    candidate_tools: torch.Tensor,
) -> torch.Tensor:
    """Keep the SBERT-selected tool set and rerank only within that set."""
    if sbert_scores.shape != structural_scores.shape:
        raise ValueError("SBERT and structural score matrices must have the same shape")
    tools = candidate_tools.bool().to(sbert_scores.device)
    selected = sbert_scores.argmax(dim=1)
    selected_tools = tools[selected]
    eligible = (tools.unsqueeze(0) == selected_tools.unsqueeze(1)).all(dim=-1)
    output = structural_scores.masked_fill(~eligible, float("-inf"))
    if not torch.isfinite(output.max(dim=1).values).all():
        raise RuntimeError("A selected SBERT tool set has no eligible structural candidate")
    return output
