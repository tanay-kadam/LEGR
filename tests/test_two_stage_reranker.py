from __future__ import annotations

import torch

from legr_experiments.two_stage_reranker import (
    V3PairReranker,
    hierarchical_scores,
    same_toolset_pair_indices,
)


def test_zero_initialized_head_starts_as_v3_cosine():
    model = V3PairReranker(embed_dim=4, hidden_dim=8, dropout=0)
    query = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    graph = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    torch.testing.assert_close(model(query, graph), (query * graph).sum(-1))


def test_hierarchy_cannot_change_sbert_selected_toolset():
    sbert = torch.tensor([[0.1, 0.9, 0.8]])
    structural = torch.tensor([[100.0, 0.1, 0.9]])
    tools = torch.tensor([[1, 0], [0, 1], [0, 1]])
    scores = hierarchical_scores(sbert, structural, tools)
    assert torch.isneginf(scores[0, 0])
    assert scores.argmax(dim=1).item() == 2


def test_pair_builder_uses_only_same_toolset_negatives():
    tools = torch.tensor([[1, 0], [1, 0], [0, 1]])
    query, positive, negative = same_toolset_pair_indices([0, 2], tools)
    assert query.tolist() == [0]
    assert positive.tolist() == [0]
    assert negative.tolist() == [1]
