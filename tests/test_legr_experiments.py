"""Fast tests for the isolated LEGR model-research framework."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from legr_experiments.config import LossConfig
from legr_experiments.graph_encoders import GraphAdapter
from legr_experiments.losses import CompositeRetrievalLoss, multi_positive_info_nce
from legr_experiments.samplers import GroupAwareBatchSampler
from legr_experiments.structures import (
    REL_DIRECT_FWD, REL_DIRECT_REV, REL_INDIRECT_FWD, REL_PARALLEL,
    build_signature, invariant_node_features,
)


VOCAB = ["a", "b", "c", "d"]


def test_relation_signature_distinguishes_direct_indirect_and_parallel():
    signature = build_signature(VOCAB, [(0, 1), (1, 2)], VOCAB, "g")
    assert signature.relation_target[0, 1] == REL_DIRECT_FWD
    assert signature.relation_target[0, 2] == REL_INDIRECT_FWD
    assert signature.relation_target[0, 3] == REL_PARALLEL
    reversed_signature = build_signature(VOCAB, [(1, 0)], VOCAB, "g")
    assert reversed_signature.relation_target[0, 1] == REL_DIRECT_REV


def test_parallel_nodes_receive_same_depth_features():
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    features = invariant_node_features(4, edges, "combined")
    assert torch.allclose(features[1], features[2])


def test_invariant_features_are_permutation_equivariant():
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(4)
    permuted_edges = inverse[edges]
    original = invariant_node_features(4, edges, "combined")
    permuted = invariant_node_features(4, permuted_edges, "combined")
    assert torch.allclose(original[permutation], permuted)


def test_multi_positive_loss_does_not_penalize_duplicate_dag():
    queries = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graphs = queries.clone()
    dag_ids = torch.tensor([0, 0, 1])
    loss = multi_positive_info_nce(queries, graphs, dag_ids, temperature=0.05)
    assert loss < 1e-3


def test_composite_loss_is_finite_with_all_heads():
    batch_size, tools = 3, 4
    output = {
        "scores": torch.eye(batch_size, requires_grad=True),
        "query_embedding": torch.nn.functional.normalize(torch.randn(batch_size, 8), dim=-1),
        "graph_embedding": torch.nn.functional.normalize(torch.randn(batch_size, 8), dim=-1),
        "tool_logits": torch.randn(batch_size, tools, requires_grad=True),
        "relation_logits": torch.randn(batch_size, tools, tools, 5, requires_grad=True),
    }
    relation_targets = torch.full((batch_size, tools, tools), -100, dtype=torch.long)
    relation_targets[:, 0, 1] = 1
    batch = {
        "dag_ids": torch.arange(batch_size),
        "group_ids": torch.tensor([0, 0, 1]),
        "tool_targets": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]]).float(),
        "relation_targets": relation_targets,
    }
    loss = CompositeRetrievalLoss(LossConfig())(output, batch)
    assert torch.isfinite(loss.total)
    loss.total.backward()


@dataclass
class Sample:
    dag_index: int
    group_index: int


def test_group_sampler_uses_one_paraphrase_per_dag_and_colocates_twins():
    samples = [Sample(dag, group) for dag, group in [(0, 0), (0, 0), (1, 0), (1, 0), (2, 1), (2, 1)]]
    sampler = GroupAwareBatchSampler(samples, batch_size=3, seed=42, drop_last=False)
    batch = next(iter(sampler))
    dags = [samples[index].dag_index for index in batch]
    assert len(dags) == len(set(dags))
    assert {0, 1}.issubset(set(dags))


def test_residual_graph_adapter_is_permutation_invariant_at_readout():
    torch.manual_seed(42)
    model = GraphAdapter(4, 8, 6, layers=2, heads=2, dropout=0.0,
                         graph_kind="residual", readout_kind="dual_attention")
    model.eval()
    node_features = torch.randn(4, 4)
    structural = torch.randn(4, 6)
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
    batch = torch.zeros(4, dtype=torch.long)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(4)
    with torch.no_grad():
        original, _ = model(node_features, structural, edges, batch)
        permuted, _ = model(
            node_features[permutation], structural[permutation], inverse[edges], batch,
        )
    assert torch.allclose(original, permuted, atol=1e-5)
