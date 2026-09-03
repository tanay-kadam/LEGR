from __future__ import annotations

import numpy as np
import torch

from legr_experiments.final_evaluation import (
    dag_clustered_paired_bootstrap,
    retrieval_diagnostics,
    vectorized_relation_scores,
)
from legr_experiments.model import LEGRResearchModel
from legr_experiments.structures import REL_IGNORE


def test_tie_aware_metrics_remove_gallery_order_advantage():
    scores = np.asarray([[1.0, 1.0, 0.0], [0.2, 0.9, 0.1]], dtype=np.float32)
    gold = [1, 1]
    query_tools = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    candidate_tools = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    metrics, details = retrieval_diagnostics(
        scores, gold, query_tools, candidate_tools, tolerance=1e-5
    )
    assert metrics["recall@1"] == 0.5
    assert metrics["tie_expected_recall@1"] == 0.75
    assert details["best_tie_ranks"].tolist() == [1, 1]
    assert details["worst_tie_ranks"].tolist() == [2, 1]


def test_true_twin_metric_excludes_singleton_toolsets():
    scores = np.asarray([[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]], dtype=np.float32)
    gold = [0, 2]
    query_tools = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    candidate_tools = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    metrics, _ = retrieval_diagnostics(scores, gold, query_tools, candidate_tools)
    assert metrics["true_twin_queries"] == 1
    assert metrics["true_twin_recall@1"] == 1.0
    assert metrics["true_twin_random_chance"] == 0.5


def test_masked_gold_is_counted_as_wrong_not_as_a_tie():
    scores = np.asarray([[-np.inf, -np.inf, 0.9]], dtype=np.float32)
    tools = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    metrics, details = retrieval_diagnostics(scores, [0], tools[[0]], tools)
    assert metrics["recall@1"] == 0.0
    assert metrics["tie_expected_recall@1"] == 0.0
    assert metrics["true_twin_recall@1"] == 0.0
    assert metrics["tie_expected_true_twin_recall@1"] == 0.0
    assert details["worst_tie_ranks"].tolist() == [3]


def test_dag_clustered_bootstrap_uses_dags_not_rows():
    left = [1, 1, 0, 0]
    right = [0, 0, 0, 0]
    result = dag_clustered_paired_bootstrap(
        left, right, ["a", "a", "b", "b"], samples=1000, seed=42
    )
    assert result["clusters"] == 2
    assert result["delta"] == 0.5
    assert result["ci95_low"] <= 0.5 <= result["ci95_high"]


def test_vectorized_relation_scores_match_model_reference():
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(3, 4, 4, 5, generator=generator)
    targets = torch.randint(0, 5, (7, 4, 4), generator=generator)
    targets[:, torch.arange(4), torch.arange(4)] = REL_IGNORE
    targets[0, 0, 1] = REL_IGNORE
    expected = LEGRResearchModel._relation_scores(logits, targets)
    actual = vectorized_relation_scores(logits, targets)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
