from __future__ import annotations

import numpy as np

from legr_experiments.functional_clusters import (
    dominant_action_label,
    embedding_diagnostics,
)


REGISTRY = {
    "lookup": "read",
    "fetch": "read",
    "change": "edit",
    "route": "orchestrate",
}


def test_dominant_action_label_uses_unique_plurality():
    label, counts = dominant_action_label(["lookup", "fetch", "change"], REGISTRY)
    assert label == "read"
    assert counts == {"read": 2, "edit": 1, "orchestrate": 0}


def test_dominant_action_label_marks_exact_ties_mixed():
    label, counts = dominant_action_label(["lookup", "change", "route"], REGISTRY)
    assert label == "mixed"
    assert counts == {"read": 1, "edit": 1, "orchestrate": 1}


def test_embedding_diagnostics_detects_separated_clusters():
    rng = np.random.default_rng(42)
    centers = np.eye(3, 8, dtype=np.float32)
    values = []
    labels = []
    for index, label in enumerate(("read", "edit", "orchestrate")):
        cluster = centers[index] + rng.normal(0, 0.025, size=(20, 8))
        cluster /= np.linalg.norm(cluster, axis=1, keepdims=True)
        values.append(cluster.astype(np.float32))
        labels.extend([label] * len(cluster))
    diagnostics = embedding_diagnostics(
        np.concatenate(values), labels, seed=42, permutations=49, bootstraps=100
    )
    assert diagnostics["knn_cv_macro_f1"] > 0.95
    assert diagnostics["between_minus_within_distance"] > 0.5
    assert diagnostics["corroborates_functional_clustering"] is True

