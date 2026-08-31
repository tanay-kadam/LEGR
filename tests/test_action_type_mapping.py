"""Action-type mapping and latent-space diagnostic tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from action_type_mapping import (  # noqa: E402
    GROUP_MIXED,
    GROUP_MOSTLY_ORCHESTRATE,
    GROUP_MOSTLY_READ,
    GROUP_MOSTLY_WRITE,
    TOOL_ACTION_TYPE,
    action_type_of,
    classify_dag_action_group,
    mapping_covers,
)
from data_synth import _FULL_TOOL_VOCAB  # noqa: E402
from latent_space_metrics import classify_support, embedding_diagnostics  # noqa: E402


def test_every_legr_tool_is_mapped():
    assert mapping_covers(_FULL_TOOL_VOCAB) == []
    assert len(TOOL_ACTION_TYPE) == 45


def test_15tool_source_of_truth():
    assert action_type_of("db_read") == "read"
    assert action_type_of("db_write") == "write"
    assert action_type_of("process_refund") == "write"
    assert action_type_of("create_ticket") == "orchestrate"


def test_unmapped_tool_raises():
    with pytest.raises(KeyError):
        action_type_of("not_a_real_tool")
    with pytest.raises(KeyError):
        classify_dag_action_group(["db_read", "not_a_real_tool"])


def test_majority_and_mixed():
    assert classify_dag_action_group(["db_read", "db_read", "create_ticket"]) == GROUP_MOSTLY_READ
    assert classify_dag_action_group(["db_write", "reset_password"]) == GROUP_MOSTLY_WRITE
    assert classify_dag_action_group(["create_ticket"]) == GROUP_MOSTLY_ORCHESTRATE
    assert classify_dag_action_group(["db_read", "db_write"]) == GROUP_MIXED
    assert classify_dag_action_group([]) == GROUP_MIXED


def test_classify_support_thresholds():
    assert classify_support(0.4, 0.5, 0.8, 0.4) == "STRONG SUPPORT"
    assert classify_support(0.12, 0.0, 0.3, 0.5) == "WEAK/PARTIAL SUPPORT"
    assert classify_support(-0.05, 0.0, 0.4, 0.5) == "NO SUPPORT"


def test_diagnostics_perfect_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.0, scale=0.01, size=(20, 8))
    b = rng.normal(loc=5.0, scale=0.01, size=(20, 8))
    embs = np.vstack([a, b])
    labels = ["mostly-read"] * 20 + ["mostly-write"] * 20
    diag = embedding_diagnostics(embs, labels, knn_k=3)
    assert diag["embedding_space"] == "original"
    assert diag["tsne_not_used_for_metrics"] is True
    assert diag["silhouette"] is not None and diag["silhouette"] > 0.5
    assert diag["neighborhood_purity"] > 0.9
    assert diag["evidence"] == "STRONG SUPPORT"


def test_diagnostics_noise_is_not_strong():
    rng = np.random.default_rng(1)
    embs = rng.normal(size=(40, 8))
    labels = (["mostly-read"] * 10 + ["mostly-write"] * 10
              + ["mostly-orchestrate"] * 10 + ["mixed"] * 10)
    diag = embedding_diagnostics(embs, labels, knn_k=3)
    assert diag["evidence"] in {"NO SUPPORT", "WEAK/PARTIAL SUPPORT"}
    assert diag["evidence"] != "STRONG SUPPORT"
