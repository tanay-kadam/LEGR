"""Regression tests for tool-count-aware raw graph generation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import vocab_config as vc


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _fresh_import_upgrade_graph(tool_count: int):
    vc.ACTIVE_TOOL_COUNT = tool_count
    _clear_modules("scripts.upgrade_graph", "utils.graph_utils", "data_synth")
    return importlib.import_module("scripts.upgrade_graph")


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    vc.ACTIVE_TOOL_COUNT = 45
    _clear_modules("scripts.upgrade_graph", "utils.graph_utils", "data_synth")


def test_upgrade_graph_respects_30tool_vocab():
    upgrade_graph = _fresh_import_upgrade_graph(30)

    assert len(upgrade_graph.TOOL_VOCAB) == 30
    generated = upgrade_graph.generate_augmented_data(existing_hashes=set(), cfg={
        "seed": 42,
        "graph": {
            "n_new_dags": 8,
            "queries_per_dag": 1,
            "new_dag_families": ["diamond", "asymmetric_fork_join", "repeated_tool"],
        },
    })
    dataset_tools = {
        tool
        for cell in generated["tools"]
        for tool in str(cell).split(";")
        if tool
    }

    assert dataset_tools <= set(upgrade_graph.TOOL_VOCAB)


def test_upgrade_graph_respects_45tool_vocab():
    upgrade_graph = _fresh_import_upgrade_graph(45)

    assert len(upgrade_graph.TOOL_VOCAB) == 45
