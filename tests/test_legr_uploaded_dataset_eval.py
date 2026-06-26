"""Read-only checks for evaluating LEGR on the uploaded 30-tool dataset."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import vocab_config as vc


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _fresh_import_eval(argv: list[str]):
    _clear_modules("eval", "train", "data_synth", "encoders", "loss")
    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        return importlib.import_module("eval")
    finally:
        sys.argv = old_argv


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    vc.ACTIVE_TOOL_COUNT = 45
    _clear_modules("eval", "train", "data_synth", "encoders", "loss")


def test_uploaded_30tool_eval_dataset_loads_without_hard_negatives():
    dataset_path = ROOT / "upgraded" / "upgraded_30tools" / "test_topology_heldout.csv"
    assert dataset_path.exists()

    eval_module = _fresh_import_eval(
        ["eval.py", "--tool_count", "30", "--checkpoint", "dummy.pt"]
    )

    resolved_dataset, resolved_hard_negatives = eval_module._resolve_eval_csv_paths(
        str(dataset_path),
        None,
        tool_count=30,
    )

    assert Path(resolved_dataset) == dataset_path
    assert resolved_hard_negatives is None

    df = pd.read_csv(dataset_path, keep_default_na=False)
    dataset = eval_module.CSVEvalDataset(df)
    assert len(dataset) == 1200
    assert dataset.num_unique_dags == 90
