"""Regression tests for the improved LEGR 30-tool dataset flow."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import vocab_config as vc


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _fresh_import(module_name: str, argv: list[str]):
    if module_name == "train":
        _clear_modules("train", "data_synth", "encoders", "loss")
    elif module_name == "eval":
        _clear_modules("eval", "train", "data_synth", "encoders", "loss")
    else:
        _clear_modules(module_name)

    old_argv = sys.argv[:]
    sys.argv = argv
    try:
        return importlib.import_module(module_name)
    finally:
        sys.argv = old_argv


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    vc.ACTIVE_TOOL_COUNT = 45
    _clear_modules(
        "train",
        "eval",
        "data_synth",
        "encoders",
        "loss",
        "scripts.prepare_legr_30tool_dataset",
    )


def _has_duplicate_tool_labels(tools_cell: object) -> bool:
    tools = [token.strip() for token in str(tools_cell).split(";") if token.strip()]
    return len(tools) != len(set(tools))


def _build_repo_30tool_dataset():
    builder = importlib.import_module("scripts.prepare_legr_30tool_dataset")
    output_dir = ROOT / "upgraded" / "upgraded_30tools"
    builder.build_upgraded_30tool_dataset(
        input_dir=ROOT / "upgraded_data" / "graph_30tools",
        output_dir=output_dir,
        seed=42,
    )
    return builder, output_dir


def test_prepare_legr_30tool_dataset_builds_expected_schema_and_splits():
    builder, output_dir = _build_repo_30tool_dataset()

    split_frames = {
        split_name: pd.read_csv(
            output_dir / f"{split_name}.csv",
            keep_default_na=False,
        )
        for split_name in builder.OUTPUT_SPLITS
    }

    assert list(split_frames["train"].columns) == builder.OUTPUT_COLUMNS

    dag_sets = {
        split_name: set(df["dag_id"].astype(int).unique().tolist())
        for split_name, df in split_frames.items()
    }
    assert len(dag_sets["train"]) == 138
    assert len(dag_sets["dev"]) == 30
    assert len(dag_sets["test_topology_heldout"]) == 30
    assert not (dag_sets["train"] & dag_sets["dev"])
    assert not (dag_sets["train"] & dag_sets["test_topology_heldout"])
    assert not (dag_sets["dev"] & dag_sets["test_topology_heldout"])

    family_sets = {
        split_name: set(df["topo_family"].tolist())
        for split_name, df in split_frames.items()
    }
    assert family_sets["train"] & family_sets["dev"]
    assert family_sets["train"] & family_sets["test_topology_heldout"]
    assert family_sets["dev"] & family_sets["test_topology_heldout"]

    saw_duplicate_rows = False
    for split_name, df in split_frames.items():
        assert set(df["split"].tolist()) == {split_name}
        assert set(df["source"].tolist()) == {"generated", "original"}
        assert (df["tools"] == df["original_tools"]).all()

        duplicate_mask = df["tools"].map(_has_duplicate_tool_labels)
        saw_duplicate_rows = saw_duplicate_rows or bool(duplicate_mask.any())
        expected_flags = duplicate_mask.map(lambda flag: "TRUE" if flag else "FALSE")

        assert (
            df["had_duplicate_node_labels"].astype(str).str.upper() == expected_flags
        ).all()
        assert (
            df["strict_fix_applied"].astype(str).str.upper() == expected_flags
        ).all()

    assert saw_duplicate_rows

    raw_hard_negatives = pd.read_csv(
        ROOT / "upgraded_data" / "graph_30tools" / "hard_negatives.csv"
    )
    built_hard_negatives = pd.read_csv(output_dir / "hard_negatives.csv")
    pd.testing.assert_frame_equal(built_hard_negatives, raw_hard_negatives)


def test_train_defaults_to_upgraded_30tool_dataset_when_tool_count_30():
    _, output_dir = _build_repo_30tool_dataset()
    train = _fresh_import("train", ["train.py", "--tool_count", "30"])

    cfg = train.TrainConfig(tool_count=30)
    train._resolve_cfg_tool_count(cfg)
    train_csv, val_csv = train._resolve_train_val_csv_paths(cfg)

    assert ROOT / Path(train_csv) == output_dir / "train.csv"
    assert ROOT / Path(val_csv) == output_dir / "dev.csv"


def test_eval_defaults_to_upgraded_30tool_dataset_when_tool_count_30():
    _, output_dir = _build_repo_30tool_dataset()
    eval_module = _fresh_import(
        "eval",
        ["eval.py", "--tool_count", "30", "--checkpoint", "dummy.pt"],
    )

    dataset_csv, hard_negative_csv = eval_module._resolve_eval_csv_paths(
        None,
        None,
        tool_count=30,
    )

    assert ROOT / Path(dataset_csv) == output_dir / "test_topology_heldout.csv"
    assert ROOT / Path(hard_negative_csv) == output_dir / "hard_negatives.csv"


def test_explicit_csv_overrides_still_win_for_30tool_defaults():
    _, output_dir = _build_repo_30tool_dataset()
    train = _fresh_import("train", ["train.py", "--tool_count", "30"])
    eval_module = _fresh_import(
        "eval",
        ["eval.py", "--tool_count", "30", "--checkpoint", "dummy.pt"],
    )

    cfg = train.TrainConfig(
        tool_count=30,
        train_csv=str(output_dir / "train.csv"),
        val_csv=str(output_dir / "dev.csv"),
    )
    train._resolve_cfg_tool_count(cfg)
    assert train._resolve_train_val_csv_paths(cfg) == (
        str(output_dir / "train.csv"),
        str(output_dir / "dev.csv"),
    )

    dataset_csv, hard_negative_csv = eval_module._resolve_eval_csv_paths(
        str(output_dir / "test_topology_heldout.csv"),
        str(output_dir / "hard_negatives.csv"),
        tool_count=30,
    )
    assert Path(dataset_csv) == output_dir / "test_topology_heldout.csv"
    assert Path(hard_negative_csv) == output_dir / "hard_negatives.csv"


def test_15tool_and_45tool_defaults_remain_unchanged():
    train = _fresh_import("train", ["train.py", "--tool_count", "15"])
    cfg = train.TrainConfig(tool_count=15)
    train._resolve_cfg_tool_count(cfg)

    assert train._default_train_val_csv_paths(15) == (None, None)
    assert train._resolve_train_val_csv_paths(cfg) == (None, None)

    eval_module = _fresh_import(
        "eval",
        ["eval.py", "--tool_count", "45", "--checkpoint", "dummy.pt"],
    )
    assert eval_module._default_eval_csv_paths(45) == (None, None)
    assert eval_module._resolve_eval_csv_paths(
        None,
        None,
        tool_count=45,
    ) == (None, None)
