"""Regression tests for LEGR tool-count tiers."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
        "experiments.run_multi_seed",
    )


def test_data_synth_build_splits_supports_30_tools(monkeypatch):
    vc.ACTIVE_TOOL_COUNT = 30
    _clear_modules("data_synth")
    data_synth = importlib.import_module("data_synth")

    # Keep the regression fast while still exercising build_splits().
    monkeypatch.setattr(data_synth, "WORKFLOW_TEMPLATES", data_synth.WORKFLOW_TEMPLATES[:60])

    train_ds, val_ds, test_ds = data_synth.build_splits(entity_variants=1, seed=42)

    assert data_synth.NUM_TOOLS == 30
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    active_tools = set(data_synth.TOOL_VOCAB)
    for ds in (train_ds, val_ds, test_ds):
        for sample in ds.samples:
            sample_tools = {
                sample.dag_nx.nodes[node]["tool"] for node in sample.dag_nx.nodes()
            }
            assert sample_tools <= active_tools


def test_train_bootstrap_resolves_tool_count_before_data_synth_import():
    vc.ACTIVE_TOOL_COUNT = 45
    train = _fresh_import("train", ["train.py", "--tool_count", "30"])

    assert vc.ACTIVE_TOOL_COUNT == 30
    assert train._TOOL_COUNT_OVERRIDE == 30
    assert train.NUM_TOOLS == 30


def test_eval_bootstrap_resolves_tool_count_before_model_sizing():
    vc.ACTIVE_TOOL_COUNT = 45
    eval_module = _fresh_import(
        "eval",
        ["eval.py", "--tool_count", "30", "--checkpoint", "dummy.pt"],
    )

    assert vc.ACTIVE_TOOL_COUNT == 30
    assert eval_module._TOOL_COUNT_OVERRIDE == 30
    assert eval_module.NUM_TOOLS == 30


class _FakeStateful:
    def __init__(self, payload):
        self.payload = payload

    def state_dict(self):
        return self.payload


def test_train_checkpoint_payload_stores_tool_count():
    train = _fresh_import("train", ["train.py", "--tool_count", "30"])
    cfg = train.TrainConfig(tool_count=30)

    payload = train._build_checkpoint_payload(
        epoch=3,
        model=_FakeStateful({"model": 1}),
        criterion=_FakeStateful({"criterion": 2}),
        optimizer=_FakeStateful({"optimizer": 3}),
        scheduler=_FakeStateful({"scheduler": 4}),
        cfg=cfg,
        val_loss=0.123,
    )

    assert payload["tool_count"] == 30
    assert payload["config"]["tool_count"] == 30


def test_eval_mismatch_error_mentions_required_tool_count():
    eval_module = _fresh_import(
        "eval",
        ["eval.py", "--tool_count", "30", "--checkpoint", "dummy.pt"],
    )

    with pytest.raises(ValueError, match="--tool_count 15"):
        eval_module._validate_checkpoint_tool_count(
            "dummy.pt",
            {"config": {"tool_count": 15}},
        )


def test_eval_accepts_matching_legacy_checkpoint_shape():
    eval_module = _fresh_import(
        "eval",
        ["eval.py", "--tool_count", "15", "--checkpoint", "dummy.pt"],
    )

    ckpt = {
        "model_state": {
            "graph_encoder.tool_embedding.weight": torch.zeros(16, 8),
        }
    }

    assert eval_module._validate_checkpoint_tool_count("dummy.pt", ckpt) == 15


def test_multi_seed_train_eval_passes_tool_count(monkeypatch):
    _clear_modules("experiments.run_multi_seed")
    run_multi_seed = importlib.import_module("experiments.run_multi_seed")

    calls: list[int | None] = []
    monkeypatch.setattr(
        run_multi_seed,
        "apply_tool_count_override",
        lambda tool_count: calls.append(tool_count) or 30,
    )

    fake_train = types.ModuleType("train")

    class FakeTrainConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_train_main(cfg):
        assert cfg.tool_count == 30
        return str(tmp_root / "seed_7" / "best_model.pt")

    fake_train.TrainConfig = FakeTrainConfig
    fake_train.main = fake_train_main

    fake_eval = types.ModuleType("eval")

    def fake_evaluate(**kwargs):
        return {"LEGR": {"recall@1": 1.0}, "BM25": {"recall@1": 0.5}}

    fake_eval.evaluate = fake_evaluate

    monkeypatch.setitem(sys.modules, "train", fake_train)
    monkeypatch.setitem(sys.modules, "eval", fake_eval)
    monkeypatch.setattr(run_multi_seed.Path, "mkdir", lambda self, parents=False, exist_ok=False: None)

    tmp_root = ROOT / "synthetic-seed-root"
    result = run_multi_seed._train_and_eval_single_seed(
        seed=7,
        checkpoint_dir=str(tmp_root),
        tool_count=30,
        epochs=1,
    )

    assert calls == [30]
    assert result["metrics"]["recall@1"] == 1.0


def test_multi_seed_eval_only_passes_tool_count(monkeypatch):
    _clear_modules("experiments.run_multi_seed")
    run_multi_seed = importlib.import_module("experiments.run_multi_seed")

    calls: list[int | None] = []
    monkeypatch.setattr(
        run_multi_seed,
        "apply_tool_count_override",
        lambda tool_count: calls.append(tool_count) or 30,
    )

    fake_eval = types.ModuleType("eval")

    def fake_evaluate(**kwargs):
        return {"LEGR": {"mrr@1": 0.75}, "S-BERT": {"mrr@1": 0.25}}

    fake_eval.evaluate = fake_evaluate
    monkeypatch.setitem(sys.modules, "eval", fake_eval)
    monkeypatch.setattr(run_multi_seed.Path, "exists", lambda self: True)

    result = run_multi_seed._eval_only_single_seed(
        seed=11,
        checkpoint_dir=str(ROOT / "synthetic-seed-root"),
        tool_count=30,
    )

    assert calls == [30]
    assert result["metrics"]["mrr@1"] == 0.75
