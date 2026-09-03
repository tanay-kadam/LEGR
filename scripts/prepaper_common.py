"""Shared, read-only helpers for the pre-paper LEGR experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GALLERY_SIZES = {15: 322, 30: 455, 45: 650}


def repo_relative(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return value.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_output_dir(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def environment_snapshot(device: torch.device) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        snapshot.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(index),
            "gpu_total_memory_bytes": int(properties.total_memory),
        })
    return snapshot


def campaign_paths(tier: int) -> dict[str, Path]:
    base = ROOT / "data" / "campaign_v4" / f"campaign_v4_{tier}tools"
    return {
        "candidate": base / "candidate_corpus.csv",
        "test": base / "test_topology_heldout.csv",
    }


def full_gallery_frame(tier: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = campaign_paths(tier)
    candidate = pd.read_csv(paths["candidate"])
    test = pd.read_csv(paths["test"])
    gallery = pd.concat(
        [candidate.drop_duplicates("dag_id"), test.drop_duplicates("dag_id")],
        ignore_index=True,
    ).drop_duplicates("dag_id").reset_index(drop=True)
    expected = EXPECTED_GALLERY_SIZES[tier]
    if len(gallery) != expected:
        raise AssertionError(
            f"Campaign V4 {tier}-tool gallery has {len(gallery)} rows; expected {expected}"
        )
    return gallery, test


def build_gallery_dataset(frame: pd.DataFrame):
    """Build the minimal interface required by eval.encode_all_dags."""
    from data_synth import build_dag, dag_canonical_hash
    from train import _parse_edges, _parse_tools

    class GalleryDataset:
        def __init__(self, rows: pd.DataFrame):
            self._dags = []
            self._keys: set[str] = set()
            for _, row in rows.iterrows():
                graph = build_dag(_parse_tools(row["tools"]), _parse_edges(row["edges"]))
                key = dag_canonical_hash(graph)
                if key not in self._keys:
                    self._keys.add(key)
                    self._dags.append(graph)
            self.num_unique_dags = len(self._dags)

        def get_unique_dag(self, index: int):
            return self._dags[index]

    dataset = GalleryDataset(frame)
    if dataset.num_unique_dags != len(frame):
        raise AssertionError(
            f"Gallery canonicalization changed size from {len(frame)} to "
            f"{dataset.num_unique_dags}"
        )
    return dataset


def checkpoint_manifest() -> dict[int, list[dict[str, Any]]]:
    campaign = ROOT / "artifacts" / "campaign_v4" / "results"
    v3_scale = ROOT / "artifacts" / "legr_model_search" / "v3_scale"
    manifest: dict[int, list[dict[str, Any]]] = {}
    for tier in (15, 30, 45):
        v3_no_ged_root = campaign if tier == 15 else v3_scale
        manifest[tier] = [
            {
                "model_id": f"v2_no_ged_{tier}t",
                "architecture": "V2",
                "objective": "InfoNCE",
                "checkpoint": campaign / f"legr_directed_toolname_no_ged_{tier}t_s42" / "best_model.pt",
                "expected_encoder": "directed_text",
                "expected_lambda_ged": 0.0,
            },
            {
                "model_id": f"v2_ged_{tier}t",
                "architecture": "V2",
                "objective": "InfoNCE+GED",
                "checkpoint": campaign / f"legr_directed_toolname_ged_{tier}t_s42" / "best_model.pt",
                "expected_encoder": "directed_text",
                "expected_lambda_ged": None,
            },
            {
                "model_id": f"v3_no_ged_{tier}t",
                "architecture": "V3",
                "objective": "InfoNCE",
                "checkpoint": v3_no_ged_root / f"legr_setgnn_tied_no_ged_{tier}t_s42" / "best_model.pt",
                "expected_encoder": "setgnn_tied",
                "expected_lambda_ged": 0.0,
            },
            {
                "model_id": f"v3_ged_{tier}t",
                "architecture": "V3",
                "objective": "InfoNCE+GED",
                "checkpoint": campaign / f"legr_setgnn_tied_ged_{tier}t_s42" / "best_model.pt",
                "expected_encoder": "setgnn_tied",
                "expected_lambda_ged": None,
            },
        ]
    return manifest


def validate_checkpoint_metadata(entry: dict[str, Any], checkpoint: dict[str, Any], tier: int) -> dict[str, Any]:
    path = Path(entry["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(path)
    config = checkpoint.get("config", {})
    encoder = str(config.get("graph_encoder_type", config.get("graph_direction", "")))
    value = float(config.get("lambda_ged", 0.0))
    if encoder != entry["expected_encoder"]:
        raise AssertionError(f"{path}: encoder={encoder}, expected {entry['expected_encoder']}")
    if int(config.get("tool_count", checkpoint.get("tool_count", -1))) != tier:
        raise AssertionError(f"{path}: checkpoint tool tier does not match {tier}")
    if entry["expected_lambda_ged"] == 0.0 and value != 0.0:
        raise AssertionError(f"{path}: expected lambda_ged=0, found {value}")
    if entry["expected_lambda_ged"] is None and value <= 0.0:
        raise AssertionError(f"{path}: expected positive GED weight, found {value}")
    return {
        "epoch": int(checkpoint.get("epoch", -1)),
        "graph_encoder_type": encoder,
        "lambda_ged": value,
        "configured_tool_count": int(config.get("tool_count", tier)),
    }
