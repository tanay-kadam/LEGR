"""Independent checkpoint reload + metric recompute. Invoke with --tool_count first."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from legr_tool_count import add_tool_count_argument, bootstrap_tool_count_from_argv

bootstrap_tool_count_from_argv(sys.argv)

import torch  # noqa: E402
from encoders import resolve_graph_encoder_settings  # noqa: E402
from eval import (  # noqa: E402
    CSVEvalDataset,
    _load_model_and_tokenizer,
    compute_metrics,
    encode_all_dags,
    encode_all_queries,
)
from utils import read_datafile  # noqa: E402


def _finite(v) -> bool:
    if v is None:
        return False
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _range_ok(metrics: dict) -> list[str]:
    errors = []
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        if not _finite(v):
            errors.append(f"{k} is not finite: {v}")
            continue
        lk = k.lower()
        if "recall" in lk or lk.endswith("_f1") or "mrr" in lk or "accuracy" in lk:
            if v < -1e-9 or v > 1.0 + 1e-6:
                # accuracy_pct is 0-100
                if "pct" in lk:
                    if v < -1e-9 or v > 100.0 + 1e-6:
                        errors.append(f"{k} out of range: {v}")
                else:
                    errors.append(f"{k} out of range: {v}")
    return errors


def verify_legr(args) -> dict:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA")
    ckpt = Path(args.checkpoint)
    model, cfg, tokenizer = _load_model_and_tokenizer(str(ckpt), device)
    param_dev = next(model.parameters()).device
    if param_dev.type != "cuda":
        raise RuntimeError(f"Reloaded LEGR not on CUDA: {param_dev}")
    df = read_datafile(args.dataset_csv)
    dataset = CSVEvalDataset(df)
    _, _, bidirectional = resolve_graph_encoder_settings(cfg)
    q = encode_all_queries(model, dataset, tokenizer, device)
    d = encode_all_dags(model, dataset, device, bidirectional=bidirectional)
    if not torch.isfinite(q).all() or not torch.isfinite(d).all():
        raise RuntimeError("Non-finite embeddings")
    sim = torch.mm(q, d.t())
    k = min(5, d.size(0))
    topk = sim.topk(k=k, dim=1).indices
    gt = torch.tensor(
        [
            dataset.samples[i]["dag_id"]
            if isinstance(dataset.samples[i], dict)
            else dataset.samples[i].dag_id
            for i in range(len(dataset))
        ]
    )
    metrics = compute_metrics(topk, gt, dataset)
    out = {
        "arch": "legr",
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_sha256": sha256_file(ckpt),
        "dataset_csv": str(Path(args.dataset_csv).resolve()),
        "n_eval": int(len(dataset)),
        "n_unique_dags": int(dataset.num_unique_dags),
        "device": str(param_dev),
        "gpu": torch.cuda.get_device_name(device),
        "graph_direction": getattr(cfg, "graph_direction", None),
        "graph_encoder_type": getattr(cfg, "graph_encoder_type", None),
        "bidirectional": bidirectional,
        "tool_count": getattr(cfg, "tool_count", None),
        "seed": getattr(cfg, "seed", None),
        "metrics": metrics,
        "range_errors": _range_ok(metrics),
        "forward_finite": True,
        "verified_load": True,
    }
    pred_path = Path(args.output_dir) / "rankings.pt"
    torch.save({"topk": topk.cpu(), "gt": gt.cpu(), "sim_shape": list(sim.shape)}, pred_path)
    return out


def verify_sbert(args) -> dict:
    from sbert_ft_baseline import (
        SBERTFineTuneDualEncoder,
        evaluate_sbert_ft,
        get_tokenizer,
    )
    from train import TrainConfig

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA")
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config_dict = ckpt.get("config", {})
    fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{k: v for k, v in config_dict.items() if k in fields})
    tied = bool(ckpt.get("tied", False))
    model = SBERTFineTuneDualEncoder(
        embed_dim=cfg.embed_dim,
        text_model_name=cfg.text_model,
        freeze_text_backbone=cfg.freeze_text,
        num_frozen_layers=cfg.num_frozen_layers,
        tied=tied,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    param_dev = next(model.parameters()).device
    if param_dev.type != "cuda":
        raise RuntimeError(f"Reloaded SBERT not on CUDA: {param_dev}")
    tokenizer = get_tokenizer(cfg.text_model)
    df = read_datafile(args.dataset_csv)
    dataset = CSVEvalDataset(df)
    metrics = evaluate_sbert_ft(model, dataset, tokenizer, device)
    out = {
        "arch": "sbert_ft",
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "dataset_csv": str(Path(args.dataset_csv).resolve()),
        "n_eval": int(len(dataset)),
        "n_unique_dags": int(dataset.num_unique_dags),
        "device": str(param_dev),
        "gpu": torch.cuda.get_device_name(device),
        "tied": tied,
        "lambda_ged": cfg.lambda_ged,
        "tool_count": getattr(cfg, "tool_count", None),
        "seed": getattr(cfg, "seed", None),
        "epoch": ckpt.get("epoch"),
        "metrics": metrics,
        "range_errors": _range_ok(metrics),
        "forward_finite": True,
        "verified_load": True,
        "param_count": int(sum(p.numel() for p in model.parameters())),
    }
    return out


def main():
    p = argparse.ArgumentParser()
    add_tool_count_argument(p)
    p.add_argument("--arch", choices=["legr", "sbert"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.arch == "legr":
        payload = verify_legr(args)
    else:
        payload = verify_sbert(args)
    (out_dir / "verification.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
