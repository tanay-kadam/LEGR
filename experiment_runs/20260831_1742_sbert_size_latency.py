"""Latency + size for one SBERT FT checkpoint. Invoke with --tool_count first."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\tkadam\LEGR")
sys.path.insert(0, str(ROOT / "src"))

from legr_tool_count import add_tool_count_argument, bootstrap_tool_count_from_argv

bootstrap_tool_count_from_argv(sys.argv)

import numpy as np
import torch

from eval import CSVEvalDataset
from sbert_ft_baseline import SBERTFineTuneDualEncoder, get_tokenizer
from train import TrainConfig
from utils import read_datafile


def main():
    p = argparse.ArgumentParser()
    add_tool_count_argument(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--n_samples", type=int, default=100)
    args = p.parse_args()
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
    tokenizer = get_tokenizer(cfg.text_model)
    ds = CSVEvalDataset(read_datafile(args.dataset_csv))
    n = min(args.n_samples, len(ds))
    for i in range(min(10, n)):
        q = ds[i]["query"]
        enc = tokenizer([q], padding=True, truncation=True, max_length=128, return_tensors="pt")
        _ = model.encode_query(enc["input_ids"].to(device), enc["attention_mask"].to(device))
    torch.cuda.synchronize()
    lat = []
    with torch.no_grad():
        for i in range(n):
            q = ds[i]["query"]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            enc = tokenizer([q], padding=True, truncation=True, max_length=128, return_tensors="pt")
            _ = model.encode_query(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
    lat.sort()
    p95 = lat[max(0, int(len(lat) * 0.95) - 1)]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rec = {
        "tools": int(args.tool_count),
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
        "param_count_total": total,
        "param_count_trainable": trainable,
        "tied": tied,
        "n_samples": n,
        "mean_latency_ms": round(float(np.mean(lat)) * 1000, 3),
        "median_latency_ms": round(float(np.median(lat)) * 1000, 3),
        "p95_latency_ms": round(p95 * 1000, 3),
        "protocol": "tokenize + encode_query, warmup 10, cuda.synchronize, 100 queries",
    }
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
