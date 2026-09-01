"""
run_campaign_v4.py — Campaign v4 Training & Evaluation Runner
==============================================================

Orchestrates the complete LEGR Campaign v4 experiment pipeline:
  - LEGR training (legacy + corrected architecture)
  - SBERT baseline
  - BM25 baseline
  - Generative LLM baselines (Llama, GPT-OSS)

Usage:
  python scripts/run_campaign_v4.py --tier 15 --seed 42
  python scripts/run_campaign_v4.py --tier 30 --seed 42 --skip-llm
  python scripts/run_campaign_v4.py --tier 15 --pilot  # quick check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, ".")

CAMPAIGN_DIR = Path("data/campaign_v4")
RESULTS_DIR = Path("artifacts/campaign_v4/results")


@dataclass
class RunConfig:
    tier: int = 15
    seed: int = 42
    epochs: int = 100
    batch_size: int = 128
    pilot: bool = False
    skip_llm: bool = False
    device: str = "cuda"


def _campaign_csv_paths(tier: int) -> Dict[str, str]:
    tier_dir = CAMPAIGN_DIR / f"campaign_v4_{tier}tools"
    return {
        "train": str(tier_dir / "train.csv"),
        "val": str(tier_dir / "dev.csv"),
        "test_indomain": str(tier_dir / "test_indomain.csv"),
        "test_heldout": str(tier_dir / "test_topology_heldout.csv"),
        "candidate": str(tier_dir / "candidate_corpus.csv"),
    }


def _run_cmd(cmd: List[str], label: str, cwd: str = ".") -> Dict:
    """Run a subprocess and capture result."""
    print(f"\n  [{label}] Running: {' '.join(cmd[:6])}...")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=7200,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0
        if not success:
            print(f"    FAILED (exit={result.returncode}, {elapsed:.1f}s)")
            if result.stderr:
                print(f"    stderr: {result.stderr[:500]}")
        else:
            print(f"    OK ({elapsed:.1f}s)")
        return {
            "label": label,
            "success": success,
            "elapsed_s": round(elapsed, 1),
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"label": label, "success": False, "error": "timeout"}
    except Exception as e:
        return {"label": label, "success": False, "error": str(e)}


def run_legr_training(
    cfg: RunConfig,
    architecture: str,
    lambda_ged: float,
    run_name: str,
) -> Dict:
    """Run a single LEGR training experiment."""
    paths = _campaign_csv_paths(cfg.tier)
    ckpt_dir = RESULTS_DIR / f"{run_name}_s{cfg.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from src.data.tool_registry import get_tools
    num_tools = len(get_tools(cfg.tier))

    cmd = [
        sys.executable, "src/train.py",
        "--train_csv", paths["train"],
        "--val_csv", paths["val"],
        "--tool_count", str(cfg.tier),
        "--seed", str(cfg.seed),
        "--epochs", str(5 if cfg.pilot else cfg.epochs),
        "--batch_size", str(cfg.batch_size),
        "--device", cfg.device,
        "--checkpoint_dir", str(ckpt_dir),
        "--lambda_ged", str(lambda_ged),
        "--wandb_run_name", run_name,
    ]

    if architecture == "setgnn_tied":
        cmd.extend([
            "--graph_direction", "setgnn_tied",
            "--graph_encoder_type", "setgnn_tied",
        ])
    elif architecture == "directed_text":
        cmd.extend([
            "--graph_direction", "directed_text",
            "--graph_encoder_type", "directed_text",
        ])
    elif architecture == "directed":
        cmd.extend(["--graph_direction", "directed", "--graph_encoder_type", "directed"])
    elif architecture == "gcn_undirected":
        cmd.extend(["--graph_direction", "gcn_undirected", "--graph_encoder_type", "gcn"])

    return _run_cmd(cmd, run_name)


def run_sbert_baseline(cfg: RunConfig, variant: str, run_name: str) -> Dict:
    """Run SBERT fine-tuning baseline."""
    paths = _campaign_csv_paths(cfg.tier)
    ckpt_dir = RESULTS_DIR / f"{run_name}_s{cfg.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "src/sbert_ft_baseline.py",
        "--train_csv", paths["train"],
        "--val_csv", paths["val"],
        "--seed", str(cfg.seed),
        "--epochs", str(3 if cfg.pilot else cfg.epochs),
        "--checkpoint_dir", str(ckpt_dir),
        "--device", cfg.device,
    ]

    if "ged" in variant:
        cmd.extend(["--lambda_ged", "0.3"])
    if "tied" in variant:
        cmd.extend(["--tied"])

    return _run_cmd(cmd, run_name)


def generate_experiment_manifest(cfg: RunConfig, run_results: List[Dict]) -> Path:
    """Generate experiment manifest for this tier/seed combination."""
    manifest = {
        "campaign": "campaign_v4",
        "tier": cfg.tier,
        "seed": cfg.seed,
        "device": cfg.device,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "runs": run_results,
    }

    try:
        import torch
        manifest["pytorch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            manifest["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    path = RESULTS_DIR / f"manifest_{cfg.tier}tools_s{cfg.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Campaign v4 experiment runner")
    parser.add_argument("--tier", type=int, default=15, choices=[15, 30, 45])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pilot", action="store_true", help="Quick smoke run (5 epochs)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM baselines")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = RunConfig(
        tier=args.tier,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        pilot=args.pilot,
        skip_llm=args.skip_llm,
        device=args.device,
    )

    paths = _campaign_csv_paths(cfg.tier)
    for name, path in paths.items():
        if not Path(path).exists():
            print(f"  ERROR: Missing {name} CSV: {path}")
            return

    print(f"{'='*60}")
    print(f"  Campaign v4 — Tier {cfg.tier}, Seed {cfg.seed}")
    print(f"  Device: {cfg.device}, Pilot: {cfg.pilot}")
    print(f"{'='*60}")

    run_results = []

    # ─── LEGR Legacy (GCN undirected, integer IDs) ───
    print("\n>>> LEGR Legacy (GCN undirected) <<<")

    r = run_legr_training(
        cfg, "gcn_undirected", lambda_ged=0.0,
        run_name=f"legr_legacy_no_ged_{cfg.tier}t",
    )
    run_results.append(r)

    ged_lambda = {15: 0.10, 30: 0.30, 45: 0.30}[cfg.tier]
    r = run_legr_training(
        cfg, "gcn_undirected", lambda_ged=ged_lambda,
        run_name=f"legr_legacy_ged_{cfg.tier}t",
    )
    run_results.append(r)

    # ─── LEGR Directed (with GED) — uses existing directed encoder ───
    print("\n>>> LEGR Directed <<<")

    r = run_legr_training(
        cfg, "directed", lambda_ged=0.0,
        run_name=f"legr_directed_no_ged_{cfg.tier}t",
    )
    run_results.append(r)

    r = run_legr_training(
        cfg, "directed", lambda_ged=ged_lambda,
        run_name=f"legr_directed_ged_{cfg.tier}t",
    )
    run_results.append(r)

    # ─── LEGR directed + tool-name MiniLM node features (encoders_v2) ───
    print("\n>>> LEGR Directed + tool-name text node features <<<")

    r = run_legr_training(
        cfg, "directed_text", lambda_ged=0.0,
        run_name=f"legr_directed_toolname_no_ged_{cfg.tier}t",
    )
    run_results.append(r)

    r = run_legr_training(
        cfg, "directed_text", lambda_ged=ged_lambda,
        run_name=f"legr_directed_toolname_ged_{cfg.tier}t",
    )
    run_results.append(r)

    # ─── LEGR set-GNN tied MiniLM (new dirs; does not replace toolname) ───
    print("\n>>> LEGR SetGNN tied MiniLM (node-set pool + directed GNN) <<<")

    r = run_legr_training(
        cfg, "setgnn_tied", lambda_ged=0.0,
        run_name=f"legr_setgnn_tied_no_ged_{cfg.tier}t",
    )
    run_results.append(r)

    r = run_legr_training(
        cfg, "setgnn_tied", lambda_ged=ged_lambda,
        run_name=f"legr_setgnn_tied_ged_{cfg.tier}t",
    )
    run_results.append(r)

    # ─── SBERT Baselines ───
    print("\n>>> SBERT Fine-tuned Baselines <<<")

    r = run_sbert_baseline(cfg, "no_ged", f"sbert_ft_no_ged_{cfg.tier}t")
    run_results.append(r)

    r = run_sbert_baseline(cfg, "ged", f"sbert_ft_ged_{cfg.tier}t")
    run_results.append(r)

    # ─── Manifest ───
    manifest = generate_experiment_manifest(cfg, run_results)
    print(f"\n  Manifest: {manifest}")

    succeeded = sum(1 for r in run_results if r.get("success"))
    total = len(run_results)
    print(f"\n{'='*60}")
    print(f"  TIER {cfg.tier}: {succeeded}/{total} experiments completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
