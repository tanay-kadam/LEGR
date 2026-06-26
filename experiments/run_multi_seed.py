"""
run_multi_seed.py — Multi-Seed Training & Evaluation
=======================================================

Runs LEGR training and evaluation across multiple random seeds to compute
mean ± std metrics for statistical significance reporting.

Supports:
    - Sequential or parallel (multi-worker) runs
    - Train+eval mode or eval-only mode (reuse existing checkpoints)
    - Aggregation of per-seed metrics into mean ± std

Results are saved to ``experiments/multi_seed_results.json``.

Usage
-----
    $ python experiments/run_multi_seed.py --seeds 42 43 44 45 46
    $ python experiments/run_multi_seed.py --seeds 42 43 44 45 46 --n_workers 5
    $ python experiments/run_multi_seed.py --mode eval_only \\
          --checkpoint_dir checkpoints --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legr_tool_count import add_tool_count_argument, apply_tool_count_override


def _train_and_eval_single_seed(
    seed: int,
    checkpoint_dir: str,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    hard_negative_csv: Optional[str] = None,
    graph_encoder_type: str = "gcn",
    epochs: int = 20,
    tool_count: Optional[int] = None,
) -> Dict:
    """Train and evaluate for a single seed."""
    apply_tool_count_override(tool_count)
    from train import TrainConfig, main as train_main
    from eval import evaluate

    seed_dir = Path(checkpoint_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        seed=seed,
        checkpoint_dir=str(seed_dir),
        graph_encoder_type=graph_encoder_type,
        epochs=epochs,
        tool_count=tool_count,
    )
    if train_csv:
        cfg.train_csv = train_csv
    if val_csv:
        cfg.val_csv = val_csv

    print(f"\n{'='*60}")
    print(f"  Seed {seed}: Training...")
    print(f"{'='*60}")

    best_path = train_main(cfg)

    print(f"\n  Seed {seed}: Evaluating...")
    results = evaluate(
        checkpoint_path=best_path,
        dataset_csv=test_csv,
        hard_negative_csv=hard_negative_csv,
        seed=seed,
    )

    return {
        "seed": seed,
        "checkpoint": best_path,
        "metrics": results.get("LEGR", {}),
        "baselines": {
            k: v for k, v in results.items() if k != "LEGR"
        },
    }


def _eval_only_single_seed(
    seed: int,
    checkpoint_dir: str,
    test_csv: Optional[str] = None,
    hard_negative_csv: Optional[str] = None,
    tool_count: Optional[int] = None,
) -> Dict:
    """Evaluate an existing checkpoint for a single seed."""
    apply_tool_count_override(tool_count)
    from eval import evaluate

    ckpt_path = Path(checkpoint_dir) / f"seed_{seed}" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  WARNING: No checkpoint at {ckpt_path}")
        return {"seed": seed, "error": f"No checkpoint at {ckpt_path}"}

    print(f"\n  Seed {seed}: Evaluating {ckpt_path}...")
    results = evaluate(
        checkpoint_path=str(ckpt_path),
        dataset_csv=test_csv,
        hard_negative_csv=hard_negative_csv,
        seed=seed,
    )

    return {
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "metrics": results.get("LEGR", {}),
        "baselines": {
            k: v for k, v in results.items() if k != "LEGR"
        },
    }


def aggregate_results(per_seed: List[Dict]) -> Dict:
    """Compute mean ± std across seeds for each model."""
    valid = [r for r in per_seed if "error" not in r]
    if not valid:
        return {}

    aggregated = {}

    # LEGR metrics
    legr_keys = set()
    for r in valid:
        legr_keys.update(r.get("metrics", {}).keys())

    legr_agg = {}
    for key in sorted(legr_keys):
        vals = [r["metrics"][key] for r in valid if key in r.get("metrics", {})]
        if vals and isinstance(vals[0], (int, float)):
            legr_agg[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n": len(vals),
            }
    aggregated["LEGR"] = legr_agg

    # Baseline metrics
    baseline_names = set()
    for r in valid:
        baseline_names.update(r.get("baselines", {}).keys())

    for bname in sorted(baseline_names):
        bkeys = set()
        for r in valid:
            bkeys.update(r.get("baselines", {}).get(bname, {}).keys())

        bagg = {}
        for key in sorted(bkeys):
            vals = [
                r["baselines"][bname][key]
                for r in valid
                if bname in r.get("baselines", {}) and key in r["baselines"].get(bname, {})
            ]
            if vals and isinstance(vals[0], (int, float)):
                bagg[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n": len(vals),
                }
        aggregated[bname] = bagg

    return aggregated


def main():
    p = argparse.ArgumentParser(description="Multi-seed LEGR training & evaluation")
    add_tool_count_argument(p)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                    help="List of random seeds")
    p.add_argument("--mode", type=str, default="train_eval",
                    choices=["train_eval", "eval_only"],
                    help="Mode: train_eval or eval_only")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                    help="Base directory for checkpoints")
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--hard_negative_csv", type=str, default=None)
    p.add_argument("--graph_encoder_type", type=str, default="gcn",
                    choices=["gcn", "gat"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n_workers", type=int, default=1,
                    help="Number of parallel workers (1 = sequential)")
    p.add_argument("--output", type=str, default="experiments/multi_seed_results.json",
                    help="Output JSON path")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  Multi-Seed Experiment")
    print(f"  Seeds: {args.seeds}")
    print(f"  Mode: {args.mode}")
    print(f"{'='*60}")

    t0 = time.time()
    per_seed_results = []

    if args.n_workers <= 1:
        for seed in args.seeds:
            if args.mode == "train_eval":
                result = _train_and_eval_single_seed(
                    seed=seed,
                    checkpoint_dir=args.checkpoint_dir,
                    train_csv=args.train_csv,
                    val_csv=args.val_csv,
                    test_csv=args.test_csv,
                    hard_negative_csv=args.hard_negative_csv,
                    graph_encoder_type=args.graph_encoder_type,
                    epochs=args.epochs,
                    tool_count=args.tool_count,
                )
            else:
                result = _eval_only_single_seed(
                    seed=seed,
                    checkpoint_dir=args.checkpoint_dir,
                    test_csv=args.test_csv,
                    hard_negative_csv=args.hard_negative_csv,
                    tool_count=args.tool_count,
                )
            per_seed_results.append(result)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {}
            for seed in args.seeds:
                if args.mode == "train_eval":
                    fut = executor.submit(
                        _train_and_eval_single_seed,
                        seed=seed,
                        checkpoint_dir=args.checkpoint_dir,
                        train_csv=args.train_csv,
                        val_csv=args.val_csv,
                        test_csv=args.test_csv,
                        hard_negative_csv=args.hard_negative_csv,
                        graph_encoder_type=args.graph_encoder_type,
                        epochs=args.epochs,
                        tool_count=args.tool_count,
                    )
                else:
                    fut = executor.submit(
                        _eval_only_single_seed,
                        seed=seed,
                        checkpoint_dir=args.checkpoint_dir,
                        test_csv=args.test_csv,
                        hard_negative_csv=args.hard_negative_csv,
                        tool_count=args.tool_count,
                    )
                futures[fut] = seed

            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    result = fut.result()
                    per_seed_results.append(result)
                except Exception as e:
                    print(f"  ERROR: Seed {seed} failed: {e}")
                    per_seed_results.append({"seed": seed, "error": str(e)})

    per_seed_results.sort(key=lambda r: r["seed"])

    aggregated = aggregate_results(per_seed_results)

    output = {
        "seeds": args.seeds,
        "mode": args.mode,
        "tool_count": args.tool_count,
        "per_seed": per_seed_results,
        "aggregated_mean_std": aggregated,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Multi-Seed Experiment Complete ({elapsed:.1f}s)")
    print(f"  Results saved to {out_path}")
    print(f"{'='*60}")

    # Print summary
    print("\n  Aggregated Results (mean ± std):")
    for model_name, metrics in aggregated.items():
        print(f"\n    {model_name}:")
        for metric, stats in metrics.items():
            print(f"      {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    print()


if __name__ == "__main__":
    main()
