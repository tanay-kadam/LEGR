#!/usr/bin/env python3
"""
Hyperparameter sweep for GED loss knobs without modifying train.py or eval.py.

For each Cartesian product of (lambda_ged, ged_scale, ged_margin), this script:
  1. Runs ``python train.py`` with a dedicated ``--checkpoint_dir``
  2. Runs ``python eval.py`` on ``best_model.pt`` and saves per-trial metrics CSV
  3. Writes a summary table and prints the best setting for your chosen metric

Weights & Biases is disabled for training subprocesses unless you pass ``--wandb``.

Examples
~~~~~~~~
::

    # Small default grid (built-in train/val splits if you omit CSVs)
    python sweep_ged_hyperparams.py

    # Same data you use for training + eval CSV for scoring
    python sweep_ged_hyperparams.py \\
        --train_csv upgraded_data/graph/train.csv \\
        --val_csv upgraded_data/graph/dev.csv \\
        --dataset_csv upgraded_data/graph/test.csv

    # Custom grid, fewer epochs for a smoke search
    python sweep_ged_hyperparams.py --epochs 5 \\
        --lambda_ged_values 0.1,0.15,0.3 --ged_scale_values 2.0,2.5 \\
        --ged_margin_values 0,0.05

    # Optimize mean GED to ground-truth top-1 (lower is better)
    python sweep_ged_hyperparams.py --metric mean_ged_error --maximize false
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
) -> None:
    print("\n  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd), env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {r.returncode}")


def _read_legr_row(eval_csv: Path) -> Dict[str, Any]:
    with open(eval_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("model") == "LEGR":
            return row
    raise ValueError(f"No LEGR row in {eval_csv}")


def _float_metric(row: Dict[str, Any], key: str) -> float:
    v = row.get(key, "")
    if v is None or v == "":
        raise KeyError(f"Metric {key!r} missing in eval row")
    return float(v)


@dataclass
class TrialResult:
    lambda_ged: float
    ged_scale: float
    ged_margin: float
    checkpoint_dir: str
    eval_csv: str
    metrics: Dict[str, Any]
    train_ok: bool
    eval_ok: bool
    error: str = ""


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep GED hyperparameters via train.py + eval.py (unchanged).",
    )
    p.add_argument(
        "--lambda_ged_values",
        type=str,
        default="0.1,0.15,0.2,0.25,0.3,0.35,0.4",
        help="Comma-separated lambda_ged values (must be > 0; 0 is ignored).",
    )
    p.add_argument(
        "--ged_scale_values",
        type=str,
        default="0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0",
        help="Comma-separated ged_scale values.",
    )
    p.add_argument(
        "--ged_margin_values",
        type=str,
        default="0.0,0.05, 0.1, 0.15, 0.2, 0.25, 0.3",
        help="Comma-separated ged_margin values.",
    )
    p.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Passed to train.py (must use with --val_csv).",
    )
    p.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Passed to train.py (must use with --train_csv).",
    )
    p.add_argument(
        "--dataset_csv",
        type=str,
        default=None,
        help="Eval dataset CSV for eval.py (omit to use eval's built-in test split).",
    )
    p.add_argument(
        "--checkpoint_root",
        type=str,
        default="sweep_checkpoints",
        help="Directory under which per-trial checkpoint folders are created.",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Where per-trial eval CSVs and the summary CSV are written.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="mrr@1",
        help="Metric name on the LEGR row of eval CSV (e.g. mrr@1, recall@1, tool_set_f1).",
    )
    p.add_argument(
        "--maximize",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=True,
        help="Maximize metric (true) or minimize (false); use false for mean_ged_error.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="If set, passed to train.py as --epochs (for quick sweeps).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="If set, passed to train.py as --device.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Allow Weights & Biases in training (default: WANDB_MODE=disabled).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the trial grid and exit without training.",
    )
    p.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on first train/eval failure instead of recording the error and continuing.",
    )
    args = p.parse_args()

    lambdas = _parse_float_list(args.lambda_ged_values)
    scales = _parse_float_list(args.ged_scale_values)
    margins = _parse_float_list(args.ged_margin_values)

    n_before = len(lambdas)
    lambdas = [lg for lg in lambdas if lg != 0.0]
    if len(lambdas) < n_before:
        print("Skipping lambda_ged=0 (this sweep only trains with GED enabled).")
    if not lambdas:
        p.error("After removing lambda_ged=0, no values remain; use positive lambda_ged only.")

    if (args.train_csv is None) ^ (args.val_csv is None):
        p.error("Provide both --train_csv and --val_csv, or neither.")

    trials: List[Tuple[float, float, float]] = [
        (lg, gs, gm) for lg in lambdas for gs in scales for gm in margins
    ]
    print(f"Planned trials: {len(trials)} (lambda_ged x ged_scale x ged_margin)")

    if args.dry_run:
        for i, (lg, gs, gm) in enumerate(trials, 1):
            print(f"  {i:3d}. lambda_ged={lg}  ged_scale={gs}  ged_margin={gm}")
        return

    py = sys.executable
    train_py = ROOT / "train.py"
    eval_py = ROOT / "eval.py"
    if not train_py.is_file() or not eval_py.is_file():
        raise FileNotFoundError("train.py and eval.py must live next to this script.")

    ckpt_root = ROOT / args.checkpoint_root
    results_dir = ROOT / args.results_dir
    ckpt_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = results_dir / f"hp_sweep_summary_{stamp}.csv"

    base_env = os.environ.copy()
    if not args.wandb:
        base_env["WANDB_MODE"] = "disabled"

    trial_results: List[TrialResult] = []

    for idx, (lg, gs, gm) in enumerate(trials, 1):
        tag = f"t{idx:03d}_lg{lg}_gs{gs}_gm{gm}"
        tag = tag.replace(".", "p")
        trial_ckpt = ckpt_root / tag
        trial_eval_csv = results_dir / f"hp_sweep_eval_{stamp}_{tag}.csv"

        train_cmd: List[str] = [
            py,
            str(train_py),
            "--checkpoint_dir",
            str(trial_ckpt),
            "--lambda_ged",
            str(lg),
            "--ged_scale",
            str(gs),
            "--ged_margin",
            str(gm),
        ]
        if args.train_csv:
            train_cmd.extend(["--train_csv", args.train_csv, "--val_csv", args.val_csv])
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(args.epochs)])
        if args.device:
            train_cmd.extend(["--device", args.device])

        eval_cmd: List[str] = [
            py,
            str(eval_py),
            "--checkpoint",
            str(trial_ckpt / "best_model.pt"),
            "--save_results",
            str(trial_eval_csv),
        ]
        if args.dataset_csv:
            eval_cmd.extend(["--dataset_csv", args.dataset_csv])

        print(f"\n{'=' * 70}\nTrial {idx}/{len(trials)}  "
              f"lambda_ged={lg}  ged_scale={gs}  ged_margin={gm}\n{'=' * 70}")

        train_ok = False
        eval_ok = False
        err = ""
        metrics: Dict[str, Any] = {}

        try:
            _run(train_cmd, cwd=ROOT, env=base_env)
            train_ok = True
        except RuntimeError as e:
            err = f"train: {e}"
            print(f"ERROR: {err}", flush=True)
            if args.fail_fast:
                raise

        if train_ok:
            if not (trial_ckpt / "best_model.pt").is_file():
                err = "train finished but best_model.pt missing"
                print(f"ERROR: {err}", flush=True)
                if args.fail_fast:
                    raise RuntimeError(err)
            else:
                try:
                    _run(eval_cmd, cwd=ROOT, env=base_env)
                    eval_ok = True
                    metrics = dict(_read_legr_row(trial_eval_csv))
                except Exception as e:
                    err = f"eval: {e}"
                    print(f"ERROR: {err}", flush=True)
                    if args.fail_fast:
                        raise

        trial_results.append(
            TrialResult(
                lambda_ged=lg,
                ged_scale=gs,
                ged_margin=gm,
                checkpoint_dir=str(trial_ckpt),
                eval_csv=str(trial_eval_csv) if eval_ok else "",
                metrics=metrics,
                train_ok=train_ok,
                eval_ok=eval_ok,
                error=err,
            )
        )

    # Summary CSV: hyperparams + LEGR metrics + status
    all_metric_keys: List[str] = []
    for tr in trial_results:
        for k in tr.metrics:
            if k != "model" and k not in all_metric_keys:
                all_metric_keys.append(k)
    all_metric_keys.sort()

    fieldnames = [
        "lambda_ged",
        "ged_scale",
        "ged_margin",
        "train_ok",
        "eval_ok",
        "error",
        "checkpoint_dir",
        "eval_csv",
    ] + all_metric_keys

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for tr in trial_results:
            row = {
                "lambda_ged": tr.lambda_ged,
                "ged_scale": tr.ged_scale,
                "ged_margin": tr.ged_margin,
                "train_ok": tr.train_ok,
                "eval_ok": tr.eval_ok,
                "error": tr.error,
                "checkpoint_dir": tr.checkpoint_dir,
                "eval_csv": tr.eval_csv,
            }
            row.update({k: tr.metrics.get(k, "") for k in all_metric_keys})
            w.writerow(row)

    print(f"\nSummary written to {summary_path}")

    # Best trial among successful evals
    ok = [tr for tr in trial_results if tr.eval_ok and args.metric in tr.metrics]
    if not ok:
        print("No successful trials with metric %r; cannot pick best." % args.metric)
        return

    def score(tr: TrialResult) -> float:
        return _float_metric(tr.metrics, args.metric)

    if args.maximize:
        best = max(ok, key=score)
        cmp = "highest"
    else:
        best = min(ok, key=score)
        cmp = "lowest"

    print(
        f"\nBest trial ({cmp} LEGR {args.metric}): "
        f"lambda_ged={best.lambda_ged}  ged_scale={best.ged_scale}  "
        f"ged_margin={best.ged_margin}"
    )
    print(f"  {args.metric} = {score(best)}")
    print(f"  checkpoint_dir: {best.checkpoint_dir}")
    print(f"  eval_csv: {best.eval_csv}")

    print(
        "\nTo reuse the best settings in a normal train run, pass e.g.\n  "
        f"--lambda_ged {best.lambda_ged} --ged_scale {best.ged_scale} "
        f"--ged_margin {best.ged_margin}"
    )


if __name__ == "__main__":
    main()
