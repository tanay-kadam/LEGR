"""
run_routing_experiments.py — Run routing evaluation across tool counts and models
=================================================================================

Executes main.py for each (model, tool_count, split) combination, saving
results to new_results/{model_slug}_{N}tools/.

Usage:
    python scripts/run_routing_experiments.py                          # 30+45 tools, both models, all splits
    python scripts/run_routing_experiments.py --tool_counts 45         # 45 tools only
    python scripts/run_routing_experiments.py --tool_counts 30 45      # both
    python scripts/run_routing_experiments.py --models llama3.2        # single model
    python scripts/run_routing_experiments.py --splits confusable_intents base_cleaned
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPLITS = [
    "routing_confusable_intents",
    "routing_base_cleaned",
    "routing_lexical_cue_reduced",
    "routing_paraphrase_test",
]

MODEL_SLUG = {
    "gpt-oss:120b-cloud": "gpt-oss",
    "llama3.2": "llama3.2",
}


def results_dir_for(model: str, tool_count: int) -> str:
    slug = MODEL_SLUG.get(model, model.replace(":", "-"))
    return f"new_results/{slug}_{tool_count}tools"


def run_split(preset: str, model: str, tool_count: int, out_dir: str) -> int:
    cmd = [
        sys.executable, str(ROOT / "src" / "main.py"),
        "--tool_count", str(tool_count),
        "--dataset_preset", preset,
        "--results_dir", out_dir,
    ]

    env = os.environ.copy()
    env["USE_OLLAMA"] = "true"
    env["OLLAMA_MODEL"] = model

    print(f"\n{'='*60}")
    print(f"  Split:      {preset}")
    print(f"  Model:      {model}")
    print(f"  Tool count: {tool_count}")
    print(f"  Output:     {out_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0

    tag = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  {preset}: {tag} in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    return result.returncode


def main():
    p = argparse.ArgumentParser(description="Run routing experiments")
    p.add_argument("--models", nargs="+",
                   default=["gpt-oss:120b-cloud", "llama3.2"],
                   help="Ollama model names (default: gpt-oss:120b-cloud llama3.2)")
    p.add_argument("--tool_counts", nargs="+", type=int,
                   default=[30, 45],
                   help="Tool counts to evaluate (default: 30 45)")
    p.add_argument("--splits", nargs="*", default=None,
                   help="Subset of splits to run (default: all 4)")
    args = p.parse_args()

    splits = args.splits or SPLITS
    valid_short = {s.replace("routing_", "") for s in SPLITS}
    resolved = []
    for s in splits:
        if s in SPLITS:
            resolved.append(s)
        elif f"routing_{s}" in SPLITS:
            resolved.append(f"routing_{s}")
        else:
            print(f"ERROR: Unknown split '{s}'. Valid: {valid_short}")
            sys.exit(1)

    combos = [
        (m, tc) for m in args.models for tc in args.tool_counts
    ]

    print(f"\n{'#'*60}")
    print(f"  Routing Experiment Suite")
    print(f"  Models:      {args.models}")
    print(f"  Tool counts: {args.tool_counts}")
    print(f"  Splits:      {len(resolved)}")
    print(f"  Total runs:  {len(combos) * len(resolved)}")
    print(f"{'#'*60}")

    t_total = time.time()
    all_results = {}

    for model, tc in combos:
        out_dir = results_dir_for(model, tc)
        print(f"\n{'#'*60}")
        print(f"  [{model}] {tc} tools  ->  {out_dir}")
        print(f"{'#'*60}")

        for preset in resolved:
            rc = run_split(preset, model, tc, out_dir)
            all_results[(model, tc, preset)] = rc

    elapsed_total = time.time() - t_total

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE ({elapsed_total:.0f}s / {elapsed_total/3600:.1f}h)")
    print(f"{'='*60}")
    for (model, tc, preset), rc in all_results.items():
        tag = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  [{model}] {tc}t {preset}: {tag}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
