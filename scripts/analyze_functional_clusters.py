"""Audit read/edit/orchestrate clustering in the winning LEGR checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legr_experiments.functional_clusters import run_audit  # noqa: E402


DEFAULT_CHECKPOINT = (
    "artifacts/legr_model_search/confirm_r1_15t_s42_30795749e6/best_model.pt"
)
DEFAULT_OUTPUT = "artifacts/legr_model_search/action_cluster_seed42"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default="data/campaign_v4")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--skip-determinism-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(
        root=ROOT,
        checkpoint=_resolve(args.checkpoint),
        data_root=_resolve(args.data_root),
        output_dir=_resolve(args.output),
        device_name=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        verify_determinism=not args.skip_determinism_check,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

