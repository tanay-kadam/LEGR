"""Run the isolated LEGR model-only search without mutating legacy assets."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from src.data.tool_registry import get_tools  # noqa: E402
from legr_experiments.config import ExperimentConfig  # noqa: E402
from legr_experiments.integrity import compare, snapshot, write_snapshot  # noqa: E402
from legr_experiments.search import (  # noqa: E402
    SearchController, architecture_configs, backbone_fusion_configs,
    mathematical_configs, reranker_configs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, choices=(15, 30, 45), default=15)
    parser.add_argument("--budget-hours", type=float, default=24.0)
    parser.add_argument("--stage", choices=("smoke", "math", "architecture", "fusion", "reranker", "all"), default="all")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--screen-epochs", type=int, default=15)
    parser.add_argument("--full-epochs", type=int, default=75)
    parser.add_argument("--output-root", default="artifacts/legr_model_search")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    integrity_path = output_root / "immutable_before.json"
    before = snapshot()
    if integrity_path.exists():
        previous = json.loads(integrity_path.read_text(encoding="utf-8"))
        drift = compare(previous, before)
        if drift["changed"] or drift["missing"]:
            raise RuntimeError(f"Immutable inputs changed since campaign start: {drift}")
    else:
        write_snapshot(integrity_path, before)

    vocabulary = list(get_tools(args.tier))
    base = ExperimentConfig(tier=args.tier)
    base.train.epochs = args.screen_epochs
    base.train.patience = min(6, args.screen_epochs)
    controller = SearchController(output_root, vocabulary, args.budget_hours)

    if args.stage == "smoke":
        configs = mathematical_configs(base)[:2]
        for cfg in configs:
            cfg.train.epochs = 1
            cfg.train.patience = 1
            cfg.train.batch_size = 16
            cfg.model.use_reranker = False
        controller.run_stage("smoke", configs, args.max_runs)
    else:
        best = deepcopy(base)
        baseline = deepcopy(base)
        baseline.name = "baseline_semantic"
        baseline.model.fusion_kind = "semantic"
        baseline.model.use_reranker = False
        baseline.train.epochs = 1
        baseline.train.patience = 1
        controller.run_stage("baseline", [baseline])
        if args.stage in ("math", "all"):
            best = controller.run_stage("math", mathematical_configs(best), args.max_runs)
        if args.stage in ("architecture", "all"):
            best = controller.run_stage("architecture", architecture_configs(best), args.max_runs)
        if args.stage in ("fusion", "all"):
            configs, skipped = backbone_fusion_configs(best)
            (output_root / "skipped_backbones.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
            best = controller.run_stage("fusion", configs, args.max_runs)
        if args.stage in ("reranker", "all"):
            best = controller.run_stage("reranker", reranker_configs(best), args.max_runs)

        if args.stage == "all" and controller.remaining_seconds() > 1800:
            finalists = controller.top_configs(4)
            confirmations = []
            for rank, finalist in enumerate(finalists):
                for seed in (42, 123, 2026):
                    cfg = deepcopy(finalist)
                    cfg.name = f"confirm_r{rank + 1}"
                    cfg.train.seed = seed
                    cfg.train.epochs = args.full_epochs
                    cfg.train.patience = 15
                    confirmations.append(cfg)
            controller.run_stage("confirmation", confirmations)

    after = snapshot()
    drift = compare(before, after)
    (output_root / "immutable_after.json").write_text(json.dumps(after, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "integrity_report.json").write_text(json.dumps(drift, indent=2), encoding="utf-8")
    if drift["changed"] or drift["missing"] or drift["added"]:
        raise RuntimeError(f"Immutable input drift detected: {drift}")
    print(json.dumps({
        "status": "complete", "timestamp": datetime.now().isoformat(),
        "runs": len(controller.records), "output_root": str(output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
