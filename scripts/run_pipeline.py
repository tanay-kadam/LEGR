"""
run_pipeline.py -- Orchestrator for the data-upgrade pipeline
=============================================================

Runs all three stages in order:
  Step 0 (optional): Regenerate pipeline input files
  1. Part A: Upgrade routing dataset
  2. Part B: Upgrade graph dataset
  3. Part C: Combined validation & reporting

Usage:
    python scripts/run_pipeline.py                  # stages 1-3 only
    python scripts/run_pipeline.py --regen-inputs   # full end-to-end (0→1→2→3)
    python scripts/run_pipeline.py --routing-only
    python scripts/run_pipeline.py --graph-only
    python scripts/run_pipeline.py --report-only
    python scripts/run_pipeline.py --tool-count 30  # route output to *_30tools/
    python scripts/run_pipeline.py --tool-count 45 --routing-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data-upgrade pipeline")
    parser.add_argument("--tool-count", type=int, default=None,
                        choices=[15, 30, 45],
                        help="Override ACTIVE_TOOL_COUNT and route output to "
                             "upgraded_data/routing_{N}tools/ etc.")
    parser.add_argument("--regen-inputs", action="store_true",
                        help="Step 0: regenerate both pipeline input CSVs before "
                             "running the three upgrade stages")
    parser.add_argument("--routing-only", action="store_true",
                        help="Only run Part A (routing upgrade)")
    parser.add_argument("--graph-only", action="store_true",
                        help="Only run Part B (graph upgrade)")
    parser.add_argument("--report-only", action="store_true",
                        help="Only run Part C (combined report)")
    args = parser.parse_args()

    if args.tool_count is not None:
        import vocab_config
        vocab_config.ACTIVE_TOOL_COUNT = args.tool_count

    routing_dir = None
    graph_dir = None
    report_dir = None
    if args.tool_count is not None:
        tc = args.tool_count
        routing_dir = f"upgraded_data/routing_{tc}tools"
        graph_dir = f"upgraded_data/graph_{tc}tools"
        report_dir = f"upgraded_data"

    run_all = not (args.routing_only or args.graph_only or args.report_only)

    tc_label = f" (tool_count={args.tool_count})" if args.tool_count else ""
    print("\n" + "=" * 60)
    print(f"  Data Upgrade Pipeline{tc_label}")
    print("=" * 60)

    t0 = time.time()

    if args.regen_inputs:
        from scripts.regenerate_inputs import main as regen_main
        regen_main()

    if run_all or args.routing_only:
        from scripts.upgrade_routing import main as routing_main
        routing_main(output_dir_override=routing_dir)

    if run_all or args.graph_only:
        from scripts.upgrade_graph import main as graph_main
        graph_main(output_dir_override=graph_dir)

    if run_all or args.report_only:
        from scripts.validate_and_report import main as report_main
        report_main(
            routing_dir_override=routing_dir,
            graph_dir_override=graph_dir,
            report_dir_override=report_dir,
        )

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
