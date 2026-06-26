"""Helpers for configuring LEGR tool-count tiers before import-time sizing."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


SUPPORTED_TOOL_COUNTS: tuple[int, ...] = (15, 30, 45)


def get_active_tool_count() -> int:
    """Return the current active LEGR vocabulary size."""
    import vocab_config as vc

    return vc.ACTIVE_TOOL_COUNT


def apply_tool_count_override(tool_count: int | None) -> int:
    """Apply a validated tool-count override to ``vocab_config``."""
    import vocab_config as vc

    if tool_count is None:
        return vc.ACTIVE_TOOL_COUNT

    if tool_count not in SUPPORTED_TOOL_COUNTS:
        raise ValueError(
            f"Unsupported tool_count={tool_count}. "
            f"Expected one of {SUPPORTED_TOOL_COUNTS}."
        )

    vc.ACTIVE_TOOL_COUNT = int(tool_count)
    return vc.ACTIVE_TOOL_COUNT


def bootstrap_tool_count_from_argv(
    argv: Sequence[str] | None = None,
) -> int | None:
    """Best-effort early ``--tool_count`` parsing for import-time modules."""
    argv = list(sys.argv if argv is None else argv)

    for idx, arg in enumerate(argv):
        if arg != "--tool_count":
            continue
        if idx + 1 >= len(argv):
            return None
        try:
            tool_count = int(argv[idx + 1])
        except ValueError:
            return None
        if tool_count in SUPPORTED_TOOL_COUNTS:
            apply_tool_count_override(tool_count)
            return tool_count
        return None

    return None


def add_tool_count_argument(
    parser: argparse.ArgumentParser,
    default: int | None = None,
) -> None:
    """Add the shared LEGR ``--tool_count`` CLI argument."""
    parser.add_argument(
        "--tool_count",
        type=int,
        default=default,
        choices=list(SUPPORTED_TOOL_COUNTS),
        metavar="N",
        help="Number of active LEGR tools (15, 30, or 45).",
    )
