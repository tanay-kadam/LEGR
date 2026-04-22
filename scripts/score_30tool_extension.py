"""
Rank candidate tools for the branch-first 30-tool extension.

This helper scores the non-frozen tool pool against the nine extension cells
formed by the new semantic and tool-bound branches. It uses local repo truth:

  - ``taxonomies._ALL_TOOL_DESCRIPTIONS``
  - ``utils.text_utils.CUE_WORDS``
  - ``utils.text_utils.INDIRECT_PHRASINGS``
  - ``utils.text_utils.CONFUSABLE_LABEL_MAP``

Usage:
    python scripts/score_30tool_extension.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from routing_tiers import EXPLICIT_ROUTING_TOOL_NAMES_15
from taxonomies import _ALL_TOOL_DESCRIPTIONS
from utils.text_utils import CONFUSABLE_LABEL_MAP, CUE_WORDS, INDIRECT_PHRASINGS


SEMANTIC_BRANCHES = (
    "Release & Platform Delivery",
    "Identity & Access Administration",
    "Reliability & Continuity Engineering",
)

TOOL_BOUND_BRANCHES = (
    "Observability & Readiness Checks",
    "Configuration & Change Execution",
    "Restriction, Recovery & Reversal",
)

CELL_TARGETS: Dict[Tuple[str, str], int] = {
    ("Release & Platform Delivery", "Observability & Readiness Checks"): 2,
    ("Release & Platform Delivery", "Configuration & Change Execution"): 2,
    ("Release & Platform Delivery", "Restriction, Recovery & Reversal"): 1,
    ("Identity & Access Administration", "Observability & Readiness Checks"): 1,
    ("Identity & Access Administration", "Configuration & Change Execution"): 2,
    ("Identity & Access Administration", "Restriction, Recovery & Reversal"): 2,
    ("Reliability & Continuity Engineering", "Observability & Readiness Checks"): 2,
    ("Reliability & Continuity Engineering", "Configuration & Change Execution"): 1,
    ("Reliability & Continuity Engineering", "Restriction, Recovery & Reversal"): 2,
}

SEMANTIC_PROFILE_WORDS: Dict[str, set[str]] = {
    "Release & Platform Delivery": {
        "application",
        "build",
        "container",
        "delivery",
        "deploy",
        "deployment",
        "dns",
        "exposure",
        "feature",
        "flag",
        "launch",
        "pipeline",
        "platform",
        "production",
        "release",
        "rollout",
        "route",
        "routing",
        "service",
        "traffic",
    },
    "Identity & Access Administration": {
        "access",
        "admin",
        "approval",
        "approve",
        "assign",
        "authorization",
        "authorize",
        "certificate",
        "credential",
        "credentials",
        "identity",
        "ownership",
        "permission",
        "permissions",
        "remove",
        "resource",
        "revoke",
        "role",
        "roles",
        "trust",
        "user",
    },
    "Reliability & Continuity Engineering": {
        "alert",
        "availability",
        "backup",
        "capacity",
        "continuity",
        "disaster",
        "failover",
        "maintenance",
        "preparedness",
        "readiness",
        "recovery",
        "resilience",
        "restore",
        "retention",
        "reliability",
        "service",
        "snapshot",
        "stress",
        "test",
        "uptime",
    },
}

TOOL_BOUND_PROFILE_WORDS: Dict[str, set[str]] = {
    "Observability & Readiness Checks": {
        "acknowledge",
        "backup",
        "benchmark",
        "certificate",
        "check",
        "inspect",
        "inspection",
        "load",
        "monitor",
        "observe",
        "pipeline",
        "preflight",
        "prepare",
        "readiness",
        "review",
        "scan",
        "signal",
        "snapshot",
        "test",
        "validate",
        "verification",
        "verify",
    },
    "Configuration & Change Execution": {
        "activate",
        "activation",
        "approve",
        "assign",
        "change",
        "configure",
        "create",
        "delivery",
        "deploy",
        "enable",
        "execution",
        "expose",
        "grant",
        "launch",
        "route",
        "routing",
        "schedule",
        "ship",
        "tag",
        "transfer",
    },
    "Restriction, Recovery & Reversal": {
        "backup",
        "block",
        "constrain",
        "constraint",
        "disable",
        "failover",
        "isolate",
        "recover",
        "recovery",
        "remove",
        "restore",
        "restrict",
        "restriction",
        "reversal",
        "reverse",
        "revoke",
        "rollback",
        "snapshot",
        "unblock",
        "undo",
    },
}

RECOMMENDED_CELL_ASSIGNMENTS: Dict[Tuple[str, str], List[str]] = {
    ("Release & Platform Delivery", "Observability & Readiness Checks"): [
        "run_pipeline",
        "run_load_test",
    ],
    ("Release & Platform Delivery", "Configuration & Change Execution"): [
        "deploy_container",
        "create_dns_record",
    ],
    ("Release & Platform Delivery", "Restriction, Recovery & Reversal"): [
        "disable_feature_flag",
    ],
    ("Identity & Access Administration", "Observability & Readiness Checks"): [
        "renew_certificate",
    ],
    ("Identity & Access Administration", "Configuration & Change Execution"): [
        "approve_access",
        "assign_role",
    ],
    ("Identity & Access Administration", "Restriction, Recovery & Reversal"): [
        "revoke_access",
        "remove_role",
    ],
    ("Reliability & Continuity Engineering", "Observability & Readiness Checks"): [
        "backup_database",
        "snapshot_vm",
    ],
    ("Reliability & Continuity Engineering", "Configuration & Change Execution"): [
        "schedule_maintenance",
    ],
    ("Reliability & Continuity Engineering", "Restriction, Recovery & Reversal"): [
        "restore_backup",
        "trigger_failover",
    ],
}


def _tokenize(parts: Iterable[str]) -> set[str]:
    words: set[str] = set()
    for part in parts:
        words.update(re.findall(r"[a-z]+", part.lower()))
    return words


def _tool_words(tool: str) -> set[str]:
    parts: List[str] = [tool.replace("_", " "), _ALL_TOOL_DESCRIPTIONS[tool]]
    parts.extend(CUE_WORDS.get(tool, set()))
    parts.extend(INDIRECT_PHRASINGS.get(tool, []))
    return _tokenize(parts)


def _cell_score(tool: str, semantic_branch: str, tool_bound_branch: str) -> tuple[int, int, int, int, int]:
    words = _tool_words(tool)
    semantic_score = len(words & SEMANTIC_PROFILE_WORDS[semantic_branch])
    tool_bound_score = len(words & TOOL_BOUND_PROFILE_WORDS[tool_bound_branch])

    semantic_margin = semantic_score - max(
        len(words & profile)
        for branch_name, profile in SEMANTIC_PROFILE_WORDS.items()
        if branch_name != semantic_branch
    )
    tool_bound_margin = tool_bound_score - max(
        len(words & profile)
        for branch_name, profile in TOOL_BOUND_PROFILE_WORDS.items()
        if branch_name != tool_bound_branch
    )

    frozen_confusable_count = sum(
        1 for label in CONFUSABLE_LABEL_MAP.get(tool, [])
        if label in EXPLICIT_ROUTING_TOOL_NAMES_15
    )
    frozen_cue_overlap = len(
        CUE_WORDS.get(tool, set())
        & set().union(*(CUE_WORDS[name] for name in EXPLICIT_ROUTING_TOOL_NAMES_15))
    )

    total_score = (
        (semantic_score * 3)
        + (tool_bound_score * 3)
        + semantic_margin
        + tool_bound_margin
        - (frozen_confusable_count * 3)
        - frozen_cue_overlap
    )
    return (
        total_score,
        semantic_score,
        tool_bound_score,
        frozen_confusable_count,
        frozen_cue_overlap,
    )


def _candidate_tools() -> List[str]:
    frozen = set(EXPLICIT_ROUTING_TOOL_NAMES_15)
    return [tool for tool in _ALL_TOOL_DESCRIPTIONS if tool not in frozen]


def rank_candidates() -> Dict[Tuple[str, str], List[tuple[str, tuple[int, int, int, int, int]]]]:
    ranked: Dict[Tuple[str, str], List[tuple[str, tuple[int, int, int, int, int]]]] = {}
    for semantic_branch, tool_bound_branch in CELL_TARGETS:
        scored = [
            (tool, _cell_score(tool, semantic_branch, tool_bound_branch))
            for tool in _candidate_tools()
        ]
        ranked[(semantic_branch, tool_bound_branch)] = sorted(
            scored,
            key=lambda item: (
                -item[1][0],
                -item[1][1],
                -item[1][2],
                item[1][3],
                item[1][4],
                item[0],
            ),
        )
    return ranked


def main() -> None:
    ranked = rank_candidates()
    selected_tools = {
        tool
        for tools in RECOMMENDED_CELL_ASSIGNMENTS.values()
        for tool in tools
    }
    print("Branch-first 30-tool extension scoring")
    print("=" * 72)
    print(f"Frozen 15 tools: {len(EXPLICIT_ROUTING_TOOL_NAMES_15)}")
    print(f"Candidate extra tools: {len(_candidate_tools())}")
    print(f"Recommended extra tools: {len(selected_tools)}")
    print("")

    for cell, limit in CELL_TARGETS.items():
        semantic_branch, tool_bound_branch = cell
        print(f"{semantic_branch} x {tool_bound_branch} (need {limit})")
        for tool, metrics in ranked[cell][:8]:
            total, semantic_score, tool_bound_score, frozen_confusable_count, frozen_cue_overlap = metrics
            marker = "  *" if tool in RECOMMENDED_CELL_ASSIGNMENTS[cell] else "   "
            print(
                f"{marker} {tool:<22}"
                f" score={total:>3}"
                f" sem={semantic_score}"
                f" tb={tool_bound_score}"
                f" frozen_conf={frozen_confusable_count}"
                f" frozen_cues={frozen_cue_overlap}"
            )
        print("")


if __name__ == "__main__":
    main()
