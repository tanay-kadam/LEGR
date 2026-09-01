"""
tool_registry.py — Campaign v4 Nested Tool Library
=====================================================

Defines the 15 / 30 / 45-tool nested vocabulary for LEGR Campaign v4.

Design principles:
  - ACTION_FIRST snake_case naming (the tool name IS the description)
  - Nested: tools_15 ⊂ tools_30 ⊂ tools_45
  - Balanced across three functional categories:
      DATA_RETRIEVAL / STATE_MODIFICATION / ORCHESTRATION
  - Read/write sibling pairs for functional-categorization evaluation
  - No separate natural-language descriptions — the name is semantic
"""

from __future__ import annotations

import csv
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List


class FunctionalCategory(str, Enum):
    DATA_RETRIEVAL = "DATA_RETRIEVAL"
    STATE_MODIFICATION = "STATE_MODIFICATION"
    ORCHESTRATION = "ORCHESTRATION"


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 1 — First 15 tools
# ═══════════════════════════════════════════════════════════════════════════

TIER1_DATA_RETRIEVAL = [
    "read_user_profile",
    "read_database_record",
    "read_access_logs",
    "check_service_status",
    "read_subscription_status",
    "scan_system_for_malware",
]

TIER1_STATE_MODIFICATION = [
    "edit_username",
    "write_database_record",
    "reset_user_password",
    "update_subscription_plan",
    "restart_service",
    "create_support_ticket",
]

TIER1_ORCHESTRATION = [
    "dispatch_message_to_usergroup",
    "escalate_case_to_human",
    "route_task_by_condition",
]

# ═══════════════════════════════════════════════════════════════════════════
#  Tier 2 — Add for 30 tools
# ═══════════════════════════════════════════════════════════════════════════

TIER2_DATA_RETRIEVAL = [
    "read_invoice",
    "read_ticket_status",
    "read_vm_status",
    "read_security_policy",
    "read_group_members",
    "read_audit_events",
]

TIER2_STATE_MODIFICATION = [
    "issue_refund",
    "provision_virtual_machine",
    "quarantine_system",
    "update_security_policy",
    "add_user_to_group",
    "append_audit_event",
]

TIER2_ORCHESTRATION = [
    "fanout_tasks_to_workers",
    "merge_parallel_results",
    "retry_failed_task",
]

# ═══════════════════════════════════════════════════════════════════════════
#  Tier 3 — Add for 45 tools
# ═══════════════════════════════════════════════════════════════════════════

TIER3_DATA_RETRIEVAL = [
    "read_payment_status",
    "read_deployment_status",
    "read_backup_status",
    "read_api_key_status",
    "read_incident_timeline",
    "read_usage_report",
]

TIER3_STATE_MODIFICATION = [
    "cancel_payment",
    "deploy_application",
    "create_backup",
    "rotate_api_key",
    "resolve_incident",
    "update_usage_quota",
]

TIER3_ORCHESTRATION = [
    "schedule_dependent_tasks",
    "wait_for_parallel_tasks",
    "dispatch_approval_workflow",
]

# ═══════════════════════════════════════════════════════════════════════════
#  Assembled tool lists
# ═══════════════════════════════════════════════════════════════════════════

TOOLS_15: List[str] = (
    TIER1_DATA_RETRIEVAL + TIER1_STATE_MODIFICATION + TIER1_ORCHESTRATION
)

TOOLS_30: List[str] = (
    TOOLS_15
    + TIER2_DATA_RETRIEVAL + TIER2_STATE_MODIFICATION + TIER2_ORCHESTRATION
)

TOOLS_45: List[str] = (
    TOOLS_30
    + TIER3_DATA_RETRIEVAL + TIER3_STATE_MODIFICATION + TIER3_ORCHESTRATION
)

TOOL_TO_CATEGORY: Dict[str, FunctionalCategory] = {}
for _t in (TIER1_DATA_RETRIEVAL + TIER2_DATA_RETRIEVAL + TIER3_DATA_RETRIEVAL):
    TOOL_TO_CATEGORY[_t] = FunctionalCategory.DATA_RETRIEVAL
for _t in (TIER1_STATE_MODIFICATION + TIER2_STATE_MODIFICATION + TIER3_STATE_MODIFICATION):
    TOOL_TO_CATEGORY[_t] = FunctionalCategory.STATE_MODIFICATION
for _t in (TIER1_ORCHESTRATION + TIER2_ORCHESTRATION + TIER3_ORCHESTRATION):
    TOOL_TO_CATEGORY[_t] = FunctionalCategory.ORCHESTRATION

READ_WRITE_SIBLING_PAIRS: List[tuple[str, str]] = [
    ("read_security_policy", "update_security_policy"),
    ("read_group_members", "add_user_to_group"),
    ("read_audit_events", "append_audit_event"),
    ("read_payment_status", "cancel_payment"),
    ("read_deployment_status", "deploy_application"),
    ("read_backup_status", "create_backup"),
    ("read_api_key_status", "rotate_api_key"),
    ("read_incident_timeline", "resolve_incident"),
]


def get_tools(tier: int) -> List[str]:
    if tier == 15:
        return list(TOOLS_15)
    if tier == 30:
        return list(TOOLS_30)
    if tier == 45:
        return list(TOOLS_45)
    raise ValueError(f"Unsupported tier: {tier}. Use 15, 30, or 45.")


def get_tool_to_index(tier: int) -> Dict[str, int]:
    return {t: i for i, t in enumerate(get_tools(tier))}


def get_category_counts(tier: int) -> Dict[str, int]:
    tools = get_tools(tier)
    counts = {c.value: 0 for c in FunctionalCategory}
    for t in tools:
        counts[TOOL_TO_CATEGORY[t].value] += 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════
#  Validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_registry() -> Dict[str, object]:
    """Run all invariant checks. Returns dict with pass/fail for each."""
    results = {}

    results["tier_sizes"] = (len(TOOLS_15) == 15 and len(TOOLS_30) == 30 and len(TOOLS_45) == 45)

    results["nested_15_in_30"] = set(TOOLS_15) < set(TOOLS_30)
    results["nested_30_in_45"] = set(TOOLS_30) < set(TOOLS_45)

    results["no_duplicates_15"] = len(TOOLS_15) == len(set(TOOLS_15))
    results["no_duplicates_30"] = len(TOOLS_30) == len(set(TOOLS_30))
    results["no_duplicates_45"] = len(TOOLS_45) == len(set(TOOLS_45))

    results["all_categorized"] = all(t in TOOL_TO_CATEGORY for t in TOOLS_45)

    c15 = get_category_counts(15)
    results["balance_15"] = (c15["DATA_RETRIEVAL"] == 6
                             and c15["STATE_MODIFICATION"] == 6
                             and c15["ORCHESTRATION"] == 3)

    c30 = get_category_counts(30)
    results["balance_30"] = (c30["DATA_RETRIEVAL"] == 12
                             and c30["STATE_MODIFICATION"] == 12
                             and c30["ORCHESTRATION"] == 6)

    c45 = get_category_counts(45)
    results["balance_45"] = (c45["DATA_RETRIEVAL"] == 18
                             and c45["STATE_MODIFICATION"] == 18
                             and c45["ORCHESTRATION"] == 9)

    results["action_first_naming"] = all(
        "_" in t and t[0].islower() and t == t.lower()
        for t in TOOLS_45
    )

    for read_tool, write_tool in READ_WRITE_SIBLING_PAIRS:
        key = f"sibling_pair_{read_tool}"
        results[key] = (read_tool in TOOL_TO_CATEGORY and write_tool in TOOL_TO_CATEGORY)

    results["all_passed"] = all(v is True for v in results.values())
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════════════════════

def export_json(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tier in (15, 30, 45):
        tools = get_tools(tier)
        payload = {
            "tier": tier,
            "tool_count": len(tools),
            "tools": tools,
            "categories": {t: TOOL_TO_CATEGORY[t].value for t in tools},
            "category_counts": get_category_counts(tier),
        }
        path = output_dir / f"tools_{tier}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_csv(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "tool_registry.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tool_name", "category", "tier", "index_15", "index_30", "index_45"])
        idx15 = get_tool_to_index(15)
        idx30 = get_tool_to_index(30)
        idx45 = get_tool_to_index(45)
        for i, tool in enumerate(TOOLS_45):
            tier = 1 if tool in set(TOOLS_15) else (2 if tool in set(TOOLS_30) else 3)
            writer.writerow([
                tool,
                TOOL_TO_CATEGORY[tool].value,
                tier,
                idx15.get(tool, ""),
                idx30.get(tool, ""),
                idx45.get(tool, ""),
            ])


if __name__ == "__main__":
    results = validate_registry()
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if results["all_passed"]:
        print("\n  All registry checks passed.")
        export_dir = Path("data/campaign_v4")
        export_json(export_dir)
        export_csv(export_dir)
        print(f"  Exported to {export_dir}")
    else:
        print("\n  REGISTRY VALIDATION FAILED — fix errors before proceeding.")
        raise SystemExit(1)
