"""
taxonomies.py -- Tier-Aware Semantic vs Tool-Bound Taxonomies
=============================================================

Defines two hierarchical classification trees over the same 45 leaf-level
API tools. The taxonomies differ only in how they group tools:

    - Semantic Taxonomy: grouped by human-facing domain area
    - Tool-Bound Taxonomy: grouped by requested operation type

The 15-tool benchmark is a special case: it uses the historical 3-branch
layout that produced the original results. The 30-tool tier uses a separate
routing-only benchmark vocabulary, while the 45-tool tier keeps the balanced
full-size 5-branch layout.
"""

from __future__ import annotations

from typing import Dict, List

from routing_benchmark_specs import (
    ROUTING_30_SEMANTIC_BRANCHES,
    ROUTING_30_TOOL_BOUND_BRANCHES,
    ROUTING_30_TOOL_DESCRIPTIONS,
)
from routing_tiers import resolve_explicit_routing_tool_names
from vocab_config import ACTIVE_TOOL_COUNT as _ACTIVE_TOOL_COUNT


# ---------------------------------------------------------------------------
# Shared tool descriptions
# ---------------------------------------------------------------------------

_ALL_TOOL_DESCRIPTIONS: Dict[str, str] = {
    # Tier 1 (tools 1-15)
    "check_status": "Read the current status, health, or progress of a system, service, or request",
    "provision_vm": "Create and allocate a new virtual machine or instance, not just restart or isolate an existing one",
    "create_dns_record": "Create or update a DNS record",
    "scan_malware": "Run a security scan to detect malware or vulnerabilities",
    "restore_backup": "Restore a system from backup",
    "log_audit_event": "Record an event in an audit or compliance log without editing database records or opening a support work item",
    "update_database": "Write, modify, or persist records in a database",
    "approve_access": "Approve an access or permission request",
    "revoke_access": "Revoke user or service access",
    "acknowledge_alert": "Acknowledge a triggered alert",
    "deploy_container": "Deploy a containerised application",
    "run_pipeline": "Trigger a CI/CD pipeline run",
    "query_database": "Read or look up existing records in a database without changing them",
    "send_notification": "Send a message, alert, or notification to a recipient without handing the case off to a human owner",
    "create_ticket": "Open a support or incident ticket for follow-up work instead of directly editing records",
    # Tier 2 (tools 16-30)
    "scale_service": "Scale a service up or down",
    "restart_service": "Stop and start an existing service or application process without provisioning new infrastructure",
    "renew_certificate": "Renew a TLS/SSL certificate",
    "backup_database": "Create a database backup or snapshot",
    "quarantine_system": "Immediately isolate a system from normal network or service access as a containment response, not a database or credential update",
    "rotate_api_key": "Rotate an API key or secret",
    "reset_password": "Reset a user's password or credentials",
    "assign_role": "Assign a role or permission to a user",
    "remove_role": "Remove a role or permission from a user",
    "run_load_test": "Run a load or stress test",
    "rollback_deployment": "Roll back a recent deployment",
    "enable_feature_flag": "Enable a feature flag",
    "generate_report": "Generate a report that summarizes existing data or compliance information",
    "migrate_database": "Migrate a database schema or data",
    "escalate_to_human": "Hand an issue off to a human operator or support agent instead of only sending a message",
    # Tier 3 (tools 31-45)
    "snapshot_vm": "Take a VM snapshot",
    "invalidate_cache": "Invalidate or flush a cache layer",
    "trigger_failover": "Trigger failover to a standby system",
    "archive_data": "Archive old data or records",
    "block_ip_address": "Block an IP address in the firewall",
    "unblock_ip_address": "Unblock a previously blocked IP address",
    "process_refund": "Issue a refund or credit for a payment or charge",
    "update_subscription": "Change a subscription plan, renewal, downgrade, upgrade, or cancellation state",
    "merge_accounts": "Merge two user or customer accounts",
    "disable_feature_flag": "Disable a feature flag",
    "tag_resource": "Tag a cloud resource for tracking",
    "create_alert_rule": "Create a monitoring alert rule",
    "export_data": "Export data for a user or system",
    "schedule_maintenance": "Schedule a maintenance window",
    "transfer_ownership": "Transfer resource ownership",
}

_ALL_TOOL_NAMES = list(_ALL_TOOL_DESCRIPTIONS.keys())


def _resolve_active_tool_names(tool_count: int) -> List[str]:
    """Return the active tool vocabulary for the requested tier."""
    explicit = resolve_explicit_routing_tool_names(tool_count)
    if explicit is not None:
        return explicit
    return _ALL_TOOL_NAMES[:tool_count]


def _active_tool_description(name: str) -> str:
    if _ACTIVE_TOOL_COUNT == 30 and name in ROUTING_30_TOOL_DESCRIPTIONS:
        return ROUTING_30_TOOL_DESCRIPTIONS[name]
    return _ALL_TOOL_DESCRIPTIONS[name]


_ACTIVE_TOOL_NAMES = _resolve_active_tool_names(_ACTIVE_TOOL_COUNT)
_ACTIVE_NAMES = set(_ACTIVE_TOOL_NAMES)
TOOL_DESCRIPTIONS = {name: _active_tool_description(name) for name in _ACTIVE_TOOL_NAMES}
ALL_TOOLS: List[str] = list(TOOL_DESCRIPTIONS.keys())


# ---------------------------------------------------------------------------
# Tier-specific 15-tool taxonomies
# ---------------------------------------------------------------------------

_SEMANTIC_BRANCHES_15: Dict[str, Dict[str, List[str] | str]] = {
    "IT Support": {
        "description": (
            "Requests involving technical support, service access, system "
            "availability, and routine operational help."
        ),
        "tools": [
            "check_status",
            "create_ticket",
            "provision_vm",
            "reset_password",
            "restart_service",
        ],
    },
    "Security & Compliance": {
        "description": (
            "Requests involving security review, incident handling, audit "
            "tracking, and compliance-related oversight."
        ),
        "tools": [
            "escalate_to_human",
            "generate_report",
            "log_audit_event",
            "quarantine_system",
            "scan_malware",
        ],
    },
    "Billing & Data Management": {
        "description": (
            "Requests involving billing actions, customer records, data "
            "management, and account-related communication."
        ),
        "tools": [
            "process_refund",
            "query_database",
            "send_notification",
            "update_database",
            "update_subscription",
        ],
    },
}

_TOOL_BOUND_BRANCHES_15: Dict[str, Dict[str, List[str] | str]] = {
    "Data Retrieval & Monitoring": {
        "description": (
            "Read-oriented, inspection, verification, logging, scanning, and "
            "reporting operations over existing information or system state."
        ),
        "tools": [
            "check_status",
            "generate_report",
            "log_audit_event",
            "query_database",
            "scan_malware",
        ],
    },
    "State Modification & Provisioning": {
        "description": (
            "Create, update, reset, refund, provisioning, and other "
            "state-changing operations affecting systems, accounts, or data."
        ),
        "tools": [
            "process_refund",
            "provision_vm",
            "reset_password",
            "update_database",
            "update_subscription",
        ],
    },
    "Communication & Orchestration": {
        "description": (
            "Notification, escalation, ticketing, restart, coordination, and "
            "response-routing operations across people or services."
        ),
        "tools": [
            "create_ticket",
            "escalate_to_human",
            "restart_service",
            "send_notification",
            "quarantine_system",
        ],
    },
}


# ---------------------------------------------------------------------------
# Explicit 30-tool routing taxonomies
# ---------------------------------------------------------------------------

_SEMANTIC_BRANCHES_30: Dict[str, Dict[str, List[str] | str]] = ROUTING_30_SEMANTIC_BRANCHES

_TOOL_BOUND_BRANCHES_30: Dict[str, Dict[str, List[str] | str]] = ROUTING_30_TOOL_BOUND_BRANCHES


# ---------------------------------------------------------------------------
# Balanced 45-tool taxonomy definitions
# ---------------------------------------------------------------------------

_SEMANTIC_BRANCHES_45: Dict[str, Dict[str, List[str] | str]] = {
    "Infrastructure Operations": {
        "description": (
            "Core infrastructure, runtime operations, and platform-level "
            "system handling."
        ),
        "tools": [
            "check_status",
            "provision_vm",
            "create_dns_record",
            "scale_service",
            "restart_service",
            "renew_certificate",
            "snapshot_vm",
            "trigger_failover",
            "tag_resource",
        ],
    },
    "Security & Data Protection": {
        "description": (
            "Security, recovery, retention, and protective controls for "
            "systems and data."
        ),
        "tools": [
            "scan_malware",
            "restore_backup",
            "log_audit_event",
            "backup_database",
            "quarantine_system",
            "rotate_api_key",
            "archive_data",
            "block_ip_address",
            "unblock_ip_address",
        ],
    },
    "User & Account Management": {
        "description": (
            "Identity, permissions, account records, and customer account "
            "lifecycle changes."
        ),
        "tools": [
            "update_database",
            "approve_access",
            "revoke_access",
            "reset_password",
            "assign_role",
            "remove_role",
            "update_subscription",
            "merge_accounts",
            "transfer_ownership",
        ],
    },
    "Deployment & Observability": {
        "description": (
            "Application delivery, rollout control, and operational "
            "observation of deployed systems."
        ),
        "tools": [
            "acknowledge_alert",
            "deploy_container",
            "run_pipeline",
            "run_load_test",
            "rollback_deployment",
            "enable_feature_flag",
            "invalidate_cache",
            "disable_feature_flag",
            "create_alert_rule",
        ],
    },
    "Communication & Data Services": {
        "description": (
            "Requests centered on data services, reporting, notifications, "
            "support workflows, and service communication."
        ),
        "tools": [
            "query_database",
            "send_notification",
            "create_ticket",
            "generate_report",
            "migrate_database",
            "escalate_to_human",
            "process_refund",
            "export_data",
            "schedule_maintenance",
        ],
    },
}

_TOOL_BOUND_BRANCHES_45: Dict[str, Dict[str, List[str] | str]] = {
    "Data Retrieval & Monitoring": {
        "description": (
            "Read, inspect, measure, or capture state without directly "
            "changing the target system."
        ),
        "tools": [
            "check_status",
            "scan_malware",
            "query_database",
            "run_load_test",
            "generate_report",
            "backup_database",
            "snapshot_vm",
            "create_alert_rule",
            "export_data",
        ],
    },
    "State Modification & Provisioning": {
        "description": (
            "Create, provision, update, or otherwise modify live system or "
            "data state."
        ),
        "tools": [
            "provision_vm",
            "update_database",
            "deploy_container",
            "scale_service",
            "enable_feature_flag",
            "migrate_database",
            "invalidate_cache",
            "update_subscription",
            "merge_accounts",
        ],
    },
    "Communication & Orchestration": {
        "description": (
            "Coordinate work, trigger workflows, notify stakeholders, or "
            "route work to another actor."
        ),
        "tools": [
            "run_pipeline",
            "send_notification",
            "create_ticket",
            "restart_service",
            "rollback_deployment",
            "escalate_to_human",
            "process_refund",
            "schedule_maintenance",
            "transfer_ownership",
        ],
    },
    "Infrastructure Lifecycle": {
        "description": (
            "Preserve, restore, secure, recover, or retire infrastructure and "
            "system state across its lifecycle."
        ),
        "tools": [
            "restore_backup",
            "log_audit_event",
            "acknowledge_alert",
            "renew_certificate",
            "quarantine_system",
            "rotate_api_key",
            "trigger_failover",
            "archive_data",
            "disable_feature_flag",
        ],
    },
    "Access Control & Configuration": {
        "description": (
            "Control permissions, credentials, network policy, and other "
            "configuration-level settings."
        ),
        "tools": [
            "create_dns_record",
            "approve_access",
            "revoke_access",
            "reset_password",
            "assign_role",
            "remove_role",
            "block_ip_address",
            "unblock_ip_address",
            "tag_resource",
        ],
    },
}


def _prune_branches(
    branches: Dict[str, Dict[str, List[str] | str]],
) -> Dict[str, Dict[str, List[str] | str]]:
    """Return a copy of branches filtered to the active tool set."""
    pruned: Dict[str, Dict[str, List[str] | str]] = {}
    for branch_name, branch_info in branches.items():
        active_tools = [tool for tool in branch_info["tools"] if tool in _ACTIVE_NAMES]
        if not active_tools:
            continue
        pruned[branch_name] = {
            "description": branch_info["description"],
            "tools": active_tools,
        }
    return pruned


def _build_semantic_taxonomy() -> Dict[str, Dict[str, List[str] | str]]:
    if _ACTIVE_TOOL_COUNT == 15:
        return _prune_branches(_SEMANTIC_BRANCHES_15)
    if _ACTIVE_TOOL_COUNT == 30:
        return _prune_branches(_SEMANTIC_BRANCHES_30)
    return _prune_branches(_SEMANTIC_BRANCHES_45)


def _build_tool_bound_taxonomy() -> Dict[str, Dict[str, List[str] | str]]:
    if _ACTIVE_TOOL_COUNT == 15:
        return _prune_branches(_TOOL_BOUND_BRANCHES_15)
    if _ACTIVE_TOOL_COUNT == 30:
        return _prune_branches(_TOOL_BOUND_BRANCHES_30)
    return _prune_branches(_TOOL_BOUND_BRANCHES_45)


SEMANTIC_TAXONOMY: Dict[str, Dict[str, List[str] | str]] = {
    "name": "Semantic Taxonomy (Baseline)",
    "branches": _build_semantic_taxonomy(),
}

TOOL_BOUND_TAXONOMY: Dict[str, Dict[str, List[str] | str]] = {
    "name": "Tool-Bound Taxonomy (Proposed)",
    "branches": _build_tool_bound_taxonomy(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_taxonomy(name: str) -> dict:
    """Return a taxonomy dict by name fragment (case-insensitive)."""
    key = name.lower()
    if "semantic" in key or "baseline" in key:
        return SEMANTIC_TAXONOMY
    if "tool" in key or "bound" in key or "proposed" in key:
        return TOOL_BOUND_TAXONOMY
    raise ValueError(f"Unknown taxonomy: {name}")


def get_branch_for_tool(taxonomy: dict, tool: str) -> str | None:
    """Return the branch name that contains tool, or None."""
    for branch_name, branch_info in taxonomy["branches"].items():
        if tool in branch_info["tools"]:
            return branch_name
    return None


def get_tools_for_branch(taxonomy: dict, branch: str) -> List[str]:
    """Return the list of tools under branch."""
    info = taxonomy["branches"].get(branch)
    if info is None:
        return []
    return list(info["tools"])


def format_taxonomy_prompt(taxonomy: dict) -> str:
    """Format a taxonomy into a text block suitable for LLM prompts."""
    lines = []
    for branch_name, branch_info in taxonomy["branches"].items():
        lines.append(f"[{branch_name}] {branch_info['description']}")
        for tool in branch_info["tools"]:
            desc = TOOL_DESCRIPTIONS.get(tool, "")
            lines.append(f"  - {tool}: {desc}")
    return "\n".join(lines)
