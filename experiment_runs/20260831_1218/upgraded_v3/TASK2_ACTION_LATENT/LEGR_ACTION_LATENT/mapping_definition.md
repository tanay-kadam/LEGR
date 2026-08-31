"""
action_type_mapping.py — LEGR tool → {read, write, orchestrate}

Deterministic mapping used by the action-type latent-space analysis.

Source of truth for the original 15 LEGR tools: the 15-tool Tool-Bound
taxonomy in ``taxonomies._TOOL_BOUND_BRANCHES_15``, with routing names
rewritten to LEGR names (``query_database`` → ``db_read``,
``update_database`` → ``db_write``).

Remaining tools (16–45) are assigned from the 45-tool Tool-Bound branches
in ``taxonomies._TOOL_BOUND_BRANCHES_45``:

* Data Retrieval & Monitoring → read
* State Modification & Provisioning → write
* Communication & Orchestration → orchestrate
* Infrastructure Lifecycle → write, except response-coordination tools
  (``acknowledge_alert``, ``trigger_failover``) → orchestrate
* Access Control & Configuration → write

Where a 15-tool label and a 45-tool branch disagree (e.g. ``process_refund``),
the 15-tool label wins.

Unmapped tool names raise ``KeyError``. They are never silently labelled mixed.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

ACTION_READ = "read"
ACTION_WRITE = "write"
ACTION_ORCHESTRATE = "orchestrate"
VALID_ACTIONS = (ACTION_READ, ACTION_WRITE, ACTION_ORCHESTRATE)

GROUP_MOSTLY_READ = "mostly-read"
GROUP_MOSTLY_WRITE = "mostly-write"
GROUP_MOSTLY_ORCHESTRATE = "mostly-orchestrate"
GROUP_MIXED = "mixed"
VALID_GROUPS = (
    GROUP_MOSTLY_READ,
    GROUP_MOSTLY_WRITE,
    GROUP_MOSTLY_ORCHESTRATE,
    GROUP_MIXED,
)

_ACTION_TO_GROUP = {
    ACTION_READ: GROUP_MOSTLY_READ,
    ACTION_WRITE: GROUP_MOSTLY_WRITE,
    ACTION_ORCHESTRATE: GROUP_MOSTLY_ORCHESTRATE,
}

# Complete 45-tool LEGR vocabulary. Keys must match data_synth.TOOL_VOCAB.
TOOL_ACTION_TYPE: dict[str, str] = {
    # --- 15-tool Tool-Bound (LEGR names) ---
    "db_read": ACTION_READ,
    "check_status": ACTION_READ,
    "scan_malware": ACTION_READ,
    "generate_report": ACTION_READ,
    "log_audit_event": ACTION_READ,
    "db_write": ACTION_WRITE,
    "process_refund": ACTION_WRITE,
    "provision_vm": ACTION_WRITE,
    "reset_password": ACTION_WRITE,
    "update_subscription": ACTION_WRITE,
    "create_ticket": ACTION_ORCHESTRATE,
    "escalate_to_human": ACTION_ORCHESTRATE,
    "restart_service": ACTION_ORCHESTRATE,
    "send_notification": ACTION_ORCHESTRATE,
    "quarantine_system": ACTION_ORCHESTRATE,
    # --- 45-tool Retrieval leftovers → read ---
    "run_load_test": ACTION_READ,
    "backup_database": ACTION_READ,
    "snapshot_vm": ACTION_READ,
    "create_alert_rule": ACTION_READ,
    "export_data": ACTION_READ,
    # --- State modification / access / lifecycle writes ---
    "deploy_container": ACTION_WRITE,
    "scale_service": ACTION_WRITE,
    "enable_feature_flag": ACTION_WRITE,
    "migrate_database": ACTION_WRITE,
    "invalidate_cache": ACTION_WRITE,
    "merge_accounts": ACTION_WRITE,
    "restore_backup": ACTION_WRITE,
    "renew_certificate": ACTION_WRITE,
    "rotate_api_key": ACTION_WRITE,
    "archive_data": ACTION_WRITE,
    "disable_feature_flag": ACTION_WRITE,
    "create_dns_record": ACTION_WRITE,
    "approve_access": ACTION_WRITE,
    "revoke_access": ACTION_WRITE,
    "assign_role": ACTION_WRITE,
    "remove_role": ACTION_WRITE,
    "block_ip_address": ACTION_WRITE,
    "unblock_ip_address": ACTION_WRITE,
    "tag_resource": ACTION_WRITE,
    # --- Orchestration leftovers ---
    "run_pipeline": ACTION_ORCHESTRATE,
    "rollback_deployment": ACTION_ORCHESTRATE,
    "schedule_maintenance": ACTION_ORCHESTRATE,
    "transfer_ownership": ACTION_ORCHESTRATE,
    "trigger_failover": ACTION_ORCHESTRATE,
    "acknowledge_alert": ACTION_ORCHESTRATE,
}


def action_type_of(tool: str, mapping: Mapping[str, str] | None = None) -> str:
    """Return the action type of a LEGR tool name. Raises KeyError if unknown."""
    table = mapping if mapping is not None else TOOL_ACTION_TYPE
    if tool not in table:
        raise KeyError(f"No action-type mapping for tool {tool!r}")
    action = table[tool]
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action type {action!r} for tool {tool!r}")
    return action


def action_counts(
    tools: Iterable[str],
    mapping: Mapping[str, str] | None = None,
) -> Counter:
    """Count action types over a DAG's tools. Unknown tools raise KeyError."""
    counts: Counter = Counter()
    for tool in tools:
        counts[action_type_of(tool, mapping)] += 1
    return counts


def classify_dag_action_group(
    tools: Iterable[str],
    mapping: Mapping[str, str] | None = None,
    majority_threshold: float = 0.5,
) -> str:
    """Label a DAG by dominant action composition.

    ``mostly-*`` if one action type is a *strict* majority (``count / n > threshold``).
    Otherwise ``mixed`` (including empty tool lists and exact 50/50 splits).
    """
    tool_list = list(tools)
    if not tool_list:
        return GROUP_MIXED
    counts = action_counts(tool_list, mapping)
    n = len(tool_list)
    top_action, top_count = counts.most_common(1)[0]
    if top_count / n > majority_threshold:
        return _ACTION_TO_GROUP[top_action]
    return GROUP_MIXED


def mapping_covers(tools: Iterable[str], mapping: Mapping[str, str] | None = None) -> list[str]:
    """Return tool names missing from the mapping (empty list if complete)."""
    table = mapping if mapping is not None else TOOL_ACTION_TYPE
    return sorted({t for t in tools if t not in table})
