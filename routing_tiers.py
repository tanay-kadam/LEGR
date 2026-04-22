"""
routing_tiers.py -- Shared explicit routing vocabularies
=======================================================

Central source of truth for the single-tool routing tiers that intentionally
depart from the default prefix slice of the full 45-tool vocabulary.

The 15-tool tier is frozen to the original benchmark vocabulary.
The 30-tool tier is a routing-only benchmark with its own vocabulary and
taxonomy pair, separate from the 45-tool DAG pipeline vocabulary.
"""

from __future__ import annotations

from routing_benchmark_specs import ROUTING_30_TOOL_NAMES


EXPLICIT_ROUTING_TOOL_NAMES_15: list[str] = [
    "check_status",
    "create_ticket",
    "escalate_to_human",
    "generate_report",
    "log_audit_event",
    "process_refund",
    "provision_vm",
    "quarantine_system",
    "query_database",
    "reset_password",
    "restart_service",
    "scan_malware",
    "send_notification",
    "update_database",
    "update_subscription",
]

EXPLICIT_ROUTING_TOOL_NAMES_30: list[str] = list(ROUTING_30_TOOL_NAMES)


def resolve_explicit_routing_tool_names(tool_count: int) -> list[str] | None:
    """Return the explicit routing vocabulary for supported tiers, else None."""
    if tool_count == 15:
        return list(EXPLICIT_ROUTING_TOOL_NAMES_15)
    if tool_count == 30:
        return list(EXPLICIT_ROUTING_TOOL_NAMES_30)
    return None
