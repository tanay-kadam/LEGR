"""
dataset.py -- Canonical Evaluation Datasets
===========================================

Provides:
    - RAW_DATASET: 80 hand-crafted queries spanning all 45 API tools,
      designed for qualitative comparative evaluation.
    - build_dataset(): Returns the canonical dataset as a DataFrame.
    - build_scaled_single_tool_dataset(): Generates a large single-tool dataset
      via data_synth for statistically powerful evaluation.

Usage (CLI)
-----------
    python dataset.py
    python dataset.py scaled 67 results/out.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from routing_benchmark_specs import (
    build_routing_30_canonical_rows,
    build_routing_30_single_tool_dataset,
    matches_routing_30_labels,
)
from routing_tiers import resolve_explicit_routing_tool_names


# ===========================================================================
# Canonical evaluation dataset
# ===========================================================================
# Each tuple: (query, ground_truth_tool, domain_note)

RAW_DATASET: List[tuple] = [
    ("can you pull up Priya's account so I can see what's currently on file",
     "query_database", ""),
    ("can you fix the email address we have listed for Priya's account",
     "update_database", ""),
    ("can you move Priya over to the annual plan instead",
     "update_subscription", ""),
    ("Priya can't get into her account anymore, help her get back in",
     "reset_password", ""),
    ("what's the current state of auth-service-14",
     "check_status", ""),
    ("auth-service-14 is stuck again, can you bounce it",
     "restart_service", ""),
    ("can you bring another machine online for the auth workload",
     "provision_vm", ""),
    ("we should open a case for the auth-service-14 issue",
     "create_ticket", ""),
    ("can you see what happened with transaction 55281",
     "query_database", ""),
    ("the charge on transaction 55281 needs to be reversed",
     "process_refund", ""),
    ("I need the numbers for yesterday's transactions in a format finance can review",
     "generate_report", ""),
    ("let finance know there was a problem with transaction 55281",
     "send_notification", ""),
    ("please check whether node-27 shows signs of anything suspicious",
     "scan_malware", ""),
    ("node-27 should be taken off the network until we understand what's happening",
     "quarantine_system", ""),
    ("make sure what happened on node-27 is recorded for compliance review",
     "log_audit_event", ""),
    ("this node-27 issue needs to be handed to a human responder now",
     "escalate_to_human", ""),
    ("can you look up the current details for Omar's subscription",
     "query_database", ""),
    ("the card on Omar's account is fine, but the plan itself needs to change",
     "update_subscription", ""),
    ("the plan is fine, but Omar's profile details need correcting",
     "update_database", ""),
    ("put together the latest usage numbers for Omar's account",
     "generate_report", ""),
    # Tier 2 tools (16-30)
    ("can you ship the latest build to the staging cluster",
     "deploy_container", ""),
    ("push the new container image to prod for the payments service",
     "deploy_container", ""),
    ("the last release broke everything, take us back to the previous version",
     "rollback_deployment", ""),
    ("revert the deployment on auth-service to what it was yesterday",
     "rollback_deployment", ""),
    ("the API key for the billing service is compromised, cycle it immediately",
     "rotate_api_key", ""),
    ("can you generate a fresh secret for the analytics endpoint",
     "rotate_api_key", ""),
    ("take a snapshot of the production database before we do anything else",
     "backup_database", ""),
    ("make sure we have a backup of the user DB before the migration",
     "backup_database", ""),
    ("the migration failed, can you restore the database from last night's snapshot",
     "restore_backup", ""),
    ("bring the staging DB back from the latest backup",
     "restore_backup", ""),
    ("traffic is spiking, we need more capacity on the API gateway",
     "scale_service", ""),
    ("scale the worker fleet down, we're way over-provisioned right now",
     "scale_service", ""),
    ("kick off the CI/CD pipeline for the frontend repo",
     "run_pipeline", ""),
    ("trigger a build and deploy for the backend service",
     "run_pipeline", ""),
    ("Priya's request for access to the analytics dashboard is pending, please approve it",
     "approve_access", ""),
    ("sign off on Omar's permission request for the staging environment",
     "approve_access", ""),
    ("Omar left the team, revoke his access to everything",
     "revoke_access", ""),
    ("cut off the contractor's permissions to the production systems",
     "revoke_access", ""),
    ("the analytics project is moving to the data team, transfer ownership",
     "transfer_ownership", ""),
    ("hand the billing-service repo over to the platform team",
     "transfer_ownership", ""),
    ("we need to schedule downtime for the payment gateway this weekend",
     "schedule_maintenance", ""),
    ("book a maintenance window for the database cluster on Saturday night",
     "schedule_maintenance", ""),
    ("move all the records older than two years into cold storage",
     "archive_data", ""),
    ("archive the legacy customer data so it stops cluttering the main DB",
     "archive_data", ""),
    ("turn on the new checkout flow for 10% of users",
     "enable_feature_flag", ""),
    ("enable the dark mode feature flag for the beta group",
     "enable_feature_flag", ""),
    ("kill the experimental search flag, it's causing issues",
     "disable_feature_flag", ""),
    ("turn off the new onboarding flow feature toggle",
     "disable_feature_flag", ""),
    ("the CDN is serving stale content, flush the cache",
     "invalidate_cache", ""),
    ("bust the Redis cache on the product catalog service",
     "invalidate_cache", ""),
    # Tier 3 tools (31-45)
    ("set up a DNS entry so the new service is reachable at api.example.com",
     "create_dns_record", ""),
    ("point the staging subdomain to the new load balancer IP",
     "create_dns_record", ""),
    ("the TLS certificate on the payment gateway expires tomorrow, renew it",
     "renew_certificate", ""),
    ("can you refresh the SSL cert for the main website before it lapses",
     "renew_certificate", ""),
    ("we're getting hammered from that IP, block it at the firewall immediately",
     "block_ip_address", ""),
    ("add the attacker's address to the deny list on the edge nodes",
     "block_ip_address", ""),
    ("turns out we accidentally blocked a legitimate partner IP, undo it",
     "unblock_ip_address", ""),
    ("remove the IP we flagged yesterday from the blocklist",
     "unblock_ip_address", ""),
    ("give Priya the editor role on the analytics dashboard",
     "assign_role", ""),
    ("Omar needs the admin permission for the staging environment",
     "assign_role", ""),
    ("strip the admin role from the intern's account",
     "remove_role", ""),
    ("take away Omar's write permissions on the production database",
     "remove_role", ""),
    ("the primary database is unreachable, switch to the standby replica",
     "trigger_failover", ""),
    ("initiate failover for the payment service to the DR site",
     "trigger_failover", ""),
    ("run a stress test against the API gateway before Black Friday",
     "run_load_test", ""),
    ("we need to benchmark the new endpoint under peak traffic conditions",
     "run_load_test", ""),
    ("take a snapshot of the production VM before we start the upgrade",
     "snapshot_vm", ""),
    ("capture the current state of the staging server as a checkpoint",
     "snapshot_vm", ""),
    ("apply the new schema migration to the production database",
     "migrate_database", ""),
    ("run the data migration script on the user-profiles DB",
     "migrate_database", ""),
    ("tag the new EC2 instances with the marketing cost center",
     "tag_resource", ""),
    ("label all the staging resources with the environment tag",
     "tag_resource", ""),
    ("set up an alert that fires when API latency exceeds 500ms",
     "create_alert_rule", ""),
    ("create a monitoring rule for disk usage on the database servers",
     "create_alert_rule", ""),
    ("acknowledge the CPU alert on the API gateway, we're already on it",
     "acknowledge_alert", ""),
    ("mark the disk space warning as seen, it's a known issue",
     "acknowledge_alert", ""),
    ("Priya has two accounts from before the migration, merge them into one",
     "merge_accounts", ""),
    ("combine Omar's old and new customer profiles into a single record",
     "merge_accounts", ""),
    ("Priya requested a full export of all her data under GDPR",
     "export_data", ""),
    ("pull a complete data dump for the finance department's year-end audit",
     "export_data", ""),
]


def build_dataset() -> pd.DataFrame:
    """Return the canonical evaluation dataset as a DataFrame."""
    from taxonomies import ALL_TOOLS
    from vocab_config import ACTIVE_TOOL_COUNT

    explicit = resolve_explicit_routing_tool_names(ACTIVE_TOOL_COUNT)
    if explicit is not None and matches_routing_30_labels(explicit):
        rows = build_routing_30_canonical_rows(seed=42)
        df = pd.DataFrame(rows, columns=["query", "ground_truth", "domain_note"])
        df.index.name = "id"
        return df

    active = set(ALL_TOOLS)
    rows = [r for r in RAW_DATASET if r[1] in active]
    df = pd.DataFrame(rows, columns=["query", "ground_truth", "domain_note"])
    df.index.name = "id"
    return df


# ===========================================================================
# Scaled single-tool dataset (via data_synth)
# ===========================================================================

TOOL_SYNTH_TO_TAXONOMY: Dict[str, str] = {
    "db_read": "query_database",
    "db_write": "update_database",
}

TAXONOMY_TO_TOOL_SYNTH: Dict[str, str] = {
    value: key for key, value in TOOL_SYNTH_TO_TAXONOMY.items()
}


def build_scaled_single_tool_dataset(
    n_per_tool: int = 67,
    seed: int = 42,
    tool_names: List[str] | None = None,
) -> pd.DataFrame:
    """Build a large single-tool dataset for statistically significant evaluation.

    Explicit routing tiers automatically use their shared benchmark-aligned tool
    list instead of the default synth prefix slice. The 30-tool routing tier is
    generated from its routing-only benchmark spec rather than the DAG vocab.
    """
    from vocab_config import ACTIVE_TOOL_COUNT

    selected_tool_names = tool_names
    if selected_tool_names is None:
        selected_tool_names = resolve_explicit_routing_tool_names(ACTIVE_TOOL_COUNT)

    if selected_tool_names is not None and matches_routing_30_labels(selected_tool_names):
        examples = build_routing_30_single_tool_dataset(
            n_per_tool=n_per_tool,
            seed=seed,
            tool_names=selected_tool_names,
        )
        rows = [
            {
                "query": ex["query"],
                "ground_truth": ex["tool"],
                "domain_note": "",
            }
            for ex in examples
        ]
        df = pd.DataFrame(rows)
        df.index.name = "id"
        return df

    from data_synth import build_single_tool_dataset as _build_synth

    synth_tool_names = None
    if selected_tool_names is not None:
        synth_tool_names = [
            TAXONOMY_TO_TOOL_SYNTH.get(tool, tool) for tool in selected_tool_names
        ]

    examples = _build_synth(
        n_per_tool=n_per_tool,
        seed=seed,
        tool_names=synth_tool_names,
    )

    rows = []
    for ex in examples:
        tool = ex["tool"]
        ground_truth = TOOL_SYNTH_TO_TAXONOMY.get(tool, tool)
        rows.append({
            "query": ex["query"],
            "ground_truth": ground_truth,
            "domain_note": "",
        })

    df = pd.DataFrame(rows)
    df.index.name = "id"
    return df


def save_scaled_single_tool_dataset(
    path: str,
    n_per_tool: int = 67,
    seed: int = 42,
) -> None:
    """Build and save the scaled dataset to CSV."""
    df = build_scaled_single_tool_dataset(n_per_tool=n_per_tool, seed=seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    print(f"Saved {len(df)} queries to {path}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "scaled":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 67
        out = sys.argv[3] if len(sys.argv) > 3 else "results/single_tool_dataset_scaled.csv"
        save_scaled_single_tool_dataset(out, n_per_tool=n)
    else:
        df = build_dataset()
        print(df.to_string())
        print(f"\n{len(df)} queries, {df['ground_truth'].nunique()} unique tools")
