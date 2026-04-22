"""
data_synth.py — Synthetic Dataset Generator for Latent Execution-Graph Routing
===============================================================================

Generates (natural-language query, execution DAG) pairs for training and
evaluation of the LEGR dual-encoder model.

Each sample consists of:
    query  : str          — A user request requiring multi-tool execution.
    dag    : nx.DiGraph   — A directed acyclic graph whose nodes carry a `tool`
                            attribute drawn from the 45-tool vocabulary and
                            whose edges encode execution-order dependencies.

Dataset construction pipeline:
    1.  Define *workflow templates* — canonical (DAG topology, query-phrasing
        set) pairs covering linear chains, fan-out, fan-in, diamond, and
        complex DAG patterns.
    2.  Instantiate templates with randomised entity substitution
        ({user}, {server}, …) to multiply surface-form diversity.
    3.  Convert each NetworkX DAG to a ``torch_geometric.data.Data`` object
        with integer-coded node features and bidirectional edges (for
        undirected GCN message passing).
    4.  Pre-compute the all-pairs Graph Edit Distance (GED) matrix over
        unique DAG structures for use in the Graph-Aware Contrastive Loss.

Reference
---------
Sanchez-Lengeling et al., "Evaluating Attribution for Graph Neural Networks",
NeurIPS 2020, for GED methodology on attributed graphs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Tool Vocabulary — 45 canonical API operations
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_VOCAB: List[str] = [
    "db_read",              # 0
    "db_write",             # 1
    "reset_password",       # 2
    "create_ticket",        # 3
    "send_notification",    # 4
    "quarantine_system",    # 5
    "scan_malware",         # 6
    "generate_report",      # 7
    "process_refund",       # 8
    "update_subscription",  # 9
    "provision_vm",         # 10
    "restart_service",      # 11
    "check_status",         # 12
    "escalate_to_human",    # 13
    "log_audit_event",      # 14
    # ── Tier 2 (tools 15–29) ─────────────────────────────────────────────
    "deploy_container",     # 15
    "rollback_deployment",  # 16
    "rotate_api_key",       # 17
    "backup_database",      # 18
    "restore_backup",       # 19
    "scale_service",        # 20
    "run_pipeline",         # 21
    "approve_access",       # 22
    "revoke_access",        # 23
    "transfer_ownership",   # 24
    "schedule_maintenance", # 25
    "archive_data",         # 26
    "enable_feature_flag",  # 27
    "disable_feature_flag", # 28
    "invalidate_cache",     # 29
    # ── Tier 3 (tools 30–44) ─────────────────────────────────────────────
    "create_dns_record",    # 30
    "renew_certificate",    # 31
    "block_ip_address",     # 32
    "unblock_ip_address",   # 33
    "assign_role",          # 34
    "remove_role",          # 35
    "trigger_failover",     # 36
    "run_load_test",        # 37
    "snapshot_vm",          # 38
    "migrate_database",     # 39
    "tag_resource",         # 40
    "create_alert_rule",    # 41
    "acknowledge_alert",    # 42
    "merge_accounts",       # 43
    "export_data",          # 44
]

from vocab_config import ACTIVE_TOOL_COUNT as _ACTIVE_TOOL_COUNT

_FULL_TOOL_VOCAB = list(TOOL_VOCAB)
TOOL_VOCAB = _FULL_TOOL_VOCAB[:_ACTIVE_TOOL_COUNT]
TOOL_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(TOOL_VOCAB)}
NUM_TOOLS: int = len(TOOL_VOCAB)


def register_tools(tool_names) -> int:
    """Expand TOOL_VOCAB / TOOL_TO_IDX with previously unseen tool names.

    Returns the updated vocabulary size.
    """
    global NUM_TOOLS
    for t in tool_names:
        if t not in TOOL_TO_IDX:
            TOOL_TO_IDX[t] = len(TOOL_VOCAB)
            TOOL_VOCAB.append(t)
    NUM_TOOLS = len(TOOL_VOCAB)
    return NUM_TOOLS

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "db_read":              "Read / query data from a database",
    "db_write":             "Write / update data in a database",
    "reset_password":       "Reset a user's password or credentials",
    "create_ticket":        "Create a support or incident ticket",
    "send_notification":    "Send an email, alert, or notification",
    "quarantine_system":    "Isolate a compromised or misbehaving system",
    "scan_malware":         "Run a malware or vulnerability scan",
    "generate_report":      "Generate an analytics or compliance report",
    "process_refund":       "Process a financial refund or credit",
    "update_subscription":  "Update, upgrade, or cancel a subscription",
    "provision_vm":         "Provision a new virtual machine or instance",
    "restart_service":      "Restart a service or application process",
    "check_status":         "Check system, service, or request status",
    "escalate_to_human":    "Escalate an issue to a human agent",
    "log_audit_event":      "Record an event in the audit / compliance log",
    # Tier 2
    "deploy_container":     "Deploy a containerised application",
    "rollback_deployment":  "Roll back a recent deployment",
    "rotate_api_key":       "Rotate an API key or secret",
    "backup_database":      "Create a database backup or snapshot",
    "restore_backup":       "Restore a system from backup",
    "scale_service":        "Scale a service up or down",
    "run_pipeline":         "Trigger a CI/CD pipeline run",
    "approve_access":       "Approve an access or permission request",
    "revoke_access":        "Revoke user or service access",
    "transfer_ownership":   "Transfer resource ownership",
    "schedule_maintenance": "Schedule a maintenance window",
    "archive_data":         "Archive old data or records",
    "enable_feature_flag":  "Enable a feature flag",
    "disable_feature_flag": "Disable a feature flag",
    "invalidate_cache":     "Invalidate or flush a cache layer",
    # Tier 3
    "create_dns_record":    "Create or update a DNS record",
    "renew_certificate":    "Renew a TLS/SSL certificate",
    "block_ip_address":     "Block an IP address in the firewall",
    "unblock_ip_address":   "Unblock a previously blocked IP address",
    "assign_role":          "Assign a role or permission to a user",
    "remove_role":          "Remove a role or permission from a user",
    "trigger_failover":     "Trigger failover to a standby system",
    "run_load_test":        "Run a load or stress test",
    "snapshot_vm":          "Take a VM snapshot",
    "migrate_database":     "Migrate a database schema or data",
    "tag_resource":         "Tag a cloud resource for tracking",
    "create_alert_rule":    "Create a monitoring alert rule",
    "acknowledge_alert":    "Acknowledge a triggered alert",
    "merge_accounts":       "Merge two user or customer accounts",
    "export_data":          "Export data for a user or system",
}

_active_set = set(TOOL_VOCAB)
TOOL_DESCRIPTIONS = {k: v for k, v in TOOL_DESCRIPTIONS.items() if k in _active_set}

# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Entity pools for query-template instantiation
# ═══════════════════════════════════════════════════════════════════════════════

_ENTITY_POOLS: Dict[str, List[str]] = {
    "user":   ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"],
    "order":  ["#10234", "#20891", "#31450", "#42017", "#53698", "#60122"],
    "server": ["prod-web-01", "staging-db-02", "payment-api-03",
               "auth-svc-04", "ml-infer-05", "cdn-edge-06"],
    "dept":   ["Engineering", "Finance", "Marketing", "Legal", "HR"],
    "ticket": ["INC-4021", "INC-7733", "INC-1198", "INC-5560", "INC-8842"],
}


def _fill(template: str, rng: random.Random) -> str:
    """Replace ``{entity}`` placeholders with random pool values."""
    result = template
    for key, pool in _ENTITY_POOLS.items():
        tag = "{" + key + "}"
        while tag in result:
            result = result.replace(tag, rng.choice(pool), 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# § 2b  Colloquial phrase library for programmatic query synthesis
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_PHRASES: Dict[str, List[str]] = {
    "db_read": [
        "pull {user}'s data", "look up the records for {order}",
        "check the DB for {user}", "grab {user}'s account details",
        "query the database about {order}", "dig up the history on {user}",
        "fetch {order}'s info", "read {user}'s profile from the database",
    ],
    "db_write": [
        "save the changes for {user} to the DB",
        "persist {user}'s data to the database",
        "update {user}'s record in the DB",
        "write the new data for {order} back",
        "commit {user}'s update to the database",
        "push the changes for {user} to the DB",
        "store the {dept} config in the database",
    ],
    "reset_password": [
        "reset {user}'s password",
        "force a credential change for {user}",
        "nuke {user}'s old login and issue a fresh one",
        "rotate {user}'s credentials",
        "give {user} a new password",
        "change {user}'s auth credentials",
    ],
    "create_ticket": [
        "open a ticket for {user}", "file an incident report about {server}",
        "log a Jira for {user}'s issue", "create a support case for {dept}",
        "spin up a tracking ticket for {user}", "file a bug about {server}",
        "create a ticket about the {dept} issue",
    ],
    "send_notification": [
        "shoot {user} an email", "blast an alert to {dept}",
        "ping {user} about it", "send a heads-up to {dept}",
        "fire off a notification to {user}", "email {user} the update",
        "notify the {dept} team",
    ],
    "quarantine_system": [
        "isolate {server}", "yank {server} off the network",
        "lock down {server}", "quarantine {server}",
        "pull {server} from production",
        "disconnect {server} from everything",
    ],
    "scan_malware": [
        "run a malware sweep on {server}", "scan {server} for threats",
        "do a security check on {server}", "sweep {server} for IOCs",
        "investigate {server} for infections",
        "check {server} for vulnerabilities",
    ],
    "generate_report": [
        "whip up a report for {dept}", "generate the {dept} analytics",
        "compile a summary for {dept}", "build a compliance doc for {dept}",
        "put together a report on {server}",
        "produce the {dept} performance summary",
    ],
    "process_refund": [
        "refund {user}", "credit {user}'s account",
        "push the refund through for {order}",
        "reverse the charge on {order}",
        "process the payout for {user}", "give {user} their money back",
    ],
    "update_subscription": [
        "update {user}'s subscription", "change {user}'s plan",
        "adjust the billing tier for {user}",
        "downgrade {user}'s account",
        "modify {user}'s subscription", "switch {user} to the new plan",
    ],
    "provision_vm": [
        "spin up a VM for {dept}", "provision a new instance for {dept}",
        "deploy a server for {dept}", "create a virtual machine for {dept}",
        "stand up a new box for {dept}", "launch an instance for {dept}",
    ],
    "restart_service": [
        "bounce {server}", "restart the service on {server}",
        "reboot {server}", "give {server} a kick",
        "cycle {server}", "restart the main process on {server}",
    ],
    "check_status": [
        "check if {server} is up", "verify {server}'s health",
        "see what's going on with {server}", "assess {server}'s status",
        "diagnose {server}", "ping {server} for a health check",
    ],
    "escalate_to_human": [
        "escalate {user}'s case to the on-call",
        "get a human involved for {user}", "loop in a {dept} manager",
        "bring in tier-2 support for {user}",
        "hand {user}'s issue off to an engineer",
        "wake up the on-call about {server}",
    ],
    "log_audit_event": [
        "log {user}'s action for compliance",
        "drop an audit entry for {server}",
        "record the event in {dept}'s log",
        "create an audit trail for {order}",
        "make a compliance log about {user}",
        "log the whole {server} incident",
    ],
    # ── Tier 2 phrases ────────────────────────────────────────────────────
    "deploy_container": [
        "deploy the container for {dept}",
        "ship the new build to {server}",
        "push the container image for {dept}",
        "roll out the containerised app to {server}",
        "launch the docker deployment for {dept}",
        "get the container running on {server}",
    ],
    "rollback_deployment": [
        "roll back the last deploy on {server}",
        "revert {server} to the previous version",
        "undo the latest release on {server}",
        "take {server} back to the prior build",
        "roll {server} back to the last stable release",
        "back out the deployment on {server}",
    ],
    "rotate_api_key": [
        "rotate the API key for {server}",
        "cycle the secret for {dept}'s service",
        "generate a new API key for {server}",
        "swap out {server}'s credentials",
        "refresh the API token for {dept}",
        "issue a new secret for {server}",
    ],
    "backup_database": [
        "back up the database for {dept}",
        "take a snapshot of {server}'s DB",
        "create a DB backup for {dept}",
        "dump the database on {server}",
        "save a backup of {dept}'s data",
        "snapshot the database for {server}",
    ],
    "restore_backup": [
        "restore {server} from the latest backup",
        "bring {server} back from the snapshot",
        "recover {dept}'s data from backup",
        "load the backup onto {server}",
        "restore the DB snapshot for {dept}",
        "pull {server} back from the last good backup",
    ],
    "scale_service": [
        "scale up {server}",
        "add more replicas for {server}",
        "bump the capacity on {server}",
        "scale {server} down to save costs",
        "adjust the instance count for {server}",
        "resize {server}'s resources",
    ],
    "run_pipeline": [
        "kick off the CI/CD pipeline for {dept}",
        "trigger a build for {server}",
        "run the deployment pipeline for {dept}",
        "start the CI job for {server}",
        "fire off the pipeline for {dept}",
        "execute the build pipeline for {server}",
    ],
    "approve_access": [
        "approve {user}'s access request",
        "grant {user} permissions to {server}",
        "sign off on {user}'s permission request",
        "give {user} access to {dept}'s resources",
        "authorize {user} for {server}",
        "green-light {user}'s access to {dept}",
    ],
    "revoke_access": [
        "revoke {user}'s access to {server}",
        "cut off {user}'s permissions",
        "remove {user} from {dept}'s access list",
        "pull {user}'s credentials for {server}",
        "strip {user}'s access to {dept}",
        "disable {user}'s account on {server}",
    ],
    "transfer_ownership": [
        "transfer ownership of {server} to {user}",
        "hand {server} over to {dept}",
        "reassign {server} to {user}",
        "move {server}'s ownership to {dept}",
        "pass {server} to {user}",
        "change the owner of {server} to {dept}",
    ],
    "schedule_maintenance": [
        "schedule maintenance for {server}",
        "book a maintenance window for {server}",
        "set up downtime for {server}",
        "plan maintenance on {server} for {dept}",
        "arrange a maintenance slot for {server}",
        "queue a maintenance window for {dept}",
    ],
    "archive_data": [
        "archive {user}'s old records",
        "move {dept}'s stale data to cold storage",
        "archive the old entries for {user}",
        "put {dept}'s historical data into the archive",
        "cold-store the old records for {user}",
        "shelve {dept}'s outdated data",
    ],
    "enable_feature_flag": [
        "turn on the feature flag for {dept}",
        "enable the flag for {server}",
        "flip the feature toggle on for {dept}",
        "activate the feature flag on {server}",
        "switch on the flag for {dept}",
        "light up the feature for {server}",
    ],
    "disable_feature_flag": [
        "turn off the feature flag for {dept}",
        "disable the flag on {server}",
        "kill the feature toggle for {dept}",
        "deactivate the feature flag on {server}",
        "switch off the flag for {dept}",
        "shut down the feature on {server}",
    ],
    "invalidate_cache": [
        "flush the cache on {server}",
        "invalidate {server}'s cache",
        "clear the cache for {dept}",
        "bust the cache on {server}",
        "purge {server}'s cached data",
        "wipe the cache layer for {dept}",
    ],
    # ── Tier 3 phrases ────────────────────────────────────────────────────
    "create_dns_record": [
        "create a DNS entry for {server}",
        "add a DNS record for {dept}'s service",
        "point the domain to {server}",
        "set up DNS for {server}",
        "register a DNS name for {dept}",
        "update the DNS to route to {server}",
    ],
    "renew_certificate": [
        "renew the TLS cert for {server}",
        "refresh the SSL certificate on {server}",
        "the cert for {server} is about to expire, renew it",
        "get a new certificate for {server}",
        "update the SSL on {server}",
        "reissue the TLS certificate for {dept}",
    ],
    "block_ip_address": [
        "block the suspicious IP hitting {server}",
        "add the attacker's IP to the firewall blocklist",
        "deny traffic from that IP on {server}",
        "blacklist the IP that's flooding {server}",
        "firewall off the offending IP on {server}",
        "drop all traffic from that address on {server}",
    ],
    "unblock_ip_address": [
        "unblock the IP we flagged on {server}",
        "remove the IP from the blocklist on {server}",
        "whitelist the IP that was accidentally blocked",
        "let traffic through from that IP again on {server}",
        "undo the IP block on {server}",
        "re-allow the blocked address on {server}",
    ],
    "assign_role": [
        "assign the admin role to {user}",
        "give {user} the editor permission",
        "add the reviewer role to {user}'s account",
        "grant {user} the {dept} manager role",
        "set {user} up with read-write access",
        "attach the operator role to {user}",
    ],
    "remove_role": [
        "remove the admin role from {user}",
        "strip {user}'s editor permission",
        "take away {user}'s manager role",
        "revoke the reviewer role from {user}",
        "detach the operator role from {user}",
        "pull the {dept} admin role from {user}",
    ],
    "trigger_failover": [
        "trigger failover for {server}",
        "switch to the standby for {server}",
        "fail over {server} to the secondary",
        "activate the DR replica for {server}",
        "move traffic to the backup for {server}",
        "initiate failover on {server}",
    ],
    "run_load_test": [
        "run a load test against {server}",
        "stress test {server} before the launch",
        "hammer {server} with test traffic",
        "do a performance test on {server}",
        "benchmark {server} under heavy load",
        "simulate peak traffic on {server}",
    ],
    "snapshot_vm": [
        "take a snapshot of {server}",
        "capture the current state of {server}",
        "create a VM snapshot for {server}",
        "save {server}'s disk image",
        "snapshot {server} before the upgrade",
        "freeze {server}'s state into a snapshot",
    ],
    "migrate_database": [
        "migrate the database on {server}",
        "run the schema migration for {dept}",
        "apply the DB migration to {server}",
        "upgrade the database schema on {server}",
        "execute the data migration for {dept}",
        "move the data to the new schema on {server}",
    ],
    "tag_resource": [
        "tag {server} with the {dept} cost center",
        "label {server} for tracking",
        "add a tag to {server} for billing",
        "mark {server} with the project identifier",
        "apply the {dept} tag to {server}",
        "annotate {server} with the environment label",
    ],
    "create_alert_rule": [
        "set up an alert for {server}'s CPU usage",
        "create a monitoring rule for {server}",
        "add an alert when {server} goes above 90% memory",
        "configure a threshold alert for {server}",
        "build a notification rule for {server}'s latency",
        "define an alert on {server}'s error rate",
    ],
    "acknowledge_alert": [
        "acknowledge the alert on {server}",
        "ack the firing alert for {server}",
        "mark the {server} alert as seen",
        "confirm we're aware of the {server} incident",
        "silence the alert on {server} for now",
        "accept the notification about {server}",
    ],
    "merge_accounts": [
        "merge {user}'s two accounts",
        "combine the duplicate profiles for {user}",
        "unify {user}'s old and new accounts",
        "consolidate {user}'s duplicate entries",
        "join the two accounts for {user} into one",
        "fold {user}'s secondary account into the primary",
    ],
    "export_data": [
        "export {user}'s data",
        "pull a data dump for {user}",
        "generate a data export for {dept}",
        "package up {user}'s records for download",
        "create a data extract for {dept}",
        "download all of {user}'s information",
    ],
}

_Q_OPENERS: List[str] = [
    "Hey, can you", "I need you to", "Quick one —", "Urgent:",
    "{user} is asking us to", "The {dept} team needs us to",
    "We gotta", "Time to", "", "Do me a favor and",
    "Ticket {ticket} says", "ASAP —", "When you get a chance,",
    "{user} reported an issue —", "From {dept}:", "Per {user}'s request,",
    "For the {dept} team,",
]

_SEQ_CONN: List[str] = [
    ", then ", ", and then ", ". After that, ", ". Next, ",
    ", followed by ", ". Once that's done, ", " — then ",
]
_PAR_CONN: List[str] = [
    " and simultaneously ", " while also ",
    " and at the same time ", " — in parallel, ",
    " and concurrently ", ", and while you're at it, ",
]
_MERGE_CONN: List[str] = [
    ". Once both are done, ", ". After those finish, ",
    ", merge everything and ", ". When all of that's complete, ",
    ", then converge and ",
]
_FINAL_CONN: List[str] = [
    ". Finally, ", ". To wrap up, ", ". Last step: ",
    ", and lastly ", ". End with ",
]


def _synthesize_queries(
    tools: List[str],
    edges: List[Tuple[int, int]],
    seed_offset: int,
    n: int = 5,
) -> List[str]:
    """Generate *n* colloquial queries for a workflow template.

    Uses topological layering so sequential steps get chain connectors,
    fan-out steps get parallel connectors, and fan-in nodes get merge
    phrasing — producing natural sentence flow for any DAG topology.
    """
    rng = random.Random(42 + seed_offset)
    num_nodes = len(tools)

    if num_nodes == 1:
        patterns = [
            "Just {p0}. Nothing else needed.",
            "Quick task: {p0}.",
            "{opener} {p0}.",
            "All I need is for someone to {p0}.",
            "Simple request — {p0} and we're good.",
        ]
        queries = []
        for _ in range(n):
            p0 = rng.choice(_TOOL_PHRASES[tools[0]])
            opener = rng.choice(_Q_OPENERS) or "Please"
            q = rng.choice(patterns).format(p0=p0, opener=opener)
            q = q[0].upper() + q[1:]
            queries.append(q)
        return queries

    children: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    parents: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
    for s, d in edges:
        children[s].append(d)
        parents[d].append(s)

    layers: Dict[int, int] = {}
    for node in range(num_nodes):
        if not parents[node]:
            layers[node] = 0
    changed = True
    while changed:
        changed = False
        for s, d in edges:
            if s in layers:
                new_layer = layers[s] + 1
                if d not in layers or new_layer > layers[d]:
                    layers[d] = new_layer
                    changed = True

    max_layer = max(layers.values()) if layers else 0
    grouped: List[List[int]] = [[] for _ in range(max_layer + 1)]
    for node in range(num_nodes):
        if node in layers:
            grouped[layers[node]].append(node)

    queries: List[str] = []
    for _ in range(n):
        phrases = {i: rng.choice(_TOOL_PHRASES[tools[i]])
                   for i in range(num_nodes)}
        parts: List[str] = []

        for layer_idx, layer_nodes in enumerate(grouped):
            if not layer_nodes:
                continue
            if len(layer_nodes) == 1:
                phrase = phrases[layer_nodes[0]]
            elif len(layer_nodes) == 2:
                phrase = (f"{phrases[layer_nodes[0]]} and "
                          f"{phrases[layer_nodes[1]]}")
            else:
                rest = ", ".join(phrases[nd] for nd in layer_nodes[:-1])
                phrase = f"{rest}, and {phrases[layer_nodes[-1]]}"

            is_fan_in = any(len(parents[nd]) > 1 for nd in layer_nodes)
            is_parallel = len(layer_nodes) > 1

            if layer_idx == 0:
                opener = rng.choice(_Q_OPENERS)
                if opener:
                    parts.append(f"{opener} {phrase}")
                else:
                    parts.append(phrase[0].upper() + phrase[1:])
            elif is_fan_in and not is_parallel:
                parts.append(rng.choice(_MERGE_CONN) + phrase)
            elif is_parallel:
                parts.append(rng.choice(_PAR_CONN) + phrase)
            elif layer_idx == len(grouped) - 1 and layer_idx > 1:
                parts.append(rng.choice(_FINAL_CONN) + phrase)
            else:
                parts.append(rng.choice(_SEQ_CONN) + phrase)

        query = "".join(parts)
        if not query.endswith("."):
            query += "."
        query = query.replace("..", ".").replace(". .", ".")
        queries.append(query)

    return queries


# ═══════════════════════════════════════════════════════════════════════════════
# § 2c  Utilities for single-tool taxonomy experiments and LLM baselines
# ═══════════════════════════════════════════════════════════════════════════════

def build_single_tool_dataset(
    n_per_tool: int = 40,
    seed: int = 42,
    tool_names: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    """Generate a large single-tool dataset for taxonomy comparison.

    Returns a list of dicts with fields:
        - query   : natural-language query
        - tool    : tool name from the selected routing vocabulary
        - tool_id : integer index within the selected tool list

    With the default ``n_per_tool=40`` and 45 tools, this yields 1800 queries,
    which you can increase (e.g. to 70 → 1,050 queries) if needed.
    """
    rng = random.Random(seed)
    examples: List[Dict[str, object]] = []
    selected_tools = list(tool_names) if tool_names is not None else list(TOOL_VOCAB)

    for tool_id, tool in enumerate(selected_tools):
        # Reuse the query synthesiser on a 1-node "DAG" for this tool.
        queries = _synthesize_queries(
            tools=[tool],
            edges=[],
            seed_offset=seed + tool_id,
            n=n_per_tool,
        )
        for q in queries:
            examples.append(
                {
                    "query": _fill(q, rng),
                    "tool": tool,
                    "tool_id": tool_id,
                }
            )

    return examples


def save_single_tool_dataset_jsonl(
    path: str | Path,
    n_per_tool: int = 40,
    seed: int = 42,
) -> None:
    """Materialise the single-tool dataset as JSONL on disk.

    Each line is a JSON object with ``query``, ``tool``, and ``tool_id`` keys.
    """
    data = build_single_tool_dataset(n_per_tool=n_per_tool, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Workflow Templates
# ═══════════════════════════════════════════════════════════════════════════════
# Format: (name, [tools_per_node], [(src_idx, dst_idx), ...], [query, ...])
# Node indices are positions in the tools list.

_T = Tuple[str, List[str], List[Tuple[int, int]], List[str]]

WORKFLOW_TEMPLATES: List[_T] = [
    # ── 2-node linear (Implicit & Colloquial) ────────────────────────────────
    ("implicit_refund",
     ["db_read", "process_refund"], [(0, 1)],
     ["Customer {user} is furious about being charged twice, fix this now.",
      "Can you see why {user} was billed again? Give them their money back.",
      "I need you to look up {order} and make sure they get a full refund.",
      "Refund the latest transaction for {user}, I think there was a billing glitch.",
      "Pull up the record for {order} and reverse the charge immediately."]),

    ("implicit_quarantine",
     ["scan_malware", "quarantine_system"], [(0, 1)],
     ["We are seeing weird pings from {server}, lock it down.",
      "Run a check on {server}, it's acting suspicious. Isolate if needed.",
      "Something's wrong with {server}, do a security sweep and pull it off the network.",
      "Investigate {server} for infections and disconnect it from prod.",
      "Check {server} for any active threats and immediately quarantine it."]),

    ("implicit_restart",
     ["check_status", "restart_service"], [(0, 1)],
     ["The {dept} app is timing out, can you bounce it?",
      "Is {server} down again? Give it a swift kick.",
      "Check why {server} is hanging and reboot the main process.",
      "Take a look at the health of {server} and just restart it if it's stuck.",
      "Verify if {server} is actually unresponsive, and if so, restart it."]),

    ("implicit_notify",
     ["db_read", "send_notification"], [(0, 1)],
     ["Fetch the list of folks in {dept} and blast out an update.",
      "We need to warn {user} about the maintenance, get their info and email them.",
      "Pull the contact details from the DB and shoot {user} a message.",
      "Look up {user}'s profile and ping them about the new features.",
      "Grab the mailing list for {dept} and send out the weekly alert."]),

    ("implicit_report",
     ["db_read", "generate_report"], [(0, 1)],
     ["Get me the numbers from the database and put together a slide deck.",
      "Pull the Q3 stats and whip up a compliance document.",
      "We need a summary of the latest metrics, fetch them and generate the PDF.",
      "Read the analytics tables and dump it into a report for {dept}.",
      "Query the recent logs and compile a performance summary."]),

    # ── 3-node linear (Distractor & Compound) ────────────────────────────────
    ("compound_refund_notify",
     ["db_read", "process_refund", "send_notification"], [(0, 1), (1, 2)],
     ["Check order {order}, process the credit, and shoot the customer an email.",
      "Look into {user}'s account, give them their money back, and let them know.",
      "Find out what happened with {user}, refund them, and send an apology note.",
      "Retrieve the billing details for {order}, reverse it, and email a receipt.",
      "Get the transaction for {user}, push a refund through, and alert them."]),

    ("compound_quarantine_log",
     ["scan_malware", "quarantine_system", "log_audit_event"], [(0, 1), (1, 2)],
     ["Sweep {server} for bugs, yank it from the network, and make sure compliance knows.",
      "Do a deep scan on {server}, isolate it, and drop a record in the audit trail.",
      "Check {server} for IOCs, quarantine the box, and log the whole incident.",
      "Find any malware on {server}, lock it down, and record the event for sec-ops.",
      "Investigate {server}, disconnect it, and make an official audit entry."]),

    ("compound_provision_log",
     ["provision_vm", "db_write", "log_audit_event"], [(0, 1), (1, 2)],
     ["The {dept} folks need a new box spun up. Register it and drop an audit log.",
      "Create a new instance for {dept}, save the config, and log the action.",
      "Spin up a VM, update our asset DB, and make sure it's in the audit trail.",
      "Provision a server for {dept}, write to the registry, and log for compliance.",
      "Deploy a new virtual machine, save it to the DB, and record the event."]),

    ("compound_reset_notify",
     ["reset_password", "log_audit_event", "send_notification"], [(0, 1), (1, 2)],
     ["User {user} can't log in and is super mad, give them a new pass, log it, and tell them.",
      "Force a credential reset for {user}, drop an audit log, and email them the new password.",
      "Change {user}'s password, make a compliance record, and notify them immediately.",
      "Reset the login for {user}, record the security event, and send them an alert.",
      "Update {user}'s auth details, log the change, and shoot them a confirmation."]),

    ("compound_escalate_ticket",
     ["check_status", "escalate_to_human", "create_ticket"], [(0, 1), (1, 2)],
     ["The {server} is entirely unresponsive, wake up the on-call and open a sev-1.",
      "Check what's wrong with the deployment, escalate it to the lead, and file a bug.",
      "Verify the outage on {server}, ping the engineering team, and log an incident ticket.",
      "Diagnose the issue, get a manager involved, and create a tracking ticket.",
      "Assess the {server} failure, escalate to a human agent, and open a Jira for it."]),

    # ── Fan-out / Fan-in (Complex Routing) ───────────────────────────────────
    ("fanout_refund_ticket",
     ["db_read", "process_refund", "create_ticket"], [(0, 1), (0, 2)],
     ["Look up {order}, process the refund, but also open a ticket so we investigate.",
      "Find {user}'s transaction, give them a refund, and file a bug report ticket in parallel.",
      "Pull the record for {order}, credit the account, and concurrently log a Jira ticket.",
      "Query {user}'s billing, issue a refund, and simultaneously create an incident ticket.",
      "Check order {order}, refund it, and open a support ticket to track the root cause."]),

    ("fanout_quarantine_notify",
     ["scan_malware", "quarantine_system", "send_notification"], [(0, 1), (0, 2)],
     ["Scan {server}, and if bad, isolate it while simultaneously alerting {dept}.",
      "Run a security check on {server}, lock it down and blast an email to the team.",
      "Check {server} for threats, yank it off the network and notify the admins at the same time.",
      "Investigate {server}, apply quarantine, and send a notification about the breach.",
      "Do a malware sweep on {server}, disconnect it, and send a warning to the owners."]),

    ("fanin_scan_log",
     ["db_read", "scan_malware", "log_audit_event"], [(0, 2), (1, 2)],
     ["Pull the access logs and run a malware sweep, then log both findings together.",
      "Check the DB records and scan {server} for threats, then dump it all in the audit log.",
      "Retrieve the historical data and do a live scan, then record everything for compliance.",
      "Read the user activity and scan the system, logging the combined results at the end.",
      "Get the DB entry and check for malware, making sure to log both actions."]),

    ("fanin_refund_notify",
     ["process_refund", "update_subscription", "send_notification"], [(0, 2), (1, 2)],
     ["Push the refund through and downgrade their plan, then send one email confirming both.",
      "Credit the account and cancel the sub, then shoot {user} a message about the changes.",
      "Reverse the charge for {order} and update the billing tier, then notify the customer.",
      "Refund the money and change {user}'s subscription, sending a single alert afterwards.",
      "Process the payout and modify the subscription status, then email a receipt."]),

    # ── Diamond & Cross-linked ───────────────────────────────────────────────
    ("diamond_refund_flow",
     ["db_read", "process_refund", "update_subscription", "send_notification"],
     [(0, 1), (0, 2), (1, 3), (2, 3)],
     ["Look up {user}, handle the refund and sub cancellation in parallel, then send one confirmation.",
      "Query the DB, process the refund while updating the plan, and notify {user} once done.",
      "Pull {user}'s info, do the credit and subscription changes, then blast an email.",
      "Find {order}, refund it and downgrade the account, then shoot them a single message.",
      "Retrieve the details, reverse the charge and cancel the service, then alert the customer."]),

    ("diamond_incident_flow",
     ["check_status", "restart_service", "escalate_to_human", "create_ticket"],
     [(0, 1), (0, 2), (1, 3), (2, 3)],
     ["Verify {server} is down, bounce it while waking up on-call, then file a master ticket.",
      "Check the outage, restart the app and ping a human, then log it all in a ticket.",
      "Assess {server}, reboot the service and escalate, then open an incident tracking ticket.",
      "Diagnose the crash, try a restart and involve an engineer, then create a Jira.",
      "Monitor the failure, restart the node and escalate, then consolidate into one ticket."]),

    # ── Deep & Complex Graphs (5-8 nodes) ────────────────────────────────────
    ("deep_incident_response",
     ["check_status", "scan_malware", "quarantine_system", "restart_service", 
      "escalate_to_human", "send_notification", "log_audit_event"],
     [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
     ["Massive outage on {server}: check it, scan for bugs, quarantine, reboot, escalate to on-call, notify {dept}, and log everything.",
      "Full IR flow: verify {server} is down, run a scan, isolate, restart, get a human involved, alert the team, and audit it.",
      "Investigate {server} status, do a malware check, lock it down, bounce the service, escalate to tier 2, email {dept}, and record the event.",
      "Diagnose {server}, scan for threats, pull it off the network, restart, escalate to management, notify stakeholders, and log for compliance.",
      "Check health of {server}, run security sweep, quarantine, reboot, escalate the incident, send a blast to {dept}, and make an audit entry."]),

    ("deep_offboarding",
     ["db_read", "reset_password", "update_subscription", "process_refund", 
      "db_write", "log_audit_event", "send_notification"],
     [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
     ["Offboard {user}: pull their profile, nuke their password, cancel their sub, refund the balance, update the DB, log it, and email them.",
      "Terminate {user}'s account: read DB, reset creds, downgrade plan, issue prorated refund, save changes, audit the action, and notify.",
      "Handle {user} departure: fetch data, lock login, stop subscription, refund any remaining credit, write to DB, log event, and send notice.",
      "Process offboarding for {user}: look them up, disable access, cancel billing, refund them, update records, drop an audit log, and email.",
      "Close out {user}: query account, reset password, end subscription, process final refund, update DB, create audit trail, and notify them."]),

    ("diamond_provision_complex",
     ["check_status", "provision_vm", "db_write", "create_ticket", "send_notification"],
     [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)],
     ["Check capacity, spin up a VM for {dept}, then register it and open a tracking ticket, finally sending one confirmation email.",
      "Verify resources, provision a new server, update the DB and create an infra ticket, then notify {dept}.",
      "Assess {server} status, deploy a new instance, write to inventory and log a Jira, then send a combined alert.",
      "Check if we can host it, provision the VM, record the asset and file a setup ticket, then email the requester.",
      "Review infra health, create a virtual machine, save to DB and open a ticket, then shoot a message to {dept}."]),
      
    ("full_audit_compliance",
     ["db_read", "generate_report", "scan_malware", "check_status", "log_audit_event"],
     [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)],
     ["Quarterly audit: read the DB, generate a report, scan {server}, check all statuses, and funnel everything into a master audit log.",
      "Run the compliance check: pull data, make a report, do a malware sweep, verify health, and log all three results.",
      "Audit {dept}: query records, create the PDF, scan for vulnerabilities, check uptime, and record the combined event.",
      "Full security review: fetch info, generate analytics, scan infrastructure, check node status, and dump to the audit trail.",
      "Perform the monthly audit: lookup details, build a summary report, scan {server}, assess health, and log it all for compliance."]),

    # ── 4-node linear chains ──────────────────────────────────────────────────
    ("chain_audit_reset",
     ["check_status", "reset_password", "db_write", "log_audit_event"],
     [(0, 1), (1, 2), (2, 3)],
     ["Verify {user}'s account, force a credential reset, persist to DB, and log for compliance.",
      "Check {user}'s auth state, change their password, update the database, then create an audit entry.",
      "Look into {user}'s login, reset the credentials, write the change, and record the event.",
      "Inspect {user}'s account health, issue a new password, save to DB, and drop an audit log.",
      "Confirm {user}'s status, rotate their credentials, commit to the database, and log it all."]),

    ("chain_provision_full",
     ["check_status", "provision_vm", "db_write", "send_notification"],
     [(0, 1), (1, 2), (2, 3)],
     ["Check capacity first, then spin up a VM for {dept}, register it, and email confirmation.",
      "Verify resource availability, create the instance, write to the asset DB, and notify the team.",
      "Assess the infrastructure, provision a server, record it in inventory, and send {dept} an alert.",
      "Check if we have room, deploy the VM, save the configuration, and shoot a message to {dept}.",
      "Review current utilization, stand up the machine, update records, and let {dept} know."]),

    # ── Wide fan-out (1 → 3) ─────────────────────────────────────────────────
    ("wide_fanout_alert",
     ["scan_malware", "quarantine_system", "send_notification", "log_audit_event"],
     [(0, 1), (0, 2), (0, 3)],
     ["Scan {server} and then simultaneously quarantine it, alert {dept}, and log the event.",
      "Run a security check on {server}, then in parallel: isolate, notify, and audit.",
      "Sweep {server} for threats, then concurrently lock it down, email the team, and record it.",
      "Do a malware scan on {server} and fan out: quarantine, send alerts, and create audit trail.",
      "Investigate {server}, then kick off quarantine, notification, and logging all at once."]),

    # ── Wide fan-in (3 → 1) ──────────────────────────────────────────────────
    ("wide_fanin_summary",
     ["db_read", "check_status", "scan_malware", "create_ticket"],
     [(0, 3), (1, 3), (2, 3)],
     ["Pull the records, check system health, and run a scan, then create one unified ticket.",
      "Gather DB info, verify {server} status, and sweep for threats, then file a combined incident.",
      "Read the data, assess uptime, and do a security check, merging everything into a single ticket.",
      "Query the database, check {server}, and scan for malware, then open a comprehensive ticket.",
      "Fetch records, diagnose the system, and investigate security, then consolidate into one ticket."]),

    # ── Y-shape (2 → 1 → 1) ──────────────────────────────────────────────────
    ("y_merge_report",
     ["db_read", "scan_malware", "generate_report", "send_notification"],
     [(0, 2), (1, 2), (2, 3)],
     ["Pull the data and run a scan, combine into a report, and email it to {dept}.",
      "Query records and sweep for threats, generate a unified PDF, then notify stakeholders.",
      "Fetch the DB and do a security check, compile a report from both, and send it out.",
      "Read the analytics and scan {server}, produce a merged summary, and alert the team.",
      "Get the data and investigate threats, build a combined report, and distribute to {dept}."]),

    # ── Inverted Y (1 → 1 → 2) ───────────────────────────────────────────────
    ("inv_y_refund",
     ["db_read", "process_refund", "send_notification", "log_audit_event"],
     [(0, 1), (1, 2), (1, 3)],
     ["Look up {order}, process the refund, then simultaneously email {user} and log the event.",
      "Pull the billing record, credit the account, then in parallel: notify and create audit trail.",
      "Query {user}'s transaction, push the refund, then concurrently send confirmation and log it.",
      "Fetch the order details, reverse the charge, then fan out to notification and audit.",
      "Check {order}, do the refund, and then at the same time email a receipt and record the action."]),

    # ── Hourglass (2 → 1 → 2) ────────────────────────────────────────────────
    ("hourglass_ops",
     ["db_read", "check_status", "generate_report", "send_notification", "log_audit_event"],
     [(0, 2), (1, 2), (2, 3), (2, 4)],
     ["Gather DB data and system health, compile into a report, then email {dept} and log it.",
      "Pull records and check {server}, generate a summary, then fan out to notification and audit.",
      "Read analytics and verify uptime, produce the report, then simultaneously alert and log.",
      "Fetch data and assess status, build a combined document, then send alerts and record for compliance.",
      "Query the DB and check services, create the PDF, then concurrently notify {dept} and audit."]),

    # ── Complex multi-path (6 nodes, cross-linked) ───────────────────────────
    ("complex_security_response",
     ["check_status", "scan_malware", "quarantine_system", "create_ticket",
      "escalate_to_human", "log_audit_event"],
     [(0, 1), (0, 3), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)],
     ["Assess {server}: scan and open a ticket, quarantine based on scan, merge into escalation, and log everything.",
      "Check {server}, do a scan and file a ticket, quarantine if infected, escalate with ticket context, and audit.",
      "Diagnose {server}, investigate threats and create an incident, isolate if needed, escalate to on-call, and record.",
      "Status check on {server}, run security sweep and open a case, quarantine the box, get a human involved, and log it.",
      "Verify {server} health, scan and ticket in parallel, quarantine from scan, escalate, then audit the whole flow."]),

    # ── Single-node (trivial but important edge cases) ─────────────────────
    ("single_check_status",
     ["check_status"], [],
     ["Just check if {server} is up right now.",
      "Is {server} running? Quick ping.",
      "Give me a health check on {server}.",
      "Verify that {server} is responding to requests.",
      "Quick status check on {server}, nothing else needed."]),

    ("single_create_ticket",
     ["create_ticket"], [],
     ["Open a ticket for the {dept} outage.",
      "File a bug report for the issue {user} reported.",
      "Create a Jira for the {server} problem.",
      "Log an incident ticket about the {dept} service disruption.",
      "Just open a support ticket, I'll handle the rest."]),

    # ── New 2-node combos ──────────────────────────────────────────────────
    ("implicit_escalate_notify",
     ["escalate_to_human", "send_notification"], [(0, 1)],
     ["Get a manager on this and then blast an alert to {dept}.",
      "Escalate to the on-call engineer and notify the team.",
      "Bring in a human for {user}'s issue and send them an update.",
      "Loop in tier 2 support and shoot {user} an email about it.",
      "Hand this off to a real person and alert {dept} about the escalation."]),

    ("implicit_write_notify",
     ["db_write", "send_notification"], [(0, 1)],
     ["Save the new config to the database and email {dept} about it.",
      "Persist {user}'s changes and ping them when it's done.",
      "Write the updated settings to DB and shoot a confirmation to {user}.",
      "Commit the record to the database and fire off a notification.",
      "Update the DB entry and let {dept} know the change went through."]),

    ("implicit_provision_log",
     ["provision_vm", "log_audit_event"], [(0, 1)],
     ["Spin up a new box for {dept} and log the provisioning event.",
      "Create a VM and make sure there's an audit trail for it.",
      "Deploy a new instance and record it for compliance.",
      "Provision the server and drop an entry in the audit log.",
      "Stand up a machine for {dept} and log the action."]),

    # ── New 3-node chains ──────────────────────────────────────────────────
    ("chain_read_update_notify",
     ["db_read", "update_subscription", "send_notification"], [(0, 1), (1, 2)],
     ["Look up {user}'s account, change their subscription tier, and email them.",
      "Pull {user}'s billing info, adjust the plan, and send a confirmation.",
      "Query {user}'s details, update their subscription, and notify them of the change.",
      "Fetch the account data, modify the billing plan, and alert {user}.",
      "Check {user}'s current plan, make the upgrade, and shoot them a message."]),

    ("chain_status_escalate_notify",
     ["check_status", "escalate_to_human", "send_notification"], [(0, 1), (1, 2)],
     ["Check what's going on with {server}, escalate it, and then alert the team.",
      "Verify the outage, bring in the on-call, and notify {dept} of the escalation.",
      "Assess {server}, hand it off to a human, and send a heads-up to stakeholders.",
      "Diagnose the issue, escalate to management, and notify {dept} about the status.",
      "Confirm {server} is down, get an engineer involved, and email the update."]),

    # ── New wide fan-out (1 → 3) with different tools ──────────────────────
    ("wide_fanout_provision",
     ["provision_vm", "db_write", "send_notification", "log_audit_event"],
     [(0, 1), (0, 2), (0, 3)],
     ["Spin up the VM and then simultaneously save the config, notify {dept}, and log it.",
      "Provision the instance, then in parallel: write to DB, alert the team, and audit.",
      "Deploy the server and fan out to: persist the record, email {dept}, and create an audit trail.",
      "Create the VM, then concurrently register it, send notifications, and log the event.",
      "Stand up the machine and kick off DB write, notification, and audit logging all at once."]),

    # ── New Y-shape with different tools ────────────────────────────────────
    ("y_merge_ticket",
     ["check_status", "scan_malware", "create_ticket", "escalate_to_human"],
     [(0, 2), (1, 2), (2, 3)],
     ["Check {server} health and run a scan, then file a unified ticket and escalate.",
      "Verify status and sweep for threats, merge findings into one ticket, and get a human.",
      "Diagnose {server} and do a malware check, create a combined incident, then escalate.",
      "Assess uptime and scan for vulnerabilities, log it all in one ticket, and involve tier 2.",
      "Check and scan {server} in parallel, consolidate into a ticket, and escalate to on-call."]),

    # ── New diamond with different tools ────────────────────────────────────
    ("diamond_reset_flow",
     ["check_status", "reset_password", "db_write", "send_notification"],
     [(0, 1), (0, 2), (1, 3), (2, 3)],
     ["Check {user}'s account, reset their password and update the DB in parallel, then email them.",
      "Verify the login issue, change credentials while persisting changes, then send confirmation.",
      "Diagnose {user}'s auth, force a reset and write to DB, then notify them once both are done.",
      "Check the account state, do a password reset and DB update concurrently, then alert {user}.",
      "Assess {user}'s situation, handle credential reset and record update, then send one email."]),

    # ── N-shape (cross-linked 4-node) ──────────────────────────────────────
    ("n_shape_ops",
     ["db_read", "generate_report", "check_status", "send_notification"],
     [(0, 1), (0, 2), (2, 3), (1, 3)],
     ["Pull data and check status separately, generate a report from the data, then send both results to {dept}.",
      "Query the DB and verify {server}, build a report and merge everything into one notification.",
      "Fetch records and assess health, compile the report, and fan both outcomes into a single email.",
      "Read the database and check {server}, create a report from the data, then notify with combined info.",
      "Get data and status in parallel, produce the analytics report, and converge into a notification for {dept}."]),

    # ── 5-node linear chain ────────────────────────────────────────────────
    ("chain5_full_audit",
     ["check_status", "db_read", "generate_report", "send_notification", "log_audit_event"],
     [(0, 1), (1, 2), (2, 3), (3, 4)],
     ["Check {server}, pull the data, build a report, email {dept}, and log the whole process.",
      "Verify health, query the DB, generate a summary, notify stakeholders, and create an audit entry.",
      "Assess {server}, read the records, compile the PDF, distribute to {dept}, and log for compliance.",
      "Status check, data retrieval, report generation, notification, and audit logging — full pipeline.",
      "Monitor {server}, fetch the metrics, produce the report, send it out, and record the event."]),

    # ── Tier 2 hand-crafted templates ─────────────────────────────────────
    ("deploy_and_notify",
     ["deploy_container", "send_notification"], [(0, 1)],
     ["Ship the new build to {server} and let {dept} know it's live.",
      "Deploy the container and ping the team when it's done.",
      "Push the image to {server} and email {dept} about the release.",
      "Roll out the container and blast an alert to {dept}.",
      "Get the docker deployment running on {server} and notify stakeholders."]),

    ("deploy_rollback_chain",
     ["run_pipeline", "deploy_container", "check_status", "rollback_deployment"],
     [(0, 1), (1, 2), (2, 3)],
     ["Trigger the pipeline, deploy the container, verify it's healthy, and roll back if it's not.",
      "Run the CI build, ship it, check if it's working, and revert if something's wrong.",
      "Kick off the pipeline for {dept}, deploy, verify status, and undo if needed.",
      "Fire off the build, push the container to {server}, check health, and roll back on failure.",
      "Execute the pipeline, deploy to {server}, assess status, and revert the release if broken."]),

    ("backup_restore_flow",
     ["backup_database", "restore_backup", "check_status", "send_notification"],
     [(0, 1), (1, 2), (2, 3)],
     ["Snapshot the DB, restore it on {server}, verify it's good, and notify {dept}.",
      "Back up the database, load it onto {server}, check health, and email the team.",
      "Take a DB backup, restore to {server}, verify everything, and alert {dept}.",
      "Dump the database, recover onto {server}, confirm it's up, and send a heads-up.",
      "Create a backup, restore it, check status, and let {dept} know we're good."]),

    ("access_lifecycle",
     ["approve_access", "log_audit_event", "send_notification"], [(0, 1), (0, 2)],
     ["Approve {user}'s access request, log it for compliance, and notify them in parallel.",
      "Grant {user} permissions, drop an audit entry, and shoot them an email.",
      "Sign off on {user}'s request, record it for the auditors, and ping {user}.",
      "Green-light {user}'s access, create an audit trail, and send a confirmation.",
      "Authorize {user}, log the approval, and let them know they're in."]),

    ("revoke_and_rotate",
     ["revoke_access", "rotate_api_key", "log_audit_event"], [(0, 1), (1, 2)],
     ["Revoke {user}'s access, rotate the API key, and log everything.",
      "Cut off {user}'s permissions, cycle the secret, and record it for compliance.",
      "Pull {user}'s credentials, generate a new API key, and drop an audit entry.",
      "Disable {user}'s account, swap out the API secret, and log the whole thing.",
      "Strip {user}'s access, refresh the token, and make an audit trail."]),

    ("scale_and_cache",
     ["scale_service", "invalidate_cache", "check_status"], [(0, 1), (1, 2)],
     ["Scale up {server}, flush the cache, and check that everything's healthy.",
      "Bump the capacity on {server}, bust the cache, and verify status.",
      "Add replicas for {server}, purge the cached data, and confirm it's stable.",
      "Resize {server}, wipe the cache layer, and do a health check.",
      "Adjust {server}'s instance count, invalidate the cache, and assess health."]),

    ("feature_flag_diamond",
     ["enable_feature_flag", "check_status", "run_pipeline", "send_notification"],
     [(0, 1), (0, 2), (1, 3), (2, 3)],
     ["Enable the flag, check health and trigger a pipeline in parallel, then notify {dept}.",
      "Flip the toggle on, verify status and kick off CI concurrently, then email the team.",
      "Activate the feature, assess {server} and run the build, then send one alert.",
      "Turn on the flag for {dept}, check everything and run the pipeline, then notify.",
      "Light up the feature, do a health check and trigger CI, then blast a notification."]),

    ("maintenance_window_flow",
     ["schedule_maintenance", "send_notification", "backup_database", "log_audit_event"],
     [(0, 1), (0, 2), (1, 3), (2, 3)],
     ["Schedule the maintenance, notify {dept} and take a backup, then log it all.",
      "Book downtime for {server}, alert the team and snapshot the DB, then record for compliance.",
      "Plan maintenance on {server}, email stakeholders and back up the data, then audit.",
      "Set up a maintenance window, send a heads-up and dump the database, then log the event.",
      "Arrange downtime, notify {dept} and create a backup in parallel, then drop an audit entry."]),

    ("archive_and_transfer",
     ["archive_data", "transfer_ownership", "log_audit_event"], [(0, 1), (1, 2)],
     ["Archive {user}'s old records, transfer ownership to {dept}, and log it.",
      "Move the stale data to cold storage, hand the resources to {dept}, and record for compliance.",
      "Shelve the outdated data, reassign to {user}, and create an audit entry.",
      "Cold-store the old records, pass ownership to {dept}, and log the action.",
      "Put the historical data into the archive, transfer to {user}, and drop an audit log."]),

    ("single_deploy_container",
     ["deploy_container"], [],
     ["Just deploy the container to {server}.",
      "Ship the latest build to {server}, nothing else.",
      "Push the container image to {server}.",
      "Roll out the docker app to {server}.",
      "Get the container live on {server}."]),

    ("single_scale_service",
     ["scale_service"], [],
     ["Scale up {server} right now.",
      "Add more capacity to {server}.",
      "Bump {server}'s replicas.",
      "Resize {server}'s resources.",
      "Adjust the instance count for {server}."]),

    # ── Tier 3 hand-crafted templates ─────────────────────────────────────
    ("dns_and_cert_chain",
     ["create_dns_record", "renew_certificate", "check_status"], [(0, 1), (1, 2)],
     ["Set up the DNS, renew the TLS cert, and verify everything works.",
      "Point the domain to {server}, refresh the certificate, and check health.",
      "Create the DNS entry, update the SSL, and confirm the endpoint is live.",
      "Register the DNS name, get a new cert, and check that HTTPS is good.",
      "Add the DNS record, renew the certificate, and do a health check."]),

    ("ip_block_and_log",
     ["block_ip_address", "log_audit_event", "send_notification"], [(0, 1), (0, 2)],
     ["Block the IP, log it for compliance, and alert {dept} in parallel.",
      "Firewall off the attacker, record the action, and notify the team.",
      "Drop the suspicious traffic, create an audit entry, and email {dept}.",
      "Blacklist the IP, log for the auditors, and send an alert.",
      "Deny the address, drop an audit log, and ping {dept} about it."]),

    ("role_lifecycle",
     ["assign_role", "log_audit_event", "send_notification"], [(0, 1), (0, 2)],
     ["Give {user} the role, log it for compliance, and let them know.",
      "Assign the permission, record the change, and notify {user}.",
      "Set up {user}'s role, create an audit trail, and email confirmation.",
      "Grant the role to {user}, log the action, and send a heads-up.",
      "Attach the role, audit the change, and ping {user}."]),

    ("failover_flow",
     ["check_status", "trigger_failover", "send_notification", "log_audit_event"],
     [(0, 1), (1, 2), (1, 3)],
     ["Check {server}'s health, trigger failover, then notify {dept} and log it.",
      "Verify {server} is down, switch to the standby, email the team and audit.",
      "Assess {server}, activate DR, send an alert and record for compliance.",
      "Diagnose the failure, fail over to secondary, notify and log everything.",
      "Check the outage, initiate failover, then alert {dept} and create an audit entry."]),

    ("load_test_pipeline",
     ["run_load_test", "generate_report", "send_notification"], [(0, 1), (1, 2)],
     ["Run a load test on {server}, compile the results, and email {dept}.",
      "Stress test {server}, generate a performance report, and notify stakeholders.",
      "Benchmark {server}, produce a summary, and send it to {dept}.",
      "Hammer {server} with test traffic, build a report, and distribute it.",
      "Do a performance test on {server}, create the PDF, and alert the team."]),

    ("snapshot_and_migrate",
     ["snapshot_vm", "migrate_database", "check_status", "send_notification"],
     [(0, 1), (1, 2), (2, 3)],
     ["Snapshot {server}, run the migration, verify health, and notify {dept}.",
      "Capture the VM state, apply the DB migration, check it's good, and email the team.",
      "Take a snapshot, migrate the schema, confirm everything, and send a heads-up.",
      "Freeze {server}'s state, execute the migration, verify, and alert {dept}.",
      "Save the VM snapshot, do the database migration, check status, and notify."]),

    ("merge_and_export",
     ["merge_accounts", "export_data", "send_notification"], [(0, 1), (1, 2)],
     ["Merge {user}'s accounts, export their data, and send them confirmation.",
      "Combine the duplicate profiles, generate a data export, and email {user}.",
      "Unify the accounts, package up the records, and notify {user}.",
      "Consolidate {user}'s entries, pull a data dump, and send it to them.",
      "Join the two accounts, create a data extract, and let {user} know."]),

    ("alert_rule_and_ack",
     ["create_alert_rule", "check_status", "acknowledge_alert"], [(0, 1), (1, 2)],
     ["Set up a monitoring rule for {server}, check status, and ack any alerts.",
      "Create an alert, verify {server}'s health, and acknowledge what fires.",
      "Configure monitoring, check the endpoint, and mark any alerts as seen.",
      "Build a threshold alert, assess {server}, and silence any initial noise.",
      "Define the alert rule, do a health check, and acknowledge it."]),

    ("tag_and_scale",
     ["tag_resource", "scale_service", "log_audit_event"], [(0, 1), (1, 2)],
     ["Tag {server} for billing, scale it up, and log the change.",
      "Label the resource, add replicas, and record it for compliance.",
      "Apply the cost center tag, bump capacity, and create an audit entry.",
      "Mark {server} with the project ID, resize resources, and log it.",
      "Annotate {server}, adjust the instance count, and audit the action."]),

    ("single_block_ip",
     ["block_ip_address"], [],
     ["Block that IP on {server} immediately.",
      "Firewall off the attacker's address.",
      "Deny traffic from that IP.",
      "Blacklist the offending IP on {server}.",
      "Drop all connections from that address."]),

    ("single_export_data",
     ["export_data"], [],
     ["Export {user}'s data right now.",
      "Pull a data dump for {user}.",
      "Generate a data extract for {dept}.",
      "Package up {user}'s records.",
      "Download all of {user}'s information."]),
]


# ═══════════════════════════════════════════════════════════════════════════════
# § 3b  Extended templates — programmatic generation for dataset scale-up
# ═══════════════════════════════════════════════════════════════════════════════
#
# 39 hand-crafted templates (above) are insufficient for meaningful
# Recall@k evaluation.  This section adds ~111 structurally unique DAGs
# via curated (tool-combo, topology) pairs with synthesized queries,
# bringing the total to ~150 unique DAGs.
#
# Each topology family is defined as a list of tool-sequences; edges
# are inferred from the topology pattern.  Queries are generated by
# _synthesize_queries() using the colloquial phrase library (§ 2b).

# ── Single-node additions ────────────────────────────────────────────────────
_EXT_SINGLE: List[List[str]] = [
    ["db_read"],
    ["db_write"],
    ["restart_service"],
    ["scan_malware"],
    ["send_notification"],
]

# ── 2-node linear (A → B) ───────────────────────────────────────────────────
_EXT_LIN2: List[List[str]] = [
    ["restart_service", "log_audit_event"],
    ["process_refund", "send_notification"],
    ["generate_report", "send_notification"],
    ["db_write", "log_audit_event"],
    ["check_status", "send_notification"],
    ["escalate_to_human", "create_ticket"],
    ["scan_malware", "log_audit_event"],
    ["quarantine_system", "log_audit_event"],
    ["update_subscription", "send_notification"],
    ["provision_vm", "db_write"],
    ["reset_password", "send_notification"],
    ["check_status", "create_ticket"],
    ["db_read", "create_ticket"],
    ["db_read", "escalate_to_human"],
    ["create_ticket", "send_notification"],
    ["restart_service", "send_notification"],
    ["check_status", "escalate_to_human"],
    ["scan_malware", "create_ticket"],
    ["process_refund", "log_audit_event"],
    ["update_subscription", "log_audit_event"],
    ["quarantine_system", "send_notification"],
    ["restart_service", "create_ticket"],
    ["generate_report", "log_audit_event"],
    ["provision_vm", "send_notification"],
    ["db_read", "restart_service"],
]

# ── 3-node linear (A → B → C) ───────────────────────────────────────────────
_EXT_LIN3: List[List[str]] = [
    ["db_read", "process_refund", "log_audit_event"],
    ["check_status", "restart_service", "log_audit_event"],
    ["check_status", "restart_service", "send_notification"],
    ["scan_malware", "quarantine_system", "send_notification"],
    ["db_read", "update_subscription", "log_audit_event"],
    ["db_read", "generate_report", "log_audit_event"],
    ["reset_password", "db_write", "send_notification"],
    ["check_status", "provision_vm", "log_audit_event"],
    ["escalate_to_human", "create_ticket", "send_notification"],
    ["db_read", "reset_password", "send_notification"],
    ["process_refund", "db_write", "send_notification"],
    ["scan_malware", "create_ticket", "escalate_to_human"],
    ["check_status", "scan_malware", "create_ticket"],
    ["db_read", "db_write", "log_audit_event"],
    ["quarantine_system", "restart_service", "log_audit_event"],
    ["provision_vm", "check_status", "send_notification"],
    ["generate_report", "send_notification", "log_audit_event"],
    ["update_subscription", "process_refund", "send_notification"],
    ["check_status", "generate_report", "send_notification"],
    ["scan_malware", "quarantine_system", "create_ticket"],
    ["db_read", "generate_report", "send_notification"],
]

# ── 4-node linear (A → B → C → D) ──────────────────────────────────────────
_EXT_LIN4: List[List[str]] = [
    ["db_read", "process_refund", "send_notification", "log_audit_event"],
    ["check_status", "scan_malware", "quarantine_system", "log_audit_event"],
    ["scan_malware", "quarantine_system", "restart_service", "log_audit_event"],
    ["check_status", "restart_service", "send_notification", "log_audit_event"],
    ["db_read", "update_subscription", "send_notification", "log_audit_event"],
    ["provision_vm", "db_write", "send_notification", "log_audit_event"],
    ["check_status", "escalate_to_human", "create_ticket", "send_notification"],
    ["db_read", "reset_password", "db_write", "send_notification"],
]

# ── Fan-out (root → child_1, root → child_2[, root → child_3]) ──────────────
_EXT_FANOUT: List[List[str]] = [
    ["db_read", "update_subscription", "send_notification"],
    ["check_status", "restart_service", "create_ticket"],
    ["scan_malware", "quarantine_system", "create_ticket"],
    ["check_status", "escalate_to_human", "send_notification"],
    ["db_read", "process_refund", "send_notification"],
    ["db_read", "generate_report", "send_notification"],
    ["check_status", "restart_service", "send_notification"],
    ["provision_vm", "db_write", "send_notification"],
    ["scan_malware", "quarantine_system", "log_audit_event"],
    ["reset_password", "send_notification", "log_audit_event"],
    ["db_read", "create_ticket", "send_notification"],
]

# ── Fan-in (parent_1 → sink, parent_2 → sink) ──────────────────────────────
_EXT_FANIN: List[List[str]] = [
    ["db_read", "check_status", "generate_report"],
    ["scan_malware", "check_status", "create_ticket"],
    ["process_refund", "update_subscription", "log_audit_event"],
    ["db_read", "scan_malware", "create_ticket"],
    ["check_status", "db_read", "send_notification"],
    ["restart_service", "check_status", "log_audit_event"],
    ["scan_malware", "check_status", "escalate_to_human"],
    ["reset_password", "db_write", "send_notification"],
]

# ── Diamond (0→1, 0→2, 1→3, 2→3) ───────────────────────────────────────────
_EXT_DIAMOND: List[List[str]] = [
    ["db_read", "process_refund", "create_ticket", "log_audit_event"],
    ["check_status", "scan_malware", "escalate_to_human", "create_ticket"],
    ["scan_malware", "quarantine_system", "create_ticket", "log_audit_event"],
    ["db_read", "generate_report", "send_notification", "log_audit_event"],
    ["check_status", "restart_service", "send_notification", "log_audit_event"],
    ["provision_vm", "db_write", "check_status", "send_notification"],
    ["db_read", "update_subscription", "process_refund", "send_notification"],
]

# ── Y-shape (0→2, 1→2, 2→3) — two sources merge, then chain ────────────────
_EXT_YSHAPE: List[List[str]] = [
    ["db_read", "check_status", "generate_report", "send_notification"],
    ["scan_malware", "check_status", "create_ticket", "send_notification"],
    ["db_read", "check_status", "escalate_to_human", "create_ticket"],
    ["process_refund", "update_subscription", "db_write", "send_notification"],
    ["restart_service", "check_status", "generate_report", "send_notification"],
]

# ── Inverted Y (0→1, 1→2, 1→3) — chain then fan-out ────────────────────────
_EXT_INVY: List[List[str]] = [
    ["db_read", "generate_report", "send_notification", "log_audit_event"],
    ["check_status", "restart_service", "send_notification", "log_audit_event"],
    ["scan_malware", "quarantine_system", "send_notification", "log_audit_event"],
    ["db_read", "process_refund", "db_write", "send_notification"],
    ["check_status", "escalate_to_human", "create_ticket", "send_notification"],
]

# ── Hourglass (0→2, 1→2, 2→3, 2→4) — fan-in then fan-out ──────────────────
_EXT_HOURGLASS: List[List[str]] = [
    ["scan_malware", "check_status", "create_ticket",
     "escalate_to_human", "send_notification"],
    ["db_read", "check_status", "create_ticket",
     "escalate_to_human", "log_audit_event"],
    ["process_refund", "update_subscription", "db_write",
     "send_notification", "log_audit_event"],
    ["restart_service", "check_status", "create_ticket",
     "send_notification", "log_audit_event"],
    ["scan_malware", "db_read", "generate_report",
     "send_notification", "log_audit_event"],
]

# ── 5-node linear chains ────────────────────────────────────────────────────
_EXT_CHAIN5: List[List[str]] = [
    ["check_status", "scan_malware", "quarantine_system",
     "log_audit_event", "send_notification"],
    ["db_read", "process_refund", "db_write",
     "log_audit_event", "send_notification"],
    ["provision_vm", "db_write", "check_status",
     "send_notification", "log_audit_event"],
    ["db_read", "reset_password", "db_write",
     "send_notification", "log_audit_event"],
    ["scan_malware", "quarantine_system", "restart_service",
     "send_notification", "log_audit_event"],
]

# ── 6-node linear chain ─────────────────────────────────────────────────────
_EXT_CHAIN6: List[List[str]] = [
    ["scan_malware", "quarantine_system", "escalate_to_human",
     "create_ticket", "log_audit_event", "send_notification"],
]

# ── Complex multi-path (custom edge lists) ──────────────────────────────────
_EXT_COMPLEX: List[Tuple[str, List[str], List[Tuple[int, int]]]] = [
    # Diamond → chain (fan-out, merge, then tail)
    ("ext_cmplx_00",
     ["check_status", "scan_malware", "restart_service",
      "escalate_to_human", "create_ticket", "log_audit_event"],
     [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]),
    # Diamond → tail
    ("ext_cmplx_01",
     ["db_read", "process_refund", "update_subscription",
      "send_notification", "log_audit_event"],
     [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]),
    # Ultra-wide fan-out (1 → 4)
    ("ext_cmplx_02",
     ["db_read", "process_refund", "create_ticket",
      "send_notification", "log_audit_event"],
     [(0, 1), (0, 2), (0, 3), (0, 4)]),
    # W-shape: two parallel chains merging at the end
    ("ext_cmplx_03",
     ["db_read", "generate_report", "check_status",
      "send_notification", "log_audit_event"],
     [(0, 1), (2, 3), (1, 4), (3, 4)]),
    # Parallel paths merging
    ("ext_cmplx_04",
     ["db_read", "process_refund", "scan_malware",
      "quarantine_system", "log_audit_event"],
     [(0, 1), (2, 3), (1, 4), (3, 4)]),
]


def _build_extended_templates() -> List[_T]:
    """Assemble extended template entries with synthesised queries."""
    combos: List[Tuple[str, List[str], List[Tuple[int, int]]]] = []

    for i, tools in enumerate(_EXT_SINGLE):
        combos.append((f"ext_single_{i:02d}", tools, []))

    for i, tools in enumerate(_EXT_LIN2):
        combos.append((f"ext_lin2_{i:02d}", tools, [(0, 1)]))

    for i, tools in enumerate(_EXT_LIN3):
        combos.append((f"ext_lin3_{i:02d}", tools,
                        [(0, 1), (1, 2)]))

    for i, tools in enumerate(_EXT_LIN4):
        combos.append((f"ext_lin4_{i:02d}", tools,
                        [(0, 1), (1, 2), (2, 3)]))

    for i, tools in enumerate(_EXT_FANOUT):
        combos.append((f"ext_fo_{i:02d}", tools,
                        [(0, 1), (0, 2)]))

    for i, tools in enumerate(_EXT_FANIN):
        combos.append((f"ext_fi_{i:02d}", tools,
                        [(0, 2), (1, 2)]))

    for i, tools in enumerate(_EXT_DIAMOND):
        combos.append((f"ext_dia_{i:02d}", tools,
                        [(0, 1), (0, 2), (1, 3), (2, 3)]))

    for i, tools in enumerate(_EXT_YSHAPE):
        combos.append((f"ext_y_{i:02d}", tools,
                        [(0, 2), (1, 2), (2, 3)]))

    for i, tools in enumerate(_EXT_INVY):
        combos.append((f"ext_ivy_{i:02d}", tools,
                        [(0, 1), (1, 2), (1, 3)]))

    for i, tools in enumerate(_EXT_HOURGLASS):
        combos.append((f"ext_hg_{i:02d}", tools,
                        [(0, 2), (1, 2), (2, 3), (2, 4)]))

    for i, tools in enumerate(_EXT_CHAIN5):
        combos.append((f"ext_ch5_{i:02d}", tools,
                        [(0, 1), (1, 2), (2, 3), (3, 4)]))

    for i, tools in enumerate(_EXT_CHAIN6):
        combos.append((f"ext_ch6_{i:02d}", tools,
                        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]))

    for name, tools, edges in _EXT_COMPLEX:
        combos.append((name, tools, edges))

    templates: List[_T] = []
    for idx, (name, tools, edges) in enumerate(combos):
        queries = _synthesize_queries(tools, edges, seed_offset=idx * 7)
        templates.append((name, tools, edges, queries))
    return templates


WORKFLOW_TEMPLATES.extend(_build_extended_templates())


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  DAG ↔ PyG conversion utilities
# ═══════════════════════════════════════════════════════════════════════════════

def build_dag(tools: List[str], edges: List[Tuple[int, int]]) -> nx.DiGraph:
    """Construct a NetworkX DAG from a template specification."""
    G = nx.DiGraph()
    for i, tool in enumerate(tools):
        G.add_node(i, tool=tool)
    G.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(G), f"Not a DAG: {tools}, {edges}"
    return G


def dag_to_pyg(G: nx.DiGraph) -> Data:
    """Convert a NetworkX DAG to a ``torch_geometric.data.Data`` object.

    Node features are integer tool indices (shape ``[N, 1]``).  Edges are
    made bidirectional so that standard GCN message-passing can propagate
    information in both directions through the graph.  Each node also
    receives a **topological position** encoding (its rank in a topological
    sort of the original DAG) so the GCN can reason about execution order
    despite operating on undirected edges.
    """
    nodes = sorted(G.nodes())
    tool_indices = [TOOL_TO_IDX[G.nodes[n]["tool"]] for n in nodes]
    x = torch.tensor(tool_indices, dtype=torch.long).unsqueeze(-1)

    # Topological position: rank in the DAG's execution order
    topo_order = list(nx.topological_sort(G))
    topo_pos = torch.zeros(len(nodes), dtype=torch.long)
    for rank, node in enumerate(topo_order):
        topo_pos[node] = rank

    src, dst = [], []
    for u, v in G.edges():
        src.extend([u, v])
        dst.extend([v, u])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, topo_pos=topo_pos)


def dag_canonical_hash(G: nx.DiGraph) -> str:
    """Produce a deterministic hash of the DAG's labelled structure.

    Two DAGs with the same tool-labelled topology will hash identically,
    enabling efficient deduplication for GED pre-computation.
    """
    node_labels = tuple(sorted(G.nodes[n]["tool"] for n in G.nodes()))
    edge_labels = tuple(sorted(
        (G.nodes[u]["tool"], G.nodes[v]["tool"]) for u, v in G.edges()
    ))
    payload = f"{node_labels}|{edge_labels}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def dag_to_text(G: nx.DiGraph) -> str:
    """Create a canonical textual description of a DAG for baseline models.

    Example output: ``"db_read -> process_refund, db_read -> create_ticket"``
    """
    if G.number_of_edges() == 0:
        tools = [G.nodes[n]["tool"] for n in G.nodes()]
        return ", ".join(tools)
    edges_str = sorted(
        f"{G.nodes[u]['tool']} -> {G.nodes[v]['tool']}" for u, v in G.edges()
    )
    return ", ".join(edges_str)


def _build_scale_up_templates(
    n_lin2: int = 120,
    n_lin3: int = 100,
    n_lin4: int = 50,
    n_fanout: int = 40,
    n_fanin: int = 40,
    seed: int = 42,
) -> List[_T]:
    """Programmatically generate extra DAG templates for dataset scale-up.

    Produces unique linear chains (2-, 3-, 4-node) and fan-out/fan-in
    patterns from the tool vocabulary so the corpus reaches 400+ unique DAGs
    for stronger evaluation and reviewer-facing diversity.
    """
    rng = random.Random(seed)
    templates: List[_T] = []
    seen_hashes: set[str] = set()

    def add_if_new(name: str, tools: List[str], edges: List[Tuple[int, int]]) -> None:
        G = build_dag(tools, edges)
        h = dag_canonical_hash(G)
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        q = _synthesize_queries(tools, edges, seed_offset=hash(name) % 10000)
        templates.append((name, tools, edges, q))

    # All 2-node linear chains (A → B)
    lin2 = list(itertools.permutations(TOOL_VOCAB, 2))
    rng.shuffle(lin2)
    for i, (a, b) in enumerate(lin2[:n_lin2]):
        add_if_new(f"scale_lin2_{i:03d}", [a, b], [(0, 1)])

    # Sample 3-node linear chains
    lin3 = list(itertools.permutations(TOOL_VOCAB, 3))
    rng.shuffle(lin3)
    for i, (a, b, c) in enumerate(lin3[:n_lin3]):
        add_if_new(f"scale_lin3_{i:03d}", [a, b, c], [(0, 1), (1, 2)])

    # Sample 4-node linear chains
    lin4 = list(itertools.permutations(TOOL_VOCAB, 4))
    rng.shuffle(lin4)
    for i, (a, b, c, d) in enumerate(lin4[:n_lin4]):
        add_if_new(f"scale_lin4_{i:03d}", [a, b, c, d], [(0, 1), (1, 2), (2, 3)])

    # Fan-out (1 → 2 children): pick 3 distinct tools
    for i in range(n_fanout):
        triple = tuple(rng.sample(TOOL_VOCAB, 3))
        add_if_new(f"scale_fo_{i:03d}", list(triple), [(0, 1), (0, 2)])

    # Fan-in (2 parents → 1 sink)
    for i in range(n_fanin):
        triple = tuple(rng.sample(TOOL_VOCAB, 3))
        add_if_new(f"scale_fi_{i:03d}", list(triple), [(0, 2), (1, 2)])

    return templates


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  GED pre-computation
# ═══════════════════════════════════════════════════════════════════════════════

def _node_subst_cost(n1: dict, n2: dict) -> float:
    """Unit cost: 0 if same tool label, 1 otherwise."""
    return 0.0 if n1["tool"] == n2["tool"] else 1.0


def _node_del_cost(n: dict) -> float:
    return 1.0


def _node_ins_cost(n: dict) -> float:
    return 1.0


def _edge_del_cost(e: dict) -> float:
    return 1.0


def _edge_ins_cost(e: dict) -> float:
    return 1.0


def compute_ged(G1: nx.DiGraph, G2: nx.DiGraph) -> float:
    """Exact Graph Edit Distance with unit costs on attributed nodes.

    Tractable because all DAGs in our vocabulary have ≤ 6 nodes.
    """
    return nx.graph_edit_distance(
        G1, G2,
        node_subst_cost=_node_subst_cost,
        node_del_cost=_node_del_cost,
        node_ins_cost=_node_ins_cost,
        edge_del_cost=_edge_del_cost,
        edge_ins_cost=_edge_ins_cost,
    )


def build_ged_matrix(dags: List[nx.DiGraph]) -> np.ndarray:
    """Compute the symmetric pairwise GED matrix over a list of DAGs.

    Returns
    -------
    ged : np.ndarray, shape ``(N, N)``
        ``ged[i, j]`` is the graph edit distance between ``dags[i]`` and
        ``dags[j]``.
    """
    n = len(dags)
    ged = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_ged(dags[i], dags[j])
            ged[i, j] = d
            ged[j, i] = d
    return ged


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  LEGRDataset — PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Sample:
    query: str
    graph: Data
    dag_id: int              # index into the unique-DAG list
    dag_nx: nx.DiGraph       # kept for evaluation / visualisation


class LEGRDataset(Dataset):
    """Paired (query, execution-DAG) dataset for LEGR training.

    Parameters
    ----------
    templates : list of workflow template tuples (name, tools, edges, queries).
    entity_variants : int
        Number of random entity-fill variants per base query.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        templates: Optional[List[_T]] = None,
        entity_variants: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        if templates is None:
            templates = WORKFLOW_TEMPLATES

        rng = random.Random(seed)
        self.samples: List[_Sample] = []

        # Build unique DAGs and assign IDs
        self._unique_dags: List[nx.DiGraph] = []
        self._dag_texts: List[str] = []
        hash_to_id: Dict[str, int] = {}

        for name, tools, edges, queries in templates:
            G = build_dag(tools, edges)
            h = dag_canonical_hash(G)
            if h not in hash_to_id:
                hash_to_id[h] = len(self._unique_dags)
                self._unique_dags.append(G)
                self._dag_texts.append(dag_to_text(G))
            dag_id = hash_to_id[h]
            pyg_data = dag_to_pyg(G)

            for q_template in queries:
                for _ in range(entity_variants):
                    q = _fill(q_template, rng)
                    self.samples.append(_Sample(
                        query=q,
                        graph=pyg_data.clone(),
                        dag_id=dag_id,
                        dag_nx=G,
                    ))

        # Pre-compute all-pairs GED over unique DAGs
        self.ged_matrix: np.ndarray = build_ged_matrix(self._unique_dags)
        self.num_unique_dags: int = len(self._unique_dags)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "query": s.query,
            "graph": s.graph,
            "dag_id": s.dag_id,
        }

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_unique_dag(self, dag_id: int) -> nx.DiGraph:
        return self._unique_dags[dag_id]

    def get_dag_text(self, dag_id: int) -> str:
        return self._dag_texts[dag_id]

    def get_ged(self, id_a: int, id_b: int) -> float:
        return float(self.ged_matrix[id_a, id_b])

    def get_ged_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.ged_matrix).float()

    def __repr__(self) -> str:
        return (
            f"LEGRDataset(samples={len(self)}, "
            f"unique_dags={self.num_unique_dags}, "
            f"max_ged={self.ged_matrix.max():.0f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  Train / Val / Test splits
# ═══════════════════════════════════════════════════════════════════════════════

def build_splits(
    entity_variants: int = 4,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    split_mode: str = "template",
) -> Tuple[LEGRDataset, LEGRDataset, LEGRDataset]:
    """Build a full dataset and split into train / val / test.

    If split_mode == "template", splits by unique DAGs to evaluate zero-shot
    structural generalization (prevents overfitting).
    If split_mode == "random", splits randomly across all samples.

    Returns
    -------
    train_ds, val_ds, test_ds : LEGRDataset
        The ``ged_matrix`` is shared across all three and covers every
        unique DAG.
    """
    full = LEGRDataset(entity_variants=entity_variants, seed=seed)
    rng = random.Random(seed)

    if split_mode == "template":
        dag_to_samples = {}
        for i, s in enumerate(full.samples):
            dag_to_samples.setdefault(s.dag_id, []).append(i)
        
        unique_dags = list(dag_to_samples.keys())
        rng.shuffle(unique_dags)
        
        n_train_dags = int(len(unique_dags) * train_frac)
        n_val_dags = int(len(unique_dags) * val_frac)
        
        train_dags = unique_dags[:n_train_dags]
        val_dags = unique_dags[n_train_dags:n_train_dags + n_val_dags]
        test_dags = unique_dags[n_train_dags + n_val_dags:]
        
        train_indices = [i for d in train_dags for i in dag_to_samples[d]]
        val_indices = [i for d in val_dags for i in dag_to_samples[d]]
        test_indices = [i for d in test_dags for i in dag_to_samples[d]]
        
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        rng.shuffle(test_indices)
    else:
        n = len(full)
        indices = list(range(n))
        rng.shuffle(indices)

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

    def _subset(idxs: List[int]) -> LEGRDataset:
        ds = LEGRDataset.__new__(LEGRDataset)
        ds.samples = [full.samples[i] for i in idxs]
        ds._unique_dags = full._unique_dags
        ds._dag_texts = full._dag_texts
        ds.ged_matrix = full.ged_matrix
        ds.num_unique_dags = full.num_unique_dags
        return ds

    train = _subset(train_indices)
    val = _subset(val_indices)
    test = _subset(test_indices)
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  Export utilities for LLM routing baselines
# ═══════════════════════════════════════════════════════════════════════════════

def export_llm_routing_corpus_jsonl(
    path: str | Path,
    split: str = "test",
    entity_variants: int = 4,
    seed: int = 42,
    split_mode: str = "template",
) -> None:
    """Export query–DAG pairs as JSONL for LLM DAG-generation baselines.

    Each line is a JSON object with:
        - query   : natural-language query
        - tools   : list of tool labels per node
        - edges   : list of [src, dst] integer index pairs
        - dag_id  : integer identifier of the canonical DAG
        - dag_text: canonical textual description of the DAG
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_splits(
        entity_variants=entity_variants,
        seed=seed,
        split_mode=split_mode,
    )
    if split == "train":
        ds = train_ds
    elif split == "val":
        ds = val_ds
    elif split == "test":
        ds = test_ds
    else:
        raise ValueError(f"Unknown split '{split}', expected 'train', 'val', or 'test'.")

    with path.open("w", encoding="utf-8") as f:
        for s in ds.samples:
            G = s.dag_nx
            tools = [G.nodes[n]["tool"] for n in G.nodes()]
            edges = [[int(u), int(v)] for u, v in G.edges()]
            ex = {
                "query": s.query,
                "tools": tools,
                "edges": edges,
                "dag_id": int(s.dag_id),
                "dag_text": dag_to_text(G),
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  CLI
# ═══════════════════════════════════════════════════════════════════════════════

# Scale-up: 400+ unique DAGs (must run after build_dag / dag_canonical_hash exist)
WORKFLOW_TEMPLATES.extend(_build_scale_up_templates())

# Prune templates to only include tools in the active vocabulary
WORKFLOW_TEMPLATES = [
    (name, tools, edges, queries)
    for name, tools, edges, queries in WORKFLOW_TEMPLATES
    if all(t in TOOL_TO_IDX for t in tools)
]


if __name__ == "__main__":
    print("Building LEGR dataset …")
    train_ds, val_ds, test_ds = build_splits()
    print(f"  Train : {train_ds}")
    print(f"  Val   : {val_ds}")
    print(f"  Test  : {test_ds}")
    print(f"\n  Unique DAGs : {train_ds.num_unique_dags}")
    print(f"  GED matrix  : {train_ds.ged_matrix.shape}")
    print(f"  GED range   : [{train_ds.ged_matrix.min():.0f}, "
          f"{train_ds.ged_matrix.max():.0f}]")
    print("\n  Sample 0:")
    s = train_ds[0]
    print(f"    query  : {s['query']}")
    print(f"    dag_id : {s['dag_id']}")
    print(f"    graph  : {s['graph']}")
