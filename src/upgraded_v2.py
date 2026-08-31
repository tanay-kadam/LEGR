"""Generate the leakage-free, non-redundant LEGR v2 graph corpora.

Writes ``upgraded_v2/upgraded_{15,30,45}tools/`` and does not touch
``upgraded/``.  Queries for a DAG use distinct sentence frames rather than
entity-slot clones of one template.  Tool assignment is domain-constrained
so edges are semantically plausible.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import pandas as pd

from utils.graph_utils import (
    _dag_to_text,
    build_dag_from_row,
    classify_topology,
    edges_to_str,
    generate_hard_negatives,
    labeled_dag_hash,
    tools_to_str,
    topology_hash,
)

ToolList = List[str]
EdgeList = List[Tuple[int, int]]
QueryList = List[str]
Recipe = Tuple[str, ToolList, EdgeList, QueryList]


FULL_TOOL_VOCAB: List[str] = [
    "db_read", "db_write", "reset_password", "create_ticket",
    "send_notification", "quarantine_system", "scan_malware",
    "generate_report", "process_refund", "update_subscription",
    "provision_vm", "restart_service", "check_status",
    "escalate_to_human", "log_audit_event",
    "deploy_container", "rollback_deployment", "rotate_api_key",
    "backup_database", "restore_backup", "scale_service",
    "run_pipeline", "approve_access", "revoke_access",
    "transfer_ownership", "schedule_maintenance", "archive_data",
    "enable_feature_flag", "disable_feature_flag", "invalidate_cache",
    "create_dns_record", "renew_certificate", "block_ip_address",
    "unblock_ip_address", "assign_role", "remove_role",
    "trigger_failover", "run_load_test", "snapshot_vm",
    "migrate_database", "tag_resource", "create_alert_rule",
    "acknowledge_alert", "merge_accounts", "export_data",
]

DOMAINS: Dict[str, List[str]] = {
    "security": [
        "scan_malware", "quarantine_system", "block_ip_address",
        "unblock_ip_address", "restart_service", "check_status",
        "create_alert_rule", "acknowledge_alert",
    ],
    "billing": [
        "db_read", "db_write", "process_refund", "update_subscription",
        "merge_accounts", "export_data", "archive_data",
    ],
    "access": [
        "reset_password", "approve_access", "revoke_access", "assign_role",
        "remove_role", "rotate_api_key", "transfer_ownership",
    ],
    "deploy": [
        "run_pipeline", "deploy_container", "rollback_deployment",
        "enable_feature_flag", "disable_feature_flag", "invalidate_cache",
        "scale_service",
    ],
    "infra": [
        "provision_vm", "restart_service", "check_status", "scale_service",
        "snapshot_vm", "tag_resource", "create_dns_record", "renew_certificate",
        "trigger_failover", "run_load_test", "schedule_maintenance",
    ],
    "dataops": [
        "backup_database", "restore_backup", "migrate_database", "db_read",
        "db_write", "archive_data", "export_data", "invalidate_cache",
    ],
    "observability": [
        "check_status", "create_alert_rule", "acknowledge_alert",
        "generate_report", "log_audit_event",
    ],
}

GLUE_TOOLS: Set[str] = {
    "create_ticket", "send_notification", "log_audit_event",
    "escalate_to_human", "generate_report",
}

ENTITY_POOLS: Dict[str, List[str]] = {
    "user": [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Farah", "Gita", "Hassan",
        "Ines", "Jules", "Kai", "Lina", "Mateo", "Nora", "Omar", "Priya",
        "Quinn", "Ravi", "Sofia", "Talia", "Uma", "Victor", "Wendy", "Yusuf",
    ],
    "order": [
        "#10234", "#20891", "#31450", "#42017", "#53698", "#60122",
        "#71003", "#82456", "#93781", "#10011", "#11880", "#12994",
        "#14002", "#15567", "#16890", "#17721",
    ],
    "server": [
        "prod-web-01", "staging-db-02", "payment-api-03", "auth-svc-04",
        "ml-infer-05", "cdn-edge-06", "cache-redis-07", "queue-rabbit-08",
        "billing-api-09", "search-idx-10", "metrics-11", "vpn-gw-12",
        "ops-bastion-13", "etl-job-14", "mail-mx-15",
    ],
    "dept": [
        "Engineering", "Finance", "Marketing", "Legal", "HR", "Operations",
        "Product", "Security", "Support", "Compliance", "Sales", "Data",
    ],
    "ticket": [
        "INC-4021", "INC-7733", "INC-1198", "INC-5560", "INC-8842",
        "INC-2234", "INC-6677", "INC-9911", "INC-3044", "INC-4488",
        "INC-5120", "INC-6401", "INC-7219", "INC-8088", "INC-9150",
    ],
}

# Action phrases.  "full" may name the action; "reduced" prefers symptoms/outcomes.
PHRASES: Dict[str, Dict[str, List[str]]] = {
    "db_read": {
        "full": ["look up {user}'s records", "pull the history on {order}",
                 "fetch {user}'s account details"],
        "reduced": ["see what we have on {user}", "open the file for {order}",
                    "find {user}'s history"],
    },
    "db_write": {
        "full": ["save the change for {user}", "write {user}'s update back",
                 "persist the {dept} config"],
        "reduced": ["make the {user} change stick", "commit what {dept} asked for",
                    "keep {user}'s new details"],
    },
    "reset_password": {
        "full": ["reset {user}'s password", "issue {user} a new login"],
        "reduced": ["let {user} get back into the account",
                    "fix {user}'s lockout"],
    },
    "create_ticket": {
        "full": ["open a ticket for {user}", "file an incident on {server}"],
        "reduced": ["start a tracking case for {user}",
                    "put the {server} issue on the board"],
    },
    "send_notification": {
        "full": ["email {user} about it", "alert the {dept} team"],
        "reduced": ["let {user} know", "give {dept} a heads-up"],
    },
    "quarantine_system": {
        "full": ["isolate {server}", "quarantine {server}"],
        "reduced": ["yank {server} off the network", "take {server} out of prod"],
    },
    "scan_malware": {
        "full": ["scan {server} for malware", "run a threat check on {server}"],
        "reduced": ["see if {server} is compromised",
                    "check {server} for anything nasty"],
    },
    "generate_report": {
        "full": ["build a {dept} report", "compile the {dept} summary"],
        "reduced": ["put the {dept} numbers into a doc",
                    "write up what {dept} needs to see"],
    },
    "process_refund": {
        "full": ["refund {user}", "reverse the charge on {order}"],
        "reduced": ["get {user} their money back", "undo the {order} charge"],
    },
    "update_subscription": {
        "full": ["change {user}'s plan", "update {user}'s subscription"],
        "reduced": ["move {user} to a different tier",
                    "fix {user}'s billing plan"],
    },
    "provision_vm": {
        "full": ["provision a VM for {dept}", "spin up an instance for {dept}"],
        "reduced": ["stand up a box for {dept}", "give {dept} a new machine"],
    },
    "restart_service": {
        "full": ["restart the service on {server}", "reboot {server}"],
        "reduced": ["bounce {server}", "give {server} a kick"],
    },
    "check_status": {
        "full": ["check whether {server} is up", "verify {server}'s health"],
        "reduced": ["see what's going on with {server}",
                    "look at {server}'s health"],
    },
    "escalate_to_human": {
        "full": ["escalate {user}'s case to on-call",
                 "loop in a {dept} manager"],
        "reduced": ["get a human on {user}'s issue",
                    "wake someone for the {server} problem"],
    },
    "log_audit_event": {
        "full": ["log this for compliance", "write an audit entry"],
        "reduced": ["leave a paper trail", "record it for {dept}"],
    },
    "deploy_container": {
        "full": ["deploy the container to {server}",
                 "roll out the {dept} image"],
        "reduced": ["ship the new {dept} build", "get the image onto {server}"],
    },
    "rollback_deployment": {
        "full": ["roll back the deploy on {server}",
                 "revert {server} to the last build"],
        "reduced": ["undo the last release on {server}",
                    "put {server} back on the last good version"],
    },
    "rotate_api_key": {
        "full": ["rotate the API key for {server}",
                 "issue a new secret for {dept}"],
        "reduced": ["cycle {server}'s credentials",
                    "replace {dept}'s leaked secret"],
    },
    "backup_database": {
        "full": ["back up {dept}'s database", "snapshot {server}'s DB"],
        "reduced": ["take a copy of {dept}'s data",
                    "save {server}'s database first"],
    },
    "restore_backup": {
        "full": ["restore {server} from backup",
                 "load {dept}'s last snapshot"],
        "reduced": ["bring {server} back from the last good copy",
                    "recover {dept}'s data"],
    },
    "scale_service": {
        "full": ["scale {server} up", "add replicas for {server}"],
        "reduced": ["give {server} more capacity",
                    "resize {server} for the load"],
    },
    "run_pipeline": {
        "full": ["run the {dept} pipeline", "trigger CI for {server}"],
        "reduced": ["kick off the {dept} build",
                    "start the {server} job"],
    },
    "approve_access": {
        "full": ["approve {user}'s access", "grant {user} rights to {server}"],
        "reduced": ["sign off on {user}'s request",
                    "let {user} into {dept}'s systems"],
    },
    "revoke_access": {
        "full": ["revoke {user}'s access", "cut {user} off {server}"],
        "reduced": ["lock {user} out", "pull {user}'s permissions"],
    },
    "transfer_ownership": {
        "full": ["transfer {server} to {user}",
                 "reassign {server} to {dept}"],
        "reduced": ["hand {server} to {user}",
                    "make {dept} the owner of {server}"],
    },
    "schedule_maintenance": {
        "full": ["schedule maintenance on {server}",
                 "book a window for {server}"],
        "reduced": ["set downtime for {server}",
                    "plan an outage window for {dept}"],
    },
    "archive_data": {
        "full": ["archive {user}'s old records",
                 "move {dept}'s stale data to cold storage"],
        "reduced": ["shelve {user}'s history",
                    "park {dept}'s old files"],
    },
    "enable_feature_flag": {
        "full": ["enable the flag for {dept}",
                 "turn the feature on for {server}"],
        "reduced": ["flip the {dept} toggle on",
                    "light up the new behaviour on {server}"],
    },
    "disable_feature_flag": {
        "full": ["disable the flag for {dept}",
                 "turn the feature off on {server}"],
        "reduced": ["kill the {dept} toggle",
                    "shut the experiment down on {server}"],
    },
    "invalidate_cache": {
        "full": ["flush the cache on {server}",
                 "invalidate {dept}'s cache"],
        "reduced": ["clear stale data on {server}",
                    "force {dept} to see fresh values"],
    },
    "create_dns_record": {
        "full": ["add a DNS record for {server}",
                 "point the {dept} domain at {server}"],
        "reduced": ["make {server} reachable by name",
                    "wire DNS for {dept}"],
    },
    "renew_certificate": {
        "full": ["renew the TLS cert on {server}",
                 "refresh SSL for {dept}"],
        "reduced": ["the {server} cert is expiring — replace it",
                    "keep {dept} HTTPS from going red"],
    },
    "block_ip_address": {
        "full": ["block the attacking IP on {server}",
                 "firewall off the source hitting {server}"],
        "reduced": ["stop that address from reaching {server}",
                    "cut off the flood against {server}"],
    },
    "unblock_ip_address": {
        "full": ["unblock the IP on {server}",
                 "remove the {server} blocklist entry"],
        "reduced": ["let that address through again on {server}",
                    "undo the accidental block on {server}"],
    },
    "assign_role": {
        "full": ["assign {user} the {dept} role",
                 "give {user} editor access"],
        "reduced": ["make {user} an admin for {dept}",
                    "set {user} up with the right permissions"],
    },
    "remove_role": {
        "full": ["remove {user}'s {dept} role",
                 "strip {user}'s editor access"],
        "reduced": ["take {user} out of {dept} admins",
                    "drop {user}'s extra privileges"],
    },
    "trigger_failover": {
        "full": ["fail {server} over to standby",
                 "activate DR for {server}"],
        "reduced": ["switch {server} onto the backup",
                    "move traffic off the sick {server}"],
    },
    "run_load_test": {
        "full": ["load-test {server}", "stress {server} before launch"],
        "reduced": ["see if {server} survives peak traffic",
                    "hammer {server} in staging"],
    },
    "snapshot_vm": {
        "full": ["snapshot {server}", "capture {server}'s disk"],
        "reduced": ["freeze {server}'s current state",
                    "save {server} before we touch it"],
    },
    "migrate_database": {
        "full": ["migrate {dept}'s schema", "apply the DB migration on {server}"],
        "reduced": ["move {dept}'s data onto the new schema",
                    "upgrade {server}'s database shape"],
    },
    "tag_resource": {
        "full": ["tag {server} for {dept}", "label {server} for billing"],
        "reduced": ["mark {server} as {dept}'s",
                    "put a cost label on {server}"],
    },
    "create_alert_rule": {
        "full": ["add an alert on {server}'s CPU",
                 "create a {server} latency rule"],
        "reduced": ["watch {server} and page us if it dies",
                    "set a tripwire on {server}"],
    },
    "acknowledge_alert": {
        "full": ["acknowledge the {server} alert",
                 "ack the firing page for {server}"],
        "reduced": ["mark the {server} page as seen",
                    "tell the system we have the {server} incident"],
    },
    "merge_accounts": {
        "full": ["merge {user}'s duplicate accounts",
                 "combine {user}'s old and new profiles"],
        "reduced": ["fold {user}'s two logins into one",
                    "clean up {user}'s duplicate records"],
    },
    "export_data": {
        "full": ["export {user}'s data", "dump {dept}'s records"],
        "reduced": ["package {user}'s information for download",
                    "give {dept} a copy of the records"],
    },
}

# (family, n_nodes, edges) — generator family names, not classify_topology.
SHAPES: Dict[str, List[Tuple[int, EdgeList]]] = {
    "single_node": [(1, [])],
    "chain_short": [(2, [(0, 1)])],
    "chain_medium": [
        (3, [(0, 1), (1, 2)]),
        (4, [(0, 1), (1, 2), (2, 3)]),
    ],
    "chain_long": [
        (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
    ],
    "fanout": [(3, [(0, 1), (0, 2)])],
    "fanin": [(3, [(0, 2), (1, 2)])],
    "wide_fanout": [(4, [(0, 1), (0, 2), (0, 3)])],
    "diamond": [(4, [(0, 1), (0, 2), (1, 3), (2, 3)])],
    "y_shape": [(4, [(0, 2), (1, 2), (2, 3)])],
    "inverted_y": [(4, [(0, 1), (1, 2), (1, 3)])],
    "hourglass": [(5, [(0, 2), (1, 2), (2, 3), (2, 4)])],
    "w_shape": [(5, [(0, 1), (2, 3), (1, 4), (3, 4)])],
    "asymmetric_fork_join": [
        (5, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]),
    ],
    "repeated_tool": [(4, [(0, 1), (1, 2), (2, 3)])],
    "double_diamond": [
        (7, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6)]),
    ],
}

TRAIN_FAMILIES: Tuple[str, ...] = (
    "chain_short", "chain_medium", "chain_long", "fanout", "fanin",
    "y_shape", "w_shape", "asymmetric_fork_join", "repeated_tool",
    "single_node",
)

TEST_FAMILIES_BASE: Tuple[str, ...] = (
    "diamond", "hourglass", "inverted_y", "wide_fanout",
)

TIER_CONFIG: Dict[int, Dict] = {
    15: {
        "train_dags": 180,
        "dev_dags": 30,
        "test_dags": 48,
        "queries_per_dag": 4,
        "test_families": list(TEST_FAMILIES_BASE),
    },
    30: {
        "train_dags": 220,
        "dev_dags": 36,
        "test_dags": 72,
        "queries_per_dag": 4,
        "test_families": list(TEST_FAMILIES_BASE),
    },
    45: {
        "train_dags": 260,
        "dev_dags": 40,
        "test_dags": 80,
        "queries_per_dag": 4,
        "test_families": list(TEST_FAMILIES_BASE) + ["double_diamond"],
    },
}

OUTPUT_COLUMNS: List[str] = [
    "query", "dag_id", "dag_text", "tools", "edges", "topo_family",
    "source", "split", "query_style",
]


def _hand_recipes() -> List[Recipe]:
    """Hand-written (tools, edges, diverse query frames).  One fill each."""
    return [
        ("implicit_refund", ["db_read", "process_refund"], [(0, 1)], [
            "Customer {user} was charged twice — look the order up and get them their money back.",
            "Can you see why {order} billed again and undo that charge?",
            "Ticket {ticket}: {user} wants a refund; confirm the record first.",
            "{user} is furious about a double charge. Fix billing, don't just apologise.",
        ]),
        ("implicit_quarantine", ["scan_malware", "quarantine_system"], [(0, 1)], [
            "{server} is sending weird pings — check it and pull it off the network if it's dirty.",
            "Security thinks {server} is compromised. Confirm, then isolate it.",
            "Don't leave {server} in prod: inspect it, then yank it.",
            "IR on {server}: prove infection, then cut it off.",
        ]),
        ("implicit_restart", ["check_status", "restart_service"], [(0, 1)], [
            "The {dept} app is timing out. Is {server} even up? Bounce it if not.",
            "Look at {server} before you reboot it — then reboot it.",
            "{server} feels stuck. Confirm it's hung, then kick the process.",
            "Ticket {ticket} says {server} is dead. Verify, then restart.",
        ]),
        ("compound_refund_notify",
         ["db_read", "process_refund", "send_notification"], [(0, 1), (1, 2)], [
            "Look up {order}, reverse the charge, and tell {user} it's done — in that order.",
            "{user} needs their money back and a receipt. Confirm the record first.",
            "Billing glitch on {order}: prove it, credit them, then email.",
            "Don't email {user} until the refund on {order} has actually gone through.",
        ]),
        ("compound_quarantine_log",
         ["scan_malware", "quarantine_system", "log_audit_event"], [(0, 1), (1, 2)], [
            "Sweep {server}, isolate it, and leave a compliance trail.",
            "If {server} is infected, take it down and record the whole thing.",
            "IR writeup needed: inspect {server}, cut it off, document it.",
            "Compliance wants an audit entry only after {server} is isolated.",
        ]),
        ("fanout_refund_ticket",
         ["db_read", "process_refund", "create_ticket"], [(0, 1), (0, 2)], [
            "Pull {order}, then in parallel refund {user} and open a case so we investigate.",
            "Same lookup, two follow-ups: credit the account and file a ticket.",
            "Don't serialise this — after you find {order}, refund and ticket together.",
            "{user} is owed money and we need a tracking case. Look them up once, then split.",
        ]),
        ("fanout_quarantine_notify",
         ["scan_malware", "quarantine_system", "send_notification"], [(0, 1), (0, 2)], [
            "Scan {server}. If it's bad, isolate it and alert {dept} at the same time.",
            "After the {server} check, lock-down and notification should race, not queue.",
            "Don't wait to tell {dept} until quarantine finishes — do both after the scan.",
            "Threat check on {server}, then simultaneously pull it and warn the owners.",
        ]),
        ("fanin_scan_log",
         ["db_read", "scan_malware", "log_audit_event"], [(0, 2), (1, 2)], [
            "Gather {user}'s records and check {server} independently, then one audit entry for both.",
            "Two inputs, one paper trail: data lookup plus {server} inspection, then log.",
            "Don't log until both the DB check and the {server} sweep are done.",
            "Run the records pull and the {server} check as siblings; merge into compliance.",
        ]),
        ("diamond_refund_flow",
         ["db_read", "process_refund", "update_subscription", "send_notification"],
         [(0, 1), (0, 2), (1, 3), (2, 3)], [
            "Look {user} up, then refund and cancel the plan in parallel; one email when both finish.",
            "After the lookup, billing and plan change fan out; notify only at the join.",
            "{user} wants out: confirm the account, handle money and plan together, then one note.",
            "Do not email until both the credit and the plan change for {user} have landed.",
        ]),
        ("diamond_incident_flow",
         ["check_status", "restart_service", "escalate_to_human", "create_ticket"],
         [(0, 1), (0, 2), (1, 3), (2, 3)], [
            "Confirm {server} is down, then bounce it while waking on-call; one ticket at the end.",
            "Health check first. Restart and escalation are siblings. Ticket is the join.",
            "{server} outage: diagnose, try a restart and get a human in parallel, then file once.",
            "Don't open the ticket until both the restart attempt and the escalation exist.",
        ]),
        ("wide_fanout_alert",
         ["scan_malware", "quarantine_system", "send_notification", "log_audit_event"],
         [(0, 1), (0, 2), (0, 3)], [
            "After scanning {server}, isolate, warn {dept}, and record it — all three at once.",
            "One scan, three follow-ups in parallel: lock-down, alert, audit.",
            "Don't sequence the aftermath of the {server} check; fan it out.",
            "Threat found on {server}? Cut it off, page {dept}, and log, concurrently.",
        ]),
        ("inv_y_refund",
         ["db_read", "process_refund", "send_notification", "log_audit_event"],
         [(0, 1), (1, 2), (1, 3)], [
            "Look up {order}, refund, then email {user} and write the audit as siblings.",
            "Chain the lookup into the credit; after that, notify and log in parallel.",
            "The refund is the fork: confirmation to {user} and compliance should split.",
            "Don't treat notify-then-log as a chain — both hang off the refund.",
        ]),
        ("hourglass_ops",
         ["db_read", "check_status", "generate_report", "send_notification", "log_audit_event"],
         [(0, 2), (1, 2), (2, 3), (2, 4)], [
            "Records and {server} health merge into one report, which then fans out to {dept} and audit.",
            "Two sources, a bottleneck doc, then parallel notify and log.",
            "Don't send {dept} anything until the combined report exists; then alert and record together.",
            "Hourglass: gather, compress into a summary, split to stakeholders and compliance.",
        ]),
        ("y_merge_report",
         ["db_read", "scan_malware", "generate_report", "send_notification"],
         [(0, 2), (1, 2), (2, 3)], [
            "Pull data and check {server} as siblings, write one report, then send it to {dept}.",
            "Merge the lookup and the {server} sweep into a doc; email is last.",
            "{dept} only wants the combined write-up, not the raw checks.",
            "Two inputs, one document, then a single send.",
        ]),
        ("single_check_status", ["check_status"], [], [
            "Just tell me if {server} is up. Nothing else.",
            "Quick health look at {server} — that's the whole request.",
            "Is {server} responding? Don't restart anything.",
            "Ticket {ticket} only asks for a status check on {server}.",
        ]),
        ("single_create_ticket", ["create_ticket"], [], [
            "Open a case for the {dept} outage and stop there.",
            "File {ticket} for what {user} reported. No diagnosis.",
            "Someone already investigated — just put {server} on the board.",
            "All I need is a tracking ticket for {user}.",
        ]),
        ("deep_incident_response",
         ["check_status", "scan_malware", "quarantine_system", "restart_service",
          "escalate_to_human", "send_notification", "log_audit_event"],
         [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)], [
            "Full IR on {server}: health, inspect, isolate, bounce, get a human, tell {dept}, document.",
            "Walk {server} through the incident ladder in order — don't skip to notify.",
            "{server} is on fire. Diagnose through isolation and restart before you escalate or email.",
            "Sequential on-call playbook for {server}, ending with {dept} and an audit line.",
        ]),
        ("deploy_diamond",
         ["run_pipeline", "deploy_container", "enable_feature_flag", "send_notification"],
         [(0, 1), (0, 2), (1, 3), (2, 3)], [
            "Kick the {dept} pipeline, then ship the image and flip the flag together; one email after both.",
            "CI first. Deploy and flag are parallel. Notify {dept} only at the join.",
            "Don't tell {dept} the feature is live until both the container and the toggle are done.",
            "Pipeline fans out to rollout and flag; stakeholders hear once.",
        ]),
        ("access_hourglass",
         ["approve_access", "assign_role", "log_audit_event", "send_notification", "create_ticket"],
         [(0, 2), (1, 2), (2, 3), (2, 4)], [
            "Approve {user} and attach the {dept} role independently, audit the pair, then email and ticket in parallel.",
            "Two access actions merge into compliance, which then splits to {user} and the board.",
            "Don't notify {user} until both approval and role exist and are logged.",
            "Access hourglass for {user}: grant + role → audit → message and case.",
        ]),
        ("backup_chain",
         ["schedule_maintenance", "backup_database", "snapshot_vm"],
         [(0, 1), (1, 2)], [
            "Book a window on {server}, dump the DB, then freeze the VM — in that order.",
            "No snapshot until the {dept} backup finished inside the maintenance slot.",
            "Protect {server} three ways, sequenced: window, database copy, disk image.",
            "Ticket {ticket}: maintenance, then backup, then snapshot. Don't reorder.",
        ]),
        ("rollback_fanout",
         ["rollback_deployment", "invalidate_cache", "send_notification"],
         [(0, 1), (0, 2)], [
            "Revert {server}, then flush cache and warn {dept} together.",
            "After the rollback, cache bust and notification are siblings.",
            "Don't wait on {dept}'s email before you clear {server}'s cache — do both.",
            "Bad release on {server}: roll it back, then fan out to cache and comms.",
        ]),
        ("failover_inv_y",
         ["check_status", "trigger_failover", "create_alert_rule", "send_notification"],
         [(0, 1), (1, 2), (1, 3)], [
            "If {server} is sick, fail it over, then set a watch and tell {dept} in parallel.",
            "Health check, then DR; after DR, monitoring and comms split.",
            "Don't page {dept} before failover. After failover, page and add a rule together.",
            "Confirm {server}, move traffic, then fork to alerting and notification.",
        ]),
        ("dns_cert_chain",
         ["create_dns_record", "renew_certificate", "run_load_test"],
         [(0, 1), (1, 2)], [
            "Point DNS at {server}, refresh TLS, then prove it holds load.",
            "No load test until {dept}'s name and cert are both live on {server}.",
            "Launch path for {server}: name, certificate, then stress.",
            "Ticket {ticket} is go-live: DNS, cert, then a traffic rehearsal.",
        ]),
        ("ip_block_diamond",
         ["check_status", "block_ip_address", "create_ticket", "log_audit_event"],
         [(0, 1), (0, 2), (1, 3), (2, 3)], [
            "Confirm the flood on {server}, then firewall the source while opening a case; audit both.",
            "Health check first. Block and ticket fan out. Compliance is the join.",
            "Don't log until both the IP block and the {server} case exist.",
            "{server} is being hit: diagnose, then cut the address and file in parallel, then record.",
        ]),
        ("merge_export_fanin",
         ["merge_accounts", "export_data", "send_notification"],
         [(0, 2), (1, 2)], [
            "Dedup {user}'s profiles and pull a data dump independently, then one email.",
            "Two jobs, one message: merge and export, then tell {user}.",
            "Don't contact {user} until both the merge and the export finished.",
            "{user} asked for a combined account and a copy of their data; notify at the end.",
        ]),
    ]


def vocab_for(tool_count: int) -> List[str]:
    if tool_count not in (15, 30, 45):
        raise ValueError(f"tool_count must be 15, 30, or 45; got {tool_count}")
    return FULL_TOOL_VOCAB[:tool_count]


def recipe_usable(recipe: Recipe, vocab: Sequence[str]) -> bool:
    _name, tools, _edges, _queries = recipe
    allowed = set(vocab)
    return all(t in allowed for t in tools)


def _tool_domain(tool: str) -> Set[str]:
    found = {name for name, members in DOMAINS.items() if tool in members}
    if tool in GLUE_TOOLS:
        found.add("glue")
    return found


def edge_compatible(src: str, dst: str) -> bool:
    if src == dst:
        return True
    if dst in GLUE_TOOLS or src in GLUE_TOOLS:
        return True
    return bool(_tool_domain(src) & _tool_domain(dst) - {"glue"}) or bool(
        _tool_domain(src) & _tool_domain(dst)
    )


def dag_compatible(tools: ToolList, edges: EdgeList) -> bool:
    n = len(tools)
    if any(s >= n or d >= n or s < 0 or d < 0 for s, d in edges):
        return False
    return all(edge_compatible(tools[s], tools[d]) for s, d in edges)


def _layers(n: int, edges: EdgeList) -> List[List[int]]:
    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    parents: Dict[int, List[int]] = {i: [] for i in range(n)}
    for s, d in edges:
        children[s].append(d)
        parents[d].append(s)
    depth = {i: 0 for i in range(n) if not parents[i]}
    changed = True
    while changed:
        changed = False
        for s, d in edges:
            if s in depth:
                cand = depth[s] + 1
                if d not in depth or cand > depth[d]:
                    depth[d] = cand
                    changed = True
    if not depth:
        return [list(range(n))]
    width = max(depth.values()) + 1
    grouped: List[List[int]] = [[] for _ in range(width)]
    for node in range(n):
        grouped[depth.get(node, 0)].append(node)
    return grouped


def _fill(template: str, rng: random.Random) -> str:
    result = template
    for key, pool in ENTITY_POOLS.items():
        tag = "{" + key + "}"
        while tag in result:
            result = result.replace(tag, rng.choice(pool), 1)
    return result


def _contains_tool_name(text: str, vocab: Sequence[str]) -> bool:
    lowered = text.lower()
    for tool in vocab:
        if tool in lowered or tool.replace("_", " ") in lowered:
            return True
    return False


def _entity_values() -> List[str]:
    values: List[str] = []
    for pool in ENTITY_POOLS.values():
        values.extend(pool)
    values.sort(key=len, reverse=True)
    return values


_ENTITY_VALUES = _entity_values()


def normalize_query(text: str) -> str:
    s = text.lower()
    for val in _ENTITY_VALUES:
        s = s.replace(val.lower(), "<e>")
    s = re.sub(r"[^a-z0-9<>\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def too_similar(candidate: str, existing: Sequence[str], threshold: float = 0.62) -> bool:
    norm = normalize_query(candidate)
    for prev in existing:
        prev_n = normalize_query(prev)
        if norm == prev_n:
            return True
        if _token_jaccard(norm, prev_n) >= threshold:
            return True
    return False


def _pick_phrase(tool: str, rng: random.Random, reduced: bool) -> str:
    pack = PHRASES[tool]
    key = "reduced" if reduced and pack["reduced"] else "full"
    return rng.choice(pack[key])


def _join_and(parts: Sequence[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _layer_phrases(
    tools: ToolList,
    layers: List[List[int]],
    rng: random.Random,
    reduced: bool,
) -> List[str]:
    out: List[str] = []
    for layer in layers:
        bits = [_pick_phrase(tools[i], rng, reduced) for i in layer]
        if len(layer) == 1:
            out.append(bits[0])
        else:
            out.append(_join_and(bits))
    return out


FrameFn = Callable[[List[str], random.Random], str]


def _frames_for(family: str, n: int, has_parallel: bool, has_merge: bool) -> List[FrameFn]:
    def goal_first(layers: List[str], rng: random.Random) -> str:
        goal = layers[-1]
        prefix = _join_and(layers[:-1]) if len(layers) > 1 else goal
        if len(layers) == 1:
            return f"That's the whole ask: {goal}."
        return f"End state we need: {goal}. Get there by {prefix}."

    def symptom(layers: List[str], rng: random.Random) -> str:
        openers = [
            "Something's off —",
            "This is blocking people —",
            "Ops is paging about this —",
            "Don't wait on a meeting —",
        ]
        body = " Then ".join(layers)
        return f"{rng.choice(openers)} {body}."

    def ticket(layers: List[str], rng: random.Random) -> str:
        body = "; ".join(layers)
        return f"Ticket {{ticket}}: {body}."

    def forward(layers: List[str], rng: random.Random) -> str:
        if len(layers) == 1:
            return f"Only this: {layers[0]}."
        mid = ", then ".join(layers[:-1])
        return f"{mid}, and finish by {layers[-1]}."

    def question(layers: List[str], rng: random.Random) -> str:
        body = ", then ".join(layers)
        return f"Can someone {body}?"

    def because(layers: List[str], rng: random.Random) -> str:
        if len(layers) < 2:
            return f"Just {layers[0]}."
        return f"We need {layers[-1]} because first {layers[0]}" + (
            f", then {', then '.join(layers[1:-1])}." if len(layers) > 2 else "."
        )

    frames: List[FrameFn] = [goal_first, symptom, ticket, forward, question, because]

    if has_parallel:
        def parallel(layers: List[str], rng: random.Random) -> str:
            return (
                f"Do not serialise the branches. Sequence is: "
                + " → then split → ".join(layers)
                + "."
            ).replace(" → then split → ", ", then in parallel / then ")

        def sibling(layers: List[str], rng: random.Random) -> str:
            return (
                "After the first step, the next actions should happen together, "
                f"not as a chain: {'; '.join(layers)}."
            )

        frames.extend([parallel, sibling])

    if has_merge:
        def merge(layers: List[str], rng: random.Random) -> str:
            return (
                f"Independent work first, one join at the end: {'; '.join(layers)}."
            )

        frames.append(merge)

    if n == 1:
        def singleton(layers: List[str], rng: random.Random) -> str:
            return f"{layers[0]}. Nothing else on this request."

        frames = [singleton, ticket, question, symptom]

    return frames


def synthesize_diverse_queries(
    tools: ToolList,
    edges: EdgeList,
    n: int,
    rng: random.Random,
    vocab: Sequence[str],
    family: str,
    hand_queries: Optional[QueryList] = None,
) -> Tuple[List[str], List[str]]:
    """Return (queries, styles).  At most one fill per frame; no entity clones."""
    queries: List[str] = []
    styles: List[str] = []

    if hand_queries:
        order = list(hand_queries)
        rng.shuffle(order)
        for i, tmpl in enumerate(order):
            if len(queries) >= n:
                break
            filled = _fill(tmpl, rng)
            if _contains_tool_name(filled, vocab):
                continue
            if too_similar(filled, queries):
                continue
            queries.append(_cap(filled))
            styles.append(f"hand_{i}")

    n_nodes = len(tools)
    layers_idx = _layers(n_nodes, edges)
    has_parallel = any(len(layer) > 1 for layer in layers_idx)
    parents = defaultdict(int)
    for _s, d in edges:
        parents[d] += 1
    has_merge = any(v > 1 for v in parents.values())
    frames = _frames_for(family, n_nodes, has_parallel, has_merge)
    rng.shuffle(frames)

    for idx, frame in enumerate(frames):
        if len(queries) >= n:
            break
        reduced = (idx % 2 == 1)
        layer_text = _layer_phrases(tools, layers_idx, rng, reduced)
        raw = frame(layer_text, rng)
        filled = _fill(raw, rng)
        if _contains_tool_name(filled, vocab):
            continue
        if too_similar(filled, queries):
            continue
        queries.append(_cap(filled))
        styles.append(f"{'reduced' if reduced else 'full'}_{frame.__name__}")

    # Last-resort: still refuse entity clones; vary reduced flag and frame index.
    attempts = 0
    while len(queries) < n and attempts < 40:
        attempts += 1
        reduced = attempts % 2 == 0
        layer_text = _layer_phrases(tools, layers_idx, rng, reduced)
        frame = frames[attempts % len(frames)]
        filled = _fill(frame(layer_text, rng), rng)
        if _contains_tool_name(filled, vocab) or too_similar(filled, queries, 0.55):
            continue
        queries.append(_cap(filled))
        styles.append(f"retry_{attempts}")

    return queries, styles


def _cap(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if not text.endswith((".", "?", "!")):
        text += "."
    return text[0].upper() + text[1:]


def _sample_domain_pool(vocab: Sequence[str], rng: random.Random) -> List[str]:
    allowed = set(vocab)
    viable = [
        name for name, members in DOMAINS.items()
        if sum(1 for t in members if t in allowed) >= 2
    ]
    if not viable:
        return list(vocab)
    name = rng.choice(viable)
    pool = [t for t in DOMAINS[name] if t in allowed]
    glue = [t for t in GLUE_TOOLS if t in allowed]
    return pool + glue


def fill_shape(
    n: int,
    edges: EdgeList,
    vocab: Sequence[str],
    rng: random.Random,
    family: str,
    max_tries: int = 80,
) -> Optional[Tuple[ToolList, EdgeList]]:
    allowed = list(vocab)
    for _ in range(max_tries):
        pool = _sample_domain_pool(vocab, rng)
        if family == "repeated_tool":
            if n < 2 or len(pool) < n - 1:
                continue
            base = rng.sample(pool, n - 1)
            tools = base + [base[0]]
        else:
            if len(pool) >= n:
                tools = rng.sample(pool, n)
            else:
                tools = [rng.choice(pool) for _ in range(n)]
        if not dag_compatible(tools, edges):
            continue
        G = build_dag_from_row(tools, edges)
        if not nx.is_directed_acyclic_graph(G):
            continue
        if n > 1 and not edges:
            continue
        return tools, list(edges)
    if n == 1 and allowed:
        return [rng.choice(allowed)], []
    return None


def _hash_dag(tools: ToolList, edges: EdgeList) -> str:
    G = build_dag_from_row(tools, edges)
    return labeled_dag_hash(G)


def collect_recipe_dags(vocab: Sequence[str]) -> List[Dict]:
    found: List[Dict] = []
    seen: Set[str] = set()
    for name, tools, edges, queries in _hand_recipes():
        if not recipe_usable((name, tools, edges, queries), vocab):
            continue
        h = _hash_dag(tools, edges)
        if h in seen:
            continue
        seen.add(h)
        family = classify_topology(edges, len(tools))
        # Prefer the intended structural family when the recipe name encodes it.
        for cand in SHAPES:
            if cand.replace("_", "") in name.replace("_", "") or name.startswith(cand):
                family = cand
                break
        if "diamond" in name:
            family = "diamond"
        elif "hourglass" in name:
            family = "hourglass"
        elif "fanout" in name and "wide" in name:
            family = "wide_fanout"
        elif name.startswith("fanout") or "fanout" in name:
            family = "wide_fanout" if len(edges) >= 3 and classify_topology(edges, len(tools)) == "wide_fanout" else "fanout"
        elif name.startswith("fanin"):
            family = "fanin"
        elif "inv_y" in name or "inverted" in name:
            family = "inverted_y"
        elif name.startswith("y_"):
            family = "y_shape"
        elif name.startswith("single"):
            family = "single_node"
        elif "chain" in name or name.startswith("compound") or name.startswith("implicit") or name.startswith("backup") or name.startswith("dns"):
            family = classify_topology(edges, len(tools))
        found.append({
            "tools": tools,
            "edges": edges,
            "family": family,
            "source": "recipe",
            "hand_queries": queries,
            "dag_hash": h,
            "name": name,
        })
    return found


def generate_typed_dags(
    vocab: Sequence[str],
    families: Sequence[str],
    n_needed: int,
    rng: random.Random,
    seen: Set[str],
) -> List[Dict]:
    results: List[Dict] = []
    attempts = 0
    max_attempts = n_needed * 40
    fam_cycle = list(families)
    while len(results) < n_needed and attempts < max_attempts:
        attempts += 1
        family = fam_cycle[attempts % len(fam_cycle)]
        variants = SHAPES[family]
        n_nodes, edges = rng.choice(variants)
        filled = fill_shape(n_nodes, edges, vocab, rng, family)
        if filled is None:
            continue
        tools, edges = filled
        h = _hash_dag(tools, edges)
        if h in seen:
            continue
        seen.add(h)
        results.append({
            "tools": tools,
            "edges": edges,
            "family": family,
            "source": "typed",
            "hand_queries": None,
            "dag_hash": h,
            "name": f"{family}_{h}",
        })
    return results


def _split_families(
    dags: List[Dict],
    cfg: Dict,
    rng: random.Random,
) -> Dict[str, List[Dict]]:
    test_fams = set(cfg["test_families"])
    test_pool = [d for d in dags if d["family"] in test_fams and d["family"] != "single_node"]
    train_pool = [d for d in dags if d["family"] not in test_fams]
    # single_node never goes to test
    rng.shuffle(test_pool)
    rng.shuffle(train_pool)

    test = test_pool[: cfg["test_dags"]]
    leftover_test = test_pool[cfg["test_dags"]:]
    # leftover held-out families are dropped, not leaked into train
    _ = leftover_test

    need_train = cfg["train_dags"] + cfg["dev_dags"]
    if len(train_pool) < need_train:
        raise RuntimeError(
            f"Only {len(train_pool)} train-family DAGs; need {need_train}. "
            f"Test-family pool was {len(test_pool)}."
        )
    if len(test) < cfg["test_dags"]:
        raise RuntimeError(
            f"Only {len(test)} test-family DAGs; need {cfg['test_dags']}."
        )

    train = train_pool[: cfg["train_dags"]]
    dev = train_pool[cfg["train_dags"]: cfg["train_dags"] + cfg["dev_dags"]]
    return {"train": train, "dev": dev, "test_topology_heldout": test}


def _rows_for_split(
    dags: List[Dict],
    split: str,
    vocab: Sequence[str],
    queries_per: int,
    rng: random.Random,
    dag_id_start: int,
) -> Tuple[List[Dict], int]:
    rows: List[Dict] = []
    dag_id = dag_id_start
    for spec in dags:
        tools = spec["tools"]
        edges = spec["edges"]
        G = build_dag_from_row(tools, edges)
        dag_text = _dag_to_text(G)
        queries, styles = synthesize_diverse_queries(
            tools, edges, queries_per, rng, vocab,
            family=spec["family"],
            hand_queries=spec.get("hand_queries"),
        )
        if not queries:
            continue
        for q, style in zip(queries, styles):
            rows.append({
                "query": q,
                "dag_id": dag_id,
                "dag_text": dag_text,
                "tools": tools_to_str(tools),
                "edges": edges_to_str(edges),
                "topo_family": spec["family"],
                "source": spec["source"],
                "split": split if split != "test_topology_heldout" else "test",
                "query_style": style,
            })
        dag_id += 1
    return rows, dag_id


def _hard_negatives_for(
    test_dags: List[Dict],
    vocab: Sequence[str],
    rng: random.Random,
    query_by_dag_id: Dict[int, str],
    dag_id_by_hash: Dict[str, int],
) -> pd.DataFrame:
    rows: List[Dict] = []
    neg_i = 0
    for spec in test_dags:
        h = spec["dag_hash"]
        dag_id = dag_id_by_hash.get(h)
        if dag_id is None:
            continue
        query = query_by_dag_id.get(dag_id, "")
        tools, edges = spec["tools"], spec["edges"]
        negatives = generate_hard_negatives(tools, edges, rng, list(vocab))
        # Prefer type-compatible swap_tools when the default draw is nonsense.
        for neg in negatives:
            if neg["negative_type"] == "swap_tools" and not dag_compatible(
                neg["neg_tools"], neg["neg_edges"]
            ):
                alt = fill_shape(len(tools), edges, vocab, rng, spec["family"])
                if alt is not None and alt[0] != tools:
                    G = build_dag_from_row(*alt)
                    neg["neg_tools"] = alt[0]
                    neg["neg_edges"] = alt[1]
                    neg["neg_dag_text"] = _dag_to_text(G)
            rows.append({
                "query": query,
                "positive_dag_id": dag_id,
                "negative_dag_id": f"neg_{neg_i}",
                "negative_type": neg["negative_type"],
                "neg_tools": tools_to_str(neg["neg_tools"]),
                "neg_edges": edges_to_str(neg["neg_edges"]),
                "neg_dag_text": neg["neg_dag_text"],
            })
            neg_i += 1
    return pd.DataFrame(rows)


def _redundancy_stats(df: pd.DataFrame) -> Dict[str, float]:
    norms = df["query"].map(normalize_query)
    n = len(norms)
    unique_norm = norms.nunique()
    # within-DAG clone rate: queries that share a normalized form
    clone_rows = 0
    for _dag, group in df.groupby("dag_id"):
        forms = group["query"].map(normalize_query)
        clone_rows += int(forms.duplicated().sum())
    return {
        "rows": float(n),
        "unique_normalized_queries": float(unique_norm),
        "normalized_diversity": round(unique_norm / max(n, 1), 4),
        "within_dag_normalized_clones": float(clone_rows),
    }


def _leakage_report(splits: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    def hashes(df: pd.DataFrame, labeled: bool) -> Set[str]:
        out: Set[str] = set()
        for _, row in df.drop_duplicates("dag_id").iterrows():
            tools = [t for t in str(row["tools"]).split(";") if t]
            edges: EdgeList = []
            raw = str(row["edges"]) if pd.notna(row["edges"]) else ""
            if raw and raw != "nan":
                for part in raw.split(";"):
                    if "->" not in part:
                        continue
                    a, b = part.split("->")
                    edges.append((int(a), int(b)))
            G = build_dag_from_row(tools, edges)
            if labeled:
                out.add(labeled_dag_hash(G))
            else:
                out.add(topology_hash(edges, len(tools)))
        return out

    train, dev, test = splits["train"], splits["dev"], splits["test_topology_heldout"]
    lt, ld, lte = hashes(train, True), hashes(dev, True), hashes(test, True)
    ut, ude, ute = hashes(train, False), hashes(dev, False), hashes(test, False)
    tf = set(train["topo_family"].unique())
    tef = set(test["topo_family"].unique())
    return {
        "labeled_overlap_train_test": len(lt & lte),
        "labeled_overlap_train_dev": len(lt & ld),
        "labeled_overlap_dev_test": len(ld & lte),
        "unlabeled_topology_overlap_train_test": len(ut & ute),
        "family_overlap_train_test": sorted(tf & tef),
        "train_families": sorted(tf),
        "test_families": sorted(tef),
        "dev_families": sorted(set(dev["topo_family"].unique())),
    }


def build_tier(
    tool_count: int,
    output_root: Path,
    seed: int = 42,
) -> Dict[str, object]:
    cfg = TIER_CONFIG[tool_count]
    vocab = vocab_for(tool_count)
    rng = random.Random(seed + tool_count)

    recipes = collect_recipe_dags(vocab)
    seen = {d["dag_hash"] for d in recipes}

    test_fams = cfg["test_families"]
    train_fams = [f for f in TRAIN_FAMILIES if f in SHAPES]

    # Over-generate so splits can be filled after family filtering.
    extra_test = generate_typed_dags(
        vocab, test_fams, cfg["test_dags"] * 3, rng, seen,
    )
    extra_train = generate_typed_dags(
        vocab, train_fams, (cfg["train_dags"] + cfg["dev_dags"]) * 3, rng, seen,
    )
    pool = recipes + extra_test + extra_train
    split_dags = _split_families(pool, cfg, rng)

    out_dir = output_root / f"upgraded_{tool_count}tools"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {}
    dag_id = 0
    dag_id_by_hash: Dict[str, int] = {}
    query_by_dag_id: Dict[int, str] = {}

    for split_name, dags in split_dags.items():
        rows, dag_id = _rows_for_split(
            dags, split_name, vocab, cfg["queries_per_dag"], rng, dag_id,
        )
        df = pd.DataFrame(rows)
        # Map hashes for the dags we actually emitted.
        emitted_ids = sorted(df["dag_id"].unique())
        for spec, eid in zip(dags, emitted_ids):
            dag_id_by_hash[spec["dag_hash"]] = int(eid)
            query_by_dag_id[int(eid)] = str(
                df.loc[df["dag_id"] == eid, "query"].iloc[0]
            )
        frames[split_name] = df
        df.to_csv(out_dir / f"{split_name}.csv", index=False)

    hn = _hard_negatives_for(
        split_dags["test_topology_heldout"], vocab, rng,
        query_by_dag_id, dag_id_by_hash,
    )
    hn.to_csv(out_dir / "hard_negatives.csv", index=False)

    all_df = pd.concat(frames.values(), ignore_index=True)
    leakage = _leakage_report(frames)
    redundancy = {
        name: _redundancy_stats(df) for name, df in frames.items()
    }
    summary = {
        "tool_count": tool_count,
        "seed": seed,
        "rows": {name: int(len(df)) for name, df in frames.items()},
        "unique_dags": {name: int(df["dag_id"].nunique()) for name, df in frames.items()},
        "families": {
            name: sorted(df["topo_family"].unique().tolist())
            for name, df in frames.items()
        },
        "leakage": leakage,
        "redundancy": redundancy,
        "hard_negatives": int(len(hn)),
        "vocab_size": len(vocab),
    }
    (out_dir / "report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _render_report(summary: Dict[str, object]) -> str:
    leak = summary["leakage"]
    lines = [
        f"# upgraded_v2 — {summary['tool_count']} tools",
        "",
        f"- seed: {summary['seed']}",
        f"- vocab: {summary['vocab_size']}",
        f"- hard negatives: {summary['hard_negatives']}",
        "",
        "## Split sizes",
        "",
        "| split | rows | unique DAGs |",
        "|---|---:|---:|",
    ]
    for name in ("train", "dev", "test_topology_heldout"):
        lines.append(
            f"| {name} | {summary['rows'][name]} | {summary['unique_dags'][name]} |"
        )
    lines += [
        "",
        "## Leakage",
        "",
        f"- labelled DAG overlap train∩test: **{leak['labeled_overlap_train_test']}**",
        f"- labelled DAG overlap train∩dev: {leak['labeled_overlap_train_dev']}",
        f"- unlabelled topology overlap train∩test: **{leak['unlabeled_topology_overlap_train_test']}**",
        f"- family overlap train∩test: {leak['family_overlap_train_test'] or 'none'}",
        f"- train families: {', '.join(leak['train_families'])}",
        f"- test families: {', '.join(leak['test_families'])}",
        "",
        "## Query redundancy (normalized entity-invariant)",
        "",
        "| split | rows | unique normalized | diversity | within-DAG clones |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in summary["redundancy"].items():
        lines.append(
            f"| {name} | {int(stats['rows'])} | "
            f"{int(stats['unique_normalized_queries'])} | "
            f"{stats['normalized_diversity']} | "
            f"{int(stats['within_dag_normalized_clones'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_all(
    output_root: Path,
    tool_counts: Sequence[int] = (15, 30, 45),
    seed: int = 42,
) -> Dict[int, Dict[str, object]]:
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for tc in tool_counts:
        summaries[tc] = build_tier(tc, output_root, seed=seed)
    combined = {str(k): v for k, v in summaries.items()}
    (output_root / "combined_report.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8",
    )
    return summaries
