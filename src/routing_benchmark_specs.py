"""
routing_benchmark_specs.py -- Routing-only benchmark specs
==========================================================

Defines benchmark-specific routing vocabularies that do not depend on the DAG
pipeline vocabulary. The 30-tool benchmark is intentionally separate from the
45-tool execution graph tool set.
"""

from __future__ import annotations

import random
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


ROUTING_30_TOOL_NAMES: List[str] = [
    "check_service_status",
    "provision_workspace",
    "reset_user_password",
    "restart_service",
    "create_support_ticket",
    "inspect_security_alerts",
    "revoke_system_access",
    "quarantine_endpoint",
    "escalate_security_incident",
    "log_compliance_event",
    "generate_access_report",
    "assign_access_role",
    "approve_access_request",
    "notify_access_change",
    "update_identity_record",
    "validate_release_readiness",
    "deploy_service_release",
    "enable_feature_flag",
    "rollback_service_release",
    "record_release_note",
    "run_load_test",
    "scale_service_capacity",
    "trigger_failover",
    "schedule_maintenance_window",
    "snapshot_system_state",
    "update_customer_record",
    "authorize_data_export",
    "process_refund",
    "send_customer_notification",
    "archive_customer_data",
]

ROUTING_30_TOOL_NAME_SET: Set[str] = set(ROUTING_30_TOOL_NAMES)

ROUTING_30_TOOL_DESCRIPTIONS: Dict[str, str] = {
    "check_service_status": "Read the current status, health, or responsiveness of a service or system.",
    "provision_workspace": "Provision a new workstation, workspace, or starter environment for a user.",
    "reset_user_password": "Reset a user's password or sign-in credentials so they can regain access.",
    "restart_service": "Restart an existing service or application process without deploying a new release.",
    "create_support_ticket": "Create a support or incident work item for follow-up handling.",
    "inspect_security_alerts": "Review security alerts or suspicious signals to understand what needs attention.",
    "revoke_system_access": "Revoke a user's or system's access to an environment or resource.",
    "quarantine_endpoint": "Isolate a compromised or risky endpoint from the wider environment.",
    "escalate_security_incident": "Escalate a security incident to a human responder or incident lead.",
    "log_compliance_event": "Record an action or incident in a compliance or audit log.",
    "generate_access_report": "Generate a report summarizing current access or permission state.",
    "assign_access_role": "Assign a role or permission set to a user or account.",
    "approve_access_request": "Approve a pending access or permission request.",
    "notify_access_change": "Notify users or teams that an access decision or permission change occurred.",
    "update_identity_record": "Update identity, account-governance, or user profile records.",
    "validate_release_readiness": "Evaluate whether a planned release is ready to go live.",
    "deploy_service_release": "Deploy or roll out a new service release into a live environment.",
    "enable_feature_flag": "Enable a feature flag or rollout gate for a target cohort or environment.",
    "rollback_service_release": "Roll back a service release to a previous version.",
    "record_release_note": "Record durable release notes describing what changed in a rollout.",
    "run_load_test": "Run a performance or load test against a service.",
    "scale_service_capacity": "Adjust service capacity by scaling the running footprint up or down.",
    "trigger_failover": "Trigger failover from a primary system to a standby path.",
    "schedule_maintenance_window": "Schedule a maintenance window or planned downtime slot.",
    "snapshot_system_state": "Capture a point-in-time snapshot or checkpoint of the current system state.",
    "update_customer_record": "Update customer, account, or order details in the system of record.",
    "authorize_data_export": "Authorize a requested export of customer or service data.",
    "process_refund": "Issue a refund or reverse an incorrect customer charge.",
    "send_customer_notification": "Send a customer-facing notification about an account or order change.",
    "archive_customer_data": "Archive older customer data into long-term storage.",
}

ROUTING_30_SEMANTIC_BRANCHES: Dict[str, Dict[str, List[str] | str]] = {
    "Service Support & Systems Administration": {
        "description": (
            "Requests involving service support, workspace setup, account recovery, and routine systems administration tasks."
        ),
        "tools": [
            "check_service_status",
            "provision_workspace",
            "reset_user_password",
            "restart_service",
            "create_support_ticket",
            "schedule_maintenance_window",
        ],
    },
    "Security, Risk & Compliance Response": {
        "description": (
            "Requests involving security review, incident handling, risk containment, and compliance response activities work."
        ),
        "tools": [
            "inspect_security_alerts",
            "revoke_system_access",
            "quarantine_endpoint",
            "escalate_security_incident",
            "log_compliance_event",
            "snapshot_system_state",
        ],
    },
    "Identity, Access & Governance Services": {
        "description": (
            "Requests involving identity records, access permissions, approval workflows, and governance decisions for accounts."
        ),
        "tools": [
            "generate_access_report",
            "assign_access_role",
            "approve_access_request",
            "notify_access_change",
            "update_identity_record",
            "authorize_data_export",
        ],
    },
    "Release & Platform Engineering": {
        "description": (
            "Requests involving release readiness, production rollouts, reliability testing, and platform delivery changes work."
        ),
        "tools": [
            "validate_release_readiness",
            "deploy_service_release",
            "enable_feature_flag",
            "rollback_service_release",
            "record_release_note",
            "run_load_test",
        ],
    },
    "Service Continuity & Customer Operations": {
        "description": (
            "Requests involving customer records, billing actions, service continuity, and ongoing operational maintenance work."
        ),
        "tools": [
            "scale_service_capacity",
            "trigger_failover",
            "update_customer_record",
            "process_refund",
            "send_customer_notification",
            "archive_customer_data",
        ],
    },
}

ROUTING_30_TOOL_BOUND_BRANCHES: Dict[str, Dict[str, List[str] | str]] = {
    "Observe, Review & Assess": {
        "description": (
            "Requests involving observation, inspection, evaluation, and evidence capture before taking follow-on actions elsewhere."
        ),
        "tools": [
            "check_service_status",
            "inspect_security_alerts",
            "generate_access_report",
            "validate_release_readiness",
            "run_load_test",
            "snapshot_system_state",
        ],
    },
    "Access Decisions & Credential Actions": {
        "description": (
            "Requests involving access decisions, credential changes, and account-state updates affecting user permissions directly."
        ),
        "tools": [
            "reset_user_password",
            "revoke_system_access",
            "assign_access_role",
            "approve_access_request",
            "notify_access_change",
            "authorize_data_export",
        ],
    },
    "Live System & Record Changes": {
        "description": (
            "Requests involving provisioning, deployment, scaling, and direct updates to live records or systems."
        ),
        "tools": [
            "provision_workspace",
            "deploy_service_release",
            "enable_feature_flag",
            "scale_service_capacity",
            "update_identity_record",
            "update_customer_record",
        ],
    },
    "Recovery, Reversal & Continuity Actions": {
        "description": (
            "Requests involving containment, rollback, restart, reversal, and continuity actions on existing systems safely."
        ),
        "tools": [
            "restart_service",
            "quarantine_endpoint",
            "rollback_service_release",
            "trigger_failover",
            "schedule_maintenance_window",
            "process_refund",
        ],
    },
    "Case, Escalation & Durable Records": {
        "description": (
            "Requests involving logging, notification, escalation, scheduling, and routing work to people or records."
        ),
        "tools": [
            "create_support_ticket",
            "escalate_security_incident",
            "log_compliance_event",
            "record_release_note",
            "send_customer_notification",
            "archive_customer_data",
        ],
    },
}

ROUTING_30_CUE_WORDS: Dict[str, Set[str]] = {
    "check_service_status": {"status", "health", "responsive", "up", "down", "check", "service", "pulse"},
    "provision_workspace": {"workspace", "workstation", "desktop", "setup", "provision", "environment", "starter"},
    "reset_user_password": {"reset", "password", "credentials", "sign-in", "login", "unlock", "access"},
    "restart_service": {"restart", "reboot", "bounce", "service", "process", "fresh", "start"},
    "create_support_ticket": {"ticket", "case", "support", "incident", "track", "reference", "file"},
    "inspect_security_alerts": {"security", "alerts", "threat", "alert", "inspect", "review", "suspicious"},
    "revoke_system_access": {"revoke", "remove", "cut", "disable", "access", "permissions", "lock"},
    "quarantine_endpoint": {"quarantine", "isolate", "contain", "endpoint", "cut", "off", "compromised"},
    "escalate_security_incident": {"escalate", "incident", "security", "human", "on-call", "lead", "responder"},
    "log_compliance_event": {"audit", "compliance", "log", "record", "entry", "trail", "document"},
    "generate_access_report": {"report", "access", "permissions", "summary", "review", "generate", "breakdown"},
    "assign_access_role": {"assign", "role", "permission", "grant", "access", "attach", "level"},
    "approve_access_request": {"approve", "approval", "authorize", "sign", "off", "request", "access"},
    "notify_access_change": {"notify", "access", "permissions", "update", "tell", "communicate", "decision"},
    "update_identity_record": {"update", "identity", "record", "profile", "governance", "correct", "account"},
    "validate_release_readiness": {"validate", "readiness", "release", "ready", "launch", "go-live", "check"},
    "deploy_service_release": {"deploy", "release", "rollout", "ship", "live", "version", "production"},
    "enable_feature_flag": {"enable", "feature", "flag", "toggle", "activate", "turn", "on"},
    "rollback_service_release": {"rollback", "revert", "undo", "previous", "release", "version", "back"},
    "record_release_note": {"release", "note", "record", "write", "entry", "changelog", "document"},
    "run_load_test": {"load", "test", "stress", "benchmark", "traffic", "performance", "pressure"},
    "scale_service_capacity": {"scale", "capacity", "instances", "replicas", "headroom", "expand", "shrink"},
    "trigger_failover": {"failover", "standby", "secondary", "backup", "switch", "takeover", "dr"},
    "schedule_maintenance_window": {"schedule", "maintenance", "window", "downtime", "planned", "book", "slot"},
    "snapshot_system_state": {"snapshot", "checkpoint", "capture", "state", "point-in-time", "freeze", "copy"},
    "update_customer_record": {"update", "customer", "record", "details", "account", "correct", "order"},
    "authorize_data_export": {"authorize", "export", "data", "approve", "handoff", "extract", "permission"},
    "process_refund": {"refund", "reverse", "charge", "money", "return", "credit", "reimburse"},
    "send_customer_notification": {"notify", "customer", "message", "send", "inform", "update", "email"},
    "archive_customer_data": {"archive", "storage", "historical", "old", "customer", "data", "retention"},
}

ROUTING_30_BASE_QUERY_TEMPLATES: Dict[str, List[str]] = {
    "check_service_status": [
        "Check whether {server} is healthy for {dept}.",
        "Give me the current status of {server} before {dept} starts work.",
        "Confirm that {server} is still responding for {dept}.",
    ],
    "provision_workspace": [
        "Provision a new workspace for {user} in {dept}.",
        "Set up a workstation environment for {user} on the {dept} team.",
        "Create a fresh desktop setup so {user} can start with {dept}.",
    ],
    "reset_user_password": [
        "Reset the password for {user} so they can get back into {server}.",
        "Restore {user}'s credentials for the {dept} systems.",
        "Help {user} sign in again by resetting their access.",
    ],
    "restart_service": [
        "Restart the service running on {server} for {dept}.",
        "Bounce the stuck process on {server} before {dept} uses it.",
        "Bring the service on {server} back up for {dept}.",
    ],
    "create_support_ticket": [
        "Create a support ticket for {user}'s issue with {server}.",
        "Open a help ticket so {dept} can track the problem on {server}.",
        "File a support case for {user} about the {server} outage.",
    ],
    "inspect_security_alerts": [
        "Inspect the security alerts coming from {server}.",
        "Review the threat alerts tied to {server} for {dept}.",
        "Look through the suspicious alert activity on {server}.",
    ],
    "revoke_system_access": [
        "Revoke {user}'s access to {server}.",
        "Remove system access for {user} across the {dept} environment.",
        "Cut off {user}'s permissions before they reach {server}.",
    ],
    "quarantine_endpoint": [
        "Quarantine the endpoint on {server} before it spreads issues.",
        "Isolate {server} from the rest of the {dept} environment.",
        "Contain the compromised endpoint on {server}.",
    ],
    "escalate_security_incident": [
        "Escalate the security incident on {server} to the on-call team.",
        "Hand the threat case for {dept} over to a human responder.",
        "Route the security incident involving {server} to the incident lead.",
    ],
    "log_compliance_event": [
        "Log a compliance event for the action taken on {server}.",
        "Record the security action for audit review in {dept}.",
        "Add an audit entry covering what happened on {server}.",
    ],
    "generate_access_report": [
        "Generate an access report for {dept}.",
        "Compile a permissions report covering {user}'s access.",
        "Produce an access summary for the systems used by {dept}.",
    ],
    "assign_access_role": [
        "Assign the needed role to {user} for {dept}.",
        "Give {user} the correct access role on {server}.",
        "Attach the new permission role for {user}.",
    ],
    "approve_access_request": [
        "Approve {user}'s access request for {server}.",
        "Sign off on the pending permission request from {user}.",
        "Authorize the access change requested by {user} for {dept}.",
    ],
    "notify_access_change": [
        "Notify {user} that their access changed.",
        "Send an access-change update to {dept}.",
        "Tell {user} about the latest permission update.",
    ],
    "update_identity_record": [
        "Update the identity record for {user}.",
        "Correct the account-governance details for {user}.",
        "Modify the identity information tied to {user}'s account.",
    ],
    "validate_release_readiness": [
        "Validate that the next release is ready for {dept}.",
        "Check release readiness before {server} goes live.",
        "Evaluate whether the rollout package is ready for production.",
    ],
    "deploy_service_release": [
        "Deploy the new service release to {server}.",
        "Roll out the latest application release for {dept}.",
        "Ship the next version of the service into production.",
    ],
    "enable_feature_flag": [
        "Enable the feature flag for {dept}.",
        "Turn on the new release flag for {user}'s cohort.",
        "Activate the feature toggle on {server}.",
    ],
    "rollback_service_release": [
        "Roll back the release on {server}.",
        "Revert the service to the previous version for {dept}.",
        "Undo the last rollout and restore the earlier build.",
    ],
    "record_release_note": [
        "Record a release note for the latest rollout.",
        "Log the release note covering what changed for {dept}.",
        "Write the release entry for the deployment on {server}.",
    ],
    "run_load_test": [
        "Run a load test against {server} for {dept}.",
        "Stress-test the service before {dept}'s big event.",
        "Benchmark how {server} behaves under heavy traffic.",
    ],
    "scale_service_capacity": [
        "Scale service capacity on {server} for {dept}.",
        "Increase capacity for the service used by {dept}.",
        "Adjust the number of running instances behind {server}.",
    ],
    "trigger_failover": [
        "Trigger failover for {server}.",
        "Switch {dept}'s service to the standby system.",
        "Move the workload over to the backup environment.",
    ],
    "schedule_maintenance_window": [
        "Schedule a maintenance window for {server}.",
        "Book downtime for {dept}'s service this weekend.",
        "Arrange a planned maintenance slot for {server}.",
    ],
    "snapshot_system_state": [
        "Take a snapshot of the current state on {server}.",
        "Capture a point-in-time system snapshot before changes.",
        "Save the current state of {server} as a checkpoint.",
    ],
    "update_customer_record": [
        "Update the customer record for {user}.",
        "Correct the account details attached to order {order}.",
        "Change the customer data on file for {user}.",
    ],
    "authorize_data_export": [
        "Authorize the data export request for {user}.",
        "Approve a customer data export for {dept}.",
        "Sign off on the export package tied to order {order}.",
    ],
    "process_refund": [
        "Process a refund for order {order}.",
        "Issue money back to {user} for the mistaken charge.",
        "Reverse the charge on {order} for {user}.",
    ],
    "send_customer_notification": [
        "Send a customer notification to {user}.",
        "Notify {user} about the latest account update.",
        "Message the customer about what changed on order {order}.",
    ],
    "archive_customer_data": [
        "Archive the customer data for {user}.",
        "Move the old records tied to order {order} into archive storage.",
        "Store the historical customer information for {dept} off the main system.",
    ],
}

ROUTING_30_INDIRECT_PHRASINGS: Dict[str, List[str]] = {
    "check_service_status": [
        "{dept} needs to know whether {server} is okay right now.",
        "We have not heard from {server} in a bit; can you take a look for {dept}?",
        "Please get a quick pulse on {server} before {dept} depends on it.",
    ],
    "provision_workspace": [
        "{user} is joining {dept} and needs somewhere to work.",
        "{dept} has a new starter and {user} still needs their environment.",
        "{user} cannot begin with {dept} until their work area is ready.",
    ],
    "reset_user_password": [
        "{user} is locked out of the tools used by {dept}.",
        "{user} cannot get back into the system this morning.",
        "Access is failing for {user} and they need to sign in again.",
    ],
    "restart_service": [
        "Something on {server} is stuck and {dept} needs it back.",
        "{dept} says the service on {server} has frozen.",
        "The process on {server} needs a fresh start.",
    ],
    "create_support_ticket": [
        "{user}'s problem with {server} needs a formal tracking record.",
        "{dept} needs a reference number for the issue on {server}.",
        "This problem for {user} should not get lost without an official case.",
    ],
    "inspect_security_alerts": [
        "{dept} is worried about the alerts coming off {server}.",
        "Something in the security feed for {server} needs a closer look.",
        "The alert stream around {server} looks suspicious.",
    ],
    "revoke_system_access": [
        "{user} should not be able to get into {server} anymore.",
        "{dept} needs {user}'s system access shut down immediately.",
        "{user} must be locked out of the environment now.",
    ],
    "quarantine_endpoint": [
        "{server} should be separated from everything else right now.",
        "{dept} cannot trust {server} to stay connected.",
        "The affected endpoint on {server} needs to be cut off.",
    ],
    "escalate_security_incident": [
        "This threat case around {server} needs a real responder now.",
        "{dept} needs human security judgment on what is happening.",
        "Automation is not enough for the incident tied to {server}.",
    ],
    "log_compliance_event": [
        "Compliance will want a record of what happened on {server}.",
        "{dept} needs a traceable audit trail for this action.",
        "There should be an official record of this security step.",
    ],
    "generate_access_report": [
        "{dept} wants a clear summary of who can access what.",
        "We need a readable breakdown of current permissions for {user}.",
        "Leadership asked for a review of access across {dept}.",
    ],
    "assign_access_role": [
        "{user} needs the right level of access for {dept}.",
        "The team cannot move until {user} has the proper role.",
        "{user} is missing the permission level required for {server}.",
    ],
    "approve_access_request": [
        "{user} is waiting for someone to say yes to the access request.",
        "{dept} cannot proceed until the pending permission request is cleared.",
        "The request from {user} still needs formal approval.",
    ],
    "notify_access_change": [
        "{user} still needs to hear that their permissions changed.",
        "{dept} has not been told about the access update yet.",
        "Someone should communicate the access decision to {user}.",
    ],
    "update_identity_record": [
        "{user}'s identity record is outdated and needs correction.",
        "The governance details on {user}'s account no longer look right.",
        "We need the identity file for {user} to reflect the latest state.",
    ],
    "validate_release_readiness": [
        "{dept} wants confidence that the release is safe to launch.",
        "We need to know whether the next rollout is actually ready.",
        "Before going live, someone should confirm the release is in good shape.",
    ],
    "deploy_service_release": [
        "{dept} is waiting for the new version to go live.",
        "The latest build needs to be put into service now.",
        "It is time to roll the new release out.",
    ],
    "enable_feature_flag": [
        "{dept} wants the new behavior made available now.",
        "The rollout switch for the feature needs to be turned on.",
        "{user}'s cohort should start seeing the new capability.",
    ],
    "rollback_service_release": [
        "The latest release needs to be taken back.",
        "{dept} wants the service returned to how it worked before.",
        "We need to undo the most recent rollout right away.",
    ],
    "record_release_note": [
        "{dept} needs a written note about what shipped.",
        "There should be a durable release record for this rollout.",
        "Someone has to capture what changed in the latest release.",
    ],
    "run_load_test": [
        "{dept} wants to know how the service behaves under pressure.",
        "We need proof that {server} can handle a traffic spike.",
        "Before the rush, someone should push {server} hard.",
    ],
    "scale_service_capacity": [
        "{dept} needs more room for incoming traffic.",
        "The service around {server} is running out of headroom.",
        "Capacity has to be adjusted before demand climbs.",
    ],
    "trigger_failover": [
        "{dept} needs the standby system to take over now.",
        "The primary path is not good enough; switch to backup.",
        "We should move the live workload onto the secondary system.",
    ],
    "schedule_maintenance_window": [
        "{dept} needs a planned downtime slot on the calendar.",
        "There should be an official window for work on {server}.",
        "We need to reserve time for planned service work.",
    ],
    "snapshot_system_state": [
        "Before we touch anything, {server} should be frozen in time.",
        "We need a recoverable picture of the system as it is now.",
        "Someone should capture the current state of {server} first.",
    ],
    "update_customer_record": [
        "{user}'s customer record is no longer correct.",
        "The details tied to order {order} need to be fixed.",
        "We need the customer information for {user} brought up to date.",
    ],
    "authorize_data_export": [
        "{user} is waiting for permission to take their data out.",
        "{dept} cannot proceed until the export request is approved.",
        "Someone needs to authorize the requested data handoff.",
    ],
    "process_refund": [
        "{user} was billed incorrectly and needs the money returned.",
        "Order {order} should not have been charged that amount.",
        "We owe {user} money back for the mistake.",
    ],
    "send_customer_notification": [
        "{user} still has not heard about the latest change.",
        "Someone needs to keep the customer informed.",
        "The update tied to order {order} should be communicated.",
    ],
    "archive_customer_data": [
        "The old customer records should be moved out of the active system.",
        "Historical information tied to order {order} belongs in archive storage.",
        "{dept} wants the stale customer data kept but out of the way.",
    ],
}

ROUTING_30_CONFUSABLE_LABEL_MAP: Dict[str, List[str]] = {
    "check_service_status": ["restart_service", "inspect_security_alerts"],
    "provision_workspace": ["create_support_ticket", "deploy_service_release"],
    "reset_user_password": ["create_support_ticket", "approve_access_request"],
    "restart_service": ["check_service_status", "quarantine_endpoint"],
    "create_support_ticket": ["provision_workspace", "escalate_security_incident"],
    "inspect_security_alerts": ["quarantine_endpoint", "check_service_status"],
    "revoke_system_access": ["quarantine_endpoint", "approve_access_request"],
    "quarantine_endpoint": ["inspect_security_alerts", "restart_service"],
    "escalate_security_incident": ["log_compliance_event", "create_support_ticket"],
    "log_compliance_event": ["inspect_security_alerts", "update_identity_record"],
    "generate_access_report": ["update_identity_record", "validate_release_readiness"],
    "assign_access_role": ["approve_access_request", "provision_workspace"],
    "approve_access_request": ["notify_access_change", "reset_user_password"],
    "notify_access_change": ["update_identity_record", "send_customer_notification"],
    "update_identity_record": ["generate_access_report", "log_compliance_event"],
    "validate_release_readiness": ["deploy_service_release", "generate_access_report"],
    "deploy_service_release": ["rollback_service_release", "provision_workspace"],
    "enable_feature_flag": ["record_release_note", "authorize_data_export"],
    "rollback_service_release": ["deploy_service_release", "trigger_failover"],
    "record_release_note": ["validate_release_readiness", "log_compliance_event"],
    "run_load_test": ["scale_service_capacity", "validate_release_readiness"],
    "scale_service_capacity": ["schedule_maintenance_window", "update_customer_record"],
    "trigger_failover": ["snapshot_system_state", "rollback_service_release"],
    "schedule_maintenance_window": ["run_load_test", "notify_access_change"],
    "snapshot_system_state": ["trigger_failover", "record_release_note"],
    "update_customer_record": ["process_refund", "provision_workspace"],
    "authorize_data_export": ["archive_customer_data", "enable_feature_flag"],
    "process_refund": ["update_customer_record", "rollback_service_release"],
    "send_customer_notification": ["archive_customer_data", "notify_access_change"],
    "archive_customer_data": ["authorize_data_export", "snapshot_system_state"],
}

ROUTING_30_CONFUSABLE_QUERY_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "check_service_status": [
        ("Before anyone restarts {server}, tell {dept} whether it is actually healthy.", "restart_service"),
        ("Instead of diving into the security feed, first tell me whether {server} is up for {dept}.", "inspect_security_alerts"),
    ],
    "provision_workspace": [
        ("Open a support case later if needed, but right now {user} needs a workspace for {dept}.", "create_support_ticket"),
        ("Do not roll out an app yet; first get {user}'s workspace ready in {dept}.", "deploy_service_release"),
    ],
    "reset_user_password": [
        ("Track the issue if you want, but {user} mainly needs help getting back into the system.", "create_support_ticket"),
        ("The approval can wait; first fix {user}'s sign-in problem.", "approve_access_request"),
    ],
    "restart_service": [
        ("Do not just report on {server}; bring the service back for {dept}.", "check_service_status"),
        ("If isolation is unnecessary, just get the service on {server} running again.", "quarantine_endpoint"),
    ],
    "create_support_ticket": [
        ("Do not provision anything yet; create a support case for {user}'s problem first.", "provision_workspace"),
        ("If this becomes bigger, someone can escalate it later, but open a tracked case now.", "escalate_security_incident"),
    ],
    "inspect_security_alerts": [
        ("Before isolating anything, review the security alerts tied to {server}.", "quarantine_endpoint"),
        ("I do not just need a health check; I need the security alerts on {server} inspected.", "check_service_status"),
    ],
    "revoke_system_access": [
        ("Do not isolate the machine; remove {user}'s access instead.", "quarantine_endpoint"),
        ("This is not an approval question anymore; shut {user} out of the system.", "approve_access_request"),
    ],
    "quarantine_endpoint": [
        ("The alerts matter, but the immediate need is to isolate {server}.", "inspect_security_alerts"),
        ("Do not just restart {server}; contain it from the rest of the environment.", "restart_service"),
    ],
    "escalate_security_incident": [
        ("We can record it later, but this security incident needs a human responder now.", "log_compliance_event"),
        ("Do not just open a generic case; escalate the security incident immediately.", "create_support_ticket"),
    ],
    "log_compliance_event": [
        ("Investigate if needed, but make sure the action on {server} is logged for compliance.", "inspect_security_alerts"),
        ("This is not just an account update; we need an audit entry for what happened.", "update_identity_record"),
    ],
    "generate_access_report": [
        ("Update the account later; first produce a report of current access for {dept}.", "update_identity_record"),
        ("This is not a release-readiness check; I need an access summary for {dept}.", "validate_release_readiness"),
    ],
    "assign_access_role": [
        ("Approval aside, just give {user} the role they need for {dept}.", "approve_access_request"),
        ("Do not provision a workspace first; assign {user} the correct access role.", "provision_workspace"),
    ],
    "approve_access_request": [
        ("Tell them later if you want, but approve {user}'s pending access request now.", "notify_access_change"),
        ("This is not a password problem; the request itself needs approval.", "reset_user_password"),
    ],
    "notify_access_change": [
        ("Do not just update the record; tell {user} that the access change happened.", "update_identity_record"),
        ("This is not a customer outreach task; notify {user} about the permission change.", "send_customer_notification"),
    ],
    "update_identity_record": [
        ("The report can come after; right now {user}'s identity record needs updating.", "generate_access_report"),
        ("Do not just log the action; correct the identity record itself.", "log_compliance_event"),
    ],
    "validate_release_readiness": [
        ("Do not deploy yet; first confirm the release is ready to go live.", "deploy_service_release"),
        ("This is not an access review; I need a readiness check for the release.", "generate_access_report"),
    ],
    "deploy_service_release": [
        ("Undo it later if needed, but the immediate job is to roll the new release out.", "rollback_service_release"),
        ("This is not workspace setup; ship the service release into production.", "provision_workspace"),
    ],
    "enable_feature_flag": [
        ("Write the note later; for now turn the feature on for {dept}.", "record_release_note"),
        ("This is not about exporting data; activate the rollout flag instead.", "authorize_data_export"),
    ],
    "rollback_service_release": [
        ("This is not a new deployment; take the service back to the previous version.", "deploy_service_release"),
        ("If failover is unnecessary, just undo the release.", "trigger_failover"),
    ],
    "record_release_note": [
        ("The readiness check can wait; what we need now is a release note.", "validate_release_readiness"),
        ("This is not a compliance record; write the note that documents the rollout.", "log_compliance_event"),
    ],
    "run_load_test": [
        ("Do not just adjust capacity yet; first stress the service and see how it behaves.", "scale_service_capacity"),
        ("This is not a launch-readiness question; I need the service load-tested.", "validate_release_readiness"),
    ],
    "scale_service_capacity": [
        ("The maintenance schedule can wait; increase capacity on the service now.", "schedule_maintenance_window"),
        ("This is not a customer-record edit; adjust the service footprint instead.", "update_customer_record"),
    ],
    "trigger_failover": [
        ("Capture the state later; switch to the standby system now.", "snapshot_system_state"),
        ("This is not just a rollout reversal; move the workload to backup.", "rollback_service_release"),
    ],
    "schedule_maintenance_window": [
        ("Test later if you want, but first put a maintenance window on the calendar.", "run_load_test"),
        ("Do not send an access update; schedule downtime for the service instead.", "notify_access_change"),
    ],
    "snapshot_system_state": [
        ("Before failing over, capture the current state of {server}.", "trigger_failover"),
        ("This is not a release note; take a system snapshot first.", "record_release_note"),
    ],
    "update_customer_record": [
        ("The refund can follow, but the customer record itself needs correction first.", "process_refund"),
        ("Do not set up new workspace gear; update the customer information on file.", "provision_workspace"),
    ],
    "authorize_data_export": [
        ("Archiving can wait; first approve the requested data export.", "archive_customer_data"),
        ("This is not a feature-toggle decision; authorize the customer data handoff.", "enable_feature_flag"),
    ],
    "process_refund": [
        ("The record update can come later; return the money for order {order} now.", "update_customer_record"),
        ("Do not roll a release back; reverse the mistaken customer charge instead.", "rollback_service_release"),
    ],
    "send_customer_notification": [
        ("Archive it later if you want, but the customer needs to be told first.", "archive_customer_data"),
        ("This is not an access-update notice; send the customer-facing message now.", "notify_access_change"),
    ],
    "archive_customer_data": [
        ("Approval can wait; move the old customer data into archive storage now.", "authorize_data_export"),
        ("Do not capture a system snapshot; archive the stale customer records instead.", "snapshot_system_state"),
    ],
}

_ENTITY_POOLS: Dict[str, List[str]] = {
    "user": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"],
    "order": ["#10234", "#20891", "#31450", "#42017", "#53698", "#60122"],
    "server": ["prod-web-01", "staging-db-02", "payment-api-03", "auth-svc-04", "ml-infer-05", "cdn-edge-06"],
    "dept": ["Engineering", "Finance", "Marketing", "Legal", "HR"],
    "ticket": ["INC-4021", "INC-7733", "INC-1198", "INC-5560", "INC-8842"],
}


def matches_routing_30_labels(labels: Iterable[str]) -> bool:
    label_set = set(labels)
    return bool(label_set) and label_set <= ROUTING_30_TOOL_NAME_SET


def _fill(template: str, rng: random.Random) -> str:
    rendered = template
    for key, pool in _ENTITY_POOLS.items():
        rendered = rendered.replace("{" + key + "}", rng.choice(pool))
    return rendered


def detect_routing_30_cue_words(query: str, label: str) -> List[str]:
    cues = ROUTING_30_CUE_WORDS.get(label, set())
    tokens = set(re.findall(r"[a-z0-9\-]+", query.lower()))
    return sorted(cues & tokens)


def cue_word_fraction_routing_30(queries: List[str], labels: List[str]) -> float:
    if not queries:
        return 0.0
    hits = sum(
        1 for query, label in zip(queries, labels)
        if detect_routing_30_cue_words(query, label)
    )
    return hits / len(queries)


def generate_routing_30_indirect_query(
    label: str,
    rng: random.Random,
    used: Optional[Set[str]] = None,
) -> str:
    templates = ROUTING_30_INDIRECT_PHRASINGS.get(label, [])
    if not templates:
        return ""
    if used is None:
        used = set()

    for _ in range(60):
        rendered = _fill(rng.choice(templates), rng)
        if rendered not in used:
            used.add(rendered)
            return rendered

    rendered = _fill(rng.choice(templates), rng)
    used.add(rendered)
    return rendered


def generate_routing_30_confusable_query(
    true_label: str,
    rng: random.Random,
    used: Optional[Set[str]] = None,
) -> Tuple[str, str, str]:
    templates = ROUTING_30_CONFUSABLE_QUERY_TEMPLATES.get(true_label, [])
    if not templates:
        return ("", true_label, "")
    if used is None:
        used = set()

    for _ in range(60):
        template, confusable = rng.choice(templates)
        rendered = _fill(template, rng)
        if rendered not in used:
            used.add(rendered)
            return (rendered, true_label, confusable)

    template, confusable = rng.choice(templates)
    rendered = _fill(template, rng)
    return (rendered, true_label, confusable)


def build_routing_30_single_tool_dataset(
    n_per_tool: int = 67,
    seed: int = 42,
    tool_names: Optional[List[str]] = None,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    examples: List[Dict[str, object]] = []
    selected_tools = list(tool_names) if tool_names is not None else list(ROUTING_30_TOOL_NAMES)

    for tool_id, tool in enumerate(selected_tools):
        templates = ROUTING_30_BASE_QUERY_TEMPLATES[tool]
        used: Set[str] = set()
        attempts = 0
        while len(used) < n_per_tool and attempts < max(200, n_per_tool * 20):
            attempts += 1
            rendered = _fill(rng.choice(templates), rng)
            if rendered in used:
                continue
            used.add(rendered)
            examples.append({
                "query": rendered,
                "tool": tool,
                "tool_id": tool_id,
            })

        while len(used) < n_per_tool:
            rendered = _fill(rng.choice(templates), rng)
            rendered = f"{rendered.rstrip('.')} for request {len(used) + 1}."
            if rendered in used:
                continue
            used.add(rendered)
            examples.append({
                "query": rendered,
                "tool": tool,
                "tool_id": tool_id,
            })

    return examples


def build_routing_30_canonical_rows(seed: int = 42) -> List[tuple[str, str, str]]:
    rng = random.Random(seed)
    rows: List[tuple[str, str, str]] = []
    for tool in ROUTING_30_TOOL_NAMES:
        for template in ROUTING_30_BASE_QUERY_TEMPLATES[tool][:2]:
            rows.append((_fill(template, rng), tool, ""))
    return rows
