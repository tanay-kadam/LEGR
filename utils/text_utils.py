"""
text_utils.py -- Text utilities for routing dataset upgrade
============================================================

Provides:
  - Per-label cue-word dictionaries (built from data_synth._TOOL_PHRASES)
  - Cue-word detection and fraction computation
  - Indirect / symptom-based phrasing maps for lexical cue reduction
  - Confusable-label query generators
  - Rule-based paraphrase generation
  - Near-duplicate detection via word-set Jaccard similarity
  - Top-n-gram extraction
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Cue-word dictionaries per label
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "it", "to", "do", "for", "of", "on", "in",
    "and", "or", "that", "this", "with", "from", "at", "by", "be", "as",
    "can", "you", "me", "my", "we", "our", "i", "up", "if", "so", "no",
    "not", "but", "all", "just", "about", "get", "has", "have", "had",
    "was", "were", "been", "its", "them", "their", "there", "than",
    "then", "some", "any", "new", "out", "now", "also", "back",
}

# Maps each label -> set of lexical cue words that are strong indicators.
# Built once; importers can call build_cue_word_dict() to regenerate.
CUE_WORDS: Dict[str, Set[str]] = {
    "query_database": {
        "look", "lookup", "pull", "fetch", "retrieve", "search", "query",
        "database", "db", "record", "records", "data", "details", "info",
        "check", "grab", "dig", "read", "profile",
    },
    "update_database": {
        "update", "save", "persist", "write", "commit", "push", "store",
        "change", "modify", "mark", "database", "db", "record", "records",
    },
    "reset_password": {
        "reset", "password", "credential", "credentials", "login", "log",
        "nuke", "rotate", "auth", "force",
    },
    "create_ticket": {
        "ticket", "open", "file", "jira", "incident", "bug", "support",
        "case", "log", "create", "tracking",
    },
    "send_notification": {
        "send", "email", "notify", "notification", "alert", "blast",
        "ping", "shoot", "fire", "message",
    },
    "quarantine_system": {
        "quarantine", "isolate", "yank", "lock", "disconnect", "pull",
        "network", "production",
    },
    "scan_malware": {
        "scan", "malware", "sweep", "threat", "threats", "vulnerability",
        "vulnerabilities", "antivirus", "ioc", "iocs", "infection",
        "infections", "security",
    },
    "generate_report": {
        "report", "generate", "compile", "summary", "analytics",
        "compliance", "pdf", "document", "build", "produce",
    },
    "process_refund": {
        "refund", "credit", "reverse", "charge", "payout", "money",
        "reimburse", "overcharged",
    },
    "update_subscription": {
        "subscription", "upgrade", "downgrade", "plan", "billing",
        "tier", "cancel", "modify", "adjust",
    },
    "provision_vm": {
        "provision", "vm", "virtual", "machine", "instance", "server",
        "deploy", "spin", "stand", "launch", "box",
    },
    "restart_service": {
        "restart", "reboot", "bounce", "kick", "cycle", "service",
        "process",
    },
    "check_status": {
        "status", "health", "check", "verify", "assess", "diagnose",
        "ping", "monitor", "uptime", "responsive", "up", "down",
    },
    "escalate_to_human": {
        "escalate", "human", "manager", "agent", "tier", "engineer",
        "on-call", "oncall", "involve", "hand",
    },
    "log_audit_event": {
        "audit", "log", "compliance", "record", "trail", "event",
        "entry", "document",
    },
    # Tier 2
    "deploy_container": {
        "deploy", "container", "docker", "ship", "image", "push",
        "roll", "launch", "containerised",
    },
    "rollback_deployment": {
        "rollback", "revert", "undo", "previous", "back", "prior",
        "deployment", "release",
    },
    "rotate_api_key": {
        "rotate", "api", "key", "secret", "token", "credential",
        "cycle", "refresh", "swap",
    },
    "backup_database": {
        "backup", "snapshot", "dump", "save", "database", "db",
        "copy", "preserve",
    },
    "restore_backup": {
        "restore", "recover", "backup", "snapshot", "bring",
        "load", "database",
    },
    "scale_service": {
        "scale", "capacity", "replica", "replicas", "resize",
        "autoscale", "instances", "bump",
    },
    "run_pipeline": {
        "pipeline", "ci", "cd", "build", "trigger", "run",
        "job", "workflow",
    },
    "approve_access": {
        "approve", "grant", "authorize", "permission", "access",
        "sign", "green-light",
    },
    "revoke_access": {
        "revoke", "remove", "cut", "strip", "disable", "access",
        "permission", "block",
    },
    "transfer_ownership": {
        "transfer", "ownership", "hand", "reassign", "move",
        "pass", "owner",
    },
    "schedule_maintenance": {
        "schedule", "maintenance", "downtime", "window", "plan",
        "book", "arrange",
    },
    "archive_data": {
        "archive", "cold", "storage", "old", "historical",
        "shelve", "stale",
    },
    "enable_feature_flag": {
        "enable", "feature", "flag", "toggle", "activate",
        "turn", "flip", "switch",
    },
    "disable_feature_flag": {
        "disable", "feature", "flag", "toggle", "deactivate",
        "turn", "kill", "switch",
    },
    "invalidate_cache": {
        "cache", "flush", "invalidate", "bust", "purge",
        "clear", "wipe",
    },
    # Tier 3
    "create_dns_record": {
        "dns", "domain", "record", "point", "subdomain",
        "cname", "route", "resolve",
    },
    "renew_certificate": {
        "certificate", "cert", "tls", "ssl", "renew",
        "expire", "refresh", "reissue",
    },
    "block_ip_address": {
        "block", "ip", "firewall", "deny", "blacklist",
        "ban", "drop", "address",
    },
    "unblock_ip_address": {
        "unblock", "ip", "whitelist", "allow", "remove",
        "undo", "address", "re-allow",
    },
    "assign_role": {
        "assign", "role", "permission", "grant", "admin",
        "editor", "attach", "add",
    },
    "remove_role": {
        "remove", "role", "permission", "strip", "revoke",
        "detach", "take", "pull",
    },
    "trigger_failover": {
        "failover", "standby", "secondary", "dr", "switch",
        "activate", "replica", "disaster",
    },
    "run_load_test": {
        "load", "test", "stress", "benchmark", "performance",
        "simulate", "traffic", "hammer",
    },
    "snapshot_vm": {
        "snapshot", "capture", "freeze", "image", "checkpoint",
        "vm", "disk", "state",
    },
    "migrate_database": {
        "migrate", "migration", "schema", "upgrade", "database",
        "apply", "evolve", "transform",
    },
    "tag_resource": {
        "tag", "label", "annotate", "mark", "resource",
        "cost", "billing", "tracking",
    },
    "create_alert_rule": {
        "alert", "rule", "monitor", "threshold", "trigger",
        "notification", "configure", "define",
    },
    "acknowledge_alert": {
        "acknowledge", "ack", "alert", "silence", "seen",
        "confirm", "accept", "dismiss",
    },
    "merge_accounts": {
        "merge", "combine", "unify", "consolidate", "duplicate",
        "join", "fold", "accounts",
    },
    "export_data": {
        "export", "download", "dump", "extract", "package",
        "gdpr", "data", "pull",
    },
}


def detect_cue_words(query: str, label: str) -> List[str]:
    """Return cue words from *label*'s dictionary that appear in *query*."""
    cues = CUE_WORDS.get(label, set())
    tokens = set(re.findall(r"[a-z0-9\-]+", query.lower()))
    return sorted(cues & tokens)


def cue_word_fraction(queries: List[str], labels: List[str]) -> float:
    """Fraction of queries that contain at least one label-specific cue word."""
    if not queries:
        return 0.0
    hits = sum(1 for q, l in zip(queries, labels) if detect_cue_words(q, l))
    return hits / len(queries)


# ─────────────────────────────────────────────────────────────────────────────
#  Indirect phrasing maps for lexical cue reduction
# ─────────────────────────────────────────────────────────────────────────────

INDIRECT_PHRASINGS: Dict[str, List[str]] = {
    "query_database": [
        "I need the details on {user}.",
        "What do we have on file for {user}?",
        "Can someone get me {user}'s information?",
        "I'm trying to find out about order {order}.",
        "Tell me everything we know about {user}.",
        "Who has the records related to {order}?",
        "{user} called in -- what's their history with us?",
        "We got a question about {order}, need context.",
        "Can you find what's associated with {user}'s account?",
        "I want to see the full picture for {order}.",
        "What's on record for {user}?",
        "Show me what we've got for {order}.",
    ],
    "update_database": [
        "{user}'s information has changed and needs correcting.",
        "We got new details for {user} that need to go into the system.",
        "The records for {user} are stale -- please fix them.",
        "{user}'s profile doesn't reflect the latest info.",
        "There are incorrect entries for {user} that need patching.",
        "The data for {user} is outdated.",
        "Please correct {user}'s address in our system.",
        "We need to amend the entry for {user}.",
        "{user}'s file needs to reflect the changes from today.",
        "Some fields for {user} are wrong -- please sort it out.",
    ],
    "reset_password": [
        "{user} can't log in anymore.",
        "{user} is locked out of their account.",
        "{user}'s credentials are expired.",
        "{user} forgot how to access the system.",
        "{user} keeps getting 'access denied'.",
        "{user} can't authenticate since this morning.",
        "{user} says the system won't let them in.",
        "We need to restore {user}'s access.",
        "{user} has been unable to sign in all day.",
        "{user} is complaining about being locked out.",
    ],
    "create_ticket": [
        "We need to track this issue with {user}.",
        "This should be documented somewhere official.",
        "Can we get a formal record of {user}'s problem?",
        "We need a paper trail for the {server} issue.",
        "This needs to be logged for follow-up.",
        "{user} reported something that needs tracking.",
        "Make sure this doesn't slip through the cracks.",
        "We should have a reference number for {user}'s issue.",
        "Someone should formalize this complaint from {user}.",
        "This needs an official entry in our tracking system.",
    ],
    "send_notification": [
        "{user} needs to know what happened.",
        "Has anyone told {dept} about this?",
        "We should let {user} know about the situation.",
        "Someone needs to loop {user} in.",
        "Make sure {dept} hears about this.",
        "{user} is waiting for an update.",
        "Keep {user} in the loop.",
        "{dept} hasn't been informed yet.",
        "Time to reach out to {user} about the outcome.",
        "Don't forget to tell {user} the news.",
    ],
    "quarantine_system": [
        "{server} is acting weird -- get it away from everything else.",
        "Something on {server} looks compromised.",
        "{server} shouldn't be connected to the rest of the network right now.",
        "We can't trust {server} anymore -- take it offline.",
        "{server} is a risk to the rest of the infrastructure.",
        "That box is behaving erratically -- cut it off.",
        "Remove {server} before it spreads to other hosts.",
        "We need {server} separated from production immediately.",
        "{server} is showing signs of compromise.",
        "Get {server} out of the environment until we know more.",
    ],
    "scan_malware": [
        "Something weird is going on with {server}.",
        "We're seeing unusual activity on {server}.",
        "{server} might have been breached.",
        "There could be something bad running on {server}.",
        "Is {server} clean? We're not sure.",
        "We need to make sure {server} is safe.",
        "There are suspicious processes on {server}.",
        "Can we verify that {server} isn't compromised?",
        "{server} has been flagged for unusual behavior.",
        "We should look into what's happening on {server}.",
    ],
    "generate_report": [
        "{dept} wants a summary of what happened.",
        "We need documentation for the stakeholders.",
        "Management is asking for numbers.",
        "Can someone put together the overview for {dept}?",
        "We owe {dept} a write-up.",
        "The quarterly figures need packaging.",
        "Stakeholders want to see the data in a digestible format.",
        "It's time for the periodic review document.",
        "We need to present the findings to leadership.",
        "Someone needs to summarize the metrics for {dept}.",
    ],
    "process_refund": [
        "{user} was charged incorrectly.",
        "{user} shouldn't have been billed for that.",
        "We owe {user} money.",
        "{user} wants their money returned.",
        "There was a billing mistake with {order}.",
        "{user} paid twice and isn't happy.",
        "The charge on {order} was a mistake.",
        "{user} says they didn't authorize this charge.",
        "We need to make {user} financially whole again.",
        "The amount on {order} needs to be reversed.",
    ],
    "update_subscription": [
        "{user} wants to change their service level.",
        "{user}'s current plan doesn't match their needs.",
        "{user} needs a different tier.",
        "The contract terms for {user} have changed.",
        "{user} asked to be moved to a different package.",
        "{user}'s service needs don't match what they're paying for.",
        "{user} wants fewer features and a lower cost.",
        "{user} is requesting more capacity.",
        "{user}'s agreement needs to reflect the new pricing.",
        "Please move {user} to the appropriate package.",
    ],
    "provision_vm": [
        "{dept} needs more compute capacity.",
        "We're short on infrastructure for {dept}.",
        "{dept} is asking for a new environment.",
        "We need another host for the {dept} workload.",
        "{dept}'s current resources aren't enough.",
        "Can we get some fresh infrastructure for {dept}?",
        "The {dept} team needs their own isolated environment.",
        "We're running out of capacity for {dept}.",
        "{dept} put in a request for new hardware resources.",
        "Someone needs to set up a new environment for {dept}.",
    ],
    "restart_service": [
        "{server} has been unresponsive.",
        "The process on {server} is hung.",
        "{server} is stuck and not serving requests.",
        "Users are complaining that {server} is down.",
        "Something on {server} stopped working.",
        "{server} hasn't responded in a while.",
        "The application on {server} is frozen.",
        "{server} needs a fresh start.",
        "Requests to {server} are timing out.",
        "The service on {server} is in a bad state.",
    ],
    "check_status": [
        "Is everything okay with {server}?",
        "What's going on with {server}?",
        "Can someone tell me if {server} is alive?",
        "We haven't heard from {server} in a while.",
        "Is {server} still working?",
        "I'd like a pulse on {server}.",
        "Any idea what state {server} is in?",
        "We need visibility into {server}'s condition.",
        "Can you confirm {server} is operational?",
        "Give me a quick look at how {server} is doing.",
    ],
    "escalate_to_human": [
        "This is beyond what automation can handle.",
        "{user}'s situation needs a real person.",
        "We need someone with judgment on this.",
        "The automated response isn't going to cut it for {user}.",
        "This requires human decision-making.",
        "Can we get a senior person to look at {user}'s case?",
        "This one needs a pair of human eyes.",
        "Automation isn't going to resolve {user}'s issue.",
        "{user}'s case is too nuanced for a bot.",
        "We need human intervention for {user}.",
    ],
    "log_audit_event": [
        "We need a record of what just happened.",
        "This action needs to be documented for regulators.",
        "Make sure there's a trace of what was done.",
        "Compliance will want to see this was recorded.",
        "We can't skip the paper trail on this one.",
        "This needs to be in the official record.",
        "The regulator will ask if we documented this.",
        "Don't forget to capture this for the auditors.",
        "We need proof that this action took place.",
        "This should be traceable later if anyone asks.",
    ],
    # Tier 2
    "deploy_container": [
        "The new version is ready to go live.",
        "{dept} has been waiting for this release.",
        "The build passed all tests and should be shipped.",
        "It's time to get the latest code running in production.",
        "The team needs the new version out there.",
        "Everything's green — let's get it deployed.",
    ],
    "rollback_deployment": [
        "The latest release is causing problems.",
        "Something broke after the last deploy.",
        "Users are complaining since the update went out.",
        "We need to go back to what was working before.",
        "The new version introduced a regression.",
        "Things were fine until the latest push.",
    ],
    "rotate_api_key": [
        "The current credentials may have been exposed.",
        "It's been too long since we changed the secret.",
        "Someone may have leaked the API token.",
        "Policy says we need fresh credentials.",
        "The key for {server} hasn't been rotated in months.",
        "We should swap out the secret just to be safe.",
    ],
    "backup_database": [
        "We're about to do something risky to the data.",
        "Let's make sure we have a safety net first.",
        "Before the migration, we need a copy of everything.",
        "We should preserve the current state of the database.",
        "If anything goes wrong, we'll need a way back.",
        "The data needs to be protected before we proceed.",
    ],
    "restore_backup": [
        "The migration went sideways and we lost data.",
        "We need to get back to where we were before.",
        "The database is in a bad state.",
        "Something corrupted the data — we need the old version.",
        "Can we undo what happened to the database?",
        "The system needs to be returned to its previous state.",
    ],
    "scale_service": [
        "The service is getting hammered with traffic.",
        "Response times are climbing — we need more capacity.",
        "We're way over-provisioned and wasting money.",
        "{server} can't handle the current load.",
        "We need to adjust the resources for {server}.",
        "Traffic patterns changed and we need to resize.",
    ],
    "run_pipeline": [
        "The code is merged and ready to be built.",
        "{dept} pushed a fix and needs it deployed.",
        "We need to get the latest changes through CI.",
        "The build hasn't run since the last commit.",
        "Can we get the automated tests and deploy going?",
        "The release is blocked until the pipeline finishes.",
    ],
    "approve_access": [
        "{user} needs to get into the system for their job.",
        "There's a pending request that's blocking {user}.",
        "{user} can't do their work without the right permissions.",
        "The request has been sitting there for days.",
        "{user}'s team lead already confirmed they need this.",
        "Can someone authorize {user}'s pending request?",
    ],
    "revoke_access": [
        "{user} no longer needs access to this system.",
        "{user} left the company and still has permissions.",
        "There's a security concern with {user}'s access level.",
        "{user} shouldn't have access to {server} anymore.",
        "The contractor's engagement ended — clean up their access.",
        "{user}'s role changed and their old permissions are a risk.",
    ],
    "transfer_ownership": [
        "The team responsible for {server} is changing.",
        "{dept} is taking over this project.",
        "The current owner is leaving and someone needs to take over.",
        "This resource needs a new responsible party.",
        "Responsibility for {server} is shifting to another group.",
        "{dept} should be the owner going forward.",
    ],
    "schedule_maintenance": [
        "{server} needs some planned downtime soon.",
        "We can't keep putting off the maintenance.",
        "The team needs a window to do upgrades.",
        "{server} has patches that require a restart.",
        "We need to coordinate a time to take {server} offline.",
        "There are pending updates that require downtime.",
    ],
    "archive_data": [
        "The old records are slowing down queries.",
        "We don't need this data in the active database anymore.",
        "Retention policy says we should move this to cold storage.",
        "The database is bloated with historical entries.",
        "These records haven't been accessed in over a year.",
        "We need to clean up {dept}'s outdated data.",
    ],
    "enable_feature_flag": [
        "The new feature is ready for a limited rollout.",
        "{dept} wants to start testing the new flow.",
        "We're ready to expose this to a subset of users.",
        "The product team wants the new experience turned on.",
        "QA signed off — time to make it available.",
        "Let's start the gradual rollout for {dept}.",
    ],
    "disable_feature_flag": [
        "The new feature is causing errors for some users.",
        "We need to pull back the experimental flow.",
        "The rollout isn't going well — shut it down.",
        "Product asked us to turn off the new experience.",
        "The feature is creating more problems than it solves.",
        "Let's kill the toggle until we fix the bugs.",
    ],
    "invalidate_cache": [
        "Users are seeing outdated information.",
        "The content hasn't updated even though the data changed.",
        "Something is stale and it's confusing users.",
        "The cached version doesn't match what's in the database.",
        "We pushed an update but the old version is still showing.",
        "The data is out of sync — probably a caching issue.",
    ],
    # Tier 3
    "create_dns_record": [
        "The new service isn't reachable by name yet.",
        "Users can't find {server} because there's no DNS entry.",
        "The subdomain doesn't resolve to anything.",
        "We need the domain to point to the new load balancer.",
        "The endpoint exists but nobody can reach it by URL.",
        "Traffic isn't being routed to the new service.",
    ],
    "renew_certificate": [
        "Browsers are showing a security warning for {server}.",
        "The HTTPS connection to {server} is being rejected.",
        "{server}'s certificate is about to expire.",
        "Users are getting 'connection not secure' errors.",
        "The secure connection to {server} stopped working.",
        "We're getting certificate expiration warnings.",
    ],
    "block_ip_address": [
        "We're being attacked from a specific address.",
        "There's suspicious traffic flooding {server}.",
        "An unknown source keeps hammering {server}.",
        "We need to stop the malicious traffic immediately.",
        "Someone is brute-forcing the login from one IP.",
        "The firewall needs to deny this traffic source.",
    ],
    "unblock_ip_address": [
        "A legitimate partner can't reach {server}.",
        "We accidentally blocked a customer's IP.",
        "The blocked address turned out to be harmless.",
        "A vendor is complaining they can't connect.",
        "The false positive is blocking real users.",
        "We need to restore connectivity for that address.",
    ],
    "assign_role": [
        "{user} needs higher permissions for their new position.",
        "{user} can't do their work without the right role.",
        "The new hire needs to be set up with proper access.",
        "{user}'s responsibilities changed and they need more access.",
        "{user} needs the ability to make changes in {dept}.",
        "Someone on {dept} needs elevated permissions.",
    ],
    "remove_role": [
        "{user} shouldn't have admin access anymore.",
        "{user}'s role changed and they have too many permissions.",
        "The intern still has write access they shouldn't have.",
        "{user} moved teams and their old role is a security risk.",
        "We need to clean up {user}'s excessive permissions.",
        "{user} no longer needs elevated access.",
    ],
    "trigger_failover": [
        "The primary system is completely unresponsive.",
        "{server} is down and we have a standby ready.",
        "We've lost the primary and need to switch over.",
        "The main database isn't accepting connections.",
        "We need to activate the disaster recovery plan.",
        "The primary is toast — time to use the secondary.",
    ],
    "run_load_test": [
        "We're launching next week and need to know if it can handle the load.",
        "How many concurrent users can {server} handle?",
        "We need to verify {server} won't fall over under peak traffic.",
        "Before the sale event, we should test our capacity.",
        "Can {server} handle 10x the normal traffic?",
        "We need performance numbers before going live.",
    ],
    "snapshot_vm": [
        "We're about to make risky changes to {server}.",
        "Let's save the current state before we do anything.",
        "We need a restore point for {server}.",
        "Before the upgrade, we should have a way to roll back.",
        "Create a safety checkpoint for {server}.",
        "We need to preserve {server}'s state before proceeding.",
    ],
    "migrate_database": [
        "The schema needs to match the new application version.",
        "There are pending database changes for the release.",
        "The data model changed and the DB needs updating.",
        "The new code requires a schema change in production.",
        "We need to evolve the database to support the new features.",
        "The tables need restructuring for the new version.",
    ],
    "tag_resource": [
        "Finance can't track costs without proper labels.",
        "We don't know which project {server} belongs to.",
        "The billing report can't attribute this to a team.",
        "Cloud costs are unattributed — we need labels.",
        "The resources need proper project identifiers.",
        "We can't do cost allocation without tags.",
    ],
    "create_alert_rule": [
        "We have no monitoring on the new service.",
        "Nobody will know if {server} goes down.",
        "We need to be notified when latency spikes.",
        "There's no warning system for disk space.",
        "We should get paged if error rates go up.",
        "The new endpoint has zero observability.",
    ],
    "acknowledge_alert": [
        "The alert keeps firing but we're already working on it.",
        "We know about the issue — stop the noise.",
        "Someone needs to confirm we've seen the warning.",
        "The pager is going off but we're aware.",
        "Mark the incident as acknowledged so it stops escalating.",
        "We're on it — just need to silence the notification.",
    ],
    "merge_accounts": [
        "{user} accidentally created a second account.",
        "There are duplicate records for the same customer.",
        "{user} has two profiles and it's causing confusion.",
        "The customer appears twice in the system.",
        "Data is split across two accounts for the same person.",
        "We need to consolidate {user}'s duplicate entries.",
    ],
    "export_data": [
        "{user} wants a copy of everything we have on them.",
        "We received a data portability request from {user}.",
        "{dept} needs the data in a format they can analyze externally.",
        "{user} is exercising their right to data access.",
        "The auditors need an extract of {dept}'s records.",
        "We need to provide {user} with their personal data.",
    ],
}

# Entity pools (mirrored from data_synth for standalone use)
_ENTITY_POOLS: Dict[str, List[str]] = {
    "user":   ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"],
    "order":  ["#10234", "#20891", "#31450", "#42017", "#53698", "#60122"],
    "server": ["prod-web-01", "staging-db-02", "payment-api-03",
               "auth-svc-04", "ml-infer-05", "cdn-edge-06"],
    "dept":   ["Engineering", "Finance", "Marketing", "Legal", "HR"],
    "ticket": ["INC-4021", "INC-7733", "INC-1198", "INC-5560", "INC-8842"],
}


def fill_entities(template: str, rng: random.Random) -> str:
    """Replace {entity} placeholders with random pool values."""
    result = template
    for key, pool in _ENTITY_POOLS.items():
        tag = "{" + key + "}"
        while tag in result:
            result = result.replace(tag, rng.choice(pool), 1)
    return result


def generate_indirect_query(
    label: str,
    rng: random.Random,
    used: Optional[Set[str]] = None,
) -> str:
    """Pick an indirect phrasing template for *label*, fill entities.

    Tracks fully-rendered outputs in *used* to avoid exact duplicates.
    """
    templates = INDIRECT_PHRASINGS.get(label, [])
    if not templates:
        return ""
    if used is None:
        used = set()

    for _ in range(30):
        tpl = rng.choice(templates)
        rendered = fill_entities(tpl, rng)
        if rendered not in used:
            used.add(rendered)
            return rendered

    # Last resort: return a novel fill even if template was reused
    rendered = fill_entities(rng.choice(templates), rng)
    used.add(rendered)
    return rendered


# ─────────────────────────────────────────────────────────────────────────────
#  Confusable-intent query generation
# ─────────────────────────────────────────────────────────────────────────────

CONFUSABLE_LABEL_MAP: Dict[str, List[str]] = {
    "query_database":       ["check_status", "generate_report"],
    "update_database":      ["reset_password", "log_audit_event"],
    "reset_password":       ["update_database", "quarantine_system"],
    "create_ticket":        ["log_audit_event", "send_notification"],
    "send_notification":    ["escalate_to_human", "create_ticket"],
    "quarantine_system":    ["restart_service", "scan_malware"],
    "scan_malware":         ["check_status", "quarantine_system"],
    "generate_report":      ["query_database", "log_audit_event"],
    "process_refund":       ["update_subscription", "update_database"],
    "update_subscription":  ["process_refund", "update_database"],
    "provision_vm":         ["restart_service", "update_database"],
    "restart_service":      ["provision_vm", "quarantine_system"],
    "check_status":         ["query_database", "scan_malware"],
    "escalate_to_human":    ["send_notification", "create_ticket"],
    "log_audit_event":      ["create_ticket", "generate_report"],
    # Tier 2
    "deploy_container":     ["run_pipeline", "provision_vm"],
    "rollback_deployment":  ["restore_backup", "restart_service"],
    "rotate_api_key":       ["reset_password", "revoke_access"],
    "backup_database":      ["archive_data", "query_database"],
    "restore_backup":       ["rollback_deployment", "restart_service"],
    "scale_service":        ["provision_vm", "restart_service"],
    "run_pipeline":         ["deploy_container", "restart_service"],
    "approve_access":       ["transfer_ownership", "reset_password"],
    "revoke_access":        ["quarantine_system", "rotate_api_key"],
    "transfer_ownership":   ["approve_access", "update_database"],
    "schedule_maintenance": ["create_ticket", "send_notification"],
    "archive_data":         ["backup_database", "update_database"],
    "enable_feature_flag":  ["deploy_container", "disable_feature_flag"],
    "disable_feature_flag": ["rollback_deployment", "enable_feature_flag"],
    "invalidate_cache":     ["restart_service", "update_database"],
    # Tier 3
    "create_dns_record":    ["update_database", "deploy_container"],
    "renew_certificate":    ["rotate_api_key", "schedule_maintenance"],
    "block_ip_address":     ["quarantine_system", "revoke_access"],
    "unblock_ip_address":   ["approve_access", "block_ip_address"],
    "assign_role":          ["approve_access", "transfer_ownership"],
    "remove_role":          ["revoke_access", "assign_role"],
    "trigger_failover":     ["restart_service", "rollback_deployment"],
    "run_load_test":        ["check_status", "scan_malware"],
    "snapshot_vm":          ["backup_database", "archive_data"],
    "migrate_database":     ["update_database", "restore_backup"],
    "tag_resource":         ["update_database", "create_ticket"],
    "create_alert_rule":    ["send_notification", "check_status"],
    "acknowledge_alert":    ["send_notification", "log_audit_event"],
    "merge_accounts":       ["update_database", "transfer_ownership"],
    "export_data":          ["query_database", "generate_report"],
}

# Queries that intentionally use wording from a confusable label but
# actually require the TRUE label.  Key = true label.
CONFUSABLE_QUERY_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "query_database": [
        ("I want to check what we have on file for {user}.", "check_status"),
        ("Can you verify the records for {order} in the system?", "check_status"),
        ("Generate a list of all entries matching {user}.", "generate_report"),
        ("I need a report of {user}'s transactions from the database.", "generate_report"),
    ],
    "update_database": [
        ("Reset the outdated values for {user} in the system.", "reset_password"),
        ("Log the corrected entry for {user}.", "log_audit_event"),
        ("Record the new address for {user} in the system.", "log_audit_event"),
    ],
    "reset_password": [
        ("Update {user}'s login so they can get back in.", "update_database"),
        ("{user}'s access needs to be fixed -- the system locked them out.", "quarantine_system"),
        ("The system is blocking {user} -- fix their credentials.", "quarantine_system"),
    ],
    "create_ticket": [
        ("Log {user}'s complaint so we can follow up.", "log_audit_event"),
        ("Notify the team by creating a formal issue for {server}.", "send_notification"),
        ("Send a record of this problem to the tracking system.", "send_notification"),
    ],
    "send_notification": [
        ("Escalate this message to {user} right away.", "escalate_to_human"),
        ("Create an alert for {dept} about the maintenance.", "create_ticket"),
        ("Make sure {user} gets the update -- reach out to them.", "escalate_to_human"),
    ],
    "quarantine_system": [
        ("Restart {server}'s isolation procedure.", "restart_service"),
        ("{server} is infected -- shut down its connections.", "scan_malware"),
        ("Scan and lock down {server} immediately.", "scan_malware"),
    ],
    "scan_malware": [
        ("Check {server} for anything suspicious.", "check_status"),
        ("{server} looks compromised -- isolate and investigate.", "quarantine_system"),
        ("Verify {server}'s security posture.", "check_status"),
    ],
    "generate_report": [
        ("Query the system and put together a summary for {dept}.", "query_database"),
        ("Log the quarterly metrics into a document.", "log_audit_event"),
        ("Pull the data and compile findings for {dept}.", "query_database"),
    ],
    "process_refund": [
        ("{user}'s subscription charge was wrong -- give them their money back.", "update_subscription"),
        ("Update {user}'s billing to reflect the returned amount.", "update_database"),
        ("Adjust {user}'s account balance after the error.", "update_subscription"),
    ],
    "update_subscription": [
        ("Refund the difference after the plan change for {user}.", "process_refund"),
        ("Update {user}'s account to a lower tier.", "update_database"),
        ("Process {user}'s request to switch plans.", "process_refund"),
    ],
    "provision_vm": [
        ("Restart the setup process for {dept}'s new environment.", "restart_service"),
        ("Deploy a fresh system for {dept} -- update the inventory too.", "update_database"),
        ("Set up another host -- the current {server} isn't cutting it.", "restart_service"),
    ],
    "restart_service": [
        ("Provision a fresh start for {server}.", "provision_vm"),
        ("The service on {server} is failing -- isolate and reboot.", "quarantine_system"),
        ("Bring {server} back to life.", "provision_vm"),
    ],
    "check_status": [
        ("Query {server} to see if it's still alive.", "query_database"),
        ("Scan {server} -- is it healthy?", "scan_malware"),
        ("Look up how {server} is performing.", "query_database"),
    ],
    "escalate_to_human": [
        ("Send this to a real person for {user}.", "send_notification"),
        ("Notify a manager that {user}'s case needs attention.", "send_notification"),
        ("Create an escalation path for {user}.", "create_ticket"),
    ],
    "log_audit_event": [
        ("File a record of what happened with {server}.", "create_ticket"),
        ("Report the incident to the compliance trail.", "generate_report"),
        ("Document the action taken on {user}'s account.", "create_ticket"),
    ],
    # Tier 2
    "deploy_container": [
        ("Run the pipeline and push the build to {server}.", "run_pipeline"),
        ("Provision a container for {dept}'s new service.", "provision_vm"),
        ("Get the latest build running on a new instance.", "provision_vm"),
    ],
    "rollback_deployment": [
        ("Restore {server} to the version before the last update.", "restore_backup"),
        ("Restart the service using the previous build.", "restart_service"),
        ("Bring {server} back to what was running before.", "restore_backup"),
    ],
    "rotate_api_key": [
        ("Reset the credentials for {server}'s API.", "reset_password"),
        ("Revoke the old key and issue a new one for {server}.", "revoke_access"),
        ("Change the secret used by {server}'s endpoint.", "reset_password"),
    ],
    "backup_database": [
        ("Archive the current state of {server}'s data.", "archive_data"),
        ("Query the database and save a copy of everything.", "query_database"),
        ("Pull all the data from the DB and preserve it.", "query_database"),
    ],
    "restore_backup": [
        ("Roll back the database to yesterday's version.", "rollback_deployment"),
        ("Restart the database from the saved copy.", "restart_service"),
        ("Revert the data to what it looked like before.", "rollback_deployment"),
    ],
    "scale_service": [
        ("Provision more instances for {server}.", "provision_vm"),
        ("Restart the service with more resources.", "restart_service"),
        ("Spin up additional capacity for {server}.", "provision_vm"),
    ],
    "run_pipeline": [
        ("Deploy the latest code through the build system.", "deploy_container"),
        ("Restart the build process for {dept}'s service.", "restart_service"),
        ("Trigger the automated deployment for {server}.", "deploy_container"),
    ],
    "approve_access": [
        ("Transfer the permissions to {user}.", "transfer_ownership"),
        ("Reset {user}'s access so they can get in.", "reset_password"),
        ("Grant {user} the credentials they need.", "reset_password"),
    ],
    "revoke_access": [
        ("Quarantine {user}'s account from the system.", "quarantine_system"),
        ("Rotate {user}'s credentials and lock them out.", "rotate_api_key"),
        ("Block {user} from accessing {server}.", "quarantine_system"),
    ],
    "transfer_ownership": [
        ("Approve {user} as the new owner of {server}.", "approve_access"),
        ("Update the database to reflect the new ownership.", "update_database"),
        ("Grant {user} full control over the resource.", "approve_access"),
    ],
    "schedule_maintenance": [
        ("Create a ticket for the planned downtime on {server}.", "create_ticket"),
        ("Notify {dept} about the upcoming maintenance.", "send_notification"),
        ("Send an alert about the scheduled outage.", "send_notification"),
    ],
    "archive_data": [
        ("Back up the old records and remove them.", "backup_database"),
        ("Update the database by clearing out stale entries.", "update_database"),
        ("Save a copy of the historical data and clean up.", "backup_database"),
    ],
    "enable_feature_flag": [
        ("Deploy the new feature to {dept}.", "deploy_container"),
        ("Disable the old behavior and switch to the new one.", "disable_feature_flag"),
        ("Roll out the container with the new feature enabled.", "deploy_container"),
    ],
    "disable_feature_flag": [
        ("Roll back the feature that was just enabled.", "rollback_deployment"),
        ("Enable the old behavior again.", "enable_feature_flag"),
        ("Revert the toggle to its previous state.", "rollback_deployment"),
    ],
    "invalidate_cache": [
        ("Restart the cache service to clear stale data.", "restart_service"),
        ("Update the cache layer with fresh data.", "update_database"),
        ("Flush everything and reload from the database.", "restart_service"),
    ],
    # Tier 3
    "create_dns_record": [
        ("Update the database with the new service endpoint.", "update_database"),
        ("Deploy the service and make it reachable by name.", "deploy_container"),
        ("Register the new endpoint in the system.", "update_database"),
    ],
    "renew_certificate": [
        ("Rotate the security credentials for {server}.", "rotate_api_key"),
        ("Schedule the cert renewal during maintenance.", "schedule_maintenance"),
        ("Swap out the expired key on {server}.", "rotate_api_key"),
    ],
    "block_ip_address": [
        ("Quarantine the traffic coming from that source.", "quarantine_system"),
        ("Revoke network access for the suspicious address.", "revoke_access"),
        ("Isolate {server} from that IP address.", "quarantine_system"),
    ],
    "unblock_ip_address": [
        ("Approve the IP that was accidentally blocked.", "approve_access"),
        ("Undo the block we put on that address.", "block_ip_address"),
        ("Grant access back to the blocked partner IP.", "approve_access"),
    ],
    "assign_role": [
        ("Approve {user} for the admin position.", "approve_access"),
        ("Transfer the admin permissions to {user}.", "transfer_ownership"),
        ("Grant {user} ownership-level access.", "approve_access"),
    ],
    "remove_role": [
        ("Revoke {user}'s elevated access.", "revoke_access"),
        ("Switch {user} back to a basic role.", "assign_role"),
        ("Strip {user}'s permissions from the system.", "revoke_access"),
    ],
    "trigger_failover": [
        ("Restart the service using the standby.", "restart_service"),
        ("Roll back to the secondary system.", "rollback_deployment"),
        ("Reboot {server} by switching to the DR replica.", "restart_service"),
    ],
    "run_load_test": [
        ("Check how {server} performs under heavy traffic.", "check_status"),
        ("Scan {server}'s performance under stress.", "scan_malware"),
        ("Verify {server} can handle the load.", "check_status"),
    ],
    "snapshot_vm": [
        ("Back up {server}'s current disk state.", "backup_database"),
        ("Archive {server}'s VM image.", "archive_data"),
        ("Save a copy of {server} before we proceed.", "backup_database"),
    ],
    "migrate_database": [
        ("Update the database schema to the new version.", "update_database"),
        ("Restore the DB from the migration artifact.", "restore_backup"),
        ("Write the schema changes to the production DB.", "update_database"),
    ],
    "tag_resource": [
        ("Update the database record with the project tag.", "update_database"),
        ("Create a ticket to track this resource.", "create_ticket"),
        ("Log the resource identifier in the system.", "update_database"),
    ],
    "create_alert_rule": [
        ("Send a notification when the threshold is exceeded.", "send_notification"),
        ("Check if {server}'s metrics need monitoring.", "check_status"),
        ("Set up email alerts for the service.", "send_notification"),
    ],
    "acknowledge_alert": [
        ("Notify the team that we've seen the alert.", "send_notification"),
        ("Log that the alert was acknowledged.", "log_audit_event"),
        ("Send confirmation that we're aware of the issue.", "send_notification"),
    ],
    "merge_accounts": [
        ("Update the database to combine the two records.", "update_database"),
        ("Transfer one account's data into the other.", "transfer_ownership"),
        ("Consolidate the records by updating the DB.", "update_database"),
    ],
    "export_data": [
        ("Query the database for all of {user}'s records.", "query_database"),
        ("Generate a report of {user}'s data.", "generate_report"),
        ("Pull all the data we have on {user}.", "query_database"),
    ],
}


def generate_confusable_query(
    true_label: str,
    rng: random.Random,
    used: Optional[Set[str]] = None,
) -> Tuple[str, str, str]:
    """Return (query, true_label, confusable_with_label) for a confusable pair.

    Tracks rendered outputs in *used* to reduce exact duplicates.
    """
    templates = CONFUSABLE_QUERY_TEMPLATES.get(true_label, [])
    if not templates:
        return ("", true_label, "")
    if used is None:
        used = set()

    for _ in range(30):
        tpl, confusable_with = rng.choice(templates)
        query = fill_entities(tpl, rng)
        if query not in used:
            used.add(query)
            return (query, true_label, confusable_with)

    tpl, confusable_with = rng.choice(templates)
    query = fill_entities(tpl, rng)
    return (query, true_label, confusable_with)


# ─────────────────────────────────────────────────────────────────────────────
#  Paraphrase generation (rule-based)
# ─────────────────────────────────────────────────────────────────────────────

_FORMAL_STARTERS = [
    "Kindly", "Please", "Would you mind", "I'd appreciate it if you could",
    "Could you please", "We request that you",
]
_INFORMAL_STARTERS = [
    "Hey, can you", "Yo,", "Quick one --", "Do me a favor and",
    "Need you to", "Gonna need you to",
]
_QUESTION_FORMS = [
    "Is it possible to {action}?",
    "Could someone {action}?",
    "Who can {action}?",
    "Would anyone be able to {action}?",
]
_PASSIVE_TEMPLATES = [
    "{action} -- that's what needs to happen.",
    "What we need is for {action}.",
    "The situation requires {action}.",
    "It would help if {action}.",
]


def generate_paraphrases(
    query: str,
    rng: random.Random,
    n: int = 4,
) -> List[str]:
    """Generate *n* rule-based paraphrase variants of *query*.

    Strategies: formality shift, question form, passive restructuring,
    word-order shuffling.
    """
    paraphrases: List[str] = []
    clean = query.rstrip(".!?").strip()
    lower = clean.lower()

    # Remove leading address forms for rewriting
    core = re.sub(
        r"^(hey,?\s*can you|please|kindly|could you|can you|"
        r"i need you to|do me a favor and|quick one\s*[-—]\s*|"
        r"urgent:\s*|yo,?\s*)\s*",
        "", lower, flags=re.IGNORECASE,
    ).strip()
    if not core:
        core = lower

    # Strategy 1: formality shift
    starter = rng.choice(_FORMAL_STARTERS)
    paraphrases.append(f"{starter} {core}.")

    # Strategy 2: informal shift
    starter = rng.choice(_INFORMAL_STARTERS)
    paraphrases.append(f"{starter} {core}.")

    # Strategy 3: question form
    tpl = rng.choice(_QUESTION_FORMS)
    paraphrases.append(tpl.format(action=core))

    # Strategy 4: passive / restructuring
    tpl = rng.choice(_PASSIVE_TEMPLATES)
    paraphrases.append(tpl.format(action=core))

    rng.shuffle(paraphrases)
    return paraphrases[:n]


# ─────────────────────────────────────────────────────────────────────────────
#  Near-duplicate detection
# ─────────────────────────────────────────────────────────────────────────────

def word_set(text: str) -> Set[str]:
    """Lowercase tokenised word set (alphanumeric + hyphens)."""
    return set(re.findall(r"[a-z0-9\-]+", text.lower()))


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def near_duplicate_rate(
    queries: List[str],
    threshold: float = 0.8,
) -> float:
    """Fraction of query pairs whose Jaccard similarity exceeds *threshold*.

    For efficiency on large lists, samples pairs rather than computing O(n^2).
    """
    n = len(queries)
    if n < 2:
        return 0.0

    max_pairs = min(50_000, n * (n - 1) // 2)
    sets = [word_set(q) for q in queries]

    if n <= 320:
        hits = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if jaccard_similarity(sets[i], sets[j]) >= threshold:
                    hits += 1
        return hits / total if total else 0.0

    rng = random.Random(42)
    hits = 0
    for _ in range(max_pairs):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 2)
        if j >= i:
            j += 1
        if jaccard_similarity(sets[i], sets[j]) >= threshold:
            hits += 1
    return hits / max_pairs


def exact_duplicate_rate(queries: List[str]) -> float:
    """Fraction of queries that are exact duplicates of another."""
    if not queries:
        return 0.0
    counts = Counter(queries)
    dupes = sum(c - 1 for c in counts.values() if c > 1)
    return dupes / len(queries)


# ─────────────────────────────────────────────────────────────────────────────
#  N-gram extraction
# ─────────────────────────────────────────────────────────────────────────────

def top_ngrams(
    queries: List[str],
    n: int = 3,
    top_k: int = 20,
) -> List[Tuple[str, int]]:
    """Return the *top_k* most frequent *n*-grams across all queries."""
    counter: Counter = Counter()
    for q in queries:
        tokens = re.findall(r"[a-z0-9\-']+", q.lower())
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            counter[gram] += 1
    return counter.most_common(top_k)
