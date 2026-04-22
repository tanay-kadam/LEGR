"""
replace_templates.py — Template Replacement & Query Augmentation Utility
==========================================================================

Provides utilities for programmatic query generation and template
expansion, supporting the data upgrade pipeline.

Features:
    - Entity-slot replacement ({user}, {server}, {dept}, etc.)
    - Phrasing-style transformation (formal, informal, question, passive)
    - Synonym-based tool-name substitution
    - Batch template expansion with deduplication
    - Export to CSV or JSONL

Used by the upgrade pipeline (scripts/upgrade_routing.py, etc.) to
generate diverse query variants while maintaining label correctness.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#  Entity pools
# ═══════════════════════════════════════════════════════════════════════════

ENTITY_POOLS: Dict[str, List[str]] = {
    "user":   ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
               "Hector", "Irene", "James", "Karen", "Leo"],
    "order":  ["#10234", "#20891", "#31450", "#42017", "#53698", "#60122",
               "#71003", "#82456", "#93781"],
    "server": ["prod-web-01", "staging-db-02", "payment-api-03",
               "auth-svc-04", "ml-infer-05", "cdn-edge-06",
               "cache-redis-07", "queue-rabbit-08"],
    "dept":   ["Engineering", "Finance", "Marketing", "Legal", "HR",
               "Operations", "Product", "Security"],
    "ticket": ["INC-4021", "INC-7733", "INC-1198", "INC-5560", "INC-8842",
               "INC-2234", "INC-6677", "INC-9911"],
}


def fill_template(template: str, rng: random.Random) -> str:
    """Replace {entity} placeholders with random pool values."""
    result = template
    for key, pool in ENTITY_POOLS.items():
        tag = "{" + key + "}"
        while tag in result:
            result = result.replace(tag, rng.choice(pool), 1)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Phrasing transformations
# ═══════════════════════════════════════════════════════════════════════════

_FORMAL_STARTERS = [
    "Kindly", "Please", "Would you mind", "I'd appreciate it if you could",
    "Could you please", "We request that you", "It would be helpful if you",
]

_INFORMAL_STARTERS = [
    "Hey, can you", "Yo,", "Quick one —", "Do me a favor and",
    "Need you to", "Gonna need you to", "Can ya",
]

_QUESTION_FORMS = [
    "Is it possible to {action}?",
    "Could someone {action}?",
    "Who can {action}?",
    "Would anyone be able to {action}?",
    "Can we get someone to {action}?",
]

_PASSIVE_TEMPLATES = [
    "{action} — that's what needs to happen.",
    "What we need is for {action}.",
    "The situation requires {action}.",
    "It would help if {action}.",
    "This is urgent: {action}.",
]


def _strip_opener(text: str) -> str:
    """Remove common openers to get the core action phrase."""
    core = re.sub(
        r"^(hey,?\s*can you|please|kindly|could you|can you|"
        r"i need you to|do me a favor and|quick one\s*[-—]\s*|"
        r"urgent:\s*|yo,?\s*|we need to|gonna need you to)\s*",
        "", text.lower(), flags=re.IGNORECASE,
    ).strip()
    return core if core else text.lower()


def rephrase_formal(query: str, rng: random.Random) -> str:
    core = _strip_opener(query.rstrip(".!?"))
    starter = rng.choice(_FORMAL_STARTERS)
    return f"{starter} {core}."


def rephrase_informal(query: str, rng: random.Random) -> str:
    core = _strip_opener(query.rstrip(".!?"))
    starter = rng.choice(_INFORMAL_STARTERS)
    return f"{starter} {core}."


def rephrase_question(query: str, rng: random.Random) -> str:
    core = _strip_opener(query.rstrip(".!?"))
    tpl = rng.choice(_QUESTION_FORMS)
    return tpl.format(action=core)


def rephrase_passive(query: str, rng: random.Random) -> str:
    core = _strip_opener(query.rstrip(".!?"))
    tpl = rng.choice(_PASSIVE_TEMPLATES)
    return tpl.format(action=core)


def generate_variants(
    query: str,
    rng: random.Random,
    n: int = 4,
    entity_fill: bool = True,
) -> List[str]:
    """Generate *n* stylistic variants of a query."""
    transformers = [rephrase_formal, rephrase_informal,
                    rephrase_question, rephrase_passive]
    variants = []
    for fn in transformers:
        v = fn(query, rng)
        if entity_fill:
            v = fill_template(v, rng)
        variants.append(v)
    rng.shuffle(variants)
    return variants[:n]


# ═══════════════════════════════════════════════════════════════════════════
#  Tool-name synonym substitution
# ═══════════════════════════════════════════════════════════════════════════

TOOL_SYNONYMS: Dict[str, List[str]] = {
    "query_database":       ["look up", "fetch data", "retrieve records", "search DB"],
    "update_database":      ["modify records", "change data", "update the DB", "patch entries"],
    "reset_password":       ["change credentials", "fix login", "restore access", "reset auth"],
    "create_ticket":        ["file a case", "open an issue", "log a ticket", "create a report"],
    "send_notification":    ["send an alert", "notify", "email", "ping"],
    "quarantine_system":    ["isolate", "lock down", "disconnect", "take offline"],
    "scan_malware":         ["security sweep", "threat check", "vulnerability scan", "investigate"],
    "generate_report":      ["compile summary", "produce report", "build document", "create overview"],
    "process_refund":       ["reverse charge", "issue credit", "refund", "reimburse"],
    "update_subscription":  ["change plan", "modify subscription", "adjust tier", "switch package"],
    "provision_vm":         ["deploy server", "spin up instance", "create VM", "launch machine"],
    "restart_service":      ["reboot", "bounce", "restart process", "cycle service"],
    "check_status":         ["health check", "verify status", "monitor", "assess state"],
    "escalate_to_human":    ["involve a person", "hand off", "get human help", "escalate"],
    "log_audit_event":      ["record for compliance", "create audit entry", "log event", "document action"],
    # Tier 2
    "deploy_container":     ["ship container", "push image", "roll out app", "launch container"],
    "rollback_deployment":  ["revert release", "undo deploy", "roll back", "go to previous version"],
    "rotate_api_key":       ["cycle secret", "refresh token", "swap key", "regenerate credentials"],
    "backup_database":      ["snapshot DB", "dump database", "save backup", "create snapshot"],
    "restore_backup":       ["recover from backup", "load snapshot", "restore DB", "bring back data"],
    "scale_service":        ["resize service", "add replicas", "adjust capacity", "bump instances"],
    "run_pipeline":         ["trigger build", "kick off CI", "start pipeline", "execute workflow"],
    "approve_access":       ["grant permission", "authorize access", "sign off request", "green-light"],
    "revoke_access":        ["remove permission", "cut access", "strip privileges", "disable account"],
    "transfer_ownership":   ["reassign owner", "hand off resource", "change ownership", "move responsibility"],
    "schedule_maintenance": ["book downtime", "plan maintenance", "arrange outage window", "set up maintenance"],
    "archive_data":         ["cold-store records", "shelve old data", "move to archive", "retire records"],
    "enable_feature_flag":  ["flip toggle on", "activate feature", "turn on flag", "switch on"],
    "disable_feature_flag": ["flip toggle off", "deactivate feature", "turn off flag", "kill toggle"],
    "invalidate_cache":     ["flush cache", "purge cached data", "bust cache", "clear cache"],
    # Tier 3
    "create_dns_record":    ["set up DNS", "add domain entry", "point domain", "register hostname"],
    "renew_certificate":    ["refresh cert", "reissue TLS", "update SSL", "extend certificate"],
    "block_ip_address":     ["ban IP", "firewall off address", "deny traffic", "blacklist source"],
    "unblock_ip_address":   ["allow IP", "whitelist address", "remove block", "re-allow traffic"],
    "assign_role":          ["grant role", "add permission", "set up access level", "attach role"],
    "remove_role":          ["strip role", "revoke permission", "detach access level", "pull role"],
    "trigger_failover":     ["switch to standby", "activate DR", "fail over", "use secondary"],
    "run_load_test":        ["stress test", "benchmark service", "simulate traffic", "performance test"],
    "snapshot_vm":          ["capture VM state", "save disk image", "create checkpoint", "freeze VM"],
    "migrate_database":     ["run schema migration", "upgrade DB", "apply migration", "evolve schema"],
    "tag_resource":         ["label resource", "annotate instance", "mark for tracking", "apply cost tag"],
    "create_alert_rule":    ["set up monitoring", "configure alert", "define threshold", "add watch rule"],
    "acknowledge_alert":    ["ack alert", "mark as seen", "silence notification", "confirm awareness"],
    "merge_accounts":       ["combine profiles", "unify records", "consolidate accounts", "join entries"],
    "export_data":          ["data dump", "pull records", "download data", "extract information"],
}


def substitute_tool_names(query: str, rng: random.Random) -> str:
    """Replace tool-related phrases with synonyms."""
    result = query
    for tool, synonyms in TOOL_SYNONYMS.items():
        tool_phrase = tool.replace("_", " ")
        if tool_phrase in result.lower():
            replacement = rng.choice(synonyms)
            result = re.sub(re.escape(tool_phrase), replacement, result,
                            flags=re.IGNORECASE, count=1)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Batch expansion
# ═══════════════════════════════════════════════════════════════════════════

def expand_templates(
    templates: List[Tuple[str, str]],
    n_entity_variants: int = 5,
    n_style_variants: int = 4,
    seed: int = 42,
    deduplicate: bool = True,
) -> List[Dict[str, str]]:
    """Expand (template, label) pairs into diverse query variants.

    Returns list of dicts with 'query', 'label', 'generation_type'.
    """
    rng = random.Random(seed)
    results: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for template, label in templates:
        for _ in range(n_entity_variants):
            filled = fill_template(template, rng)
            if deduplicate and filled in seen:
                continue
            seen.add(filled)
            results.append({
                "query": filled,
                "label": label,
                "generation_type": "entity_fill",
            })

            variants = generate_variants(filled, rng, n=n_style_variants,
                                         entity_fill=False)
            for v in variants:
                if deduplicate and v in seen:
                    continue
                seen.add(v)
                results.append({
                    "query": v,
                    "label": label,
                    "generation_type": "style_variant",
                })

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Export utilities
# ═══════════════════════════════════════════════════════════════════════════

def export_csv(rows: List[Dict], path: str) -> None:
    """Export expanded queries to CSV."""
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Exported {len(df)} rows to {path}")


def export_jsonl(rows: List[Dict], path: str) -> None:
    """Export expanded queries to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Exported {len(rows)} rows to {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("replace_templates.py — Template expansion utility")
    print("  Use as a library: from replace_templates import expand_templates")
    print("  Or run: python replace_templates.py <input.csv> <output.csv>")

    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        n_entity = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        n_style = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

        df = pd.read_csv(input_path)
        query_col = "query" if "query" in df.columns else df.columns[0]
        label_col = "label" if "label" in df.columns else (
            "ground_truth" if "ground_truth" in df.columns else df.columns[1]
        )

        templates = list(zip(df[query_col].tolist(), df[label_col].tolist()))
        expanded = expand_templates(
            templates,
            n_entity_variants=n_entity,
            n_style_variants=n_style,
            seed=seed,
        )
        export_csv(expanded, output_path)
