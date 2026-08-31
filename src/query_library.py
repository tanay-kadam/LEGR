"""
query_library.py — Phrasing resources for v2 query regeneration
================================================================

Holds the surface-form inventory used by ``scripts/regenerate_queries_v2.py``:
an expanded connector/opener/closer set and split-partitionable entity pools.

The per-tool phrase library is read out of ``data_synth.py`` with ``ast`` instead
of being imported. ``data_synth`` imports torch and torch_geometric at module
level, and query regeneration needs only the string literals, so parsing keeps
the regeneration path installable with pandas alone.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path
from typing import Dict, List, Sequence

_DATA_SYNTH = Path(__file__).with_name("data_synth.py")


def _load_literal(name: str):
    """Return the value of a module-level literal assignment in data_synth.py."""
    tree = ast.parse(_DATA_SYNTH.read_text(encoding="utf-8"))
    for node in tree.body:
        target_names: List[str] = []
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value = node.value
        elif isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        if name in target_names and value is not None:
            return ast.literal_eval(value)
    raise KeyError(f"{name} not found in {_DATA_SYNTH}")


TOOL_PHRASES: Dict[str, List[str]] = _load_literal("_TOOL_PHRASES")
TOOL_VOCAB: List[str] = _load_literal("TOOL_VOCAB")


# ═══════════════════════════════════════════════════════════════════════════
#  Connectors — expanded from data_synth._synthesize_queries
#
#  Each connector is followed by a bare imperative phrase, so forms requiring a
#  gerund are excluded: v1's ". End with " and ", followed by " produced
#  "End with apply the tag" and are dropped rather than carried over.
# ═══════════════════════════════════════════════════════════════════════════

SEQ_CONN: List[str] = [
    ", then ", ", and then ", ". After that, ", ". Next, ", ". Then, ",
    ". Once that's done, ", " — then ", ". Then go ahead and ",
    ". When that lands, ", ". As soon as that clears, ", ". Following that, ",
    ". With that in place, ", ", after which please ", ". Subsequently, ",
    ". Only once that succeeds, ", ". On completion of that, ",
    ". That done, ", ". Right after, ",
]

PAR_CONN: List[str] = [
    " and simultaneously ", " and also ", " and at the same time ",
    " — in parallel, ", " and concurrently ", ", and while you're at it, ",
    " and in the same pass ", " and alongside that ", " and together with that ",
    " and in tandem ", " — both that and ", " and side by side with it ",
    " plus, at once, ", " and jointly ", " and in the same breath ",
]

MERGE_CONN: List[str] = [
    ". Once both are done, ", ". After those finish, ", ", merge everything and ",
    ". When all of that's complete, ", ", then converge and ",
    ". After both branches land, ", ". Once those have all settled, ",
    ". With both results in hand, ", ". Bringing those together, ",
    ". After everything upstream completes, ", ". Once each of those returns, ",
    ". Consolidating all of that, ", ". When every branch is finished, ",
]

FINAL_CONN: List[str] = [
    ". Finally, ", ". To wrap up, ", ". Last step: ", ", and lastly ",
    ". The last thing is to ", ". And to finish, ", ". Last of all, ",
    ". To close this out, ", ". Lastly, ", ". The final step: ",
    ". Wrapping up, ",
]

OPENERS: List[str] = [
    "Hey, can you", "I need you to", "Quick one —", "Urgent:",
    "{user} is asking us to", "The {dept} team needs us to", "We gotta",
    "Time to", "", "Do me a favor and", "Ticket {ticket} says", "ASAP —",
    "When you get a chance,", "{user} reported an issue —", "From {dept}:",
    "Per {user}'s request,", "For the {dept} team,",
    "Following up on {ticket} —", "{dept} escalated this:",
    "Can you please", "Need a hand:", "Priority request —",
    "Heads-up, we need to", "Sometime today, please", "As discussed with {user},",
    "Per the runbook,", "{user} flagged this —", "On behalf of {dept},",
    "Small ask:", "Whenever you're free,", "This came in from {dept}:",
    "Logging this as {ticket} —", "Kicking this off:", "Requesting the following:",
    "Before end of day,", "Next up:", "Please action {ticket}:",
    "Got a request from {user} —", "Standard procedure here:",
    "Handing this to you:", "{dept} raised {ticket} —", "Need this handled:",
    "Straightforward one:", "New request from {user}:",
]

CLOSERS: List[str] = [
    "", "", "", "", "", "", "",
    " Thanks.", " Appreciate it.", " This is blocking {dept}.",
    " {user} is waiting on this.", " Tracking under {ticket}.",
    " Let me know when it's done.", " Needs to be done today.",
    " Ping me if anything looks off.", " This one's time-sensitive.",
    " Flag any issues to {dept}.", " Confirm once complete.",
]

SINGLE_NODE_PATTERNS: List[str] = [
    "Just {p0}. Nothing else needed.",
    "Quick task: {p0}.",
    "{opener} {p0}.",
    "All I need is for someone to {p0}.",
    "Simple request — {p0} and we're good.",
    "One thing only: {p0}.",
    "Can someone {p0}? That's the whole ask.",
    "{opener} {p0} — single step, nothing follows.",
    "Standalone request: {p0}.",
    "No dependencies here, just {p0}.",
    "{opener} {p0}. That's it.",
    "Minor one — {p0}.",
    "Single action needed: {p0}.",
    "Please {p0}; nothing downstream.",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Entity pools — enlarged so names cannot be memorised across splits
# ═══════════════════════════════════════════════════════════════════════════

ENTITY_POOLS: Dict[str, List[str]] = {
    "user": [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hector",
        "Irene", "James", "Karen", "Leo", "Maya", "Nadia", "Omar", "Priya",
        "Quinn", "Rosa", "Sam", "Tariq", "Uma", "Victor", "Wendy", "Xiang",
        "Yara", "Zach", "Anita", "Boris", "Clara", "Dmitri", "Elena", "Farid",
        "Greta", "Hugo", "Ines", "Jonas",
    ],
    "order": [
        "#10234", "#20891", "#31450", "#42017", "#53698", "#60122", "#71003",
        "#82456", "#93781", "#11290", "#22874", "#33061", "#44519", "#55937",
        "#66208", "#77345", "#88614", "#99072", "#10588", "#21763", "#32940",
        "#43127", "#54306", "#65482",
    ],
    "server": [
        "prod-web-01", "staging-db-02", "payment-api-03", "auth-svc-04",
        "ml-infer-05", "cdn-edge-06", "redis-cache-07", "queue-rabbit-08",
        "search-es-09", "batch-worker-10", "gateway-nginx-11", "metrics-prom-12",
        "logs-loki-13", "vault-secrets-14", "billing-svc-15", "notify-svc-16",
        "report-gen-17", "sync-daemon-18", "archive-store-19", "dns-resolver-20",
        "mail-relay-21", "node-backup-22", "canary-web-23", "replica-db-24",
    ],
    "dept": [
        "Engineering", "Finance", "Marketing", "Legal", "HR", "Operations",
        "Product", "Security", "Support", "Sales", "Compliance", "Research",
        "Procurement", "Facilities", "Data", "Platform", "Design", "Payroll",
    ],
    "ticket": [
        "INC-4021", "INC-7733", "INC-1198", "INC-5560", "INC-8842", "INC-2234",
        "INC-6677", "INC-9911", "INC-3305", "INC-4498", "INC-5127", "INC-6390",
        "INC-7014", "INC-8256", "INC-9483", "INC-1672", "INC-2805", "INC-3941",
        "INC-4160", "INC-5273", "INC-6584", "INC-7896", "INC-8107", "INC-9328",
    ],
}

# Fractions of each pool reserved for train / dev / test. Disjoint by
# construction so an entity name seen at training time never reappears in a
# held-out split.
_SPLIT_FRACTIONS: Dict[str, tuple] = {
    "train": (0.00, 0.50),
    "dev": (0.50, 0.70),
    "test": (0.70, 1.00),
}


def partition_entity_pools(split: str) -> Dict[str, List[str]]:
    """Return the entity pools restricted to *split*'s disjoint slice."""
    key = "train" if split.startswith("train") else "dev" if split.startswith("dev") else "test"
    lo_frac, hi_frac = _SPLIT_FRACTIONS[key]
    out: Dict[str, List[str]] = {}
    for name, pool in ENTITY_POOLS.items():
        lo = int(round(lo_frac * len(pool)))
        hi = int(round(hi_frac * len(pool)))
        slice_ = pool[lo:hi]
        if not slice_:
            raise ValueError(f"entity pool {name!r} too small to partition for {split!r}")
        out[name] = slice_
    return out


def fill_entities(text: str, pools: Dict[str, Sequence[str]], rng: random.Random) -> str:
    """Replace ``{entity}`` placeholders using *pools*.

    Templates write the possessive as ``{dept}'s``, which reads wrong for pool
    values already ending in "s" ("Facilities's"), so those collapse to a bare
    apostrophe instead.
    """
    result = text
    for key, pool in pools.items():
        tag = "{" + key + "}"
        choices = list(pool)
        while tag in result:
            value = rng.choice(choices)
            idx = result.index(tag)
            after = result[idx + len(tag):]
            if value.endswith("s") and after.startswith("'s"):
                result = result[:idx] + value + "'" + after[2:]
            else:
                result = result[:idx] + value + after
    return result
