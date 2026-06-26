"""
validate_and_report.py -- Part C: Combined validation and reporting
===================================================================

Reads outputs from Parts A and B, produces:
  - upgraded_data/combined_report.json
  - upgraded_data/combined_report.md
  - upgraded_data/README.md (dataset card + benchmark caveats)
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.text_utils import (
    cue_word_fraction,
    exact_duplicate_rate,
    near_duplicate_rate,
)


def load_config() -> Dict:
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    print(f"  WARNING: {path} not found, using empty DataFrame")
    return pd.DataFrame()


def _safe_load_json(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  WARNING: {path} not found")
    return {}


def entropy(counts: Dict) -> float:
    """Shannon entropy of a count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Build combined report
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_report(root: Path, cfg: Dict) -> Dict:
    """Aggregate Part A and Part B reports into a single analysis."""
    routing_dir = root / cfg["routing"]["output_dir"]
    graph_dir = root / cfg["graph"]["output_dir"]

    routing_report = _safe_load_json(routing_dir / "report.json")
    topo_report = _safe_load_json(graph_dir / "topology_report.json")

    # ── Routing analysis ─────────────────────────────────────────────────
    routing_analysis = {}

    orig = routing_report.get("original", {})
    lex = routing_report.get("lexical_cue_reduced", {})

    routing_analysis["lexical_shortcut_pressure"] = {
        "before": orig.get("cue_word_fraction", "N/A"),
        "after_lexical_reduced": lex.get("cue_word_fraction", "N/A"),
        "improvement": (
            round(orig.get("cue_word_fraction", 0) - lex.get("cue_word_fraction", 0), 4)
            if isinstance(orig.get("cue_word_fraction"), (int, float))
            and isinstance(lex.get("cue_word_fraction"), (int, float))
            else "N/A"
        ),
    }

    # Class balance
    balance = {}
    for split_name in ["original", "lexical_cue_reduced", "confusable_intents",
                        "paraphrase_heldout_train", "paraphrase_heldout_test"]:
        cc = routing_report.get(split_name, {}).get("class_counts", {})
        if cc:
            vals = list(cc.values())
            balance[split_name] = {
                "min": min(vals),
                "max": max(vals),
                "std": round((sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)) ** 0.5, 2),
            }
    routing_analysis["class_balance"] = balance

    routing_analysis["paraphrase_family_overlap"] = routing_report.get(
        "paraphrase_family_overlap", "N/A"
    )

    pairs = routing_report.get("confusable_label_pairs", [])
    routing_analysis["confusable_label_pairs_covered"] = len(pairs)
    routing_analysis["confusable_pairs"] = pairs

    # ── Graph analysis ───────────────────────────────────────────────────
    graph_analysis = {}

    td = topo_report.get("topology_diversity", {})
    graph_analysis["topology_diversity"] = {
        "before": td.get("before", "N/A"),
        "after": td.get("after_base_plus_aug", "N/A"),
    }

    base_fams = topo_report.get("base_topo_families", {})
    combined_fams = topo_report.get("combined_topo_families", {})
    graph_analysis["motif_concentration"] = {
        "before_entropy": round(entropy(base_fams), 3) if base_fams else "N/A",
        "after_entropy": round(entropy(combined_fams), 3) if combined_fams else "N/A",
        "before_families": len(base_fams),
        "after_families": len(combined_fams),
    }

    graph_analysis["hard_negative_availability"] = {
        "total": topo_report.get("hard_negative_count", 0),
        "types": topo_report.get("hard_negative_types", {}),
    }

    overlap = topo_report.get("train_test_family_overlap", [])
    train_fams = topo_report.get("train_families", [])
    test_fams = topo_report.get("test_families", [])
    graph_analysis["topology_heldout_evaluation"] = {
        "train_families": len(train_fams),
        "test_families": len(test_fams),
        "overlap_families": len(overlap),
        "is_nontrivial": len(overlap) < len(test_fams),
        "held_out_test_families": sorted(set(test_fams) - set(train_fams)),
    }

    return {
        "routing_analysis": routing_analysis,
        "graph_analysis": graph_analysis,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Generate README.md
# ─────────────────────────────────────────────────────────────────────────────

def write_readme(report: Dict, out_dir: Path) -> None:
    """Write upgraded_data/README.md with dataset cards and caveats."""
    ra = report.get("routing_analysis", {})
    ga = report.get("graph_analysis", {})

    lines = [
        "# Upgraded Benchmark Datasets",
        "",
        "This directory contains strengthened versions of the single-tool routing",
        "and multi-step DAG retrieval benchmarks, designed to resist shortcut learning",
        "and provide harder, more realistic evaluation.",
        "",
        "## What Changed and Why It Matters",
        "",
        "### Routing Benchmark",
        "",
        "The original single-tool routing dataset (`single_tool_1005.csv`) was class-balanced",
        "and clean, but vulnerable to keyword matching -- most queries contained direct lexical",
        "cues that trivially revealed the target tool (e.g., 'reset password' -> reset_password).",
        "",
        "We produced three new evaluation splits that address distinct weaknesses:",
        "",
        "1. **lexical_cue_reduced.csv** -- Queries rewritten to use indirect phrasings",
        "   (symptoms, outcomes, context) instead of explicit tool/action keywords.",
        f"   Cue-word fraction dropped from {ra.get('lexical_shortcut_pressure', {}).get('before', '?')} "
        f"to {ra.get('lexical_shortcut_pressure', {}).get('after_lexical_reduced', '?')}.",
        "",
        "2. **confusable_intents.csv** -- Queries intentionally surface-similar to confusable",
        "   labels (e.g., a query that sounds like 'check_status' but actually requires",
        f"   'query_database'). Covers {ra.get('confusable_label_pairs_covered', '?')} label pairs.",
        "",
        "3. **paraphrase_heldout_train/test.csv** -- Paraphrase families split across",
        "   train/test so models cannot memorize phrasing patterns.",
        f"   Family overlap between splits: {ra.get('paraphrase_family_overlap', '?')}.",
        "",
        "### DAG / Graph Retrieval Benchmark",
        "",
        "The original multi-step DAG dataset (`test_corpus_for_all_models.csv`) had valid",
        "DAGs but limited topology diversity, making it vulnerable to motif memorisation.",
        "",
        "We strengthened it with:",
        "",
        f"1. **Topology diversity**: Increased from {ga.get('topology_diversity', {}).get('before', '?')} "
        f"to {ga.get('topology_diversity', {}).get('after', '?')} unique topologies.",
        "",
        "2. **New topology families**: Added diamond, asymmetric fork-join, deep asymmetric",
        "   merge, multi-branch, repeated-tool patterns, long branched chains, double diamonds,",
        "   W-shapes, hourglasses, and wide fan-out with depth.",
        "",
        f"3. **Hard negatives**: {ga.get('hard_negative_availability', {}).get('total', '?')} "
        "hard-negative DAGs (same tools/different edges, same topology/different tools,",
        "   single-edge perturbations, extra distractor nodes, missing dependencies).",
        "",
        "4. **Topology-held-out splits**: Train/dev/test where some topology families only",
        f"   appear in test ({len(ga.get('topology_heldout_evaluation', {}).get('held_out_test_families', []))} "
        "families held out).",
        "",
        "## File Structure",
        "",
        "```",
        "upgraded_data/",
        "  routing/",
        "    base_cleaned.csv              # Original data, validated",
        "    lexical_cue_reduced.csv       # Indirect phrasings",
        "    confusable_intents.csv        # Surface-similar cross-label queries",
        "    paraphrase_heldout_train.csv  # Train split (family-separated)",
        "    paraphrase_heldout_test.csv   # Test split (family-separated)",
        "    report.json                   # Validation statistics",
        "    report.md                     # Human-readable report",
        "  graph/",
        "    base_validated.csv            # Original data, validated",
        "    generated_augmented.csv       # New diverse-topology DAGs",
        "    hard_negatives.csv            # Hard-negative DAG variants",
        "    train.csv                     # Training split",
        "    dev.csv                       # Development split",
        "    test_topology_heldout.csv     # Topology-held-out test",
        "    topology_report.json          # Topology analysis",
        "    topology_report.md            # Human-readable topology report",
        "  combined_report.json            # Combined before/after analysis",
        "  combined_report.md              # Human-readable combined report",
        "  README.md                       # This file",
        "```",
        "",
        "## Reproducibility",
        "",
        "All outputs are deterministic given the same random seed (default: 42).",
        "Configuration is in `configs/pipeline_config.json`.",
        "",
        "```powershell",
        "python scripts/run_pipeline.py",
        "```",
        "",
        "## Dataset Card",
        "",
        "| Property | Routing | Graph |",
        "|----------|---------|-------|",
        "| Source | Synthetic (rule-based generation) | Synthetic (template + rule-based) |",
        "| Language | English | English + DAG notation |",
        "| Domain | IT operations, 15-tool API routing | Multi-step workflow orchestration |",
        "| License | Research use | Research use |",
        "| Sensitive data | No PII (synthetic names only) | No PII |",
        "",
        "## Benchmark Caveats",
        "",
        "- All queries are synthetically generated. While phrasing diversity has been",
        "  increased, the vocabulary and scenario space is bounded by the template library.",
        "- Indirect phrasings in `lexical_cue_reduced` were handcrafted; an LLM-assisted",
        "  expansion pass would further improve diversity (optional hook available).",
        "- Hard negatives are structurally plausible but not semantically validated --",
        "  some may be ambiguous edge cases.",
        "- The topology-held-out split assumes that topology families transfer meaningfully;",
        "  this is a reasonable but untested assumption for graph-structure-aware models.",
        "- Entity pools are small (7 users, 6 servers, etc.); downstream models could",
        "  overfit to specific entity names across splits.",
        "",
    ]

    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_combined_report_md(report: Dict, path: Path) -> None:
    """Write the combined Markdown report."""
    ra = report.get("routing_analysis", {})
    ga = report.get("graph_analysis", {})

    lines = [
        "# Combined Benchmark Strengthening Report",
        "",
        "## Routing Benchmark",
        "",
        "### Lexical Shortcut Pressure",
        "",
    ]
    lsp = ra.get("lexical_shortcut_pressure", {})
    lines.append(f"- Before: {lsp.get('before', 'N/A')}")
    lines.append(f"- After (lexical_cue_reduced): {lsp.get('after_lexical_reduced', 'N/A')}")
    lines.append(f"- Improvement: {lsp.get('improvement', 'N/A')}")
    lines.append("")

    lines.append("### Class Balance\n")
    balance = ra.get("class_balance", {})
    for split, stats in balance.items():
        lines.append(f"- **{split}**: min={stats['min']}, max={stats['max']}, std={stats['std']}")
    lines.append("")

    lines.append(f"### Paraphrase Separation: family overlap = {ra.get('paraphrase_family_overlap', 'N/A')}\n")

    lines.append(f"### Confusable Label Coverage: {ra.get('confusable_label_pairs_covered', 'N/A')} pairs\n")
    for p in ra.get("confusable_pairs", []):
        lines.append(f"- {p.get('true_label', '?')} <-> {p.get('confusable_with', '?')}")
    lines.append("")

    lines.append("## Graph Benchmark\n")

    td = ga.get("topology_diversity", {})
    lines.append("### Topology Diversity\n")
    lines.append(f"- Before: {td.get('before', 'N/A')} unique topologies")
    lines.append(f"- After: {td.get('after', 'N/A')} unique topologies")
    lines.append("")

    mc = ga.get("motif_concentration", {})
    lines.append("### Motif Concentration (Shannon Entropy)\n")
    lines.append(f"- Before: {mc.get('before_entropy', 'N/A')} ({mc.get('before_families', '?')} families)")
    lines.append(f"- After: {mc.get('after_entropy', 'N/A')} ({mc.get('after_families', '?')} families)")
    lines.append("")

    hn = ga.get("hard_negative_availability", {})
    lines.append("### Hard Negative Availability\n")
    lines.append(f"- Total: {hn.get('total', 0)}")
    for t, c in sorted(hn.get("types", {}).items()):
        lines.append(f"- {t}: {c}")
    lines.append("")

    the = ga.get("topology_heldout_evaluation", {})
    lines.append("### Topology-Held-Out Evaluation\n")
    lines.append(f"- Train families: {the.get('train_families', '?')}")
    lines.append(f"- Test families: {the.get('test_families', '?')}")
    lines.append(f"- Overlap: {the.get('overlap_families', '?')}")
    lines.append(f"- Is nontrivial: {the.get('is_nontrivial', '?')}")
    held = the.get("held_out_test_families", [])
    if held:
        lines.append(f"- Held-out test-only families: {', '.join(held)}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    routing_dir_override: str | None = None,
    graph_dir_override: str | None = None,
    report_dir_override: str | None = None,
) -> None:
    print("\n" + "=" * 60)
    print("  Part C: Combined Validation & Reporting")
    print("=" * 60 + "\n")

    cfg = load_config()
    root = Path(__file__).resolve().parent.parent

    if routing_dir_override:
        cfg["routing"]["output_dir"] = routing_dir_override
    if graph_dir_override:
        cfg["graph"]["output_dir"] = graph_dir_override

    report_out = report_dir_override or cfg["report"]["output_dir"]
    out_dir = root / report_out
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_combined_report(root, cfg)

    # Save JSON
    json_path = out_dir / "combined_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved combined_report.json")

    # Save Markdown
    write_combined_report_md(report, out_dir / "combined_report.md")
    print(f"  Saved combined_report.md")

    # Generate README
    write_readme(report, out_dir)
    print(f"  Saved README.md")

    # Print summary
    ra = report.get("routing_analysis", {})
    ga = report.get("graph_analysis", {})
    lsp = ra.get("lexical_shortcut_pressure", {})

    print(f"\n  Summary:")
    print(f"    Routing cue-word fraction: {lsp.get('before', '?')} -> {lsp.get('after_lexical_reduced', '?')}")
    print(f"    Paraphrase family overlap: {ra.get('paraphrase_family_overlap', '?')}")
    print(f"    Confusable pairs covered : {ra.get('confusable_label_pairs_covered', '?')}")
    td = ga.get("topology_diversity", {})
    print(f"    Topology diversity       : {td.get('before', '?')} -> {td.get('after', '?')}")
    mc = ga.get("motif_concentration", {})
    print(f"    Motif entropy            : {mc.get('before_entropy', '?')} -> {mc.get('after_entropy', '?')}")
    hn = ga.get("hard_negative_availability", {})
    print(f"    Hard negatives           : {hn.get('total', '?')}")
    the = ga.get("topology_heldout_evaluation", {})
    print(f"    Topology held-out nontrivial: {the.get('is_nontrivial', '?')}")
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
