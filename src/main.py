"""
main.py — Experiment Entry Point
==================================
Runs the full Semantic-vs-Tool-Bound comparative routing experiment and
prints a formatted summary report to the terminal.

Usage
-----
    $ cp .env.example .env
    $ # Edit .env and set GEMINI_API_KEY=... or USE_OLLAMA=true
    $ python main.py                    # 50-query canonical dataset
    $ python main.py --scaled           # 1,005 queries (n_per_tool=67) for statistical power
    $ python main.py --scaled --n_per_tool 34   # 510 queries
    $ python main.py --dataset_path results/single_tool_1000.csv   # pre-generated CSV
    $ python main.py --tool_count 30 --dataset_preset routing_base_cleaned
    $ python main.py --tool_count 45 --dataset_preset routing_confusable_intents

The script will:
    1.  Load the dataset (50-query, scaled, or from file).
    2.  Route every query through the **Semantic** taxonomy (baseline).
    3.  Route every query through the **Tool-Bound** taxonomy (proposed).
    4.  Print aggregate metrics side-by-side.
    5.  Export the full per-query log to ``results/experiment_log.csv``
        (or ``new_results/{model}_{N}tools/`` when ``--tool_count`` is set).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import pandas as pd

# Load .env before reading GEMINI_API_KEY / USE_OLLAMA
load_dotenv()

# ── Early --tool_count handling ───────────────────────────────────────────
# vocab_config.ACTIVE_TOOL_COUNT is consumed at import time by taxonomies.py,
# so we must override it *before* importing evaluator / dataset / routers.
import vocab_config as _vc

_TOOL_COUNT_OVERRIDE: Optional[int] = None
for _i, _a in enumerate(sys.argv):
    if _a == "--tool_count" and _i + 1 < len(sys.argv):
        try:
            _TOOL_COUNT_OVERRIDE = int(sys.argv[_i + 1])
            _vc.ACTIVE_TOOL_COUNT = _TOOL_COUNT_OVERRIDE
        except ValueError:
            pass
        break

from evaluator import (
    run_experiment,
    per_api_accuracy,
    misrouted_queries,
    branch_health_report,
    confusable_pair_analysis,
)
from taxonomies import SEMANTIC_TAXONOMY, TOOL_BOUND_TAXONOMY
from dataset import build_dataset, build_scaled_single_tool_dataset
from utils import read_datafile


# ═══════════════════════════════════════════════════════════════════════════
#  Pretty-printing helpers
# ═══════════════════════════════════════════════════════════════════════════

DIVIDER = "=" * 72
THIN_DIVIDER = "-" * 72


def print_header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_side_by_side(sem: dict, tb: dict) -> None:
    """Print a two-column comparison of aggregate metrics."""
    label_width = 30
    col_width = 20

    header = (
        f"{'Metric':<{label_width}}"
        f"{'Semantic':>{col_width}}"
        f"{'Tool-Bound':>{col_width}}"
        f"{'Δ (TB − Sem)':>{col_width}}"
    )
    print(f"\n{header}")
    print(THIN_DIVIDER)

    rows = [
        ("Routing Accuracy (%)",    "accuracy_pct",                  "%"),
        ("Branch Accuracy (%)",     "branch_accuracy_pct",           "%"),
        ("Branch OK / Tool Wrong",  "branch_correct_tool_wrong",     ""),
        ("  (% of queries)",        "branch_correct_tool_wrong_pct", "%"),
        ("Hallucinated Tools",      "hallucinated_tools",            ""),
        ("  (% of queries)",        "hallucination_rate_pct",        "%"),
        ("Error Propagations",      "error_propagations",            ""),
        ("Mean Latency (s)",        "mean_latency_s",                "s"),
        ("Median Latency (s)",      "median_latency_s",              "s"),
        ("P95 Latency (s)",         "p95_latency_s",                 "s"),
        ("Total Tokens",            "total_tokens",                  ""),
        ("Mean Tokens / Query",     "mean_tokens_per_query",         ""),
        ("Prompt Tokens",           "total_prompt_tokens",           ""),
        ("Completion Tokens",       "total_completion_tokens",       ""),
        ("Est. Cost (USD)",         "estimated_cost_usd",            "$"),
    ]

    for label, key, _unit in rows:
        sv = sem[key]
        tv = tb[key]
        delta = round(tv - sv, 4) if isinstance(tv, (int, float)) else "—"
        sign = "+" if isinstance(delta, (int, float)) and delta > 0 else ""
        print(
            f"{label:<{label_width}}"
            f"{sv:>{col_width}}"
            f"{tv:>{col_width}}"
            f"{sign}{delta:>{col_width - 1}}"
        )

    print(THIN_DIVIDER)


def print_per_api(sem_records, tb_records) -> None:
    """Print per-API accuracy tables for both taxonomies."""
    for label, records in [
        ("Semantic Taxonomy", sem_records),
        ("Tool-Bound Taxonomy", tb_records),
    ]:
        print(f"\n  Per-API Accuracy — {label}")
        print(THIN_DIVIDER)
        df = per_api_accuracy(records)
        for _, row in df.iterrows():
            bar = "█" * int(row["accuracy_pct"] / 5)
            print(
                f"  {row['ground_truth']:<24s} "
                f"{int(row['correct']):>2}/{int(row['total']):<2}  "
                f"{row['accuracy_pct']:>5.1f}%  {bar}"
            )
        print()


def _fmt(val) -> str:
    """Format a value for display; use — for None/NaN."""
    if val is None or (isinstance(val, float) and val != val):  # NaN
        return "—"
    s = str(val).strip()
    return s if s else "—"


def print_misrouted(sem_records, tb_records) -> None:
    """Print queries that were incorrectly routed."""
    for label, records in [
        ("Semantic Taxonomy", sem_records),
        ("Tool-Bound Taxonomy", tb_records),
    ]:
        mis = misrouted_queries(records)
        if mis.empty:
            print(f"\n  {label}: No misrouted queries — 100% accuracy!")
            continue

        print(f"\n  Misrouted Queries — {label} ({len(mis)} failures)")
        print(THIN_DIVIDER)
        for _, row in mis.iterrows():
            print(f"  Q{int(row['query_id']):>2}: {row['query'][:65]}")
            got = _fmt(row["predicted_tool"])
            print(f"       Expected: {row['ground_truth']:<22s}  Got: {got}")
            err = _fmt(row["error"])
            if err != "—":
                print(f"       ⚠ {err}")
            print()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Run Semantic vs Tool-Bound taxonomy routing experiment.",
    )
    p.add_argument(
        "--tool_count",
        type=int,
        default=None,
        choices=[15, 30, 45],
        metavar="N",
        help=(
            "Number of active tools (15, 30, or 45). When set, dataset presets "
            "resolve to upgraded_data/routing_{N}tools/ and results are saved to "
            "new_results/{model}_{N}tools/."
        ),
    )
    p.add_argument(
        "--scaled",
        action="store_true",
        help="Use scaled single-tool dataset (500–1000+ queries) for statistically sound evaluation.",
    )
    p.add_argument(
        "--n_per_tool",
        type=int,
        default=67,
        metavar="N",
        help="Queries per tool for --scaled (default 67 -> 1,005 total). Use 34 for ~510.",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        metavar="PATH",
        help="Load dataset from CSV. Supports canonical columns (query, ground_truth) and upgraded columns (transformed_query, label). Overrides --scaled if set.",
    )
    p.add_argument(
        "--dataset_preset",
        type=str,
        default=None,
        choices=[
            "single_tool_1005",
            "routing_base_cleaned",
            "routing_lexical_cue_reduced",
            "routing_confusable_intents",
            "routing_paraphrase_train",
            "routing_paraphrase_test",
        ],
        help=(
            "Use a built-in dataset path preset for routing experiments. "
            "Ignored if --dataset_path is provided."
        ),
    )
    p.add_argument(
        "--query_col",
        type=str,
        default=None,
        metavar="COL",
        help="Optional query column name override for custom CSVs.",
    )
    p.add_argument(
        "--label_col",
        type=str,
        default=None,
        metavar="COL",
        help="Optional label column name override for custom CSVs.",
    )
    p.add_argument(
        "--save_dataset",
        type=str,
        default=None,
        metavar="PATH",
        help="Save the dataset to CSV before running (e.g. results/single_tool_1000.csv).",
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Override the output directory for results "
            "(e.g. new_results/gpt-oss_45tools). When omitted, the directory "
            "is auto-derived from --tool_count and the model name."
        ),
    )
    return p.parse_args()


# Maps preset names to the filename inside each routing data directory.
_ROUTING_PRESET_FILES = {
    "routing_base_cleaned": "base_cleaned.csv",
    "routing_lexical_cue_reduced": "lexical_cue_reduced.csv",
    "routing_confusable_intents": "confusable_intents.csv",
    "routing_paraphrase_train": "paraphrase_heldout_train.csv",
    "routing_paraphrase_test": "paraphrase_heldout_test.csv",
}

# Short split labels used for output file naming (experiment_log_{split}.csv).
_PRESET_SPLIT_LABELS = {
    "routing_base_cleaned": "base_cleaned",
    "routing_lexical_cue_reduced": "lexical_cue_reduced",
    "routing_confusable_intents": "confusable_intents",
    "routing_paraphrase_train": "paraphrase_train",
    "routing_paraphrase_test": "paraphrase_test",
}


def _resolve_dataset_preset(
    preset: Optional[str],
    tool_count: Optional[int] = None,
) -> Optional[str]:
    if preset is None:
        return None

    if tool_count is not None and preset in _ROUTING_PRESET_FILES:
        return f"upgraded_data/routing_{tool_count}tools/{_ROUTING_PRESET_FILES[preset]}"

    mapping = {
        "single_tool_1005": "results/single_tool_1005.csv",
        "routing_base_cleaned": "upgraded_data/routing/base_cleaned.csv",
        "routing_lexical_cue_reduced": "upgraded_data/routing/lexical_cue_reduced.csv",
        "routing_confusable_intents": "upgraded_data/routing/confusable_intents.csv",
        "routing_paraphrase_train": "upgraded_data/routing/paraphrase_heldout_train.csv",
        "routing_paraphrase_test": "upgraded_data/routing/paraphrase_heldout_test.csv",
        "graph_train": "new_dataset/train.parquet",
        "graph_dev": "new_dataset/dev.parquet",
        "graph_test": "new_dataset/test_topology_heldout.parquet",
    }
    return mapping[preset]


def _canonicalise_dataset_columns(
    df: pd.DataFrame,
    dataset_path: str,
    query_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """Map dataset columns to canonical query / ground_truth schema."""
    if query_col is None:
        for cand in ("query", "transformed_query", "text", "utterance"):
            if cand in df.columns:
                query_col = cand
                break
    if label_col is None:
        for cand in ("ground_truth", "label", "tool", "target"):
            if cand in df.columns:
                label_col = cand
                break

    if query_col is None or label_col is None:
        print(
            f"ERROR: Could not infer query/label columns for {dataset_path}.\n"
            f"  Found columns: {list(df.columns)}\n"
            "  Provide overrides with --query_col and --label_col."
        )
        sys.exit(1)

    if query_col not in df.columns or label_col not in df.columns:
        print(
            f"ERROR: Requested columns not found in {dataset_path}.\n"
            f"  query_col={query_col}, label_col={label_col}\n"
            f"  Available: {list(df.columns)}"
        )
        sys.exit(1)

    out = df.copy()
    if query_col != "query":
        out = out.rename(columns={query_col: "query"})
    if label_col != "ground_truth":
        out = out.rename(columns={label_col: "ground_truth"})

    out = out.dropna(subset=["query", "ground_truth"]).reset_index(drop=True)
    return out


def _derive_model_slug(model_label: str) -> str:
    """Derive a filesystem-safe model slug from the display label."""
    slug = model_label.lower().replace(" ", "-").replace("(", "").replace(")", "")
    for ch in "/:*?\"<>|":
        slug = slug.replace(ch, "")
    return slug


def main() -> None:
    args = parse_args()
    tool_count = args.tool_count or _TOOL_COUNT_OVERRIDE
    use_ollama = os.environ.get("USE_OLLAMA", "").strip().lower() in ("1", "true", "yes")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    client = None
    model = ""
    llm_backend = None

    if use_ollama:
        from llm_backends import OllamaBackend
        llm_backend = OllamaBackend(model_name=ollama_model)
        model_label = f"Ollama ({ollama_model})"
    elif api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        model = "gemini-3-flash-preview"
        model_label = model
    else:
        print(
            "ERROR: No LLM configured.\n"
            "  Option 1 — Use Ollama (local, no API key):\n"
            "    Set USE_OLLAMA=true in .env\n"
            "    Optional: OLLAMA_MODEL=llama3.2 (default)\n"
            "    Run: ollama pull llama3.2\n"
            "  Option 2 — Use Gemini:\n"
            "    Set GEMINI_API_KEY=... in .env\n"
            "    Get a free key at: https://aistudio.google.com/app/apikey"
        )
        sys.exit(1)

    dataset_path = args.dataset_path or _resolve_dataset_preset(
        args.dataset_preset, tool_count=tool_count,
    )

    # Dataset: from file/preset, scaled, or canonical 50-query
    if dataset_path:
        raw_df = read_datafile(dataset_path)
        df = _canonicalise_dataset_columns(
            raw_df,
            dataset_path=dataset_path,
            query_col=args.query_col,
            label_col=args.label_col,
        )
        dataset_label = f"{len(df)} queries (from {dataset_path})"
    elif args.scaled:
        df = build_scaled_single_tool_dataset(n_per_tool=args.n_per_tool, seed=42)
        dataset_label = f"{len(df)} queries (scaled, n_per_tool={args.n_per_tool})"
        if args.save_dataset:
            Path(args.save_dataset).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.save_dataset, index=True)
            print(f"  Dataset saved to {args.save_dataset}")
    else:
        df = build_dataset()
        dataset_label = f"{len(df)} synthetic queries × {df['ground_truth'].nunique()} APIs"

    tc_label = f" [{tool_count} tools]" if tool_count else ""
    print_header("Agentic Tool-Bound Taxonomies — Comparative Experiment")
    print(f"  Model       : {model_label}")
    print(f"  Tool count  : {tool_count or _vc.ACTIVE_TOOL_COUNT}")
    print(f"  Dataset     : {dataset_label}")
    print(f"  Taxonomies  : Semantic (baseline) vs Tool-Bound (proposed)")

    # Run the full experiment
    sem_metrics, tb_metrics, sem_records, tb_records = run_experiment(
        client=client,
        model=model,
        inter_query_delay=0.3,
        llm_backend=llm_backend,
        df=df,
    )

    # ── Summary report ────────────────────────────────────────────────────
    print_header("RESULTS — Aggregate Metrics")
    print_side_by_side(sem_metrics, tb_metrics)

    print_header("RESULTS — Per-API Accuracy Breakdown")
    print_per_api(sem_records, tb_records)

    print_header("RESULTS — Misrouted Queries (Qualitative)")
    print_misrouted(sem_records, tb_records)

    # ── Branch health ─────────────────────────────────────────────────────
    print_header("DIAGNOSTICS — Branch Health")
    for label, tax in [
        ("Semantic", SEMANTIC_TAXONOMY),
        ("Tool-Bound", TOOL_BOUND_TAXONOMY),
    ]:
        report = branch_health_report(tax)
        warn = " *** IMBALANCED ***" if report["imbalanced"] else ""
        print(f"\n  {label} ({report['total_tools']} tools, "
              f"max branch {report['max_branch_pct']:.1f}%){warn}")
        print(THIN_DIVIDER)
        for bname, binfo in report["branches"].items():
            bar = "█" * binfo["count"]
            print(f"    {bname:<38s} {binfo['count']:>2} tools  "
                  f"({binfo['pct']:>5.1f}%)  {bar}")
        print()

    # ── Confusable pair analysis ──────────────────────────────────────────
    print_header("DIAGNOSTICS — Confusable Pairs")
    for label, records, tax in [
        ("Semantic", sem_records, SEMANTIC_TAXONOMY),
        ("Tool-Bound", tb_records, TOOL_BOUND_TAXONOMY),
    ]:
        pairs_df = confusable_pair_analysis(records, tax)
        if pairs_df.empty:
            print(f"\n  {label}: No confusable pairs detected.")
            continue
        cross = pairs_df[pairs_df["confusion_type"] == "cross_branch"]
        within = pairs_df[pairs_df["confusion_type"] == "within_branch"]
        print(f"\n  {label}: {len(cross)} cross-branch, "
              f"{len(within)} within-branch confusion pairs")
        print(THIN_DIVIDER)
        for _, row in pairs_df.head(15).iterrows():
            tag = "CROSS" if row["confusion_type"] == "cross_branch" else "within"
            print(f"    {row['ground_truth']:<24s} -> {row['predicted']:<24s} "
                  f"x{row['count']}  [{tag}]")
        print()

    # ── Export full log to CSV ────────────────────────────────────────────
    split_label = _PRESET_SPLIT_LABELS.get(args.dataset_preset or "")

    if args.results_dir:
        results_dir = Path(args.results_dir)
        log_name = f"experiment_log_{split_label}.csv" if split_label else "experiment_log.csv"
        summary_name = f"summary_metrics_{split_label}.csv" if split_label else "summary_metrics.csv"
    elif tool_count and split_label:
        model_slug = _derive_model_slug(model_label)
        results_dir = Path("new_results") / f"{model_slug}_{tool_count}tools"
        log_name = f"experiment_log_{split_label}.csv"
        summary_name = f"summary_metrics_{split_label}.csv"
    else:
        results_dir = Path("results")
        log_name = "experiment_log.csv"
        summary_name = "summary_metrics.csv"

    results_dir.mkdir(parents=True, exist_ok=True)

    all_records = [r.__dict__ for r in sem_records] + [
        r.__dict__ for r in tb_records
    ]
    log_df = pd.DataFrame(all_records)
    csv_path = results_dir / log_name
    log_df.to_csv(csv_path, index=False)

    summary_df = pd.DataFrame([sem_metrics, tb_metrics])
    summary_path = results_dir / summary_name
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  Full per-query log exported to:  {csv_path}")
    print(f"  Summary metrics exported to:     {summary_path}")

    print_header("EXPERIMENT COMPLETE")

    # Final verdict
    delta = tb_metrics["accuracy_pct"] - sem_metrics["accuracy_pct"]
    if delta > 0:
        print(
            f"\n  ✓ Tool-Bound taxonomy outperformed Semantic by "
            f"+{delta:.1f} percentage points in routing accuracy."
        )
    elif delta < 0:
        print(
            f"\n  ✗ Semantic taxonomy outperformed Tool-Bound by "
            f"+{abs(delta):.1f} percentage points in routing accuracy."
        )
    else:
        print("\n  — Both taxonomies achieved identical routing accuracy.")

    print()


if __name__ == "__main__":
    main()
