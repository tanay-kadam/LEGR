"""
evaluator.py — Comparative Experiment Runner
===============================================

Runs both taxonomies against a dataset and computes aggregate metrics.

Exports:
    - ``run_experiment()``: Run both taxonomies, return metrics + records.
    - ``per_api_accuracy()``: Per-tool accuracy breakdown.
    - ``misrouted_queries()``: DataFrame of incorrectly routed queries.
    - ``branch_health_report()``: Branch size balance analysis.
    - ``confusable_pair_analysis()``: Cross-branch vs within-branch confusion.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dataset import build_dataset
from llm_backends import OllamaBackend
from routers import RoutingResult, hierarchical_route
from taxonomies import (
    ALL_TOOLS,
    SEMANTIC_TAXONOMY,
    TOOL_BOUND_TAXONOMY,
    get_branch_for_tool,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Run a single taxonomy
# ═══════════════════════════════════════════════════════════════════════════

def _run_taxonomy(
    taxonomy: dict,
    df: pd.DataFrame,
    client=None,
    model: str = "",
    llm_backend: Optional[OllamaBackend] = None,
    inter_query_delay: float = 0.3,
) -> Tuple[Dict[str, Any], List[RoutingResult]]:
    """Route every query in *df* through *taxonomy* and compute metrics."""
    records: List[RoutingResult] = []
    taxonomy_name = taxonomy["name"]

    for idx, row in df.iterrows():
        query = row["query"]
        ground_truth = row["ground_truth"]
        query_id = int(idx) if not isinstance(idx, int) else idx

        result = hierarchical_route(
            query=query,
            taxonomy=taxonomy,
            client=client,
            model=model,
            llm_backend=llm_backend,
        )

        predicted = result.get("predicted_tool")
        branch = result.get("selected_branch")
        hallucinated = result.get("hallucinated", False)
        correct = (predicted is not None and
                   predicted.lower() == ground_truth.lower())

        expected_branch = get_branch_for_tool(taxonomy, ground_truth)
        branch_correct = (branch is not None and expected_branch is not None and
                          branch.lower() == expected_branch.lower())

        rec = RoutingResult(
            query_id=query_id,
            query=query,
            ground_truth=ground_truth,
            taxonomy_name=taxonomy_name,
            predicted_tool=predicted,
            selected_branch=branch,
            correct=correct,
            branch_correct=branch_correct,
            hallucinated=hallucinated,
            latency_s=result.get("latency_s", 0.0),
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
            error=result.get("error"),
        )
        records.append(rec)

        if inter_query_delay > 0:
            time.sleep(inter_query_delay)

    metrics = _compute_aggregate_metrics(records, taxonomy_name)
    return metrics, records


def _compute_aggregate_metrics(
    records: List[RoutingResult],
    taxonomy_name: str,
) -> Dict[str, Any]:
    """Compute aggregate metrics from a list of routing results."""
    n = len(records)
    if n == 0:
        return {"taxonomy": taxonomy_name, "total_queries": 0}

    correct = sum(1 for r in records if r.correct)
    branch_correct = sum(1 for r in records if r.branch_correct)
    branch_correct_tool_wrong = sum(
        1 for r in records if r.branch_correct and not r.correct
    )
    hallucinated_tools = sum(1 for r in records if r.hallucinated)
    errors = sum(1 for r in records if r.error)

    latencies = [r.latency_s for r in records]
    total_prompt = sum(r.prompt_tokens for r in records)
    total_completion = sum(r.completion_tokens for r in records)
    total_tokens = sum(r.total_tokens for r in records)

    sorted_lat = sorted(latencies)
    p95_idx = max(0, int(len(sorted_lat) * 0.95) - 1)

    return {
        "taxonomy": taxonomy_name,
        "total_queries": n,
        "correct": correct,
        "accuracy_pct": round(correct / n * 100, 1),
        "branch_correct": branch_correct,
        "branch_accuracy_pct": round(branch_correct / n * 100, 1),
        "branch_correct_tool_wrong": branch_correct_tool_wrong,
        "branch_correct_tool_wrong_pct": round(branch_correct_tool_wrong / n * 100, 1),
        "hallucinated_tools": hallucinated_tools,
        "hallucination_rate_pct": round(hallucinated_tools / n * 100, 1),
        "error_propagations": errors,
        "error_rate_pct": round(errors / n * 100, 1),
        "mean_latency_s": round(statistics.mean(latencies), 3),
        "median_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(sorted_lat[p95_idx], 3),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "mean_tokens_per_query": round(total_tokens / n, 1),
        "estimated_cost_usd": 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Branch health diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def branch_health_report(taxonomy: dict) -> Dict[str, Any]:
    """Analyze branch size balance for a taxonomy.

    Returns a dict with per-branch tool counts, the maximum branch
    percentage, and a warning flag if any branch exceeds 33% of tools.
    """
    branches = taxonomy.get("branches", {})
    total_tools = sum(len(b["tools"]) for b in branches.values())
    if total_tools == 0:
        return {
            "taxonomy": taxonomy.get("name", ""),
            "total_tools": 0,
            "branches": {},
            "max_branch_pct": 0.0,
            "imbalanced": False,
        }

    branch_sizes = {}
    max_pct = 0.0
    for name, info in branches.items():
        count = len(info["tools"])
        pct = round(count / total_tools * 100, 1)
        branch_sizes[name] = {"count": count, "pct": pct}
        max_pct = max(max_pct, pct)

    return {
        "taxonomy": taxonomy.get("name", ""),
        "total_tools": total_tools,
        "branches": branch_sizes,
        "max_branch_pct": max_pct,
        "imbalanced": max_pct > 33.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Confusable pair analysis
# ═══════════════════════════════════════════════════════════════════════════

def confusable_pair_analysis(
    records: List[RoutingResult],
    taxonomy: dict,
) -> pd.DataFrame:
    """Find (ground_truth, predicted) confusion pairs in failures.

    Each pair is tagged as ``within_branch`` (both tools share a branch,
    so the branch step succeeded) or ``cross_branch`` (tools are in
    different branches, so the branch step was the root cause).
    """
    pair_counts: Counter = Counter()
    for r in records:
        if not r.correct and r.predicted_tool:
            pair_counts[(r.ground_truth, r.predicted_tool)] += 1

    if not pair_counts:
        return pd.DataFrame()

    rows = []
    for (gt, pred), count in pair_counts.most_common():
        gt_branch = get_branch_for_tool(taxonomy, gt)
        pred_branch = get_branch_for_tool(taxonomy, pred)
        same_branch = (
            gt_branch is not None
            and pred_branch is not None
            and gt_branch == pred_branch
        )
        rows.append({
            "ground_truth": gt,
            "predicted": pred,
            "count": count,
            "gt_branch": gt_branch or "UNKNOWN",
            "pred_branch": pred_branch or "UNKNOWN",
            "confusion_type": "within_branch" if same_branch else "cross_branch",
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    client=None,
    model: str = "",
    inter_query_delay: float = 0.3,
    llm_backend: Optional[OllamaBackend] = None,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict, Dict, List[RoutingResult], List[RoutingResult]]:
    """Run the full comparative experiment (Semantic vs Tool-Bound).

    Returns (sem_metrics, tb_metrics, sem_records, tb_records).
    """
    if df is None:
        df = build_dataset()

    print(f"\n  Running Semantic Taxonomy ({len(df)} queries)...")
    sem_metrics, sem_records = _run_taxonomy(
        SEMANTIC_TAXONOMY, df,
        client=client, model=model,
        llm_backend=llm_backend,
        inter_query_delay=inter_query_delay,
    )

    print(f"  Running Tool-Bound Taxonomy ({len(df)} queries)...")
    tb_metrics, tb_records = _run_taxonomy(
        TOOL_BOUND_TAXONOMY, df,
        client=client, model=model,
        llm_backend=llm_backend,
        inter_query_delay=inter_query_delay,
    )

    return sem_metrics, tb_metrics, sem_records, tb_records


def per_api_accuracy(records: List[RoutingResult]) -> pd.DataFrame:
    """Compute per-tool accuracy from routing records."""
    by_tool: Dict[str, Dict[str, int]] = {}
    for r in records:
        gt = r.ground_truth
        if gt not in by_tool:
            by_tool[gt] = {"correct": 0, "total": 0}
        by_tool[gt]["total"] += 1
        if r.correct:
            by_tool[gt]["correct"] += 1

    rows = []
    for gt in sorted(by_tool.keys()):
        c = by_tool[gt]["correct"]
        t = by_tool[gt]["total"]
        rows.append({
            "ground_truth": gt,
            "correct": c,
            "total": t,
            "accuracy_pct": round(c / t * 100, 1) if t > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def misrouted_queries(records: List[RoutingResult]) -> pd.DataFrame:
    """Return a DataFrame of incorrectly routed queries."""
    wrong = [r for r in records if not r.correct]
    if not wrong:
        return pd.DataFrame()

    rows = []
    for r in wrong:
        rows.append({
            "query_id": r.query_id,
            "query": r.query,
            "ground_truth": r.ground_truth,
            "predicted_tool": r.predicted_tool,
            "selected_branch": r.selected_branch,
            "hallucinated": r.hallucinated,
            "error": r.error,
        })
    return pd.DataFrame(rows)
