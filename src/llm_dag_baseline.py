"""
llm_dag_baseline.py — LLM-Based DAG Generation Baseline
==========================================================

Evaluates how well an LLM can directly generate execution DAGs from
natural-language queries, without a trained retrieval model.

The LLM receives a query and must output:
    - The set of tools needed (from the 45-tool vocabulary)
    - The execution-order edges between them

Results are compared against ground-truth DAGs using tool-set F1 and
Graph Edit Distance (GED).

Input formats:
    - ``.jsonl``: Each line has query, tools, edges, dag_id, dag_text
    - ``.csv``: Columns: query, tools, edges (plus optional dag_id, dag_text)

Usage
-----
    $ python llm_dag_baseline.py --input outputs/legr_llm_test.jsonl
    $ python llm_dag_baseline.py --input upgraded_data/graph/test_topology_heldout.csv \\
                                  --provider ollama --model llama3.2 --max_examples 200
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from legr_tool_count import add_tool_count_argument, bootstrap_tool_count_from_argv

_TOOL_COUNT_OVERRIDE = bootstrap_tool_count_from_argv(sys.argv)

from data_synth import (
    TOOL_VOCAB,
    TOOL_DESCRIPTIONS,
    build_dag,
    compute_ged,
    export_llm_routing_corpus_jsonl,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Load corpus
# ═══════════════════════════════════════════════════════════════════════════

def _load_corpus_rows(path: str) -> List[Dict]:
    """Load evaluation corpus from JSONL or CSV."""
    p = Path(path)

    if p.suffix == ".jsonl":
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    # CSV format
    df = pd.read_csv(p)
    rows = []
    for _, row in df.iterrows():
        tools_str = str(row.get("tools", ""))
        edges_str = str(row.get("edges", ""))

        tools = [t.strip() for t in tools_str.split(";") if t.strip()]

        edges = []
        if edges_str and edges_str != "nan":
            for part in edges_str.split(";"):
                part = part.strip()
                m = re.match(r"(\d+)\s*->\s*(\d+)", part)
                if m:
                    edges.append([int(m.group(1)), int(m.group(2))])

        rows.append({
            "query": row.get("query", ""),
            "tools": tools,
            "edges": edges,
            "dag_id": row.get("dag_id", 0),
            "dag_text": row.get("dag_text", ""),
        })

    return rows


# ═══════════════════════════════════════════════════════════════════════════
#  LLM DAG generation
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are an API workflow planner. Given a user query, output a JSON object with:
- "tools": a list of tool names needed (from the vocabulary below)
- "edges": a list of [source_index, target_index] pairs representing execution order

Tool vocabulary:
{tool_list}

Rules:
- Only use tools from the vocabulary above
- Indices in edges refer to positions in the tools list (0-based)
- The result must form a valid DAG (no cycles)
- Output valid JSON only, no explanation"""


def _build_system_prompt() -> str:
    tool_list = "\n".join(f"  - {t}: {TOOL_DESCRIPTIONS[t]}" for t in TOOL_VOCAB)
    return _SYSTEM_PROMPT.format(tool_list=tool_list)


def _parse_llm_response(text: str) -> Tuple[List[str], List[List[int]]]:
    """Parse LLM JSON response into (tools, edges)."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return [], []
        else:
            return [], []

    tools = data.get("tools", [])
    edges = data.get("edges", [])

    if not isinstance(tools, list):
        tools = []
    if not isinstance(edges, list):
        edges = []

    tools = [str(t).strip() for t in tools if str(t).strip() in TOOL_VOCAB]
    valid_edges = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            s, d = int(e[0]), int(e[1])
            if 0 <= s < len(tools) and 0 <= d < len(tools) and s != d:
                valid_edges.append([s, d])

    return tools, valid_edges


def _call_llm(
    query: str,
    system_prompt: str,
    provider: str,
    model: str,
    client=None,
    ollama_backend=None,
    request_timeout_s: float | None = None,
) -> str:
    """Call the LLM and return raw text response."""
    if provider == "ollama":
        if ollama_backend is None:
            from llm_backends import OllamaBackend
            ollama_backend = OllamaBackend(
                model_name=model,
                timeout_s=request_timeout_s,
            )
        resp = ollama_backend.call(system_prompt, query)
        return resp.text
    elif provider == "gemini":
        from llm_backends import call_gemini
        if client is None:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY", "")
            client = genai.Client(api_key=api_key)
        resp = call_gemini(client, model, system_prompt, query)
        return resp.text
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _compute_tool_f1(gt_tools: List[str], pred_tools: List[str]) -> float:
    gt_set = set(gt_tools)
    pred_set = set(pred_tools)
    if not gt_set and not pred_set:
        return 1.0

    intersection = gt_set & pred_set
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _load_progress(progress_path: str, max_examples: int) -> Dict[int, Dict[str, Any]]:
    """Load per-example progress records from JSONL if available."""
    path = Path(progress_path)
    if not path.exists():
        return {}

    completed: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("example_index")
            if not isinstance(idx, int):
                continue
            if 0 <= idx < max_examples:
                completed[idx] = rec
    return completed


def _append_progress_record(progress_path: str, record: Dict[str, Any]) -> None:
    """Append one JSONL progress record."""
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _aggregate_progress(
    completed_records: Dict[int, Dict[str, Any]],
    *,
    provider: str,
    model: str,
    total_examples: int,
) -> Dict[str, Any]:
    """Aggregate final metrics from saved per-example records."""
    ordered = [completed_records[i] for i in sorted(completed_records)]

    tool_f1s = [float(rec.get("tool_f1", 0.0)) for rec in ordered]
    ged_errors = [
        float(rec["ged_error"])
        for rec in ordered
        if rec.get("ged_error") is not None
    ]
    latencies = [
        float(rec["latency_s"])
        for rec in ordered
        if rec.get("latency_s") is not None
    ]
    parse_failures = sum(1 for rec in ordered if rec.get("parse_failure"))
    exact_matches = sum(int(rec.get("exact_match", 0)) for rec in ordered)

    results = {
        "provider": provider,
        "model": model,
        "total_examples": total_examples,
        "completed_examples": len(ordered),
        "parse_failures": parse_failures,
        "tool_set_f1": round(np.mean(tool_f1s), 4) if tool_f1s else 0.0,
        "mean_ged_error": round(np.mean(ged_errors), 4) if ged_errors else 0.0,
        "exact_match_rate": round(exact_matches / max(total_examples, 1), 4),
    }

    if latencies:
        results["mean_latency_s"] = round(np.mean(latencies), 3)
        results["median_latency_s"] = round(np.median(latencies), 3)
        latencies.sort()
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        results["p95_latency_s"] = round(latencies[p95_idx], 3)

    return results


def evaluate_llm_baseline(
    corpus: List[Dict],
    provider: str = "ollama",
    model: str = "llama3.2",
    max_examples: int = 0,
    inter_query_delay: float = 0.5,
    request_timeout_s: float | None = 120.0,
    progress_path: str | None = None,
    resume: bool = True,
    save_results_path: str | None = None,
    progress_every: int = 20,
) -> Dict[str, Any]:
    """Run the LLM DAG generation baseline on the corpus."""
    system_prompt = _build_system_prompt()

    client = None
    ollama_backend = None

    if provider == "ollama":
        from llm_backends import OllamaBackend
        ollama_backend = OllamaBackend(
            model_name=model,
            timeout_s=request_timeout_s,
        )
    elif provider == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        client = genai.Client(api_key=api_key)

    examples = corpus[:max_examples] if max_examples > 0 else corpus
    completed_records: Dict[int, Dict[str, Any]] = {}
    if progress_path and resume:
        completed_records = _load_progress(progress_path, len(examples))
        if completed_records:
            print(
                f"  Resuming from {len(completed_records)} completed examples "
                f"using {progress_path}"
            )
            if save_results_path:
                _save_results(
                    save_results_path,
                    _aggregate_progress(
                        completed_records,
                        provider=provider,
                        model=model,
                        total_examples=len(examples),
                    ),
                )

    print(f"\n  Evaluating {len(examples)} examples with {provider}/{model}...")

    for i, ex in enumerate(examples):
        if i in completed_records:
            if progress_every > 0 and (i + 1) % progress_every == 0:
                print(
                    f"    Completed {len(completed_records)}/{len(examples)} "
                    f"(latest example {i + 1})..."
                )
            continue

        query = ex["query"]
        gt_tools = ex["tools"]
        gt_edges = [tuple(e) for e in ex["edges"]]

        t0 = time.perf_counter()

        try:
            raw = _call_llm(
                query, system_prompt, provider, model,
                client=client, ollama_backend=ollama_backend,
                request_timeout_s=request_timeout_s,
            )
            pred_tools, pred_edges = _parse_llm_response(raw)
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            record = {
                "example_index": i,
                "latency_s": round(time.perf_counter() - t0, 6),
                "parse_failure": True,
                "tool_f1": 0.0,
                "ged_error": None,
                "exact_match": 0,
                "error": str(e),
            }
            completed_records[i] = record
            if progress_path:
                _append_progress_record(progress_path, record)
            if save_results_path and ((i + 1) % 20 == 0 or i == len(examples) - 1):
                _save_results(
                    save_results_path,
                    _aggregate_progress(
                        completed_records,
                        provider=provider,
                        model=model,
                        total_examples=len(examples),
                    ),
                )
            if progress_every > 0 and (
                (i + 1) % progress_every == 0 or i == len(examples) - 1
            ):
                print(
                    f"    Completed {len(completed_records)}/{len(examples)} "
                    f"(latest example {i + 1})..."
                )
            continue

        latency_s = time.perf_counter() - t0

        if not pred_tools:
            record = {
                "example_index": i,
                "latency_s": round(latency_s, 6),
                "parse_failure": True,
                "tool_f1": 0.0,
                "ged_error": None,
                "exact_match": 0,
                "error": "empty_or_invalid_prediction",
            }
            completed_records[i] = record
            if progress_path:
                _append_progress_record(progress_path, record)
            if save_results_path and ((i + 1) % 20 == 0 or i == len(examples) - 1):
                _save_results(
                    save_results_path,
                    _aggregate_progress(
                        completed_records,
                        provider=provider,
                        model=model,
                        total_examples=len(examples),
                    ),
                )
            if progress_every > 0 and (
                (i + 1) % progress_every == 0 or i == len(examples) - 1
            ):
                print(
                    f"    Completed {len(completed_records)}/{len(examples)} "
                    f"(latest example {i + 1})..."
                )
            continue

        tool_f1 = _compute_tool_f1(gt_tools, pred_tools)
        ged = None
        exact_match = 0
        try:
            gt_G = build_dag(gt_tools, gt_edges)
            pred_G = build_dag(pred_tools, [tuple(e) for e in pred_edges])
            ged = compute_ged(gt_G, pred_G)
            if ged == 0:
                exact_match = 1
        except Exception:
            pass

        record = {
            "example_index": i,
            "latency_s": round(latency_s, 6),
            "parse_failure": False,
            "tool_f1": round(tool_f1, 6),
            "ged_error": ged,
            "exact_match": exact_match,
        }
        completed_records[i] = record
        if progress_path:
            _append_progress_record(progress_path, record)
        if save_results_path and ((i + 1) % 20 == 0 or i == len(examples) - 1):
            _save_results(
                save_results_path,
                _aggregate_progress(
                    completed_records,
                    provider=provider,
                    model=model,
                    total_examples=len(examples),
                ),
            )

        if inter_query_delay > 0:
            time.sleep(inter_query_delay)

        if progress_every > 0 and (
            (i + 1) % progress_every == 0 or i == len(examples) - 1
        ):
            print(
                f"    Completed {len(completed_records)}/{len(examples)} "
                f"(latest example {i + 1})..."
            )

    return _aggregate_progress(
        completed_records,
        provider=provider,
        model=model,
        total_examples=len(examples),
    )


def _save_results(path: str, results: Dict[str, Any]) -> None:
    """Save results as JSON or single-row CSV based on the file suffix."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results.keys()))
            writer.writeheader()
            writer.writerow(results)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="LLM DAG generation baseline")
    add_tool_count_argument(p, default=_TOOL_COUNT_OVERRIDE)
    p.add_argument("--input", type=str, default="outputs/legr_llm_test.jsonl",
                    help="JSONL or CSV input file with query/tools/edges")
    p.add_argument("--provider", type=str, default="ollama",
                    choices=["ollama", "gemini"],
                    help="LLM provider (default: ollama)")
    p.add_argument("--model", type=str, default="llama3.2",
                    help="Model name (default: llama3.2)")
    p.add_argument("--max_examples", type=int, default=0,
                    help="Max examples to evaluate (0 = all)")
    p.add_argument("--inter_query_delay", type=float, default=0.5,
                    help="Delay between queries in seconds")
    p.add_argument("--request_timeout_s", type=float, default=120.0,
                    help="Per-request timeout in seconds for Ollama calls")
    p.add_argument("--progress_path", type=str, default=None,
                    help="Optional JSONL path for per-example progress and resume")
    p.add_argument("--progress_every", type=int, default=20,
                    help="Print progress every N examples (0 disables updates)")
    p.add_argument("--no_resume", action="store_true",
                    help="Ignore any existing progress file and start fresh")
    p.add_argument("--generate_corpus", action="store_true",
                    help="Generate the test corpus JSONL before evaluating")
    p.add_argument("--save_results", type=str, default=None,
                    help="Save results to JSON file")
    args = p.parse_args()

    if args.generate_corpus:
        print("  Generating test corpus...")
        export_llm_routing_corpus_jsonl(args.input, split="test")
        print(f"  Saved to {args.input}")

    corpus = _load_corpus_rows(args.input)
    print(f"  Loaded {len(corpus)} examples from {args.input}")

    results = evaluate_llm_baseline(
        corpus,
        provider=args.provider,
        model=args.model,
        max_examples=args.max_examples,
        inter_query_delay=args.inter_query_delay,
        request_timeout_s=args.request_timeout_s,
        progress_path=args.progress_path,
        resume=not args.no_resume,
        save_results_path=args.save_results,
        progress_every=args.progress_every,
    )

    print(f"\n  Results:")
    for k, v in results.items():
        print(f"    {k}: {v}")

    if args.save_results:
        _save_results(args.save_results, results)
        print(f"\n  Saved to {args.save_results}")


if __name__ == "__main__":
    main()
