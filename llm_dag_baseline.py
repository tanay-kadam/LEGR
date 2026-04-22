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
            data = json.loads(json_match.group())
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
) -> str:
    """Call the LLM and return raw text response."""
    if provider == "ollama":
        if ollama_backend is None:
            from llm_backends import OllamaBackend
            ollama_backend = OllamaBackend(model_name=model)
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

def evaluate_llm_baseline(
    corpus: List[Dict],
    provider: str = "ollama",
    model: str = "llama3.2",
    max_examples: int = 0,
    inter_query_delay: float = 0.5,
) -> Dict[str, Any]:
    """Run the LLM DAG generation baseline on the corpus."""
    system_prompt = _build_system_prompt()

    client = None
    ollama_backend = None

    if provider == "ollama":
        from llm_backends import OllamaBackend
        ollama_backend = OllamaBackend(model_name=model)
    elif provider == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        client = genai.Client(api_key=api_key)

    examples = corpus[:max_examples] if max_examples > 0 else corpus

    tool_f1s = []
    ged_errors = []
    exact_matches = 0
    parse_failures = 0
    latencies = []

    print(f"\n  Evaluating {len(examples)} examples with {provider}/{model}...")

    for i, ex in enumerate(examples):
        query = ex["query"]
        gt_tools = ex["tools"]
        gt_edges = [tuple(e) for e in ex["edges"]]

        t0 = time.perf_counter()

        try:
            raw = _call_llm(
                query, system_prompt, provider, model,
                client=client, ollama_backend=ollama_backend,
            )
            pred_tools, pred_edges = _parse_llm_response(raw)
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            parse_failures += 1
            continue

        latencies.append(time.perf_counter() - t0)

        if not pred_tools:
            parse_failures += 1
            tool_f1s.append(0.0)
            continue

        # Tool set F1
        gt_set = set(gt_tools)
        pred_set = set(pred_tools)
        if gt_set or pred_set:
            intersection = gt_set & pred_set
            precision = len(intersection) / len(pred_set) if pred_set else 0
            recall = len(intersection) / len(gt_set) if gt_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            tool_f1s.append(f1)
        else:
            tool_f1s.append(1.0)

        # GED
        try:
            gt_G = build_dag(gt_tools, gt_edges)
            pred_G = build_dag(pred_tools, [tuple(e) for e in pred_edges])
            ged = compute_ged(gt_G, pred_G)
            ged_errors.append(ged)
            if ged == 0:
                exact_matches += 1
        except Exception:
            pass

        if inter_query_delay > 0:
            time.sleep(inter_query_delay)

        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(examples)}...")

    n = len(examples)
    results = {
        "provider": provider,
        "model": model,
        "total_examples": n,
        "parse_failures": parse_failures,
        "tool_set_f1": round(np.mean(tool_f1s), 4) if tool_f1s else 0.0,
        "mean_ged_error": round(np.mean(ged_errors), 4) if ged_errors else 0.0,
        "exact_match_rate": round(exact_matches / max(n, 1), 4),
    }

    if latencies:
        results["mean_latency_s"] = round(np.mean(latencies), 3)
        results["median_latency_s"] = round(np.median(latencies), 3)
        latencies.sort()
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        results["p95_latency_s"] = round(latencies[p95_idx], 3)

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="LLM DAG generation baseline")
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
    )

    print(f"\n  Results:")
    for k, v in results.items():
        print(f"    {k}: {v}")

    if args.save_results:
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {args.save_results}")


if __name__ == "__main__":
    main()
