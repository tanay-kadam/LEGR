"""
dag_extract.py — Validated DAG Extraction (Stage 0)
=====================================================

Extracts execution DAGs from natural-language agent traces using an LLM,
then enforces structural validity via confidence-weighted cycle breaking.

The extraction-then-enforce-acyclicity pattern is adapted from the DAG-LLM
Pipeline (krumiaa/DAGLLM, Apache-2.0), retargeted from causal/affective
concepts to a closed tool vocabulary for agentic workflow planning.

Reference
---------
Krumiaa, "DAG-LLM Pipeline: From Fuzzy Worlds to Causal Graphs",
https://doi.org/10.5281/zenodo.17210060, 2025.
https://github.com/krumiaa/DAGLLM (Apache-2.0 License)

Usage
-----
    $ python src/dag_extract.py --input traces.txt --provider ollama --model llama3.2
    $ python src/dag_extract.py --input traces.txt --output corpus.csv --provider gemini
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
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from legr_tool_count import add_tool_count_argument, bootstrap_tool_count_from_argv

_TOOL_COUNT_OVERRIDE = bootstrap_tool_count_from_argv(sys.argv)

from data_synth import TOOL_VOCAB, TOOL_DESCRIPTIONS, build_dag, dag_to_text
from utils.graph_utils import classify_topology


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt for confidence-weighted DAG extraction
# ═══════════════════════════════════════════════════════════════════════════

_EXTRACTION_PROMPT = """\
You are an API workflow planner. Given a user request, extract a structured \
execution plan as a JSON object with:

- "tools": a list of tool names needed (from the vocabulary below)
- "edges": a list of objects, each with:
    - "source": index of the source tool in the tools list (0-based)
    - "target": index of the target tool in the tools list (0-based)
    - "confidence": integer 0-100 indicating how confident you are that \
this dependency is required

Tool vocabulary:
{tool_list}

Rules:
- Only use tools from the vocabulary above
- Indices in edges refer to positions in the tools list (0-based)
- Assign lower confidence (30-60) to edges you are uncertain about
- Assign higher confidence (70-100) to edges that are clearly implied
- Output valid JSON only, no explanation"""


def _build_extraction_prompt() -> str:
    tool_list = "\n".join(f"  - {t}: {TOOL_DESCRIPTIONS[t]}" for t in TOOL_VOCAB)
    return _EXTRACTION_PROMPT.format(tool_list=tool_list)


# ═══════════════════════════════════════════════════════════════════════════
#  LLM response parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_extraction_response(
    text: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Parse LLM JSON into (tools, edges_with_confidence).

    Each edge dict has keys: source, target, confidence.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return [], []
        else:
            return [], []

    tools = data.get("tools", [])
    raw_edges = data.get("edges", [])

    if not isinstance(tools, list):
        tools = []
    if not isinstance(raw_edges, list):
        raw_edges = []

    tools = [str(t).strip() for t in tools if str(t).strip() in TOOL_VOCAB]
    n = len(tools)

    edges: List[Dict[str, Any]] = []
    for e in raw_edges:
        if isinstance(e, dict):
            try:
                s = int(e.get("source", -1))
                t = int(e.get("target", -1))
            except (TypeError, ValueError):
                continue
            conf = e.get("confidence", 50)
            try:
                conf = int(conf)
            except (TypeError, ValueError):
                conf = 50
            conf = max(0, min(100, conf))
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            try:
                s, t = int(e[0]), int(e[1])
            except (TypeError, ValueError):
                continue
            conf = int(e[2]) if len(e) > 2 else 50
            conf = max(0, min(100, conf))
        else:
            continue

        if 0 <= s < n and 0 <= t < n and s != t:
            edges.append({"source": s, "target": t, "confidence": conf})

    return tools, edges


# ═══════════════════════════════════════════════════════════════════════════
#  Cycle detection and confidence-weighted repair
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIDENCE = 50


def detect_cycles(
    tools: List[str],
    edges: List[Dict[str, Any]],
) -> bool:
    """Return True if the edge set contains a cycle."""
    G = nx.DiGraph()
    G.add_nodes_from(range(len(tools)))
    G.add_edges_from((e["source"], e["target"]) for e in edges)
    return not nx.is_directed_acyclic_graph(G)


def break_cycles_min_confidence(
    tools: List[str],
    edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove lowest-confidence edges on cycles until the graph is a DAG.

    Adapted from krumiaa/DAGLLM's ``_break_cycles_min_conf`` (Apache-2.0).

    Returns (clean_edges, removed_edges).
    """
    current = list(edges)
    removed: List[Dict[str, Any]] = []

    while True:
        G = nx.DiGraph()
        G.add_nodes_from(range(len(tools)))
        for e in current:
            G.add_edge(e["source"], e["target"])

        if nx.is_directed_acyclic_graph(G):
            break

        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break

        cycle_pairs = {(u, v) for u, v, _ in cycle}

        candidate_indices = [
            i for i, e in enumerate(current)
            if (e["source"], e["target"]) in cycle_pairs
        ]

        if not candidate_indices:
            break

        victim = min(
            candidate_indices,
            key=lambda i: current[i].get("confidence", DEFAULT_CONFIDENCE),
        )
        removed.append(current[victim])
        del current[victim]

    return current, removed


def enforce_dag(
    tools: List[str],
    edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Validate and repair a predicted graph to ensure DAG structure.

    Returns (clean_edges, removed_edges, had_cycle).
    """
    had_cycle = detect_cycles(tools, edges)
    if not had_cycle:
        return edges, [], False
    clean, removed = break_cycles_min_confidence(tools, edges)
    return clean, removed, True


# ═══════════════════════════════════════════════════════════════════════════
#  Structural validity checking (usable on any predicted graph)
# ═══════════════════════════════════════════════════════════════════════════

def check_structural_validity(
    tools: List[str],
    edges: List[Tuple[int, int]] | List[List[int]],
) -> Dict[str, Any]:
    """Check structural properties of a predicted DAG without repairing it.

    Returns a dict with: is_dag, has_cycle, n_cycle_edges, is_connected.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(len(tools)))
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            G.add_edge(int(e[0]), int(e[1]))

    is_dag = nx.is_directed_acyclic_graph(G)

    n_cycle_edges = 0
    if not is_dag:
        try:
            cycle = nx.find_cycle(G, orientation="original")
            n_cycle_edges = len(cycle)
        except nx.NetworkXNoCycle:
            pass

    is_connected = nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else True

    return {
        "is_dag": is_dag,
        "has_cycle": not is_dag,
        "n_cycle_edges": n_cycle_edges,
        "is_connected": is_connected,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Full extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def extract_dag(
    query: str,
    provider: str = "ollama",
    model: str = "llama3.2",
    client=None,
    ollama_backend=None,
    request_timeout_s: float | None = 120.0,
) -> Dict[str, Any]:
    """Extract a validated DAG from a natural-language query.

    Returns a dict with: tools, edges (index pairs), confidence_per_edge,
    had_cycle, removed_edges, latency_s, parse_failure, topo_family.
    """
    system_prompt = _build_extraction_prompt()

    t0 = time.perf_counter()

    try:
        if provider == "ollama":
            if ollama_backend is None:
                from llm_backends import OllamaBackend
                ollama_backend = OllamaBackend(
                    model_name=model, timeout_s=request_timeout_s,
                )
            resp = ollama_backend.call(system_prompt, query)
            raw = resp.text
        elif provider == "gemini":
            from llm_backends import call_gemini
            if client is None:
                from google import genai
                api_key = os.environ.get("GEMINI_API_KEY", "")
                client = genai.Client(api_key=api_key)
            resp = call_gemini(client, model, system_prompt, query)
            raw = resp.text
        else:
            raise ValueError(f"Unknown provider: {provider}")

        tools, conf_edges = parse_extraction_response(raw)
    except Exception as exc:
        return {
            "tools": [],
            "edges": [],
            "confidence_per_edge": [],
            "had_cycle": False,
            "removed_edges": [],
            "latency_s": round(time.perf_counter() - t0, 6),
            "parse_failure": True,
            "error": str(exc),
            "topo_family": "empty",
        }

    latency_s = round(time.perf_counter() - t0, 6)

    if not tools:
        return {
            "tools": [],
            "edges": [],
            "confidence_per_edge": [],
            "had_cycle": False,
            "removed_edges": [],
            "latency_s": latency_s,
            "parse_failure": True,
            "error": "empty_or_invalid_prediction",
            "topo_family": "empty",
        }

    clean_edges, removed_edges, had_cycle = enforce_dag(tools, conf_edges)

    index_edges = [(e["source"], e["target"]) for e in clean_edges]
    confidences = [e["confidence"] for e in clean_edges]

    topo_family = classify_topology(index_edges, len(tools))

    return {
        "tools": tools,
        "edges": index_edges,
        "confidence_per_edge": confidences,
        "had_cycle": had_cycle,
        "removed_edges": [
            (e["source"], e["target"], e["confidence"]) for e in removed_edges
        ],
        "latency_s": latency_s,
        "parse_failure": False,
        "topo_family": topo_family,
    }


def results_to_corpus_row(
    result: Dict[str, Any],
    query: str,
    dag_id: int,
) -> Dict[str, str]:
    """Convert an extraction result to the standard corpus CSV schema."""
    tools_str = ";".join(result["tools"])
    edges_str = ";".join(
        f"{s}->{t}" for s, t in result["edges"]
    )

    G = None
    dag_text = ""
    if result["tools"] and not result.get("parse_failure"):
        try:
            G = build_dag(result["tools"], result["edges"])
            dag_text = dag_to_text(G)
        except (AssertionError, Exception):
            dag_text = ""

    return {
        "query": query,
        "dag_id": dag_id,
        "dag_text": dag_text,
        "tools": tools_str,
        "edges": edges_str,
        "topo_family": result.get("topo_family", ""),
        "source": "extracted",
        "split": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Stage 0: Extract validated DAGs from text traces"
    )
    add_tool_count_argument(p, default=_TOOL_COUNT_OVERRIDE)
    p.add_argument("--input", type=str, required=True,
                    help="Input text file (one trace per line) or CSV with 'query' column")
    p.add_argument("--output", type=str, default=None,
                    help="Output CSV path (default: extracted_corpus.csv)")
    p.add_argument("--provider", type=str, default="ollama",
                    choices=["ollama", "gemini"])
    p.add_argument("--model", type=str, default="llama3.2")
    p.add_argument("--max_examples", type=int, default=0,
                    help="Max traces to process (0 = all)")
    p.add_argument("--request_timeout_s", type=float, default=120.0)
    args = p.parse_args()

    input_path = Path(args.input)
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
        queries = df["query"].tolist()
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]

    if args.max_examples > 0:
        queries = queries[:args.max_examples]

    output_path = args.output or "extracted_corpus.csv"

    ollama_backend = None
    client = None
    if args.provider == "ollama":
        from llm_backends import OllamaBackend
        ollama_backend = OllamaBackend(
            model_name=args.model, timeout_s=args.request_timeout_s,
        )
    elif args.provider == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        client = genai.Client(api_key=api_key)

    rows = []
    n_cycles = 0
    n_failures = 0

    print(f"  Extracting DAGs from {len(queries)} traces...")
    for i, query in enumerate(queries):
        result = extract_dag(
            query, provider=args.provider, model=args.model,
            client=client, ollama_backend=ollama_backend,
            request_timeout_s=args.request_timeout_s,
        )
        if result.get("parse_failure"):
            n_failures += 1
        if result.get("had_cycle"):
            n_cycles += 1

        row = results_to_corpus_row(result, query, dag_id=i)
        rows.append(row)

        if (i + 1) % 20 == 0 or i == len(queries) - 1:
            print(f"    {i + 1}/{len(queries)} done "
                  f"(cycles repaired: {n_cycles}, failures: {n_failures})")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"\n  Saved {len(rows)} rows to {output_path}")
    print(f"  Cycles repaired: {n_cycles}/{len(rows)}")
    print(f"  Parse failures:  {n_failures}/{len(rows)}")


if __name__ == "__main__":
    main()
