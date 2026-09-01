"""
azure_query_gen.py — Campaign v4 Azure OpenAI Query Generator
==============================================================

Generates natural language queries for DAGs using Azure OpenAI.
Queries describe the dependency logic without exposing topology labels.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.dag_generator import dag_to_text
from src.data.tool_registry import TOOL_TO_CATEGORY


@dataclass
class AzureQueryResult:
    dag_hash: str
    queries: List[Dict[str, str]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None
    attempt: int = 1


@dataclass
class AzureBudgetTracker:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_calls: int = 0
    failed_calls: int = 0
    total_latency_s: float = 0.0
    cost_per_1k_prompt: float = 0.002
    cost_per_1k_completion: float = 0.002

    @property
    def estimated_cost(self) -> float:
        return (
            self.total_prompt_tokens / 1000 * self.cost_per_1k_prompt
            + self.total_completion_tokens / 1000 * self.cost_per_1k_completion
        )

    def record(self, result: AzureQueryResult):
        self.total_calls += 1
        self.total_prompt_tokens += result.prompt_tokens
        self.total_completion_tokens += result.completion_tokens
        self.total_latency_s += result.latency_s
        if result.error:
            self.failed_calls += 1

    def to_dict(self) -> Dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "total_latency_s": round(self.total_latency_s, 2),
            "estimated_cost_usd": round(self.estimated_cost, 4),
        }


def _build_query_prompt(
    tools: List[str],
    edges: List[Tuple[int, int]],
    family: str,
    n_queries: int = 6,
    conditions: Optional[List[str]] = None,
) -> str:
    """Build the Azure OpenAI prompt for query generation.

    The prompt describes the dependency logic without exposing topology labels.
    """
    if conditions is None:
        conditions = [
            "standard",
            "paraphrase",
            "structural_clear",
            "structural_paraphrase",
            "lexical",
            "confusable",
        ]

    readable_tools = [t.replace("_", " ") for t in tools]
    n = len(tools)

    dep_lines = []
    for u, v in edges:
        dep_lines.append(f"  - \"{readable_tools[u]}\" must complete before \"{readable_tools[v]}\" begins")

    independent_groups = []
    for i in range(n):
        predecessors = {u for u, v in edges if v == i}
        successors = {v for u, v in edges if u == i}
        if not predecessors:
            independent_groups.append(readable_tools[i])

    has_parallel = False
    for i in range(n):
        children = [v for u, v in edges if u == i]
        if len(children) > 1:
            has_parallel = True
            break
    for i in range(n):
        parents = [u for u, v in edges if v == i]
        if len(parents) > 1:
            has_parallel = True
            break

    structure_hints = []
    if has_parallel:
        structure_hints.append(
            "This workflow has PARALLEL branches. "
            "Your queries MUST linguistically express which tasks happen simultaneously "
            "and which tasks depend on others finishing first. "
            "Use phrases like 'at the same time', 'in parallel', 'after both complete', "
            "'once X and Y are done', 'simultaneously', 'concurrently'."
        )
    if len(edges) == 0:
        structure_hints.append("This is a single task with no dependencies.")
    elif all(edges[i][1] == edges[i][0] + 1 for i in range(len(edges))):
        structure_hints.append(
            "This is a sequential pipeline. Express steps in order using "
            "'first', 'then', 'next', 'after that', 'finally'."
        )

    deps_text = "\n".join(dep_lines) if dep_lines else "  (no dependencies — single step)"
    hints_text = "\n".join(structure_hints) if structure_hints else ""

    condition_instructions = []
    for i, cond in enumerate(conditions[:n_queries]):
        if cond == "standard":
            condition_instructions.append(
                f"Query {i+1} (standard): A clear, natural request describing this workflow."
            )
        elif cond == "paraphrase":
            condition_instructions.append(
                f"Query {i+1} (paraphrase): Rephrase the workflow using different words and sentence structure."
            )
        elif cond == "structural_clear":
            condition_instructions.append(
                f"Query {i+1} (structural_clear): Explicitly describe the dependency structure "
                f"(which tasks are parallel, which are sequential, what depends on what)."
            )
        elif cond == "structural_paraphrase":
            condition_instructions.append(
                f"Query {i+1} (structural_paraphrase): Like structural_clear but with varied phrasing."
            )
        elif cond == "lexical":
            condition_instructions.append(
                f"Query {i+1} (lexical): Use indirect language — avoid using the exact tool action words. "
                f"For example, instead of 'read the user profile', say 'look up account details'."
            )
        elif cond == "confusable":
            condition_instructions.append(
                f"Query {i+1} (confusable): The query should sound like it could refer to a "
                f"DIFFERENT workflow but actually describes THIS exact workflow."
            )

    cond_text = "\n".join(condition_instructions)

    prompt = f"""You are generating natural language requests for an AI agent workflow system.

Given a workflow with these tools and dependencies, generate {n_queries} diverse natural language queries 
that a user might write to request this EXACT workflow.

## Tools in this workflow:
{chr(10).join(f"  {i}. {readable_tools[i]}" for i in range(n))}

## Dependencies (ordering constraints):
{deps_text}

{hints_text}

## Query requirements:
{cond_text}

## CRITICAL RULES:
1. Each query MUST describe THIS EXACT workflow — same tools, same dependency order.
2. Do NOT mention topology names like "diamond", "chain", "fork-join", etc.
3. Do NOT number the steps unless it helps clarity.
4. Use natural, conversational language that a real user would write.
5. For parallel branches, ALWAYS express which tasks run simultaneously vs sequentially.
6. Vary sentence structure, vocabulary, and complexity across queries.

## Output format:
Return a JSON array of objects, each with "query" and "condition" fields.
Example: [{{"query": "First check the status, then restart the service", "condition": "standard"}}]
"""
    return prompt


def test_azure_connectivity() -> Dict:
    """Run a minimal Azure OpenAI connectivity test."""
    from src.llm_backends import create_llm_provider, safe_error_message

    result = {
        "success": False,
        "deployment": "",
        "endpoint_hostname": "",
        "error": None,
        "latency_s": 0,
        "sample_response": "",
    }

    try:
        provider = create_llm_provider("azure_openai")
        result["deployment"] = provider.model_name
        result["endpoint_hostname"] = getattr(provider, "endpoint", "")

        t0 = time.perf_counter()
        response = provider.call(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'hello' in exactly one word.",
            temperature=0.0,
            max_tokens=10,
        )
        result["latency_s"] = round(time.perf_counter() - t0, 3)
        result["sample_response"] = response.text[:100]
        result["prompt_tokens"] = response.usage.prompt_tokens
        result["completion_tokens"] = response.usage.completion_tokens
        result["success"] = bool(response.text.strip())

    except Exception as e:
        result["error"] = safe_error_message(e)

    return result


def generate_queries_for_dag(
    dag: Dict,
    n_queries: int = 6,
    conditions: Optional[List[str]] = None,
    max_retries: int = 3,
) -> AzureQueryResult:
    """Generate queries for a single DAG using Azure OpenAI."""
    from src.llm_backends import create_llm_provider, safe_error_message

    provider = create_llm_provider("azure_openai")

    prompt = _build_query_prompt(
        tools=dag["tools"],
        edges=dag["edges"],
        family=dag["family"],
        n_queries=n_queries,
        conditions=conditions,
    )

    result = AzureQueryResult(dag_hash=dag["labeled_hash"])

    for attempt in range(1, max_retries + 1):
        result.attempt = attempt
        try:
            t0 = time.perf_counter()
            response = provider.call(
                system_prompt="You generate natural language workflow requests. Return valid JSON only.",
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=1024,
            )
            result.latency_s = time.perf_counter() - t0
            result.prompt_tokens = response.usage.prompt_tokens
            result.completion_tokens = response.usage.completion_tokens

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            if isinstance(parsed, list):
                result.queries = [
                    {
                        "query": item.get("query", ""),
                        "condition": item.get("condition", "standard"),
                    }
                    for item in parsed
                    if isinstance(item, dict) and item.get("query", "").strip()
                ]
                if result.queries:
                    return result

            result.error = f"Invalid response format on attempt {attempt}"

        except json.JSONDecodeError as e:
            result.error = f"JSON parse error: {e}"
        except Exception as e:
            result.error = safe_error_message(e)

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    return result


def run_azure_pilot(
    dags: List[Dict],
    n_per_tier: int = 10,
    queries_per_dag: int = 4,
    output_dir: str | Path = "artifacts/campaign_v4",
) -> Dict:
    """Run a small Azure pilot to estimate cost and validate generation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budget = AzureBudgetTracker()
    pilot_results = []

    conn_test = test_azure_connectivity()
    if not conn_test["success"]:
        return {
            "success": False,
            "connectivity": conn_test,
            "error": "Azure connectivity test failed",
        }

    sample_dags = dags[:n_per_tier]

    for i, dag in enumerate(sample_dags):
        print(f"  Pilot {i+1}/{len(sample_dags)}: {dag['family']} ({dag['num_nodes']} nodes)...")
        result = generate_queries_for_dag(dag, n_queries=queries_per_dag)
        budget.record(result)

        pilot_results.append({
            "dag_hash": result.dag_hash,
            "family": dag["family"],
            "n_queries_generated": len(result.queries),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": round(result.latency_s, 2),
            "error": result.error,
            "sample_query": result.queries[0]["query"] if result.queries else None,
        })

    total_dags_estimate = len(dags)
    scale_factor = total_dags_estimate / max(len(sample_dags), 1)
    projected_cost = budget.estimated_cost * scale_factor

    report = {
        "connectivity": conn_test,
        "pilot_summary": {
            "dags_tested": len(sample_dags),
            "successful": sum(1 for r in pilot_results if r["n_queries_generated"] > 0),
            "failed": sum(1 for r in pilot_results if r["n_queries_generated"] == 0),
            "budget": budget.to_dict(),
        },
        "cost_projection": {
            "total_dags_to_generate": total_dags_estimate,
            "projected_cost_usd": round(projected_cost, 2),
            "budget_limit_usd": 30.0,
            "within_budget": projected_cost <= 30.0,
        },
        "pilot_details": pilot_results,
    }

    report_path = output_dir / "azure_budget_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Pilot report: {report_path}")
    print(f"  Estimated cost: ${projected_cost:.2f} (budget: $30)")

    return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.data.dag_generator import generate_campaign_dags

    print("=== Azure OpenAI Connectivity Test ===")
    conn = test_azure_connectivity()
    print(json.dumps(conn, indent=2))

    if conn["success"]:
        print("\n=== Running Azure Pilot (15-tool tier, 10 DAGs) ===")
        result = generate_campaign_dags(tier=15, target_unique_dags=50, seed=42)
        pilot = run_azure_pilot(result["all_dags"][:10], n_per_tier=10, queries_per_dag=4)
    else:
        print("\n  Azure not available. Using local template queries.")
        print("  The campaign can proceed with local queries for now.")
