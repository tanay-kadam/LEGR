"""
routers.py -- Two-Step Hierarchical LLM Router
==============================================

Implements a two-step routing strategy:
    1. Branch selection
    2. Tool selection within the chosen branch

Both steps use structured output constrained to the currently valid branch/tool
choices. Fallback parsing is conservative: invalid branch outputs are rejected
cleanly, and invalid tool outputs are blocked before they can propagate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, create_model

from llm_backends import OllamaBackend, call_gemini
from taxonomies import ALL_TOOLS, TOOL_DESCRIPTIONS, format_taxonomy_prompt, get_tools_for_branch


@dataclass
class RoutingResult:
    query_id: int = 0
    query: str = ""
    ground_truth: str = ""
    taxonomy_name: str = ""
    predicted_tool: str | None = None
    selected_branch: str | None = None
    correct: bool = False
    branch_correct: bool = False
    hallucinated: bool = False
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None


_BRANCH_SYSTEM_PROMPT = """\
You are an intent-routing classifier. Given a user query and a set of API branches, select the single branch whose primary purpose best matches what the request is asking to do.

Base the decision on the requested action and intended outcome, not on entity names, department labels, or surface-level keywords.
Choose the branch for the direct operation the user wants completed, not a secondary side effect, prerequisite, or likely downstream follow-up task.
When a request could involve multiple related actions, select the branch that best matches the main action explicitly requested by the user.

Available branches:
{branch_text}

Return the exact branch name from the list above."""

_TOOL_SYSTEM_PROMPT = """\
You are an API-tool selector. Given a user query, select the single API tool from the list below that most directly carries out the requested action.

Choose only from the listed tools. Do not return a branch name, paraphrase, or a tool that is not shown below.
Prefer the most direct operation named in the request, not a related task that might happen before or after it.

Branch: {branch}
Available tools:
{tool_list}

Return the exact tool name from the list above."""


def _build_tool_list(taxonomy: dict, branch: str) -> str:
    tools = get_tools_for_branch(taxonomy, branch)
    lines = []
    for tool in tools:
        desc = TOOL_DESCRIPTIONS.get(tool, "")
        lines.append(f"  - {tool}: {desc}")
    return "\n".join(lines)


def _build_branch_text(taxonomy: dict) -> str:
    """Format branch options for the branch-selection step."""
    if len(ALL_TOOLS) == 15 and all(
        len(branch_info["tools"]) == 5
        for branch_info in taxonomy["branches"].values()
    ):
        lines = []
        for branch_name, branch_info in taxonomy["branches"].items():
            lines.append(f"  - {branch_name}: {branch_info['description']}")
        return "\n".join(lines)
    return format_taxonomy_prompt(taxonomy)


def _choice_literal(options: List[str]):
    """Build a Literal type whose values are restricted to *options*."""
    return Literal.__getitem__(tuple(options))


def _build_selection_model(
    model_name: str,
    field_name: str,
    options: List[str],
) -> type[BaseModel]:
    """Create a Pydantic model with one enum-like choice field."""
    if not options:
        raise ValueError(f"{model_name} requires at least one option")

    choice_type = _choice_literal(options)
    field_desc = f"Select exactly one of: {', '.join(options)}"
    return create_model(
        model_name,
        **{
            field_name: (choice_type, Field(description=field_desc)),
            "reasoning": (
                str,
                Field(default="", description="Brief reasoning for the choice"),
            ),
        },
    )


def _match_choice(raw_choice: str | None, options: List[str]) -> str | None:
    """Conservatively match raw text to one valid choice."""
    if raw_choice is None:
        return None

    normalized = raw_choice.strip().lower()
    if not normalized:
        return None

    for option in options:
        if option.lower() == normalized:
            return option

    mentioned = [option for option in options if option.lower() in normalized]
    if len(mentioned) == 1:
        return mentioned[0]

    close = get_close_matches(normalized, [option.lower() for option in options], n=1, cutoff=0.85)
    if close:
        for option in options:
            if option.lower() == close[0]:
                return option

    return None


def _validate_tool(
    raw_tool: str | None,
    branch_tools: List[str],
    all_tools: List[str],
) -> tuple[str | None, bool]:
    """Validate a predicted tool name against valid tool lists."""
    if raw_tool is None:
        return None, False

    normalized = raw_tool.strip().lower()
    if not normalized:
        return None, False

    for tool in branch_tools:
        if tool.lower() == normalized:
            return tool, False

    for tool in all_tools:
        if tool.lower() == normalized:
            return tool, False

    close = get_close_matches(normalized, [tool.lower() for tool in all_tools], n=1, cutoff=0.7)
    if close:
        for tool in all_tools:
            if tool.lower() == close[0]:
                return tool, False

    return None, True


def _extract_tool_candidate(response_text: str | None, branch_tools: List[str]) -> str | None:
    """Recover a tool choice from raw response text when structured parsing fails."""
    if response_text is None:
        return None

    matched = _match_choice(response_text, branch_tools)
    if matched is not None:
        return matched

    text = response_text.strip()
    if not text:
        return None

    for line in text.splitlines():
        candidate = _match_choice(line, branch_tools)
        if candidate is not None:
            return candidate

        if ":" in line:
            _, _, tail = line.partition(":")
            candidate = _match_choice(tail, branch_tools)
            if candidate is not None:
                return candidate

    return None


def hierarchical_route(
    query: str,
    taxonomy: dict,
    client=None,
    model: str = "",
    llm_backend: Optional[OllamaBackend] = None,
) -> Dict[str, Any]:
    """Route a query through a two-step hierarchical classification."""
    branch_text = _build_branch_text(taxonomy)
    total_prompt = 0
    total_completion = 0
    error_msg = None

    all_valid_tools = list(ALL_TOOLS)
    branch_names = list(taxonomy["branches"].keys())
    branch_model = _build_selection_model("BranchSelectionModel", "branch", branch_names)

    t0 = time.perf_counter()
    branch_sys = _BRANCH_SYSTEM_PROMPT.format(branch_text=branch_text)

    try:
        if llm_backend is not None:
            resp1 = llm_backend.call(branch_sys, query, branch_model)
        elif client is not None:
            resp1 = call_gemini(client, model, branch_sys, query, branch_model)
        else:
            raise RuntimeError("No LLM backend configured")
    except Exception as exc:
        latency = time.perf_counter() - t0
        return {
            "predicted_tool": None,
            "selected_branch": None,
            "latency_s": latency,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "error": f"Branch selection failed: {exc}",
            "hallucinated": False,
        }

    total_prompt += resp1.usage.prompt_tokens
    total_completion += resp1.usage.completion_tokens

    if resp1.parsed and hasattr(resp1.parsed, "branch"):
        selected_branch = str(resp1.parsed.branch)
    else:
        selected_branch = _match_choice(resp1.text, branch_names)

    if selected_branch is None:
        latency = time.perf_counter() - t0
        return {
            "predicted_tool": None,
            "selected_branch": None,
            "latency_s": latency,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "error": "Invalid branch selection from LLM response",
            "hallucinated": False,
        }

    branch_tools = get_tools_for_branch(taxonomy, selected_branch)
    if not branch_tools:
        latency = time.perf_counter() - t0
        return {
            "predicted_tool": None,
            "selected_branch": None,
            "latency_s": latency,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "error": "Selected branch did not resolve to any tools",
            "hallucinated": False,
        }

    tool_model = _build_selection_model(
        f"ToolSelectionModel_{selected_branch.replace(' ', '_').replace('&', 'and')}",
        "tool",
        branch_tools,
    )
    tool_sys = _TOOL_SYSTEM_PROMPT.format(
        branch=selected_branch,
        tool_list=_build_tool_list(taxonomy, selected_branch),
    )

    try:
        if llm_backend is not None:
            resp2 = llm_backend.call(tool_sys, query, tool_model)
        elif client is not None:
            resp2 = call_gemini(client, model, tool_sys, query, tool_model)
        else:
            raise RuntimeError("No LLM backend configured")
    except Exception as exc:
        latency = time.perf_counter() - t0
        return {
            "predicted_tool": None,
            "selected_branch": selected_branch,
            "latency_s": latency,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "error": f"Tool selection failed: {exc}",
            "hallucinated": False,
        }

    total_prompt += resp2.usage.prompt_tokens
    total_completion += resp2.usage.completion_tokens

    raw_tool = None
    if resp2.parsed and hasattr(resp2.parsed, "tool"):
        parsed_tool = str(resp2.parsed.tool).strip()
        if parsed_tool:
            raw_tool = parsed_tool

    if raw_tool is None:
        raw_tool = _extract_tool_candidate(resp2.text, branch_tools)

    if raw_tool is None:
        latency = time.perf_counter() - t0
        return {
            "predicted_tool": None,
            "selected_branch": selected_branch,
            "latency_s": latency,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "error": "Invalid tool selection from LLM response",
            "hallucinated": False,
        }

    predicted_tool, hallucinated = _validate_tool(raw_tool, branch_tools, all_valid_tools)
    if hallucinated:
        error_msg = f"Hallucinated tool name: '{raw_tool}'"
    elif predicted_tool is None:
        error_msg = "Could not validate tool selection"

    latency = time.perf_counter() - t0
    return {
        "predicted_tool": predicted_tool,
        "selected_branch": selected_branch,
        "latency_s": latency,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "error": error_msg,
        "hallucinated": hallucinated,
    }
