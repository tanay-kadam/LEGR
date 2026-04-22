from __future__ import annotations

from dataclasses import dataclass

from llm_backends import LLMResponse, TokenUsage
from routers import hierarchical_route
from taxonomies import TOOL_BOUND_TAXONOMY


@dataclass
class _FakeBackend:
    responses: list[LLMResponse]
    call_count: int = 0

    def call(self, system_prompt: str, user_prompt: str, response_schema=None):
        self.call_count += 1
        if not self.responses:
            raise AssertionError("No fake responses remaining")
        return self.responses.pop(0)


def _resp(text: str, parsed=None) -> LLMResponse:
    return LLMResponse(text=text, parsed=parsed, usage=TokenUsage())


def test_invalid_branch_selection_stops_before_tool_step():
    backend = _FakeBackend(
        responses=[
            _resp("not_a_real_branch"),
        ]
    )

    result = hierarchical_route(
        query="Please check whether auth-service-14 is healthy.",
        taxonomy=TOOL_BOUND_TAXONOMY,
        llm_backend=backend,
    )

    assert result["predicted_tool"] is None
    assert result["selected_branch"] is None
    assert "Invalid branch selection" in (result["error"] or "")
    assert backend.call_count == 1


def test_valid_text_branch_and_tool_are_normalized_to_exact_choices():
    backend = _FakeBackend(
        responses=[
            _resp("Data Retrieval & Monitoring"),
            _resp("query_database"),
        ]
    )

    result = hierarchical_route(
        query="Can you look up Priya's account details?",
        taxonomy=TOOL_BOUND_TAXONOMY,
        llm_backend=backend,
    )

    assert result["selected_branch"] == "Data Retrieval & Monitoring"
    assert result["predicted_tool"] == "query_database"
    assert result["error"] is None
    assert backend.call_count == 2


def test_blank_parsed_tool_falls_back_to_raw_text_choice():
    class _ParsedBlankTool:
        tool = ""

    backend = _FakeBackend(
        responses=[
            _resp("Communication & Orchestration"),
            _resp('{"tool": ""}\nescalate_to_human', parsed=_ParsedBlankTool()),
        ]
    )

    result = hierarchical_route(
        query="Please hand this issue to a human operator.",
        taxonomy=TOOL_BOUND_TAXONOMY,
        llm_backend=backend,
    )

    assert result["selected_branch"] == "Communication & Orchestration"
    assert result["predicted_tool"] == "escalate_to_human"
    assert result["error"] is None
    assert backend.call_count == 2


def test_invalid_raw_tool_text_is_still_rejected():
    class _ParsedBlankTool:
        tool = ""

    backend = _FakeBackend(
        responses=[
            _resp("Communication & Orchestration"),
            _resp('{"tool": ""}\nnot_a_real_tool', parsed=_ParsedBlankTool()),
        ]
    )

    result = hierarchical_route(
        query="Please hand this issue to a human operator.",
        taxonomy=TOOL_BOUND_TAXONOMY,
        llm_backend=backend,
    )

    assert result["predicted_tool"] is None
    assert result["selected_branch"] == "Communication & Orchestration"
    assert "Invalid tool selection" in (result["error"] or "")
    assert backend.call_count == 2
