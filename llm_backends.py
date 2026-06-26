"""
llm_backends.py — LLM Backend Abstraction Layer
==================================================

Provides a uniform interface for structured LLM calls across:
    - Google Gemini (via ``google-genai``)
    - Ollama (via ``ollama`` Python client)

Both backends accept a system prompt, user prompt, and optional Pydantic
response schema, returning either a parsed Pydantic model or raw text.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════
#  Token usage tracking
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    text: str = ""
    parsed: Any = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_s: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Gemini backend (via google-genai)
# ═══════════════════════════════════════════════════════════════════════════

def call_gemini(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: Optional[Type[BaseModel]] = None,
) -> LLMResponse:
    """Call the Gemini API with optional structured output."""
    from google.genai import types

    t0 = time.perf_counter()

    config_kwargs: Dict[str, Any] = {
        "system_instruction": system_prompt,
    }
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )

    latency = time.perf_counter() - t0

    raw_text = response.text or ""

    usage = TokenUsage()
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        usage.prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
        usage.completion_tokens = getattr(um, "candidates_token_count", 0) or 0

    parsed = None
    if response_schema is not None and raw_text:
        try:
            parsed = response_schema.model_validate_json(raw_text)
        except Exception:
            try:
                data = json.loads(raw_text)
                parsed = response_schema.model_validate(data)
            except Exception:
                pass

    return LLMResponse(text=raw_text, parsed=parsed, usage=usage, latency_s=latency)


# ═══════════════════════════════════════════════════════════════════════════
#  Ollama backend
# ═══════════════════════════════════════════════════════════════════════════

class OllamaBackend:
    """Wrapper around the Ollama Python client for structured LLM calls."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str | None = None,
        timeout_s: float | None = None,
    ):
        import ollama as _ollama
        self._ollama = _ollama
        self.model_name = model_name
        client_kwargs: Dict[str, Any] = {}
        if timeout_s is not None and timeout_s > 0:
            client_kwargs["timeout"] = timeout_s
        self._client = (
            _ollama.Client(host=base_url, **client_kwargs)
            if base_url else _ollama.Client(**client_kwargs)
        )

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama."""
        t0 = time.perf_counter()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: Dict[str, Any] = {}
        if response_schema is not None:
            kwargs["format"] = response_schema.model_json_schema()

        response = self._client.chat(
            model=self.model_name,
            messages=messages,
            **kwargs,
        )

        latency = time.perf_counter() - t0

        raw_text = ""
        if hasattr(response, "message") and response.message:
            raw_text = response.message.content or ""
        elif isinstance(response, dict):
            raw_text = response.get("message", {}).get("content", "")

        usage = TokenUsage()
        if hasattr(response, "prompt_eval_count"):
            usage.prompt_tokens = response.prompt_eval_count or 0
            usage.completion_tokens = getattr(response, "eval_count", 0) or 0
        elif isinstance(response, dict):
            usage.prompt_tokens = response.get("prompt_eval_count", 0) or 0
            usage.completion_tokens = response.get("eval_count", 0) or 0

        parsed = None
        if response_schema is not None and raw_text:
            try:
                parsed = response_schema.model_validate_json(raw_text)
            except Exception:
                try:
                    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text)
                    cleaned = cleaned.replace("```", "").strip()
                    data = json.loads(cleaned)
                    parsed = response_schema.model_validate(data)
                except Exception:
                    pass

        return LLMResponse(text=raw_text, parsed=parsed, usage=usage, latency_s=latency)
