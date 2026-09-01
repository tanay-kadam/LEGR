"""Shared LLM provider layer for Gemini, Azure OpenAI, and Ollama.

The routing and DAG pipelines consume :class:`LLMResponse` through the
``call`` method. ``generate`` is the smaller provider-neutral interface used
by setup checks and other callers that only need text.
"""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Type, runtime_checkable
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel


CANONICAL_LLAMA_MODEL = "llama3.2:3b"
CANONICAL_GPT_OSS_MODEL = "gpt-oss:120b-cloud"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_PROVIDER_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "llm_providers.json"
)


class LLMConfigurationError(RuntimeError):
    """Raised when a provider profile is absent or incomplete."""


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
    provider: str = ""
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Interface shared by local and hosted generation providers."""

    provider_name: str
    model_name: str

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> LLMResponse: ...

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str: ...


def _parse_structured_text(
    raw_text: str,
    response_schema: Optional[Type[BaseModel]],
) -> Any:
    if response_schema is None or not raw_text:
        return None
    try:
        return response_schema.model_validate_json(raw_text)
    except Exception:
        try:
            cleaned = re.sub(r"```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            cleaned = cleaned.replace("```", "").strip()
            return response_schema.model_validate(json.loads(cleaned))
        except Exception:
            return None


def _openai_strict_json_schema(response_schema: Type[BaseModel]) -> Dict[str, Any]:
    """Normalize a Pydantic schema for OpenAI strict structured outputs."""
    schema = deepcopy(response_schema.model_json_schema())

    def normalize(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["additionalProperties"] = False
                node["required"] = list(properties)
            for value in node.values():
                normalize(value)
        elif isinstance(node, list):
            for value in node:
                normalize(value)

    normalize(schema)
    return schema


def safe_error_message(exc: BaseException) -> str:
    """Return a diagnostic message with environment credentials redacted."""
    message = str(exc)
    for name, value in os.environ.items():
        if len(value) >= 6 and any(
            token in name.upper() for token in ("KEY", "TOKEN", "SECRET")
        ):
            message = message.replace(value, "[REDACTED]")
    message = re.sub(
        r"(?i)(api[-_]?key|authorization|token|sig)=([^&\s]+)",
        r"\1=[REDACTED]",
        message,
    )
    return message[:1000]


def categorize_llm_error(exc: BaseException) -> str:
    """Map provider-specific exceptions to stable, non-secret categories."""
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    if status == 401 or "authentication" in name:
        return "authentication"
    if status == 403 or "permission" in name:
        return "permission"
    if status == 404 or "notfound" in name or "not_found" in name:
        return "deployment_or_model_not_found"
    if status == 429 or "ratelimit" in name or "rate_limit" in name:
        return "rate_limit"
    if "timeout" in name:
        return "timeout"
    if "connection" in name or "connect" in name:
        return "connection"
    if isinstance(exc, (LLMConfigurationError, ValueError)):
        return "configuration"
    return "api_error"


def endpoint_hostname(endpoint: str) -> str:
    """Return only the hostname portion of an endpoint for safe reports."""
    return urlparse(endpoint).hostname or ""


def call_gemini(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: Optional[Type[BaseModel]] = None,
) -> LLMResponse:
    """Call the existing Gemini integration with optional structured output."""
    from google.genai import types

    t0 = time.perf_counter()
    config_kwargs: Dict[str, Any] = {"system_instruction": system_prompt}
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    raw_text = response.text or ""
    usage = TokenUsage()
    if getattr(response, "usage_metadata", None):
        usage.prompt_tokens = (
            getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        )
        usage.completion_tokens = (
            getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        )
    return LLMResponse(
        text=raw_text,
        parsed=_parse_structured_text(raw_text, response_schema),
        usage=usage,
        latency_s=time.perf_counter() - t0,
        provider="gemini",
        model=model,
    )


class OllamaBackend:
    """Ollama client compatible with the repository's existing ``call`` API."""

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str = CANONICAL_LLAMA_MODEL,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ):
        import ollama as _ollama

        self._ollama = _ollama
        self.model_name = model_name
        self.base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        ).rstrip("/")
        self.timeout_s = timeout_s
        client_kwargs: Dict[str, Any] = {}
        if timeout_s is not None and timeout_s > 0:
            client_kwargs["timeout"] = timeout_s
        self._client = _ollama.Client(host=self.base_url, **client_kwargs)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: Dict[str, Any] = {
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }
        if response_schema is not None:
            kwargs["format"] = response_schema.model_json_schema()

        response = self._client.chat(
            model=self.model_name,
            messages=messages,
            **kwargs,
        )
        raw_text = ""
        if getattr(response, "message", None):
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

        return LLMResponse(
            text=raw_text,
            parsed=_parse_structured_text(raw_text, response_schema),
            usage=usage,
            latency_s=time.perf_counter() - t0,
            provider=self.provider_name,
            model=self.model_name,
            metadata={"base_url": self.base_url},
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        return self.call(
            "", prompt, temperature=temperature, max_tokens=max_tokens
        ).text


class AzureOpenAIBackend:
    """Azure OpenAI v1 chat-completions provider."""

    provider_name = "azure_openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment: str | None = None,
        timeout_s: float | None = 120.0,
    ):
        from openai import OpenAI

        values = {
            "AZURE_OPENAI_API_KEY": api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT": deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
        }
        missing = [name for name, value in values.items() if not str(value or "").strip()]
        if missing:
            raise LLMConfigurationError(
                "Missing required Azure OpenAI environment variables: "
                + ", ".join(missing)
            )

        self.endpoint = str(values["AZURE_OPENAI_ENDPOINT"]).rstrip("/")
        if self.endpoint.lower().endswith("/openai/v1"):
            self.base_url = self.endpoint + "/"
        else:
            self.base_url = self.endpoint + "/openai/v1/"
        self.model_name = str(values["AZURE_OPENAI_DEPLOYMENT"])
        self.timeout_s = timeout_s
        kwargs: Dict[str, Any] = {
            "api_key": values["AZURE_OPENAI_API_KEY"],
            "base_url": self.base_url,
        }
        if timeout_s is not None and timeout_s > 0:
            kwargs["timeout"] = timeout_s
        self._client = OpenAI(**kwargs)

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        request: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": _openai_strict_json_schema(response_schema),
                },
            }

        response = self._client.chat.completions.create(**request)
        raw_text = ""
        if response.choices:
            raw_text = response.choices[0].message.content or ""
        usage_obj = getattr(response, "usage", None)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        )
        metadata: Dict[str, Any] = {
            "endpoint_hostname": endpoint_hostname(self.endpoint),
            "api_mode": "v1",
            "deployment": self.model_name,
        }
        if response.choices:
            metadata["finish_reason"] = response.choices[0].finish_reason
        return LLMResponse(
            text=raw_text,
            parsed=_parse_structured_text(raw_text, response_schema),
            usage=usage,
            latency_s=time.perf_counter() - t0,
            provider=self.provider_name,
            model=self.model_name,
            metadata=metadata,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        return self.call(
            "", prompt, temperature=temperature, max_tokens=max_tokens
        ).text


def load_provider_profiles(
    config_path: str | Path = DEFAULT_PROVIDER_CONFIG,
) -> Dict[str, Dict[str, Any]]:
    path = Path(config_path)
    if not path.is_file():
        raise LLMConfigurationError(f"Provider config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMConfigurationError(f"Invalid provider config: {path}") from exc
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise LLMConfigurationError("Provider config must contain non-empty 'profiles'")
    return profiles


def create_llm_provider(
    profile_name: str,
    *,
    config_path: str | Path = DEFAULT_PROVIDER_CONFIG,
    timeout_s: float | None = 120.0,
) -> LLMProvider:
    """Create one configured provider without exposing credential values."""
    load_dotenv()
    profiles = load_provider_profiles(config_path)
    if profile_name not in profiles:
        raise LLMConfigurationError(
            f"Unknown LLM profile '{profile_name}'. Available: {', '.join(sorted(profiles))}"
        )
    profile = profiles[profile_name]
    provider_type = profile.get("provider")

    if provider_type == "azure_openai":
        def required_env(field: str) -> str:
            env_name = str(profile.get(field, ""))
            value = os.environ.get(env_name, "") if env_name else ""
            if not value.strip():
                raise LLMConfigurationError(
                    f"Missing required Azure OpenAI environment variable: {env_name or field}"
                )
            return value

        return AzureOpenAIBackend(
            endpoint=required_env("endpoint_env"),
            api_key=required_env("api_key_env"),
            deployment=required_env("deployment_env"),
            timeout_s=timeout_s,
        )

    if provider_type == "ollama":
        canonical_model = str(profile.get("model", "")).strip()
        if canonical_model not in {CANONICAL_LLAMA_MODEL, CANONICAL_GPT_OSS_MODEL}:
            raise LLMConfigurationError(
                f"Profile '{profile_name}' does not use a canonical Ollama model"
            )
        model_env = str(profile.get("model_env", ""))
        configured_model = os.environ.get(model_env, "").strip() if model_env else ""
        if configured_model and configured_model != canonical_model:
            raise LLMConfigurationError(
                f"{model_env} must be exactly '{canonical_model}' for profile '{profile_name}'"
            )
        base_env = str(profile.get("base_url_env", "OLLAMA_BASE_URL"))
        base_url = (
            os.environ.get(base_env, "").strip()
            or str(profile.get("default_base_url", DEFAULT_OLLAMA_BASE_URL))
        )
        return OllamaBackend(
            model_name=canonical_model,
            base_url=base_url,
            timeout_s=timeout_s,
        )

    raise LLMConfigurationError(
        f"Unsupported provider type '{provider_type}' in profile '{profile_name}'"
    )
