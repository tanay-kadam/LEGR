from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import BaseModel

import llm_backends
from llm_backends import (
    CANONICAL_GPT_OSS_MODEL,
    CANONICAL_LLAMA_MODEL,
    AzureOpenAIBackend,
    LLMConfigurationError,
    OllamaBackend,
    create_llm_provider,
    load_provider_profiles,
    safe_error_message,
)


class _Choice(BaseModel):
    tool: str
    reasoning: str = ""


def test_provider_config_locks_exact_model_identifiers():
    profiles = load_provider_profiles()
    assert profiles["ollama_llama"]["model"] == CANONICAL_LLAMA_MODEL
    assert profiles["ollama_gpt_oss"]["model"] == CANONICAL_GPT_OSS_MODEL


def test_factory_rejects_noncanonical_model_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_LLAMA_MODEL", "llama3.2")
    with pytest.raises(LLMConfigurationError, match="must be exactly"):
        create_llm_provider("ollama_llama")


def test_factory_reports_missing_azure_variable_names_without_values(monkeypatch):
    monkeypatch.setattr(llm_backends, "load_dotenv", lambda: None)
    for name in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMConfigurationError, match="AZURE_OPENAI_ENDPOINT"):
        create_llm_provider("azure_openai")


def test_ollama_backend_normalizes_text_metadata_and_options(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def chat(self, **kwargs):
            captured["chat"] = kwargs
            return SimpleNamespace(
                message=SimpleNamespace(content='{"tool":"query_database"}'),
                prompt_eval_count=7,
                eval_count=3,
            )

    fake_module = ModuleType("ollama")
    fake_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "ollama", fake_module)

    backend = OllamaBackend(
        model_name=CANONICAL_LLAMA_MODEL,
        base_url="http://localhost:11434/",
        timeout_s=5,
    )
    response = backend.call(
        "select a tool",
        "read the record",
        _Choice,
        temperature=0.0,
        max_tokens=32,
    )

    assert response.text == '{"tool":"query_database"}'
    assert response.parsed.tool == "query_database"
    assert response.provider == "ollama"
    assert response.model == CANONICAL_LLAMA_MODEL
    assert response.usage.total_tokens == 10
    assert captured["init"]["host"] == "http://localhost:11434"
    assert captured["chat"]["options"] == {"temperature": 0.0, "num_predict": 32}


def test_azure_backend_normalizes_response_without_printing_key(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"tool":"query_database","reasoning":"test"}'
                    ),
                    finish_reason="stop",
                )],
                usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_module = ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    secret = "never-print-this-key"
    backend = AzureOpenAIBackend(
        api_key=secret,
        endpoint="https://sample.openai.azure.com/",
        deployment="routing-deployment",
        timeout_s=10,
    )
    response = backend.call(
        "Select a tool",
        "Look up the record",
        response_schema=_Choice,
        max_tokens=8,
    )

    assert response.parsed.tool == "query_database"
    assert captured["request"]["model"] == "routing-deployment"
    assert captured["init"]["base_url"] == "https://sample.openai.azure.com/openai/v1/"
    schema = captured["request"]["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"tool", "reasoning"}
    assert secret not in str(backend.__dict__)


def test_safe_error_message_redacts_environment_credentials(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "super-secret-value")
    message = safe_error_message(RuntimeError("failed with super-secret-value"))
    assert "super-secret-value" not in message
    assert "[REDACTED]" in message
