# LEGR Model Integration Audit

Date: 2026-08-31

## Existing integration

- `src/llm_backends.py` already provided `LLMResponse`, token accounting,
  Gemini calls, and an `OllamaBackend` based on the Ollama Python client.
- `src/main.py` passed the Ollama backend through `evaluator.py` into the
  two-stage router in `routers.py`.
- `src/dag_extract.py` and `src/llm_dag_baseline.py` also called the same
  Ollama backend for execution-graph generation.
- `.env` loading already used `python-dotenv`; Gemini and legacy Ollama
  selection used environment variables.
- Routing output parsing already constrained choices with Pydantic schemas and
  rejected invalid branch/tool selections. DAG parsing already handled plain
  and fenced JSON and surfaced empty predictions as parse failures.

## Reused

- The existing backend response type, Ollama client path, router, evaluator,
  structured Pydantic schemas, DAG parser, and dotenv loading are retained.
- Gemini compatibility remains available and is not part of this setup's live
  acceptance criteria.

## Gaps found before changes

- No Azure OpenAI client, provider factory, or Azure environment contract.
- No configuration file that selected a provider without editing Python.
- Active defaults used `llama3.2`, and provider/model selection was not
  centralized or validated.
- `.gitignore` covered `.env` but not `.env.local` or `.env.*.local`.
- The active virtual environment lacked both the `openai` and `ollama` Python
  packages even though `ollama` was listed in requirements.
- The Ollama executable was absent and `localhost:11434` refused connections.
- No setup smoke-test scripts or setup result artifacts existed.

## Changes made

- Normalized Azure OpenAI and Ollama behind a shared provider protocol and
  factory while preserving `OllamaBackend.call(...)`.
- Azure uses the v1 endpoint (`/openai/v1/`) with endpoint, API key, and model
  deployment; no dated `api-version` value is required.
- Added configuration profiles for `azure_openai`, `ollama_llama`, and
  `ollama_gpt_oss`.
- Locked configured Ollama model IDs to `llama3.2:3b` and the user-selected
  `gpt-oss:120b-cloud`. The GPT-OSS profile uses Ollama Cloud rather than local
  inference.
- Added secret-safe environment templates, validation, reports, smoke tests,
  and configuration-driven pipeline selection.

No datasets, training code, benchmark runs, or experiment results are changed
by this setup.
