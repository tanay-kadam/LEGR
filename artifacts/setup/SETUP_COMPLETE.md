# LEGR Model Setup Handoff

Overall status: **PASS**

## Azure OpenAI

Status: **PASS**

Configured variables required:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_DEPLOYMENT
```

The provider uses Azure OpenAI v1 at `/openai/v1/`; no dated API-version value
is required. A live request to deployment `gpt-4o` returned the exact
`AZURE_OK` marker. Credentials remained in the gitignored `.env` and were not
written to the report.

After adding them to the gitignored `.env`, run:

```powershell
.\venv\Scripts\python.exe tools\test_azure_openai.py
.\venv\Scripts\python.exe tools\test_llm_providers.py
```

## Ollama

Status: **PASS**

Ollama version: `0.33.2`

Installed and verified targets:

```text
llama3.2:3b
gpt-oss:120b-cloud
```

`llama3.2:3b` completed local inference. Per the user's updated requirement,
GPT-OSS uses Ollama Cloud rather than the local `gpt-oss:20b` model and
completed cloud inference through the same Ollama client.

## Provider smoke tests

- Azure OpenAI direct inference: **PASS**
- Llama 3.2 3B: **PASS**
- GPT-OSS 120B Cloud: **PASS**

## LEGR integration smoke test

- Ollama / `llama3.2:3b`: **PASS**
- Ollama / `gpt-oss:120b-cloud`: **PASS**
- Azure OpenAI / `gpt-4o`: **PASS**

All three required profiles completed live inference. Each profile also
completed the shared provider → router → parser → evaluator path and returned
the expected `query_database` prediction.

## Regression and security

- Test suite: **PASS** — 146 passed, 1 expected failure.
- `.env`, `.env.local`, and `.env.*.local`: **ignored**.
- Credential-pattern scan of relevant source/config/report files: **PASS**.
- Dataset generation, training, full benchmarks, and experiment results: **not run or modified**.

## Manager review

| Check | Answer |
|---|---:|
| Azure is used only as an API service | YES |
| Azure credentials are externalized | YES |
| Actual Azure inference succeeded | YES |
| `llama3.2:3b` is installed and callable | YES |
| User-selected `gpt-oss:120b-cloud` is installed and callable | YES |
| Exact model identifiers are documented and configured | YES |
| LEGR can invoke both Ollama targets | YES |
| LEGR can invoke Azure OpenAI v1 | YES |
| Provider selection is configuration-driven | YES |
| Failures are surfaced clearly | YES |
| Secrets are protected | YES |

## Completion

No setup blocker remains. The stored aggregate provider-smoke artifact can be
refreshed later if a single-timestamp rerun of all three providers is desired.
