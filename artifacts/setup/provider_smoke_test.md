# Provider Smoke Test

Timestamp (UTC): 2026-08-31T23:23:53.152353+00:00

| Profile | Provider | Model/Deployment | Generation | Integration | Status | Latency (s) |
|---|---|---|---:|---:|---:|---:|
| azure_openai | - | - | FAIL | FAIL | FAIL | 0.001 |

- `azure_openai` error (configuration): Missing required Azure OpenAI environment variable: AZURE_OPENAI_ENDPOINT
| ollama_llama | ollama | llama3.2:3b | PASS | PASS | PASS | 2.926 |
| ollama_gpt_oss | ollama | gpt-oss:120b-cloud | PASS | PASS | PASS | 5.144 |
