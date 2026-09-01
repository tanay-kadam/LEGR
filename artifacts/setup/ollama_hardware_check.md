# Ollama Hardware Check

- Timestamp (UTC): 2026-08-31T23:22:39.797984+00:00
- Operating system: Windows-10-10.0.26200-SP0
- CPU: Intel64 Family 6 Model 143 Stepping 8, GenuineIntel
- RAM: 127.25 GiB
- GPU: NVIDIA RTX 6000 Ada Generation, 49140 MiB, 596.36

## Model runtime

| Model | Inference | Latency (s) | Notes |
|---|---:|---:|---|
| `llama3.2:3b` | PASS | 2.721 | Successful local inference; execution device is managed by Ollama. |
| `gpt-oss:120b-cloud` | PASS | 2.792 | Successful Ollama Cloud inference; local GPU is not used for model execution. |

CUDA is not a setup requirement. Any CPU fallback reported by Ollama is accepted
but must remain visible in runtime diagnostics; model substitution is not allowed.
