"""Smoke-test all configured providers through the shared LEGR router."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from evaluator import _run_taxonomy
from llm_backends import (
    categorize_llm_error,
    create_llm_provider,
    safe_error_message,
)


JSON_PATH = ROOT / "artifacts" / "setup" / "provider_smoke_test.json"
MD_PATH = ROOT / "artifacts" / "setup" / "provider_smoke_test.md"

SMOKE_TAXONOMY = {
    "name": "Provider Integration Smoke",
    "branches": {
        "Data Retrieval": {
            "description": "Read or inspect existing state without modifying it.",
            "tools": ["check_status", "query_database", "generate_report"],
        }
    },
}
SMOKE_DATA = pd.DataFrame([
    {
        "query": "Look up the existing database record for customer 42.",
        "ground_truth": "query_database",
    }
])


def main() -> int:
    load_dotenv(ROOT / ".env")
    rows = []
    for profile in ("azure_openai", "ollama_llama", "ollama_gpt_oss"):
        row = {
            "profile": profile,
            "provider": "",
            "model_or_deployment": "",
            "generation_status": "FAIL",
            "integration_status": "FAIL",
            "status": "FAIL",
            "latency_s": None,
            "prediction": None,
            "error_category": "",
            "error": "",
        }
        started = time.perf_counter()
        try:
            provider = create_llm_provider(profile, timeout_s=600.0)
            row["provider"] = provider.provider_name
            row["model_or_deployment"] = provider.model_name
            text = provider.generate(
                "Reply with exactly: PROVIDER_OK", temperature=0.0, max_tokens=256
            ).strip()
            if not text:
                raise RuntimeError("Provider returned an empty response")
            row["generation_status"] = "PASS"

            metrics, records = _run_taxonomy(
                SMOKE_TAXONOMY,
                SMOKE_DATA,
                llm_backend=provider,
                inter_query_delay=0.0,
            )
            record = records[0]
            row["prediction"] = record.predicted_tool
            if (
                metrics.get("total_queries") == 1
                and record.error is None
                and record.predicted_tool is not None
            ):
                row["integration_status"] = "PASS"
                row["status"] = "PASS"
            else:
                row["error_category"] = "integration_or_parse_failure"
                row["error"] = record.error or "Router returned no prediction"
        except Exception as exc:
            row["error_category"] = categorize_llm_error(exc)
            row["error"] = safe_error_message(exc)
        row["latency_s"] = round(time.perf_counter() - started, 3)
        rows.append(row)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": all(row["status"] == "PASS" for row in rows),
        "providers": rows,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Provider Smoke Test",
        "",
        f"Timestamp (UTC): {payload['timestamp_utc']}",
        "",
        "| Profile | Provider | Model/Deployment | Generation | Integration | Status | Latency (s) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['profile']} | {row['provider'] or '-'} | "
            f"{row['model_or_deployment'] or '-'} | {row['generation_status']} | "
            f"{row['integration_status']} | {row['status']} | {row['latency_s']} |"
        )
        if row["error"]:
            lines.append(
                f"\n- `{row['profile']}` error ({row['error_category']}): {row['error']}"
            )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Provider smoke test: {'PASS' if payload['all_passed'] else 'FAIL'}")
    print(f"JSON: {JSON_PATH}")
    print(f"Markdown: {MD_PATH}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
