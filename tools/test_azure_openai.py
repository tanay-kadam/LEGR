"""Run one minimal Azure OpenAI inference and write a secret-safe report."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from llm_backends import (
    categorize_llm_error,
    create_llm_provider,
    endpoint_hostname,
    safe_error_message,
)


REPORT_PATH = ROOT / "artifacts" / "setup" / "azure_openai_check.txt"


def main() -> int:
    load_dotenv(ROOT / ".env")
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL",
        "endpoint_hostname": endpoint_hostname(
            os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        ) or "NOT_CONFIGURED",
        "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "NOT_CONFIGURED"),
        "api_mode": "v1",
        "response_received": "false",
        "exact_marker": "false",
        "error_category": "",
        "error": "",
    }
    try:
        provider = create_llm_provider("azure_openai", timeout_s=60.0)
        text = provider.generate(
            "Reply with exactly: AZURE_OK", temperature=0.0, max_tokens=16
        ).strip()
        result["response_received"] = str(bool(text)).lower()
        result["exact_marker"] = str("AZURE_OK" in text.upper()).lower()
        if text:
            result["status"] = "PASS"
        else:
            result["error_category"] = "empty_response"
            result["error"] = "Azure returned an empty text response"
    except Exception as exc:
        result["error_category"] = categorize_llm_error(exc)
        result["error"] = safe_error_message(exc)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(f"{key}: {value}" for key, value in result.items()) + "\n",
        encoding="utf-8",
    )
    print(f"Azure OpenAI: {result['status']}")
    print(f"Report: {REPORT_PATH}")
    if result["error"]:
        print(f"Error category: {result['error_category']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
