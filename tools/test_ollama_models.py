"""Verify the Ollama service and both canonical local models."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from llm_backends import (
    CANONICAL_GPT_OSS_MODEL,
    CANONICAL_LLAMA_MODEL,
    categorize_llm_error,
    create_llm_provider,
    safe_error_message,
)


REPORT_PATH = ROOT / "artifacts" / "setup" / "ollama_model_check.txt"
HARDWARE_PATH = ROOT / "artifacts" / "setup" / "ollama_hardware_check.md"


def _run(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return completed.returncode, (completed.stdout or completed.stderr).strip()
    except Exception as exc:
        return 1, safe_error_message(exc)


def _ollama_executable() -> str:
    discovered = shutil.which("ollama")
    if discovered:
        return discovered
    if os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return str(candidate)
    return "ollama"


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _ram_gib() -> float | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.ullTotalPhys / (1024 ** 3), 2)


def _write_hardware_report(model_results: list[dict]) -> None:
    _, gpu = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    cpu = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")
    lines = [
        "# Ollama Hardware Check",
        "",
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Operating system: {platform.platform()}",
        f"- CPU: {cpu}",
        f"- RAM: {_ram_gib() or 'unknown'} GiB",
        f"- GPU: {gpu or 'not detected'}",
        "",
        "## Model runtime",
        "",
        "| Model | Inference | Latency (s) | Notes |",
        "|---|---:|---:|---|",
    ]
    for item in model_results:
        if item.get("error"):
            notes = item["error"]
        elif item["model"].endswith("-cloud"):
            notes = "Successful Ollama Cloud inference; local GPU is not used for model execution."
        else:
            notes = "Successful local inference; execution device is managed by Ollama."
        lines.append(
            f"| `{item['model']}` | {item['status']} | "
            f"{item.get('latency_s', '')} | {notes} |"
        )
    lines.extend([
        "",
        "CUDA is not a setup requirement. Any CPU fallback reported by Ollama is accepted",
        "but must remain visible in runtime diagnostics; model substitution is not allowed.",
    ])
    HARDWARE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    load_dotenv(ROOT / ".env")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    version_rc, version_text = _run([_ollama_executable(), "--version"])
    installed: list[str] = []
    api_version = ""
    service_error = ""
    try:
        api_version = str(_get_json(f"{base_url}/api/version").get("version", ""))
        installed = [
            str(model.get("name", ""))
            for model in _get_json(f"{base_url}/api/tags").get("models", [])
        ]
    except Exception as exc:
        service_error = safe_error_message(exc)

    tests = [
        ("ollama_llama", CANONICAL_LLAMA_MODEL, "Reply with exactly: LLAMA_OK"),
        ("ollama_gpt_oss", CANONICAL_GPT_OSS_MODEL, "Reply with exactly: GPT_OSS_OK"),
    ]
    model_results = []
    for profile, model, prompt in tests:
        item = {
            "model": model,
            "installed": model in installed,
            "status": "FAIL",
            "latency_s": "",
            "response_received": False,
            "exact_marker": False,
            "error_category": "",
            "error": "",
        }
        if not item["installed"]:
            item["error_category"] = "deployment_or_model_not_found"
            item["error"] = f"Required model is not installed: {model}"
        elif service_error:
            item["error_category"] = "connection"
            item["error"] = service_error
        else:
            started = time.perf_counter()
            try:
                token_limit = 256 if model == CANONICAL_GPT_OSS_MODEL else 32
                text = create_llm_provider(profile, timeout_s=600.0).generate(
                    prompt, temperature=0.0, max_tokens=token_limit
                ).strip()
                item["latency_s"] = round(time.perf_counter() - started, 3)
                item["response_received"] = bool(text)
                expected = "LLAMA_OK" if model == CANONICAL_LLAMA_MODEL else "GPT_OSS_OK"
                item["exact_marker"] = expected in text.upper()
                if text:
                    item["status"] = "PASS"
                else:
                    item["error_category"] = "empty_response"
                    item["error"] = "Ollama returned an empty text response"
            except Exception as exc:
                item["latency_s"] = round(time.perf_counter() - started, 3)
                item["error_category"] = categorize_llm_error(exc)
                item["error"] = safe_error_message(exc)
        model_results.append(item)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}",
        f"ollama_cli_status: {'PASS' if version_rc == 0 else 'FAIL'}",
        f"ollama_version_cli: {version_text or 'NOT_AVAILABLE'}",
        f"ollama_version_api: {api_version or 'NOT_AVAILABLE'}",
        f"base_url: {base_url}",
        f"service_status: {'PASS' if api_version else 'FAIL'}",
        f"service_error: {service_error}",
        "installed_models: " + (", ".join(installed) if installed else "NONE"),
    ]
    for item in model_results:
        prefix = item["model"]
        for key in (
            "installed", "status", "latency_s", "response_received",
            "exact_marker", "error_category", "error",
        ):
            lines.append(f"{prefix}.{key}: {item[key]}")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_hardware_report(model_results)

    overall = version_rc == 0 and bool(api_version) and all(
        item["status"] == "PASS" for item in model_results
    )
    print(f"Ollama models: {'PASS' if overall else 'FAIL'}")
    print(f"Report: {REPORT_PATH}")
    print(f"Hardware: {HARDWARE_PATH}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
