"""Quick test to see raw GPT-OSS output and debug parse failures."""
import sys
sys.path.insert(0, "src")

from llm_backends import create_llm_provider

backend = create_llm_provider("ollama_gpt_oss", timeout_s=120.0)
print("Testing with max_tokens=1024...")

TOOLS = [
    "check_service_status", "create_support_ticket",
    "dispatch_message_to_usergroup", "edit_username",
    "escalate_case_to_human", "read_access_logs",
    "read_database_record", "read_subscription_status",
    "read_user_profile", "reset_user_password",
    "restart_service", "route_task_by_condition",
    "scan_system_for_malware", "update_subscription_plan",
    "write_database_record",
]

tool_list = "\n".join(f"  - {t}" for t in TOOLS)

SYSTEM = f"""You are an API workflow planner. Given a user query, output a JSON object with:
- "tools": a list of tool names needed (from the vocabulary below)
- "edges": a list of [source_index, target_index] pairs representing execution order

Tool vocabulary:
{tool_list}

Rules:
- Only use tools from the vocabulary above
- Indices in edges refer to positions in the tools list (0-based)
- The result must form a valid DAG (no cycles)
- Output ONLY valid JSON, no explanation or markdown"""

queries = [
    "Start by checking the service status. Once done, escalate the case and check the user profile simultaneously. After both finish, update the subscription.",
    "Read the database record and then write it back.",
    "Scan the system for malware, then restart the service.",
]

for i, q in enumerate(queries):
    print(f"\n{'='*60}")
    print(f"Query {i+1}: {q}")
    print(f"{'='*60}")
    resp = backend.call(SYSTEM, q, max_tokens=1024)
    print(f"RAW repr: {repr(resp.text[:500])}")
    print(f"DISPLAY:\n{resp.text}")
    print()

    import re, json
    cleaned = re.sub(r"```(?:json)?\s*", "", resp.text)
    cleaned = cleaned.replace("```", "").strip()
    try:
        data = json.loads(cleaned)
        print(f"PARSED OK: {data}")
    except json.JSONDecodeError as e:
        print(f"PARSE FAILED: {e}")
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                print(f"REGEX FALLBACK OK: {data}")
            except json.JSONDecodeError as e2:
                print(f"REGEX FALLBACK ALSO FAILED: {e2}")
                print(f"MATCHED TEXT: {json_match.group()[:300]}")
