"""Dump the raw OpenAI-spec payload from ccvibe for one chat_with_tools call.

We ask Claude to call a single trivially-shaped tool. Whatever Claude/ccvibe
sends back in `choices[0].message.tool_calls`, we print verbatim — so we can
see if `arguments` is a string-of-JSON, a nested dict, or something else."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ENV = Path(__file__).parent.parent / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import httpx  # noqa: E402

api_key = os.environ["OFFERGUIDE_LLM_API_KEY"]
base_url = os.environ["OFFERGUIDE_LLM_BASE_URL"]
model = os.environ["OFFERGUIDE_LLM_MODEL"]

# Use the same /v1 path-resolution logic as LLMClient
if "/v1" not in base_url and "/chat" not in base_url:
    base_url = base_url + "/v1"

print(f"[debug] POST {base_url}/chat/completions  model={model}")

resp = httpx.post(
    f"{base_url}/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You can call the get_weather tool. When the user asks for weather, call it with the city."},
            {"role": "user", "content": "What's the weather in 北京 today?"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name in Chinese"},
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }],
        "tool_choice": "auto",
        "temperature": 0,
        "stream": False,
    },
    timeout=60.0,
)

print(f"[debug] HTTP {resp.status_code}")
print()
data = resp.json()

print("=== full payload ===")
print(json.dumps(data, ensure_ascii=False, indent=2))

print("\n=== inspecting tool_calls ===")
choices = data.get("choices", [])
if choices:
    msg = choices[0].get("message", {})
    print(f"finish_reason: {choices[0].get('finish_reason')}")
    print(f"content: {msg.get('content')!r}")
    tc = msg.get("tool_calls", [])
    print(f"tool_calls count: {len(tc)}")
    for i, t in enumerate(tc):
        print(f"\n--- tool_call[{i}] ---")
        print(json.dumps(t, ensure_ascii=False, indent=2))
        fn = t.get("function", {})
        args_raw = fn.get("arguments")
        print(f"\n  arguments type: {type(args_raw).__name__}")
        print(f"  arguments value: {args_raw!r}")
        if isinstance(args_raw, str):
            try:
                parsed = json.loads(args_raw)
                print(f"  ✓ parsed json: {parsed!r}")
            except json.JSONDecodeError as e:
                print(f"  ✗ json decode error: {e}")
