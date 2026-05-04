"""Minimal LLM client for OpenAI-compatible endpoints — DeepSeek by default.

Why direct httpx and not the `openai` SDK: the openai package is a 50MB+
dependency and we use ~1% of its surface area. The chat completions schema
hasn't moved meaningfully in two years; a direct call keeps the dep tree clean
and behavior obvious.

Provider configuration is by env var (so SKILL helpers, tests, CI can all swap
in stubs without code changes):

    DEEPSEEK_API_KEY      # mandatory for DeepSeek calls
    DEEPSEEK_BASE_URL     # optional, defaults to https://api.deepseek.com
    OFFERGUIDE_DEFAULT_MODEL  # optional, defaults to deepseek-v4-flash

Confirmed model ids from https://api-docs.deepseek.com/quick_start/pricing
(2026-04-28): deepseek-v4-flash, deepseek-v4-pro, deepseek-chat, deepseek-reasoner.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .pricing import estimate_cost_usd as _estimate_cost  # noqa: E402

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

Role = Literal["system", "user", "assistant"]


def _parse_tool_arguments(args_raw: str) -> dict[str, Any]:
    """Parse a tool-call ``arguments`` JSON string, robust to ccvibe quirks.

    Standard OpenAI: ``arguments`` is a JSON-string of an object.
    ccvibe (Claude proxy) bug: sometimes returns CONCATENATED JSON objects
    like ``"{}{\\"city\\": \\"x\\"}"`` — strict ``json.loads`` raises
    ``Extra data`` and we'd lose the real arguments. Caught in W13 first
    dogfood (model called read_job 5 times, args always empty).

    Strategy:
      1. ``json.loads`` happy path
      2. Use ``raw_decode`` to walk forward, collecting each valid object
      3. Return the FIRST non-empty dict found (or the last one if all empty)
      4. On total failure, return empty dict — caller surfaces the error
         to the model so it can self-correct
    """
    if not args_raw:
        return {}
    s = args_raw.strip()
    # Happy path: clean single object
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: walk through concatenated objects
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    idx = 0
    while idx < len(s):
        # Skip whitespace
        while idx < len(s) and s[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(s):
            break
        try:
            obj, end = decoder.raw_decode(s, idx)
        except json.JSONDecodeError:
            # Couldn't parse from here — give up
            break
        if isinstance(obj, dict):
            found.append(obj)
        idx = end

    if not found:
        return {}
    # Prefer the first non-empty dict; fall back to the last one
    for obj in found:
        if obj:
            return obj
    return found[-1]


def _strip_md_codefence(s: str) -> str:
    """Strip ```...``` and ```json...``` fences from an LLM response.

    Used when json_mode=True to handle Claude-family models that wrap
    JSON in markdown. Returns ``s`` unchanged if no fences detected.

    Handles three common shapes:
        ```json\\n{...}\\n```
        ```\\n{...}\\n```
        leading/trailing whitespace + optional fences
    """
    text = s.strip()
    if not text.startswith("```"):
        return text
    # Strip the opening fence (```json\n or ```\n)
    nl_idx = text.find("\n")
    if nl_idx == -1:
        return text
    text = text[nl_idx + 1 :]
    # Strip the closing fence — find the last ``` and cut there
    close_idx = text.rfind("```")
    if close_idx != -1:
        text = text[:close_idx]
    return text.strip()


def _normalize_base_url(raw: str) -> str:
    """Make a user-supplied base URL POST-able as ``{base}/chat/completions``.

    DeepSeek's official endpoint accepts ``/chat/completions`` directly off
    the host root (``https://api.deepseek.com/chat/completions`` works). Most
    other OpenAI-compatible proxies (one-api, ccvibe, FastGPT, OpenRouter,
    Together, etc.) namespace their chat endpoint under ``/v1``. Auto-append
    ``/v1`` when the user gave us a bare host so a hand-rolled
    ``BASE_URL=https://my-proxy.example.com`` still works without surgery.

    Rules:
      - if path already contains ``/v1`` or ``/chat`` → leave alone
      - if it's the official DeepSeek host → leave alone
      - otherwise append ``/v1``
    """
    raw = raw.rstrip("/")
    if not raw:
        return DEFAULT_DEEPSEEK_BASE
    # Already has a versioned path — trust the user
    if "/v1" in raw or "/chat" in raw:
        return raw
    # Official DeepSeek endpoint accepts /chat/completions off the root
    if raw.startswith("https://api.deepseek.com"):
        return raw
    # Custom proxy with bare host — assume OpenAI-standard /v1 namespace
    return raw + "/v1"


class LLMError(RuntimeError):
    """Wraps any non-2xx / malformed response so callers can catch one type."""


@dataclass
class ToolCall:
    """One function-call the model decided to make.

    OpenAI/Claude tool-call payload:
      ``{"id": "...", "type": "function",
         "function": {"name": "skill_name", "arguments": "<json string>"}}``
    We surface ``arguments`` already-parsed as a dict (best-effort) — the
    model often returns malformed JSON for ``arguments``, so callers should
    handle missing/unexpected keys defensively.
    """
    id: str
    name: str
    arguments: dict[str, Any]
    arguments_raw: str = ""
    """Original JSON-string of arguments — preserved so we can echo it back
    verbatim in the assistant message when continuing the multi-turn
    tool-call loop (echoing parsed-then-re-serialized JSON loses key order
    and can confuse some providers' state tracking)."""


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw: dict[str, Any] | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    """When the model used a tool, this is non-empty AND ``content`` may be
    empty (or contain a brief 'thinking' preamble depending on provider).
    Empty when no tool was called."""
    finish_reason: str = ""
    """OpenAI-spec finish_reason: 'stop' | 'tool_calls' | 'length' | ...
    Useful for the agent loop to detect ``length`` (context cap hit) vs a
    clean stop."""


class LLMClient:
    """Synchronous chat-completions client. One instance is fine for many calls."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        raw_base = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or DEFAULT_DEEPSEEK_BASE
        ).rstrip("/")
        self.base_url = _normalize_base_url(raw_base)
        self.default_model = (
            default_model or os.environ.get("OFFERGUIDE_DEFAULT_MODEL") or DEFAULT_MODEL
        )
        self._http = httpx.Client(timeout=timeout_s)

    def chat(
        self,
        messages: list[Mapping[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError(
                "No API key configured. Set DEEPSEEK_API_KEY or pass api_key= to LLMClient."
            )
        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if extra:
            body.update(extra)

        t0 = time.monotonic()
        try:
            resp = self._http.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP transport error: {e}") from e

        latency_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            raise LLMError(
                f"LLM HTTP {resp.status_code}: {resp.text[:400]}"
            )
        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned non-JSON body: {e}") from e

        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"LLM response missing choices[0].message.content: {payload}") from e

        # Strip ``` ``` markdown fences from JSON-mode responses. Anthropic Claude
        # (whether direct or via OpenAI-compat proxies like ccvibe) wraps structured
        # outputs in ```json ... ``` even when response_format=json_object is sent —
        # OpenAI-spec only the latter, but Claude inherits the markdown habit. Centralize
        # the stripping here so every json_mode call site gets a clean parse downstream.
        if json_mode and content:
            content = _strip_md_codefence(content)

        usage = payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        actual_model = payload.get("model", body["model"])
        return LLMResponse(
            content=content,
            model=actual_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_estimate_cost(model=actual_model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            latency_ms=latency_ms,
            raw=payload,
        )

    def chat_with_tools(
        self,
        messages: list[Mapping[str, Any]],
        *,
        tools: list[Mapping[str, Any]],
        model: str | None = None,
        temperature: float = 0.4,
        tool_choice: str = "auto",
        cache_system_prompt: bool = True,
        extra: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Chat with OpenAI-spec function/tool calling.

        Use this — not ``chat()`` — when you want the model to **decide**
        which tool to call. The agent loop in ``offerguide.agent.loop``
        is the primary user. ``tool_choice``:

          - ``"auto"`` (default): model picks tool or replies directly
          - ``"none"``: model must reply directly
          - ``"required"``: model must pick a tool (some providers ignore)

        ``cache_system_prompt=True`` (W13.8): tag the system message with
        Anthropic's prompt-cache marker so subsequent iterations within
        the same trajectory hit the prompt cache (50% input-token cost
        reduction past the first call). ccvibe / OpenAI-compat proxies
        that don't support cache_control silently ignore the marker —
        no behavioral degradation either way.

        Returns LLMResponse where ``tool_calls`` is non-empty if the model
        called a tool; ``content`` is the (possibly empty) text the model
        emitted alongside the tool call. Caller is responsible for
        appending the assistant message + tool results to ``messages`` and
        looping back here for the next decision.
        """
        if not self.api_key:
            raise LLMError(
                "No API key configured. Set OFFERGUIDE_LLM_API_KEY (or DEEPSEEK_API_KEY)."
            )
        # Mutate-copy so we can stamp cache_control on the system msg
        prepared_messages = [dict(m) for m in messages]
        if cache_system_prompt and prepared_messages:
            sys_msg = prepared_messages[0]
            if sys_msg.get("role") == "system" and isinstance(sys_msg.get("content"), str):
                # Convert content from string to content-block list with cache_control
                # marker. Anthropic & OpenAI both accept this shape; non-Anthropic
                # backends just ignore the cache_control field.
                sys_msg["content"] = [
                    {
                        "type": "text",
                        "text": sys_msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    },
                ]

        body: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": prepared_messages,
            "temperature": temperature,
            "stream": False,
            "tools": [dict(t) for t in tools],
            "tool_choice": tool_choice,
        }
        if extra:
            body.update(extra)

        t0 = time.monotonic()
        try:
            resp = self._http.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
        except httpx.HTTPError as e:
            raise LLMError(f"HTTP transport error: {e}") from e

        latency_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code != 200:
            raise LLMError(
                f"LLM HTTP {resp.status_code}: {resp.text[:600]}"
            )
        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned non-JSON body: {e}") from e

        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"LLM response missing choices[0].message: {payload}") from e

        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason") or ""

        # Parse tool_calls (OpenAI canonical shape; Claude-via-proxy returns same)
        tool_calls: list[ToolCall] = []
        for tc in (message.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or ""
            args_parsed: dict[str, Any] = {}
            if isinstance(args_raw, str) and args_raw:
                # ccvibe / Claude proxies sometimes emit concatenated JSON
                # objects in ``arguments`` — use the robust parser, not strict
                # json.loads. See ``_parse_tool_arguments`` docstring.
                args_parsed = _parse_tool_arguments(args_raw)
            elif isinstance(args_raw, dict):
                args_parsed = args_raw
                args_raw = json.dumps(args_raw, ensure_ascii=False)
            tool_calls.append(ToolCall(
                id=str(tc.get("id") or f"call_{len(tool_calls)}"),
                name=str(fn.get("name") or ""),
                arguments=args_parsed,
                arguments_raw=args_raw if isinstance(args_raw, str) else "",
            ))

        usage = payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        actual_model = payload.get("model", body["model"])
        return LLMResponse(
            content=content,
            model=actual_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=_estimate_cost(model=actual_model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            latency_ms=latency_ms,
            raw=payload,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    def close(self) -> None:
        self._http.close()
