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
from dataclasses import dataclass
from typing import Any, Literal

import httpx

DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

Role = Literal["system", "user", "assistant"]


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
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    raw: dict[str, Any] | None = None


class LLMClient:
    """Synchronous chat-completions client. One instance is fine for many calls."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout_s: float = 60.0,
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

        usage = payload.get("usage", {})
        return LLMResponse(
            content=content,
            model=payload.get("model", body["model"]),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            raw=payload,
        )

    def close(self) -> None:
        self._http.close()
