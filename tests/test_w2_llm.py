"""LLM client tests — mock httpx so we never make a real API call in CI."""

from __future__ import annotations

import json

import httpx
import pytest

from offerguide.llm import LLMClient, LLMError


def _client_with_transport(handler) -> LLMClient:
    """Build an LLMClient backed by httpx.MockTransport."""
    transport = httpx.MockTransport(handler)
    c = LLMClient(api_key="test-key", base_url="https://api.deepseek.com")
    c._http = httpx.Client(transport=transport, timeout=10.0)
    return c


def test_chat_posts_to_chat_completions_with_correct_shape() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": "hi back"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    c = _client_with_transport(handler)
    resp = c.chat([{"role": "user", "content": "hi"}], temperature=0.5)

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["body"]["model"] == "deepseek-v4-flash"  # default
    assert captured["body"]["temperature"] == 0.5
    assert captured["body"]["stream"] is False
    assert "response_format" not in captured["body"]  # json_mode defaults False

    assert resp.content == "hi back"
    assert resp.prompt_tokens == 5
    assert resp.completion_tokens == 2


def test_json_mode_sets_response_format() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "deepseek-v4-flash",
                "choices": [{"message": {"content": '{"x": 1}'}}],
            },
        )

    c = _client_with_transport(handler)
    c.chat([{"role": "user", "content": "json please"}], json_mode=True)

    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_chat_raises_on_non_2xx() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream busted")

    c = _client_with_transport(handler)
    with pytest.raises(LLMError, match="HTTP 500"):
        c.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_on_missing_choices() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m"})

    c = _client_with_transport(handler)
    with pytest.raises(LLMError, match="missing choices"):
        c.chat([{"role": "user", "content": "hi"}])


def test_chat_raises_when_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    c = LLMClient()
    with pytest.raises(LLMError, match="No API key"):
        c.chat([{"role": "user", "content": "hi"}])


def test_default_model_overridable_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OFFERGUIDE_DEFAULT_MODEL", "deepseek-v4-pro")
    c = LLMClient(api_key="x")
    assert c.default_model == "deepseek-v4-pro"
