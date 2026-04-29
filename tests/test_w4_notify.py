"""Notifier adapters — mock httpx for feishu/telegram, capture stdout for console."""

from __future__ import annotations

import io
import json

import httpx
import pytest

from offerguide.config import Settings
from offerguide.ui.notify import (
    ConsoleNotifier,
    FeishuNotifier,
    TelegramNotifier,
    make_notifier,
)

# ---- console ----


def test_console_notifier_writes_lines() -> None:
    buf = io.StringIO()
    n = ConsoleNotifier(stream=buf)
    res = n.notify(title="hello", body="line1\nline2", level="warn")
    assert res.ok is True
    assert res.channel == "console"
    out = buf.getvalue()
    assert "[WARN] hello" in out
    assert "line1" in out
    assert "line2" in out


# ---- feishu ----


def _patch_feishu(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    def fake_init(self, webhook_url, *, timeout_s: float = 10.0):
        if not webhook_url:
            raise ValueError("FeishuNotifier needs a webhook URL")
        self._url = webhook_url
        self._http = httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout_s)

    monkeypatch.setattr(FeishuNotifier, "__init__", fake_init)


def test_feishu_posts_text_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"code": 0, "msg": "ok"})

    _patch_feishu(monkeypatch, handler)
    n = FeishuNotifier("https://open.feishu.cn/open-apis/bot/v2/hook/T")
    res = n.notify(title="新 JD", body="match=0.7", level="info")

    assert res.ok is True
    assert res.channel == "feishu"
    assert captured["url"].endswith("/T")
    assert captured["body"]["msg_type"] == "text"
    assert "新 JD" in captured["body"]["content"]["text"]
    assert "match=0.7" in captured["body"]["content"]["text"]


def test_feishu_returns_error_on_non_zero_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 1234, "msg": "bot disabled"})

    _patch_feishu(monkeypatch, handler)
    n = FeishuNotifier("https://x")
    res = n.notify(title="t", body="b")
    assert res.ok is False
    assert "bot disabled" in (res.error or "")


def test_feishu_returns_error_on_http_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream busted")

    _patch_feishu(monkeypatch, handler)
    n = FeishuNotifier("https://x")
    res = n.notify(title="t", body="b")
    assert res.ok is False
    assert "503" in (res.error or "")


def test_feishu_init_rejects_blank_url() -> None:
    with pytest.raises(ValueError):
        FeishuNotifier("")


# ---- telegram ----


def _patch_telegram(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    def fake_init(self, *, bot_token: str, chat_id: str, timeout_s: float = 10.0):
        if not bot_token or not chat_id:
            raise ValueError("TelegramNotifier needs bot_token and chat_id")
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._http = httpx.Client(transport=httpx.MockTransport(handler), timeout=timeout_s)

    monkeypatch.setattr(TelegramNotifier, "__init__", fake_init)


def test_telegram_posts_with_html_parse_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})

    _patch_telegram(monkeypatch, handler)
    n = TelegramNotifier(bot_token="bot:abc", chat_id="999")
    res = n.notify(title="新 JD <Hot>", body="prob=0.8", level="high")

    assert res.ok is True
    assert "/botbot:abc/sendMessage" in captured["url"]
    assert captured["body"]["chat_id"] == "999"
    assert captured["body"]["parse_mode"] == "HTML"
    assert "&lt;Hot&gt;" in captured["body"]["text"]  # escaped
    assert "🔥" in captured["body"]["text"]  # high level prefix


def test_telegram_returns_error_when_api_says_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": False, "error_code": 400, "description": "chat not found"}
        )

    _patch_telegram(monkeypatch, handler)
    n = TelegramNotifier(bot_token="x", chat_id="y")
    res = n.notify(title="t", body="b")
    assert res.ok is False
    assert "chat not found" in (res.error or "")


def test_telegram_truncates_long_body(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"ok": True})

    _patch_telegram(monkeypatch, handler)
    n = TelegramNotifier(bot_token="x", chat_id="y")
    n.notify(title="t", body="a" * 10000)
    assert "(truncated)" in captured["body"]["text"]
    assert len(captured["body"]["text"]) <= 4096


def test_telegram_init_rejects_missing_pieces() -> None:
    with pytest.raises(ValueError):
        TelegramNotifier(bot_token="", chat_id="x")
    with pytest.raises(ValueError):
        TelegramNotifier(bot_token="x", chat_id="")


# ---- factory ----


def test_make_notifier_falls_back_to_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """If config says feishu but URL missing, factory falls back to console."""
    s = Settings(notify_channel="feishu")
    n = make_notifier(s)
    assert n.name == "console"


def test_make_notifier_picks_feishu_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(
        notify_channel="feishu",
        feishu_webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/T",
    )
    n = make_notifier(s)
    assert n.name == "feishu"


def test_make_notifier_picks_telegram_when_configured() -> None:
    s = Settings(
        notify_channel="telegram",
        telegram_bot_token="bot:x",
        telegram_chat_id="123",
    )
    n = make_notifier(s)
    assert n.name == "telegram"
