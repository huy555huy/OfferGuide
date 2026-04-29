"""Settings — env-var driven config loading."""

from __future__ import annotations

import pytest

from offerguide.config import Settings


def test_defaults_when_env_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "DEEPSEEK_API_KEY",
        "FEISHU_WEBHOOK_URL",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "OFFERGUIDE_NOTIFY",
        "OFFERGUIDE_RESUME_PDF",
    ):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.deepseek_api_key is None
    assert s.feishu_webhook_url is None
    assert s.notify_channel == "console"
    assert s.notify_ready() is True  # console always ready


def test_picks_up_deepseek_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    s = Settings.from_env()
    assert s.deepseek_api_key == "sk-test"


def test_notify_ready_for_each_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OFFERGUIDE_NOTIFY", "feishu")
    monkeypatch.setenv("FEISHU_WEBHOOK_URL", "https://x")
    s = Settings.from_env()
    assert s.notify_channel == "feishu"
    assert s.notify_ready() is True

    monkeypatch.setenv("OFFERGUIDE_NOTIFY", "telegram")
    monkeypatch.delenv("FEISHU_WEBHOOK_URL", raising=False)
    s = Settings.from_env()
    assert s.notify_ready() is False  # telegram needs both token + chat_id

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    s = Settings.from_env()
    assert s.notify_ready() is True


def test_notify_channel_falls_back_on_garbage_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OFFERGUIDE_NOTIFY", "skywriting")
    s = Settings.from_env()
    assert s.notify_channel == "console"
