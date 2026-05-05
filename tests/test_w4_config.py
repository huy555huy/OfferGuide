"""Settings — env-var driven config loading."""

from __future__ import annotations

import pytest

from offerguide.config import Settings, _load_dotenv_if_present


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


# ────────── W14.10 dotenv autoload ──────────


class TestDotenvAutoload:
    """The .env file in the project root is auto-loaded by Settings.from_env()
    so users don't need to remember to `source .env`. The README has always
    told them to put credentials in .env; previously nothing read it."""

    def test_loads_keys_from_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OFFERGUIDE_SKIP_DOTENV", raising=False)
        monkeypatch.delenv("MYAPP_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("MYAPP_KEY=injected_value\n", encoding="utf-8")
        n = _load_dotenv_if_present(env)
        assert n == 1
        import os
        assert os.environ.get("MYAPP_KEY") == "injected_value"

    def test_does_not_override_existing_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OFFERGUIDE_SKIP_DOTENV", raising=False)
        monkeypatch.setenv("MYAPP_KEY", "shell_wins")
        env = tmp_path / ".env"
        env.write_text("MYAPP_KEY=dotenv_value\n", encoding="utf-8")
        n = _load_dotenv_if_present(env)
        assert n == 0  # nothing injected — pre-existing wins
        import os
        assert os.environ["MYAPP_KEY"] == "shell_wins"

    def test_strips_matched_quotes(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OFFERGUIDE_SKIP_DOTENV", raising=False)
        monkeypatch.delenv("Q1", raising=False)
        monkeypatch.delenv("Q2", raising=False)
        monkeypatch.delenv("Q3", raising=False)
        env = tmp_path / ".env"
        env.write_text(
            'Q1="double quoted"\n'
            "Q2='single quoted'\n"
            'Q3=no quotes\n',
            encoding="utf-8",
        )
        _load_dotenv_if_present(env)
        import os
        assert os.environ["Q1"] == "double quoted"
        assert os.environ["Q2"] == "single quoted"
        assert os.environ["Q3"] == "no quotes"

    def test_skips_comments_and_blank_lines(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OFFERGUIDE_SKIP_DOTENV", raising=False)
        monkeypatch.delenv("REAL_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text(
            "# this is a comment\n"
            "\n"
            "REAL_KEY=value\n"
            "  # indented comment\n",
            encoding="utf-8",
        )
        n = _load_dotenv_if_present(env)
        assert n == 1
        import os
        assert os.environ["REAL_KEY"] == "value"

    def test_skip_dotenv_env_bypasses_loading(self, tmp_path, monkeypatch):
        """Critical: tests must never accidentally read the developer's real
        .env (would exfiltrate API keys into pytest output / CI logs).
        OFFERGUIDE_SKIP_DOTENV=1 (set in tests/conftest.py) must short-circuit."""
        monkeypatch.setenv("OFFERGUIDE_SKIP_DOTENV", "1")
        monkeypatch.delenv("WOULD_BE_LOADED", raising=False)
        env = tmp_path / ".env"
        env.write_text("WOULD_BE_LOADED=secret\n", encoding="utf-8")
        n = _load_dotenv_if_present(env)
        assert n == 0
        import os
        assert "WOULD_BE_LOADED" not in os.environ

    def test_missing_file_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OFFERGUIDE_SKIP_DOTENV", raising=False)
        n = _load_dotenv_if_present(tmp_path / "nonexistent.env")
        assert n == 0
