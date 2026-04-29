"""Runtime configuration — env-var driven, single source of truth.

Why env vars rather than YAML for W4: nothing here is structurally complex
enough to deserve a config file. Notifier choice is one enum, API keys are
opaque strings, paths are paths. YAML can land in W7 if ambient scheduling
needs more knobs.

Settings.from_env() is called once at app/CLI startup; every component takes
the resulting Settings object explicitly so tests can pass synthetic configs
without monkey-patching the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

NotifyChannel = Literal["console", "feishu", "telegram"]


@dataclass(frozen=True)
class Settings:
    """All runtime configuration. Frozen → safe to share across threads/coroutines."""

    # LLM
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    default_model: str = "deepseek-v4-flash"

    # Notification — at most one of feishu/telegram is used per send,
    # picked from `notify_channel` (or per-call override).
    feishu_webhook_url: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    notify_channel: NotifyChannel = "console"

    # Storage
    db_path: Path = Path(".offerguide/store.db")
    resume_pdf: Path | None = None

    # Web UI
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    @classmethod
    def from_env(cls) -> Settings:
        """Build a Settings from environment variables. Missing → defaults."""
        notify_raw = os.environ.get("OFFERGUIDE_NOTIFY", "console").lower()
        notify_channel: NotifyChannel = (
            notify_raw if notify_raw in ("console", "feishu", "telegram") else "console"  # type: ignore[assignment]
        )
        return cls(
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY") or None,
            deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            default_model=os.environ.get("OFFERGUIDE_DEFAULT_MODEL", "deepseek-v4-flash"),
            feishu_webhook_url=os.environ.get("FEISHU_WEBHOOK_URL") or None,
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None,
            notify_channel=notify_channel,
            db_path=Path(os.environ.get("OFFERGUIDE_DB", ".offerguide/store.db")),
            resume_pdf=(
                Path(os.environ["OFFERGUIDE_RESUME_PDF"])
                if "OFFERGUIDE_RESUME_PDF" in os.environ
                else None
            ),
            web_host=os.environ.get("OFFERGUIDE_HOST", "127.0.0.1"),
            web_port=int(os.environ.get("OFFERGUIDE_PORT", "8000")),
        )

    def notify_ready(self, channel: NotifyChannel | None = None) -> bool:
        """Whether the picked channel actually has the credentials it needs."""
        ch = channel or self.notify_channel
        if ch == "console":
            return True
        if ch == "feishu":
            return bool(self.feishu_webhook_url)
        if ch == "telegram":
            return bool(self.telegram_bot_token and self.telegram_chat_id)
        return False
