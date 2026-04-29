"""Notification adapters — feishu / telegram / console, runtime-selectable.

Use `make_notifier(settings)` to get the channel the user configured. For
test/CI scenarios with no creds, this falls back to ConsoleNotifier so the
agent never crashes mid-flow because of unconfigured notifications.
"""

from __future__ import annotations

from ...config import Settings
from ._base import Notifier, NotifyLevel, NotifyResult
from .console import ConsoleNotifier
from .feishu import FeishuNotifier
from .telegram import TelegramNotifier


def make_notifier(settings: Settings) -> Notifier:
    """Build the configured channel; fall through to console if creds are missing."""
    ch = settings.notify_channel
    if ch == "feishu" and settings.feishu_webhook_url:
        return FeishuNotifier(settings.feishu_webhook_url)
    if ch == "telegram" and settings.telegram_bot_token and settings.telegram_chat_id:
        return TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
    return ConsoleNotifier()


__all__ = [
    "ConsoleNotifier",
    "FeishuNotifier",
    "Notifier",
    "NotifyLevel",
    "NotifyResult",
    "TelegramNotifier",
    "make_notifier",
]
