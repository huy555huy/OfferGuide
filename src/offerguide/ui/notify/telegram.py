"""Telegram bot notifier — POSTs to api.telegram.org/bot<TOKEN>/sendMessage.

Two pieces of config: a bot token (from @BotFather) and a chat_id (the user's
own private chat with the bot, or a group id). To get the chat_id, message the
bot once and check `getUpdates`, or use @userinfobot.

We use HTML parse_mode for richer formatting; both `<b>` and a generous subset
of HTML are supported. Telegram caps messages at 4096 chars — long bodies are
truncated with a clear marker rather than failing.
"""

from __future__ import annotations

import httpx

from ._base import NotifyLevel, NotifyResult

_LEVEL_PREFIX = {"info": "📨", "warn": "⚠️", "high": "🔥"}
_MAX_LEN = 4000  # leave headroom under 4096 for the title prefix


class TelegramNotifier:
    name = "telegram"

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        timeout_s: float = 10.0,
    ) -> None:
        if not bot_token or not chat_id:
            raise ValueError("TelegramNotifier needs bot_token and chat_id")
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self._chat_id = chat_id
        self._http = httpx.Client(timeout=timeout_s)

    def notify(
        self,
        *,
        title: str,
        body: str,
        level: NotifyLevel = "info",
    ) -> NotifyResult:
        prefix = _LEVEL_PREFIX.get(level, "📨")
        text = f"{prefix} <b>{_html_escape(title)}</b>\n\n{_html_escape(body)}"
        if len(text) > _MAX_LEN:
            text = text[:_MAX_LEN] + "\n…(truncated)"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            resp = self._http.post(self._url, json=payload)
        except httpx.HTTPError as e:
            return NotifyResult(ok=False, channel=self.name, error=str(e))
        if resp.status_code != 200:
            return NotifyResult(
                ok=False,
                channel=self.name,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
        try:
            data = resp.json()
        except Exception as e:  # pragma: no cover
            return NotifyResult(ok=False, channel=self.name, error=f"non-JSON: {e}")
        if not data.get("ok"):
            return NotifyResult(
                ok=False,
                channel=self.name,
                error=data.get("description") or str(data),
                raw=data,
            )
        return NotifyResult(ok=True, channel=self.name, raw=data)

    def close(self) -> None:
        self._http.close()


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
