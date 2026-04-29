"""Notifier interface + level enum.

Every concrete adapter (`feishu`, `telegram`, `console`) returns the same
NotifyResult so callers can record success/failure into the inbox feedback
table and treat all channels uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

NotifyLevel = Literal["info", "warn", "high"]
"""Severity hint. Adapters can use it to format differently (color/emoji/parse mode)."""


@dataclass(frozen=True)
class NotifyResult:
    ok: bool
    channel: str
    error: str | None = None
    raw: dict | None = None


class Notifier(Protocol):
    name: str

    def notify(
        self,
        *,
        title: str,
        body: str,
        level: NotifyLevel = "info",
    ) -> NotifyResult: ...
