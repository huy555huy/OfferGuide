"""Console notifier — stdout, used as the offline / no-config default."""

from __future__ import annotations

import sys
from typing import ClassVar

from ._base import NotifyLevel, NotifyResult


class ConsoleNotifier:
    name = "console"

    _PREFIX: ClassVar[dict[str, str]] = {
        "info": "[INFO]",
        "warn": "[WARN]",
        "high": "[HIGH]",
    }

    def __init__(self, *, stream=None) -> None:
        self._stream = stream or sys.stdout

    def notify(
        self,
        *,
        title: str,
        body: str,
        level: NotifyLevel = "info",
    ) -> NotifyResult:
        prefix = self._PREFIX.get(level, "[INFO]")
        print(f"{prefix} {title}", file=self._stream)
        for line in body.splitlines():
            print(f"      {line}", file=self._stream)
        return NotifyResult(ok=True, channel=self.name)
