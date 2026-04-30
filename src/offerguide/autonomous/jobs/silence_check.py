"""Daily silence check — wraps ``workers.tracker.tracker_run``.

The tracker scans non-terminal applications and writes synthetic
``silent_check`` events when an application has been silent for ≥ 7 /
14 / 30 days, plus an inbox notification + push notification.

Idempotent: re-running on the same day is safe (the
``_max_alerted_threshold`` check in tracker prevents duplicate alerts).
"""

from __future__ import annotations

from typing import Any

from ...ui.notify import ConsoleNotifier
from ..scheduler import JobContext, JobSpec


def run(ctx: JobContext) -> dict[str, Any]:
    """One sweep + notify pass. Returns counters for logging."""
    from ...workers import tracker

    notifier = ctx.notifier or ConsoleNotifier()
    counters = tracker.tracker_run(ctx.store, notifier=notifier)
    return counters


# Daily 09:00 Asia/Shanghai
SILENCE_CHECK_JOB = JobSpec(
    name="silence_check",
    func=run,
    trigger="cron",
    trigger_kwargs={"hour": 9, "minute": 0},
)
