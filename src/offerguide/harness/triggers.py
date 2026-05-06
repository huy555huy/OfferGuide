"""Trigger system — event-driven primary path + cron heartbeat fallback.

Per W15 §11.5 E: cron heartbeat retreats from "main path" (W14.18) to
"fallback safety net". The **main** way the agent wakes is:

1. **Job-hunt lifecycle events** — user pasted JD / marked applied /
   received interview / etc. These are fire-and-forget: trigger.fire()
   inserts a queued event, then the next harness loop picks it up.

2. **Agent-scheduled wakes** — when the agent calls
   `schedule_next_wake(7d, 'check X')`, a row goes into
   harness_scheduled_wakes. The scheduler polls and wakes when due.

3. **Cron heartbeat** — every 6 hours, wake the agent unconditionally.
   Catches any state drift the agent missed; agent looks around, no-ops
   if nothing to do (cheap).

All three triggers funnel into the same `loop.run(trigger=..., deps=...)`
call. The trigger object's `kind` + `detail` field framing tells the agent
what woke it.

Q3 (W15.13 review answer) — STATUS:
- ``schedule_next_wake`` path is **fully wired**: agent calls tool →
  row in harness_scheduled_wakes → scheduler.poll_pending picks up →
  cron tick fires it. End-to-end works.
- ``fire_event`` + ``_poll_unprocessed_events`` are **scaffolding for
  future UI lifecycle buttons** (e.g. user clicks "I applied" on a job
  card → fires user_marked_applied event → next cron wakes agent).
  Not yet wired in UI; planned for W15.14+. The path is tested
  (test_w15_harness.TestTriggers) but unused in production. Kept
  because cost-of-keeping is low and removing means re-deriving later.
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..memory import Store
from . import _schema
from .loop import TriggerEvent

log = logging.getLogger(__name__)


# ── Lifecycle event kinds (canonical names) ───────────────────────────


# Events the agent should wake on immediately:
USER_LIFECYCLE_EVENTS: tuple[str, ...] = (
    "user_paste_jd",
    "user_marked_applied",
    "user_received_interview",
    "user_finished_interview",
    "user_received_offer",
    "user_received_rejection",
    "user_message",  # free-form chat
)


# ── Public API ────────────────────────────────────────────────────────


@dataclass
class PendingTrigger:
    """One trigger waiting to be processed by the loop."""

    source: str
    """'event' | 'scheduled' | 'cron'"""

    trigger_event: TriggerEvent
    """Pass directly to loop.run(trigger=...)."""

    cleanup: Any = None
    """Optional callback the loop runs after success (e.g. mark
    scheduled_wake as fired). Signature: cleanup() -> None."""


def fire_event(
    store: Store, *, event_kind: str, detail: dict[str, Any] | None = None,
) -> None:
    """Record a job-hunt lifecycle event. Triggers next-poll wake.

    Idempotent on reasonable retries — the agent will see the event in
    its harness_events table and decide what to do.
    """
    _schema.init_harness_schema(store)
    detail = detail or {}
    job_id = detail.get("job_id") if isinstance(detail.get("job_id"), int) else None
    note_text = detail.get("note", "") or ""
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO harness_events(kind, job_id, note, source) "
            "VALUES (?, ?, ?, ?)",
            (event_kind, job_id, note_text[:1000], "user"),
        )
    log.info("fired event: %s (job_id=%s)", event_kind, job_id)


def poll_pending(store: Store) -> list[PendingTrigger]:
    """Find all triggers ready to fire right now.

    Order: scheduled wakes first (most specific), then unprocessed
    events (in arrival order). Cron heartbeats are not surfaced here —
    they're driven by APScheduler externally.

    The caller is expected to iterate and call ``loop.run(trigger=t.trigger_event)``
    on each, then call ``t.cleanup()`` after success.
    """
    _schema.init_harness_schema(store)
    out: list[PendingTrigger] = []
    out.extend(_poll_scheduled_wakes(store))
    out.extend(_poll_unprocessed_events(store))
    return out


def make_cron_heartbeat() -> TriggerEvent:
    """Cron-driven wake. Used by the scheduler's heartbeat job."""
    return TriggerEvent(kind="cron", detail={"timestamp": _now_iso()})


def make_user_input_trigger(message: str) -> TriggerEvent:
    """User typed something in the chat input — wake the agent now."""
    return TriggerEvent(kind="user_input", detail={"message": message[:2000]})


# ── Internals ─────────────────────────────────────────────────────────


def _poll_scheduled_wakes(store: Store) -> Iterable[PendingTrigger]:
    """Find all scheduled wakes whose fire_at <= now and not yet fired."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, reason, fire_at FROM harness_scheduled_wakes "
            "WHERE fired_at IS NULL AND fire_at <= julianday('now') "
            "ORDER BY fire_at ASC LIMIT 20",
        ).fetchall()

    out = []
    for row in rows:
        wake_id, reason, fire_at = int(row[0]), row[1], row[2]
        out.append(PendingTrigger(
            source="scheduled",
            trigger_event=TriggerEvent(
                kind="scheduled",
                detail={"wake_id": wake_id, "reason": reason, "fire_at": fire_at},
            ),
            cleanup=_make_wake_cleanup(store, wake_id),
        ))
    return out


def _poll_unprocessed_events(store: Store) -> Iterable[PendingTrigger]:
    """Find lifecycle events recorded since the last harness run.

    We use a sentinel: events with id > the last `harness_runs.trigger_detail`
    that processed an event-kind. This is approximate — at-least-once
    semantics — but the agent is expected to be idempotent (worldview
    is the source of truth, events are nudges).
    """
    with store.connect() as conn:
        last_processed = conn.execute(
            "SELECT MAX(CAST(json_extract(trigger_detail, '$.event_id') AS INTEGER)) "
            "FROM harness_runs WHERE trigger_kind = 'event'",
        ).fetchone()
        cursor = (last_processed[0] if last_processed and last_processed[0] else 0) or 0

        rows = conn.execute(
            "SELECT id, kind, job_id, note FROM harness_events "
            "WHERE id > ? AND source = 'user' "
            "ORDER BY id ASC LIMIT 10",
            (cursor,),
        ).fetchall()

    out = []
    for row in rows:
        event_id, kind, job_id, note = int(row[0]), row[1], row[2], row[3]
        out.append(PendingTrigger(
            source="event",
            trigger_event=TriggerEvent(
                kind="event",
                detail={
                    "event_id": event_id,
                    "event": kind,
                    "job_id": job_id,
                    "note": note,
                },
            ),
            # No cleanup — the harness_runs row IS the cursor advance
        ))
    return out


def _make_wake_cleanup(store: Store, wake_id: int):
    def _cleanup() -> None:
        with store.connect() as conn:
            conn.execute(
                "UPDATE harness_scheduled_wakes "
                "SET fired_at = julianday('now') WHERE id = ?",
                (wake_id,),
            )
    return _cleanup


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()
