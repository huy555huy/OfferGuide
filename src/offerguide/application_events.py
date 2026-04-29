"""Application-event log — the single source of truth for application lifecycle.

Why an event log instead of a single mutable status field:

- An application's history matters as much as its current state. "投递 → HR 看了
  → 沉默 7 天 → 笔试 → 一面挂了" is a *sequence* — flattening it to one status
  loses the timing information needed for any reply-rate or response-latency
  analysis.
- Silence is queryable: "applications whose latest event is older than 14 days
  with no `replied|assessment|interview` event" — a single status field can't
  express this without sentinel values that pollute the schema.
- It's append-only, so the log can be replayed to derive any view we want
  later (status snapshots, survival curves, conversion funnels).

The `applications.status` column still exists as a denormalized convenience
field, but the source of truth is the latest event. Use ``derive_status()``
when you need the current state from the event log.

Sources of events (W5' surface; richer integrations come later):

- ``manual``    — user logs an event from the inbox UI (W6+ tracking dashboard)
- ``email``     — parsed from a future email integration
- ``platform``  — pulled from a platform's API / page parse
- ``calendar``  — derived from interview invites
- ``inferred``  — synthetic: e.g. silent_check events written by the cron job
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .memory import Store

EventKind = Literal[
    "submitted",
    "viewed",
    "replied",
    "assessment",
    "interview",
    "rejected",
    "offer",
    "withdrawn",
    "silent_check",
]
"""Allowed event kinds. Anything else raises ValueError on insert.

Adding a new kind requires updating the SQL CHECK list (none today, but the
Python validator is the gate) and any consumers that branch on kind. Keep this
list small and concrete — bespoke metadata belongs in `payload`."""

EventSource = Literal["manual", "email", "platform", "calendar", "inferred"]

_VALID_KINDS: frozenset[str] = frozenset(EventKind.__args__)
_VALID_SOURCES: frozenset[str] = frozenset(EventSource.__args__)


@dataclass(frozen=True)
class ApplicationEvent:
    id: int
    application_id: int
    kind: EventKind
    occurred_at: float
    source: EventSource
    payload: dict[str, Any]


def record(
    store: Store,
    *,
    application_id: int,
    kind: EventKind,
    source: EventSource = "manual",
    occurred_at: float | None = None,
    payload: dict[str, Any] | None = None,
) -> ApplicationEvent:
    """Append one event to the log. ``occurred_at=None`` defaults to now (julianday).

    Validates ``kind``/``source`` against the Literal lists. Foreign-key
    enforcement on ``application_id`` is handled by SQLite when PRAGMA
    foreign_keys is on (Store enables it on connect).
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"unknown event kind {kind!r}; must be one of {sorted(_VALID_KINDS)}"
        )
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"unknown event source {source!r}; must be one of {sorted(_VALID_SOURCES)}"
        )
    payload_json = json.dumps(payload or {}, ensure_ascii=False)

    with store.connect() as conn:
        if occurred_at is None:
            cur = conn.execute(
                "INSERT INTO application_events(application_id, kind, source, payload_json) "
                "VALUES (?,?,?,?) RETURNING id, occurred_at",
                (application_id, kind, source, payload_json),
            )
        else:
            cur = conn.execute(
                "INSERT INTO application_events(application_id, kind, occurred_at, source, payload_json) "
                "VALUES (?,?,?,?,?) RETURNING id, occurred_at",
                (application_id, kind, occurred_at, source, payload_json),
            )
        row = cur.fetchone()

    return ApplicationEvent(
        id=int(row[0]),
        application_id=application_id,
        kind=kind,
        occurred_at=float(row[1]),
        source=source,
        payload=json.loads(payload_json),
    )


def list_events(
    store: Store, application_id: int, *, limit: int = 200
) -> list[ApplicationEvent]:
    """Return the application's events oldest-first (i.e. lifecycle order)."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, application_id, kind, occurred_at, source, payload_json "
            "FROM application_events WHERE application_id = ? "
            "ORDER BY occurred_at ASC, id ASC LIMIT ?",
            (application_id, limit),
        ).fetchall()
    return [_row_to_event(r) for r in rows]


def latest(store: Store, application_id: int) -> ApplicationEvent | None:
    """Most-recent event for the application, or None if there are no events yet."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, application_id, kind, occurred_at, source, payload_json "
            "FROM application_events WHERE application_id = ? "
            "ORDER BY occurred_at DESC, id DESC LIMIT 1",
            (application_id,),
        ).fetchone()
    return _row_to_event(row) if row else None


def derive_status(store: Store, application_id: int) -> str:
    """Current status derived from the event log.

    Returns the latest event's ``kind``, or ``'no_events'`` if no events exist.
    Callers that need richer derivation (e.g. "submitted but silent for 14d")
    should use :func:`silence_age_days` alongside this.
    """
    last = latest(store, application_id)
    return last.kind if last else "no_events"


def silence_age_days(store: Store, application_id: int, *, now: float | None = None) -> float | None:
    """Days since the latest non-synthetic event. None if no events yet.

    "Synthetic" means events with source ``'inferred'`` — those are created by
    the silence cron itself, so counting them as "activity" would mask real
    silence. Returns 0.0 if the latest non-synthetic event is in the future
    (clock skew safety).
    """
    with store.connect() as conn:
        row = conn.execute(
            "SELECT occurred_at, COALESCE(?, julianday('now')) "
            "FROM application_events WHERE application_id = ? AND source != 'inferred' "
            "ORDER BY occurred_at DESC, id DESC LIMIT 1",
            (now, application_id),
        ).fetchone()
    if row is None:
        return None
    return max(0.0, float(row[1]) - float(row[0]))


def _row_to_event(row: tuple) -> ApplicationEvent:
    return ApplicationEvent(
        id=int(row[0]),
        application_id=int(row[1]),
        kind=row[2],
        occurred_at=float(row[3]),
        source=row[4],
        payload=json.loads(row[5]) if row[5] else {},
    )
