"""Application-event log — the single source of truth for application lifecycle.

Why an event log instead of a single mutable status field:

- An application's history matters as much as its current state. "投递 → HR 看了
  → 笔试 → 一面" is a *sequence* — flattening it to one status
  loses the timing information needed for any reply-rate or response-latency
  analysis.
- Time since the latest real event remains queryable without adding synthetic
  reminder events or sentinel statuses.
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
- ``inferred``  — an event inferred from external evidence rather than entered manually
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
    "interview_cancelled",
    "rejected",
    "offer",
    "withdrawn",
]
"""Allowed event kinds. Anything else raises ValueError on insert.

Adding a new kind requires updating the SQL CHECK list (none today, but the
Python validator is the gate) and any consumers that branch on kind. Keep this
list small and concrete — bespoke metadata belongs in `payload`."""

EventSource = Literal["manual", "email", "platform", "calendar", "inferred"]
CalendarAction = Literal["scheduled", "cancelled"]
CalendarRecordState = Literal["recorded", "duplicate", "stale"]

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


@dataclass(frozen=True)
class CalendarRecordOutcome:
    """Result of applying one versioned calendar message to the event log."""

    state: CalendarRecordState
    event: ApplicationEvent

    @property
    def created(self) -> bool:
        return self.state == "recorded"


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


def record_calendar_event(
    store: Store,
    *,
    application_id: int,
    uid: str,
    sequence: int,
    action: CalendarAction,
    payload: dict[str, Any],
) -> CalendarRecordOutcome:
    """Record a calendar revision exactly once and reject stale replays.

    RFC 5546 cancellation messages commonly reuse the latest REQUEST sequence,
    so identity includes the action as well as UID and SEQUENCE. A cancellation
    at the same sequence is therefore one additional real lifecycle event; an
    exact replay is idempotent. Older sequences, and a REQUEST replay after a
    cancellation at the same sequence, are stale and do not mutate state.
    """
    normalized_uid = uid.strip()
    if not normalized_uid:
        raise ValueError("calendar event UID is required")
    if sequence < 0:
        raise ValueError("calendar event SEQUENCE must be non-negative")
    if action not in ("scheduled", "cancelled"):
        raise ValueError(f"unknown calendar action: {action!r}")

    kind: EventKind = (
        "interview_cancelled" if action == "cancelled" else "interview"
    )
    normalized_payload = {
        **payload,
        "ics_uid": normalized_uid,
        "ics_sequence": sequence,
        "calendar_action": action,
    }
    payload_json = json.dumps(normalized_payload, ensure_ascii=False)

    with store.connect() as conn:
        # Serialize read-before-write so simultaneous uploads cannot append the
        # same calendar revision twice without requiring a second receipt table.
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT id, application_id, kind, occurred_at, source, payload_json "
            "FROM application_events "
            "WHERE application_id = ? AND source = 'calendar' "
            "ORDER BY id ASC",
            (application_id,),
        ).fetchall()
        matching: list[tuple[ApplicationEvent, int, str]] = []
        for row in rows:
            event = _row_to_event(row)
            event_uid = str(event.payload.get("ics_uid") or "").strip()
            if event_uid != normalized_uid:
                continue
            raw_sequence = event.payload.get("ics_sequence", 0)
            try:
                event_sequence = int(raw_sequence)
            except (TypeError, ValueError):
                continue
            event_action = str(event.payload.get("calendar_action") or "scheduled")
            matching.append((event, event_sequence, event_action))

        max_sequence = max(
            (event_sequence for _event, event_sequence, _action in matching),
            default=-1,
        )
        if sequence < max_sequence:
            latest_event = max(matching, key=lambda item: (item[1], item[0].id))[0]
            return CalendarRecordOutcome(state="stale", event=latest_event)

        cancelled_at_sequence = any(
            event_sequence == sequence and event_action == "cancelled"
            for _event, event_sequence, event_action in matching
        )
        if action == "scheduled" and cancelled_at_sequence:
            latest_event = max(matching, key=lambda item: (item[1], item[0].id))[0]
            return CalendarRecordOutcome(state="stale", event=latest_event)

        for event, event_sequence, event_action in matching:
            if event_sequence == sequence and event_action == action:
                return CalendarRecordOutcome(state="duplicate", event=event)

        row = conn.execute(
            "INSERT INTO application_events(application_id, kind, source, payload_json) "
            "VALUES (?, ?, 'calendar', ?) RETURNING id, occurred_at",
            (application_id, kind, payload_json),
        ).fetchone()
        if row is None:  # pragma: no cover - SQLite RETURNING contract
            raise RuntimeError("calendar event insert returned no row")
        event = ApplicationEvent(
            id=int(row[0]),
            application_id=application_id,
            kind=kind,
            occurred_at=float(row[1]),
            source="calendar",
            payload=normalized_payload,
        )
        from .state_machine import sync_status_in_transaction

        sync_status_in_transaction(
            conn,
            application_id,
            event.kind,
            event.payload,
        )
        return CalendarRecordOutcome(state="recorded", event=event)


def has_calendar_uid(
    store: Store,
    *,
    application_id: int,
    uid: str,
) -> bool:
    """Whether this application has already accepted a calendar UID."""
    normalized_uid = uid.strip()
    if not normalized_uid:
        return False
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT payload_json FROM application_events "
            "WHERE application_id = ? AND source = 'calendar'",
            (application_id,),
        ).fetchall()
    for row in rows:
        try:
            payload = json.loads(row[0]) if row[0] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("ics_uid") or "").strip() == normalized_uid:
            return True
    return False


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
    Callers that need elapsed time since the latest real event can use
    :func:`silence_age_days` alongside this.
    """
    last = latest(store, application_id)
    return last.kind if last else "no_events"


def silence_age_days(store: Store, application_id: int, *, now: float | None = None) -> float | None:
    """Days since the latest non-synthetic event. None if no events yet.

    Inferred events are excluded so elapsed time reflects confirmed lifecycle
    activity. Returns 0.0 if the latest non-inferred event is in the future
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
