"""Application status state machine.

Maps event kinds from ``application_events`` to the denormalized
``applications.status`` field.  The event log is the source of truth;
the status column is a convenience for quick filtering and display.

Call :func:`sync_status` after recording any event to keep the two in
sync. Events without a status mapping return *None* and make no writes.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .memory import Store

# ── event kind → application status ─────────────────────────────────
# Events not listed here don't change the status.

EVENT_STATUS_MAP: dict[str, str] = {
    "submitted":  "applied",
    "viewed":     "viewed",
    "replied":    "hr_replied",
    "assessment": "written_test",
    "interview":  "1st_interview",   # default; payload.round overrides
    "rejected":   "rejected",
    "offer":      "offer",
    "withdrawn":  "withdrawn",
}

# For ``interview`` events, ``payload["round"]`` can narrow the status.
INTERVIEW_ROUND_MAP: dict[str, str] = {
    "笔试": "written_test",
    "一面": "1st_interview",
    "二面": "2nd_interview",
    "三面": "final_interview",
    "终面": "final_interview",
    "HR": "final_interview",
    "HR面": "final_interview",
}

TERMINAL_STATUSES: frozenset[str] = frozenset({"rejected", "offer", "withdrawn"})

_STATUS_PROGRESS_RANK: dict[str, int] = {
    "considered": 0,
    "applied": 1,
    "viewed": 2,
    "hr_replied": 3,
    "screening": 3,
    "written_test": 4,
    "1st_interview": 5,
    "2nd_interview": 6,
    "final_interview": 7,
}


def status_for_event(
    kind: str,
    payload: dict[str, Any] | None = None,
) -> str | None:
    """Derive the new application status from an event kind + payload.

    Returns *None* for events that don't change status.
    """
    if kind == "interview" and payload:
        round_hint = _normalize_round(payload.get("round"))
        if round_hint in INTERVIEW_ROUND_MAP:
            return INTERVIEW_ROUND_MAP[round_hint]
    return EVENT_STATUS_MAP.get(kind)


def next_status(
    current_status: str,
    kind: str,
    payload: dict[str, Any] | None = None,
) -> str | None:
    """Return a lifecycle transition without regressing confirmed progress."""
    candidate = status_for_event(kind, payload)
    if candidate is None:
        return None
    if current_status in TERMINAL_STATUSES and candidate not in TERMINAL_STATUSES:
        return current_status
    current_rank = _STATUS_PROGRESS_RANK.get(current_status)
    candidate_rank = _STATUS_PROGRESS_RANK.get(candidate)
    if (
        current_rank is not None
        and candidate_rank is not None
        and candidate_rank < current_rank
    ):
        return current_status
    return candidate


def sync_status_in_transaction(
    conn: sqlite3.Connection,
    application_id: int,
    kind: str,
    payload: dict[str, Any] | None = None,
) -> str | None:
    """Apply one monotonic transition using the caller's open transaction."""
    row = conn.execute(
        "SELECT status FROM applications WHERE id = ?",
        (application_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"application#{application_id} does not exist")
    current_status = str(row[0] or "considered")
    updated_status = next_status(current_status, kind, payload)
    if updated_status is None:
        return None
    if updated_status != current_status:
        conn.execute(
            "UPDATE applications SET status = ?, "
            "last_status_change = julianday('now') WHERE id = ?",
            (updated_status, application_id),
        )
    return updated_status


def sync_status(
    store: Store,
    application_id: int,
    kind: str,
    payload: dict[str, Any] | None = None,
) -> str | None:
    """Update ``applications.status`` based on an event kind.

    Returns the new status string, or *None* if the event doesn't change
    status.  Safe to call for any event kind.
    """
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        return sync_status_in_transaction(conn, application_id, kind, payload)


def _normalize_round(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    compact = "".join(text.split())
    if compact.lower() in {"hr", "hr面", "hrinterview", "hrround"}:
        return "HR面" if compact.lower() != "hr" else "HR"
    return compact
