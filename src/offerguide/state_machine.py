"""Application status state machine.

Maps event kinds from ``application_events`` to the denormalized
``applications.status`` field.  The event log is the source of truth;
the status column is a convenience for quick filtering and display.

Call :func:`sync_status` after recording any event to keep the two in
sync.  It's safe to call on any event kind — ``silent_check`` and other
non-status-changing events return *None* and make no writes.
"""

from __future__ import annotations

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
    "终面": "final_interview",
    "HR":   "final_interview",
}

TERMINAL_STATUSES: frozenset[str] = frozenset({"rejected", "offer", "withdrawn"})


def status_for_event(
    kind: str,
    payload: dict[str, Any] | None = None,
) -> str | None:
    """Derive the new application status from an event kind + payload.

    Returns *None* for events that don't change status (``silent_check``).
    """
    if kind == "silent_check":
        return None
    if kind == "interview" and payload:
        round_hint = payload.get("round", "")
        if round_hint in INTERVIEW_ROUND_MAP:
            return INTERVIEW_ROUND_MAP[round_hint]
    return EVENT_STATUS_MAP.get(kind)


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
    new_status = status_for_event(kind, payload)
    if new_status is None:
        return None
    with store.connect() as conn:
        conn.execute(
            "UPDATE applications SET status = ?, last_status_change = julianday('now') "
            "WHERE id = ?",
            (new_status, application_id),
        )
    return new_status
