"""Inbox — the agent's HITL queue.

Every action that touches the world (sending a notification, marking a job as
"considered", saving an analyze_gaps suggestion as an applied edit) goes
through here. The agent enqueues; the user decides via the web UI; the
decision unblocks downstream automation.

Design choice for W4: we model HITL as a plain SQLite queue rather than using
LangGraph's `interrupt()` / `Command(resume=...)` mechanism. Why:

- Agent invocations stay synchronous and side-effect-free; the inbox is the
  only mutable handoff point. Easier to test, easier to reason about.
- The UI doesn't have to keep a graph thread alive while waiting for a human.
- W7's ambient scheduler can layer LangGraph interrupt() on top later if a
  long-running agent thread genuinely needs to suspend mid-graph; W4 doesn't.

Status state machine: pending → (approved | rejected | dismissed). Once
decided, an item is immutable — re-deciding requires a new inbox item.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from .memory import Store

InboxStatus = Literal["pending", "approved", "rejected", "dismissed"]
InboxKind = Literal[
    "consider_jd",          # "I think this JD is worth applying — agent's recommendation"
    "review_suggestion",    # "Here's a tailoring patch — accept it?"
    "apply_decision",       # "Confirm: log this as applied?"
    "interview_scheduled",  # "An interview was scheduled — confirm details"
    "ambient_alert",        # W7+: ambient scheduler found something noteworthy
]


@dataclass(frozen=True)
class InboxItem:
    id: int
    kind: InboxKind
    title: str
    body: str | None
    payload: dict[str, Any]
    status: InboxStatus
    created_at: float
    decided_at: float | None
    decision_note: str | None


def enqueue(
    store: Store,
    *,
    kind: InboxKind,
    title: str,
    body: str | None = None,
    payload: dict[str, Any] | None = None,
) -> InboxItem:
    """Push a new pending item onto the inbox. Returns the persisted item."""
    payload_json = json.dumps(payload or {}, ensure_ascii=False)
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO inbox_items(kind, title, body, payload_json, status) "
            "VALUES (?,?,?,?,'pending')",
            (kind, title, body, payload_json),
        )
        new_id = int(cur.lastrowid or 0)
    fetched = get(store, new_id)
    if fetched is None:
        raise RuntimeError(f"inbox enqueue failed: id={new_id} not retrievable")
    return fetched


def list_items(
    store: Store,
    *,
    status: InboxStatus | None = "pending",
    limit: int = 100,
) -> list[InboxItem]:
    """List items, newest first. `status=None` → all statuses."""
    where = "WHERE status = ?" if status else ""
    params: tuple = (status, limit) if status else (limit,)
    with store.connect() as conn:
        rows = conn.execute(
            f"SELECT id, kind, title, body, payload_json, status, created_at, "
            f"decided_at, decision_note FROM inbox_items {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
    return [_row_to_item(r) for r in rows]


def get(store: Store, item_id: int) -> InboxItem | None:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, kind, title, body, payload_json, status, created_at, "
            "decided_at, decision_note FROM inbox_items WHERE id = ?",
            (item_id,),
        ).fetchone()
    return _row_to_item(row) if row else None


def decide(
    store: Store,
    item_id: int,
    *,
    decision: Literal["approved", "rejected", "dismissed"],
    note: str | None = None,
) -> InboxItem:
    """Mark a pending item as decided. Errors if the item isn't pending."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT status FROM inbox_items WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"inbox item {item_id} not found")
        if row[0] != "pending":
            raise ValueError(f"inbox item {item_id} already decided: {row[0]}")
        conn.execute(
            "UPDATE inbox_items SET status = ?, decided_at = julianday('now'), "
            "decision_note = ? WHERE id = ?",
            (decision, note, item_id),
        )
    fetched = get(store, item_id)
    assert fetched is not None
    return fetched


def _row_to_item(row: tuple) -> InboxItem:
    return InboxItem(
        id=row[0],
        kind=row[1],
        title=row[2],
        body=row[3],
        payload=json.loads(row[4]) if row[4] else {},
        status=row[5],
        created_at=row[6],
        decided_at=row[7],
        decision_note=row[8],
    )
