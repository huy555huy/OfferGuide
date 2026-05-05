"""Inbox — the agent's HITL queue.

Every action that touches the world (sending a notification, marking a job as
"considered", saving an analyze_gaps suggestion as an applied edit) goes
through here. The agent enqueues; the user decides via the web UI; the
decision unblocks downstream automation.

Status state machine: pending → (approved | rejected | dismissed). Once
decided, an item is immutable — re-deciding requires a new inbox item.

W13.3 agent-suggestion loop:
- Agent (during a trajectory) calls ``write_suggestion`` action tool to enqueue
  an ``agent_suggestion`` item attributing the suggestion to a specific
  (skill_run_id, skill_name, skill_version) tuple.
- User opens /inbox, decides approve / reject / dismiss.
- ``decide()`` writes a ``user_thumbs`` signal to ``evolution_signals``
  attributed to the originating skill — this is the **highest-weight signal**
  in the evolution fitness calculation (weight=2.0 vs critic=1.0).

This closes the loop: real human preference flows back into prompt evolution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from .memory import Store

log = logging.getLogger(__name__)

InboxStatus = Literal["pending", "approved", "rejected", "dismissed"]
InboxKind = Literal[
    "agent_suggestion",     # W13.3 — anything the central agent loop proposes
    "consider_jd",          # legacy — pre-W13 hardcoded "consider this JD" enqueue
    "review_suggestion",    # legacy
    "apply_decision",       # legacy
    "interview_scheduled",  # legacy
    "ambient_alert",        # legacy
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
    # W13.3 — agent-suggestion attribution (None for legacy items)
    source_agent_run_id: int | None = None
    source_skill_name: str | None = None
    source_skill_version: str | None = None
    proposed_action: dict[str, Any] | None = None


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


def enqueue_agent_suggestion(
    store: Store,
    *,
    title: str,
    body: str,
    source_agent_run_id: int | None = None,
    source_skill_name: str | None = None,
    source_skill_version: str | None = None,
    source_skill_run_id: int | None = None,
    proposed_action: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> InboxItem:
    """W13.3 — preferred path for agent-generated suggestions.

    Records the agent's reasoning + (optionally) what tool it'd execute if
    user approves. The attribution fields make the user's later thumbs flow
    back into evolution_signals correctly.

    ``proposed_action`` shape:
        {"tool": "tailor_resume", "args": {"job_id": 42}}

    The route handler that processes user-approved suggestions can read this
    and either show a "do it now" button or auto-execute on approve.
    """
    p = dict(payload or {})
    if source_skill_run_id is not None:
        p["source_skill_run_id"] = source_skill_run_id

    payload_json = json.dumps(p, ensure_ascii=False)
    proposed_json = (
        json.dumps(proposed_action, ensure_ascii=False)
        if proposed_action else None
    )

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO inbox_items("
            "  kind, title, body, payload_json, status, "
            "  source_agent_run_id, source_skill_name, source_skill_version, "
            "  proposed_action_json"
            ") VALUES ('agent_suggestion', ?, ?, ?, 'pending', ?, ?, ?, ?)",
            (title, body, payload_json,
             source_agent_run_id, source_skill_name, source_skill_version,
             proposed_json),
        )
        new_id = int(cur.lastrowid or 0)
    fetched = get(store, new_id)
    assert fetched is not None
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
            f"decided_at, decision_note, source_agent_run_id, source_skill_name, "
            f"source_skill_version, proposed_action_json "
            f"FROM inbox_items {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
    return [_row_to_item(r) for r in rows]


def get(store: Store, item_id: int) -> InboxItem | None:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, kind, title, body, payload_json, status, created_at, "
            "decided_at, decision_note, source_agent_run_id, source_skill_name, "
            "source_skill_version, proposed_action_json "
            "FROM inbox_items WHERE id = ?",
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
    """Mark a pending item as decided. Errors if the item isn't pending.

    W13.3: when an ``agent_suggestion`` item is approved or rejected, this
    function ALSO writes a ``user_thumbs`` signal to ``evolution_signals``
    attributed to the originating SKILL. Dismissed items don't generate a
    signal (user said "neither yes nor no").
    """
    with store.connect() as conn:
        row = conn.execute(
            "SELECT status, kind, source_skill_name, source_skill_version, "
            "       source_agent_run_id, payload_json "
            "FROM inbox_items WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"inbox item {item_id} not found")
        cur_status, kind, sk_name, sk_ver, agent_run_id, payload_json = row
        if cur_status != "pending":
            raise ValueError(
                f"inbox item {item_id} already decided: {cur_status}"
            )
        conn.execute(
            "UPDATE inbox_items SET status = ?, decided_at = julianday('now'), "
            "decision_note = ? WHERE id = ?",
            (decision, note, item_id),
        )

    # W13.3 — fan out user's thumbs to evolution_signals
    if kind == "agent_suggestion" and decision in ("approved", "rejected") and sk_name:
        try:
            from .evolution.signals import record_user_thumbs
            # Try to find a real skill_run_id for attribution; fall back to None
            sr_id: int | None = None
            try:
                payload = json.loads(payload_json or "{}")
                sr_id = payload.get("source_skill_run_id")
                if sr_id is not None:
                    sr_id = int(sr_id)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            record_user_thumbs(
                store,
                skill_name=sk_name,
                skill_version=sk_ver or "?",
                skill_run_id=sr_id,
                thumbs=1 if decision == "approved" else -1,
                notes=(
                    f"inbox#{item_id} {decision}"
                    + (f" (agent_run#{agent_run_id})" if agent_run_id else "")
                    + (f": {note}" if note else "")
                )[:300],
            )
        except Exception as e:
            log.debug("failed to record_user_thumbs for inbox#%d: %s", item_id, e)

    fetched = get(store, item_id)
    assert fetched is not None
    return fetched


def _row_to_item(row: Any) -> InboxItem:
    # W14.8: typed as Any (not tuple[Any, ...]) because pyright narrows the
    # variadic tuple to a chain of fixed-length tuples after `len(row) > 12`,
    # then complains about every row[N]. sqlite3 row arity is dynamic anyway
    # — schema migrations may add trailing columns and we handle that with
    # the explicit len() guards.
    proposed_action = None
    if len(row) > 12 and row[12]:
        try:
            proposed_action = json.loads(row[12])
        except json.JSONDecodeError:
            proposed_action = None
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
        source_agent_run_id=row[9] if len(row) > 9 else None,
        source_skill_name=row[10] if len(row) > 10 else None,
        source_skill_version=row[11] if len(row) > 11 else None,
        proposed_action=proposed_action,
    )
