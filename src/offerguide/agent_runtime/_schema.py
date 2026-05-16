"""Application-loop DB tables.

These tables are loop telemetry/state (not domain state like jobs/applications).
They live alongside the existing OfferGuide schema in the same SQLite
file, but the agent loop owns reading/writing them.

Why not put in `memory/db.py`? Conceptually these belong to the agent runtime
layer, not the core data model. Keeping them here lets us evolve runtime
internals without churning the central DB module.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ..memory import Store

log = logging.getLogger(__name__)


# Scheduled wakes: agent-driven self-scheduling. When the agent calls
# `schedule_next_wake(when, why)` we insert a row; the trigger system
# polls this table to know when to wake the agent next.
SCHEMA_SCHEDULED_WAKES = """
CREATE TABLE IF NOT EXISTS harness_scheduled_wakes (
    id INTEGER PRIMARY KEY,
    fire_at REAL NOT NULL,           -- julianday timestamp
    reason TEXT NOT NULL,
    requested_by_run_id INTEGER,     -- soft FK to harness_runs (no constraint)
    fired_at REAL,                   -- NULL when not yet fired
    fired_run_id INTEGER,            -- which run picked this up
    created_at REAL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS idx_harness_scheduled_wakes_pending
    ON harness_scheduled_wakes(fire_at) WHERE fired_at IS NULL;
"""

# Events log: agent-driven record of "things that happened in this user's
# job hunt". Read-only audit trail for the agent + UI; agent writes via
# `record_event(kind, job_id, note)` tool.
SCHEMA_EVENTS = """
CREATE TABLE IF NOT EXISTS harness_events (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,              -- 'applied' | 'interview_received' | 'interview_done' | 'rejected' | 'offer' | 'note' | ...
    job_id INTEGER,                  -- soft FK to jobs
    note TEXT,
    source TEXT NOT NULL DEFAULT 'agent',  -- 'agent' | 'user' | 'extension'
    created_at REAL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS idx_harness_events_job
    ON harness_events(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_harness_events_kind
    ON harness_events(kind, created_at DESC);
"""

# Per-run record: every agent runtime loop run gets a row. Used for cost
# tracking, /debug telemetry, and as the soft FK target for self-notes
# / scheduled wakes / events.
SCHEMA_RUNS = """
CREATE TABLE IF NOT EXISTS harness_runs (
    id INTEGER PRIMARY KEY,
    trigger_kind TEXT NOT NULL,      -- 'cron' | 'event' | 'user_input' | 'scheduled'
    trigger_detail TEXT,             -- JSON: source event id / user message / etc
    started_at REAL DEFAULT (julianday('now')),
    ended_at REAL,
    iterations INTEGER DEFAULT 0,
    tool_calls_json TEXT,            -- JSON array of tool names called
    final_text TEXT,
    cost_usd REAL DEFAULT 0,
    status TEXT DEFAULT 'running',   -- 'running' | 'ok' | 'error'
    error_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_harness_runs_recent
    ON harness_runs(started_at DESC);
"""


SCHEMA_WORK_ITEMS = """
CREATE TABLE IF NOT EXISTS agent_work_items (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source_kind TEXT NOT NULL,        -- user_input | event | scheduled | agent
    source_ref TEXT,                  -- event id / wake id / run id / free-form
    status TEXT NOT NULL DEFAULT 'open',
        -- open | in_progress | blocked | waiting | done | dismissed
    priority INTEGER NOT NULL DEFAULT 50,
    job_id INTEGER,
    due_at REAL,
    summary TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    next_action TEXT NOT NULL DEFAULT '',
    last_run_id INTEGER,
    created_at REAL DEFAULT (julianday('now')),
    updated_at REAL DEFAULT (julianday('now')),
    closed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_agent_work_items_status
    ON agent_work_items(status, priority DESC, due_at, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_work_items_source
    ON agent_work_items(source_kind, source_ref);
CREATE INDEX IF NOT EXISTS idx_agent_work_items_job
    ON agent_work_items(job_id, status);
"""


_ALL_SCHEMAS = [
    SCHEMA_SCHEDULED_WAKES,
    SCHEMA_EVENTS,
    SCHEMA_RUNS,
    SCHEMA_WORK_ITEMS,
]

ALL_WORK_ITEM_STATUSES = (
    "open",
    "in_progress",
    "blocked",
    "waiting",
    "done",
    "dismissed",
)
ACTIVE_WORK_ITEM_STATUSES = ("open", "in_progress", "blocked", "waiting")
CLOSED_WORK_ITEM_STATUSES = ("done", "dismissed")


@dataclass(frozen=True)
class WorkItem:
    id: int
    title: str
    source_kind: str
    source_ref: str | None
    status: str
    priority: int
    job_id: int | None
    due_at: float | None
    summary: str
    evidence: dict[str, Any]
    next_action: str
    last_run_id: int | None


def init_agent_runtime_schema(store: Store) -> None:
    """Idempotent: create runtime telemetry tables if missing.

    Call this once at agent runtime startup (before first run).
    """
    with store.connect() as conn:
        for ddl in _ALL_SCHEMAS:
            conn.executescript(ddl)
    log.debug("agent runtime schema initialized (4 tables)")


def ensure_work_item(
    store: Store,
    *,
    title: str,
    source_kind: str,
    source_ref: str | None = None,
    job_id: int | None = None,
    summary: str = "",
    evidence: dict[str, Any] | None = None,
    next_action: str = "",
    priority: int = 50,
    status: str = "open",
    due_at: float | None = None,
) -> int:
    """Create or refresh one persistent unit of agent-owned work.

    A work item is the runtime's concrete representation of an open loop.
    It is deliberately small and factual; the model still decides how to
    advance it, but it no longer has to rediscover the task from logs.
    """
    init_agent_runtime_schema(store)
    source_ref_text = str(source_ref) if source_ref is not None else None
    evidence_json = json.dumps(evidence or {}, ensure_ascii=False)
    with store.connect() as conn:
        existing = None
        if source_ref_text is not None:
            existing = conn.execute(
                "SELECT id FROM agent_work_items "
                "WHERE source_kind = ? AND source_ref = ? "
                "ORDER BY id DESC LIMIT 1",
                (source_kind, source_ref_text),
            ).fetchone()
        if existing is not None:
            item_id = int(existing[0])
            conn.execute(
                "UPDATE agent_work_items SET "
                "title = ?, job_id = COALESCE(?, job_id), "
                "summary = COALESCE(NULLIF(?, ''), summary), "
                "evidence_json = ?, next_action = ?, "
                "priority = MAX(priority, ?), status = CASE "
                "  WHEN status IN ('done', 'dismissed') THEN status ELSE ? END, "
                "due_at = COALESCE(?, due_at), updated_at = julianday('now') "
                "WHERE id = ?",
                (
                    title[:240],
                    job_id,
                    summary[:1000],
                    evidence_json,
                    next_action[:500],
                    priority,
                    status,
                    due_at,
                    item_id,
                ),
            )
            return item_id
        cur = conn.execute(
            "INSERT INTO agent_work_items("
            "title, source_kind, source_ref, status, priority, job_id, due_at, "
            "summary, evidence_json, next_action"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
            (
                title[:240],
                source_kind,
                source_ref_text,
                status,
                priority,
                job_id,
                due_at,
                summary[:1000],
                evidence_json,
                next_action[:500],
            ),
        )
        return int(cur.fetchone()[0])


def list_active_work_items(store: Store, *, limit: int = 8) -> list[WorkItem]:
    init_agent_runtime_schema(store)
    placeholders = ", ".join("?" for _ in ACTIVE_WORK_ITEM_STATUSES)
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, title, source_kind, source_ref, status, priority, job_id, "
            "due_at, summary, evidence_json, next_action, last_run_id "
            "FROM agent_work_items "
            f"WHERE status IN ({placeholders}) "
            "ORDER BY priority DESC, due_at IS NULL, due_at ASC, updated_at DESC "
            "LIMIT ?",
            (*ACTIVE_WORK_ITEM_STATUSES, limit),
        ).fetchall()
    return [_row_to_work_item(row) for row in rows]


def update_work_item(
    store: Store,
    *,
    work_item_id: int,
    status: str,
    summary: str | None = None,
    evidence: dict[str, Any] | None = None,
    next_action: str | None = None,
    last_run_id: int | None = None,
    due_at: float | None = None,
) -> WorkItem:
    """Update one durable unit of agent-owned work.

    This is the runtime counterpart to tool calls that actually advance a
    task. It deliberately updates the work item itself instead of writing a
    separate "decision" log: later wakes should resume from the owned work
    state, not from a proof that the model filled out a form.
    """
    init_agent_runtime_schema(store)
    if status not in ALL_WORK_ITEM_STATUSES:
        allowed = ", ".join(ALL_WORK_ITEM_STATUSES)
        raise ValueError(f"invalid work item status {status!r}; expected one of {allowed}")

    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, title, source_kind, source_ref, status, priority, job_id, "
            "due_at, summary, evidence_json, next_action, last_run_id "
            "FROM agent_work_items WHERE id = ?",
            (work_item_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"work item {work_item_id} not found")

        current = _row_to_work_item(row)
        new_summary = current.summary if summary is None else summary[:1000]
        new_evidence = current.evidence if evidence is None else evidence
        if not isinstance(new_evidence, dict):
            new_evidence = {"value": new_evidence}
        new_next_action = (
            current.next_action if next_action is None else next_action[:500]
        )
        new_due_at = None if status in CLOSED_WORK_ITEM_STATUSES else (
            due_at if due_at is not None else current.due_at
        )
        closed_sql = (
            "julianday('now')" if status in CLOSED_WORK_ITEM_STATUSES else "NULL"
        )

        conn.execute(
            "UPDATE agent_work_items SET status = ?, summary = ?, "
            "evidence_json = ?, next_action = ?, last_run_id = COALESCE(?, last_run_id), "
            "due_at = ?, updated_at = julianday('now'), "
            f"closed_at = {closed_sql} WHERE id = ?",
            (
                status,
                new_summary,
                json.dumps(new_evidence, ensure_ascii=False),
                new_next_action,
                last_run_id,
                new_due_at,
                work_item_id,
            ),
        )
        updated = conn.execute(
            "SELECT id, title, source_kind, source_ref, status, priority, job_id, "
            "due_at, summary, evidence_json, next_action, last_run_id "
            "FROM agent_work_items WHERE id = ?",
            (work_item_id,),
        ).fetchone()
    return _row_to_work_item(updated)


def attach_work_items_to_run(
    store: Store, *, run_id: int, work_item_ids: list[int],
) -> None:
    if not work_item_ids:
        return
    init_agent_runtime_schema(store)
    with store.connect() as conn:
        conn.executemany(
            "UPDATE agent_work_items SET last_run_id = ?, "
            "status = CASE WHEN status = 'open' THEN 'in_progress' ELSE status END, "
            "updated_at = julianday('now') WHERE id = ?",
            [(run_id, item_id) for item_id in work_item_ids],
        )


def prepare_trigger_work_items(
    store: Store, *, kind: str, detail: dict[str, Any],
) -> list[int]:
    """Materialize a trigger into durable work before the model runs."""
    init_agent_runtime_schema(store)
    if kind == "user_input":
        message = (detail.get("message") or "").strip()
        if not message:
            return []
        return [
            ensure_work_item(
                store,
                title=_title_from_user_message(message),
                source_kind="user_input",
                source_ref=_stable_ref(message),
                summary=message[:1000],
                evidence={"user_message": message[:2000]},
                next_action="理解用户目标; 能自己推进就行动, 缺关键事实才 ask_user",
                priority=90,
            )
        ]
    if kind == "event":
        event_id = detail.get("event_id")
        event_kind = str(detail.get("event") or "event")
        job_id = detail.get("job_id")
        job_id = job_id if isinstance(job_id, int) else None
        note = str(detail.get("note") or "")
        return [
            ensure_work_item(
                store,
                title=_work_item_title_for_event(event_kind, job_id=job_id),
                source_kind="event",
                source_ref=str(event_id) if event_id is not None else event_kind,
                job_id=job_id,
                summary=note[:1000],
                evidence=dict(detail),
                next_action=_next_action_for_event(event_kind),
                priority=_priority_for_event(event_kind),
            )
        ]
    if kind == "scheduled":
        wake_id = detail.get("wake_id")
        reason = str(detail.get("reason") or "scheduled wake")
        return [
            ensure_work_item(
                store,
                title=f"Scheduled follow-up: {reason[:160]}",
                source_kind="scheduled",
                source_ref=str(wake_id) if wake_id is not None else reason[:160],
                summary=reason[:1000],
                evidence=dict(detail),
                next_action="检查当初留下的开放回路; 有结果就更新, 没结果就判断是否继续等待",
                priority=80,
            )
        ]
    return []


def _row_to_work_item(row: Any) -> WorkItem:
    try:
        evidence = json.loads(row[9] or "{}")
    except json.JSONDecodeError:
        evidence = {"raw": row[9]}
    if not isinstance(evidence, dict):
        evidence = {"value": evidence}
    return WorkItem(
        id=int(row[0]),
        title=row[1],
        source_kind=row[2],
        source_ref=row[3],
        status=row[4],
        priority=int(row[5] or 0),
        job_id=row[6],
        due_at=row[7],
        summary=row[8] or "",
        evidence=evidence,
        next_action=row[10] or "",
        last_run_id=row[11],
    )


def _title_from_user_message(message: str) -> str:
    compact = " ".join(message.split())
    if len(compact) <= 80:
        return f"User request: {compact}"
    return f"User request: {compact[:77]}..."


def _stable_ref(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _work_item_title_for_event(event_kind: str, *, job_id: int | None) -> str:
    labels = {
        "user_paste_jd": "Process pasted JD",
        "user_marked_applied": "Own post-apply follow-up",
        "user_received_interview": "Prepare interview response",
        "user_finished_interview": "Capture interview outcome",
        "user_received_offer": "Track offer decision",
        "user_received_rejection": "Learn from rejection",
    }
    suffix = f" for job#{job_id}" if job_id else ""
    return f"{labels.get(event_kind, event_kind)}{suffix}"


def _next_action_for_event(event_kind: str) -> str:
    mapping = {
        "user_paste_jd": "ingest/score/tailor only if the JD and candidate evidence justify it",
        "user_marked_applied": "decide whether a post-apply prep/follow-up loop is useful",
        "user_received_interview": "prepare only the missing high-value interview material",
        "user_finished_interview": "record outcome and update lessons",
        "user_received_offer": "track decision constraints and next deadline",
        "user_received_rejection": "extract one learning without overfitting",
    }
    return mapping.get(event_kind, "inspect event and decide whether it changes any open loop")


def _priority_for_event(event_kind: str) -> int:
    if event_kind in {"user_received_interview", "user_received_offer"}:
        return 95
    if event_kind in {"user_paste_jd", "user_marked_applied"}:
        return 85
    if event_kind in {"user_finished_interview", "user_received_rejection"}:
        return 75
    return 60


# Backward-compatible alias for old callers. SQLite table names still use
# `harness_*` for migration safety, but new Python code should call
# `init_agent_runtime_schema`.
init_harness_schema = init_agent_runtime_schema
