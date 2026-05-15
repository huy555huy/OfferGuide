"""Application-loop DB tables.

These tables are loop telemetry/state (not domain state like jobs/applications).
They live alongside the existing OfferGuide schema in the same SQLite
file, but the agent loop owns reading/writing them.

Why not put in `memory/db.py`? Conceptually these belong to the agent runtime
layer, not the core data model. Keeping them here lets us evolve runtime
internals without churning the central DB module.
"""

from __future__ import annotations

import logging

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


_ALL_SCHEMAS = [SCHEMA_SCHEDULED_WAKES, SCHEMA_EVENTS, SCHEMA_RUNS]


def init_agent_runtime_schema(store: Store) -> None:
    """Idempotent: create runtime telemetry tables if missing.

    Call this once at agent runtime startup (before first run).
    """
    with store.connect() as conn:
        for ddl in _ALL_SCHEMAS:
            conn.executescript(ddl)
    log.debug("agent runtime schema initialized (3 tables)")


# Backward-compatible alias for old callers. SQLite table names still use
# `harness_*` for migration safety, but new Python code should call
# `init_agent_runtime_schema`.
init_harness_schema = init_agent_runtime_schema
