"""SQLite store — local-first, single-user, schema documented inline.

Tables (additional vector tables in `vec.py`):

- `profile` — single-row JSON blob with the user's preferences and parsed resume
- `jobs` — every JD ever scouted; raw_text is LLM-facing only, structured platform
  fields live in extras_json
- `applications` — one row per JD-the-user-actually-decided-to-pursue. The status
  field is now denormalized; source of truth is the latest application_events row.
- `application_events` — append-only event log (submitted/viewed/replied/...). Lets
  silence (no event for N days) be queried, and gives us t0/event timing for any
  future survival analysis.
- `skill_runs` — every SKILL invocation (input/output/cost). The trainset for GEPA evolution.
- `feedback` — generic signals from reality (HR replied, suggestion accepted, ...).
- `interviews` — scheduled interviews, with prep notes and reflection
- `evolution_log` — one row per GEPA evolution run
- `inbox_items` — HITL queue (W4)
- `interview_experiences` — 面经 corpus for prepare_interview RAG
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profile (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    data_json TEXT NOT NULL,
    updated_at REAL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT NOT NULL,        -- 'nowcoder' / 'boss_extension' / 'manual' / ...
    source_id    TEXT,
    url          TEXT,
    title        TEXT,
    company      TEXT,
    location     TEXT,
    raw_text     TEXT NOT NULL,                 -- LLM-readable canonical JD text only.
    extras_json  TEXT NOT NULL DEFAULT '{}',    -- platform-native structured fields:
                                                 -- {"salaryMin":15,"avgProcessRate":38,...}.
                                                 -- W3' fix: was previously concatenated into raw_text,
                                                 -- which polluted LLM context AND made the platform's
                                                 -- structured fields (e.g. nowcoder's avgProcessRate
                                                 -- — actual platform-measured reply rate) unqueryable.
    content_hash TEXT NOT NULL,
    fetched_at   REAL DEFAULT (julianday('now')),
    UNIQUE(source, content_hash)
);

CREATE TABLE IF NOT EXISTS applications (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id             INTEGER NOT NULL REFERENCES jobs(id),
    status             TEXT NOT NULL,   -- 'considered'|'applied'|'hr_replied'|'screening'|
                                        -- 'written_test'|'1st_interview'|'2nd_interview'|
                                        -- 'final_interview'|'offer'|'rejected'|'withdrawn'
                                        -- NOTE: with the application_events table this becomes
                                        -- a denormalized convenience field; the source of truth
                                        -- is the latest application_events row.
    applied_at         REAL,
    last_status_change REAL DEFAULT (julianday('now')),
    notes              TEXT
);

-- Append-only event log for application lifecycles. Status is derived from the
-- latest event of an application, not stored as a single mutable field. This is
-- what makes silence (no event for N days) a queryable concept and gives us the
-- t0/event timing needed for any future survival analysis.
CREATE TABLE IF NOT EXISTS application_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id  INTEGER NOT NULL REFERENCES applications(id),
    kind            TEXT NOT NULL,
        -- 'submitted'   first push to the platform / company
        -- 'viewed'      HR / recruiter opened the application
        -- 'replied'     any human reply (positive or negative)
        -- 'assessment'  written test / OA / coding screen
        -- 'interview'   any interview round (subdivide via payload.round)
        -- 'rejected'    explicit rejection
        -- 'offer'       offer extended
        -- 'withdrawn'   user withdrew
        -- 'silent_check' synthetic event written when N-day silence is detected
    occurred_at     REAL NOT NULL DEFAULT (julianday('now')),
    source          TEXT NOT NULL,    -- 'manual'|'email'|'platform'|'calendar'|'inferred'
    payload_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS skill_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name    TEXT NOT NULL,
    skill_version TEXT NOT NULL,
    input_hash    TEXT NOT NULL,
    input_json    TEXT NOT NULL,
    output_json   TEXT NOT NULL,
    cost_usd      REAL,
    latency_ms    INTEGER,
    created_at    REAL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    target_kind TEXT NOT NULL,        -- 'application'|'interview'|'skill_run'|...
    target_id   INTEGER NOT NULL,
    kind        TEXT NOT NULL,        -- 'reply_received'|'interview_question_match'|...
    value_json  TEXT NOT NULL,
    created_at  REAL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS interviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id  INTEGER NOT NULL REFERENCES applications(id),
    scheduled_at    REAL,
    type            TEXT,             -- '笔试'|'一面'|'二面'|'终面'|'HR'
    prep_notes      TEXT,
    actual_questions TEXT,
    reflection      TEXT,
    created_at      REAL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS evolution_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name     TEXT NOT NULL,
    parent_version TEXT,
    new_version    TEXT NOT NULL,
    metric_name    TEXT NOT NULL,
    metric_before  REAL,
    metric_after   REAL,
    notes          TEXT,
    created_at     REAL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS inbox_items (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT NOT NULL,    -- 'consider_jd'|'apply_decision'|'review_suggestion'|...
    title          TEXT NOT NULL,
    body           TEXT,
    payload_json   TEXT NOT NULL,    -- structured refs to jobs/skill_runs/applications/...
    status         TEXT NOT NULL DEFAULT 'pending',  -- pending|approved|rejected|dismissed
    created_at     REAL DEFAULT (julianday('now')),
    decided_at     REAL,
    decision_note  TEXT
);

-- ``interview_experiences`` is the umbrella corpus table for ANY high-signal
-- evidence about a company: 面经, offer 复盘, 项目分享, 一面挂经验, etc.
-- The name is historical (W4 only stored 面经); ``content_kind`` distinguishes
-- the modern entries while old rows default to 'interview'.
--
-- Quality columns (W11+) carry the classifier verdict so successful-profile
-- synthesis can filter out 卖课 / 引流 / fake content. quality_score is
-- 0..1; >= 0.6 = trustworthy, < 0.4 = drop. quality_signals_json captures the
-- evidence (e.g. {has_specific_timeline: true, has_marketer_signals: false}).
CREATE TABLE IF NOT EXISTS interview_experiences (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    company                TEXT NOT NULL,
    role_hint              TEXT,             -- 岗位线索（"AI 算法"/"前端"/...）— 可空
    raw_text               TEXT NOT NULL,
    source                 TEXT NOT NULL,    -- 'nowcoder_discuss'|'manual_paste'|'1point3acres'|...
    source_url             TEXT,
    content_hash           TEXT NOT NULL,
    content_kind           TEXT NOT NULL DEFAULT 'interview',  -- 'interview'|'offer_post'|'reflection'|'project_share'|'other'
    quality_score          REAL NOT NULL DEFAULT 0.5,           -- 0..1, agent-classified trustworthiness
    quality_signals_json   TEXT NOT NULL DEFAULT '{}',          -- structured evidence
    quality_classified_at  REAL,                                -- NULL = not yet classified
    created_at             REAL DEFAULT (julianday('now')),
    UNIQUE(source, content_hash)
);

-- Per-company brief maintained by the autonomous agent.
-- The agent reads recent interview_experiences + application_events
-- + skill_runs and produces a compact JSON brief that overrides
-- hardcoded heuristics (COMPANY_APPLICATION_LIMITS) when newer signal
-- says the policy changed.
CREATE TABLE IF NOT EXISTS company_briefs (
    company           TEXT PRIMARY KEY,
    brief_json        TEXT NOT NULL,    -- {summary, current_app_limit, interview_style, recent_signals[], hiring_trend, confidence}
    last_updated_at   REAL DEFAULT (julianday('now')),
    update_count      INTEGER NOT NULL DEFAULT 1
);

-- STAR + Reflection story bank — behavioral interview answers the user
-- has rehearsed. Borrowed pattern from Career-Ops (MIT, santifer):
-- accumulate 5-10 master narratives across evaluations rather than
-- regenerating every time. Tagged so prepare_interview /
-- deep_project_prep can pull thematically relevant ones at retrieval
-- time. Theme tags: 'collaboration' | 'conflict' | 'failure' |
-- 'learning' | 'leadership' | 'ambiguity' | 'tradeoff' | etc.
CREATE TABLE IF NOT EXISTS behavioral_stories (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    title             TEXT NOT NULL,    -- short label ('法至产品分歧 / RemeDi 训练崩溃 / ...')
    situation         TEXT NOT NULL,    -- S in STAR
    task              TEXT NOT NULL,    -- T
    action            TEXT NOT NULL,    -- A
    result            TEXT NOT NULL,    -- R
    reflection        TEXT,             -- + reflective learning the story conveys
    tags_json         TEXT NOT NULL DEFAULT '[]',  -- list[str] — themes
    used_count        INTEGER NOT NULL DEFAULT 0,  -- bumped when a SKILL retrieves this story
    confidence        REAL NOT NULL DEFAULT 0.5,   -- user's self-rated readiness (0-1)
    created_at        REAL DEFAULT (julianday('now'))
);

-- ``user_facts`` is the long-term memory layer (W12, mem0 v3-style).
-- Single-pass ADD-only: new facts append, never UPDATE/DELETE — accumulation
-- of evidence beats clobber-update for downstream retrieval recall.
-- Each row links back to the SKILL run that produced it via ``source_run_id``,
-- so a stale fact can always be traced back to "which SKILL output, when".
CREATE TABLE IF NOT EXISTS user_facts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_text       TEXT NOT NULL,                    -- 一句话事实, e.g. "用户 RemeDi 项目 AUC 0.83"
    kind            TEXT NOT NULL,                    -- profile|preference|experience|feedback|project|company_signal
    source_skill    TEXT,                             -- which SKILL extracted it
    source_run_id   INTEGER,                          -- skill_runs.id (FK soft-link)
    confidence      REAL NOT NULL DEFAULT 0.5,        -- 0..1
    entities_json   TEXT NOT NULL DEFAULT '[]',       -- list[str] — companies / projects / skills
    used_count      INTEGER NOT NULL DEFAULT 0,       -- bumped each retrieve call
    created_at      REAL DEFAULT (julianday('now')),
    last_used_at    REAL,
    UNIQUE(fact_text)
);

CREATE INDEX IF NOT EXISTS idx_user_facts_kind  ON user_facts(kind);
CREATE INDEX IF NOT EXISTS idx_user_facts_used  ON user_facts(used_count, last_used_at);

CREATE INDEX IF NOT EXISTS idx_jobs_source         ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_apps_job            ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_apps_status         ON applications(status);
CREATE INDEX IF NOT EXISTS idx_runs_skill          ON skill_runs(skill_name, created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_target     ON feedback(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_evolution_skill     ON evolution_log(skill_name, created_at);
CREATE INDEX IF NOT EXISTS idx_inbox_status        ON inbox_items(status, created_at);
CREATE INDEX IF NOT EXISTS idx_interview_company   ON interview_experiences(company, created_at);
CREATE INDEX IF NOT EXISTS idx_app_events_app      ON application_events(application_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_app_events_kind     ON application_events(kind, occurred_at);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Forward-only ALTER for column additions on databases that pre-date them.

    SQLite has no ``IF NOT EXISTS`` for ADD COLUMN, so we check ``pragma_table_info``
    first. New tables are handled by the idempotent CREATE TABLE IF NOT EXISTS in
    `_SCHEMA`. This function only deals with adding columns to existing tables.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "extras_json" not in cols:
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN extras_json TEXT NOT NULL DEFAULT '{}'"
        )

    # interview_experiences quality + content_kind columns (W11)
    ie_cols = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(interview_experiences)"
        ).fetchall()
    }
    if ie_cols:  # only if the table exists
        if "content_kind" not in ie_cols:
            conn.execute(
                "ALTER TABLE interview_experiences "
                "ADD COLUMN content_kind TEXT NOT NULL DEFAULT 'interview'"
            )
        if "quality_score" not in ie_cols:
            conn.execute(
                "ALTER TABLE interview_experiences "
                "ADD COLUMN quality_score REAL NOT NULL DEFAULT 0.5"
            )
        if "quality_signals_json" not in ie_cols:
            conn.execute(
                "ALTER TABLE interview_experiences "
                "ADD COLUMN quality_signals_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "quality_classified_at" not in ie_cols:
            conn.execute(
                "ALTER TABLE interview_experiences "
                "ADD COLUMN quality_classified_at REAL"
            )


class Store:
    """Thin connection-per-call wrapper. Keep operations short — SQLite is fine
    for single-user local-first; no connection pooling needed at our scale."""

    def __init__(self, db_path: str | Path = ".offerguide/store.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self, *, with_vec: bool = False) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # safe + fast for single-user
        if with_vec:
            from .vec import attach_vec

            attach_vec(conn)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create the relational schema if missing, plus run any forward migrations.

        Idempotent on both fresh and pre-existing databases. CREATE TABLE handles
        new tables; `_migrate()` handles ADD COLUMN cases SQLite can't express
        idempotently.
        """
        with self.connect() as conn:
            conn.executescript(_SCHEMA)
            _migrate(conn)

    def health_check(self) -> dict[str, int]:
        """Return row counts per table — useful for the example script."""
        with self.connect() as conn:
            tables = [
                "profile",
                "jobs",
                "applications",
                "application_events",
                "skill_runs",
                "feedback",
                "interviews",
                "evolution_log",
                "inbox_items",
                "interview_experiences",
                "company_briefs",
                "behavioral_stories",
            ]
            return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
