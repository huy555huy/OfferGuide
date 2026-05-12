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
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    kind                   TEXT NOT NULL,
        -- 'agent_suggestion' (W13.3, the new canonical kind for things agent proposes)
        -- Legacy: 'consider_jd' | 'apply_decision' | 'review_suggestion' | 'interview_scheduled' | 'ambient_alert'
    title                  TEXT NOT NULL,
    body                   TEXT,
    payload_json           TEXT NOT NULL,
    status                 TEXT NOT NULL DEFAULT 'pending',  -- pending | approved | rejected | dismissed
    created_at             REAL DEFAULT (julianday('now')),
    decided_at             REAL,
    decision_note          TEXT,
    -- W13.3 agent-suggestion attribution: when an agent_suggestion is approved
    -- or rejected, the user's thumbs becomes a user_thumbs signal in
    -- evolution_signals, attributed to (skill_name, skill_version, run_id)
    -- so GEPA can use it as the highest-weight feedback in its fitness calc.
    source_agent_run_id    INTEGER,                          -- FK soft-link to agent_runs.id
    source_skill_name      TEXT,
    source_skill_version   TEXT,
    proposed_action_json   TEXT,      -- {"tool": "tailor_resume", "args": {...}} (optional)
    -- W14.20 — for kind='question' items, the multi-choice options the user
    -- picks from. JSON list of {id, label}. NULL for non-question kinds.
    question_options_json  TEXT
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

-- ``daemon_runs`` records each scheduled-job execution so the UI can show
-- "is the autonomous daemon actually running, and what did each job do
-- last night". Without this, users have no signal between "daemon
-- crashed silently" and "daemon is humming".
CREATE TABLE IF NOT EXISTS daemon_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name     TEXT NOT NULL,
    started_at   REAL NOT NULL DEFAULT (julianday('now')),
    ended_at     REAL,
    status       TEXT NOT NULL,        -- 'running' | 'ok' | 'error'
    summary_json TEXT NOT NULL DEFAULT '{}',
    error_text   TEXT
);
CREATE INDEX IF NOT EXISTS idx_daemon_runs_job ON daemon_runs(job_name, started_at);

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

-- ─────────────────────── W13.1 Evolution closed loop ──────────────────────
-- The W6 GEPA framework was theoretically sound but never had a real signal
-- pipeline (the trainset was hand-stitched from skill_runs without a quality
-- label). The W13.1 redesign:
--
--   skill_run → critic_score (W13)        ┐
--   user clicks 👍/👎 in /agent inbox     ├─→ evolution_signals
--   application status reaches offer/rej  ┘     ↓
--                                            fitness
--                                              ↓
--                              meta_evolve_skill (a SKILL the agent calls)
--                                              ↓
--                            generates N variants → tests on real recent inputs
--                                              ↓
--                                   skill_variants (status=shadow)
--                                              ↓
--                                       gray-release as canary
--                                              ↓
--                            agent observes; promote winner / rollback loser
--
-- This table replaces "trust the metric you wrote in evolution/metrics.py"
-- with "trust the multi-source feedback signal stream."

CREATE TABLE IF NOT EXISTS evolution_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name      TEXT NOT NULL,
    skill_version   TEXT NOT NULL,                     -- which version of the SKILL produced this signal
    skill_run_id    INTEGER,                           -- soft FK to skill_runs.id; null when signal is post-hoc
    signal_kind     TEXT NOT NULL,
        -- 'critic'        — W13 self-critique LLM judge (0..1)
        -- 'user_thumbs'   — user 👍/👎 in /agent suggestions inbox (-1 / +1)
        -- 'app_outcome'   — application this run influenced reached an outcome (1=offer/interview, 0=rejected/silent)
        -- 'follow_through' — user actually executed the suggestion (1) vs ignored (0)
        -- 'eval_synthetic' — meta_evolve_skill's synthetic eval on a variant (0..1)
    signal_value    REAL NOT NULL,                     -- normalized 0..1 (or -1..1 for thumbs)
    signal_weight   REAL NOT NULL DEFAULT 1.0,         -- importance weight in fitness aggregation
    notes           TEXT,
    created_at      REAL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS idx_evo_signals_skill   ON evolution_signals(skill_name, skill_version, created_at);
CREATE INDEX IF NOT EXISTS idx_evo_signals_kind    ON evolution_signals(signal_kind, created_at);
CREATE INDEX IF NOT EXISTS idx_evo_signals_run     ON evolution_signals(skill_run_id);

-- ─────────────────────── W13.6 long-horizon goals ────────────────────
-- A real agent has a north star, not just per-tick reactive tasks. This
-- table holds the user's actual job-search goals (target offer date,
-- target companies, minimum offer count). The agent reads them on every
-- wake and reasons EXPLICITLY against them: "with X days left + Y apps
-- in flight + Z fitness on the offer-rate funnel, am I on track?"
--
-- Status: 'active' (currently pursuing) | 'paused' (user stopped) |
--         'achieved' (hit it) | 'abandoned' (gave up)
CREATE TABLE IF NOT EXISTS user_goals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
        -- e.g. "拿到 1 个 AI Agent 暑期实习 offer"
    description     TEXT,
        -- Free-form context the agent reads (target companies, deal-breakers,
        -- nice-to-haves, what success looks like to user)
    target_date     TEXT,
        -- ISO 8601 date string (e.g. "2026-07-15"); null for open-ended goals
    target_metric   TEXT,
        -- Free-form: "1 offer" / "5 interviews" / "10 quality applications"
    status          TEXT NOT NULL DEFAULT 'active',
        -- 'active' | 'paused' | 'achieved' | 'abandoned'
    created_at      REAL DEFAULT (julianday('now')),
    updated_at      REAL DEFAULT (julianday('now')),
    achieved_at     REAL,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_user_goals_status ON user_goals(status);

-- ──────────── W13.6 agent self-observations ──────────────────────────
-- meta_reflect tool writes here when the agent looks back at its own
-- run history + user thumbs and notices a pattern about its own behavior
-- ("I keep suggesting follow_up but user dismisses 80% of them"). Future
-- agent runs read these as part of the snapshot — gentle pressure to
-- avoid repeating mistakes.
--
-- This is meta-cognition: the agent learning ABOUT itself, not just
-- about the user. user_facts is "what I know about the user";
-- agent_self_observations is "what I've noticed about my own pattern".
CREATE TABLE IF NOT EXISTS agent_self_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    observation     TEXT NOT NULL,
        -- One sentence the agent wrote about itself
    pattern_kind    TEXT NOT NULL,
        -- 'overreach' | 'underreach' | 'tone' | 'wrong_priority' | 'repeated_mistake' | 'success_pattern'
    evidence_json   TEXT NOT NULL DEFAULT '{}',
        -- {sample_runs: [N, M, ...], thumbs_count: ..., notes: ...}
    valid_until     REAL,
        -- Some observations expire (e.g. "user is busy this week"); after
        -- this julianday, exclude from snapshot. Null = always valid.
    created_at      REAL DEFAULT (julianday('now')),
    superseded_by   INTEGER REFERENCES agent_self_observations(id)
        -- When the agent later notices a contradicting pattern, it links
        -- the new observation here so we can see the chain of self-correction
);
CREATE INDEX IF NOT EXISTS idx_agent_self_obs_validity
    ON agent_self_observations(valid_until, created_at);

-- ──────────── W14.20 agent self-notes (working memory) ────────────────
-- The cron heartbeat wakes the agent every hour but the agent has no
-- memory of "what I was about to do last wake but didn't get to". Without
-- this, the agent re-derives priorities from snapshot every wake — wastes
-- LLM cost + may forget multi-step plans (e.g. "kicked off discover, will
-- score next time once new JDs land").
--
-- Each note is a **plain text reminder the agent wrote to its future self**.
-- Different from agent_self_observations (which is meta-cognition about the
-- agent's pattern); this is operational ("next wake: score the 4 jobs that
-- just landed").
--
-- Notes are surfaced in the snapshot at the top of each wake — the agent
-- sees them and decides whether to act on or clear them.
CREATE TABLE IF NOT EXISTS agent_self_notes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    body            TEXT NOT NULL,
        -- One paragraph the agent wrote to itself, in its own voice
    note_kind       TEXT NOT NULL DEFAULT 'todo',
        -- 'todo'          — concrete next action ("score 4 new JDs")
        -- 'observation'   — passive note ("user_facts mention 字节 application is silent")
        -- 'context'       — session continuity ("we are in mid-iteration on tailoring resume for X")
    valid_until     REAL,
        -- After this julianday, the note is auto-stale and not surfaced
    cleared_at      REAL,
        -- When the agent decided this todo is done; clears it from snapshot
    cleared_reason  TEXT,
        -- "did it" / "no longer relevant" / "user said no"
    related_run_id  INTEGER,
        -- (soft FK to agent_runs.id) The agent_run that created this note,
        -- for audit trail. Not enforced because: (a) the note may outlive
        -- the run row in long-term cleanup; (b) tests want to write notes
        -- without first creating an agent_runs row.
    created_at      REAL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS idx_self_notes_active
    ON agent_self_notes(cleared_at, valid_until, created_at DESC);

-- ``skill_variants`` is the version registry for any SKILL that's been evolved.
-- The original SKILL.md on disk is always implicitly version 0 (the seed).
-- meta_evolve_skill writes new rows here as 'shadow'; the gray-release loop
-- promotes one to 'canary' (small traffic %), then to 'live' (replaces seed),
-- or kills it as 'failed'.
--
-- SkillRuntime version routing reads this table on every invoke: rows with
-- status='live' or 'canary' override the on-disk SKILL.md's body. This keeps
-- the on-disk file as the audited seed but lets prod traffic see the evolved
-- prompt without filesystem mutation (which made the W6 design unauditable).
CREATE TABLE IF NOT EXISTS skill_variants (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name          TEXT NOT NULL,
    version             TEXT NOT NULL,
    parent_version      TEXT,                          -- evolved from this version (the seed is null)
    body_md             TEXT NOT NULL,                 -- the actual SKILL.md body (the prompt)
    spec_json           TEXT NOT NULL DEFAULT '{}',    -- serialized SkillSpec frontmatter overrides (e.g. output_schema changes)
    status              TEXT NOT NULL,
        -- 'shadow'  — generated by meta_evolve_skill, not in production
        -- 'canary'  — limited-traffic A/B vs current live
        -- 'live'    — production version (only one per skill_name at a time)
        -- 'retired' — replaced by a newer live (kept for audit + rollback)
        -- 'failed'  — canary lost the A/B and was killed
    canary_traffic_pct  REAL NOT NULL DEFAULT 0.0,     -- 0..1, what fraction of invocations route to this variant when status='canary'
    fitness_score       REAL,                          -- aggregate fitness over its evolution_signals; null if no signals yet
    notes               TEXT,
    created_at          REAL DEFAULT (julianday('now')),
    promoted_at         REAL,                          -- when status moved to 'canary' or 'live'
    UNIQUE(skill_name, version)
);
CREATE INDEX IF NOT EXISTS idx_variants_skill_status ON skill_variants(skill_name, status);
CREATE INDEX IF NOT EXISTS idx_variants_status       ON skill_variants(status);

-- ``agent_runs`` — DEPRECATED. W13 central-agent-loop run log. The W21
-- refactor retired AgentLoop in favor of the harness (see ``harness_runs``
-- in ``harness/_schema.py``). This table is kept ONLY so historical data
-- from pre-W21 installs is still readable; no code writes new rows here.
-- The UI reads exclusively from harness_runs. New deployments should treat
-- this table as inert.
CREATE TABLE IF NOT EXISTS agent_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger_kind       TEXT NOT NULL,
    goal               TEXT NOT NULL,
    started_at         REAL NOT NULL DEFAULT (julianday('now')),
    ended_at           REAL,
    status             TEXT NOT NULL DEFAULT 'running',  -- 'running' | 'ok' | 'error'
    iterations         INTEGER NOT NULL DEFAULT 0,
    final_answer       TEXT,
    trajectory_json    TEXT NOT NULL DEFAULT '[]',
    critic_score       REAL,                              -- 0..1 from self-critique pass
    critic_notes       TEXT,
    cost_usd           REAL NOT NULL DEFAULT 0.0,
    latency_ms         INTEGER,
    error_text         TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_started ON agent_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status  ON agent_runs(status, started_at);

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
    # W14.12: jobs.created_at — needed for "agent 本周自动找到 N 个 JD" home
    # weekly report. Defaults to julianday('now') so existing rows get a
    # post-migration timestamp (close enough for stats; the alternative was
    # a more expensive backfill from extras_json which not all sources fill).
    if "created_at" not in cols:
        conn.execute(
            "ALTER TABLE jobs ADD COLUMN created_at REAL NOT NULL DEFAULT (julianday('now'))"
        )

    # inbox_items agent-suggestion columns (W13.3)
    inbox_cols = {
        row[1] for row in conn.execute(
            "PRAGMA table_info(inbox_items)"
        ).fetchall()
    }
    if inbox_cols:  # only if the table exists
        if "source_agent_run_id" not in inbox_cols:
            conn.execute(
                "ALTER TABLE inbox_items ADD COLUMN source_agent_run_id INTEGER"
            )
        if "source_skill_name" not in inbox_cols:
            conn.execute(
                "ALTER TABLE inbox_items ADD COLUMN source_skill_name TEXT"
            )
        if "source_skill_version" not in inbox_cols:
            conn.execute(
                "ALTER TABLE inbox_items ADD COLUMN source_skill_version TEXT"
            )
        if "proposed_action_json" not in inbox_cols:
            conn.execute(
                "ALTER TABLE inbox_items ADD COLUMN proposed_action_json TEXT"
            )
        # W14.20 — for kind='question' inbox items, options the user can
        # pick from. JSON list of {id, label, [hint]}. NULL for non-question kinds.
        if "question_options_json" not in inbox_cols:
            conn.execute(
                "ALTER TABLE inbox_items ADD COLUMN question_options_json TEXT"
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
