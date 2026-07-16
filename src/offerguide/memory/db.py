"""SQLite store — local-first, single-user, schema documented inline.

Tables (additional vector tables in `vec.py`):

- `master_resume` — the current PDF evidence and user-confirmed semantic master
- `jobs` — every JD ever scouted; raw_text is LLM-facing only, structured platform
  fields live in extras_json
- `applications` — one row per JD-the-user-actually-decided-to-pursue. The status
  field is now denormalized; source of truth is the latest application_events row.
- `resume_workspaces` — the one current semantic resume and PDF per application
- `application_events` — append-only event log (submitted/viewed/replied/...)
  preserving the actual application history and timing.
- `skill_runs` — every SKILL invocation (input/output/cost). The trainset for GEPA evolution.
- `feedback` — generic signals from reality (HR replied, suggestion accepted, ...).
- `interviews` — scheduled interviews, with prep notes and reflection
- `evolution_log` — one row per GEPA evolution run
- `inbox_items` — HITL queue (W4)
- `project_records` — user's truthful project fact vault for resume/interview grounding
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

_RETIRED_TABLES = (
    "post_apply_materials",
    "interview_experiences",
    "company_briefs",
    "user_keywords",
)

_RETIRED_TRIGGERS = (
    "trg_post_apply_requires_submitted_workspace_insert",
    "trg_post_apply_requires_submitted_workspace_update",
)

_RETIRED_DERIVED_TABLES = ("daemon_runs",)

_SCHEMA = """
-- The one current master resume. The PDF text is immutable evidence; the
-- semantic document is the user-reviewable interpretation reused by resume
-- workspaces. A failed/empty extraction is rejected before it reaches here.
CREATE TABLE IF NOT EXISTS master_resume (
    id                       INTEGER PRIMARY KEY CHECK (id = 1),
    source_path              TEXT NOT NULL,
    source_sha256            TEXT NOT NULL CHECK (length(source_sha256) = 64),
    extracted_text           TEXT NOT NULL CHECK (length(trim(extracted_text)) > 0),
    semantic_document_json   TEXT NOT NULL DEFAULT '{}'
                                 CHECK (json_valid(semantic_document_json)
                                    AND json_type(semantic_document_json) = 'object'),
    semantic_status          TEXT NOT NULL DEFAULT 'draft'
                                 CHECK (semantic_status IN ('draft', 'confirmed')),
    created_at               REAL NOT NULL DEFAULT (julianday('now')),
    updated_at               REAL NOT NULL DEFAULT (julianday('now')),
    confirmed_at             REAL,
    CHECK (
        (semantic_status = 'draft' AND confirmed_at IS NULL)
        OR (semantic_status = 'confirmed' AND confirmed_at IS NOT NULL)
    )
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
-- It preserves actual lifecycle timing without synthetic reminder events.
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
    occurred_at     REAL NOT NULL DEFAULT (julianday('now')),
    source          TEXT NOT NULL,    -- 'manual'|'email'|'platform'|'calendar'|'inferred'
    payload_json    TEXT NOT NULL DEFAULT '{}'
);

-- One mutable resume workspace per application. It contains the exact context
-- used for editing, one semantic ResumeDocument, and its matching PDF. Drafts
-- are updated in place. The explicit submit transaction changes status once;
-- SQLite then prevents any further mutation or deletion of employer-received
-- material. Post-application research is owned by the research-agent schema.
CREATE TABLE IF NOT EXISTS resume_workspaces (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id           INTEGER NOT NULL UNIQUE REFERENCES applications(id),
    status                   TEXT NOT NULL DEFAULT 'draft'
                                 CHECK (status IN ('draft', 'submitted')),
    job_snapshot_json        TEXT NOT NULL
                                 CHECK (json_valid(job_snapshot_json)
                                    AND json_type(job_snapshot_json) = 'object'),
    master_source_sha256     TEXT NOT NULL CHECK (length(master_source_sha256) = 64),
    context_json             TEXT NOT NULL DEFAULT '{}'
                                 CHECK (json_valid(context_json)
                                    AND json_type(context_json) = 'object'),
    resume_document_json     TEXT NOT NULL DEFAULT '{}'
                                 CHECK (json_valid(resume_document_json)
                                    AND json_type(resume_document_json) = 'object'),
    pdf_path                 TEXT,
    pdf_sha256               TEXT CHECK (pdf_sha256 IS NULL OR length(pdf_sha256) = 64),
    apply_pack_json          TEXT NOT NULL DEFAULT '{}'
                                 CHECK (json_valid(apply_pack_json)
                                    AND json_type(apply_pack_json) = 'object'),
    created_at               REAL NOT NULL DEFAULT (julianday('now')),
    updated_at               REAL NOT NULL DEFAULT (julianday('now')),
    submitted_at             REAL,
    CHECK ((pdf_path IS NULL) = (pdf_sha256 IS NULL)),
    CHECK (
        (status = 'draft' AND submitted_at IS NULL)
        OR (status = 'submitted' AND submitted_at IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS idx_resume_workspaces_status
    ON resume_workspaces(status, updated_at DESC);

CREATE TRIGGER IF NOT EXISTS trg_resume_workspaces_one_per_job
BEFORE INSERT ON resume_workspaces
FOR EACH ROW
WHEN EXISTS (
    SELECT 1
    FROM resume_workspaces existing
    JOIN applications existing_app ON existing_app.id = existing.application_id
    JOIN applications new_app ON new_app.id = NEW.application_id
    WHERE existing_app.job_id = new_app.job_id
)
BEGIN
    SELECT RAISE(ABORT, 'a job can have only one resume workspace');
END;

CREATE TRIGGER IF NOT EXISTS trg_resume_workspaces_submitted_immutable
BEFORE UPDATE ON resume_workspaces
FOR EACH ROW
WHEN OLD.status = 'submitted'
BEGIN
    SELECT RAISE(ABORT, 'submitted resume workspace is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_resume_workspaces_submitted_no_delete
BEFORE DELETE ON resume_workspaces
FOR EACH ROW
WHEN OLD.status = 'submitted'
BEGIN
    SELECT RAISE(ABORT, 'submitted resume workspace is immutable');
END;

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
    proposed_action_json   TEXT,      -- optional agent action payload
    -- W14.20 — for kind='question' items, the multi-choice options the user
    -- picks from. JSON list of {id, label}. NULL for non-question kinds.
    question_options_json  TEXT
);

-- STAR + Reflection story bank — behavioral interview answers the user
-- has rehearsed. Borrowed pattern from Career-Ops (MIT, santifer):
-- accumulate 5-10 master narratives across evaluations rather than
-- regenerating every time. The user can review these stories or explicitly
-- add them to a submitted application's interview context. Theme tags:
-- 'collaboration' | 'conflict' | 'failure' |
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

-- Project fact vault — private grounding material for resume tailoring and
-- project deep-dive prep. This is deliberately NOT a "metrics brag" table:
-- the user records mainstream direction, real work, contribution boundary,
-- artifacts, and "do not claim" guardrails. Downstream SKILLs may use it
-- as evidence, but must not invent percent lifts / rankings / scale numbers
-- unless the project_outputs or evidence fields explicitly support them.
-- `market_context` stores how similar projects/products are commonly
-- explained in public launch posts, docs, papers, or open-source READMEs.
-- It is expression reference only, never evidence that the user achieved
-- the same metrics, scale, or novelty.
CREATE TABLE IF NOT EXISTS project_records (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    title                 TEXT NOT NULL,
    mainstream_direction  TEXT NOT NULL,
        -- e.g. LLM application / recommender systems / backend system / CV / data analysis
    typical_problem       TEXT,
        -- What this direction usually solves, and which sub-problem this project touched.
    project_task          TEXT NOT NULL,
        -- What the project actually set out to complete, without over-claiming.
    my_work               TEXT NOT NULL,
        -- What the user personally did; should distinguish team work vs own work.
    method_route          TEXT,
        -- Mainstream method/framework/model/engineering route and why it was chosen.
    market_context        TEXT,
        -- Public-context summary: how comparable projects describe the problem,
        -- value, audience, and system shape. Not a claim about this project.
    reference_sources     TEXT,
        -- Source titles/URLs used for market_context.
    contribution_type     TEXT NOT NULL DEFAULT 'main_contribution',
        -- method_innovation | engineering_improvement | application_transfer |
        -- process_improvement | integration | reproduction | main_contribution
    contribution_detail   TEXT,
        -- A conservative description of innovation/improvement/contribution.
    key_difficulties      TEXT,
        -- Real blockers: data quality, model instability, integration, deployment, etc.
    resolution_process    TEXT,
        -- How the user diagnosed/tried/settled. Not a polished fake solution.
    project_outputs       TEXT,
        -- Artifacts: code, report, demo, model, comparison table, deployment, etc.
    evidence              TEXT,
        -- Links/paths/screenshots/reports/logs that back claims.
    askable_points        TEXT,
        -- Interview/professor follow-up points the user can actually answer.
    expression_boundary   TEXT,
        -- Can say / be careful saying / do not say.
    do_not_claim          TEXT,
        -- Explicit claims that should never be generated into a resume/interview answer.
    tags_json             TEXT NOT NULL DEFAULT '[]',
    confidence            REAL NOT NULL DEFAULT 0.5,
        -- User-rated defensibility/readiness, 0..1.
    created_at            REAL DEFAULT (julianday('now')),
    updated_at            REAL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS idx_project_records_direction
    ON project_records(mainstream_direction);
CREATE INDEX IF NOT EXISTS idx_project_records_confidence
    ON project_records(confidence, updated_at);

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
-- refactor retired AgentLoop in favor of the agent runtime (see ``harness_runs``
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
    _migrate_known_job_urls(conn)

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

    project_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(project_records)").fetchall()
    }
    if project_cols:  # only if the table exists
        if "market_context" not in project_cols:
            conn.execute("ALTER TABLE project_records ADD COLUMN market_context TEXT")
        if "reference_sources" not in project_cols:
            conn.execute("ALTER TABLE project_records ADD COLUMN reference_sources TEXT")

    _migrate_draft_resume_workspaces(conn)


    # A retired discovery experiment duplicated the current domain Agent state.
    # These tables only contain derived control-plane data;
    # jobs, applications, skill runs, and submitted artifacts remain intact.
    for table in (
        "agent_self_notes",
        "discovery_attempts",
        "discovery_outcomes",
        "discovery_misses",
        "coverage_snapshots",
        "company_watchlist",
        "source_health",
        "source_probes",
        "discovery_queries",
        "discovery_runs",
        "job_intelligence",
    ):
        conn.execute(f"DROP TABLE IF EXISTS {table}")


def _migrate_known_job_urls(conn: sqlite3.Connection) -> None:
    """Repair known stale URLs in existing local stores."""
    rows = conn.execute(
        "SELECT id, source, source_id, url, extras_json FROM jobs "
        "WHERE source = 'tencent_campus' AND url LIKE 'https://join.qq.com/jobdesc.html?postId=%'"
    ).fetchall()
    for jid, source, source_id, url, extras_json in rows:
        parsed = urlparse(str(url or ""))
        values = parse_qs(parsed.query)
        post_id = next(
            (
                items[0].strip()
                for key in ("postid", "postId")
                if (items := values.get(key)) and items[0].strip()
            ),
            str(source_id or "").strip(),
        )
        if source != "tencent_campus" or not post_id:
            continue
        new_url = "https://join.qq.com/post_detail.html?" + urlencode(
            {"postid": post_id}
        )
        extras = _loads_obj(extras_json)
        migrations = extras.setdefault("migrations", [])
        if isinstance(migrations, list):
            migrations.append({
                "kind": "repair_tencent_campus_url",
                "from": url,
                "to": new_url,
            })
        conn.execute(
            "UPDATE jobs SET url = ?, extras_json = ? WHERE id = ?",
            (new_url, json.dumps(extras, ensure_ascii=False), jid),
        )


def _migrate_draft_resume_workspaces(conn: sqlite3.Connection) -> None:
    """Remove deleted context/package concepts from mutable drafts."""
    rows = conn.execute(
        "SELECT id, context_json, apply_pack_json FROM resume_workspaces WHERE status = 'draft'"
    ).fetchall()
    for workspace_id, raw_context, raw_pack in rows:
        context = _loads_obj(raw_context)
        context_changed = False
        for field in ("user_supplements", "references", "omitted_materials"):
            if field in context:
                context.pop(field)
                context_changed = True

        saved = _loads_obj(raw_pack)
        assistant = saved.get("assistant")
        pack_changed = False
        if isinstance(assistant, dict) and "message" not in assistant:
            intro = assistant.get("self_intro_snippet")
            message = intro.get("text") if isinstance(intro, dict) else None
            old_answers = assistant.get("qa_templates")
            form_answers: list[dict[str, str]] = []
            if isinstance(old_answers, list):
                for item in old_answers:
                    if not isinstance(item, dict):
                        continue
                    question = str(item.get("question") or "").strip()
                    answer = str(item.get("answer") or "").strip()
                    if question and answer:
                        form_answers.append({"question": question, "answer": answer})
            checks = assistant.get("pre_submit_checklist")
            pre_submit_checks = (
                [str(item).strip() for item in checks if str(item).strip()]
                if isinstance(checks, list)
                else []
            )
            if (isinstance(message, str) and message.strip()) or form_answers:
                saved["assistant"] = {
                    "message": (
                        message.strip() if isinstance(message, str) and message.strip() else None
                    ),
                    "form_answers": form_answers,
                    "pre_submit_checks": pre_submit_checks,
                }
                pack_changed = True

        current_assistant = saved.get("assistant")
        if isinstance(current_assistant, dict):
            checks = current_assistant.get("pre_submit_checks")
            if isinstance(checks, list):
                cleaned_checks = [
                    item
                    for item in checks
                    if not any(
                        deleted_field in str(item)
                        for deleted_field in (
                            "self_intro_snippet",
                            "qa_templates",
                            "pre_submit_checklist",
                        )
                    )
                ]
                if cleaned_checks != checks:
                    saved["assistant"] = {
                        **current_assistant,
                        "pre_submit_checks": cleaned_checks,
                    }
                    pack_changed = True
        if not context_changed and not pack_changed:
            continue
        conn.execute(
            "UPDATE resume_workspaces SET context_json = ?, apply_pack_json = ?, "
            "updated_at = julianday('now') "
            "WHERE id = ? AND status = 'draft'",
            (
                json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                json.dumps(saved, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                workspace_id,
            ),
        )


def _loads_obj(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _drop_retired_tables(conn: sqlite3.Connection) -> None:
    """Remove obsolete empty tables without silently discarding user data."""
    existing = {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        if str(row[0]) in _RETIRED_TABLES
    }
    populated = {
        table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in _RETIRED_TABLES
        if table in existing
    }
    populated = {table: count for table, count in populated.items() if count > 0}
    if populated:
        details = ", ".join(f"{table}={count}" for table, count in populated.items())
        raise RuntimeError(
            "refusing to drop retired tables with saved rows; export or migrate "
            f"their data before retrying: {details}"
        )

    for trigger in _RETIRED_TRIGGERS:
        conn.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
    for table in _RETIRED_TABLES:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    for table in _RETIRED_DERIVED_TABLES:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')


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
            _drop_retired_tables(conn)
            conn.executescript(_SCHEMA)
            _migrate(conn)

    def health_check(self) -> dict[str, int]:
        """Return row counts per table — useful for the example script."""
        with self.connect() as conn:
            tables = [
                "master_resume",
                "jobs",
                "applications",
                "application_events",
                "resume_workspaces",
                "skill_runs",
                "feedback",
                "interviews",
                "evolution_log",
                "inbox_items",
                "behavioral_stories",
                "project_records",
            ]
            return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
