"""Idempotent schema initialization for the new interview research domain."""

from __future__ import annotations

import sqlite3

from ..memory import Store

_SOURCE_LINK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interview_research_source_links (
    workspace_id       INTEGER NOT NULL
                           REFERENCES interview_research_contexts(workspace_id)
                           ON DELETE CASCADE,
    evidence_id        TEXT NOT NULL CHECK (length(trim(evidence_id)) > 0),
    evidence_text_sha256 TEXT NOT NULL CHECK (length(evidence_text_sha256) = 64),
    subject_token      TEXT NOT NULL CHECK (length(subject_token) = 64),
    context_revision   INTEGER NOT NULL CHECK (context_revision > 0),
    assessment_json    TEXT NOT NULL CHECK (
                           json_valid(assessment_json)
                           AND json_type(assessment_json) = 'object'
                       ),
    contract_version   INTEGER NOT NULL DEFAULT 1 CHECK (contract_version > 0),
    created_at         REAL NOT NULL DEFAULT (julianday('now')),
    updated_at         REAL NOT NULL DEFAULT (julianday('now')),
    PRIMARY KEY(workspace_id, evidence_id, subject_token)
);
"""

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS interview_research_contexts (
    workspace_id             INTEGER PRIMARY KEY
                                 REFERENCES resume_workspaces(id) ON DELETE CASCADE,
    application_id           INTEGER NOT NULL UNIQUE REFERENCES applications(id),
    revision                 INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
    current_result_revision  INTEGER NOT NULL DEFAULT 0
                                 CHECK (current_result_revision >= 0),
    evidence_revision        INTEGER NOT NULL DEFAULT 0
                                 CHECK (evidence_revision >= 0),
    created_at               REAL NOT NULL DEFAULT (julianday('now')),
    updated_at               REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TRIGGER IF NOT EXISTS trg_interview_research_context_requires_submission
BEFORE INSERT ON interview_research_contexts
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1 FROM resume_workspaces rw
    WHERE rw.id = NEW.workspace_id
      AND rw.application_id = NEW.application_id
      AND rw.status = 'submitted'
)
BEGIN
    SELECT RAISE(ABORT, 'interview research requires the matching submitted workspace');
END;

CREATE TABLE IF NOT EXISTS interview_research_context_updates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id  INTEGER NOT NULL
                      REFERENCES interview_research_contexts(workspace_id) ON DELETE CASCADE,
    revision      INTEGER NOT NULL CHECK (revision > 1),
    kind          TEXT NOT NULL CHECK (length(trim(kind)) > 0),
    content_json  TEXT NOT NULL CHECK (
                      json_valid(content_json) AND json_type(content_json) = 'object'
                  ),
    created_at    REAL NOT NULL DEFAULT (julianday('now')),
    UNIQUE(workspace_id, revision)
);

{_SOURCE_LINK_TABLE_SQL}

CREATE TABLE IF NOT EXISTS interview_answer_set_revisions (
    workspace_id       INTEGER NOT NULL
                           REFERENCES interview_research_contexts(workspace_id)
                           ON DELETE CASCADE,
    result_revision    INTEGER NOT NULL CHECK (result_revision > 0),
    application_id     INTEGER NOT NULL REFERENCES applications(id),
    subject_token      TEXT NOT NULL CHECK (length(subject_token) = 64),
    context_revision   INTEGER NOT NULL CHECK (context_revision > 0),
    answer_set_json    TEXT NOT NULL CHECK (
                           json_valid(answer_set_json)
                           AND json_type(answer_set_json) = 'object'
                       ),
    evidence_ids_json  TEXT NOT NULL CHECK (
                           json_valid(evidence_ids_json)
                           AND json_type(evidence_ids_json) = 'array'
                       ),
    contract_version   INTEGER NOT NULL DEFAULT 1 CHECK (contract_version > 0),
    evidence_revision  INTEGER NOT NULL DEFAULT 0 CHECK (evidence_revision >= 0),
    created_at         REAL NOT NULL DEFAULT (julianday('now')),
    PRIMARY KEY(workspace_id, result_revision)
);

CREATE INDEX IF NOT EXISTS idx_interview_research_sources_evidence
    ON interview_research_source_links(evidence_id);
CREATE INDEX IF NOT EXISTS idx_interview_answer_set_application
    ON interview_answer_set_revisions(application_id, created_at DESC);
"""


def init_interview_research_schema(store: Store) -> None:
    """Create only the interview research tables, after the core Store schema."""
    with store.connect() as conn:
        conn.executescript(_SCHEMA)
        # executescript may commit before returning. Start an explicit transaction so
        # renaming, copying, and replacing the legacy table either all succeed or all
        # roll back together.
        conn.execute("BEGIN IMMEDIATE")
        _migrate_source_link_identity(conn)
        _migrate_answer_set_column(conn)
        _add_contract_versions(conn)
        _retire_legacy_material_table(conn)


def _add_contract_versions(conn: sqlite3.Connection) -> None:
    context_columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(interview_research_contexts)"
        ).fetchall()
    }
    if "evidence_revision" not in context_columns:
        conn.execute(
            "ALTER TABLE interview_research_contexts "
            "ADD COLUMN evidence_revision INTEGER NOT NULL DEFAULT 0 "
            "CHECK (evidence_revision >= 0)"
        )
    source_columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(interview_research_source_links)"
        ).fetchall()
    }
    if "contract_version" not in source_columns:
        conn.execute(
            "ALTER TABLE interview_research_source_links "
            "ADD COLUMN contract_version INTEGER NOT NULL DEFAULT 1 "
            "CHECK (contract_version > 0)"
        )
    material_columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(interview_answer_set_revisions)"
        ).fetchall()
    }
    if "contract_version" not in material_columns:
        conn.execute(
            "ALTER TABLE interview_answer_set_revisions "
            "ADD COLUMN contract_version INTEGER NOT NULL DEFAULT 1 "
            "CHECK (contract_version > 0)"
        )
    if "evidence_revision" not in material_columns:
        conn.execute(
            "ALTER TABLE interview_answer_set_revisions "
            "ADD COLUMN evidence_revision INTEGER NOT NULL DEFAULT 0 "
            "CHECK (evidence_revision >= 0)"
        )


def _migrate_answer_set_column(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute(
            "PRAGMA table_info(interview_answer_set_revisions)"
        ).fetchall()
    }
    if "material_json" in columns and "answer_set_json" not in columns:
        conn.execute(
            "ALTER TABLE interview_answer_set_revisions "
            "RENAME COLUMN material_json TO answer_set_json"
        )


def _migrate_source_link_identity(conn: sqlite3.Connection) -> None:
    primary_key = tuple(
        name
        for _position, name in sorted(
            (int(row[5]), str(row[1]))
            for row in conn.execute(
                "PRAGMA table_info(interview_research_source_links)"
            ).fetchall()
            if int(row[5]) > 0
        )
    )
    expected = ("workspace_id", "evidence_id", "subject_token")
    if primary_key == expected:
        return
    legacy = ("workspace_id", "evidence_id")
    if primary_key != legacy:
        raise RuntimeError(
            "unsupported interview source assessment identity: "
            f"{primary_key!r}"
        )

    conn.execute("DROP INDEX IF EXISTS idx_interview_research_sources_evidence")
    conn.execute(
        "ALTER TABLE interview_research_source_links "
        "RENAME TO interview_research_source_links_legacy"
    )
    conn.execute(_SOURCE_LINK_TABLE_SQL)
    conn.execute(
        "INSERT INTO interview_research_source_links("
        "workspace_id, evidence_id, evidence_text_sha256, subject_token, "
        "context_revision, assessment_json, created_at, updated_at"
        ") SELECT workspace_id, evidence_id, evidence_text_sha256, subject_token, "
        "context_revision, assessment_json, created_at, updated_at "
        "FROM interview_research_source_links_legacy"
    )
    conn.execute("DROP TABLE interview_research_source_links_legacy")
    conn.execute(
        "CREATE INDEX idx_interview_research_sources_evidence "
        "ON interview_research_source_links(evidence_id)"
    )


def _retire_legacy_material_table(conn: sqlite3.Connection) -> None:
    """Discard model-generated prep packs that cannot become real-source Q&A."""
    legacy_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'interview_prep_material_revisions'"
    ).fetchone()
    if legacy_exists is not None:
        conn.execute("DROP TABLE interview_prep_material_revisions")
    conn.execute(
        "UPDATE interview_research_contexts AS context "
        "SET current_result_revision = 0, updated_at = julianday('now') "
        "WHERE current_result_revision > 0 AND NOT EXISTS ("
        "SELECT 1 FROM interview_answer_set_revisions AS result "
        "WHERE result.workspace_id = context.workspace_id "
        "AND result.result_revision = context.current_result_revision"
        ")"
    )


__all__ = ["init_interview_research_schema"]
