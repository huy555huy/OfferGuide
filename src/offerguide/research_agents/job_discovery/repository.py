"""SQLite persistence for job-discovery domain state.

The schema is intentionally initialized from this module instead of expanding
the legacy store schema.  This makes the new owner usable before the old
discovery paths are removed and keeps its revision rules locally auditable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ...memory import Store
from ...platforms._spec import RawJob, content_hash
from ..revisions import RevisionGuard, RevisionSnapshot, StaleRevisionError
from .models import (
    CandidateEvidence,
    JobDiscoverySnapshot,
    JobEvidenceCatalogItem,
    JobEvidenceCatalogPage,
    JobEvidenceKind,
    JobPostingEvidence,
    JobSearchContext,
    JobSelectionItem,
    JobSelectionSet,
    JobSourceStatus,
)
from .quality import (
    evidence_quote_matches_source,
    require_grounded_selection_reason,
    require_substantive_job_description,
)

_INITIAL_CATALOG_PAGE_SIZE = 25

_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_search_context (
    singleton              INTEGER PRIMARY KEY CHECK (singleton = 1),
    revision               INTEGER NOT NULL CHECK (revision > 0),
    intent_text            TEXT NOT NULL CHECK (length(trim(intent_text)) > 0),
    hard_constraints_json  TEXT NOT NULL DEFAULT '[]'
                               CHECK (json_valid(hard_constraints_json)
                                  AND json_type(hard_constraints_json) = 'array'),
    feedback_json          TEXT NOT NULL DEFAULT '[]'
                               CHECK (json_valid(feedback_json)
                                  AND json_type(feedback_json) = 'array'),
    created_at             REAL NOT NULL DEFAULT (julianday('now')),
    updated_at             REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS job_posting_evidence (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id                  INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
    stable_key              TEXT NOT NULL UNIQUE,
    evidence_kind           TEXT NOT NULL DEFAULT 'web'
                                CHECK (evidence_kind IN ('web', 'platform_adapter')),
    source_evidence_id      INTEGER NOT NULL,
    source_name             TEXT NOT NULL,
    source_job_id           TEXT,
    canonical_url           TEXT NOT NULL,
    company                 TEXT NOT NULL,
    title                   TEXT NOT NULL,
    location                TEXT,
    recruitment_type        TEXT,
    page_time_json          TEXT NOT NULL DEFAULT '[]'
                                CHECK (json_valid(page_time_json)
                                   AND json_type(page_time_json) = 'array'),
    jd_text                 TEXT NOT NULL CHECK (length(trim(jd_text)) > 0),
    source_status           TEXT NOT NULL DEFAULT 'unknown'
                                CHECK (source_status IN ('open', 'closed', 'unknown')),
    content_sha256          TEXT NOT NULL CHECK (length(content_sha256) = 64),
    checked_at              REAL NOT NULL,
    last_seen_at            REAL NOT NULL,
    created_at              REAL NOT NULL DEFAULT (julianday('now')),
    updated_at              REAL NOT NULL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS idx_job_posting_source_evidence
    ON job_posting_evidence(source_evidence_id);
CREATE INDEX IF NOT EXISTS idx_job_posting_status
    ON job_posting_evidence(source_status, checked_at DESC);

CREATE TABLE IF NOT EXISTS job_discovery_current (
    singleton          INTEGER PRIMARY KEY CHECK (singleton = 1),
    context_revision   INTEGER NOT NULL CHECK (context_revision > 0),
    result_revision    INTEGER NOT NULL CHECK (result_revision > 0),
    selection_json     TEXT NOT NULL
                           CHECK (json_valid(selection_json)
                              AND json_type(selection_json) = 'object'),
    published_at       REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TABLE IF NOT EXISTS job_discovery_selection_revisions (
    result_revision    INTEGER PRIMARY KEY CHECK (result_revision > 0),
    context_revision   INTEGER NOT NULL CHECK (context_revision > 0),
    selection_json     TEXT NOT NULL
                           CHECK (json_valid(selection_json)
                              AND json_type(selection_json) = 'object'),
    published_at       REAL NOT NULL
);
"""


class JobDiscoveryError(RuntimeError):
    """Base error for invalid job-discovery state changes."""


class JobDiscoveryNotConfiguredError(JobDiscoveryError):
    pass


class JobDiscoveryRevisionConflict(JobDiscoveryError):
    """A stale run or editor attempted to overwrite newer user-visible state."""


class JobDiscoveryEvidenceError(JobDiscoveryError):
    pass


def init_job_discovery_schema(store: Store) -> None:
    """Create only the new job-discovery tables; safe to call repeatedly."""

    with store.connect() as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        columns = {
            str(row[1]) for row in conn.execute(
                "PRAGMA table_info(job_posting_evidence)"
            ).fetchall()
        }
        if "job_id" not in columns:
            # Compatibility for local databases created during the short-lived
            # pre-integration build. Production schemas are created with the
            # NOT NULL/UNIQUE definition above.
            conn.execute(
                "ALTER TABLE job_posting_evidence ADD COLUMN job_id INTEGER "
                "REFERENCES jobs(id)"
            )
        if "evidence_kind" not in columns:
            conn.execute(
                "ALTER TABLE job_posting_evidence ADD COLUMN evidence_kind TEXT "
                "NOT NULL DEFAULT 'web'"
            )
            if _table_exists(conn, "source_subject_evidence"):
                conn.execute(
                    "UPDATE job_posting_evidence SET evidence_kind = 'platform_adapter' "
                    "WHERE source_evidence_id IN ("
                    "SELECT evidence_id FROM source_subject_evidence "
                    "WHERE purpose = 'verified platform job detail'"
                    ")"
                )
        _backfill_legacy_job_ids(conn)
        _enforce_migrated_job_id_invariants(conn)
        _migrate_current_selection_snapshots(conn)
        _backfill_selection_history(conn)


class JobDiscoveryRepository:
    def __init__(self, store: Store) -> None:
        self.store = store

    def replace_search_context(
        self,
        *,
        intent: str,
        hard_constraints: list[str] | None = None,
        feedback: list[str] | None = None,
        expected_revision: int | None,
    ) -> JobSearchContext:
        """Create or CAS-update the one user-visible search context."""

        hard_constraints = hard_constraints or []
        feedback = feedback or []
        provisional_revision = 1 if expected_revision is None else expected_revision + 1
        value = JobSearchContext(
            revision=provisional_revision,
            intent=intent,
            hard_constraints=hard_constraints,
            feedback=feedback,
        )
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT revision FROM job_search_context WHERE singleton = 1"
            ).fetchone()
            if row is None:
                if expected_revision is not None:
                    raise JobDiscoveryRevisionConflict(
                        "search context does not exist at the expected revision"
                    )
                conn.execute(
                    "INSERT INTO job_search_context("
                    "singleton, revision, intent_text, hard_constraints_json, feedback_json"
                    ") VALUES (1, 1, ?, ?, ?)",
                    (
                        value.intent,
                        _json_array(value.hard_constraints),
                        _json_array(value.feedback),
                    ),
                )
            else:
                current_revision = int(row["revision"])
                if expected_revision != current_revision:
                    raise JobDiscoveryRevisionConflict(
                        f"search context is revision {current_revision}, not {expected_revision}"
                    )
                conn.execute(
                    "UPDATE job_search_context SET revision = ?, intent_text = ?, "
                    "hard_constraints_json = ?, feedback_json = ?, "
                    "updated_at = julianday('now') WHERE singleton = 1",
                    (
                        value.revision,
                        value.intent,
                        _json_array(value.hard_constraints),
                        _json_array(value.feedback),
                    ),
                )
        return value

    def get_search_context(self) -> JobSearchContext | None:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT revision, intent_text, hard_constraints_json, feedback_json "
                "FROM job_search_context WHERE singleton = 1"
            ).fetchone()
        return _search_context_from_row(row) if row is not None else None

    def save_job_evidence(self, evidence: JobPostingEvidence) -> JobPostingEvidence:
        """Upsert a verified job page by source identity, preserving its row id."""

        require_substantive_job_description(
            evidence.jd_text,
            identity_values=(
                evidence.company,
                evidence.title,
                evidence.location,
                evidence.recruitment_type,
                *evidence.page_time_information,
            ),
        )
        canonical_url = canonicalize_job_url(evidence.canonical_url)
        identity_updates: dict[str, Any] = {}
        if evidence.evidence_kind == "web":
            hostname = urlsplit(canonical_url).hostname
            if not hostname:
                raise ValueError("web job URL must include a hostname")
            identity_updates = {
                "source_name": hostname.lower(),
                "source_job_id": None,
            }
        normalized = evidence.model_copy(
            update={
                **identity_updates,
                "canonical_url": canonical_url,
                "content_sha256": hashlib.sha256(
                    evidence.jd_text.encode("utf-8")
                ).hexdigest(),
            }
        )
        provisional_stable_key = job_stable_key(
            evidence_kind=normalized.evidence_kind,
            source_name=normalized.source_name,
            source_job_id=normalized.source_job_id,
            canonical_url=canonical_url,
        )
        page_time_json = _json_array(normalized.page_time_information)
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            if normalized.evidence_kind == "web":
                platform_rows = conn.execute(
                    "SELECT * FROM job_posting_evidence "
                    "WHERE evidence_kind = 'platform_adapter' ORDER BY id ASC"
                ).fetchall()
                existing_platform = next(
                    (
                        row
                        for row in platform_rows
                        if canonicalize_job_url(str(row["canonical_url"]))
                        == normalized.canonical_url
                    ),
                    None,
                )
                if existing_platform is not None:
                    # The generic page is still retained in SourceEvidenceStore,
                    # but it cannot relabel or rewrite higher-fidelity adapter
                    # evidence merely because both resolve to the same URL.
                    conn.execute(
                        "UPDATE job_posting_evidence SET checked_at = julianday('now'), "
                        "last_seen_at = julianday('now'), updated_at = julianday('now') "
                        "WHERE id = ?",
                        (int(existing_platform["id"]),),
                    )
                    saved = conn.execute(
                        "SELECT * FROM job_posting_evidence WHERE id = ?",
                        (int(existing_platform["id"]),),
                    ).fetchone()
                    return _job_evidence_from_row(saved)
            application_job_id, effective_source_job_id = self._materialize_application_job(
                conn, normalized, stable_key=provisional_stable_key
            )
            normalized = normalized.model_copy(
                update={"source_job_id": effective_source_job_id}
            )
            stable_key = job_stable_key(
                evidence_kind=normalized.evidence_kind,
                source_name=normalized.source_name,
                source_job_id=normalized.source_job_id,
                canonical_url=normalized.canonical_url,
            )
            row = conn.execute(
                "SELECT id, source_job_id FROM job_posting_evidence WHERE stable_key = ?",
                (stable_key,),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT id, source_job_id FROM job_posting_evidence WHERE job_id = ?",
                    (application_job_id,),
                ).fetchone()
            if (
                row is not None
                and row["source_job_id"]
                and normalized.source_job_id
                and str(row["source_job_id"]) != normalized.source_job_id
            ):
                raise JobDiscoveryEvidenceError(
                    "the same application job has conflicting stable source job ids"
                )
            if row is None:
                evidence_id = int(
                    conn.execute(
                        "INSERT INTO job_posting_evidence("
                        "job_id, stable_key, evidence_kind, source_evidence_id, "
                        "source_name, source_job_id, "
                        "canonical_url, company, title, location, recruitment_type, "
                        "page_time_json, jd_text, source_status, content_sha256, "
                        "checked_at, last_seen_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                        "julianday('now'), julianday('now')) RETURNING id",
                        (
                            application_job_id,
                            stable_key,
                            normalized.evidence_kind,
                            normalized.source_evidence_id,
                            normalized.source_name,
                            normalized.source_job_id,
                            normalized.canonical_url,
                            normalized.company,
                            normalized.title,
                            normalized.location,
                            normalized.recruitment_type,
                            page_time_json,
                            normalized.jd_text,
                            normalized.source_status,
                            normalized.content_sha256,
                        ),
                    ).fetchone()[0]
                )
            else:
                evidence_id = int(row["id"])
                conn.execute(
                    "UPDATE job_posting_evidence SET job_id = ?, stable_key = ?, "
                    "evidence_kind = ?, source_evidence_id = ?, "
                    "source_name = ?, source_job_id = ?, canonical_url = ?, "
                    "company = ?, title = ?, location = ?, recruitment_type = ?, "
                    "page_time_json = ?, jd_text = ?, source_status = ?, "
                    "content_sha256 = ?, checked_at = julianday('now'), "
                    "last_seen_at = julianday('now'), updated_at = julianday('now') "
                    "WHERE id = ?",
                    (
                        application_job_id,
                        stable_key,
                        normalized.evidence_kind,
                        normalized.source_evidence_id,
                        normalized.source_name,
                        normalized.source_job_id,
                        normalized.canonical_url,
                        normalized.company,
                        normalized.title,
                        normalized.location,
                        normalized.recruitment_type,
                        page_time_json,
                        normalized.jd_text,
                        normalized.source_status,
                        normalized.content_sha256,
                        evidence_id,
                    ),
                )
            saved = conn.execute(
                "SELECT * FROM job_posting_evidence WHERE id = ?", (evidence_id,)
            ).fetchone()
        return _job_evidence_from_row(saved)

    def record_source_check(
        self,
        *,
        canonical_url: str,
        source_status: JobSourceStatus,
    ) -> JobPostingEvidence | None:
        """Overlay a deterministic fetch outcome without rewriting the saved JD."""
        url = canonicalize_job_url(canonical_url)
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM job_posting_evidence ORDER BY id ASC"
            ).fetchall()
            row = next(
                (
                    item
                    for item in rows
                    if canonicalize_job_url(str(item["canonical_url"])) == url
                ),
                None,
            )
            if row is None:
                return None
            conn.execute(
                "UPDATE job_posting_evidence SET source_status = ?, "
                "checked_at = julianday('now'), updated_at = julianday('now') "
                "WHERE id = ?",
                (source_status, int(row["id"])),
            )
            updated = conn.execute(
                "SELECT * FROM job_posting_evidence WHERE id = ?",
                (int(row["id"]),),
            ).fetchone()
        return _job_evidence_from_row(updated)

    def record_user_closed_report(
        self, job_evidence_id: int
    ) -> JobPostingEvidence | None:
        """Mark the exact evidence row closed from an explicit user report."""

        if job_evidence_id < 1:
            raise ValueError("job_evidence_id must be positive")
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "UPDATE job_posting_evidence SET source_status = 'closed', "
                "checked_at = julianday('now'), updated_at = julianday('now') "
                "WHERE id = ? RETURNING *",
                (job_evidence_id,),
            ).fetchone()
        return _job_evidence_from_row(row) if row is not None else None

    @staticmethod
    def _materialize_application_job(
        conn: sqlite3.Connection,
        evidence: JobPostingEvidence,
        *,
        stable_key: str,
    ) -> tuple[int, str | None]:
        """Upsert the application target that apply-pack already consumes.

        The new domain uses source identity, not the legacy content hash, to
        keep one ``jobs.id`` stable while the source page changes.
        """

        row: sqlite3.Row | None = None
        if evidence.evidence_kind == "platform_adapter" and evidence.source_job_id:
            row = conn.execute(
                "SELECT * FROM jobs WHERE source = ? AND source_id = ? "
                "ORDER BY id ASC LIMIT 1",
                (evidence.source_name, evidence.source_job_id),
            ).fetchone()
        if row is None:
            if evidence.evidence_kind == "web":
                candidates = conn.execute("SELECT * FROM jobs ORDER BY id ASC").fetchall()
            else:
                candidates = conn.execute(
                    "SELECT * FROM jobs WHERE source = ? ORDER BY id ASC",
                    (evidence.source_name,),
                ).fetchall()
            for candidate in candidates:
                raw_url = candidate["url"]
                if not raw_url:
                    continue
                try:
                    candidate_url = canonicalize_job_url(str(raw_url))
                except ValueError:
                    continue
                if candidate_url == evidence.canonical_url:
                    row = candidate
                    break

        existing_source_job_id = (
            str(row["source_id"]).strip() if row is not None and row["source_id"] else None
        )
        if (
            evidence.evidence_kind == "platform_adapter"
            and
            evidence.source_job_id
            and existing_source_job_id
            and evidence.source_job_id != existing_source_job_id
        ):
            raise JobDiscoveryEvidenceError(
                "the same source URL is already bound to another stable source job id"
            )
        effective_source_job_id = (
            evidence.source_job_id or existing_source_job_id
            if evidence.evidence_kind == "platform_adapter"
            else None
        )
        bound_platform_identity = None
        if row is not None and evidence.evidence_kind == "web":
            bound_platform_identity = conn.execute(
                "SELECT source_name, source_job_id FROM job_posting_evidence "
                "WHERE job_id = ? AND evidence_kind = 'platform_adapter'",
                (int(row["id"]),),
            ).fetchone()
        materialized_source = (
            str(bound_platform_identity["source_name"])
            if bound_platform_identity is not None
            else evidence.source_name
        )
        materialized_source_job_id = (
            str(bound_platform_identity["source_job_id"])
            if bound_platform_identity is not None
            and bound_platform_identity["source_job_id"]
            else effective_source_job_id
        )

        raw_job = RawJob(
            source=materialized_source,
            source_id=materialized_source_job_id,
            url=evidence.canonical_url,
            company=evidence.company,
            title=evidence.title,
            location=evidence.location,
            raw_text=evidence.jd_text,
            extras={},
        )
        base_hash = content_hash(raw_job)
        target_id = int(row["id"]) if row is not None else None
        compatible_hash = _available_job_content_hash(
            conn,
            source=materialized_source,
            base_hash=base_hash,
            stable_key=stable_key,
            target_id=target_id,
        )
        existing_extras = _stored_json_object(row["extras_json"]) if row is not None else {}
        existing_extras.update(
            {
                "source_evidence_id": evidence.source_evidence_id,
                "source_status": evidence.source_status,
                "page_time_information": evidence.page_time_information,
            }
        )
        extras_json = json.dumps(
            existing_extras, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if target_id is None:
            job_id = int(
                conn.execute(
                    "INSERT INTO jobs("
                    "source, source_id, url, title, company, location, raw_text, "
                    "extras_json, content_hash"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                    (
                        evidence.source_name,
                        effective_source_job_id,
                        evidence.canonical_url,
                        evidence.title,
                        evidence.company,
                        evidence.location,
                        evidence.jd_text,
                        extras_json,
                        compatible_hash,
                    ),
                ).fetchone()[0]
            )
            return job_id, effective_source_job_id
        if evidence.evidence_kind == "web":
            if bound_platform_identity is None:
                conn.execute(
                    "UPDATE jobs SET source = ?, source_id = NULL, url = ?, title = ?, "
                    "company = ?, location = ?, raw_text = ?, extras_json = ?, "
                    "content_hash = ?, fetched_at = julianday('now') WHERE id = ?",
                    (
                        evidence.source_name,
                        evidence.canonical_url,
                        evidence.title,
                        evidence.company,
                        evidence.location,
                        evidence.jd_text,
                        extras_json,
                        compatible_hash,
                        target_id,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE jobs SET url = ?, title = ?, company = ?, location = ?, "
                    "raw_text = ?, extras_json = ?, content_hash = ?, "
                    "fetched_at = julianday('now') WHERE id = ?",
                    (
                        evidence.canonical_url,
                        evidence.title,
                        evidence.company,
                        evidence.location,
                        evidence.jd_text,
                        extras_json,
                        compatible_hash,
                        target_id,
                    ),
                )
        else:
            conn.execute(
                "UPDATE jobs SET source_id = ?, url = ?, title = ?, company = ?, "
                "location = ?, raw_text = ?, extras_json = ?, content_hash = ?, "
                "fetched_at = julianday('now') WHERE id = ?",
                (
                    effective_source_job_id,
                    evidence.canonical_url,
                    evidence.title,
                    evidence.company,
                    evidence.location,
                    evidence.jd_text,
                    extras_json,
                    compatible_hash,
                    target_id,
                ),
            )
        return target_id, effective_source_job_id

    def get_job_evidence(self, job_evidence_id: int) -> JobPostingEvidence | None:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM job_posting_evidence WHERE id = ?",
                (job_evidence_id,),
            ).fetchone()
        return _job_evidence_from_row(row) if row is not None else None

    def list_job_evidence(self) -> list[JobPostingEvidence]:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM job_posting_evidence ORDER BY checked_at DESC, id DESC"
            ).fetchall()
        return [_job_evidence_from_row(row) for row in rows]

    def list_job_evidence_catalog(
        self,
        *,
        offset: int = 0,
        limit: int = _INITIAL_CATALOG_PAGE_SIZE,
    ) -> JobEvidenceCatalogPage:
        if offset < 0:
            raise ValueError("job catalog offset must be non-negative")
        if not 1 <= limit <= 100:
            raise ValueError("job catalog limit must be between 1 and 100")
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            total = int(
                conn.execute("SELECT COUNT(*) FROM job_posting_evidence").fetchone()[0]
            )
            rows = conn.execute(
                "SELECT id, job_id, source_name, source_job_id, canonical_url, "
                "company, title, location, recruitment_type, source_status, checked_at "
                "FROM job_posting_evidence "
                "ORDER BY checked_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        items = [
            JobEvidenceCatalogItem(
                job_evidence_id=int(row["id"]),
                job_id=int(row["job_id"]),
                source_name=str(row["source_name"]),
                source_job_id=row["source_job_id"],
                canonical_url=str(row["canonical_url"]),
                company=str(row["company"]),
                title=str(row["title"]),
                location=row["location"],
                recruitment_type=row["recruitment_type"],
                source_status=cast(JobSourceStatus, str(row["source_status"])),
                checked_at=float(row["checked_at"]),
            )
            for row in rows
        ]
        consumed = offset + len(items)
        return JobEvidenceCatalogPage(
            offset=offset,
            limit=limit,
            total=total,
            items=items,
            next_offset=consumed if consumed < total else None,
        )

    def get_current_selection(self) -> JobSelectionSet | None:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT selection_json, published_at FROM job_discovery_current "
                "WHERE singleton = 1"
            ).fetchone()
        if row is None:
            return None
        payload = _json_object(row["selection_json"], label="current selection")
        payload["published_at"] = float(row["published_at"])
        try:
            return JobSelectionSet.model_validate(payload)
        except ValueError:
            return None

    def get_selection_revision(self, result_revision: int) -> JobSelectionSet | None:
        """Read one immutable publication used by a user-visible action link."""
        if result_revision < 1:
            raise ValueError("result revision must be positive")
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT selection_json, published_at "
                "FROM job_discovery_selection_revisions WHERE result_revision = ?",
                (result_revision,),
            ).fetchone()
        if row is None:
            return None
        payload = _json_object(row["selection_json"], label="selection revision")
        payload["published_at"] = float(row["published_at"])
        try:
            return JobSelectionSet.model_validate(payload)
        except ValueError:
            # Keep the revision number reserved, but never make an old result
            # without current evidence/grounding actionable.
            return None

    def current_result_revision(self) -> int:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT MAX(result_revision) FROM ("
                "SELECT result_revision FROM job_discovery_current "
                "UNION ALL SELECT result_revision FROM job_discovery_selection_revisions"
                ")"
            ).fetchone()
        return int(row[0]) if row is not None and row[0] is not None else 0

    def load_snapshot(self, candidate_evidence: CandidateEvidence) -> JobDiscoverySnapshot:
        context = self.get_search_context()
        if context is None:
            raise JobDiscoveryNotConfiguredError("job search context has not been created")
        current_selection = self.get_current_selection()
        return JobDiscoverySnapshot(
            search_context=context,
            candidate_evidence=candidate_evidence,
            recorded_job_catalog=self.list_job_evidence_catalog(),
            current_selection=current_selection,
            current_result_revision=self.current_result_revision(),
        )

    def publish_selection(
        self,
        *,
        expected_context_revision: int,
        expected_result_revision: int,
        items: list[JobSelectionItem],
        candidate_evidence: CandidateEvidence,
        coverage_summary: str,
        evidence_gaps: list[str],
    ) -> JobSelectionSet:
        """Atomically replace the current result if both bound revisions still match."""

        guard: RevisionGuard[JobSelectionSet] = RevisionGuard(self._revision_snapshot)

        def _write(conn: sqlite3.Connection) -> JobSelectionSet:
            conn.row_factory = sqlite3.Row
            context_row = conn.execute(
                "SELECT revision, intent_text, hard_constraints_json, feedback_json "
                "FROM job_search_context "
                "WHERE singleton = 1 AND revision = ?",
                (expected_context_revision,),
            ).fetchone()
            if context_row is None:
                raise JobDiscoveryRevisionConflict(
                    "job search context changed before publication"
                )
            published_items = self._selection_items_with_snapshots(
                conn,
                items,
                search_context=_search_context_from_row(context_row),
                candidate_evidence=candidate_evidence,
            )
            candidate = JobSelectionSet(
                context_revision=expected_context_revision,
                result_revision=expected_result_revision + 1,
                items=published_items,
                coverage_summary=coverage_summary,
                evidence_gaps=evidence_gaps,
            )
            payload = candidate.model_dump(mode="json", exclude={"published_at"})
            selection_json = json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            published_at = float(
                conn.execute("SELECT julianday('now')").fetchone()[0]
            )
            conn.execute(
                "INSERT INTO job_discovery_selection_revisions("
                "result_revision, context_revision, selection_json, published_at"
                ") VALUES (?, ?, ?, ?)",
                (
                    candidate.result_revision,
                    candidate.context_revision,
                    selection_json,
                    published_at,
                ),
            )
            conn.execute(
                "INSERT INTO job_discovery_current("
                "singleton, context_revision, result_revision, selection_json, published_at"
                ") VALUES (1, ?, ?, ?, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET "
                "context_revision = excluded.context_revision, "
                "result_revision = excluded.result_revision, "
                "selection_json = excluded.selection_json, "
                "published_at = excluded.published_at",
                (
                    candidate.context_revision,
                    candidate.result_revision,
                    selection_json,
                    published_at,
                ),
            )
            return candidate.model_copy(update={"published_at": published_at})

        try:
            return guard.compare_and_swap(
                self.store,
                expected_subject_revision=expected_context_revision,
                expected_result_revision=expected_result_revision,
                write=_write,
            )
        except StaleRevisionError as exc:
            raise JobDiscoveryRevisionConflict(str(exc)) from exc

    def assert_revisions(
        self, *, expected_context_revision: int, expected_result_revision: int
    ) -> None:
        """Check whether a run is still entitled to confirm the result unchanged."""

        guard: RevisionGuard[None] = RevisionGuard(self._revision_snapshot)
        try:
            with self.store.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                guard.assert_current(
                    conn,
                    expected_subject_revision=expected_context_revision,
                    expected_result_revision=expected_result_revision,
                )
        except StaleRevisionError as exc:
            raise JobDiscoveryRevisionConflict(str(exc)) from exc

    @staticmethod
    def _revision_snapshot(conn: sqlite3.Connection) -> RevisionSnapshot:
        context_row = conn.execute(
            "SELECT revision FROM job_search_context WHERE singleton = 1"
        ).fetchone()
        result_row = conn.execute(
            "SELECT MAX(result_revision) FROM ("
            "SELECT result_revision FROM job_discovery_current "
            "UNION ALL SELECT result_revision FROM job_discovery_selection_revisions"
            ")"
        ).fetchone()
        return RevisionSnapshot(
            subject_revision=int(context_row[0]) if context_row is not None else 0,
            result_revision=(
                int(result_row[0])
                if result_row is not None and result_row[0] is not None
                else 0
            ),
        )

    @staticmethod
    def _selection_items_with_snapshots(
        conn: sqlite3.Connection,
        items: list[JobSelectionItem],
        *,
        search_context: JobSearchContext,
        candidate_evidence: CandidateEvidence,
    ) -> list[JobSelectionItem]:
        published: list[JobSelectionItem] = []
        grounding_sources = {
            "search-context:intent": search_context.intent,
            **{
                f"search-context:hard-constraint:{index}": value
                for index, value in enumerate(search_context.hard_constraints, start=1)
            },
            **{
                f"search-context:feedback:{index}": value
                for index, value in enumerate(search_context.feedback, start=1)
            },
            **{
                document.reference: document.text
                for document in candidate_evidence.documents
            },
        }
        for item in items:
            grounding_source_texts: list[str] = []
            if not item.grounding_quotes:
                raise JobDiscoveryEvidenceError(
                    "every selected job requires an exact search-context or candidate quote"
                )
            for grounding in item.grounding_quotes:
                source_text = grounding_sources.get(grounding.reference)
                if source_text is None:
                    raise JobDiscoveryEvidenceError(
                        "selection grounding reference is not in the current Agent snapshot: "
                        f"{grounding.reference}"
                    )
                if not evidence_quote_matches_source(grounding.quote, source_text):
                    raise JobDiscoveryEvidenceError(
                        "selection grounding quote is not a copied passage from "
                        f"{grounding.reference}: {grounding.quote}"
                    )
                grounding_source_texts.append(source_text)
            row = conn.execute(
                "SELECT * FROM job_posting_evidence WHERE id = ?",
                (item.job_evidence_id,),
            ).fetchone()
            if row is None:
                raise JobDiscoveryEvidenceError(
                    f"job evidence {item.job_evidence_id} does not exist"
                )
            evidence = _job_evidence_from_row(row)
            if (
                evidence.job_id is None
                or not evidence.canonical_url.strip()
                or not evidence.jd_text.strip()
            ):
                raise JobDiscoveryEvidenceError(
                    f"job evidence {item.job_evidence_id} is incomplete"
                )
            try:
                require_substantive_job_description(
                    evidence.jd_text,
                    identity_values=(
                        evidence.company,
                        evidence.title,
                        evidence.location,
                        evidence.recruitment_type,
                        *evidence.page_time_information,
                    ),
                )
                require_grounded_selection_reason(
                    item.why_worth_attention,
                    grounding_quotes=(quote.quote for quote in item.grounding_quotes),
                    grounding_source_texts=grounding_source_texts,
                    job_text=evidence.jd_text,
                )
            except ValueError as exc:
                raise JobDiscoveryEvidenceError(str(exc)) from exc
            published.append(item.model_copy(update={"job_evidence": evidence}))
        return published


def _backfill_legacy_job_ids(conn: sqlite3.Connection) -> None:
    """Attach evidence written by the pre-integration schema to real jobs rows."""

    rows = conn.execute(
        "SELECT evidence.* FROM job_posting_evidence AS evidence "
        "LEFT JOIN jobs ON jobs.id = evidence.job_id "
        "WHERE evidence.job_id IS NULL OR jobs.id IS NULL "
        "ORDER BY evidence.id ASC"
    ).fetchall()
    for row in rows:
        evidence = _job_evidence_from_row(row)
        try:
            canonical_url = canonicalize_job_url(evidence.canonical_url)
        except ValueError as exc:
            raise JobDiscoveryError(
                f"cannot migrate job evidence {evidence.id}: invalid source URL"
            ) from exc
        normalized = evidence.model_copy(update={"canonical_url": canonical_url})
        provisional_key = job_stable_key(
            evidence_kind=normalized.evidence_kind,
            source_name=normalized.source_name,
            source_job_id=normalized.source_job_id,
            canonical_url=normalized.canonical_url,
        )
        job_id, effective_source_job_id = (
            JobDiscoveryRepository._materialize_application_job(
                conn,
                normalized,
                stable_key=provisional_key,
            )
        )
        normalized = normalized.model_copy(
            update={"source_job_id": effective_source_job_id}
        )
        stable_key = job_stable_key(
            evidence_kind=normalized.evidence_kind,
            source_name=normalized.source_name,
            source_job_id=normalized.source_job_id,
            canonical_url=normalized.canonical_url,
        )
        collision = conn.execute(
            "SELECT id FROM job_posting_evidence WHERE stable_key = ? AND id != ?",
            (stable_key, evidence.id),
        ).fetchone()
        conn.execute(
            "UPDATE job_posting_evidence SET job_id = ?, source_job_id = ?, "
            "canonical_url = ?, stable_key = ?, updated_at = julianday('now') "
            "WHERE id = ?",
            (
                job_id,
                effective_source_job_id,
                canonical_url,
                stable_key if collision is None else row["stable_key"],
                evidence.id,
            ),
        )


def _enforce_migrated_job_id_invariants(conn: sqlite3.Connection) -> None:
    """Finish the temporary nullable-column migration as the final strict schema."""

    duplicate_job_ids = conn.execute(
        "SELECT job_id FROM job_posting_evidence WHERE job_id IS NOT NULL "
        "GROUP BY job_id HAVING COUNT(*) > 1"
    ).fetchall()
    cleared_current_selection = False
    for duplicate in duplicate_job_ids:
        rows = conn.execute(
            "SELECT id FROM job_posting_evidence WHERE job_id = ? "
            "ORDER BY updated_at DESC, checked_at DESC, id DESC",
            (duplicate["job_id"],),
        ).fetchall()
        loser_ids = [int(row["id"]) for row in rows[1:]]
        conn.executemany(
            "DELETE FROM job_posting_evidence WHERE id = ?",
            [(evidence_id,) for evidence_id in loser_ids],
        )
        cleared_current_selection = True
    if cleared_current_selection:
        # JSON selections cannot be repointed without changing what the agent
        # actually published. Force a fresh publication instead.
        conn.execute("DELETE FROM job_discovery_current")

    for row in conn.execute("SELECT * FROM job_posting_evidence ORDER BY id").fetchall():
        evidence = _job_evidence_from_row(row)
        stable_key = job_stable_key(
            evidence_kind=evidence.evidence_kind,
            source_name=evidence.source_name,
            source_job_id=evidence.source_job_id,
            canonical_url=evidence.canonical_url,
        )
        collision = conn.execute(
            "SELECT id FROM job_posting_evidence WHERE stable_key = ? AND id != ?",
            (stable_key, evidence.id),
        ).fetchone()
        if collision is not None:
            raise JobDiscoveryError(
                "legacy job evidence contains conflicting source identities"
            )
        conn.execute(
            "UPDATE job_posting_evidence SET stable_key = ? WHERE id = ?",
            (stable_key, evidence.id),
        )

    invalid = conn.execute(
        "SELECT evidence.id FROM job_posting_evidence AS evidence "
        "LEFT JOIN jobs ON jobs.id = evidence.job_id "
        "WHERE evidence.job_id IS NULL OR jobs.id IS NULL LIMIT 1"
    ).fetchone()
    if invalid is not None:
        raise JobDiscoveryError(
            f"job evidence {int(invalid['id'])} could not be attached to a real job"
        )
    duplicate = conn.execute(
        "SELECT job_id FROM job_posting_evidence GROUP BY job_id "
        "HAVING COUNT(*) > 1 LIMIT 1"
    ).fetchone()
    if duplicate is not None:
        raise JobDiscoveryError("job evidence migration left duplicate application jobs")

    table_info = {
        str(row["name"]): row
        for row in conn.execute("PRAGMA table_info(job_posting_evidence)").fetchall()
    }
    if int(table_info["job_id"]["notnull"]) != 1:
        _rebuild_job_posting_evidence_table(conn)

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_posting_source_evidence "
        "ON job_posting_evidence(source_evidence_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_posting_status "
        "ON job_posting_evidence(source_status, checked_at DESC)"
    )


def _rebuild_job_posting_evidence_table(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS job_posting_evidence_migrated")
    conn.execute(
        """
        CREATE TABLE job_posting_evidence_migrated (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id                  INTEGER NOT NULL UNIQUE REFERENCES jobs(id),
            stable_key              TEXT NOT NULL UNIQUE,
            evidence_kind           TEXT NOT NULL DEFAULT 'web'
                                        CHECK (evidence_kind IN ('web', 'platform_adapter')),
            source_evidence_id      INTEGER NOT NULL,
            source_name             TEXT NOT NULL,
            source_job_id           TEXT,
            canonical_url           TEXT NOT NULL,
            company                 TEXT NOT NULL,
            title                   TEXT NOT NULL,
            location                TEXT,
            recruitment_type        TEXT,
            page_time_json          TEXT NOT NULL DEFAULT '[]'
                                        CHECK (json_valid(page_time_json)
                                           AND json_type(page_time_json) = 'array'),
            jd_text                 TEXT NOT NULL CHECK (length(trim(jd_text)) > 0),
            source_status           TEXT NOT NULL DEFAULT 'unknown'
                                        CHECK (source_status IN ('open', 'closed', 'unknown')),
            content_sha256          TEXT NOT NULL CHECK (length(content_sha256) = 64),
            checked_at              REAL NOT NULL,
            last_seen_at            REAL NOT NULL,
            created_at              REAL NOT NULL DEFAULT (julianday('now')),
            updated_at              REAL NOT NULL DEFAULT (julianday('now'))
        )
        """
    )
    columns = (
        "id, job_id, stable_key, evidence_kind, source_evidence_id, "
        "source_name, source_job_id, "
        "canonical_url, company, title, location, recruitment_type, page_time_json, "
        "jd_text, source_status, content_sha256, checked_at, last_seen_at, "
        "created_at, updated_at"
    )
    conn.execute(
        f"INSERT INTO job_posting_evidence_migrated({columns}) "
        f"SELECT {columns} FROM job_posting_evidence"
    )
    conn.execute("DROP TABLE job_posting_evidence")
    conn.execute(
        "ALTER TABLE job_posting_evidence_migrated RENAME TO job_posting_evidence"
    )


def _migrate_current_selection_snapshots(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT selection_json FROM job_discovery_current WHERE singleton = 1"
    ).fetchone()
    if row is None:
        return
    try:
        payload = _json_object(row["selection_json"], label="current selection")
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("selection items must be a list")
        migrated_items: list[dict[str, Any]] = []
        changed = False
        for raw_item in raw_items:
            if not isinstance(raw_item, Mapping):
                raise ValueError("selection item must be an object")
            item = dict(raw_item)
            if item.get("job_evidence") is None:
                evidence_id = item.get("job_evidence_id")
                if not isinstance(evidence_id, int) or isinstance(evidence_id, bool):
                    raise ValueError("selection evidence id must be an integer")
                evidence_row = conn.execute(
                    "SELECT * FROM job_posting_evidence WHERE id = ?",
                    (evidence_id,),
                ).fetchone()
                if evidence_row is None:
                    raise ValueError("selection references missing evidence")
                item["job_evidence"] = _job_evidence_from_row(evidence_row).model_dump(
                    mode="json"
                )
                changed = True
            migrated_items.append(item)
        payload["items"] = migrated_items
        selection = JobSelectionSet.model_validate(payload)
    except (JobDiscoveryError, TypeError, ValueError):
        # Preserve the revision high-water mark so it can never be reused, but
        # remove the ungrounded/dangling result from the current actionable slot.
        conn.execute(
            "INSERT OR IGNORE INTO job_discovery_selection_revisions("
            "result_revision, context_revision, selection_json, published_at"
            ") SELECT result_revision, context_revision, selection_json, published_at "
            "FROM job_discovery_current WHERE singleton = 1"
        )
        conn.execute("DELETE FROM job_discovery_current WHERE singleton = 1")
        return
    if not changed:
        return
    migrated_json = json.dumps(
        selection.model_dump(mode="json", exclude={"published_at"}),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    conn.execute(
        "UPDATE job_discovery_current SET selection_json = ? WHERE singleton = 1",
        (migrated_json,),
    )


def _backfill_selection_history(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT context_revision, result_revision, selection_json, published_at "
        "FROM job_discovery_current WHERE singleton = 1"
    ).fetchone()
    if row is None:
        return
    existing = conn.execute(
        "SELECT context_revision, selection_json FROM job_discovery_selection_revisions "
        "WHERE result_revision = ?",
        (int(row["result_revision"]),),
    ).fetchone()
    if existing is not None:
        if (
            int(existing["context_revision"]) != int(row["context_revision"])
            or str(existing["selection_json"]) != str(row["selection_json"])
        ):
            raise JobDiscoveryError(
                "selection history conflicts with the current published result"
            )
        return
    conn.execute(
        "INSERT INTO job_discovery_selection_revisions("
        "result_revision, context_revision, selection_json, published_at"
        ") VALUES (?, ?, ?, ?)",
        (
            int(row["result_revision"]),
            int(row["context_revision"]),
            str(row["selection_json"]),
            float(row["published_at"]),
        ),
    )


def canonicalize_job_url(url: str) -> str:
    """Normalize identity-relevant URL parts without guessing source semantics."""

    value = url.strip()
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("job URL must be an absolute HTTP(S) URL")
    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query_items = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    return urlunsplit((scheme, host, path, urlencode(sorted(query_items)), ""))


def job_stable_key(
    *,
    evidence_kind: str,
    source_name: str,
    source_job_id: str | None,
    canonical_url: str,
) -> str:
    if evidence_kind == "web":
        return f"web-url:{canonicalize_job_url(canonical_url)}"
    if evidence_kind != "platform_adapter":
        raise ValueError("unknown job evidence kind")
    source = source_name.strip().lower()
    if not source:
        raise ValueError("source name must not be blank")
    if source_job_id and source_job_id.strip():
        return f"source-id:{source}:{source_job_id.strip()}"
    return f"source-url:{source}:{canonicalize_job_url(canonical_url)}"


def _search_context_from_row(row: sqlite3.Row) -> JobSearchContext:
    return JobSearchContext(
        revision=int(row["revision"]),
        intent=str(row["intent_text"]),
        hard_constraints=_json_string_list(
            row["hard_constraints_json"], label="hard constraints"
        ),
        feedback=_json_string_list(row["feedback_json"], label="feedback"),
    )


def _job_evidence_from_row(row: sqlite3.Row) -> JobPostingEvidence:
    return JobPostingEvidence(
        id=int(row["id"]),
        job_id=int(row["job_id"]) if row["job_id"] is not None else None,
        evidence_kind=cast(JobEvidenceKind, str(row["evidence_kind"])),
        source_evidence_id=int(row["source_evidence_id"]),
        source_name=str(row["source_name"]),
        source_job_id=str(row["source_job_id"]) if row["source_job_id"] else None,
        canonical_url=str(row["canonical_url"]),
        company=str(row["company"]),
        title=str(row["title"]),
        location=str(row["location"]) if row["location"] else None,
        recruitment_type=(
            str(row["recruitment_type"]) if row["recruitment_type"] else None
        ),
        page_time_information=_json_string_list(
            row["page_time_json"], label="page time information"
        ),
        jd_text=str(row["jd_text"]),
        source_status=cast(JobSourceStatus, str(row["source_status"])),
        checked_at=float(row["checked_at"]),
        last_seen_at=float(row["last_seen_at"]),
        content_sha256=str(row["content_sha256"]),
    )


def _json_array(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _json_string_list(raw: str, *, label: str) -> list[str]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise JobDiscoveryError(f"stored {label} is invalid JSON") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise JobDiscoveryError(f"stored {label} must be a string list")
    return value


def _json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise JobDiscoveryError(f"stored {label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise JobDiscoveryError(f"stored {label} must be an object")
    return dict(value)


def _stored_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def _available_job_content_hash(
    conn: sqlite3.Connection,
    *,
    source: str,
    base_hash: str,
    stable_key: str,
    target_id: int | None,
) -> str:
    collision = conn.execute(
        "SELECT id FROM jobs WHERE source = ? AND content_hash = ?",
        (source, base_hash),
    ).fetchone()
    if collision is None or (target_id is not None and int(collision[0]) == target_id):
        return base_hash
    # The legacy table forbids two same-source rows with identical content,
    # even when the source supplies distinct stable position ids. Preserve
    # those identities with a deterministic compatibility hash; the new
    # discovery system never uses this field for identity.
    return hashlib.sha256(f"{base_hash}\0{stable_key}".encode()).hexdigest()
