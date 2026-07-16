"""Persistence for the one resume workspace attached to an application.

The workspace is mutable only while it is a draft. Submission freezes the
employer-received material and records the application lifecycle transition in
the same SQLite transaction. Post-submission research binds to that frozen
workspace without rewriting it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ..memory import Store
from .master import MasterResumeDocument, MasterResumeSource
from .models import ApplicationPackage, ResumeContext, ResumeDocument

WorkspaceStatus = Literal["draft", "submitted"]
SemanticStatus = Literal["draft", "confirmed"]

_WORKSPACE_COLUMNS = (
    "id, application_id, status, job_snapshot_json, master_source_sha256, "
    "context_json, resume_document_json, pdf_path, pdf_sha256, "
    "apply_pack_json, created_at, updated_at, submitted_at"
)


class ResumeWorkspaceError(RuntimeError):
    """Invalid resume workspace state or operation."""


class MasterResumeNotFoundError(ResumeWorkspaceError):
    pass


class WorkspaceNotFoundError(ResumeWorkspaceError):
    pass


class WorkspaceConflictError(ResumeWorkspaceError):
    pass


class WorkspaceSubmittedError(ResumeWorkspaceError):
    pass


class WorkspaceNotReadyError(ResumeWorkspaceError):
    pass


class ArtifactIntegrityError(ResumeWorkspaceError):
    pass


@dataclass(frozen=True, slots=True)
class MasterResume:
    source_path: str
    source_sha256: str
    extracted_text: str
    semantic_document: dict[str, Any]
    semantic_status: SemanticStatus
    created_at: float
    updated_at: float
    confirmed_at: float | None


@dataclass(frozen=True, slots=True)
class ResumeWorkspace:
    id: int
    application_id: int
    status: WorkspaceStatus
    job_snapshot: dict[str, Any]
    master_source_sha256: str
    context: dict[str, Any]
    resume_document: dict[str, Any]
    pdf_path: str | None
    pdf_sha256: str | None
    apply_pack: dict[str, Any]
    created_at: float
    updated_at: float
    submitted_at: float | None

    @property
    def is_submitted(self) -> bool:
        return self.status == "submitted"


class ResumeWorkspaceRepository:
    def __init__(self, store: Store) -> None:
        self.store = store

    def save_master(
        self,
        *,
        source_path: str | Path,
        source_sha256: str,
        extracted_text: str,
        semantic_document: Mapping[str, Any],
        confirmed: bool = False,
    ) -> MasterResume:
        """Replace the one current master after verifying its source PDF."""
        path, sha = _verified_file(source_path, source_sha256, label="master PDF")
        text = str(extracted_text or "").strip()
        if not text:
            raise ValueError("master resume extraction produced no text")
        semantic = MasterResumeDocument.model_validate(semantic_document)
        if semantic.source_sha256 != sha:
            raise WorkspaceConflictError("master semantic document does not match the source PDF")
        if semantic.confirmed_by_user != confirmed:
            raise WorkspaceConflictError(
                "master semantic confirmation does not match the stored status"
            )
        semantic_json = _dump_object(semantic.model_dump(mode="json"), "semantic_document")
        status = "confirmed" if confirmed else "draft"

        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO master_resume("
                "id, source_path, source_sha256, extracted_text, "
                "semantic_document_json, semantic_status, confirmed_at"
                ") VALUES (1, ?, ?, ?, ?, ?, "
                "CASE WHEN ? = 'confirmed' THEN julianday('now') ELSE NULL END) "
                "ON CONFLICT(id) DO UPDATE SET "
                "source_path = excluded.source_path, "
                "source_sha256 = excluded.source_sha256, "
                "extracted_text = excluded.extracted_text, "
                "semantic_document_json = excluded.semantic_document_json, "
                "semantic_status = excluded.semantic_status, "
                "updated_at = julianday('now'), "
                "confirmed_at = excluded.confirmed_at",
                (path, sha, text, semantic_json, status, status),
            )
            return _master_row(_master_row_from_db(conn))

    def get_master(self) -> MasterResume | None:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = _master_row_from_db(conn)
        return _master_row(row) if row is not None else None

    def effective_master_text(self, source: MasterResumeSource | None) -> str | None:
        """Prefer the user's confirmed semantic text for the configured PDF."""
        if source is None:
            return None
        stored = self.get_master()
        if (
            stored is None
            or stored.source_sha256 != source.sha256
            or stored.semantic_status != "confirmed"
        ):
            return source.extracted_text
        semantic = MasterResumeDocument.model_validate(stored.semantic_document)
        if not semantic.confirmed_by_user or semantic.source_sha256 != source.sha256:
            raise WorkspaceConflictError("confirmed master resume is inconsistent")
        return semantic.semantic_text

    def application_for_job(self, job_id: int, *, create: bool) -> tuple[int | None, bool]:
        """Resolve the application that owns a job's sole resume workspace."""
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job_exists = conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job_exists is None:
                raise WorkspaceNotFoundError(f"job {job_id} not found")
            rows = conn.execute(
                "SELECT a.id, rw.id FROM applications a "
                "LEFT JOIN resume_workspaces rw ON rw.application_id = a.id "
                "WHERE a.job_id = ? ORDER BY a.id ASC",
                (job_id,),
            ).fetchall()
            if len(rows) > 1:
                raise WorkspaceConflictError(
                    f"job {job_id} has multiple application records; "
                    "a job-level action cannot guess which submission is current"
                )
            if rows:
                return int(rows[0][0]), False
            if not create:
                return None, False
            application_id = int(
                conn.execute(
                    "INSERT INTO applications(job_id, status) "
                    "VALUES (?, 'considered') RETURNING id",
                    (job_id,),
                ).fetchone()[0]
            )
            return application_id, True

    def get(self, application_id: int) -> ResumeWorkspace | None:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = _workspace_row_for_application(conn, application_id)
        return _workspace_row(row) if row is not None else None

    def save_draft(
        self,
        application_id: int,
        *,
        job_snapshot: Mapping[str, Any],
        master_source_sha256: str,
        context: Mapping[str, Any],
        resume_document: Mapping[str, Any],
        pdf_path: str | Path,
        pdf_sha256: str,
        apply_pack: Mapping[str, Any],
    ) -> ResumeWorkspace:
        """Atomically create or replace every part of one usable draft.

        Model editing, rendering, and application-copy generation happen before
        this call. A failure in any of those stages therefore leaves the last
        complete draft untouched instead of mixing old and new revisions.
        """
        job_json = _dump_object(job_snapshot, "job_snapshot")
        master_sha = _sha256(master_source_sha256, "master_source_sha256")
        context_json = _validated_context_json(context, master_sha)
        document = ResumeDocument.model_validate(resume_document)
        document_json = _dump_object(document.model_dump(mode="json"), "resume_document")
        _require_complete_apply_pack(apply_pack)
        pack_json = _dump_object(apply_pack, "apply_pack")
        path, pdf_sha = _verified_file(pdf_path, pdf_sha256, label="resume PDF")

        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            job_id = _application_job_id(conn, application_id)
            _validate_job_snapshot(job_snapshot, job_id)
            row = _workspace_row_for_application(conn, application_id)
            if row is None:
                _require_current_master(conn, master_sha)
                workspace_id = int(
                    conn.execute(
                        "INSERT INTO resume_workspaces("
                        "application_id, job_snapshot_json, master_source_sha256, "
                        "context_json, resume_document_json, pdf_path, pdf_sha256, "
                        "apply_pack_json"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                        (
                            application_id,
                            job_json,
                            master_sha,
                            context_json,
                            document_json,
                            path,
                            pdf_sha,
                            pack_json,
                        ),
                    ).fetchone()[0]
                )
                return _workspace_row(_workspace_row_by_id(conn, workspace_id))
            if row["status"] != "draft":
                raise WorkspaceSubmittedError("submitted resume workspace is immutable")
            if row["job_snapshot_json"] != job_json:
                raise WorkspaceConflictError("the draft job snapshot cannot change")
            previous_master_sha = str(row["master_source_sha256"])
            if master_sha != previous_master_sha:
                _require_current_master(conn, master_sha)
            conn.execute(
                "UPDATE resume_workspaces SET master_source_sha256 = ?, "
                "context_json = ?, resume_document_json = ?, pdf_path = ?, "
                "pdf_sha256 = ?, apply_pack_json = ?, updated_at = julianday('now') "
                "WHERE id = ?",
                (
                    master_sha,
                    context_json,
                    document_json,
                    path,
                    pdf_sha,
                    pack_json,
                    int(row["id"]),
                ),
            )
            return _workspace_row(_workspace_row_by_id(conn, int(row["id"])))

    def submit(self, application_id: int) -> ResumeWorkspace:
        """Freeze the workspace and record submission as one atomic transition."""
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            _application_job_id(conn, application_id)
            row = _workspace_row_for_application(conn, application_id)
            if row is None:
                raise WorkspaceNotFoundError(
                    f"application {application_id} has no resume workspace"
                )
            if row["status"] == "submitted":
                return _workspace_row(row)
            document = _load_object(row["resume_document_json"], "resume_document")
            if not document or not row["pdf_path"] or not row["pdf_sha256"]:
                raise WorkspaceNotReadyError(
                    "the draft needs a ResumeDocument and verified PDF before submission"
                )
            apply_pack = _load_object(row["apply_pack_json"], "apply_pack")
            _require_complete_apply_pack(apply_pack)
            _verified_file(
                str(row["pdf_path"]),
                str(row["pdf_sha256"]),
                label="resume PDF",
            )
            workspace_id = int(row["id"])
            conn.execute(
                "UPDATE resume_workspaces SET status = 'submitted', "
                "submitted_at = julianday('now'), updated_at = julianday('now') "
                "WHERE id = ?",
                (workspace_id,),
            )
            conn.execute(
                "UPDATE applications SET status = 'applied', "
                "applied_at = COALESCE(applied_at, julianday('now')), "
                "last_status_change = julianday('now') WHERE id = ?",
                (application_id,),
            )
            conn.execute(
                "INSERT INTO application_events("
                "application_id, kind, source, payload_json"
                ") VALUES (?, 'submitted', 'manual', ?)",
                (
                    application_id,
                    json.dumps(
                        {"workspace_id": workspace_id},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            return _workspace_row(_workspace_row_by_id(conn, workspace_id))

def _application_job_id(conn: sqlite3.Connection, application_id: int) -> int:
    row = conn.execute(
        "SELECT job_id FROM applications WHERE id = ?",
        (application_id,),
    ).fetchone()
    if row is None:
        raise WorkspaceNotFoundError(f"application {application_id} not found")
    return int(row[0])


def _validate_job_snapshot(snapshot: Mapping[str, Any], job_id: int) -> None:
    embedded = snapshot.get("job_id", snapshot.get("id"))
    if embedded is None:
        raise WorkspaceConflictError("job_snapshot must contain job_id or id")
    try:
        embedded_id = int(embedded)
    except (TypeError, ValueError) as exc:
        raise WorkspaceConflictError("job_snapshot has an invalid job id") from exc
    if embedded_id != job_id:
        raise WorkspaceConflictError("job_snapshot does not match the application")


def _require_current_master(conn: sqlite3.Connection, source_sha256: str) -> None:
    row = conn.execute(
        "SELECT source_sha256, semantic_status FROM master_resume WHERE id = 1"
    ).fetchone()
    if row is None:
        raise MasterResumeNotFoundError("master resume has not been saved")
    if str(row[0]) != source_sha256:
        raise WorkspaceConflictError("workspace master source is not the current master")
    if str(row[1]) != "confirmed":
        raise WorkspaceNotReadyError("master semantic document has not been confirmed")


def _master_row_from_db(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT source_path, source_sha256, extracted_text, "
        "semantic_document_json, semantic_status, created_at, updated_at, confirmed_at "
        "FROM master_resume WHERE id = 1"
    ).fetchone()


def _workspace_row_for_application(
    conn: sqlite3.Connection,
    application_id: int,
) -> sqlite3.Row | None:
    return conn.execute(
        f"SELECT {_WORKSPACE_COLUMNS} FROM resume_workspaces WHERE application_id = ?",
        (application_id,),
    ).fetchone()


def _workspace_row_by_id(
    conn: sqlite3.Connection,
    workspace_id: int,
) -> sqlite3.Row:
    row = conn.execute(
        f"SELECT {_WORKSPACE_COLUMNS} FROM resume_workspaces WHERE id = ?",
        (workspace_id,),
    ).fetchone()
    if row is None:
        raise WorkspaceNotFoundError(f"resume workspace {workspace_id} not found")
    return row


def _master_row(row: sqlite3.Row | None) -> MasterResume:
    if row is None:
        raise MasterResumeNotFoundError("master resume has not been saved")
    return MasterResume(
        source_path=str(row["source_path"]),
        source_sha256=str(row["source_sha256"]),
        extracted_text=str(row["extracted_text"]),
        semantic_document=_load_object(row["semantic_document_json"], "semantic_document"),
        semantic_status=cast(SemanticStatus, str(row["semantic_status"])),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        confirmed_at=(float(row["confirmed_at"]) if row["confirmed_at"] is not None else None),
    )


def _workspace_row(row: sqlite3.Row) -> ResumeWorkspace:
    return ResumeWorkspace(
        id=int(row["id"]),
        application_id=int(row["application_id"]),
        status=cast(WorkspaceStatus, str(row["status"])),
        job_snapshot=_load_object(row["job_snapshot_json"], "job_snapshot"),
        master_source_sha256=str(row["master_source_sha256"]),
        context=_load_object(row["context_json"], "context"),
        resume_document=_load_object(row["resume_document_json"], "resume_document"),
        pdf_path=str(row["pdf_path"]) if row["pdf_path"] else None,
        pdf_sha256=str(row["pdf_sha256"]) if row["pdf_sha256"] else None,
        apply_pack=_load_object(row["apply_pack_json"], "apply_pack"),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        submitted_at=(float(row["submitted_at"]) if row["submitted_at"] is not None else None),
    )


def _dump_object(value: Mapping[str, Any], field: str) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_complete_apply_pack(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("apply_pack must be a mapping")
    assistant = value.get("assistant")
    if not isinstance(assistant, Mapping) or assistant.get("_error"):
        raise WorkspaceNotReadyError(
            "the draft needs a valid application package before it can replace or freeze the current one"
        )
    try:
        ApplicationPackage.model_validate(assistant)
    except Exception as exc:
        raise WorkspaceNotReadyError(
            "the draft needs a valid application package before it can replace or freeze the current one"
        ) from exc


def _validated_context_json(
    value: Mapping[str, Any] | None,
    master_source_sha256: str,
) -> str:
    if value is None:
        raise ValueError("context must not be empty")
    context = ResumeContext.model_validate(value)
    if context.master_source.sha256 != master_source_sha256:
        raise WorkspaceConflictError("resume context does not match the workspace master source")
    return _dump_object(context.model_dump(mode="json", exclude_none=True), "context")


def _load_object(raw: str | None, field: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError) as exc:
        raise ResumeWorkspaceError(f"invalid {field} JSON") from exc
    if not isinstance(value, dict):
        raise ResumeWorkspaceError(f"{field} JSON must be an object")
    return cast(dict[str, Any], value)


def _sha256(value: str, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{field} must be a hexadecimal SHA-256")
    return normalized


def _verified_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[str, str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ArtifactIntegrityError(f"{label} file does not exist: {source}")
    expected = _sha256(expected_sha256, f"{label} SHA-256")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ArtifactIntegrityError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return str(source), expected
