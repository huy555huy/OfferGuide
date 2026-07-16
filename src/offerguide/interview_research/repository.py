"""Persistence and publish-time evidence checks for interview research."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from typing import Any

from ..memory import Store
from .models import (
    EvidenceDocument,
    InterviewAnswerSet,
    InterviewResearchSubject,
    InterviewSourceAssessment,
    PublishedInterviewMaterial,
)

EvidenceLoader = Callable[[str], EvidenceDocument | None]
CURRENT_INTERVIEW_EVIDENCE_CONTRACT_VERSION = 3
CURRENT_INTERVIEW_MATERIAL_CONTRACT_VERSION = 1

_PROJECT_GROUNDING_FIELDS = (
    "title",
    "mainstream_direction",
    "typical_problem",
    "project_task",
    "my_work",
    "method_route",
    "contribution_type",
    "contribution_detail",
    "key_difficulties",
    "resolution_process",
    "project_outputs",
    "evidence",
    "askable_points",
    "expression_boundary",
    "do_not_claim",
    "tags",
)


class InterviewResearchError(RuntimeError):
    """Base error for invalid interview research state or operations."""


class SubmittedWorkspaceRequiredError(InterviewResearchError):
    pass


class InterviewResearchConflictError(InterviewResearchError):
    pass


class InterviewEvidenceError(InterviewResearchError):
    pass


class InterviewResearchRepository:
    """Own the one research context and current material per frozen submission."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def ensure_subject(
        self,
        application_id: int,
        submitted_workspace_id: int,
    ) -> InterviewResearchSubject:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            self._ensure_context_row(conn, application_id, submitted_workspace_id)
            return self._load_subject(conn, application_id, submitted_workspace_id)

    def load_subject(
        self,
        application_id: int,
        submitted_workspace_id: int,
    ) -> InterviewResearchSubject:
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            self._require_context_row(conn, application_id, submitted_workspace_id)
            return self._load_subject(conn, application_id, submitted_workspace_id)

    def add_context_update(
        self,
        application_id: int,
        submitted_workspace_id: int,
        *,
        kind: str,
        content: Mapping[str, Any],
        expected_revision: int | None = None,
    ) -> InterviewResearchSubject:
        update_kind = str(kind or "").strip()
        if update_kind != "user_provided_source":
            raise ValueError("interview research context only accepts a user-provided source")
        content_json = _object_json(content, "research context update")
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = self._ensure_context_row(conn, application_id, submitted_workspace_id)
            old_revision = int(row["revision"])
            if expected_revision is not None and expected_revision != old_revision:
                raise InterviewResearchConflictError(
                    "interview research context changed before the update"
                )
            new_revision = old_revision + 1
            updated = conn.execute(
                "UPDATE interview_research_contexts SET revision = ?, "
                "updated_at = julianday('now') "
                "WHERE workspace_id = ? AND application_id = ? AND revision = ?",
                (
                    new_revision,
                    submitted_workspace_id,
                    application_id,
                    old_revision,
                ),
            )
            if updated.rowcount != 1:
                raise InterviewResearchConflictError(
                    "interview research context changed before the update"
                )
            conn.execute(
                "INSERT INTO interview_research_context_updates("
                "workspace_id, revision, kind, content_json) VALUES (?, ?, ?, ?)",
                (submitted_workspace_id, new_revision, update_kind, content_json),
            )
            return self._load_subject(conn, application_id, submitted_workspace_id)

    def assess_source(
        self,
        application_id: int,
        submitted_workspace_id: int,
        *,
        evidence: EvidenceDocument,
        assessment: InterviewSourceAssessment,
        expected_subject_token: str,
    ) -> None:
        if not evidence.complete:
            raise InterviewEvidenceError(
                "a partially read source cannot be assessed as interview evidence"
            )
        if not evidence.attached_to_subject:
            raise InterviewEvidenceError(
                "source evidence is not attached to this submitted workspace revision"
            )
        _validate_assessment_questions(assessment, evidence.text)
        evidence_hash = hashlib.sha256(evidence.text.encode("utf-8")).hexdigest()
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            subject = self._load_subject(conn, application_id, submitted_workspace_id)
            if subject.subject_token != expected_subject_token:
                raise InterviewResearchConflictError(
                    "interview research context changed before source assessment"
                )
            changed = conn.execute(
                "INSERT INTO interview_research_source_links("
                "workspace_id, evidence_id, evidence_text_sha256, subject_token, "
                "context_revision, assessment_json, contract_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(workspace_id, evidence_id, subject_token) DO UPDATE SET "
                "evidence_text_sha256 = excluded.evidence_text_sha256, "
                "context_revision = excluded.context_revision, "
                "assessment_json = excluded.assessment_json, "
                "contract_version = excluded.contract_version, "
                "updated_at = julianday('now') "
                "WHERE interview_research_source_links.evidence_text_sha256 "
                "!= excluded.evidence_text_sha256 "
                "OR interview_research_source_links.context_revision "
                "!= excluded.context_revision "
                "OR interview_research_source_links.assessment_json "
                "!= excluded.assessment_json "
                "OR interview_research_source_links.contract_version "
                "!= excluded.contract_version",
                (
                    submitted_workspace_id,
                    evidence.evidence_id,
                    evidence_hash,
                    subject.subject_token,
                    subject.context_revision,
                    _model_json(assessment),
                    CURRENT_INTERVIEW_EVIDENCE_CONTRACT_VERSION,
                ),
            )
            if changed.rowcount:
                conn.execute(
                    "UPDATE interview_research_contexts "
                    "SET evidence_revision = evidence_revision + 1, "
                    "updated_at = julianday('now') "
                    "WHERE workspace_id = ? AND application_id = ?",
                    (submitted_workspace_id, application_id),
                )

    def publish(
        self,
        application_id: int,
        submitted_workspace_id: int,
        *,
        answer_set: InterviewAnswerSet,
        expected_subject_token: str,
        expected_result_revision: int,
        evidence_loader: EvidenceLoader,
    ) -> PublishedInterviewMaterial:
        used_ids = _used_evidence_ids(answer_set)
        evidence_by_id = _load_answer_evidence(used_ids, evidence_loader)

        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            subject = self._load_subject(conn, application_id, submitted_workspace_id)
            if subject.subject_token != expected_subject_token:
                raise InterviewResearchConflictError(
                    "interview research subject changed; this material is stale"
                )
            if subject.result_revision != expected_result_revision:
                raise InterviewResearchConflictError(
                    "current interview material changed; this material is stale"
                )

            current_links = self._current_source_links(
                conn,
                submitted_workspace_id,
                subject.subject_token,
                used_ids,
            )
            _validate_material_evidence(answer_set, evidence_by_id, current_links)
            _validate_grounding(answer_set, subject)

            new_revision = expected_result_revision + 1
            created_at = float(conn.execute("SELECT julianday('now')").fetchone()[0])
            published = PublishedInterviewMaterial(
                application_id=application_id,
                submitted_workspace_id=submitted_workspace_id,
                subject_token=subject.subject_token,
                context_revision=subject.context_revision,
                result_revision=new_revision,
                contract_version=CURRENT_INTERVIEW_MATERIAL_CONTRACT_VERSION,
                evidence_revision=subject.evidence_revision,
                answer_set=answer_set,
                used_evidence_ids=used_ids,
                created_at=created_at,
            )
            conn.execute(
                "INSERT INTO interview_answer_set_revisions("
                "workspace_id, result_revision, application_id, subject_token, "
                "context_revision, answer_set_json, evidence_ids_json, contract_version, "
                "evidence_revision, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submitted_workspace_id,
                    new_revision,
                    application_id,
                    subject.subject_token,
                    subject.context_revision,
                    _model_json(answer_set),
                    json.dumps(used_ids, ensure_ascii=False, separators=(",", ":")),
                    CURRENT_INTERVIEW_MATERIAL_CONTRACT_VERSION,
                    subject.evidence_revision,
                    created_at,
                ),
            )
            updated = conn.execute(
                "UPDATE interview_research_contexts "
                "SET current_result_revision = ?, updated_at = julianday('now') "
                "WHERE workspace_id = ? AND application_id = ? "
                "AND revision = ? AND current_result_revision = ?",
                (
                    new_revision,
                    submitted_workspace_id,
                    application_id,
                    subject.context_revision,
                    expected_result_revision,
                ),
            )
            if updated.rowcount != 1:
                raise InterviewResearchConflictError(
                    "interview research changed during publication"
                )
            return published

    def validate_answer_set(
        self,
        application_id: int,
        submitted_workspace_id: int,
        *,
        answer_set: InterviewAnswerSet,
        expected_subject_token: str,
        evidence_loader: EvidenceLoader,
    ) -> None:
        """Validate a staged answer without writing a competing result version."""
        used_ids = _used_evidence_ids(answer_set)
        evidence_by_id = _load_answer_evidence(used_ids, evidence_loader)
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            subject = self._load_subject(conn, application_id, submitted_workspace_id)
            if subject.subject_token != expected_subject_token:
                raise InterviewResearchConflictError(
                    "interview research subject changed; this material is stale"
                )
            current_links = self._current_source_links(
                conn,
                submitted_workspace_id,
                subject.subject_token,
                None,
            )
            links = {
                evidence_id: current_links[evidence_id]
                for evidence_id in used_ids
                if evidence_id in current_links
            }
            _validate_material_evidence(answer_set, evidence_by_id, links)
            _validate_grounding(answer_set, subject)

    def get_current(
        self,
        application_id: int,
        submitted_workspace_id: int,
    ) -> PublishedInterviewMaterial | None:
        return self.load_subject(application_id, submitted_workspace_id).current_material

    def _ensure_context_row(
        self,
        conn: sqlite3.Connection,
        application_id: int,
        submitted_workspace_id: int,
    ) -> sqlite3.Row:
        _submitted_workspace(conn, application_id, submitted_workspace_id)
        conn.execute(
            "INSERT INTO interview_research_contexts(workspace_id, application_id) "
            "VALUES (?, ?) ON CONFLICT(workspace_id) DO NOTHING",
            (submitted_workspace_id, application_id),
        )
        return self._require_context_row(conn, application_id, submitted_workspace_id)

    def _require_context_row(
        self,
        conn: sqlite3.Connection,
        application_id: int,
        submitted_workspace_id: int,
    ) -> sqlite3.Row:
        row = conn.execute(
            "SELECT workspace_id, application_id, revision, current_result_revision, "
            "evidence_revision, created_at, updated_at FROM interview_research_contexts "
            "WHERE workspace_id = ? AND application_id = ?",
            (submitted_workspace_id, application_id),
        ).fetchone()
        if row is None:
            _submitted_workspace(conn, application_id, submitted_workspace_id)
            raise SubmittedWorkspaceRequiredError(
                "interview research context has not been initialized"
            )
        return row

    def _load_subject(
        self,
        conn: sqlite3.Connection,
        application_id: int,
        submitted_workspace_id: int,
    ) -> InterviewResearchSubject:
        context_row = self._require_context_row(conn, application_id, submitted_workspace_id)
        workspace = _submitted_workspace(conn, application_id, submitted_workspace_id)
        frozen_submission = _workspace_payload(workspace)
        project_vault = _matching_project_vault(conn, frozen_submission)
        preparation_notes = _preparation_notes(frozen_submission)
        context_updates = _context_updates(conn, submitted_workspace_id)
        subject_token = _subject_token(
            application_id=application_id,
            submitted_workspace_id=submitted_workspace_id,
            context_revision=int(context_row["revision"]),
            frozen_submission=frozen_submission,
            project_vault=project_vault,
            preparation_notes=preparation_notes,
            context_updates=context_updates,
        )
        agent_subject_revision = _agent_subject_revision(subject_token)
        source_assessments = _source_assessments(conn, submitted_workspace_id, subject_token)
        attached_sources = _attached_sources(
            conn,
            application_id,
            submitted_workspace_id,
            agent_subject_revision,
        )
        current = _current_material(
            conn,
            application_id,
            submitted_workspace_id,
            int(context_row["current_result_revision"]),
        )
        return InterviewResearchSubject(
            application_id=application_id,
            submitted_workspace_id=submitted_workspace_id,
            subject_token=subject_token,
            agent_subject_revision=agent_subject_revision,
            context_revision=int(context_row["revision"]),
            result_revision=int(context_row["current_result_revision"]),
            evidence_revision=int(context_row["evidence_revision"]),
            frozen_submission=frozen_submission,
            project_vault=project_vault,
            preparation_notes=preparation_notes,
            context_updates=context_updates,
            source_assessments=source_assessments,
            attached_sources=attached_sources,
            current_material=current,
            current_material_is_stale=(
                current is not None
                and (
                    current.subject_token != subject_token
                    or current.contract_version < CURRENT_INTERVIEW_MATERIAL_CONTRACT_VERSION
                    or current.evidence_revision != int(context_row["evidence_revision"])
                )
            ),
        )

    def _current_source_links(
        self,
        conn: sqlite3.Connection,
        submitted_workspace_id: int,
        subject_token: str,
        evidence_ids: list[str] | None,
    ) -> dict[str, dict[str, Any]]:
        if evidence_ids == []:
            return {}
        sql = (
            "SELECT evidence_id, evidence_text_sha256, assessment_json, "
            "contract_version "
            "FROM interview_research_source_links "
            "WHERE workspace_id = ? AND subject_token = ?"
        )
        params: tuple[Any, ...] = (submitted_workspace_id, subject_token)
        if evidence_ids is not None:
            placeholders = ",".join("?" for _ in evidence_ids)
            sql += f" AND evidence_id IN ({placeholders})"
            params = (*params, *evidence_ids)
        rows = conn.execute(sql, params).fetchall()
        return {
            str(row["evidence_id"]): {
                "text_sha256": str(row["evidence_text_sha256"]),
                "assessment": _load_object(row["assessment_json"]),
                "contract_version": int(row["contract_version"]),
            }
            for row in rows
        }


def _submitted_workspace(
    conn: sqlite3.Connection,
    application_id: int,
    submitted_workspace_id: int,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT id, application_id, status, job_snapshot_json, master_source_sha256, "
        "context_json, resume_document_json, pdf_path, pdf_sha256, apply_pack_json, "
        "created_at, updated_at, submitted_at FROM resume_workspaces "
        "WHERE id = ? AND application_id = ?",
        (submitted_workspace_id, application_id),
    ).fetchone()
    if row is None:
        raise SubmittedWorkspaceRequiredError(
            "application and submitted workspace do not identify the same submission"
        )
    if str(row["status"]) != "submitted":
        raise SubmittedWorkspaceRequiredError(
            "interview research requires an actually submitted workspace"
        )
    return row


def _workspace_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "application_id": int(row["application_id"]),
        "status": str(row["status"]),
        "job_snapshot": _load_object(row["job_snapshot_json"]),
        "master_source_sha256": str(row["master_source_sha256"]),
        "context": _load_object(row["context_json"]),
        "resume_document": _load_object(row["resume_document_json"]),
        "pdf_path": str(row["pdf_path"]) if row["pdf_path"] else None,
        "pdf_sha256": str(row["pdf_sha256"]) if row["pdf_sha256"] else None,
        "apply_pack": _load_object(row["apply_pack_json"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "submitted_at": float(row["submitted_at"]),
    }


def _matching_project_vault(
    conn: sqlite3.Connection,
    frozen_submission: Mapping[str, Any],
) -> list[dict[str, Any]]:
    context = frozen_submission.get("context")
    project_facts = context.get("project_facts") if isinstance(context, Mapping) else None
    project_ids: list[int] = []
    if isinstance(project_facts, list):
        for fact in project_facts:
            if not isinstance(fact, Mapping):
                continue
            raw_project_id = fact.get("project_id")
            if raw_project_id is None:
                continue
            try:
                project_id = int(str(raw_project_id))
            except (TypeError, ValueError):
                continue
            if project_id not in project_ids:
                project_ids.append(project_id)
    if not project_ids:
        return []
    placeholders = ",".join("?" for _ in project_ids)
    rows = conn.execute(
        f"SELECT * FROM project_records WHERE id IN ({placeholders})",
        tuple(project_ids),
    ).fetchall()
    by_id: dict[int, dict[str, Any]] = {}
    for row in rows:
        item = dict(row)
        raw_tags = item.pop("tags_json", "[]")
        try:
            tags = json.loads(raw_tags or "[]")
        except (json.JSONDecodeError, TypeError):
            tags = []
        item["tags"] = tags if isinstance(tags, list) else []
        by_id[int(item["id"])] = item
    return [by_id[project_id] for project_id in project_ids if project_id in by_id]


def _preparation_notes(frozen_submission: Mapping[str, Any]) -> list[dict[str, Any]]:
    apply_pack = frozen_submission.get("apply_pack")
    notes = apply_pack.get("preparation_notes") if isinstance(apply_pack, Mapping) else None
    if not isinstance(notes, list):
        return []
    return [dict(item) for item in notes if isinstance(item, Mapping)]


def _context_updates(
    conn: sqlite3.Connection,
    submitted_workspace_id: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, revision, kind, content_json, created_at "
        "FROM interview_research_context_updates WHERE workspace_id = ? "
        "ORDER BY revision ASC",
        (submitted_workspace_id,),
    ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "revision": int(row["revision"]),
            "kind": str(row["kind"]),
            "content": _load_object(row["content_json"]),
            "created_at": float(row["created_at"]),
        }
        for row in rows
    ]


def _source_assessments(
    conn: sqlite3.Connection,
    submitted_workspace_id: int,
    subject_token: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT evidence_id, evidence_text_sha256, subject_token, context_revision, "
        "assessment_json, contract_version, created_at, updated_at "
        "FROM interview_research_source_links WHERE workspace_id = ? "
        "ORDER BY updated_at DESC, evidence_id ASC",
        (submitted_workspace_id,),
    ).fetchall()
    return [
        {
            "evidence_id": str(row["evidence_id"]),
            "evidence_text_sha256": str(row["evidence_text_sha256"]),
            "assessment": _load_object(row["assessment_json"]),
            "contract_version": int(row["contract_version"]),
            "assessed_for_subject_token": str(row["subject_token"]),
            "context_revision": int(row["context_revision"]),
            "is_current": str(row["subject_token"]) == subject_token,
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }
        for row in rows
    ]


def _attached_sources(
    conn: sqlite3.Connection,
    application_id: int,
    submitted_workspace_id: int,
    current_subject_revision: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT evidence.id AS evidence_id, evidence.title, evidence.final_url, "
        "evidence.provenance, evidence.body_sha256, evidence.fetched_at, "
        "length(evidence.text_content) AS total_chars, link.subject_revision, "
        "link.purpose, link.attached_at "
        "FROM source_subject_evidence link "
        "JOIN source_evidence evidence ON evidence.id = link.evidence_id "
        "WHERE link.subject_kind = 'interview_research' AND link.subject_id = ? "
        "ORDER BY evidence.fetched_at DESC, evidence.id DESC, link.attached_at ASC",
        (_source_subject_id(application_id, submitted_workspace_id),),
    ).fetchall()
    sources: dict[str, dict[str, Any]] = {}
    for row in rows:
        evidence_id = str(row["evidence_id"])
        source = sources.get(evidence_id)
        if source is None:
            source = {
                "evidence_id": evidence_id,
                "title": str(row["title"] or row["final_url"]),
                "url": str(row["final_url"]),
                "provenance": str(row["provenance"]),
                "body_sha256": str(row["body_sha256"]),
                "fetched_at": float(row["fetched_at"]),
                "total_chars": int(row["total_chars"]),
                "subject_revisions": [],
                "purposes": [],
                "attached_at": float(row["attached_at"]),
            }
            sources[evidence_id] = source
        revision = int(row["subject_revision"])
        if revision not in source["subject_revisions"]:
            source["subject_revisions"].append(revision)
        purpose = str(row["purpose"] or "").strip()
        if purpose and purpose not in source["purposes"]:
            source["purposes"].append(purpose)
        source["attached_at"] = max(
            float(source["attached_at"]),
            float(row["attached_at"]),
        )
    for source in sources.values():
        source["subject_revisions"].sort()
        source["attached_to_current_subject"] = (
            current_subject_revision in source["subject_revisions"]
        )
    return list(sources.values())


def _current_material(
    conn: sqlite3.Connection,
    application_id: int,
    submitted_workspace_id: int,
    result_revision: int,
) -> PublishedInterviewMaterial | None:
    if result_revision == 0:
        return None
    row = conn.execute(
        "SELECT subject_token, context_revision, answer_set_json, evidence_ids_json, "
        "contract_version, evidence_revision, created_at "
        "FROM interview_answer_set_revisions "
        "WHERE workspace_id = ? AND result_revision = ? AND application_id = ?",
        (submitted_workspace_id, result_revision, application_id),
    ).fetchone()
    if row is None:
        raise InterviewResearchConflictError("current interview material pointer is invalid")
    evidence_ids = json.loads(row["evidence_ids_json"])
    contract_version = int(row["contract_version"])
    if contract_version < CURRENT_INTERVIEW_MATERIAL_CONTRACT_VERSION:
        return None
    return PublishedInterviewMaterial(
        application_id=application_id,
        submitted_workspace_id=submitted_workspace_id,
        subject_token=str(row["subject_token"]),
        context_revision=int(row["context_revision"]),
        result_revision=result_revision,
        contract_version=contract_version,
        evidence_revision=int(row["evidence_revision"]),
        answer_set=_load_answer_set(
            row["answer_set_json"],
        ),
        used_evidence_ids=[str(value) for value in evidence_ids],
        created_at=float(row["created_at"]),
    )


def _subject_token(**payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _source_subject_id(application_id: int, submitted_workspace_id: int) -> str:
    return f"application:{application_id}:workspace:{submitted_workspace_id}"


def _agent_subject_revision(subject_token: str) -> int:
    """Represent the full content token in runtimes whose revision is an integer."""
    return int(subject_token[:15], 16)


def _used_evidence_ids(answer_set: InterviewAnswerSet) -> list[str]:
    used: list[str] = []
    for answer in answer_set.answers:
        for citation in answer.source_citations:
            if citation.evidence_id not in used:
                used.append(citation.evidence_id)
    return used


def _validate_assessment_questions(
    assessment: InterviewSourceAssessment,
    source_text: str,
) -> None:
    for question in assessment.actual_questions:
        if question not in source_text:
            raise InterviewEvidenceError("source assessment question is not in the source body")


def _load_answer_evidence(
    used_ids: list[str],
    evidence_loader: EvidenceLoader,
) -> dict[str, EvidenceDocument]:
    evidence_by_id: dict[str, EvidenceDocument] = {}
    for evidence_id in used_ids:
        evidence = evidence_loader(evidence_id)
        if evidence is None:
            raise InterviewEvidenceError(f"source evidence {evidence_id!r} does not exist")
        if evidence.evidence_id != evidence_id:
            raise InterviewEvidenceError("source evidence identity mismatch")
        if not evidence.complete:
            raise InterviewEvidenceError(f"source evidence {evidence_id!r} is incomplete")
        if not evidence.attached_to_subject:
            raise InterviewEvidenceError(
                f"source evidence {evidence_id!r} is not attached to this subject"
            )
        evidence_by_id[evidence_id] = evidence
    return evidence_by_id


def _validate_material_evidence(
    answer_set: InterviewAnswerSet,
    evidence_by_id: Mapping[str, EvidenceDocument],
    links: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(evidence_by_id) != set(links):
        missing = sorted(set(evidence_by_id) - set(links))
        raise InterviewEvidenceError(
            "material cites source evidence not assessed for this submission: " + ", ".join(missing)
        )
    for answer in answer_set.answers:
        for citation in answer.source_citations:
            evidence = evidence_by_id[citation.evidence_id]
            link = links[citation.evidence_id]
            if int(link.get("contract_version") or 1) < CURRENT_INTERVIEW_EVIDENCE_CONTRACT_VERSION:
                raise InterviewEvidenceError(
                    "source assessment predates the current evidence contract; "
                    "read and assess it again"
                )
            text_hash = hashlib.sha256(evidence.text.encode("utf-8")).hexdigest()
            if text_hash != link["text_sha256"]:
                raise InterviewEvidenceError(
                    f"source evidence {citation.evidence_id!r} changed after assessment"
                )
            if citation.quote not in evidence.text:
                raise InterviewEvidenceError(
                    f"citation quote is not in source {citation.evidence_id!r}"
                )
            assessment = InterviewSourceAssessment.model_validate(link["assessment"])
            if assessment.decision != "accepted":
                raise InterviewEvidenceError(
                    "source is not accepted for this application's interview question pool"
                )
            if citation.quote not in assessment.actual_questions:
                raise InterviewEvidenceError(
                    "citation is not an actual question preserved from the source"
                )


def _validate_grounding(
    answer_set: InterviewAnswerSet,
    subject: InterviewResearchSubject,
) -> None:
    frozen = subject.frozen_submission
    job_snapshot = frozen.get("job_snapshot")
    frozen_context = frozen.get("context")
    context_job = frozen_context.get("job") if isinstance(frozen_context, Mapping) else None
    job_texts = _nonblank_strings(
        job_snapshot.get("raw_text") if isinstance(job_snapshot, Mapping) else None,
        context_job.get("jd_text") if isinstance(context_job, Mapping) else None,
    )
    resume_strings = tuple(_resume_visible_strings(frozen.get("resume_document")))
    projects_by_id = {
        str(item["id"]): item for item in subject.project_vault if item.get("id") is not None
    }
    notes_by_index = {str(index): note for index, note in enumerate(subject.preparation_notes)}
    for answer in answer_set.answers:
        for grounding in answer.grounding:
            if grounding.kind == "job_description":
                _require_quote_fragment(grounding.quote, job_texts, "frozen job description")
            elif grounding.kind == "submitted_resume":
                _require_complete_field_quote(grounding.quote, resume_strings, "submitted resume")
            elif grounding.kind == "project_vault":
                project = projects_by_id.get(str(grounding.reference))
                if project is None:
                    raise InterviewEvidenceError("material cites an unrelated Project Vault record")
                _require_complete_field_quote(
                    grounding.quote,
                    tuple(_project_grounding_strings(project)),
                    f"Project Vault record {grounding.reference}",
                )
            elif grounding.kind == "preparation_note":
                note = notes_by_index.get(str(grounding.reference))
                if note is None:
                    raise InterviewEvidenceError("material cites an unknown preparation note index")
                _require_complete_field_quote(
                    grounding.quote,
                    tuple(_string_leaves(note)),
                    f"preparation note {grounding.reference}",
                )


def _nonblank_strings(*values: Any) -> tuple[str, ...]:
    return tuple(value for value in values if isinstance(value, str) and value)


def _string_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        leaves: list[str] = []
        for item in value.values():
            leaves.extend(_string_leaves(item))
        return leaves
    if isinstance(value, list | tuple):
        leaves = []
        for item in value:
            leaves.extend(_string_leaves(item))
        return leaves
    return []


def _resume_visible_strings(value: Any) -> list[str]:
    """Return candidate content while excluding layout and section labels.

    The submitted ResumeDocument is structured, so traversing arbitrary key names
    is unnecessary and would make a heading such as ``项目`` look like candidate
    evidence. RichText spans are joined before exposure so emphasis cannot split
    a negation from the rest of its visible sentence.
    """
    if not isinstance(value, Mapping):
        return []
    strings: list[str] = []
    header = value.get("header")
    if isinstance(header, Mapping):
        strings.extend(_resume_text_value(header.get("name")))
        strings.extend(_resume_text_value(header.get("lines")))

    sections = value.get("sections")
    if not isinstance(sections, list):
        return strings
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        entries = section.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                rows = entry.get("rows")
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, Mapping):
                            continue
                        strings.extend(_resume_text_value(row.get("left")))
                        strings.extend(_resume_text_value(row.get("right")))
                blocks = entry.get("blocks")
                if isinstance(blocks, list):
                    for block in blocks:
                        if isinstance(block, Mapping):
                            strings.extend(_resume_text_value(block.get("content")))
                # Read historical pre-row workspaces without treating their kind or
                # layout flags as evidence.
                strings.extend(_resume_text_value(entry.get("heading")))
                strings.extend(_resume_text_value(entry.get("aside")))
        # Historical flat sections stored visible copy directly on the section.
        strings.extend(_resume_text_value(section.get("body")))
        strings.extend(_resume_text_value(section.get("content")))
    return strings


def _resume_text_value(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        plain_text = _rich_text_plain_text(value)
        return [plain_text] if plain_text and plain_text.strip() else []
    if isinstance(value, list | tuple):
        strings: list[str] = []
        for item in value:
            strings.extend(_resume_text_value(item))
        return strings
    return []


def _rich_text_plain_text(value: Mapping[str, Any]) -> str | None:
    """Join one validated Resume RichText value before applying evidence checks.

    Keeping the full visible line together prevents emphasis span boundaries from
    turning a negated sentence into independently citable positive fragments.
    """
    spans = value.get("spans")
    if not isinstance(spans, list) or not spans:
        return None
    pieces: list[str] = []
    for span in spans:
        if not isinstance(span, Mapping) or not isinstance(span.get("text"), str):
            return None
        pieces.append(str(span["text"]))
    return "".join(pieces)


def _project_grounding_strings(project: Mapping[str, Any]) -> list[str]:
    """Exclude public wording research and storage metadata from candidate evidence."""
    leaves: list[str] = []
    for field in _PROJECT_GROUNDING_FIELDS:
        leaves.extend(_string_leaves(project.get(field)))
    return leaves


def _require_quote_fragment(
    quote: str,
    source_strings: tuple[str, ...],
    source_label: str,
) -> None:
    if any(quote in source for source in source_strings):
        return
    raise InterviewEvidenceError(f"grounding quote is not present in the {source_label}")


def _require_complete_field_quote(
    quote: str,
    source_strings: tuple[str, ...],
    source_label: str,
) -> None:
    if any(quote == source.strip() for source in source_strings if source.strip()):
        return
    raise InterviewEvidenceError(
        f"grounding quote is not present as a complete field in the {source_label}"
    )


def _load_object(raw: str | bytes | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError) as exc:
        raise InterviewResearchConflictError("stored interview research JSON is invalid") from exc
    if not isinstance(value, dict):
        raise InterviewResearchConflictError("stored interview research JSON is not an object")
    return value


def _load_answer_set(raw: str | bytes | None) -> InterviewAnswerSet:
    return InterviewAnswerSet.model_validate(_load_object(raw))


def _object_json(value: Mapping[str, Any], label: str) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc


def _model_json(value: Any) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "CURRENT_INTERVIEW_EVIDENCE_CONTRACT_VERSION",
    "CURRENT_INTERVIEW_MATERIAL_CONTRACT_VERSION",
    "EvidenceLoader",
    "InterviewEvidenceError",
    "InterviewResearchConflictError",
    "InterviewResearchError",
    "InterviewResearchRepository",
    "SubmittedWorkspaceRequiredError",
]
