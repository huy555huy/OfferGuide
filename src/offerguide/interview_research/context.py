"""Model-facing context projections for one submitted application.

Persistence records are intentionally richer than a research model needs.  This
module is the boundary between those records and model input: it emits explicit
allowlists and replaces database identities with references that live for one run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .models import InterviewResearchSubject, InterviewSourceAssessment

_SOURCE_TYPE_LABELS = {
    "firsthand_interview": "当事人的面试经历",
    "official_recruiting": "公司官方招聘信息",
    "official_company": "公司官方资料",
    "job_posting": "岗位页面",
    "personal_commentary": "个人观点",
    "question_bank": "题库或面试指南",
    "repost": "转载内容",
    "other": "来源类型尚未确认",
}
_PROVENANCE_LABELS = {
    "web": "公开网页",
    "user_provided": "用户提供的正文",
}
_PROJECT_FIELDS = (
    ("title", "项目名称"),
    ("mainstream_direction", "项目方向"),
    ("typical_problem", "要解决的问题"),
    ("project_task", "项目任务"),
    ("my_work", "候选人的实际工作"),
    ("method_route", "方法路线"),
    ("contribution_detail", "贡献边界"),
    ("key_difficulties", "关键困难"),
    ("resolution_process", "解决过程"),
    ("project_outputs", "项目产出"),
    ("evidence", "可核验材料"),
    ("askable_points", "可追问内容"),
    ("expression_boundary", "表达边界"),
    ("do_not_claim", "不得声称"),
    ("tags", "标签"),
)


@dataclass(slots=True)
class InterviewRunReferences:
    """Bidirectional, in-memory references authorized for exactly one run."""

    _source_ref_to_id: dict[str, int] = field(default_factory=dict)
    _source_id_to_ref: dict[int, str] = field(default_factory=dict)
    _context_ref_to_key: dict[str, tuple[str, str]] = field(default_factory=dict)
    _context_key_to_ref: dict[tuple[str, str], str] = field(default_factory=dict)
    _next_context_number: dict[str, int] = field(default_factory=dict)

    @classmethod
    def for_subject(cls, subject: InterviewResearchSubject) -> InterviewRunReferences:
        refs = cls()
        allowed_source_ids: set[int] = set()
        for item in subject.attached_sources:
            if item.get("provenance") not in {"web", "user_provided"}:
                continue
            evidence_id = _positive_int(item.get("evidence_id"))
            if evidence_id is not None:
                allowed_source_ids.add(evidence_id)
                refs.source_ref(evidence_id)
        for item in subject.source_assessments:
            evidence_id = _positive_int(item.get("evidence_id"))
            if evidence_id is not None and evidence_id in allowed_source_ids:
                refs.source_ref(evidence_id)
        return refs

    def source_ref(self, evidence_id: int) -> str:
        if evidence_id < 1:
            raise ValueError("source identity must be positive")
        existing = self._source_id_to_ref.get(evidence_id)
        if existing is not None:
            return existing
        reference = f"source-{len(self._source_id_to_ref) + 1}"
        self._source_id_to_ref[evidence_id] = reference
        self._source_ref_to_id[reference] = evidence_id
        return reference

    def resolve_source(self, reference: str) -> int:
        try:
            return self._source_ref_to_id[reference.strip()]
        except KeyError as exc:
            raise ValueError("unknown or unavailable source reference") from exc

    def existing_source_ref(self, evidence_id: int) -> str | None:
        return self._source_id_to_ref.get(evidence_id)

    def context_ref(self, kind: str, real_reference: str, *, prefix: str) -> str:
        key = (kind, str(real_reference))
        existing = self._context_key_to_ref.get(key)
        if existing is not None:
            return existing
        number = self._next_context_number.get(prefix, 0) + 1
        self._next_context_number[prefix] = number
        reference = f"{prefix}-{number}"
        self._context_key_to_ref[key] = reference
        self._context_ref_to_key[reference] = key
        return reference

    def resolve_context(self, kind: str, reference: str) -> str:
        try:
            stored_kind, real_reference = self._context_ref_to_key[reference.strip()]
        except KeyError as exc:
            raise ValueError("unknown or unavailable context reference") from exc
        if stored_kind != kind:
            raise ValueError("context reference does not match its evidence kind")
        return real_reference

    def existing_context_ref(self, kind: str, real_reference: str) -> str | None:
        return self._context_key_to_ref.get((kind, str(real_reference)))

    @property
    def visible_references(self) -> frozenset[str]:
        return frozenset((*self._source_ref_to_id, *self._context_ref_to_key))


def build_research_packet(
    subject: InterviewResearchSubject,
    references: InterviewRunReferences,
    *,
    current_evidence_contract_version: int,
) -> dict[str, Any]:
    """Return only facts the research/writing model can use productively."""

    return {
        "target_job": job_for_model(subject.frozen_submission),
        "saved_sources": sources_for_model(
            subject,
            references,
            current_evidence_contract_version=current_evidence_contract_version,
        ),
    }


def build_writing_packet(
    subject: InterviewResearchSubject,
    references: InterviewRunReferences,
    *,
    current_evidence_contract_version: int,
) -> dict[str, Any]:
    """Fresh writing context with no tool handles, failures, history, or old draft."""

    return {
        "target_job": job_for_model(subject.frozen_submission),
        "submitted_resume": resume_for_model(subject.frozen_submission),
        "claim_boundaries": preparation_notes_for_model(subject, references),
        "candidate_project_facts": projects_for_model(subject, references),
        "accepted_interview_experiences": accepted_sources_for_writing(
            subject,
            references,
            current_evidence_contract_version=current_evidence_contract_version,
        ),
    }


def job_for_model(frozen_submission: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = frozen_submission.get("job_snapshot")
    context = frozen_submission.get("context")
    context_job = context.get("job") if isinstance(context, Mapping) else None
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    context_job = context_job if isinstance(context_job, Mapping) else {}
    return _drop_empty(
        {
            "company": snapshot.get("company") or context_job.get("company"),
            "role": snapshot.get("title") or context_job.get("title"),
            "location": snapshot.get("location") or context_job.get("location"),
            "description": (
                snapshot.get("raw_text") or snapshot.get("jd_text") or context_job.get("jd_text")
            ),
            "public_url": _public_http_url(snapshot.get("url") or context_job.get("source_url")),
        }
    )


def resume_for_model(frozen_submission: Mapping[str, Any]) -> dict[str, Any]:
    """Project submitted resume content while omitting its contact/header and layout."""

    document = frozen_submission.get("resume_document")
    if not isinstance(document, Mapping):
        return {"sections": []}
    projected_sections: list[dict[str, Any]] = []
    sections = document.get("sections")
    if not isinstance(sections, list):
        return {"sections": []}
    for raw_section in sections:
        if not isinstance(raw_section, Mapping):
            continue
        section: dict[str, Any] = {
            "title": _plain_text(raw_section.get("title")),
            "entries": [],
        }
        entries = raw_section.get("entries")
        if isinstance(entries, list):
            for raw_entry in entries:
                if not isinstance(raw_entry, Mapping):
                    continue
                entry: dict[str, Any] = {"headings": [], "content": []}
                rows = raw_entry.get("rows")
                if isinstance(rows, list):
                    for raw_row in rows:
                        if not isinstance(raw_row, Mapping):
                            continue
                        left = _plain_text(raw_row.get("left"))
                        right = _plain_text(raw_row.get("right"))
                        if left or right:
                            entry["headings"].append(
                                _drop_empty(
                                    {
                                        "main": left,
                                        "supporting": right,
                                    }
                                )
                            )
                blocks = raw_entry.get("blocks")
                if isinstance(blocks, list):
                    for raw_block in blocks:
                        if not isinstance(raw_block, Mapping):
                            continue
                        text = _plain_text(raw_block.get("content"))
                        if text:
                            entry["content"].append(
                                _drop_empty(
                                    {
                                        "format": raw_block.get("kind"),
                                        "text": text,
                                    }
                                )
                            )
                for legacy_key in ("heading", "aside"):
                    legacy = _plain_text(raw_entry.get(legacy_key))
                    if legacy:
                        entry["headings"].append({"main": legacy})
                if entry["headings"] or entry["content"]:
                    section["entries"].append(entry)
        legacy_content = _plain_text(raw_section.get("body") or raw_section.get("content"))
        if legacy_content:
            section["entries"].append(
                {
                    "headings": [],
                    "content": [{"text": legacy_content}],
                }
            )
        if section["title"] or section["entries"]:
            projected_sections.append(section)
    return {"sections": projected_sections}


def preparation_notes_for_model(
    subject: InterviewResearchSubject,
    references: InterviewRunReferences | None = None,
) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for index, note in enumerate(subject.preparation_notes):
        if not isinstance(note, Mapping):
            continue
        projected = _drop_empty(
            {
                "claim_on_submitted_resume": note.get("claim"),
                "preparation_needed": note.get("note"),
            }
        )
        if not projected:
            continue
        if references is not None:
            projected = {
                "reference": references.context_ref("preparation_note", str(index), prefix="note"),
                **projected,
            }
        notes.append(projected)
    return notes


def projects_for_model(
    subject: InterviewResearchSubject,
    references: InterviewRunReferences | None = None,
) -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    for project in subject.project_vault:
        if not isinstance(project, Mapping) or project.get("id") is None:
            continue
        facts: list[dict[str, Any]] = []
        for raw_name, label in _PROJECT_FIELDS:
            value = project.get(raw_name)
            if isinstance(value, str) and value.strip():
                facts.append({"field": label, "value": value.strip()})
            elif raw_name == "tags" and isinstance(value, list):
                clean_tags = [str(item).strip() for item in value if str(item).strip()]
                if clean_tags:
                    facts.append({"field": label, "value": clean_tags})
        if not facts:
            continue
        item: dict[str, Any] = {"facts_and_boundaries": facts}
        if references is not None:
            item["reference"] = references.context_ref(
                "project_vault", str(project["id"]), prefix="project"
            )
        projects.append(item)
    return projects


def sources_for_model(
    subject: InterviewResearchSubject,
    references: InterviewRunReferences,
    *,
    current_evidence_contract_version: int,
) -> list[dict[str, Any]]:
    current_assessments: dict[str, Mapping[str, Any]] = {}
    for item in subject.source_assessments:
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id and item.get("is_current"):
            current_assessments[evidence_id] = item

    sources: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in subject.attached_sources:
        if source.get("provenance") not in {"web", "user_provided"}:
            continue
        evidence_id = _positive_int(source.get("evidence_id"))
        if evidence_id is None or evidence_id in seen:
            continue
        seen.add(evidence_id)
        assessment_row = current_assessments.get(str(evidence_id))
        item: dict[str, Any] = _drop_empty(
            {
                "source_ref": references.source_ref(evidence_id),
                "title": source.get("title"),
                "public_url": _public_http_url(source.get("url")),
                "obtained_via": _PROVENANCE_LABELS.get(
                    str(source.get("provenance") or ""), "已保存来源"
                ),
                "availability": (
                    "可直接读取"
                    if source.get("attached_to_current_subject")
                    else "较早研究保存；使用前先重新挂接并完整读取"
                ),
            }
        )
        if (
            isinstance(assessment_row, Mapping)
            and int(assessment_row.get("contract_version") or 1)
            >= current_evidence_contract_version
            and isinstance(assessment_row.get("assessment"), Mapping)
        ):
            assessment = InterviewSourceAssessment.model_validate(assessment_row["assessment"])
            if assessment.decision == "accepted":
                item["accepted_interview_experience"] = {
                    "source_type": assessment.source_kind,
                    "rationale": assessment.rationale,
                    "actual_questions": list(assessment.actual_questions),
                    "next_action": (
                        "Read the immutable saved source completely, then use the "
                        "answer writer. Do not reassess it unless its evidence body changes."
                    ),
                }
            else:
                item["assessment"] = _drop_empty(
                    {
                        "status": "rejected",
                        "source_type": assessment.source_kind,
                        "rationale": assessment.rationale,
                        "actual_questions": list(assessment.actual_questions),
                    }
                )
        else:
            item["assessment_status"] = "使用前需要完整读取并重新评估"
        sources.append(item)
    return sources


def accepted_sources_for_writing(
    subject: InterviewResearchSubject,
    references: InterviewRunReferences,
    *,
    current_evidence_contract_version: int,
) -> list[dict[str, Any]]:
    attached = {
        str(item.get("evidence_id") or ""): item
        for item in subject.attached_sources
        if str(item.get("evidence_id") or "") and item.get("provenance") in {"web", "user_provided"}
    }
    sources: list[dict[str, Any]] = []
    for item in subject.source_assessments:
        evidence_id = str(item.get("evidence_id") or "")
        raw_assessment = item.get("assessment")
        source = attached.get(evidence_id)
        if (
            not evidence_id
            or item.get("is_current") is not True
            or int(item.get("contract_version") or 1) < current_evidence_contract_version
            or not isinstance(raw_assessment, Mapping)
            or not isinstance(source, Mapping)
        ):
            continue
        assessment = InterviewSourceAssessment.model_validate(raw_assessment)
        if assessment.decision != "accepted" or not assessment.actual_questions:
            continue
        numeric_id = _positive_int(evidence_id)
        if numeric_id is None:
            continue
        sources.append(
            _drop_empty(
                {
                    "source_ref": references.source_ref(numeric_id),
                    "title": source.get("title"),
                    "public_url": _public_http_url(source.get("url")),
                    "source_type": assessment.source_kind,
                    "actual_questions": list(assessment.actual_questions),
                }
            )
        )
    return sources


def source_type_label(value: str) -> str:
    cleaned = value.strip()
    return _SOURCE_TYPE_LABELS.get(cleaned, cleaned or "来源类型尚未确认")


def _plain_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Mapping):
        return ""
    spans = value.get("spans")
    if not isinstance(spans, list):
        return ""
    return "".join(
        str(span.get("text") or "") for span in spans if isinstance(span, Mapping)
    ).strip()


def _drop_empty(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item is not None and item != "" and item != [] and item != {}
    }


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _public_http_url(value: Any) -> str | None:
    url = str(value or "").strip()
    return url if url.startswith(("http://", "https://")) else None


__all__ = [
    "InterviewRunReferences",
    "build_research_packet",
    "build_writing_packet",
    "job_for_model",
    "preparation_notes_for_model",
    "projects_for_model",
    "resume_for_model",
    "source_type_label",
]
