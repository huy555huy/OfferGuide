"""Domain models for source-grounded interview questions and answers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SourceAssessmentDecision = Literal["accepted", "rejected"]
SourceKind = str
GroundingKind = Literal[
    "job_description",
    "submitted_resume",
    "project_vault",
    "preparation_note",
]


class SourceCitation(BaseModel):
    """An exact quote anchoring a material block to a fetched source."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    quote: str = Field(
        description=(
            "Exact original question preserved in the accepted source assessment. "
            "This remains verbatim even when the displayed question minimally "
            "normalizes a company or role reference for the current target."
        )
    )

    @field_validator("evidence_id", "quote")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source citation fields must not be blank")
        return cleaned


class GroundingReference(BaseModel):
    """A verbatim quote from frozen candidate/application evidence."""

    model_config = ConfigDict(extra="forbid")

    kind: GroundingKind
    reference: str | None = Field(
        default=None,
        description=(
            "Project Vault record id or zero-based preparation-note index. "
            "Omit for the frozen JD and submitted resume."
        ),
    )
    quote: str = Field(
        description=(
            "Exact character-for-character quote copied from a string value in the "
            "identified frozen source; never a paraphrase or model-authored summary."
        )
    )

    @field_validator("reference")
    @classmethod
    def _normalize_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("quote")
    @classmethod
    def _quote_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("grounding quote must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _identified_grounding_requires_reference(self) -> GroundingReference:
        if self.kind in {"project_vault", "preparation_note"} and self.reference is None:
            raise ValueError(f"{self.kind} grounding requires a reference")
        if self.kind in {"job_description", "submitted_resume"} and self.reference is not None:
            raise ValueError(f"{self.kind} grounding does not accept a reference")
        return self


class InterviewQuestionAnswer(BaseModel):
    """One source-grounded question and the candidate-specific answer to it."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        description=(
            "User-facing form of one or more cited source questions. It may restore an "
            "omitted subject or minimally normalize a source company/role reference, "
            "but must not add a question absent from the cited accepted sources."
        )
    )
    answer: str
    source_citations: list[SourceCitation] = Field(default_factory=list)
    grounding: list[GroundingReference] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _discard_legacy_answer_relationship(cls, value: Any) -> Any:
        """Read material published before answer-level relationship buckets were removed."""

        if isinstance(value, Mapping) and "relationship" in value:
            value = dict(value)
            value.pop("relationship", None)
        return value

    @field_validator("question", "answer")
    @classmethod
    def _visible_text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("interview question and answer must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _question_must_cite_a_real_source(self) -> InterviewQuestionAnswer:
        if not self.source_citations:
            raise ValueError("every interview question requires a source citation")
        return self


class InterviewAnswerSet(BaseModel):
    """The only model-authored result accepted by the domain publish tool."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "not_found"] = "answered"
    answers: list[InterviewQuestionAnswer] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_answers(self) -> InterviewAnswerSet:
        if self.status == "answered" and not self.answers:
            raise ValueError("an answered result requires at least one question and answer")
        if self.status == "not_found" and self.answers:
            raise ValueError("a not-found result cannot contain generated questions")
        return self


class InterviewSourceAssessment(BaseModel):
    """Whether one fetched page contains actual questions from a real interview."""

    model_config = ConfigDict(extra="forbid")

    decision: SourceAssessmentDecision
    source_kind: SourceKind = "other"
    rationale: str
    actual_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_relationship(cls, value: Any) -> Any:
        """Read assessments written before all similar-role sources shared one pool."""

        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        legacy = normalized.pop("relationship", None)
        if "decision" not in normalized and legacy is not None:
            normalized["decision"] = (
                "accepted" if legacy in {"direct", "adjacent", "accepted"} else legacy
            )
        return normalized

    @field_validator("source_kind")
    @classmethod
    def _normalize_source_kind(cls, value: str) -> str:
        return value.strip() or "other"

    @field_validator("rationale")
    @classmethod
    def _rationale_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source assessment rationale must not be blank")
        return cleaned

    @field_validator("actual_questions")
    @classmethod
    def _lists_must_not_contain_blanks(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source assessment lists must not contain blank items")
        return cleaned

    @model_validator(mode="after")
    def _accepted_sources_must_have_questions(
        self,
    ) -> InterviewSourceAssessment:
        if self.decision == "accepted" and not self.actual_questions:
            raise ValueError("an accepted interview source must contain actual questions")
        return self


class EvidenceDocument(BaseModel):
    """Minimal adapter result required for deterministic domain validation."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    url: str
    title: str
    text: str
    fetched_at: float | None = None
    complete: bool = True
    attached_to_subject: bool = False

    @field_validator("evidence_id", "url", "title", "text")
    @classmethod
    def _document_fields_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("evidence document fields must not be blank")
        return cleaned


class PublishedInterviewMaterial(BaseModel):
    """One immutable material revision; the repository owns current selection."""

    model_config = ConfigDict(extra="forbid")

    application_id: int
    submitted_workspace_id: int
    subject_token: str
    context_revision: int
    result_revision: int
    contract_version: int = Field(default=1, ge=1)
    evidence_revision: int = Field(default=0, ge=0)
    answer_set: InterviewAnswerSet
    used_evidence_ids: list[str]
    created_at: float


class InterviewResearchSubject(BaseModel):
    """Complete context handed to the domain Agent without character slicing."""

    model_config = ConfigDict(extra="forbid")

    application_id: int
    submitted_workspace_id: int
    subject_token: str
    agent_subject_revision: int
    context_revision: int
    result_revision: int
    evidence_revision: int = Field(default=0, ge=0)
    frozen_submission: dict[str, Any]
    project_vault: list[dict[str, Any]]
    preparation_notes: list[dict[str, Any]]
    context_updates: list[dict[str, Any]]
    source_assessments: list[dict[str, Any]]
    attached_sources: list[dict[str, Any]] = Field(default_factory=list)
    current_material: PublishedInterviewMaterial | None = None
    current_material_is_stale: bool = False


__all__ = [
    "EvidenceDocument",
    "GroundingKind",
    "GroundingReference",
    "InterviewAnswerSet",
    "InterviewQuestionAnswer",
    "InterviewResearchSubject",
    "InterviewSourceAssessment",
    "PublishedInterviewMaterial",
    "SourceAssessmentDecision",
    "SourceCitation",
    "SourceKind",
]
