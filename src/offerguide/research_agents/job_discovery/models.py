"""Domain contracts for model-driven job discovery.

These models deliberately contain no scores, probability buckets, keyword
matrices, or user-specific career taxonomy.  They describe the complete
evidence and the one ordered decision surface that the user can act on.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobSearchContext(BaseModel):
    """The user-visible, revisioned meaning of the current job search."""

    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=1)
    intent: str
    hard_constraints: list[str] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)

    @field_validator("intent")
    @classmethod
    def _intent_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("job search intent must not be blank")
        return value

    @field_validator("hard_constraints", "feedback")
    @classmethod
    def _semantic_lines_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value).strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("job search context lines must not be blank")
        return cleaned


class CandidateEvidenceDocument(BaseModel):
    """One complete, confirmed candidate document supplied to the agent."""

    model_config = ConfigDict(extra="forbid")

    reference: str
    kind: str
    title: str
    text: str

    @field_validator("reference", "kind", "title", "text")
    @classmethod
    def _candidate_fields_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate evidence fields must not be blank")
        return value


class CandidateEvidence(BaseModel):
    """All confirmed resume and project evidence available for this run."""

    model_config = ConfigDict(extra="forbid")

    documents: list[CandidateEvidenceDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references_must_be_unique(self) -> CandidateEvidence:
        references = [document.reference for document in self.documents]
        if len(references) != len(set(references)):
            raise ValueError("candidate evidence references must be unique")
        return self


JobSourceStatus = Literal["open", "closed", "unknown"]
JobEvidenceKind = Literal["web", "platform_adapter"]


def platform_job_reference(
    *,
    source_evidence_id: int,
    source_name: str,
    source_job_id: str | None,
    canonical_url: str,
) -> str:
    """Return the opaque selector for one deterministic adapter result."""

    payload = json.dumps(
        {
            "source_evidence_id": source_evidence_id,
            "source_name": source_name,
            "source_job_id": source_job_id,
            "canonical_url": canonical_url,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "platform-job:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JobPostingEvidence(BaseModel):
    """A complete JD anchored to one successfully read source page."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = Field(default=None, ge=1)
    job_id: int | None = Field(default=None, ge=1)
    evidence_kind: JobEvidenceKind = "web"
    source_evidence_id: int = Field(ge=1)
    source_name: str
    source_job_id: str | None = None
    canonical_url: str
    company: str
    title: str
    location: str | None = None
    recruitment_type: str | None = None
    page_time_information: list[str] = Field(default_factory=list)
    jd_text: str
    source_status: JobSourceStatus = "unknown"
    checked_at: float | None = None
    last_seen_at: float | None = None
    content_sha256: str | None = None

    @field_validator("source_name", "canonical_url", "company", "title", "jd_text")
    @classmethod
    def _required_fields_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("job evidence fields must not be blank")
        return value

    @field_validator("source_job_id", "location", "recruitment_type")
    @classmethod
    def _optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("page_time_information")
    @classmethod
    def _time_information_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value).strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("page time information must not contain blank entries")
        return cleaned


class SelectionGroundingQuote(BaseModel):
    """An exact candidate/search-context quote supporting a selection reason."""

    model_config = ConfigDict(extra="forbid")

    reference: str
    quote: str

    @field_validator("reference", "quote")
    @classmethod
    def _grounding_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("selection grounding reference and quote must not be blank")
        return value

    @field_validator("quote")
    @classmethod
    def _quote_must_identify_evidence(cls, value: str) -> str:
        if sum(character.isalnum() for character in value) < 2:
            raise ValueError("selection grounding quote is too weak to identify evidence")
        return value


class JobSelectionItem(BaseModel):
    """The agent's current semantic judgment about one verified posting."""

    model_config = ConfigDict(extra="forbid")

    job_evidence_id: int = Field(ge=1)
    job_evidence: JobPostingEvidence | None = Field(
        default=None,
        description=(
            "The immutable evidence snapshot resolved by the repository at publication. "
            "Agent draft arguments omit this field."
        ),
    )
    why_worth_attention: str
    grounding_quotes: list[SelectionGroundingQuote] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)

    @field_validator("why_worth_attention")
    @classmethod
    def _reason_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("selection reason must not be blank")
        return value

    @field_validator("concerns", "unknowns")
    @classmethod
    def _judgment_lines_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value).strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("selection concerns and unknowns must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _snapshot_identity_must_match(self) -> JobSelectionItem:
        if self.job_evidence is not None and self.job_evidence.id != self.job_evidence_id:
            raise ValueError("selection evidence snapshot does not match job_evidence_id")
        quote_keys = [(item.reference, item.quote) for item in self.grounding_quotes]
        if len(quote_keys) != len(set(quote_keys)):
            raise ValueError("selection grounding quotes must be unique")
        return self


class JobSelectionSet(BaseModel):
    """The sole current ordered shortlist for one search-context revision."""

    model_config = ConfigDict(extra="forbid")

    context_revision: int = Field(ge=1)
    result_revision: int = Field(ge=1)
    items: list[JobSelectionItem] = Field(default_factory=list)
    coverage_summary: str
    evidence_gaps: list[str] = Field(default_factory=list)
    published_at: float | None = None

    @field_validator("coverage_summary")
    @classmethod
    def _coverage_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("selection coverage summary must not be blank")
        return value

    @field_validator("evidence_gaps")
    @classmethod
    def _gaps_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [str(value).strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("selection evidence gaps must not be blank")
        return cleaned

    @model_validator(mode="after")
    def _selection_must_be_coherent(self) -> JobSelectionSet:
        job_ids = [item.job_evidence_id for item in self.items]
        if len(job_ids) != len(set(job_ids)):
            raise ValueError("a job can appear only once in the current selection")
        if any(item.job_evidence is None for item in self.items):
            raise ValueError("published selection items require immutable evidence snapshots")
        if any(not item.grounding_quotes for item in self.items):
            raise ValueError("published selection items require evidence grounding")
        if not self.items and not self.evidence_gaps:
            raise ValueError("an empty selection must explain the remaining evidence gap")
        return self


class JobEvidenceCatalogItem(BaseModel):
    """Lightweight history entry; the complete JD is read only when needed."""

    model_config = ConfigDict(extra="forbid")

    job_evidence_id: int = Field(ge=1)
    job_id: int = Field(ge=1)
    source_name: str
    source_job_id: str | None = None
    canonical_url: str
    company: str
    title: str
    location: str | None = None
    recruitment_type: str | None = None
    source_status: JobSourceStatus
    checked_at: float


class JobEvidenceCatalogPage(BaseModel):
    """One bounded page of recorded jobs available to the Agent."""

    model_config = ConfigDict(extra="forbid")

    offset: int = Field(ge=0)
    limit: int = Field(ge=1)
    total: int = Field(ge=0)
    items: list[JobEvidenceCatalogItem] = Field(default_factory=list)
    next_offset: int | None = Field(default=None, ge=0)


class JobDiscoverySnapshot(BaseModel):
    """Complete authoritative input loaded at the start of an agent run."""

    model_config = ConfigDict(extra="forbid")

    search_context: JobSearchContext
    candidate_evidence: CandidateEvidence
    recorded_job_catalog: JobEvidenceCatalogPage
    current_selection: JobSelectionSet | None = None
    current_result_revision: int = Field(ge=0)
