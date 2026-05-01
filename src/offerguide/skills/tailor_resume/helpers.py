"""Pydantic schema for tailor_resume SKILL — anti-fabrication enforced.

This SKILL takes a master resume + JD and produces a tailored markdown
version. The key design constraint is **what it MUST NOT do**:

1. Never invent experiences (internships, projects, awards) absent from
   the master resume
2. Never modify externally-verifiable hard facts (school, degree,
   graduation date, internship duration, paper titles)
3. Never inflate numbers (AUC 0.04 lift cannot become 5%)
4. Never claim tech stack the master resume doesn't mention

Each accepted change is logged in ``change_log`` so the user can audit
the diff. Refused changes — i.e. things the LLM was tempted to do but
caught itself on — go into ``cannot_fake_warnings`` so the GEPA reflection
model can train against fabrication temptation.

Pydantic ``extra='forbid'`` plus length caps on bullet text keep the LLM
from sneaking unstructured "creative additions" past us.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChangeKind = Literal[
    "reword",
    "reorder",
    "emphasize",
    "drop",
    "ats_keyword_add",
]


class ChangeLogEntry(BaseModel):
    """One round-trippable diff against the master resume."""
    model_config = ConfigDict(extra="forbid")

    section: str
    """Which section of the resume this change touches, e.g.
    '项目经历 - RemeDi' or '技能 - 编程语言'."""

    kind: ChangeKind
    before: str
    """The original phrasing / order in master_resume.
    Trimmed to <= 200 chars; the audit needs to be readable, not exhaustive."""

    after: str
    """The new phrasing / new position. Same length cap."""

    rationale: str
    """One-sentence justification — must reference JD line, profile field,
    or ATS keyword. Catches the LLM if it tries 'looks better' style
    rationales (the SKILL body explicitly bans those)."""


class FitEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before: float = Field(ge=0.0, le=1.0)
    """Estimated probability of landing an interview with the master
    resume, given this JD."""

    after: float = Field(ge=0.0, le=1.0)
    """Same, but with the tailored version."""

    rationale: str
    """Where the lift comes from (or doesn't, if before == after)."""


class TailorResumeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    role_focus: str
    """One-sentence positioning of which role this tailored resume targets."""

    tailored_markdown: str
    """The complete, paste-ready tailored resume in markdown.
    The SKILL body caps it to ~1.5 pages (~600 chars per section)."""

    change_log: list[ChangeLogEntry] = Field(default_factory=list)
    """Every change relative to master_resume. Drives the diff UI."""

    ats_keywords_used: list[str] = Field(default_factory=list)
    """JD keywords actually present in tailored_markdown — every one
    must be substring-verifiable in tailored_markdown for the change
    to count."""

    ats_keywords_missing: list[str] = Field(default_factory=list)
    """JD keywords the LLM wanted to add but couldn't ground in
    master_resume — surfaced as gaps for the user to address (or
    skip if they're not authentic)."""

    cannot_fake_warnings: list[str] = Field(default_factory=list)
    """Self-audit: things the LLM caught itself almost doing —
    'tempted to add a 字节实习 but master_resume only mentions 法至,
    refused'. Empty list is good; non-empty is honest. The GEPA
    metric uses presence-of-warnings as a positive signal (the LLM
    is being honest about edge cases)."""

    fit_estimate: FitEstimate
    """Quantified before/after improvement estimate with reasoning."""

    suggested_filename: str
    """Format: <姓名>_<公司>_<岗位>_<YYYY-MM-DD>.pdf — derived from
    master resume + this run's metadata. Used by the export route."""

    def changes_by_kind(self) -> dict[str, int]:
        """Group change_log by kind — used by the UI to show
        '5 reword + 2 reorder + 8 ats_keyword_add' summary."""
        counts: dict[str, int] = {}
        for c in self.change_log:
            counts[c.kind] = counts.get(c.kind, 0) + 1
        return counts

    def lift(self) -> float:
        """fit_estimate.after - before. Negative means tailoring hurt
        (rare, but possible if the LLM misread the JD)."""
        return self.fit_estimate.after - self.fit_estimate.before

    def has_unfabricated_audit(self) -> bool:
        """True when cannot_fake_warnings is non-empty — the LLM
        actively reasoned about temptation. Slightly counter-intuitive,
        but a model that produces 0 warnings on a borderline-hard JD is
        either too easy a JD, or hiding fabrication. Used as a
        sanity-check signal in the GEPA metric."""
        return len(self.cannot_fake_warnings) > 0
