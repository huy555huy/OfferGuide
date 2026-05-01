"""Pydantic schema for successful_profile SKILL output.

The SKILL takes ``(company, role_hint, high_quality_samples_json)`` and
emits a structured profile of "what a successful candidate at this
company / role looks like": their background, skills, projects, the
questions they got asked, and—most important—why they got the offer.

The strict evidence requirement (every claim must trace back to a sample
in ``high_quality_samples_json``) is enforced at three layers:

1. SKILL body explicitly bans fabrication
2. ``evidence_sources`` field lists every source the synthesis used —
   the UI shows them as clickable links so the user can audit
3. ``evidence_count`` on common_questions counts how many samples
   referenced the same question (1 = drop it, that's a one-off)

This SKILL is the input to ``profile_resume_gap`` — only after we have
a profile can we compute "user has X / lacks Y / can fix Z short-term /
can't fix W". It's also the input to the ``/profile/{company}`` UI
page.

Validation contract: callers run ``SuccessfulProfileResult.model_validate``
on the LLM output. Any schema violation surfaces as a structured failure
the GEPA reflection model can act on when the SKILL evolves.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ContentKind = Literal[
    "interview", "offer_post", "project_share", "reflection", "other"
]


class BackgroundPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    education_level: str = ""
    """Aggregate education-level pattern. e.g. "硕士占多数, 本科少数". Empty
    when samples don't reveal it (then ``uncertainty_notes`` says so)."""

    school_tier: str = ""
    """e.g. "985 + 211 头部". Empty when not inferable."""

    majors: list[str] = Field(default_factory=list)
    """Top-3 most common majors across samples."""

    internships: list[str] = Field(default_factory=list)
    """Typical internship patterns mentioned (e.g. "美团 / 字节 / 头部
    AI 实验室一段")."""

    competitions: list[str] = Field(default_factory=list)
    """Algorithm / project competitions mentioned."""

    publications: list[str] = Field(default_factory=list)
    """Papers / open-source / blog signals."""


class SkillPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    must_have: list[str] = Field(default_factory=list)
    """Skills observed in ≥ 70% of samples — these are the floor."""

    highly_valued: list[str] = Field(default_factory=list)
    """Skills observed in ~50% — these are bonus, not required."""

    differentiators: list[str] = Field(default_factory=list)
    """Rare-but-powerful skills (1-2 samples mention) that correlate with
    the offer in those specific cases. The user might be able to focus
    on these to stand out."""


class ProjectPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    typical_project_themes: list[str] = Field(default_factory=list)
    common_tech_stacks: list[str] = Field(default_factory=list)
    scale_signals: list[str] = Field(default_factory=list)
    """e.g. "百万 DAU" / "10w+ QPS" / "AUC 提升 0.04". Honest signal:
    no scale → smaller projects → still valid for some companies."""
    outcome_signals: list[str] = Field(default_factory=list)


class CommonQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    category: Literal[
        "technical", "behavioral", "system_design",
        "company_specific", "project_deep_dive",
    ]
    evidence_count: int = Field(ge=1)
    """Number of samples in which this question (or near-paraphrase)
    appeared. Drop questions with evidence_count == 1 — those are
    one-off, not a pattern."""


class InterviewPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    common_questions: list[CommonQuestion] = Field(default_factory=list)
    behavioral_themes: list[str] = Field(default_factory=list)
    decision_factors: list[str] = Field(default_factory=list)


class EvidenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    """Original source identifier — 'nowcoder' / 'manual_paste' / etc."""

    kind: ContentKind
    url: str = ""
    """Empty when manually pasted."""

    sample_id: int | None = None
    """``interview_experiences.id`` of the source row, when persisted —
    lets the UI link back."""


class SuccessfulProfileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    role_focus: str
    evidence_count: int = Field(ge=0)
    evidence_kinds: list[str] = Field(default_factory=list)
    """e.g. ["offer_post", "interview", "project_share"] — what mix
    of evidence types contributed."""

    background_pattern: BackgroundPattern
    skill_pattern: SkillPattern
    project_pattern: ProjectPattern
    interview_pattern: InterviewPattern

    why_they_passed: list[str] = Field(default_factory=list)
    """3-5 aggregated reasons. Each must trace back to evidence — the
    SKILL body insists on '(来自 X 条 offer_post)'-style attribution."""

    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    """Every sample that contributed. UI surfaces as clickable links."""

    uncertainty_notes: list[str] = Field(default_factory=list)
    """Honest "this is the part we're not sure about" — small samples,
    stale data, single source, etc."""

    def confidence(self) -> float:
        """Heuristic confidence in the profile.

        Computed from evidence_count and unique_kinds — more diverse
        evidence → higher confidence. UI uses this to decide when to
        warn the user that the profile is preliminary.
        """
        # Base on count: 3+ samples → 0.5, 5+ → 0.7, 8+ → 0.85
        n = self.evidence_count
        if n >= 8:
            base = 0.85
        elif n >= 5:
            base = 0.7
        elif n >= 3:
            base = 0.5
        elif n >= 1:
            base = 0.3
        else:
            base = 0.0
        # Bonus for diverse kinds
        kinds_bonus = 0.05 * min(3, len(set(self.evidence_kinds)))
        # Penalty if there are uncertainty_notes
        penalty = min(0.15, 0.05 * len(self.uncertainty_notes))
        return max(0.0, min(1.0, base + kinds_bonus - penalty))
