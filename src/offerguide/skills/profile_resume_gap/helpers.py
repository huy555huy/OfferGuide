"""Pydantic schema for profile_resume_gap SKILL — the 4-bucket gap analysis.

This SKILL takes a ``SuccessfulProfile`` and the user's resume, then sorts
every meaningful claim from the profile into one of four buckets:

1. **已具备** — resume already evidences this; user is set
2. **短期能补 (≤2 周)** — can be acquired through self-study/projects in days/weeks
3. **短期补不了** — needs months / internships / publications to acquire
4. **不能编** — externally verifiable facts (school, internship company, etc.)

The 4-bucket split is the difference between "a generic gap analysis" and
"actionable pre-application briefing". Telling a user "you lack 3 months of
production ML experience" + "you can spend 2 weeks doing X open-source
project to demonstrate similar capability" is far more useful than a flat
"you don't match these 5 requirements".

The "不能编" bucket is the **honesty enforcement** layer — explicitly
labels which gaps cannot be papered over and tells the user how to
reframe (not fake) honestly. This is intentional design against the
common Chinese 简历"包装" temptation that backfires in 背调.

Validation contract: callers run ``ProfileResumeGapResult.model_validate``
on the LLM output. Pydantic ``extra='forbid'`` ensures the LLM doesn't
sneak extra unverified fields in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GapVerdict = Literal["go", "maybe", "hold", "skip"]
StrengthLevel = Literal["strong", "moderate", "weak"]


class HaveItem(BaseModel):
    """已具备桶 — user already has evidence in resume."""
    model_config = ConfigDict(extra="forbid")

    topic: str
    """Which capability/project/experience this is."""

    evidence_in_resume: str
    """The exact resume line/section that supports this claim."""

    evidence_in_profile: str
    """The successful_profile field that requires this."""

    strength: StrengthLevel
    """How strong the evidence is — strong/moderate/weak."""


class ShortTermFixItem(BaseModel):
    """短期能补桶 — knowledge/tooling gap closable in ≤2 weeks."""
    model_config = ConfigDict(extra="forbid")

    topic: str
    why_missing: str
    """What in the resume reveals this gap."""

    concrete_action: str
    """Specific actionable steps with time budget. ≥1 verb,
    ≥1 time estimate, no vague language. SKILL body bans 'improve
    fundamentals' style suggestions."""

    estimated_hours: int = Field(ge=1)
    """Total hours the user should budget. Used by the UI to flag
    'too expensive for a 2-week sprint' if estimated_hours > 80."""

    skill_signal_after: str
    """The new resume bullet user can write *after* completing the
    action. Not the action itself — the deliverable that goes into
    the resume."""


class CannotShortTermItem(BaseModel):
    """短期补不了桶 — experience-class gap needing months+."""
    model_config = ConfigDict(extra="forbid")

    topic: str
    why_missing: str

    min_time_to_acquire: str
    """e.g. '≥3 个月' / '≥1 段大厂实习' / '≥1 篇顶会论文'.
    Honest minimum-floor estimate."""

    alternative_demonstration: str
    """How to compensate without faking it — open-source projects,
    paper reproduction, blog series, etc."""


class CannotFakeItem(BaseModel):
    """不能编桶 — externally verifiable fact (school, company, dates)."""
    model_config = ConfigDict(extra="forbid")

    topic: str

    why_unfakeable: str
    """Which verification channel exposes a fake claim — 学信网 /
    背调 / 官方榜单 / 公开 GitHub commit history etc."""

    reframe_strategy: str
    """How to honestly present the fact, not how to hide it.
    e.g. 'school is not 985 → emphasize ACM medal + paper reproduction'.
    The SKILL body insists this be honest reframing, not concealment."""


class ApplyAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: GapVerdict
    rationale_chinese: str
    """One-line reason in Chinese, aggregating the 4 buckets."""

    top_3_pre_apply_actions: list[str] = Field(default_factory=list)
    """The most ROI-positive 3 actions to take *before* applying.
    Ordered most-impactful first."""


class GapCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    covered_profile_fields: int = Field(ge=0)
    """How many fields from the profile this gap actually evaluated.
    Lower = the LLM didn't have data to assess parts of the profile."""

    skipped_due_to_low_evidence: list[str] = Field(default_factory=list)
    """Which profile fields were skipped because evidence was weak."""


class ProfileResumeGapResult(BaseModel):
    """Top-level 4-bucket gap analysis with apply advice."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    company: str
    role_focus: str

    # Buckets — Chinese-named via aliases so JSON survives the LLM
    # (which prefers natural Chinese keys) but Python access stays
    # english-attribute-friendly.
    have: list[HaveItem] = Field(default_factory=list, alias="已具备")
    short_term_fix: list[ShortTermFixItem] = Field(
        default_factory=list, alias="短期能补 (≤2周)",
    )
    cannot_short_term: list[CannotShortTermItem] = Field(
        default_factory=list, alias="短期补不了",
    )
    cannot_fake: list[CannotFakeItem] = Field(
        default_factory=list, alias="不能编",
    )

    apply_advice: ApplyAdvice = Field(alias="投递建议")
    calibration: GapCalibration

    def total_gaps(self) -> int:
        """All non-have items — quick "how many things to address" count."""
        return (
            len(self.short_term_fix)
            + len(self.cannot_short_term)
            + len(self.cannot_fake)
        )

    def short_term_total_hours(self) -> int:
        """Sum of estimated_hours across short_term_fix items.
        UI uses this to surface "if you do all short-term actions,
        budget X hours"."""
        return sum(item.estimated_hours for item in self.short_term_fix)

    def has_unfakeable_blocker(self) -> bool:
        """True when 不能编 has any items — those are the absolute
        floor the user can't change. UI surfaces these prominently."""
        return len(self.cannot_fake) > 0
