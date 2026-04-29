"""Pydantic schema for prepare_interview SKILL output.

The SKILL takes ``(company, job_text, user_profile, past_experiences)``
and emits a structured prep packet: company snapshot, ranked likely
questions across 5 categories, focus areas, and weak spots.

Validation contract: callers (agent, examples, GEPA metric) run
``PrepareInterviewResult.model_validate(...)`` on the LLM output. Any
extra keys / missing keys / out-of-range values surface as a structured
failure that becomes feedback for the GEPA reflection model when this
SKILL is evolved.

Evolution path: a per-SKILL trainset is needed. The natural metric is
the **interview-question hit rate** — fraction of `expected_questions`
that actually appeared in the user's interview. That trainset arrives
once we have ~30 dogfood interview reflections. Until then this SKILL
runs as v0.1.0 hand-written prompt; evolution lands when there's data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionCategory = Literal[
    "technical",
    "behavioral",
    "system_design",
    "company_specific",
    "project_deep_dive",
]


class InterviewQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    """The interview question itself, in Chinese (or matching the JD language)."""

    category: QuestionCategory
    """Which of the 5 dimensions — see SKILL body for what each means."""

    likelihood: float = Field(ge=0.0, le=1.0)
    """Calibrated probability this exact (or near-paraphrase) question gets asked.

    The SKILL prompt asks the LLM to be honest about uncertainty —
    0.5 means "I don't know", not a default placeholder. The hit-rate
    metric (eventual) will train this calibration."""

    rationale: str
    """Why this question is likely — JD line, profile match, past
    experience clue, etc. Keeps the LLM grounded in evidence."""


class PrepareInterviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_snapshot: str
    """2-3 Chinese sentences: business focus, tech stack tendency,
    interview style cues from the JD or past 面经."""

    expected_questions: list[InterviewQuestion]
    """Up to 8 ranked by `likelihood` desc. Must span >= 3 of the 5
    categories — interview prep that's all-technical or all-behavioral
    fails users in the actual interview."""

    prep_focus_areas: list[str] = Field(default_factory=list)
    """3-5 study topics — concrete enough to action ("Transformer
    数学推导" not "深度学习基础")."""

    weak_spots: list[str] = Field(default_factory=list)
    """User's weakest spots given JD requirements. Honest, not flattering."""

    def top_questions(self, n: int = 5) -> list[InterviewQuestion]:
        """Most-likely n questions — what the user should rehearse first."""
        return sorted(self.expected_questions, key=lambda q: q.likelihood, reverse=True)[:n]

    def categories_covered(self) -> set[QuestionCategory]:
        """Which of the 5 categories the LLM produced questions for —
        used by the agent layer to warn if coverage is too narrow."""
        return {q.category for q in self.expected_questions}

    def coverage_warning(self) -> str | None:
        """Returns a warning string if fewer than 3 categories are covered."""
        n = len(self.categories_covered())
        if n < 3:
            return (
                f"⚠️ 只覆盖 {n}/5 个题型类别——实际面试可能问到其它类别。"
                " 建议手动补充其它类别的备战。"
            )
        return None
