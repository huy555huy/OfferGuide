"""Pydantic models for apply_assistant SKILL output validation.

Mirrors the SKILL.md output_schema. The /apply UI route uses these to
render the package + enable per-section copy-to-clipboard buttons.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SelfIntroSnippet(BaseModel):
    """3-line self-intro for Boss直聘 / 牛客 first message."""
    model_config = ConfigDict(extra="forbid")

    platform_hint: Literal["boss_zhipin", "niuke", "generic"] = "boss_zhipin"
    text: str = Field(..., min_length=20, max_length=400,
                       description="The actual paste-ready text, 80-150 字 ideal")
    rationale: str = Field(..., min_length=5, max_length=200)


class QATemplate(BaseModel):
    """One application form question + answer template."""
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=2, max_length=200)
    category: Literal[
        "motivation", "fit", "logistics", "salary", "weakness", "other"
    ]
    answer: str = Field(..., min_length=20, max_length=600)
    anti_patterns: list[str] = Field(default_factory=list, max_length=4)
    personalization_score: float = Field(..., ge=0.0, le=1.0)


class SubmissionStrategy(BaseModel):
    """Per-job submission strategy hints."""
    model_config = ConfigDict(extra="forbid")

    best_time_window: str = Field(..., min_length=2, max_length=120)
    platform_specific_tips: list[str] = Field(default_factory=list, max_length=6)
    follow_up_plan: str = Field(..., min_length=5, max_length=300)
    expected_response_window_days: int = Field(..., ge=1, le=60)


class ApplyPackage(BaseModel):
    """Top-level apply_assistant SKILL output."""
    model_config = ConfigDict(extra="forbid")

    company: str
    role_focus: str
    self_intro_snippet: SelfIntroSnippet
    qa_templates: list[QATemplate] = Field(default_factory=list, max_length=8)
    submission_strategy: SubmissionStrategy
    pre_submit_checklist: list[str] = Field(default_factory=list, max_length=8)
    skip_reasons: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("qa_templates")
    @classmethod
    def _at_most_six_qa(cls, v: list[QATemplate]) -> list[QATemplate]:
        # Schema says 3-6, allow 8 in pydantic for proxies that occasionally
        # over-generate, but warn (caller can trim)
        return v[:6] if len(v) > 6 else v

    @property
    def should_skip(self) -> bool:
        """Convenience: is this a 'don't apply' verdict?"""
        return bool(self.skip_reasons)
