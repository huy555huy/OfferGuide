"""Pydantic schema for analyze_gaps SKILL output.

Two-level structure: keyword_gaps (raw keyword presence) + suggestions
(actionable patches). Each suggestion carries an explicit `ai_risk` so the
agent can warn the user before they accept anything that looks AI-generated —
this is the OfferGuide answer to the 49% auto-dismiss-AI-resume rate
([gettailor 2026](https://www.gettailor.ai/blog/ai-resume-detection)).

`AnalyzeGapsResult` is what `score_match`/the W4 agent will validate against.
Validation failure → GEPA gets structured feedback the optimizer can act on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Importance = Literal["high", "medium", "low"]
SuggestionAction = Literal["add", "emphasize", "reword"]
AiRisk = Literal["low", "medium", "high"]


class KeywordGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jd_keyword: str
    in_resume: bool
    importance: Importance
    evidence_in_jd: str
    """The literal JD sentence(s) that mention this keyword — keeps the LLM honest."""


class Suggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section: str
    """Where in the resume the change applies — '项目经历' / '技能' / '教育' / etc."""

    action: SuggestionAction
    """`add` (new bullet/keyword), `emphasize` (call out something already there),
    `reword` (rephrase one specific line)."""

    current_text: str | None
    """If `action` is emphasize/reword, the existing snippet being modified.
    Null when `action='add'`."""

    proposed_addition: str
    """1-2 sentences the user can copy-paste verbatim. Not a rewrite of a section."""

    reason: str
    """Which JD requirement this addresses."""

    ai_risk: AiRisk
    """How likely this addition would trigger an AI-detection filter. See SKILL body."""

    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeGapsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    """2-3 Chinese sentences: overall match + the 1-2 biggest gaps."""

    keyword_gaps: list[KeywordGap]
    suggestions: list[Suggestion]

    do_not_add: list[str] = Field(default_factory=list)
    """Things the LLM was tempted to suggest but explicitly should NOT — fabricated
    experience, marketing fluff, etc. Visible to the user as a transparency note."""

    ai_detection_warnings: list[str] = Field(default_factory=list)
    """Holistic observations about how the suggested edits might shift the resume's
    overall AI-detection risk (style consistency, jargon density, etc.)."""

    def high_risk_count(self) -> int:
        """Number of suggestions flagged ai_risk=high — surfaced in the agent UI."""
        return sum(1 for s in self.suggestions if s.ai_risk == "high")

    def deal_breaker_keyword_gaps(self) -> list[KeywordGap]:
        """High-importance keywords that aren't in the resume — hard blockers for HR."""
        return [g for g in self.keyword_gaps if g.importance == "high" and not g.in_resume]
