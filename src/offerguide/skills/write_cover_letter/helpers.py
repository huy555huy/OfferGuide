"""Pydantic schema for ``write_cover_letter`` SKILL output.

Inspired by `Career-Ops <https://github.com/santifer/career-ops>`_ (MIT)
and `Resume-Matcher <https://github.com/srbhr/Resume-Matcher>`_ (Apache 2.0)
which both ship cover-letter generation modules. We adopt the
**6-block evaluation framework** from Career-Ops at output level:

1. ``opening_hook`` — what makes this candidate match this role specifically
2. ``narrative_body`` — 1-3 paragraphs grounded in profile evidence
3. ``customization_signals`` — concrete proofs of read-the-JD-carefully
4. ``ats_keywords_used`` — list of JD keywords woven in naturally
5. ``closing_call_to_action`` — explicit ask + availability
6. ``personalization_score`` — self-rated 0-1 (how customized vs boilerplate)

Why ``ats_keywords_used`` and ``ai_risk_warnings`` are explicit fields:
49% of HR systems auto-dismiss AI-detected text. By exposing them in
the schema, the user (and downstream metrics) can audit whether the
LLM smuggled in giveaway phrases like "I am writing to express my
interest in" or "leverage my skills".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CoverLetterTone = Literal[
    "formal",        # 正式、严谨（金融 / 国企 / 大厂 HR 通用）
    "warm_concise",  # 温和但简洁（互联网大厂技术岗推荐）
    "enthusiastic",  # 热情（创业公司 / agentic 岗）
    "conservative",  # 保守（学术 / 研究院）
]


class CoverLetterResult(BaseModel):
    """The full cover letter packet returned by the SKILL."""

    model_config = ConfigDict(extra="forbid")

    opening_hook: str
    """1-2 sentences. The 'why this candidate, why this role' opener.
    Borrowed from Career-Ops: must reference a *specific* JD requirement
    or company signal — generic 'I am excited to apply' fails this."""

    narrative_body: list[str] = Field(min_length=1, max_length=4)
    """1-4 paragraph blocks. Each grounds a JD keyword in the candidate's
    actual project / experience. No filler."""

    closing_call_to_action: str
    """Explicit ask + availability ('I can start in May 2026 and would
    welcome a conversation about ...'). Anti-passive."""

    customization_signals: list[str] = Field(default_factory=list, max_length=5)
    """Concrete signals the writer read the JD carefully — e.g.
    'mentioned the team's recent paper on X', 'addressed the 5x scale
    requirement specifically'. Helps differentiate from boilerplate."""

    ats_keywords_used: list[str] = Field(default_factory=list, max_length=15)
    """JD keywords woven into the letter naturally. Surfaced so the
    user can audit naturalness (15+ keywords stuffed = ATS-detectable)."""

    ai_risk_warnings: list[str] = Field(default_factory=list, max_length=5)
    """Phrases the writer chose that *could* trigger AI-detection
    filters. Honest self-audit — empty list when the writer dodged them."""

    suggested_tone: CoverLetterTone
    """The writer's tone choice, with rationale."""

    personalization_score: float = Field(ge=0.0, le=1.0)
    """Writer's self-rated 0-1 — how customized vs how boilerplate.
    >= 0.7 = strongly customized; < 0.4 = generic, user should reject."""

    overall_word_count: int = Field(ge=50, le=600)
    """Approximate word count (the LLM counts its own output). Cover
    letters that hit > 500 words for a校招 internship are too long."""

    def render_plain(self) -> str:
        """Render to plain text (paragraphs separated by blank lines)."""
        parts = [self.opening_hook]
        parts.extend(self.narrative_body)
        parts.append(self.closing_call_to_action)
        return "\n\n".join(p.strip() for p in parts if p)

    def has_high_ai_risk(self) -> bool:
        """True if more than 2 ai_risk_warnings — user should review."""
        return len(self.ai_risk_warnings) > 2
