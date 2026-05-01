"""Pydantic schema for mock_interview SKILL — turn-based mock interview.

This SKILL is the missing middle of OfferGuide's interview prep loop:

    prepare_interview (predicts questions)
            ↓
    [USER PRACTICES — currently a black box]
            ↓
    post_interview_reflection (analyzes real transcript)

mock_interview is a *real* practice phase: agent plays interviewer,
asks one question at a time, scores the user's answer on 4 dimensions,
gives concrete improvements + a model-answer skeleton, then adapts
difficulty for the next round. After 6-8 rounds, transcript auto-feeds
back into post_interview_reflection — every mock session generates
one more piece of dogfood data for prepare_interview's calibration.

Why turn-based not whole-batch:
- Real interviews are turn-based — practicing whole-batch trains the
  wrong skill (essay-writing instead of conversational depth)
- Difficulty can adapt: a strong answer earns a harder follow-up
- Coverage check happens during the session, not after — model can
  switch category if all 3 questions so far were technical

Why text-only (no STT):
- Whisper adds 200MB+ dep + cost per session; text-only mode loses
  fluency signal but keeps content signal — and for校招技术面 content
  is 80% of the score
- Voice mode is W14+ candidate after dogfood proves it's worth it

Validation contract: caller invokes once per turn with the previous
turn_history. The single SKILL output covers BOTH evaluation of the
last answer AND the next question — fewer round-trips, simpler client.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionCategory = Literal[
    "technical", "behavioral", "system_design",
    "company_specific", "project_deep_dive",
]
Difficulty = Literal["easy", "medium", "hard"]
SessionStatus = Literal["in_progress", "complete"]


class ScoringDimensions(BaseModel):
    """4-axis breakdown — caller can analyze which dim user is weakest on."""
    model_config = ConfigDict(extra="forbid")

    factual_accuracy: float = Field(ge=0.0, le=1.0)
    """公式 / 算法 / 框架原理是否正确."""

    depth: float = Field(ge=0.0, le=1.0)
    """是否答到了边界条件 / 取舍 / 对比 alternative."""

    structure: float = Field(ge=0.0, le=1.0)
    """答题结构清晰度 (high-level → detail / STAR / 逻辑顺序)."""

    evidence: float = Field(ge=0.0, le=1.0)
    """是否用具体案例 / 数字 / 项目支撑."""


class TurnEvaluation(BaseModel):
    """Evaluation of one user answer (the last turn)."""
    model_config = ConfigDict(extra="forbid")

    question: str
    user_answer: str
    score: float = Field(ge=0.0, le=1.0)
    """Aggregate score — calibrated to real校招 distribution (mean ~0.55,
    not centered on 0.7 like an over-eager LLM might do)."""

    scoring_dimensions: ScoringDimensions
    strengths: list[str] = Field(default_factory=list, max_length=3)
    """Only list what was actually good. SKILL body bans filler praise."""

    improvements: list[str] = Field(default_factory=list, max_length=3)
    """Each item must be 'next sentence should say X instead of Y' specific —
    SKILL body explicitly bans '加强深度' style vagueness."""

    model_answer_skeleton: str
    """3-5 bullet skeleton, NOT a full model answer. Lets user
    self-rehearse without spoon-feeding."""

    follow_up_likely: str
    """What a real interviewer would probe next — gives user a hint
    about how to push their answer in real interviews."""


class NextQuestion(BaseModel):
    """The next question to ask (or None if session is complete)."""
    model_config = ConfigDict(extra="forbid")

    question: str
    category: QuestionCategory
    difficulty: Difficulty
    rationale: str
    """One-sentence why this question, this difficulty, now."""

    expected_aspects: list[str] = Field(default_factory=list, min_length=3, max_length=5)
    """3-5 bullets of what a complete answer should cover. Used by the
    next-turn evaluation to score depth."""


class SessionSummary(BaseModel):
    """End-of-session aggregate — only present when status == complete."""
    model_config = ConfigDict(extra="forbid")

    rounds_played: int = Field(ge=1)
    average_score: float = Field(ge=0.0, le=1.0)
    weakest_dimension: str
    """Which of factual_accuracy/depth/structure/evidence had the lowest
    average — directs user to focused practice."""

    strongest_dimension: str
    top_3_takeaways: list[str] = Field(default_factory=list, max_length=3)
    ready_for_real_interview: bool
    """SKILL body's threshold: average_score >= 0.7 AND no dimension < 0.4."""

    rationale: str


class MockInterviewResult(BaseModel):
    """One turn's full output — handles both evaluation and next question."""
    model_config = ConfigDict(extra="forbid")

    company: str
    role_focus: str
    turn_index: int = Field(ge=1)

    evaluation_of_last_answer: TurnEvaluation | None = None
    """None on the very first turn (no previous answer to evaluate)."""

    next_question: NextQuestion | None = None
    """None on the final turn when session_status='complete'."""

    session_status: SessionStatus
    session_summary: SessionSummary | None = None
    """Required when session_status='complete', else None."""

    def is_first_turn(self) -> bool:
        """Used by UI to render correct widget (no eval block on turn 1)."""
        return self.evaluation_of_last_answer is None and self.turn_index == 1

    def is_complete(self) -> bool:
        return self.session_status == "complete"

    def categories_covered_in_history(
        self, prior_questions: list[NextQuestion],
    ) -> set[QuestionCategory]:
        """Coverage check used by the UI to surface 'all-technical' warning."""
        return {q.category for q in prior_questions}
