"""Pydantic schema for ``post_interview_reflection`` SKILL output.

Borrowed pattern from `Pytai <https://github.com/getFrontend/app-ai-interviews>`_
and `GPTInterviewer <https://github.com/jiatastic/GPTInterviewer>`_ (MIT):
post-interview transcript analysis is the **closure of the dogfood loop**.
Without it we predict (prepare_interview / deep_project_prep) but never
learn whether predictions were correct.

The SKILL takes:
- ``company``: which company this interview was for
- ``prep_questions_json``: the prediction the agent made (from prep_run /
  deep_prep_run) — JSON-stringified list of {question, category,
  likelihood, rationale}
- ``actual_transcript``: user's free-form recap of what was actually
  asked + how they did

It outputs the **structured feedback** that auto-updates:
1. ``story_bank`` — promising STAR moments → new behavioral_stories
2. ``company_briefs`` — interview_style refinement based on observed
   patterns
3. The user's own next-interview prep (which weak spots to practice)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionMatchKind = Literal[
    "exact",       # the exact question we predicted came up
    "paraphrase",  # same intent, different wording
    "category",    # same category but different specific question
    "miss",        # we predicted, they didn't ask
]


class QuestionMatch(BaseModel):
    """One prediction-vs-reality outcome."""

    model_config = ConfigDict(extra="forbid")

    predicted_question: str
    """The question text from the prep output."""

    predicted_likelihood: float = Field(ge=0.0, le=1.0)
    """The likelihood we assigned at prediction time. Used for
    calibration analysis (high-likelihood + miss = bad calibration)."""

    match_kind: QuestionMatchKind
    """How the prediction fared in the actual interview."""

    actual_question: str | None
    """If matched, the actual question wording (often slightly different
    from the predicted one — useful evidence)."""

    user_self_rating: float | None = Field(default=None, ge=0.0, le=1.0)
    """User's own assessment of how they answered, if reported.
    None when the user didn't rate this question."""


class SurpriseQuestion(BaseModel):
    """A question the agent did NOT predict but was asked anyway."""

    model_config = ConfigDict(extra="forbid")

    question: str
    category: Literal[
        "technical", "behavioral", "system_design",
        "company_specific", "project_deep_dive",
    ]
    why_we_missed: str
    """Honest analysis of why our prediction missed this. Examples:
    'company_specific: didn't have signal on this team's recent work',
    'tradeoff: we under-weighted challenge questions for this seniority'."""


class StorySuggestion(BaseModel):
    """A STAR story candidate worth adding to the story bank."""

    model_config = ConfigDict(extra="forbid")

    title: str
    """Short label, like the user would title it."""

    suggested_situation: str
    """Drafted from the transcript. The user reviews + edits before
    actually inserting into behavioral_stories."""

    suggested_task: str
    suggested_action: str
    suggested_result: str
    suggested_reflection: str | None = None

    suggested_tags: list[str] = Field(default_factory=list, max_length=4)
    """Theme tags from the RECOMMENDED_TAGS vocabulary."""

    triggered_by: str
    """The actual question / topic in the transcript that surfaced
    this story. Anchors the suggestion in evidence."""


class BriefDelta(BaseModel):
    """A proposed update to the company_briefs row."""

    model_config = ConfigDict(extra="forbid")

    interview_style_addition: str | None = None
    """Append to interview_style if not None."""

    new_recent_signals: list[str] = Field(default_factory=list, max_length=4)
    """Add these to the brief's recent_signals."""

    confidence_adjustment: float = 0.0
    """Add to brief.confidence (clamped to [0, 1] downstream).
    Positive = our predictions held up; negative = miscalibration."""


class PostInterviewReflection(BaseModel):
    """The full reflection packet from one interview round."""

    model_config = ConfigDict(extra="forbid")

    company: str
    """Echoed from input."""

    hit_rate: float = Field(ge=0.0, le=1.0)
    """Fraction of predicted questions that surfaced in some form
    (exact + paraphrase + category, divided by total predicted)."""

    matched_predictions: list[QuestionMatch] = Field(default_factory=list)
    """One entry per predicted question — match_kind says how it went."""

    surprises: list[SurpriseQuestion] = Field(default_factory=list, max_length=8)
    """Questions asked that we didn't predict — most useful learning."""

    user_performance_summary: str
    """2-3 Chinese sentences. Honest take on how the user did."""

    suggested_stories: list[StorySuggestion] = Field(default_factory=list, max_length=4)
    """STAR moments from the transcript that should become master
    stories. User reviews before inserting."""

    brief_delta: BriefDelta
    """Suggested updates to the company brief. Auto-applied (gracefully)
    by the post-interview endpoint."""

    weak_spots_to_practice: list[str] = Field(default_factory=list, max_length=5)
    """Concrete topics the user should drill before the next round."""

    def calibration_score(self) -> float:
        """Per-question calibration: high-likelihood predictions should
        be more likely matched. Mean abs error between predicted_likelihood
        and 1.0-if-matched-else-0.0. Lower = better calibrated."""
        if not self.matched_predictions:
            return 0.0
        n = len(self.matched_predictions)
        err = 0.0
        for m in self.matched_predictions:
            actual = 0.0 if m.match_kind == "miss" else 1.0
            err += abs(m.predicted_likelihood - actual)
        return err / n
