"""Pydantic schema for deep_project_prep SKILL output.

Contract: from (company, JD, profile) → emit per-project deep prep.

Per-project we want:
- A 1-sentence summary of the project from the interviewer's POV
- The technical claims the simulated panel will probe
- A handful of probing questions, each typed (foundational / deep_dive
  / challenge / tradeoff / extension), each carrying an *answer outline*
  the user can rehearse, plus likely follow-ups
- Honest weak points with a mitigation-narrative for each

Plus three cross-cutting groups:
- ``cross_project_questions`` — questions that require the candidate to
  contrast or unify multiple projects ("which of these gave you the most
  ownership?")
- ``behavioral_questions_tailored`` — STAR-style behavioral questions
  rewritten to reference the user's actual experience, not generic
- ``company_style_summary`` — a brief description of how the target
  company tends to probe this kind of role (drives question style)

The schema uses ``extra='forbid'`` so any GEPA-generated prompt that
emits stray fields fails validation and produces a strong negative
signal back to the reflection LM.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

QuestionType = Literal[
    "foundational",
    "deep_dive",
    "challenge",
    "tradeoff",
    "extension",
]
"""What kind of probing this question represents.

- ``foundational``  basic concept the project depends on (Transformer attn,
                    diffusion forward process, …)
- ``deep_dive``     implementation specifics ("how did you handle KV cache?")
- ``challenge``     question the design decision ("why not ReAct?")
- ``tradeoff``      compare alternatives ("when would you NOT use this?")
- ``extension``     beyond the project scope ("how would this scale 10x?")
"""


class ProbingQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    type: QuestionType
    likelihood: float = Field(ge=0.0, le=1.0)
    """Calibrated probability the question (or near-paraphrase) gets asked."""

    rationale: str
    """Why THIS company on THIS role would ask THIS question.
    Anchored to JD requirements or known company style."""

    answer_outline: list[str] = Field(min_length=2, max_length=6)
    """Bullet anchors for the user to rehearse — NOT a full essay.
    Each bullet should be a concrete fact / claim / number, not vague
    ("be confident") guidance."""

    followups: list[str] = Field(default_factory=list, max_length=3)
    """1-3 plausible follow-up questions the interviewer might fire next."""


class WeakPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weakness: str
    """Honest assessment of where THIS project is weakest given JD demands.
    No flattering — recruiters can smell it."""

    mitigation: str
    """How to *reframe* the weakness as a narrative — not "lie", but
    "tell the truth strategically"."""

    likely_question: str
    """The form the weakness is most likely to surface as in the panel."""


class ProjectAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str
    """As it appears (or canonically labelled) on the resume."""

    project_summary: str
    """One Chinese sentence — interviewer's TL;DR of what this project is."""

    technical_claims: list[str] = Field(min_length=2, max_length=8)
    """Distinct technical claims the project makes (extracted from profile).
    Each should be probe-able — not "我用 Python", more
    "实现了 evidence-centric 上下文管理机制"."""

    probing_questions: list[ProbingQuestion] = Field(min_length=3, max_length=8)
    """The actual prep set for this project. Must span ≥ 3 question types
    so the user isn't blindsided in any direction."""

    weak_points: list[WeakPoint] = Field(default_factory=list, max_length=4)
    """Where the project is most attackable — honest list."""

    def question_types_covered(self) -> set[QuestionType]:
        return {q.type for q in self.probing_questions}


class DeepProjectPrepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_style_summary: str
    """How THIS company tends to probe THIS kind of role. 2-3 sentences
    in Chinese, anchored in known signal (past 面经, JD style, public
    interview blog posts) — say so explicitly when there's no signal."""

    projects_analyzed: list[ProjectAnalysis] = Field(min_length=1, max_length=4)
    """Up to 4 projects ranked by JD-relevance. The SKILL is responsible
    for picking which projects from the resume merit deep prep."""

    cross_project_questions: list[ProbingQuestion] = Field(default_factory=list, max_length=5)
    """Questions that span projects. Often asked at staff/principal level
    panels to probe meta-thinking."""

    behavioral_questions_tailored: list[ProbingQuestion] = Field(
        default_factory=list, max_length=5
    )
    """STAR questions rewritten to reference the user's actual experience.
    Generic ('讲一个跨团队协作') is forbidden; specific
    ('在法至科技实习时跟产品对齐 agent 工作流，分歧怎么解决的') wins."""

    def all_questions(self) -> list[ProbingQuestion]:
        """Flat list across projects + cross + behavioral, useful for
        rendering a single ranked agenda in the UI."""
        out: list[ProbingQuestion] = list(self.cross_project_questions)
        out.extend(self.behavioral_questions_tailored)
        for p in self.projects_analyzed:
            out.extend(p.probing_questions)
        return out

    def total_question_count(self) -> int:
        return len(self.all_questions())

    def weakest_spots(self, limit: int = 5) -> list[WeakPoint]:
        """Concatenated weak points across all projects, capped."""
        out: list[WeakPoint] = []
        for p in self.projects_analyzed:
            out.extend(p.weak_points)
        return out[:limit]
