"""Pydantic schema for the score_match SKILL's output + calibration helpers.

The SKILL body asks the LLM to emit JSON shaped like `ScoreMatchResult`. Callers
(`agent`, `examples/score_jd.py`, GEPA's metric function) validate the LLM's
JSON via `ScoreMatchResult.model_validate(...)`. Validation failure surfaces a
structured error that the GEPA loop can use as feedback for prompt evolution.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ScoreDimensions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tech: float = Field(ge=0.0, le=1.0)
    exp: float = Field(ge=0.0, le=1.0)
    company_tier: float = Field(ge=0.0, le=1.0)


class ScoreMatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str
    dimensions: ScoreDimensions
    deal_breakers: list[str] = Field(default_factory=list)


def calibration_floor(raw_score: float, n_samples: int) -> float:
    """Bayesian shrinkage of a raw probability toward 0.5 when n is small.

    Used when the agent has too few historical reply-rate observations to
    trust the raw model probability — `n=0` → returns 0.5 exactly, and the
    pull weakens as `n` grows. Will be invoked from the score_match prompt's
    tool sub-call once the agent loop is wired in W4.
    """
    if n_samples <= 0:
        return 0.5
    weight = n_samples / (n_samples + 10)
    return raw_score * weight + 0.5 * (1 - weight)
