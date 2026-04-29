"""User profile schema — what we know about the user beyond their resume text.

W1 stores only the parsed raw text and minimal metadata. The structured fields
(`education`, `experience`, `skills`, `preferences`) get LLM-populated in W2 by
the profile-build SKILL. They're defined here now so the storage shape is
stable from day one and so type-checkers catch typos.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EducationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    school: str
    degree: str | None = None
    major: str | None = None
    start: str | None = None  # 'YYYY-MM' free-form is fine for now
    end: str | None = None
    notes: str | None = None


class ExperienceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    org: str | None = None
    start: str | None = None
    end: str | None = None
    bullets: list[str] = Field(default_factory=list)


class JobPreferences(BaseModel):
    """What the user is looking for. Drives the score_match SKILL."""

    model_config = ConfigDict(extra="forbid")

    target_roles: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    cities: list[str] = Field(default_factory=list)
    salary_floor: int | None = None
    company_tiers: list[str] = Field(default_factory=list)
    """Free-form labels the user cares about — '大厂' / '独角兽' / 'startup' / '外企' / etc."""


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    preferences: JobPreferences = Field(default_factory=JobPreferences)

    raw_resume_text: str = ""
    """The full extracted resume text. Source of truth for any LLM-driven re-parse."""

    source_pdf: str | None = None
    """Original PDF path. Kept for re-extraction when pypdf or the prompt evolves."""
