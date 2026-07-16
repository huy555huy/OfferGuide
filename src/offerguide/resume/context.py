"""Build the one complete context used by every resume editing operation."""

from __future__ import annotations

from collections.abc import Sequence

from .master import MasterResumeDocument, MasterResumeSource
from .models import (
    ProjectFact,
    ResumeContext,
    ResumeDocument,
    ResumeJobContext,
)


def build_resume_context(
    *,
    job: ResumeJobContext,
    master_source: MasterResumeSource,
    master_document: MasterResumeDocument,
    project_facts: Sequence[ProjectFact] = (),
    feedback: Sequence[str] = (),
    current_resume: ResumeDocument | None = None,
) -> ResumeContext:
    """Assemble inputs without slicing, summarizing, inferring, or dropping material."""
    return ResumeContext(
        job=job,
        master_source=master_source,
        master_document=master_document,
        project_facts=list(project_facts),
        feedback=_copy_text_sequence("feedback", feedback),
        current_resume=current_resume,
    )


def _copy_text_sequence(name: str, values: Sequence[str]) -> list[str]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings, not one string")
    copied = list(values)
    for value in copied:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain only strings")
        if not value.strip():
            raise ValueError(f"{name} must not contain blank items")
    return copied


__all__ = ["build_resume_context"]
