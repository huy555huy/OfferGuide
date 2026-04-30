"""SKILL-specific adapters for GEPA evolution.

Each evolvable SKILL ships a module here that satisfies the
``SkillAdapter`` Protocol from ``_base.py`` — it provides the trainset,
metric function, and axis names the runner needs.

Adding a new evolvable SKILL is a matter of:
1. Drop a module ``adapters/<skill_name>.py``
2. Register it in ``REGISTRY`` below
3. The runner picks it up automatically by name; the CLI works for free.
"""

from __future__ import annotations

from types import ModuleType

from . import analyze_gaps, deep_project_prep, prepare_interview, score_match
from ._base import MetricBreakdown, SkillAdapter, aggregate, parse_json_output

REGISTRY: dict[str, ModuleType] = {
    "score_match": score_match,
    "analyze_gaps": analyze_gaps,
    "prepare_interview": prepare_interview,
    "deep_project_prep": deep_project_prep,
}


def get_adapter(skill_name: str) -> ModuleType:
    """Return the adapter module for ``skill_name``.

    Raises ``KeyError`` with the list of known SKILLs if the name isn't
    registered. The runner catches this to give the user a helpful error.
    """
    if skill_name not in REGISTRY:
        raise KeyError(
            f"no evolution adapter for {skill_name!r}; "
            f"known: {sorted(REGISTRY)}"
        )
    return REGISTRY[skill_name]


def list_evolvable_skills() -> list[str]:
    """All SKILL names that have an evolution adapter registered."""
    return sorted(REGISTRY)


__all__ = [
    "REGISTRY",
    "MetricBreakdown",
    "SkillAdapter",
    "aggregate",
    "analyze_gaps",
    "deep_project_prep",
    "get_adapter",
    "list_evolvable_skills",
    "parse_json_output",
    "prepare_interview",
    "score_match",
]
