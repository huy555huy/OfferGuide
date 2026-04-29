"""Cross-platform job representation.

Every platform adapter (`nowcoder`, `manual`, future `boss_extension`) returns
`RawJob` instances. The `Scout` worker is platform-agnostic and works against
this shape only — adapters can be added without touching Scout.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RawJob:
    """A single JD as seen by exactly one platform adapter."""

    source: str
    """Platform identifier — 'nowcoder' / 'boss_extension' / 'manual' / ..."""

    title: str
    raw_text: str
    """Canonical descriptive text for this job (responsibilities + requirements +
    salary + location + ...). Source of truth for the LLM and for dedup hashing."""

    source_id: str | None = None
    """Platform-internal stable id (e.g. nowcoder's `446211`). None for manual paste."""

    url: str | None = None
    company: str | None = None
    location: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    """Platform-specific raw fields: salary range, edu level, avg_process_rate, etc.
    Stored as JSON in the `jobs` table for downstream tools / SKILLs to consult."""


def canonical_text(rj: RawJob) -> str:
    """A single normalized string suitable for both LLM consumption and content hashing.

    Stable across re-runs: same fields → same string → same hash → dedup works.
    """
    return "\n".join(
        line
        for line in [
            f"标题: {rj.title}",
            f"公司: {rj.company}" if rj.company else "",
            f"地点: {rj.location}" if rj.location else "",
            "",
            rj.raw_text.strip(),
        ]
        if line != ""
    ).strip()


def content_hash(rj: RawJob) -> str:
    """Stable SHA-256 hash over the canonical text — feeds the `jobs.content_hash` UNIQUE constraint."""
    return hashlib.sha256(canonical_text(rj).encode("utf-8")).hexdigest()
