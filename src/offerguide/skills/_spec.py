"""Static specification of a SKILL — the on-disk format borrowed from Hermes Agent.

A SKILL is a directory containing exactly one `SKILL.md` (YAML frontmatter +
markdown body) and zero or more helper Python scripts. The markdown body IS the
prompt template — that's why GEPA can evolve it as plain text without touching
any agent runtime code.

Field naming follows Hermes' on-disk format (so a Hermes skill can be loaded
verbatim) plus three OfferGuide-specific extensions for the self-evolution loop:
`inputs`, `output_schema`, `evolved_at` / `parent_version`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class SkillSpec:
    """Parsed shape of a SKILL on disk. Immutable so it can be cached and hashed."""

    name: str
    description: str
    version: str
    body: str
    """The markdown body — the actual prompt template that the agent renders.
    GEPA evolution mutates this field (and rewrites the SKILL.md on disk)."""

    triggers: tuple[str, ...] = ()
    """Phrases that activate this SKILL. Used by the agent's intent router; not
    consulted when the agent calls the SKILL by name as a tool."""

    tags: tuple[str, ...] = ()
    author: str | None = None
    license: str | None = None

    inputs: tuple[str, ...] = ()
    """Names of the inputs this SKILL expects. Tool argument schemas are derived
    from these names paired with the `output_schema` description."""

    output_schema: str | None = None
    """Free-form description of the SKILL's output. Kept as a string (not JSON
    schema) so it can be evolved by GEPA alongside the body."""

    evolved_at: datetime | None = None
    parent_version: str | None = None
    """Set by the GEPA evolution runner when this SKILL was generated as an
    evolved variant of an earlier version."""

    helper_scripts: tuple[Path, ...] = ()
    source_path: Path | None = None
