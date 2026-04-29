"""Parse SKILL.md files on disk into SkillSpec instances.

Format (borrowed from Hermes Agent — see ATTRIBUTION.md):

    ---
    name: score_match
    description: ...
    version: 0.1.0
    triggers: [...]
    ...
    ---

    # Markdown body (this becomes the prompt template)

The frontmatter is delimited by literal `---` lines. Anything after the closing
`---` is the body. Helper Python scripts in the same directory (excluding
`__init__.py`) are auto-discovered and exposed via `SkillSpec.helper_scripts`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from ._spec import SkillSpec

_FRONTMATTER_DELIM = "---"


class SkillParseError(ValueError):
    """Raised when a SKILL.md is malformed."""


def load_skill(skill_dir: Path | str) -> SkillSpec:
    """Load a single SKILL from `skill_dir/SKILL.md`."""
    skill_dir = Path(skill_dir)
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise SkillParseError(f"No SKILL.md in {skill_dir}")

    raw = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw, source=skill_md)
    meta = yaml.safe_load(frontmatter) or {}

    if "name" not in meta or "description" not in meta or "version" not in meta:
        raise SkillParseError(
            f"{skill_md}: SKILL frontmatter requires `name`, `description`, `version`"
        )

    helpers = tuple(
        sorted(
            p
            for p in skill_dir.glob("*.py")
            if p.name not in {"__init__.py", "_spec.py", "_loader.py"}
        )
    )

    return SkillSpec(
        name=meta["name"],
        description=meta["description"],
        version=str(meta["version"]),
        body=body.strip(),
        triggers=_as_tuple_str(meta.get("triggers")),
        tags=_as_tuple_str(meta.get("tags")),
        author=meta.get("author"),
        license=meta.get("license"),
        inputs=_as_tuple_str(meta.get("inputs")),
        output_schema=meta.get("output_schema"),
        evolved_at=_parse_dt(meta.get("evolved_at")),
        parent_version=meta.get("parent_version"),
        helper_scripts=helpers,
        source_path=skill_md,
    )


def discover_skills(root: Path | str) -> list[SkillSpec]:
    """Find every SKILL.md under `root`, returned in deterministic name order."""
    root = Path(root)
    found: list[SkillSpec] = []
    for skill_md in sorted(root.rglob("SKILL.md")):
        found.append(load_skill(skill_md.parent))
    return found


def _split_frontmatter(text: str, source: Path) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip() != _FRONTMATTER_DELIM:
        raise SkillParseError(f"{source}: missing opening `---` frontmatter delimiter")
    end_idx: int | None = None
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip() == _FRONTMATTER_DELIM:
            end_idx = i
            break
    if end_idx is None:
        raise SkillParseError(f"{source}: missing closing `---` frontmatter delimiter")
    return "".join(lines[1:end_idx]), "".join(lines[end_idx + 1 :])


def _as_tuple_str(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    raise SkillParseError(f"Expected list-of-string or string, got {type(value).__name__}")


def _parse_dt(value: object) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise SkillParseError(f"Expected ISO datetime or null, got {type(value).__name__}")
