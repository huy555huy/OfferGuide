"""W15.15 — Anthropic Agent Skills Open Standard compatibility regression.

Anthropic Skills (2026-01 Open Standard) requires:
1. Folder containing one ``SKILL.md`` per skill
2. YAML frontmatter at top with ``name`` (str) + ``description`` (str)
3. ``description`` should be ≤1024 chars and tell the model **when** to use it

Our SkillSpec (Hermes-inherited format) goes beyond with version / inputs /
triggers / output_schema, but those are silently ignored by Anthropic loaders.
The same SKILL.md can be:
- Loaded by our SkillRuntime (full feature set)
- Loaded as a Claude Code / Codex plugin (name+description only)

This test ensures we don't drift out of compliance — every SKILL must keep
its name + description fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def _all_skill_md() -> list[Path]:
    return sorted(SKILLS_ROOT.glob("*/SKILL.md"))


@pytest.mark.parametrize("skill_md", _all_skill_md(), ids=lambda p: p.parent.name)
class TestAnthropicSkillsCompat:
    def test_starts_with_yaml_frontmatter(self, skill_md: Path) -> None:
        text = skill_md.read_text(encoding="utf-8")
        assert text.startswith("---\n"), (
            f"{skill_md.parent.name}/SKILL.md must start with YAML frontmatter"
        )

    def test_has_name_and_description(self, skill_md: Path) -> None:
        text = skill_md.read_text(encoding="utf-8")
        # crude YAML key check (don't import yaml here — test should
        # work without the project's yaml dep installed)
        first_block = text.split("---\n", 2)
        assert len(first_block) >= 3, (
            f"{skill_md.parent.name}/SKILL.md must have closed --- frontmatter"
        )
        fm = first_block[1]
        assert any(line.startswith("name:") for line in fm.splitlines()), (
            f"{skill_md.parent.name}: name: missing from frontmatter"
        )
        assert any(line.startswith("description:") for line in fm.splitlines()), (
            f"{skill_md.parent.name}: description: missing from frontmatter"
        )

    def test_description_is_concise(self, skill_md: Path) -> None:
        """Anthropic spec says description ≤ 1024 chars + ideally 1-2 sentences."""
        text = skill_md.read_text(encoding="utf-8")
        first_block = text.split("---\n", 2)
        fm = first_block[1]
        for line in fm.splitlines():
            if line.startswith("description:"):
                desc = line[len("description:"):].strip().strip('"').strip("'")
                # If description spans multiple lines (YAML multi-line), this
                # check would underestimate. We tolerate that — most SKILL
                # descriptions are single-line.
                assert len(desc) <= 1024, (
                    f"{skill_md.parent.name}: description too long ({len(desc)} chars)"
                )
                # Heuristic: should have at least 30 chars (not a stub)
                assert len(desc) >= 20, (
                    f"{skill_md.parent.name}: description too short to be useful"
                )
                return
        pytest.fail(f"{skill_md.parent.name}: description: line not found")


def test_at_least_one_skill_exists() -> None:
    """Sanity: skill discovery isn't broken (e.g. wrong path)."""
    skills = _all_skill_md()
    assert skills, f"expected at least one SKILL.md in {SKILLS_ROOT}"
