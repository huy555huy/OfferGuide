"""Evolution diff — reads ``evolution_log`` + the ``SKILL.md.v<n>.bak``
backup file and produces a human-readable before/after report.

This is the "what did GEPA actually learn" deliverable for the README and
blog post — the on-disk evidence that the evolution loop is doing real work.

Data sources:
- ``evolution_log`` row holds skill_name, versions, metric_before/after,
  and a JSON ``notes`` blob with the full per-axis breakdown
- ``<skill_dir>/SKILL.md.v<parent_version>.bak`` holds the parent body
- ``<skill_dir>/SKILL.md`` holds the evolved body

Usage::

    python -m offerguide.evolution diff score_match
    python -m offerguide.evolution diff score_match --markdown > diff.md
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path

from ..memory import Store


@dataclass(frozen=True)
class EvolutionRecord:
    """One row of evolution_log, parsed into typed fields."""

    id: int
    skill_name: str
    parent_version: str | None
    new_version: str
    metric_name: str
    metric_before_total: float
    metric_after_total: float
    metric_breakdown_before: dict[str, float]
    """Per-axis baseline (prob/recall/anti/total/n) — empty if `notes` was empty."""

    metric_breakdown_after: dict[str, float]
    delta_total: float
    notes_extra: str
    created_at: float


@dataclass(frozen=True)
class DiffReport:
    record: EvolutionRecord
    parent_body: str
    current_body: str
    unified_diff: str
    """The git-style unified diff of the SKILL.md body. Empty string when
    parent body is unavailable (no .bak file)."""

    parent_available: bool


# ── DB lookup ──────────────────────────────────────────────────────


def latest_evolution(store: Store, skill_name: str) -> EvolutionRecord | None:
    """Most-recent evolution_log row for `skill_name`, or None if no evolutions."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, skill_name, parent_version, new_version, metric_name, "
            "metric_before, metric_after, notes, created_at "
            "FROM evolution_log WHERE skill_name = ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (skill_name,),
        ).fetchone()
    return _row_to_record(row) if row else None


def list_evolutions(store: Store, skill_name: str) -> list[EvolutionRecord]:
    """All evolution_log rows for `skill_name`, newest first."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, skill_name, parent_version, new_version, metric_name, "
            "metric_before, metric_after, notes, created_at "
            "FROM evolution_log WHERE skill_name = ? "
            "ORDER BY created_at DESC, id DESC",
            (skill_name,),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def _row_to_record(row: tuple) -> EvolutionRecord:
    notes_raw = row[7] or "{}"
    try:
        notes = json.loads(notes_raw)
    except json.JSONDecodeError:
        notes = {}
    return EvolutionRecord(
        id=int(row[0]),
        skill_name=row[1],
        parent_version=row[2],
        new_version=row[3],
        metric_name=row[4],
        metric_before_total=float(row[5] or 0.0),
        metric_after_total=float(row[6] or 0.0),
        metric_breakdown_before=dict(notes.get("metric_before", {})),
        metric_breakdown_after=dict(notes.get("metric_after", {})),
        delta_total=float(notes.get("delta_total",
                                    float(row[6] or 0.0) - float(row[5] or 0.0))),
        notes_extra=str(notes.get("extra", "")),
        created_at=float(row[8]),
    )


# ── SKILL.md body parsing ──────────────────────────────────────────


def extract_body(skill_md_text: str) -> str:
    """Strip the YAML frontmatter (between two ``---`` lines) and return just the body.

    Tolerant of files with no frontmatter: returns the full text in that case.
    """
    lines = skill_md_text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return skill_md_text
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[i + 1 :]).lstrip("\n")
    return skill_md_text  # unclosed frontmatter — return full text as fallback


# ── DiffReport construction ────────────────────────────────────────


def build_diff_report(
    store: Store,
    skills_root: Path,
    skill_name: str,
) -> DiffReport | None:
    """Assemble a DiffReport for the latest evolution of `skill_name`.

    Returns *None* if there's no evolution_log entry for the SKILL yet.
    The ``parent_available`` flag is False when the `.bak` file is missing
    (e.g., when the user evolved with ``--no-backup``); in that case
    ``unified_diff`` is empty.
    """
    record = latest_evolution(store, skill_name)
    if record is None:
        return None

    skill_dir = skills_root / skill_name
    current_md = skill_dir / "SKILL.md"
    if not current_md.exists():
        raise FileNotFoundError(f"current SKILL.md not found: {current_md}")
    current_body = extract_body(current_md.read_text(encoding="utf-8"))

    parent_body = ""
    parent_available = False
    if record.parent_version:
        bak = skill_dir / f"SKILL.md.v{record.parent_version}.bak"
        if bak.exists():
            parent_body = extract_body(bak.read_text(encoding="utf-8"))
            parent_available = True

    if parent_available:
        diff_lines = difflib.unified_diff(
            parent_body.splitlines(keepends=True),
            current_body.splitlines(keepends=True),
            fromfile=f"SKILL.md (parent v{record.parent_version})",
            tofile=f"SKILL.md (evolved v{record.new_version})",
            n=3,
        )
        unified_diff = "".join(diff_lines)
    else:
        unified_diff = ""

    return DiffReport(
        record=record,
        parent_body=parent_body,
        current_body=current_body,
        unified_diff=unified_diff,
        parent_available=parent_available,
    )


# ── Markdown rendering (for README / blog paste) ───────────────────


def render_markdown(report: DiffReport) -> str:
    """Render a DiffReport as a markdown document suitable for README/blog paste."""
    r = report.record
    lines: list[str] = []
    lines.append(f"# `{r.skill_name}` — GEPA Evolution Report")
    lines.append("")
    lines.append(f"- **Parent version**: `{r.parent_version or '(none)'}`")
    lines.append(f"- **Evolved version**: `{r.new_version}`")
    lines.append(f"- **Run**: evolution_log #{r.id}")
    if r.notes_extra:
        lines.append(f"- **Run config**: {r.notes_extra}")
    lines.append("")

    # Headline metric
    delta = r.delta_total
    arrow = "↑" if delta >= 0 else "↓"
    lines.append("## Metric — total")
    lines.append("")
    lines.append(
        f"| baseline | evolved | Δ |\n"
        f"|---|---|---|\n"
        f"| {r.metric_before_total:.3f} | {r.metric_after_total:.3f} "
        f"| **{arrow} {abs(delta):+.3f}** |"
    )
    lines.append("")

    # Per-axis breakdown if available
    if r.metric_breakdown_before and r.metric_breakdown_after:
        axes = sorted(
            set(r.metric_breakdown_before) | set(r.metric_breakdown_after)
        )
        lines.append("## Per-axis breakdown")
        lines.append("")
        lines.append("| axis | baseline | evolved | Δ |")
        lines.append("|---|---|---|---|")
        for ax in axes:
            b = r.metric_breakdown_before.get(ax, 0.0)
            a = r.metric_breakdown_after.get(ax, 0.0)
            d = a - b
            sign = "+" if d >= 0 else ""
            lines.append(f"| {ax} | {b:.3f} | {a:.3f} | {sign}{d:.3f} |")
        lines.append("")

    # Prompt diff
    lines.append("## Prompt body diff")
    lines.append("")
    if not report.parent_available:
        lines.append(
            f"_Parent backup `.bak` file not found "
            f"(SKILL.md.v{r.parent_version}.bak). "
            f"Re-run evolution without `--no-backup` to regenerate._"
        )
    elif not report.unified_diff:
        lines.append("_Parent and evolved bodies are identical._")
    else:
        lines.append("```diff")
        lines.append(report.unified_diff.rstrip())
        lines.append("```")

    return "\n".join(lines)
