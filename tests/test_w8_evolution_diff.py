"""W8 — Evolution diff tests.

Exercises:
- evolution_log row → EvolutionRecord parsing (notes JSON unmarshal)
- SKILL.md frontmatter stripping → just the body
- Unified diff generation between parent .bak and current SKILL.md
- Markdown rendering for README/blog paste
- CLI subcommand: `python -m offerguide.evolution diff <skill>`
- Graceful handling when there's no evolution_log entry yet
- Graceful handling when the .bak file is missing
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import offerguide
from offerguide.evolution.diff import (
    EvolutionRecord,
    build_diff_report,
    extract_body,
    latest_evolution,
    list_evolutions,
    render_markdown,
)

# ── fixtures ───────────────────────────────────────────────────────


def _make_store(tmp_path: Path) -> offerguide.Store:
    store = offerguide.Store(tmp_path / "evo.db")
    store.init_schema()
    return store


def _insert_evolution(
    store: offerguide.Store,
    *,
    skill_name: str,
    parent_version: str,
    new_version: str,
    metric_before: float,
    metric_after: float,
    breakdown_before: dict | None = None,
    breakdown_after: dict | None = None,
    extra: str = "",
) -> int:
    notes = json.dumps(
        {
            "metric_before": breakdown_before or {},
            "metric_after": breakdown_after or {},
            "delta_total": metric_after - metric_before,
            "extra": extra,
        }
    )
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO evolution_log(skill_name, parent_version, new_version, "
            "metric_name, metric_before, metric_after, notes) "
            "VALUES (?, ?, ?, 'score_match_total', ?, ?, ?)",
            (skill_name, parent_version, new_version, metric_before, metric_after, notes),
        )
    return int(cur.lastrowid or 0)


def _make_skill_dir(
    tmp_path: Path,
    *,
    skill_name: str,
    parent_version: str,
    parent_body: str,
    current_version: str,
    current_body: str,
) -> Path:
    """Create a fake skill directory with both SKILL.md and SKILL.md.v<parent>.bak."""
    skill_dir = tmp_path / "skills" / skill_name
    skill_dir.mkdir(parents=True)

    parent_md = (
        f"---\nname: {skill_name}\nversion: {parent_version}\nparent_version: null\n---\n"
        f"{parent_body}\n"
    )
    current_md = (
        f"---\nname: {skill_name}\nversion: {current_version}\n"
        f"parent_version: {parent_version}\nevolved_at: '2026-04-29T12:00:00+00:00'\n---\n"
        f"{current_body}\n"
    )
    (skill_dir / f"SKILL.md.v{parent_version}.bak").write_text(parent_md)
    (skill_dir / "SKILL.md").write_text(current_md)
    return skill_dir.parent  # skills root


# ═══════════════════════════════════════════════════════════════════
# FRONTMATTER STRIPPING
# ═══════════════════════════════════════════════════════════════════


class TestExtractBody:
    def test_strips_frontmatter(self) -> None:
        text = "---\nname: x\nversion: 1.0.0\n---\n\nThis is the body.\n"
        assert extract_body(text).strip() == "This is the body."

    def test_returns_full_text_when_no_frontmatter(self) -> None:
        text = "Just plain text, no frontmatter.\n"
        assert extract_body(text) == text

    def test_handles_empty_string(self) -> None:
        assert extract_body("") == ""

    def test_unclosed_frontmatter_returns_full(self) -> None:
        text = "---\nname: x\nno closing delimiter\n"
        assert extract_body(text) == text


# ═══════════════════════════════════════════════════════════════════
# DB LOOKUP
# ═══════════════════════════════════════════════════════════════════


class TestLatestEvolution:
    def test_returns_none_when_no_rows(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert latest_evolution(store, "score_match") is None

    def test_returns_most_recent(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _insert_evolution(
            store,
            skill_name="score_match",
            parent_version="0.2.0",
            new_version="0.2.1",
            metric_before=0.50,
            metric_after=0.65,
            extra="auto=light",
        )
        rec = latest_evolution(store, "score_match")
        assert rec is not None
        assert isinstance(rec, EvolutionRecord)
        assert rec.parent_version == "0.2.0"
        assert rec.new_version == "0.2.1"
        assert rec.metric_before_total == pytest.approx(0.50)
        assert rec.metric_after_total == pytest.approx(0.65)
        assert rec.delta_total == pytest.approx(0.15)
        assert rec.notes_extra == "auto=light"

    def test_picks_latest_when_multiple(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _insert_evolution(
            store, skill_name="score_match",
            parent_version="0.2.0", new_version="0.2.1",
            metric_before=0.50, metric_after=0.55,
        )
        _insert_evolution(
            store, skill_name="score_match",
            parent_version="0.2.1", new_version="0.2.2",
            metric_before=0.55, metric_after=0.72,
        )
        rec = latest_evolution(store, "score_match")
        assert rec is not None
        assert rec.parent_version == "0.2.1"
        assert rec.new_version == "0.2.2"

    def test_filters_by_skill_name(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _insert_evolution(
            store, skill_name="score_match",
            parent_version="0.2.0", new_version="0.2.1",
            metric_before=0.5, metric_after=0.6,
        )
        _insert_evolution(
            store, skill_name="analyze_gaps",
            parent_version="0.1.0", new_version="0.1.1",
            metric_before=0.4, metric_after=0.5,
        )
        rec = latest_evolution(store, "score_match")
        assert rec is not None
        assert rec.skill_name == "score_match"

    def test_breakdown_parsed_from_notes(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _insert_evolution(
            store,
            skill_name="score_match",
            parent_version="0.2.0",
            new_version="0.2.1",
            metric_before=0.50,
            metric_after=0.72,
            breakdown_before={"total": 0.50, "prob": 0.5, "recall": 0.5, "anti": 0.5},
            breakdown_after={"total": 0.72, "prob": 0.7, "recall": 0.8, "anti": 0.6},
        )
        rec = latest_evolution(store, "score_match")
        assert rec is not None
        assert rec.metric_breakdown_before["recall"] == pytest.approx(0.5)
        assert rec.metric_breakdown_after["recall"] == pytest.approx(0.8)


class TestListEvolutions:
    def test_returns_empty_list_when_none(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert list_evolutions(store, "score_match") == []

    def test_returns_all_newest_first(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _insert_evolution(
            store, skill_name="score_match",
            parent_version="0.2.0", new_version="0.2.1",
            metric_before=0.5, metric_after=0.6,
        )
        _insert_evolution(
            store, skill_name="score_match",
            parent_version="0.2.1", new_version="0.2.2",
            metric_before=0.6, metric_after=0.7,
        )
        rows = list_evolutions(store, "score_match")
        assert len(rows) == 2
        assert rows[0].new_version == "0.2.2"
        assert rows[1].new_version == "0.2.1"


# ═══════════════════════════════════════════════════════════════════
# DIFF REPORT
# ═══════════════════════════════════════════════════════════════════


class TestBuildDiffReport:
    def test_returns_none_when_no_evolution(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        (skills_root / "score_match").mkdir()
        (skills_root / "score_match" / "SKILL.md").write_text("---\nname: x\n---\nbody")

        report = build_diff_report(store, skills_root, "score_match")
        assert report is None

    def test_full_report_with_backup_present(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills_root = _make_skill_dir(
            tmp_path,
            skill_name="score_match",
            parent_version="0.2.0",
            parent_body="ORIGINAL prompt body line 1.\nORIGINAL line 2.",
            current_version="0.2.1",
            current_body="EVOLVED prompt body line 1.\nORIGINAL line 2.",
        )
        _insert_evolution(
            store,
            skill_name="score_match",
            parent_version="0.2.0",
            new_version="0.2.1",
            metric_before=0.50,
            metric_after=0.72,
            breakdown_before={"total": 0.50, "prob": 0.5, "recall": 0.5, "anti": 0.5},
            breakdown_after={"total": 0.72, "prob": 0.7, "recall": 0.8, "anti": 0.6},
        )

        report = build_diff_report(store, skills_root, "score_match")
        assert report is not None
        assert report.parent_available is True
        assert "ORIGINAL prompt body line 1" in report.parent_body
        assert "EVOLVED prompt body line 1" in report.current_body
        assert "ORIGINAL prompt body line 1." in report.unified_diff
        assert "EVOLVED prompt body line 1." in report.unified_diff

    def test_handles_missing_bak_gracefully(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "score_match"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: score_match\nversion: 0.2.1\nparent_version: 0.2.0\n---\nevolved body\n"
        )
        # NOTE: no .bak file
        _insert_evolution(
            store,
            skill_name="score_match",
            parent_version="0.2.0",
            new_version="0.2.1",
            metric_before=0.5,
            metric_after=0.6,
        )

        report = build_diff_report(store, skills_root, "score_match")
        assert report is not None
        assert report.parent_available is False
        assert report.unified_diff == ""
        assert report.parent_body == ""


# ═══════════════════════════════════════════════════════════════════
# MARKDOWN RENDERING
# ═══════════════════════════════════════════════════════════════════


class TestRenderMarkdown:
    def test_renders_metric_delta(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills_root = _make_skill_dir(
            tmp_path,
            skill_name="score_match",
            parent_version="0.2.0",
            parent_body="OLD body",
            current_version="0.2.1",
            current_body="NEW body",
        )
        _insert_evolution(
            store, skill_name="score_match",
            parent_version="0.2.0", new_version="0.2.1",
            metric_before=0.50, metric_after=0.72,
            breakdown_before={"total": 0.50, "prob": 0.5, "recall": 0.5, "anti": 0.5},
            breakdown_after={"total": 0.72, "prob": 0.7, "recall": 0.8, "anti": 0.6},
        )
        report = build_diff_report(store, skills_root, "score_match")
        assert report is not None

        md = render_markdown(report)
        assert "score_match" in md
        assert "0.2.0" in md
        assert "0.2.1" in md
        # Delta present
        assert "+0.220" in md or "0.220" in md
        # Per-axis breakdown table
        assert "| recall |" in md
        # Diff block
        assert "```diff" in md
        assert "OLD body" in md or "NEW body" in md

    def test_renders_warning_when_bak_missing(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills_root = tmp_path / "skills"
        skill_dir = skills_root / "score_match"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: score_match\nversion: 0.2.1\nparent_version: 0.2.0\n---\nevolved body\n"
        )
        _insert_evolution(
            store, skill_name="score_match",
            parent_version="0.2.0", new_version="0.2.1",
            metric_before=0.5, metric_after=0.6,
        )

        report = build_diff_report(store, skills_root, "score_match")
        assert report is not None
        md = render_markdown(report)
        assert "Parent backup" in md or "not found" in md


# ═══════════════════════════════════════════════════════════════════
# CLI SUBCOMMAND
# ═══════════════════════════════════════════════════════════════════


class TestDiffCLI:
    def test_diff_cli_returns_1_when_no_evolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from offerguide.evolution.cli import main

        # Point the CLI at a fresh DB with no evolution_log entries
        monkeypatch.setenv("OFFERGUIDE_DB", str(tmp_path / "empty.db"))
        # Use the real score_match skill directory — only the DB is empty
        rc = main(["diff", "score_match"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "No evolution_log" in captured.err

    def test_diff_cli_returns_2_for_unknown_skill(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        from offerguide.evolution.cli import main

        monkeypatch.setenv("OFFERGUIDE_DB", str(tmp_path / "x.db"))
        rc = main(["diff", "ghost_skill_does_not_exist"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "no SKILL directory" in captured.err
