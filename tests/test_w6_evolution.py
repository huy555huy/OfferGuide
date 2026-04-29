"""W6 GEPA evolution — golden trainset, metric, SKILL writeback, log row.

Skips the actual GEPA optimizer (it needs a live LLM). The runner is split so
``run_gepa`` is mockable; the surrounding pipeline is exercised here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import offerguide
from offerguide.evolution import (
    GOLDEN_EXAMPLES,
    GoldenExample,
    aggregate,
    bump_version,
    log_evolution,
    score_match_metric,
    split_train_val,
    write_evolved_skill,
)
from offerguide.evolution.metrics import MetricBreakdown
from offerguide.skills import load_skill

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ---------- Golden trainset structure ---------------------------------


def test_golden_examples_have_valid_probability_ranges() -> None:
    for ex in GOLDEN_EXAMPLES:
        low, high = ex.expected_probability_range
        assert 0.0 <= low <= high <= 1.0, f"{ex.name}: range out of [0,1]"
        assert high - low <= 0.40, f"{ex.name}: range too wide ({low}-{high})"


def test_golden_examples_have_unique_names() -> None:
    names = [ex.name for ex in GOLDEN_EXAMPLES]
    assert len(names) == len(set(names))


def test_golden_examples_span_all_three_bands() -> None:
    bands = {ex.band for ex in GOLDEN_EXAMPLES}
    assert bands == {"fit", "misfit", "middle"}


def test_split_train_val_is_deterministic() -> None:
    a_train, a_val = split_train_val(seed=42)
    b_train, b_val = split_train_val(seed=42)
    assert [e.name for e in a_train] == [e.name for e in b_train]
    assert [e.name for e in a_val] == [e.name for e in b_val]


def test_split_train_val_balances_bands() -> None:
    """Each split should contain at least one example of each band."""
    train, val = split_train_val()
    assert {e.band for e in train} == {"fit", "misfit", "middle"}
    assert {e.band for e in val} == {"fit", "misfit", "middle"}


# ---------- Metric ---------------------------------------------------


def _golden(**overrides) -> GoldenExample:
    base = {
        "name": "test",
        "band": "middle",
        "job_text": "...",
        "user_profile": "...",
        "expected_probability_range": (0.4, 0.6),
        "must_mention": (),
        "must_not_mention": (),
    }
    base.update(overrides)
    return GoldenExample(**base)


def test_metric_full_score_when_in_band_and_recall_perfect() -> None:
    g = _golden(
        expected_probability_range=(0.4, 0.6),
        must_mention=("Python",),
    )
    pred = {
        "probability": 0.5,
        "reasoning": "用户精通 Python",
        "dimensions": {"tech": 0.7, "exp": 0.5, "company_tier": 0.6},
        "deal_breakers": [],
    }
    m = score_match_metric(g, pred)
    assert m.total == pytest.approx(1.0)
    assert m.prob_score == 1.0
    assert m.recall_score == 1.0
    assert m.anti_score == 1.0


def test_metric_probability_outside_band_linear_penalty() -> None:
    g = _golden(expected_probability_range=(0.6, 0.8))
    pred = {"probability": 0.4, "reasoning": "x", "dimensions": {}}
    m = score_match_metric(g, pred)
    # 0.4 is 0.2 below the band → score = max(0, 1 - 4*0.2) = 0.2
    assert m.prob_score == pytest.approx(0.2, abs=1e-6)


def test_metric_zero_score_for_unparseable_output() -> None:
    g = _golden()
    m = score_match_metric(g, "this is not json")
    assert m.total == 0.0
    assert "OUTPUT_PARSE_FAILURE" in m.feedback


def test_metric_zero_score_for_invalid_probability() -> None:
    g = _golden()
    m = score_match_metric(g, {"probability": "not a number", "reasoning": "x"})
    assert m.prob_score == 0.0


def test_metric_recall_partial_credit_when_some_mentions_hit() -> None:
    g = _golden(must_mention=("Python", "PyTorch", "Java"))
    pred = {
        "probability": 0.5,  # in band
        "reasoning": "熟悉 Python 和 PyTorch，但不会 Java",  # all 3 mentioned
        "dimensions": {},
    }
    m = score_match_metric(g, pred)
    assert m.recall_score == 1.0

    pred2 = {**pred, "reasoning": "熟悉 Python，PyTorch 没用过"}  # 2 of 3
    m2 = score_match_metric(g, pred2)
    assert m2.recall_score == pytest.approx(2 / 3)


def test_metric_anti_false_positive_penalty() -> None:
    g = _golden(must_not_mention=("强匹配", "高度契合"))
    pred = {
        "probability": 0.5,
        "reasoning": "整体属于强匹配，技术高度契合",  # both forbidden words appear
        "dimensions": {},
    }
    m = score_match_metric(g, pred)
    # 2 forbidden hits → 1 - 2*0.5 = 0.0
    assert m.anti_score == 0.0


def test_metric_uses_deal_breakers_for_recall() -> None:
    """When the must_mention term appears in deal_breakers (not reasoning), still counts."""
    g = _golden(must_mention=("5 年",))
    pred = {
        "probability": 0.05,
        "reasoning": "硬性年限不符",
        "dimensions": {"tech": 0.1, "exp": 0.05, "company_tier": 0.5},
        "deal_breakers": ["JD 要求 5 年经验，应届不符"],
    }
    m = score_match_metric(g, pred)
    assert m.recall_score == 1.0


def test_metric_breakdown_feedback_is_human_readable() -> None:
    g = _golden(name="ali_ai", expected_probability_range=(0.6, 0.8), must_mention=("Agent",))
    pred = {"probability": 0.3, "reasoning": "缺乏 Agent 经验"}
    m = score_match_metric(g, pred)
    assert "ali_ai" in m.feedback
    # Should mention the prob is "偏低于"
    assert "偏低" in m.feedback


def test_aggregate_means_per_axis() -> None:
    bs = [
        MetricBreakdown(
            total=0.8, prob_score=1.0, recall_score=0.6, anti_score=1.0, feedback=""
        ),
        MetricBreakdown(
            total=0.6, prob_score=0.5, recall_score=0.8, anti_score=0.5, feedback=""
        ),
    ]
    agg = aggregate(bs)
    assert agg["total"] == pytest.approx(0.7)
    assert agg["prob"] == pytest.approx(0.75)
    assert agg["recall"] == pytest.approx(0.7)
    assert agg["anti"] == pytest.approx(0.75)
    assert agg["n"] == 2


def test_aggregate_handles_empty_list() -> None:
    agg = aggregate([])
    assert agg["n"] == 0
    assert agg["total"] == 0.0


# ---------- Version bump ---------------------------------------------


def test_bump_version_semver() -> None:
    assert bump_version("0.2.0") == "0.2.1"
    assert bump_version("1.0.0") == "1.0.1"
    assert bump_version("10.5.99") == "10.5.100"


def test_bump_version_non_semver_appends_evolved() -> None:
    assert bump_version("alpha") == "alpha.evolved"
    assert bump_version("0.2") == "0.2.evolved"


# ---------- SKILL writeback ------------------------------------------


def test_write_evolved_skill_overwrites_with_new_version_and_parent(tmp_path: Path) -> None:
    # Copy score_match SKILL into a fresh tmp dir
    src = SKILLS_ROOT / "score_match"
    dst = tmp_path / "score_match"
    shutil.copytree(src, dst)

    parent = load_skill(dst)
    new_body = "你是测试 prompt v999"
    written = write_evolved_skill(
        dst, parent_spec=parent, new_instructions=new_body, new_version="9.9.9"
    )
    assert written == dst / "SKILL.md"

    reloaded = load_skill(dst)
    assert reloaded.version == "9.9.9"
    assert reloaded.parent_version == parent.version
    assert reloaded.evolved_at is not None
    assert reloaded.body == new_body
    # Original spec-required fields preserved
    assert reloaded.name == "score_match"
    assert reloaded.inputs == ("job_text", "user_profile")


def test_write_evolved_skill_writes_backup_by_default(tmp_path: Path) -> None:
    src = SKILLS_ROOT / "score_match"
    dst = tmp_path / "score_match"
    shutil.copytree(src, dst)
    parent = load_skill(dst)

    write_evolved_skill(dst, parent_spec=parent, new_instructions="new")

    backup = dst / f"SKILL.md.v{parent.version}.bak"
    assert backup.exists()
    # Backup contains the original version line, NOT the new one
    backup_text = backup.read_text(encoding="utf-8")
    assert f"version: {parent.version}" in backup_text


def test_write_evolved_skill_no_backup_when_disabled(tmp_path: Path) -> None:
    src = SKILLS_ROOT / "score_match"
    dst = tmp_path / "score_match"
    shutil.copytree(src, dst)
    parent = load_skill(dst)
    write_evolved_skill(
        dst, parent_spec=parent, new_instructions="new", backup=False
    )
    assert not (dst / f"SKILL.md.v{parent.version}.bak").exists()


def test_write_evolved_skill_idempotent_within_round(tmp_path: Path) -> None:
    """Calling twice should still leave a coherent SKILL.md (no double frontmatter)."""
    src = SKILLS_ROOT / "score_match"
    dst = tmp_path / "score_match"
    shutil.copytree(src, dst)
    parent = load_skill(dst)
    write_evolved_skill(dst, parent_spec=parent, new_instructions="round1")

    # Second call uses the now-mutated parent_spec
    parent2 = load_skill(dst)
    write_evolved_skill(dst, parent_spec=parent2, new_instructions="round2")

    final = load_skill(dst)
    assert final.body == "round2"
    text = (dst / "SKILL.md").read_text(encoding="utf-8")
    # Exactly two `---` delimiters (one open, one close)
    assert text.count("\n---\n") + text.count("---\n") <= 3


# ---------- evolution_log ---------------------------------------------


def test_log_evolution_persists_metric_delta(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "evo.db")
    store.init_schema()

    log_id = log_evolution(
        store,
        skill_name="score_match",
        parent_version="0.2.0",
        new_version="0.2.1",
        metric_before={"total": 0.50, "prob": 0.5, "recall": 0.5, "anti": 0.5},
        metric_after={"total": 0.62, "prob": 0.6, "recall": 0.6, "anti": 0.7},
        notes="auto=light",
    )
    assert log_id > 0

    with store.connect() as conn:
        row = conn.execute(
            "SELECT skill_name, parent_version, new_version, metric_before, metric_after, notes "
            "FROM evolution_log WHERE id = ?",
            (log_id,),
        ).fetchone()
    name, parent, new, mb, ma, notes = row
    assert name == "score_match"
    assert parent == "0.2.0"
    assert new == "0.2.1"
    assert mb == pytest.approx(0.50)
    assert ma == pytest.approx(0.62)
    notes_payload = json.loads(notes)
    assert notes_payload["delta_total"] == pytest.approx(0.12, abs=1e-6)
    assert notes_payload["extra"] == "auto=light"
