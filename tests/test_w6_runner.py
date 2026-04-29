"""Runner pipeline tests — exercise build_trainset / make_metric / evaluate_module
without making real LLM calls. The actual ``run_gepa`` is mocked so we can
assert the orchestration around it."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import dspy
import pytest

import offerguide
from offerguide.evolution import (
    GoldenExample,
    build_trainset,
    make_score_match_metric,
)
from offerguide.evolution.runner import evaluate_module
from offerguide.skills import load_skill

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def _example(**kwargs) -> GoldenExample:
    base = {
        "name": "stub",
        "band": "fit",
        "job_text": "JD goes here",
        "user_profile": "profile here",
        "expected_probability_range": (0.5, 0.7),
        "must_mention": (),
        "must_not_mention": (),
    }
    base.update(kwargs)
    return GoldenExample(**base)


def test_build_trainset_attaches_golden_name_and_inputs() -> None:
    examples = [
        _example(name="ex1", job_text="JD-A", user_profile="P-A"),
        _example(name="ex2", job_text="JD-B", user_profile="P-B"),
    ]
    out = build_trainset(examples, input_names=["job_text", "user_profile"])
    assert len(out) == 2
    assert out[0].job_text == "JD-A"
    assert out[0]._golden_name == "ex1"
    assert set(out[0].inputs().keys()) == {"job_text", "user_profile"}


def test_make_score_match_metric_returns_prediction_with_score_and_feedback() -> None:
    examples = [_example(name="ex1", expected_probability_range=(0.5, 0.7))]
    metric = make_score_match_metric(examples)

    fake_example = dspy.Example(
        job_text="x", user_profile="y", _golden_name="ex1"
    ).with_inputs("job_text", "user_profile")
    fake_pred = dspy.Prediction(
        response_json='{"probability": 0.6, "reasoning": "ok", "dimensions": {}}'
    )
    result = metric(fake_example, fake_pred)
    assert isinstance(result, dspy.Prediction)
    assert result.score == pytest.approx(1.0)
    assert "ex1" in result.feedback


def test_make_score_match_metric_handles_unknown_example() -> None:
    metric = make_score_match_metric([])
    bad_ex = dspy.Example(job_text="x", _golden_name="ghost").with_inputs("job_text")
    pred = dspy.Prediction(response_json="{}")
    result = metric(bad_ex, pred)
    assert result.score == 0.0
    assert "missing _golden_name" in result.feedback or "_golden_name" in result.feedback


class _StubStudent(dspy.Module):
    """Returns a canned Prediction without touching any LM."""

    def __init__(self, response: str) -> None:
        super().__init__()
        self._response = response

    def forward(self, **_inputs: Any) -> dspy.Prediction:
        return dspy.Prediction(response_json=self._response)


def test_evaluate_module_aggregates_per_axis() -> None:
    examples = [
        _example(
            name="a",
            expected_probability_range=(0.6, 0.8),
            must_mention=("PyTorch",),
        ),
        _example(
            name="b",
            expected_probability_range=(0.0, 0.1),
            must_mention=("不符",),
        ),
    ]
    canned = '{"probability": 0.7, "reasoning": "用户熟悉 PyTorch", "dimensions": {}}'
    student = _StubStudent(canned)
    agg = evaluate_module(
        student=student, examples=examples, input_names=["job_text", "user_profile"]
    )
    # Example a: prob 0.7 is in [0.6, 0.8] → 1.0; PyTorch mentioned → 1.0
    # Example b: prob 0.7 outside [0.0, 0.1] by 0.6 → max(0, 1-4*0.6) = 0; "不符" not in reasoning → 0
    # Aggregate per-axis means
    assert 0.0 < agg["total"] < 1.0
    assert agg["n"] == 2


def test_evaluate_module_treats_forward_exception_as_failure() -> None:
    examples = [_example(name="x")]

    class _Boom(dspy.Module):
        def forward(self, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated network failure")

    agg = evaluate_module(
        student=_Boom(), examples=examples, input_names=["job_text", "user_profile"]
    )
    # Forward crashed → unparseable output → score 0
    assert agg["total"] == 0.0
    assert agg["n"] == 1


def test_evolve_skill_end_to_end_with_mocked_gepa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mock dspy.GEPA to bypass the LLM but exercise everything else.

    The mocked GEPA returns the student unchanged but with new instructions
    set, simulating a successful evolution. The test asserts:
    - SKILL.md is overwritten with the new instructions
    - parent_version + evolved_at are set
    - evolution_log row is created with metric_before/metric_after
    - backup file is written
    """
    src = SKILLS_ROOT / "score_match"
    dst_root = tmp_path / "skills"
    dst_root.mkdir()
    skill_dir = dst_root / "score_match"
    shutil.copytree(src, skill_dir)
    parent = load_skill(skill_dir)

    store = offerguide.Store(tmp_path / "evo.db")
    store.init_schema()

    # Stub LLMs — they never get called because GEPA itself is mocked
    fake_main_lm = object()
    fake_reflection_lm = object()

    # Mock evaluate_module to return deterministic before/after metrics
    from offerguide.evolution import runner as runner_mod

    eval_calls: list[dspy.Module] = []

    def fake_evaluate(*, student, examples, input_names):
        eval_calls.append(student)
        # First call (baseline) returns lower; second call (post) returns higher
        is_first = len(eval_calls) == 1
        return {
            "total": 0.50 if is_first else 0.72,
            "prob": 0.5 if is_first else 0.7,
            "recall": 0.5 if is_first else 0.8,
            "anti": 0.5 if is_first else 0.6,
            "n": 4,
        }

    monkeypatch.setattr(runner_mod, "evaluate_module", fake_evaluate)

    # Mock dspy.configure (avoid having to wire a real LM)
    monkeypatch.setattr(dspy, "configure", lambda **_kw: None)

    # Mock run_gepa: return the student with a fresh instructions string
    def fake_run_gepa(*, student, **_kwargs):
        student.predict.signature = student.predict.signature.with_instructions(
            "EVOLVED PROMPT BODY (mocked)"
        )
        return student

    monkeypatch.setattr(runner_mod, "run_gepa", fake_run_gepa)

    result = runner_mod.evolve_skill(
        skill_dir,
        store=store,
        main_lm=fake_main_lm,  # type: ignore[arg-type]
        reflection_lm=fake_reflection_lm,  # type: ignore[arg-type]
        auto="light",
    )

    # Result fields
    assert result.skill_name == "score_match"
    assert result.parent_version == parent.version
    assert result.new_version != parent.version
    assert result.metric_before["total"] == pytest.approx(0.50)
    assert result.metric_after["total"] == pytest.approx(0.72)

    # SKILL.md was overwritten
    reloaded = load_skill(skill_dir)
    assert "EVOLVED PROMPT BODY (mocked)" in reloaded.body
    assert reloaded.parent_version == parent.version
    assert reloaded.evolved_at is not None

    # Backup file exists
    backup = skill_dir / f"SKILL.md.v{parent.version}.bak"
    assert backup.exists()

    # evolution_log row
    with store.connect() as conn:
        row = conn.execute(
            "SELECT skill_name, parent_version, new_version, metric_before, metric_after "
            "FROM evolution_log WHERE id = ?",
            (result.evolution_log_id,),
        ).fetchone()
    assert row[0] == "score_match"
    assert row[1] == parent.version
    assert row[3] == pytest.approx(0.50)
    assert row[4] == pytest.approx(0.72)
