"""W13.7 cost tracking + circuit breaker + held-out eval framework."""

from __future__ import annotations

from pathlib import Path

import pytest

import offerguide
from offerguide.eval.runner import (
    EvalCase,
    _score_case,
    load_cases,
)
from offerguide.llm import LLMResponse
from offerguide.llm.pricing import estimate_cost_usd, is_known_model
from offerguide.skills import SkillRuntime, SkillSpec

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# Pricing
# ═══════════════════════════════════════════════════════════════════


class TestPricing:
    def test_known_model_priced(self):
        cost = estimate_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=1000,
            completion_tokens=200,
        )
        # 1000 * 3.00 / 1M + 200 * 15.00 / 1M = 0.003 + 0.003 = 0.006
        assert cost == pytest.approx(0.006, abs=0.001)

    def test_deepseek_cheaper_than_claude(self):
        cost_ds = estimate_cost_usd(
            model="deepseek-v4-flash",
            prompt_tokens=10000,
            completion_tokens=5000,
        )
        cost_cl = estimate_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=10000,
            completion_tokens=5000,
        )
        assert cost_ds < cost_cl

    def test_unknown_model_uses_fallback(self):
        cost = estimate_cost_usd(
            model="nonexistent-model-xyz",
            prompt_tokens=1000,
            completion_tokens=200,
        )
        # Falls back to mid-range estimate (claude-sonnet-tier)
        assert cost > 0
        assert not is_known_model("nonexistent-model-xyz")

    def test_prefix_match_for_dated_variants(self):
        """claude-sonnet-4-6-20251022 should match claude-sonnet-4-6 base."""
        cost = estimate_cost_usd(
            model="claude-sonnet-4-6-20251022",
            prompt_tokens=1000,
            completion_tokens=200,
        )
        expected = estimate_cost_usd(
            model="claude-sonnet-4-6",
            prompt_tokens=1000,
            completion_tokens=200,
        )
        assert cost == pytest.approx(expected, abs=0.0001)

    def test_zero_tokens_zero_cost(self):
        assert (
            estimate_cost_usd(
                model="claude-sonnet-4-6",
                prompt_tokens=0,
                completion_tokens=0,
            )
            == 0
        )


# ═══════════════════════════════════════════════════════════════════
# Cost flows through skill_runs
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "cost.db")
    s.init_schema()
    return s


class _CostStubLLM:
    """Returns LLMResponse with explicit cost_usd set."""

    def __init__(self, cost: float = 0.05):
        self._cost = cost

    def chat(self, messages, **kw):
        return LLMResponse(
            content='{"probability": 0.7, "reasoning": "ok"}',
            model="claude-sonnet-4-6",
            prompt_tokens=1000,
            completion_tokens=200,
            cost_usd=self._cost,
        )


class TestCostPersistence:
    def test_skill_run_records_cost(self, store):
        seed_spec = SkillSpec(
            name="test_cost",
            description="d",
            version="0.1.0",
            body="b",
            inputs=("x",),
        )
        rt = SkillRuntime(llm=_CostStubLLM(cost=0.05), store=store)
        result = rt.invoke(seed_spec, {"x": "y"})
        assert result.cost_usd == pytest.approx(0.05)

        with store.connect() as conn:
            row = conn.execute(
                "SELECT cost_usd FROM skill_runs WHERE id = ?",
                (result.skill_run_id,),
            ).fetchone()
        assert row[0] == pytest.approx(0.05)

    def test_zero_cost_when_stub_absent(self, store):
        """Old stub-style LLMs without cost_usd should not crash; record 0."""

        class _OldStub:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")  # no cost set

        seed_spec = SkillSpec(
            name="test",
            description="d",
            version="0.1.0",
            body="b",
            inputs=(),
        )
        rt = SkillRuntime(llm=_OldStub(), store=store)
        result = rt.invoke(seed_spec, {})
        assert result.cost_usd == 0.0


# ═══════════════════════════════════════════════════════════════════
# Eval runner
# ═══════════════════════════════════════════════════════════════════


class TestEvalScoring:
    def test_passes_when_all_assertions_satisfied(self):
        case = EvalCase(
            name="ok",
            inputs={},
            expected={
                "json_valid": True,
                "must_contain_keys": ["a"],
                "must_contain_phrases_in_output": ["foo"],
                "score_range": {"a": [0.0, 1.0]},
            },
        )
        failures = _score_case(case, parsed={"a": 0.5}, raw_text="this has foo in it")
        assert failures == []

    def test_fails_missing_key(self):
        case = EvalCase(
            name="x", inputs={}, expected={"json_valid": True, "must_contain_keys": ["xxx"]}
        )
        failures = _score_case(case, parsed={"yyy": 1}, raw_text="")
        assert any("xxx" in f for f in failures)

    def test_fails_missing_phrase(self):
        case = EvalCase(name="x", inputs={}, expected={"must_contain_phrases_in_output": ["LoRA"]})
        failures = _score_case(case, parsed=None, raw_text="lots of words but no L word")
        assert any("LoRA" in f for f in failures)

    def test_fails_forbidden_phrase(self):
        case = EvalCase(name="x", inputs={}, expected={"must_not_contain_phrases": ["编造"]})
        failures = _score_case(case, parsed=None, raw_text="建议编造一个项目")
        assert any("编造" in f for f in failures)

    def test_fails_score_range(self):
        case = EvalCase(name="x", inputs={}, expected={"score_range": {"prob": [0.4, 0.6]}})
        failures = _score_case(case, parsed={"prob": 0.9}, raw_text="")
        assert any("0.9" in f and "0.4" in f for f in failures)

    def test_score_range_passes_in_bounds(self):
        case = EvalCase(name="x", inputs={}, expected={"score_range": {"prob": [0.4, 0.6]}})
        failures = _score_case(case, parsed={"prob": 0.5}, raw_text="")
        assert failures == []


class TestEvalDatasets:
    def test_unknown_skill_returns_empty(self):
        assert load_cases("nonexistent_skill_xyz") == []


class TestRunEvalIntegration:
    def test_runner_executes_against_stub_and_aggregates(self, store):
        """Smoke: runner invokes SKILL through SkillRuntime + collects results."""
        seed_spec = SkillSpec(
            name="example_eval",
            description="d",
            version="0.1.0",
            body="output JSON with probability + reasoning",
            inputs=("job_text", "user_profile"),
        )
        rt = SkillRuntime(llm=_CostStubLLM(cost=0.01), store=store)
        # Minimal cases (don't actually rely on the on-disk dataset for this test)
        cases = [
            EvalCase(
                name="t1",
                inputs={"job_text": "x", "user_profile": "y"},
                expected={
                    "json_valid": True,
                    "must_contain_keys": ["probability"],
                },
            ),
        ]
        from offerguide.eval.runner import run_eval_for_skill

        report = run_eval_for_skill(runtime=rt, spec=seed_spec, cases=cases)
        assert report.total == 1
        assert report.passed == 1
        assert report.results[0].cost_usd == pytest.approx(0.01)

    def test_runner_marks_failure_when_assertion_fails(self, store):
        seed_spec = SkillSpec(
            name="x",
            description="d",
            version="0.1.0",
            body="b",
            inputs=("a",),
        )
        rt = SkillRuntime(llm=_CostStubLLM(cost=0.01), store=store)
        cases = [
            EvalCase(
                name="will_fail",
                inputs={"a": "x"},
                expected={
                    "must_contain_phrases_in_output": ["this string is not in output"],
                },
            ),
        ]
        from offerguide.eval.runner import run_eval_for_skill

        report = run_eval_for_skill(runtime=rt, spec=seed_spec, cases=cases)
        assert report.passed == 0
        assert report.results[0].failures
