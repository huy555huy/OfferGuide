"""W8' — Tests for the per-SKILL evolution adapters.

Covers analyze_gaps and prepare_interview adapters; the score_match
adapter is exercised by the existing W6 tests via backward-compat
shims (``offerguide.evolution.metrics.score_match_metric``).

Each adapter is tested for:
- Valid JSON that satisfies the schema scores > 0
- Invalid JSON / schema-violating JSON scores 0 across all axes
- Each scoring axis has at least one targeted test
- ``split_train_val`` is deterministic and stratified
"""

from __future__ import annotations

import json

import pytest

from offerguide.evolution.adapters import (
    analyze_gaps,
    get_adapter,
    list_evolvable_skills,
    prepare_interview,
    score_match,
)
from offerguide.evolution.adapters._base import MetricBreakdown, aggregate

# ═══════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════


class TestRegistry:
    def test_all_skills_registered(self) -> None:
        assert set(list_evolvable_skills()) == {
            "score_match",
            "analyze_gaps",
            "prepare_interview",
            "deep_project_prep",
            "compare_jobs",
        }

    def test_get_adapter_returns_module(self) -> None:
        assert get_adapter("score_match") is score_match
        assert get_adapter("analyze_gaps") is analyze_gaps
        assert get_adapter("prepare_interview") is prepare_interview

    def test_get_adapter_raises_for_unknown(self) -> None:
        with pytest.raises(KeyError, match="known"):
            get_adapter("ghost_skill")

    def test_each_adapter_has_required_attrs(self) -> None:
        for name in list_evolvable_skills():
            adapter = get_adapter(name)
            assert hasattr(adapter, "name")
            assert hasattr(adapter, "INPUT_NAMES")
            assert hasattr(adapter, "METRIC_AXES")
            assert hasattr(adapter, "EXAMPLES")
            assert hasattr(adapter, "metric")
            assert hasattr(adapter, "split_train_val")
            assert adapter.name == name
            assert len(adapter.EXAMPLES) > 0


# ═══════════════════════════════════════════════════════════════════
# ANALYZE_GAPS adapter
# ═══════════════════════════════════════════════════════════════════


def _good_gaps_output(
    *,
    keyword_gaps: list[dict] | None = None,
    suggestions: list[dict] | None = None,
) -> dict:
    return {
        "summary": "总体匹配中等，缺少 C/C++ 与 LangGraph",
        "keyword_gaps": keyword_gaps
        or [
            {
                "jd_keyword": "C/C++",
                "in_resume": False,
                "importance": "high",
                "evidence_in_jd": "扎实的 C/C++/Java（至少一门）",
            },
            {
                "jd_keyword": "LangGraph",
                "in_resume": False,
                "importance": "medium",
                "evidence_in_jd": "熟悉 LangChain / LangGraph 加分",
            },
        ],
        "suggestions": suggestions
        or [
            {
                "section": "技能",
                "action": "add",
                "current_text": None,
                "proposed_addition": "C++ (本科课程实践)",
                "reason": "JD 要求 C/C++/Java",
                "ai_risk": "low",
                "confidence": 0.6,
            },
            {
                "section": "项目经历",
                "action": "emphasize",
                "current_text": "Deep Research Agent",
                "proposed_addition": "强调对 LangGraph 类工具的快速上手能力",
                "reason": "JD 提到 LangGraph 加分项",
                "ai_risk": "medium",
                "confidence": 0.5,
            },
            {
                "section": "技能",
                "action": "add",
                "current_text": None,
                "proposed_addition": "Java (基础语法)",
                "reason": "JD 备选语言之一",
                "ai_risk": "low",
                "confidence": 0.4,
            },
        ],
        "do_not_add": [],
        "ai_detection_warnings": [],
    }


def _gaps_example(name: str = "ali_agent_intern_gaps"):
    return next(e for e in analyze_gaps.EXAMPLES if e.name == name)


class TestAnalyzeGapsMetric:
    def test_invalid_json_scores_zero(self) -> None:
        ex = _gaps_example()
        result = analyze_gaps.metric(ex, "not valid json {")
        assert result.total == 0.0
        assert all(v == 0.0 for v in result.breakdown.values())
        assert "OUTPUT_PARSE_FAILURE" in result.feedback

    def test_schema_violation_scores_zero(self) -> None:
        ex = _gaps_example()
        # Missing required `keyword_gaps` field → schema fails
        bad = {"summary": "x", "suggestions": [], "do_not_add": [], "ai_detection_warnings": []}
        result = analyze_gaps.metric(ex, json.dumps(bad))
        assert result.total == 0.0
        assert result.breakdown["schema"] == 0.0
        assert "schema 失败" in result.feedback

    def test_extra_keys_rejected(self) -> None:
        ex = _gaps_example()
        bad = _good_gaps_output()
        bad["secret_field"] = "should fail extra='forbid'"
        result = analyze_gaps.metric(ex, json.dumps(bad))
        assert result.total == 0.0
        assert result.breakdown["schema"] == 0.0

    def test_valid_output_scores_positive(self) -> None:
        ex = _gaps_example()
        out = _good_gaps_output()
        result = analyze_gaps.metric(ex, json.dumps(out))
        assert result.total > 0
        assert result.breakdown["schema"] == 1.0
        assert isinstance(result, MetricBreakdown)

    def test_keyword_recall_partial(self) -> None:
        ex = _gaps_example()
        # Only flag C/C++ — miss LangGraph, Java, etc.
        out = _good_gaps_output(
            keyword_gaps=[
                {
                    "jd_keyword": "C/C++",
                    "in_resume": False,
                    "importance": "high",
                    "evidence_in_jd": "扎实的 C/C++",
                }
            ],
        )
        result = analyze_gaps.metric(ex, json.dumps(out))
        # Expected keywords: C/C++, C++, Java, LangChain, LangGraph (5)
        # Hits: C/C++ and C++ (substring of "C/C++") → 2/5
        assert 0.3 < result.breakdown["keyword_recall"] < 0.5

    def test_keyword_recall_full(self) -> None:
        ex = _gaps_example()
        out = _good_gaps_output(
            keyword_gaps=[
                {"jd_keyword": k, "in_resume": False, "importance": "high", "evidence_in_jd": "x"}
                for k in ["C/C++", "C++", "Java", "LangChain", "LangGraph"]
            ],
        )
        result = analyze_gaps.metric(ex, json.dumps(out))
        assert result.breakdown["keyword_recall"] == pytest.approx(1.0)

    def test_count_in_range_optimal(self) -> None:
        ex = _gaps_example()
        # 3 suggestions is in [3, 8]
        result = analyze_gaps.metric(ex, json.dumps(_good_gaps_output()))
        assert result.breakdown["count"] == 1.0

    def test_count_too_few_penalized(self) -> None:
        ex = _gaps_example()
        out = _good_gaps_output(
            suggestions=[
                {
                    "section": "技能",
                    "action": "add",
                    "current_text": None,
                    "proposed_addition": "C++",
                    "reason": "JD 要求",
                    "ai_risk": "low",
                    "confidence": 0.5,
                }
            ],
        )
        result = analyze_gaps.metric(ex, json.dumps(out))
        # 1 suggestion < 3 min → penalized
        assert result.breakdown["count"] < 1.0

    def test_ai_risk_floor_satisfied(self) -> None:
        # The ats_buzzword_gaps example requires at least 1 high-risk suggestion
        ex = next(e for e in analyze_gaps.EXAMPLES if e.name == "ats_buzzword_gaps")
        good_with_high = _good_gaps_output()
        good_with_high["suggestions"][0]["ai_risk"] = "high"
        result = analyze_gaps.metric(ex, json.dumps(good_with_high))
        assert result.breakdown["ai_risk"] == 1.0

    def test_ai_risk_floor_violated(self) -> None:
        ex = next(e for e in analyze_gaps.EXAMPLES if e.name == "ats_buzzword_gaps")
        # No high-risk → fail floor
        result = analyze_gaps.metric(ex, json.dumps(_good_gaps_output()))
        assert result.breakdown["ai_risk"] == 0.0


class TestAnalyzeGapsSplit:
    def test_deterministic(self) -> None:
        a1, b1 = analyze_gaps.split_train_val(seed=42)
        a2, b2 = analyze_gaps.split_train_val(seed=42)
        assert [e.name for e in a1] == [e.name for e in a2]
        assert [e.name for e in b1] == [e.name for e in b2]

    def test_total_count_preserved(self) -> None:
        train, val = analyze_gaps.split_train_val()
        assert len(train) + len(val) == len(analyze_gaps.EXAMPLES)


# ═══════════════════════════════════════════════════════════════════
# PREPARE_INTERVIEW adapter
# ═══════════════════════════════════════════════════════════════════


def _good_prep_output(
    *,
    questions: list[dict] | None = None,
) -> dict:
    return {
        "company_snapshot": "字节跳动 Seed，主攻 LLM 后训练。JD 要求 PyTorch + RL。",
        "expected_questions": questions
        or [
            {
                "question": "讲讲 attention 缩放为什么除以 √d",
                "category": "technical",
                "likelihood": 0.85,
                "rationale": "JD 提到 Transformer 内部机制，面经里多次出现",
            },
            {
                "question": "讲一个跨团队协作的例子",
                "category": "behavioral",
                "likelihood": 0.55,
                "rationale": "校招通用 STAR 题；面试官常用此题筛选 PyTorch 项目协作经验",
            },
            {
                "question": "你做的 RemeDi loss 曲线长什么样",
                "category": "project_deep_dive",
                "likelihood": 0.75,
                "rationale": "项目深挖，简历提到 RemeDi 和 GRPO",
            },
            {
                "question": "如何调试 GRPO 训练不稳",
                "category": "technical",
                "likelihood": 0.60,
                "rationale": "JD 要求 RLHF 经验；用户简历用过 GRPO",
            },
            {
                "question": "DeepSpeed ZeRO-2 vs ZeRO-3 区别？",
                "category": "company_specific",
                "likelihood": 0.40,
                "rationale": "字节训练规模大，DeepSpeed/Megatron 必问",
            },
        ],
        "prep_focus_areas": ["Transformer 数学", "GRPO/PPO 调参", "项目讲述"],
        "weak_spots": ["Megatron 没用过", "DPO 不熟"],
    }


def _prep_example(name: str = "bytedance_seed_with_面经"):
    return next(e for e in prepare_interview.EXAMPLES if e.name == name)


class TestPrepareInterviewMetric:
    def test_invalid_json_scores_zero(self) -> None:
        ex = _prep_example()
        result = prepare_interview.metric(ex, "not valid json")
        assert result.total == 0.0
        assert all(v == 0.0 for v in result.breakdown.values())

    def test_schema_violation_scores_zero(self) -> None:
        ex = _prep_example()
        bad = {"company_snapshot": "x"}  # missing required fields
        result = prepare_interview.metric(ex, json.dumps(bad))
        assert result.total == 0.0
        assert result.breakdown["schema"] == 0.0

    def test_likelihood_out_of_range_rejected(self) -> None:
        ex = _prep_example()
        out = _good_prep_output()
        out["expected_questions"][0]["likelihood"] = 1.5  # > 1
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["schema"] == 0.0

    def test_valid_output_scores_high(self) -> None:
        ex = _prep_example()
        result = prepare_interview.metric(ex, json.dumps(_good_prep_output()))
        assert result.total > 0.5
        assert result.breakdown["schema"] == 1.0

    def test_coverage_full_when_3_plus_categories(self) -> None:
        ex = _prep_example()
        out = _good_prep_output()
        # Has technical, behavioral, project_deep_dive, company_specific = 4 categories
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["coverage"] == 1.0

    def test_coverage_low_when_only_one_category(self) -> None:
        ex = _prep_example()
        # All technical
        out = _good_prep_output(
            questions=[
                {
                    "question": f"Q{i}",
                    "category": "technical",
                    "likelihood": 0.5 + i * 0.05,
                    "rationale": "JD 要求 PyTorch",
                }
                for i in range(5)
            ],
        )
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["coverage"] < 0.5  # 1/3

    def test_count_in_range(self) -> None:
        ex = _prep_example()
        out = _good_prep_output()  # 5 questions ∈ [5,8]
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["count"] == 1.0

    def test_count_too_few_penalized(self) -> None:
        ex = _prep_example()
        out = _good_prep_output(questions=_good_prep_output()["expected_questions"][:2])
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["count"] < 1.0

    def test_grounded_rationales_full(self) -> None:
        ex = _prep_example()
        out = _good_prep_output()  # rationales mention RemeDi/PyTorch/Transformer
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["grounded"] >= 0.6

    def test_grounded_rationales_zero_when_all_generic(self) -> None:
        ex = _prep_example()
        out = _good_prep_output(
            questions=[
                {
                    "question": f"Q{i}",
                    "category": "technical" if i % 2 == 0 else "behavioral",
                    "likelihood": 0.5,
                    "rationale": "通用大厂常考题",  # no profile/jd keywords
                }
                for i in range(5)
            ],
        )
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["grounded"] == 0.0

    def test_calibration_penalty_for_uniform_likelihoods(self) -> None:
        ex = _prep_example()
        out = _good_prep_output(
            questions=[
                {
                    "question": f"Q{i}",
                    "category": "technical" if i < 3 else "behavioral",
                    "likelihood": 0.7,  # all same
                    "rationale": "JD 要求 PyTorch",
                }
                for i in range(5)
            ],
        )
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["calibration"] == 0.0

    def test_calibration_full_for_spread_likelihoods(self) -> None:
        ex = _prep_example()
        out = _good_prep_output()  # likelihoods range 0.40 to 0.85
        result = prepare_interview.metric(ex, json.dumps(out))
        assert result.breakdown["calibration"] >= 0.5


class TestPrepareInterviewSplit:
    def test_deterministic(self) -> None:
        a1, b1 = prepare_interview.split_train_val(seed=0)
        a2, b2 = prepare_interview.split_train_val(seed=0)
        assert [e.name for e in a1] == [e.name for e in a2]
        assert [e.name for e in b1] == [e.name for e in b2]

    def test_total_count_preserved(self) -> None:
        train, val = prepare_interview.split_train_val()
        assert len(train) + len(val) == len(prepare_interview.EXAMPLES)

    def test_each_band_has_at_least_one_in_val(self) -> None:
        _, val = prepare_interview.split_train_val()
        bands = {e.band for e in val}
        # Every band gets at least one val sample (stratified)
        all_bands = {e.band for e in prepare_interview.EXAMPLES}
        assert bands == all_bands


# ═══════════════════════════════════════════════════════════════════
# AGGREGATE — works for arbitrary axis names
# ═══════════════════════════════════════════════════════════════════


class TestAggregateGeneric:
    def test_handles_score_match_axes(self) -> None:
        """Backward-compat axis names from score_match."""
        ms = [
            MetricBreakdown(total=0.8, breakdown={"prob": 1.0, "recall": 0.6, "anti": 1.0}),
            MetricBreakdown(total=0.6, breakdown={"prob": 0.5, "recall": 0.8, "anti": 0.5}),
        ]
        agg = aggregate(ms)
        assert agg["total"] == pytest.approx(0.7)
        assert agg["prob"] == pytest.approx(0.75)

    def test_handles_analyze_gaps_axes(self) -> None:
        ms = [
            MetricBreakdown(
                total=0.8,
                breakdown={"schema": 1.0, "keyword_recall": 0.7, "ai_risk": 1.0, "count": 1.0},
            ),
            MetricBreakdown(
                total=0.5,
                breakdown={"schema": 1.0, "keyword_recall": 0.3, "ai_risk": 0.0, "count": 1.0},
            ),
        ]
        agg = aggregate(ms)
        assert "keyword_recall" in agg
        assert agg["keyword_recall"] == pytest.approx(0.5)
        assert agg["ai_risk"] == pytest.approx(0.5)

    def test_handles_mixed_axes(self) -> None:
        """If two metrics have different axis sets, axes from either appear."""
        ms = [
            MetricBreakdown(total=0.5, breakdown={"a": 0.5}),
            MetricBreakdown(total=0.5, breakdown={"b": 0.5}),
        ]
        agg = aggregate(ms)
        assert agg["a"] == 0.25  # mean of 0.5 and 0.0 (missing → 0)
        assert agg["b"] == 0.25
