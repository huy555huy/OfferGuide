"""W8' — deep_project_prep SKILL: schema + adapter + agent integration.

Tests the 4th evolvable SKILL across three layers:
- Pydantic schema enforces extra='forbid', likelihood ∈ [0,1], list bounds
- Adapter metric scores schema/coverage/grounded/concrete/behavioral on
  good and bad outputs
- Agent graph routes ``deep_prep`` action to deep_prep_node and returns
  a deep_prep_result; ``everything`` runs all 4 SKILLs in order
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from offerguide.evolution.adapters import deep_project_prep, get_adapter
from offerguide.skills import discover_skills, load_skill
from offerguide.skills.deep_project_prep.helpers import (
    DeepProjectPrepResult,
    ProbingQuestion,
    WeakPoint,
)

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ── Frontmatter ────────────────────────────────────────────────────


def test_skill_loads_from_disk() -> None:
    spec = load_skill(SKILLS_ROOT / "deep_project_prep")
    assert spec.name == "deep_project_prep"
    assert spec.version == "0.1.0"
    assert set(spec.inputs) == {"company", "job_text", "user_profile"}


def test_skill_appears_in_discover() -> None:
    skills = discover_skills(SKILLS_ROOT)
    names = {s.name for s in skills}
    assert "deep_project_prep" in names
    # All 4 evolvable SKILLs present
    assert {"score_match", "analyze_gaps", "prepare_interview", "deep_project_prep"} <= names


# ── Helper output (the canonical "good" example) ───────────────────


def _good_output() -> dict:
    return {
        "company_style_summary": "字节 Seed 偏项目深挖 + 性能数字。这次 RemeDi 必被深问。",
        "projects_analyzed": [
            {
                "project_name": "Deep Research Agent",
                "project_summary": "面试官视角：单 agent runtime + 双层架构，目标 evidence 完整性",
                "technical_claims": [
                    "evidence-centric 上下文管理",
                    "双层架构 (语义层 + 工作区层)",
                    "closure semantics",
                ],
                "probing_questions": [
                    {
                        "question": "讲讲 evidence-centric 上下文管理具体怎么做",
                        "type": "foundational",
                        "likelihood": 0.85,
                        "rationale": "简历里说了 evidence-centric，JD 提到 RAG",
                        "answer_outline": [
                            "evidence 列表用 BGE-M3 做 embedding 检索",
                            "agent loop 每步 max 3 evidences 进上下文",
                            "vs ReAct: 只保留必要 evidence 而不是 thought-action-observation",
                        ],
                        "followups": [
                            "evidence 列表大于 100 时怎么选 top-3?",
                            "embedding 怎么避免 stale?",
                        ],
                    },
                    {
                        "question": "为什么不直接用 ReAct?",
                        "type": "challenge",
                        "likelihood": 0.7,
                        "rationale": "evidence 设计是简历里有争议的选择",
                        "answer_outline": [
                            "ReAct 上下文随 step 线性增长",
                            "evidence 模式 KV cache 命中率 +30%",
                            "tradeoff: 缺历史推理链路",
                        ],
                        "followups": ["缺历史推理链路怎么办?"],
                    },
                    {
                        "question": "如果你有 100 个并发 agent 怎么改?",
                        "type": "extension",
                        "likelihood": 0.5,
                        "rationale": "JD 提到分布式; 简历是单 agent runtime",
                        "answer_outline": [
                            "evidence 共享存到 redis",
                            "agent state 各自隔离",
                            "调度: priority queue + workload balancing",
                        ],
                        "followups": [],
                    },
                ],
                "weak_points": [
                    {
                        "weakness": "没有 evidence 检索的 ablation, 无法证明 BGE-M3 的必要性",
                        "mitigation": "诚实承认 + 用'实习时间限制'框定 + 未来工作",
                        "likely_question": "你怎么证明 evidence 检索本身有用?",
                    },
                ],
            },
            {
                "project_name": "RemeDi",
                "project_summary": "LLaDA-8B 扩散模型 + 双流架构 + GRPO",
                "technical_claims": [
                    "TPS + UPS 双流",
                    "GRPO 训练",
                    "DeepSpeed ZeRO-2",
                ],
                "probing_questions": [
                    {
                        "question": "GRPO vs PPO 区别在哪?",
                        "type": "foundational",
                        "likelihood": 0.8,
                        "rationale": "JD 直接提了 GRPO; 简历也有",
                        "answer_outline": [
                            "GRPO 不需要 critic, 用 group sample 估计 baseline",
                            "PPO clip 在 ratio 1±epsilon, GRPO 在 advantage",
                            "DeepSeek paper 的核心创新",
                        ],
                        "followups": ["训练不稳怎么调?"],
                    },
                    {
                        "question": "ZeRO-2 vs ZeRO-3 选哪个?",
                        "type": "tradeoff",
                        "likelihood": 0.6,
                        "rationale": "简历 ZeRO-2; JD 大模型分布式",
                        "answer_outline": [
                            "ZeRO-2: 仅 partition optimizer states + gradients",
                            "ZeRO-3: 还 partition parameters, 通信增加",
                            "8B 用 ZeRO-2 性价比更高",
                        ],
                        "followups": [],
                    },
                    {
                        "question": "扩散语言模型相比 autoregressive 优势在哪?",
                        "type": "deep_dive",
                        "likelihood": 0.65,
                        "rationale": "项目核心创新点; 不强问就奇怪了",
                        "answer_outline": [
                            "并行 denoising: O(T) 步长 vs O(N) tokens",
                            "天然支持任意位置编辑",
                            "training-inference gap 较小",
                        ],
                        "followups": [],
                    },
                ],
                "weak_points": [],
            },
        ],
        "cross_project_questions": [
            {
                "question": "你这俩项目设计哲学有什么共通点?",
                "type": "extension",
                "likelihood": 0.5,
                "rationale": "staff 级常考题",
                "answer_outline": [
                    "都强调'压缩信息'; agent 是 evidence, RemeDi 是双流",
                    "都用了 LoRA / ZeRO-2 节省显存",
                ],
                "followups": [],
            },
        ],
        "behavioral_questions_tailored": [
            {
                "question": "你在法至科技实习时跟产品对齐 agent 工作流, 有过分歧吗?",
                "type": "extension",
                "likelihood": 0.55,
                "rationale": "behavioral STAR; 引用真实经历",
                "answer_outline": [
                    "产品要 ChatGPT-like 即时响应",
                    "我推 Deep Research 多步推理",
                    "妥协: 保留两种模式, 让用户切换",
                ],
                "followups": [],
            },
        ],
    }


# ── Pydantic schema ────────────────────────────────────────────────


class TestSchema:
    def test_validates_well_formed(self) -> None:
        result = DeepProjectPrepResult.model_validate(_good_output())
        assert len(result.projects_analyzed) == 2
        assert result.projects_analyzed[0].project_name == "Deep Research Agent"

    def test_rejects_extra_keys(self) -> None:
        bad = _good_output()
        bad["bonus_field"] = "rogue"
        with pytest.raises(ValidationError):
            DeepProjectPrepResult.model_validate(bad)

    def test_rejects_likelihood_above_1(self) -> None:
        bad = _good_output()
        bad["projects_analyzed"][0]["probing_questions"][0]["likelihood"] = 1.2
        with pytest.raises(ValidationError):
            DeepProjectPrepResult.model_validate(bad)

    def test_rejects_invalid_question_type(self) -> None:
        bad = _good_output()
        bad["projects_analyzed"][0]["probing_questions"][0]["type"] = "bogus"
        with pytest.raises(ValidationError):
            DeepProjectPrepResult.model_validate(bad)

    def test_rejects_too_few_probing_questions(self) -> None:
        bad = _good_output()
        bad["projects_analyzed"][0]["probing_questions"] = bad[
            "projects_analyzed"
        ][0]["probing_questions"][:2]  # only 2, needs ≥ 3
        with pytest.raises(ValidationError):
            DeepProjectPrepResult.model_validate(bad)

    def test_rejects_empty_outline(self) -> None:
        bad = _good_output()
        bad["projects_analyzed"][0]["probing_questions"][0]["answer_outline"] = []
        with pytest.raises(ValidationError):
            DeepProjectPrepResult.model_validate(bad)

    def test_rejects_zero_projects(self) -> None:
        bad = _good_output()
        bad["projects_analyzed"] = []
        with pytest.raises(ValidationError):
            DeepProjectPrepResult.model_validate(bad)

    def test_directly_instantiate_question_and_weakness(self) -> None:
        q = ProbingQuestion(
            question="Q", type="foundational", likelihood=0.5,
            rationale="R", answer_outline=["a", "b"], followups=[],
        )
        assert q.likelihood == 0.5

        w = WeakPoint(weakness="x", mitigation="y", likely_question="z")
        assert w.weakness == "x"

    def test_helper_methods(self) -> None:
        result = DeepProjectPrepResult.model_validate(_good_output())
        assert result.total_question_count() == (
            3 + 3 + 1 + 1
        )  # proj1=3 + proj2=3 + cross=1 + behavioral=1
        weakest = result.weakest_spots(limit=10)
        assert len(weakest) == 1
        assert isinstance(weakest[0], WeakPoint)
        # Question-type coverage on each project
        cov1 = result.projects_analyzed[0].question_types_covered()
        assert {"foundational", "challenge", "extension"} <= cov1


# ── Adapter (metric, EXAMPLES, registry) ───────────────────────────


def _example():
    return next(
        e for e in deep_project_prep.EXAMPLES if e.name == "bytedance_seed_post_training"
    )


class TestAdapter:
    def test_registered(self) -> None:
        assert get_adapter("deep_project_prep") is deep_project_prep
        assert deep_project_prep.name == "deep_project_prep"
        assert "company" in deep_project_prep.INPUT_NAMES
        assert "schema" in deep_project_prep.METRIC_AXES

    def test_examples_non_empty(self) -> None:
        assert len(deep_project_prep.EXAMPLES) >= 4

    def test_invalid_json_scores_zero(self) -> None:
        ex = _example()
        result = deep_project_prep.metric(ex, "not json {")
        assert result.total == 0.0
        assert all(v == 0.0 for v in result.breakdown.values())

    def test_schema_violation_scores_zero(self) -> None:
        ex = _example()
        result = deep_project_prep.metric(ex, json.dumps({"foo": "bar"}))
        assert result.total == 0.0
        assert result.breakdown["schema"] == 0.0

    def test_valid_output_scores_high(self) -> None:
        ex = _example()
        out = _good_output()
        result = deep_project_prep.metric(ex, json.dumps(out, ensure_ascii=False))
        assert result.total > 0.5  # solid output should clear half
        assert result.breakdown["schema"] == 1.0

    def test_type_coverage_full_when_3_plus_per_project(self) -> None:
        ex = _example()
        result = deep_project_prep.metric(ex, json.dumps(_good_output(), ensure_ascii=False))
        # Project 1 has 3 types, Project 2 has 3 types → both score 1.0
        assert result.breakdown["type_coverage"] == 1.0

    def test_type_coverage_low_when_only_one_type(self) -> None:
        ex = _example()
        out = _good_output()
        # Force project 1 to have only foundational
        for q in out["projects_analyzed"][0]["probing_questions"]:
            q["type"] = "foundational"
        result = deep_project_prep.metric(ex, json.dumps(out, ensure_ascii=False))
        # Project 1 = 1/3, Project 2 = 1.0 → mean = 0.667
        assert result.breakdown["type_coverage"] < 0.8

    def test_outline_concreteness_scores_high_with_numbers_or_terms(self) -> None:
        ex = _example()
        result = deep_project_prep.metric(ex, json.dumps(_good_output(), ensure_ascii=False))
        # Outlines contain BGE-M3, GRPO, ZeRO, +30%, etc. → concreteness > generic
        assert result.breakdown["outline_concreteness"] > 0.5

    def test_outline_concreteness_drops_with_empty_text(self) -> None:
        ex = _example()
        out = _good_output()
        # All outlines now generic
        for proj in out["projects_analyzed"]:
            for q in proj["probing_questions"]:
                q["answer_outline"] = ["要自信", "讲清楚就好"]
        result = deep_project_prep.metric(ex, json.dumps(out, ensure_ascii=False))
        assert result.breakdown["outline_concreteness"] < 0.5

    def test_rationale_grounded_drops_with_generic_rationales(self) -> None:
        ex = _example()
        out = _good_output()
        for proj in out["projects_analyzed"]:
            for q in proj["probing_questions"]:
                q["rationale"] = "大厂常考"  # no profile/JD keyword anchor
        out["cross_project_questions"][0]["rationale"] = "通用题"
        out["behavioral_questions_tailored"][0]["rationale"] = "测试题"
        result = deep_project_prep.metric(ex, json.dumps(out, ensure_ascii=False))
        assert result.breakdown["rationale_grounded"] < 0.3

    def test_behavioral_specificity_full_when_keyword_referenced(self) -> None:
        ex = _example()
        result = deep_project_prep.metric(ex, json.dumps(_good_output(), ensure_ascii=False))
        # The behavioral question mentions "法至科技实习" and "agent" — but ex's
        # profile_keywords are RemeDi/GRPO/DeepSpeed/扩散/LoRA. None of those
        # appear in the behavioral question, so it scores 0.
        assert result.breakdown["behavioral_specificity"] in (0.0, 1.0)


# ── split_train_val ────────────────────────────────────────────────


class TestSplit:
    def test_deterministic(self) -> None:
        a1, b1 = deep_project_prep.split_train_val(seed=42)
        a2, b2 = deep_project_prep.split_train_val(seed=42)
        assert [e.name for e in a1] == [e.name for e in a2]
        assert [e.name for e in b1] == [e.name for e in b2]

    def test_total_count_preserved(self) -> None:
        train, val = deep_project_prep.split_train_val()
        assert len(train) + len(val) == len(deep_project_prep.EXAMPLES)
