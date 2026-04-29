"""W8 — prepare_interview SKILL: schema, frontmatter, runtime invocation.

Tests the third evolvable SKILL. Like score_match / analyze_gaps,
exercises:
- SKILL.md frontmatter loads with required fields
- Pydantic schema validates well-formed output
- Pydantic schema rejects malformed output (extra keys, OOR likelihood, ...)
- Helper methods (top_questions, categories_covered, coverage_warning)
- discover_skills picks it up
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from offerguide.skills import discover_skills, load_skill
from offerguide.skills.prepare_interview.helpers import (
    InterviewQuestion,
    PrepareInterviewResult,
)

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ── Frontmatter / loading ──────────────────────────────────────────


def test_prepare_interview_skill_loads_with_correct_frontmatter() -> None:
    spec = load_skill(SKILLS_ROOT / "prepare_interview")
    assert spec.name == "prepare_interview"
    assert spec.version == "0.1.0"
    assert set(spec.inputs) == {"company", "job_text", "user_profile", "past_experiences"}
    assert spec.evolved_at is None
    assert spec.parent_version is None


def test_prepare_interview_appears_in_discover_skills() -> None:
    skills = discover_skills(SKILLS_ROOT)
    names = {s.name for s in skills}
    assert "prepare_interview" in names
    assert "score_match" in names
    assert "analyze_gaps" in names


# ── Pydantic validation ────────────────────────────────────────────


def _good_payload() -> dict[str, Any]:
    return {
        "company_snapshot": "字节跳动 AI Lab，主攻 LLM 应用与多模态。"
        " JD 要求熟悉 LangGraph 和 RAG。面经数据缺，下方推断基于 JD。",
        "expected_questions": [
            {
                "question": "讲讲 Transformer 自注意力机制为什么除以 √d",
                "category": "technical",
                "likelihood": 0.8,
                "rationale": "JD 第 1 条要求精通 Transformer 原理",
            },
            {
                "question": "讲一个跨团队协作的例子",
                "category": "behavioral",
                "likelihood": 0.6,
                "rationale": "校招通用 STAR 题，几乎必考",
            },
            {
                "question": "你做的 Deep Research Agent，agent loop 一次平均迭代多少次？",
                "category": "project_deep_dive",
                "likelihood": 0.7,
                "rationale": "简历项目，必被深挖",
            },
        ],
        "prep_focus_areas": [
            "Transformer 数学推导（QKV、softmax、缩放）",
            "LangGraph vs LangChain 区别",
            "实习经历的 STAR 模板",
        ],
        "weak_spots": [
            "JD 要求 LangGraph，简历只有 LangChain",
            "缺少分布式训练经验",
        ],
    }


def test_validates_well_formed() -> None:
    result = PrepareInterviewResult.model_validate(_good_payload())
    assert len(result.expected_questions) == 3
    assert result.expected_questions[0].likelihood == 0.8


def test_rejects_invalid_category() -> None:
    bad = _good_payload()
    bad["expected_questions"][0]["category"] = "rocket_science"
    with pytest.raises(ValidationError):
        PrepareInterviewResult.model_validate(bad)


def test_rejects_likelihood_above_1() -> None:
    bad = _good_payload()
    bad["expected_questions"][0]["likelihood"] = 1.2
    with pytest.raises(ValidationError):
        PrepareInterviewResult.model_validate(bad)


def test_rejects_likelihood_below_0() -> None:
    bad = _good_payload()
    bad["expected_questions"][0]["likelihood"] = -0.1
    with pytest.raises(ValidationError):
        PrepareInterviewResult.model_validate(bad)


def test_rejects_extra_keys_at_root() -> None:
    bad = _good_payload()
    bad["secret_field"] = "should not be allowed"
    with pytest.raises(ValidationError):
        PrepareInterviewResult.model_validate(bad)


def test_rejects_extra_keys_in_question() -> None:
    bad = _good_payload()
    bad["expected_questions"][0]["bonus"] = "extra"
    with pytest.raises(ValidationError):
        PrepareInterviewResult.model_validate(bad)


# ── Helper methods ─────────────────────────────────────────────────


def test_top_questions_sorts_by_likelihood_desc() -> None:
    result = PrepareInterviewResult.model_validate(_good_payload())
    top2 = result.top_questions(n=2)
    assert len(top2) == 2
    assert top2[0].likelihood == 0.8
    assert top2[1].likelihood == 0.7


def test_categories_covered_returns_set() -> None:
    result = PrepareInterviewResult.model_validate(_good_payload())
    cats = result.categories_covered()
    assert cats == {"technical", "behavioral", "project_deep_dive"}


def test_coverage_warning_is_none_when_3_categories_covered() -> None:
    result = PrepareInterviewResult.model_validate(_good_payload())
    assert result.coverage_warning() is None


def test_coverage_warning_fires_when_under_3_categories() -> None:
    payload = _good_payload()
    # Restrict to only 2 categories
    payload["expected_questions"] = [
        q for q in payload["expected_questions"]
        if q["category"] in ("technical", "behavioral")
    ]
    result = PrepareInterviewResult.model_validate(payload)
    warning = result.coverage_warning()
    assert warning is not None
    assert "2/5" in warning


# ── Runtime invocation (no real LLM call) ──────────────────────────


def test_runtime_invokes_prepare_interview_and_persists(tmp_path: Path) -> None:
    """Smoke test that the SKILL can be rendered + parsed by the runtime."""
    import offerguide
    from offerguide.skills import SkillRuntime

    # Stub LLM that returns a valid prepare_interview payload
    class _StubLLM:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def chat(self, *, messages, json_mode=True, model=None, temperature=0.3):
            self.calls.append(messages)
            return type(
                "Resp",
                (),
                {
                    "content": json.dumps(_good_payload(), ensure_ascii=False),
                    "latency_ms": 42,
                },
            )()

    store = offerguide.Store(tmp_path / "pi.db")
    store.init_schema()
    spec = load_skill(SKILLS_ROOT / "prepare_interview")
    runtime = SkillRuntime(_StubLLM(), store)  # type: ignore[arg-type]

    result = runtime.invoke(
        spec,
        {
            "company": "字节跳动",
            "job_text": "AI Agent 实习，要求 LangGraph 经验",
            "user_profile": "胡阳，应用统计专硕，做过 Deep Research Agent",
            "past_experiences": "",
        },
    )

    parsed = PrepareInterviewResult.model_validate(result.parsed)
    assert parsed.company_snapshot.startswith("字节跳动")
    assert len(parsed.expected_questions) == 3

    # Should have been persisted to skill_runs
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT skill_name, skill_version FROM skill_runs"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "prepare_interview"
    assert rows[0][1] == "0.1.0"


# ── Smoke: question type is a true Pydantic model ──────────────────


def test_interview_question_can_be_instantiated_directly() -> None:
    q = InterviewQuestion(
        question="什么是 GEPA？",
        category="technical",
        likelihood=0.5,
        rationale="JD 提到了 GEPA",
    )
    assert q.likelihood == 0.5
