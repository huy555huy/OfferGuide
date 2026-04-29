"""W3 analyze_gaps SKILL tests — schema, loadability, runtime invocation with stub LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import offerguide
from offerguide.llm import LLMResponse
from offerguide.skills import SkillRuntime, load_skill
from offerguide.skills.analyze_gaps.helpers import (
    AnalyzeGapsResult,
    KeywordGap,
    Suggestion,
)

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


class _StubLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages,
        *,
        model=None,
        temperature=0.3,
        json_mode=False,
        extra=None,
    ) -> LLMResponse:
        self.calls.append({
            "messages": list(messages),
            "model": model,
            "json_mode": json_mode,
        })
        return LLMResponse(content=self._content, model="stub")


# ---- SKILL.md loadability + frontmatter ----------------------------------


def test_analyze_gaps_skill_loads_with_correct_frontmatter() -> None:
    spec = load_skill(SKILLS_ROOT / "analyze_gaps")
    assert spec.name == "analyze_gaps"
    assert spec.version == "0.1.0"
    assert spec.inputs == ("job_text", "user_profile")
    assert "ai-detection-aware" in spec.tags
    assert "这份简历差什么" in spec.triggers
    # Body must mention all 4 core principles so GEPA has a grammar to mutate
    body = spec.body
    assert "不重写" in body
    assert "不编造" in body
    assert "AI 风险" in body
    assert "keyword" in body.lower()


def test_analyze_gaps_appears_in_discover_skills() -> None:
    skills = offerguide.discover_skills(SKILLS_ROOT)
    names = {s.name for s in skills}
    assert {"score_match", "analyze_gaps"} <= names


# ---- Pydantic schema -----------------------------------------------------


_SAMPLE_RESULT = {
    "summary": "整体匹配度 60%。最大 gap 是没有 LLM 训练经验。",
    "keyword_gaps": [
        {
            "jd_keyword": "PyTorch",
            "in_resume": True,
            "importance": "high",
            "evidence_in_jd": "熟悉深度学习主流框架（PyTorch/TensorFlow）",
        },
        {
            "jd_keyword": "C++",
            "in_resume": False,
            "importance": "medium",
            "evidence_in_jd": "具备扎实的 C/C++/Java（至少一门）基础",
        },
    ],
    "suggestions": [
        {
            "section": "技能",
            "action": "add",
            "current_text": None,
            "proposed_addition": "C/C++（本科课程级，能读懂主流开源项目）",
            "reason": "JD 要求 C/C++/Java 至少一门，简历需要至少 mention",
            "ai_risk": "low",
            "confidence": 0.8,
        },
        {
            "section": "项目经历",
            "action": "emphasize",
            "current_text": "RemeDi 项目：实现 SFT 训练管线",
            "proposed_addition": "在 SFT 训练中实现了基于噪声调度的动态数据增强（模拟掩码与错误 Token）",
            "reason": "JD 要求精通 SFT/RL，建议把已有的具体技术细节 surface 出来",
            "ai_risk": "low",
            "confidence": 0.9,
        },
    ],
    "do_not_add": ["不要建议加 LLM 预训练经验——用户没做过"],
    "ai_detection_warnings": [],
}


def test_analyze_gaps_result_validates_well_formed() -> None:
    parsed = AnalyzeGapsResult.model_validate(_SAMPLE_RESULT)
    assert parsed.summary.startswith("整体匹配度")
    assert len(parsed.keyword_gaps) == 2
    assert len(parsed.suggestions) == 2
    assert isinstance(parsed.keyword_gaps[0], KeywordGap)
    assert isinstance(parsed.suggestions[0], Suggestion)


def test_analyze_gaps_result_helpers() -> None:
    parsed = AnalyzeGapsResult.model_validate(_SAMPLE_RESULT)
    assert parsed.high_risk_count() == 0  # both suggestions are low risk

    # add a high-risk one
    payload = {**_SAMPLE_RESULT}
    payload["suggestions"] = [
        *payload["suggestions"],
        {
            "section": "自我评价",
            "action": "add",
            "current_text": None,
            "proposed_addition": "热情主动，赋能团队，驱动业务增长",
            "reason": "soft-skills",
            "ai_risk": "high",
            "confidence": 0.4,
        },
    ]
    parsed2 = AnalyzeGapsResult.model_validate(payload)
    assert parsed2.high_risk_count() == 1


def test_analyze_gaps_result_deal_breaker_helper() -> None:
    parsed = AnalyzeGapsResult.model_validate(_SAMPLE_RESULT)
    # PyTorch is high importance but already in resume → not a deal-breaker
    # C++ is medium importance not in resume → not a deal-breaker (only high counts)
    assert parsed.deal_breaker_keyword_gaps() == []

    payload = {**_SAMPLE_RESULT}
    payload["keyword_gaps"] = [
        *payload["keyword_gaps"],
        {
            "jd_keyword": "RLHF 经验",
            "in_resume": False,
            "importance": "high",
            "evidence_in_jd": "需 RLHF 经验",
        },
    ]
    parsed2 = AnalyzeGapsResult.model_validate(payload)
    deal_breakers = parsed2.deal_breaker_keyword_gaps()
    assert len(deal_breakers) == 1
    assert deal_breakers[0].jd_keyword == "RLHF 经验"


def test_analyze_gaps_rejects_invalid_action() -> None:
    bad = {**_SAMPLE_RESULT}
    bad["suggestions"] = [{**_SAMPLE_RESULT["suggestions"][0], "action": "rewrite_everything"}]
    with pytest.raises(ValidationError):
        AnalyzeGapsResult.model_validate(bad)


def test_analyze_gaps_rejects_invalid_ai_risk() -> None:
    bad = {**_SAMPLE_RESULT}
    bad["suggestions"] = [{**_SAMPLE_RESULT["suggestions"][0], "ai_risk": "extreme"}]
    with pytest.raises(ValidationError):
        AnalyzeGapsResult.model_validate(bad)


def test_analyze_gaps_rejects_extra_keys_at_root() -> None:
    bad = {**_SAMPLE_RESULT, "uninvited": True}
    with pytest.raises(ValidationError):
        AnalyzeGapsResult.model_validate(bad)


def test_analyze_gaps_rejects_confidence_out_of_range() -> None:
    bad = {**_SAMPLE_RESULT}
    bad["suggestions"] = [{**_SAMPLE_RESULT["suggestions"][0], "confidence": 1.5}]
    with pytest.raises(ValidationError):
        AnalyzeGapsResult.model_validate(bad)


# ---- runtime invocation --------------------------------------------------


def test_runtime_invokes_analyze_gaps_and_persists(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "analyze_gaps")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()

    canned = json.dumps(_SAMPLE_RESULT, ensure_ascii=False)
    llm = _StubLLM(canned)
    rt = SkillRuntime(llm, store)  # type: ignore[arg-type]

    result = rt.invoke(
        spec,
        {"job_text": "AI Agent JD ...", "user_profile": "胡阳 资料 ..."},
    )

    assert result.parsed is not None
    parsed = AnalyzeGapsResult.model_validate(result.parsed)
    assert len(parsed.suggestions) == 2

    # System prompt should be the SKILL body (which mentions 不重写)
    msgs = llm.calls[0]["messages"]
    assert "不重写" in msgs[0]["content"]
    # User message must include both inputs labelled
    assert "### job_text" in msgs[1]["content"]
    assert "### user_profile" in msgs[1]["content"]

    # Persisted to skill_runs alongside score_match runs (separate skill_name)
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT skill_name, skill_version FROM skill_runs"
        ).fetchall()
    assert ("analyze_gaps", "0.1.0") in rows
