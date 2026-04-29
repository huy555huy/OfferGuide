"""Agent graph — routing on `requested_action`, summary composition, error handling.

We pass a fake `runtime` that doesn't touch the LLM — just records calls and
returns canned SkillResults. This exercises the graph topology end-to-end
without depending on DEEPSEEK_API_KEY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import offerguide
from offerguide.agent.graph import _format_summary, build_graph
from offerguide.skills import SkillResult, discover_skills

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


class _FakeRuntime:
    """Mimics SkillRuntime.invoke(spec, inputs) → SkillResult, no LLM call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._next_run_id = 100

    def invoke(self, spec, inputs, **_):
        self._next_run_id += 1
        self.calls.append((spec.name, inputs))
        if spec.name == "score_match":
            parsed = {
                "probability": 0.72,
                "reasoning": "技术匹配较高",
                "dimensions": {"tech": 0.8, "exp": 0.6, "company_tier": 0.7},
                "deal_breakers": [],
            }
        else:
            parsed = {
                "summary": "整体匹配 70%，主要 gap：缺少 RLHF 经验",
                "keyword_gaps": [],
                "suggestions": [
                    {
                        "section": "技能",
                        "action": "add",
                        "current_text": None,
                        "proposed_addition": "PyTorch 2.x",
                        "reason": "JD 明确要求",
                        "ai_risk": "low",
                        "confidence": 0.9,
                    }
                ],
                "do_not_add": [],
                "ai_detection_warnings": [],
            }
        return SkillResult(
            raw_text=json.dumps(parsed),
            parsed=parsed,
            skill_name=spec.name,
            skill_version=spec.version,
            skill_run_id=self._next_run_id,
            input_hash="x" * 64,
            cost_usd=0.0,
            latency_ms=10,
        )


def test_score_only_dispatches_only_score() -> None:
    skills = discover_skills(SKILLS_ROOT)
    rt = _FakeRuntime()
    graph = build_graph(skills=skills, runtime=rt)

    result = graph.invoke(
        {
            "requested_action": "score",
            "job_text": "AI Agent at ByteDance",
            "user_profile_text": "胡阳 ...",
        }
    )
    assert [c[0] for c in rt.calls] == ["score_match"]
    assert result["score_result"]["probability"] == 0.72
    assert result.get("gaps_result") is None
    assert "校准概率" in result["final_response"]


def test_gaps_only_dispatches_only_gaps() -> None:
    skills = discover_skills(SKILLS_ROOT)
    rt = _FakeRuntime()
    graph = build_graph(skills=skills, runtime=rt)

    result = graph.invoke(
        {"requested_action": "gaps", "job_text": "x", "user_profile_text": "y"}
    )
    assert [c[0] for c in rt.calls] == ["analyze_gaps"]
    assert result.get("score_result") is None
    assert result["gaps_result"]["summary"].startswith("整体匹配")
    assert "差距与建议" in result["final_response"]


def test_score_and_gaps_dispatches_both_in_order() -> None:
    skills = discover_skills(SKILLS_ROOT)
    rt = _FakeRuntime()
    graph = build_graph(skills=skills, runtime=rt)

    result = graph.invoke(
        {
            "requested_action": "score_and_gaps",
            "job_text": "x",
            "user_profile_text": "y",
        }
    )
    assert [c[0] for c in rt.calls] == ["score_match", "analyze_gaps"]
    assert result["score_result"]["probability"] == 0.72
    assert result["gaps_result"]["summary"].startswith("整体匹配")
    response = result["final_response"]
    assert "## 匹配评分" in response
    assert "## 差距与建议" in response


def test_runtime_none_routes_through_summarize_with_error() -> None:
    """No runtime configured → score_node returns error → summarize still produces text."""
    skills = discover_skills(SKILLS_ROOT)
    graph = build_graph(skills=skills, runtime=None)
    result = graph.invoke(
        {"requested_action": "score", "job_text": "x", "user_profile_text": "y"}
    )
    assert "no SkillRuntime" in result.get("error", "")
    assert "出错" in result["final_response"]


def test_format_summary_handles_empty_state() -> None:
    out = _format_summary({})  # type: ignore[arg-type]
    assert "没有可显示的结果" in out


def test_format_summary_renders_deal_breakers() -> None:
    state: dict = {
        "score_result": {
            "probability": 0.05,
            "reasoning": "硬性不匹配",
            "dimensions": {"tech": 0.1, "exp": 0.05, "company_tier": 0.5},
            "deal_breakers": ["要求 5 年经验，应届"],
        }
    }
    out = _format_summary(state)  # type: ignore[arg-type]
    assert "Deal-breakers" in out
    assert "5 年经验" in out


def test_format_summary_caps_suggestions_to_six() -> None:
    state: dict = {
        "gaps_result": {
            "summary": "...",
            "keyword_gaps": [],
            "suggestions": [
                {
                    "section": f"sec{i}",
                    "action": "add",
                    "current_text": None,
                    "proposed_addition": f"sug {i}",
                    "reason": "r",
                    "ai_risk": "low",
                    "confidence": 0.5,
                }
                for i in range(20)
            ],
            "do_not_add": [],
            "ai_detection_warnings": [],
        }
    }
    out = _format_summary(state)  # type: ignore[arg-type]
    # Only first 6 should appear
    assert out.count("**[") == 6
    assert "[6]" in out and "[7]" not in out


def test_public_api_still_exposes_build_graph() -> None:
    assert hasattr(offerguide, "build_graph")
    assert callable(offerguide.build_graph)
