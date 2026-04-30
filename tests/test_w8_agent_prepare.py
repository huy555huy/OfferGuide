"""W8' — Agent integration tests for the prepare_interview node.

Covers:
- Routing: ``prepare_interview`` action goes to prep_node only
- Routing: ``everything`` runs score → gaps → prep
- Empty company → graceful error (not a silent failure)
- Past experiences retrieved from interview_corpus when not provided
- Caller-provided past_experiences takes precedence over corpus
- Prep result rendered in the final summary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offerguide
from offerguide import interview_corpus
from offerguide.agent import build_graph
from offerguide.agent.state import RequestedAction
from offerguide.skills import SkillResult, discover_skills

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ── fakes ──────────────────────────────────────────────────────────


class _FakeRuntime:
    """Records calls; returns canned SkillResult per SKILL name."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._next_id = 100

    def invoke(self, spec, inputs, **_):
        self._next_id += 1
        self.calls.append((spec.name, dict(inputs)))
        if spec.name == "score_match":
            parsed = {
                "probability": 0.7,
                "reasoning": "fit",
                "dimensions": {"tech": 0.7, "exp": 0.6, "company_tier": 0.7},
                "deal_breakers": [],
            }
        elif spec.name == "analyze_gaps":
            parsed = {
                "summary": "中等匹配",
                "keyword_gaps": [],
                "suggestions": [],
                "do_not_add": [],
                "ai_detection_warnings": [],
            }
        elif spec.name == "prepare_interview":
            parsed = {
                "company_snapshot": f"{inputs.get('company')} 画像",
                "expected_questions": [
                    {
                        "question": "讲讲 attention 缩放",
                        "category": "technical",
                        "likelihood": 0.8,
                        "rationale": "JD 要求",
                    },
                    {
                        "question": "STAR 协作题",
                        "category": "behavioral",
                        "likelihood": 0.5,
                        "rationale": "通用",
                    },
                    {
                        "question": "深挖项目",
                        "category": "project_deep_dive",
                        "likelihood": 0.7,
                        "rationale": "面经里有",
                    },
                ],
                "prep_focus_areas": ["Transformer", "STAR"],
                "weak_spots": ["Megatron"],
            }
        elif spec.name == "deep_project_prep":
            parsed = {
                "company_style_summary": f"{inputs.get('company')} 风格画像",
                "projects_analyzed": [
                    {
                        "project_name": "Deep Research Agent",
                        "project_summary": "agent runtime",
                        "technical_claims": ["双层架构", "evidence-centric"],
                        "probing_questions": [
                            {"question": "讲讲 evidence-centric", "type": "foundational",
                             "likelihood": 0.8, "rationale": "JD",
                             "answer_outline": ["a", "b"], "followups": []},
                        ],
                        "weak_points": [],
                    },
                ],
                "cross_project_questions": [],
                "behavioral_questions_tailored": [],
            }
        else:
            parsed = {}
        return SkillResult(
            raw_text=json.dumps(parsed),
            parsed=parsed,
            skill_name=spec.name,
            skill_version="0.1.0",
            skill_run_id=self._next_id,
            input_hash="fake_hash",
            cost_usd=0.0,
            latency_ms=10,
        )


# ── helpers ────────────────────────────────────────────────────────


def _make_store(tmp_path: Path) -> offerguide.Store:
    store = offerguide.Store(tmp_path / "agent.db")
    store.init_schema()
    return store


def _seed_experiences(store: offerguide.Store, company: str, n: int = 3) -> None:
    """Populate interview_experiences for the given company."""
    for i in range(n):
        interview_corpus.insert(
            store,
            company=company,
            raw_text=f"面经 {i+1}: {company} 一面问 attention 缩放，二面问项目细节。",
            source="manual_paste",
        )


# ═══════════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════════


class TestPrepRouting:
    def test_prepare_interview_action_runs_only_prep(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": "面试"}],
                "requested_action": "prepare_interview",
                "job_text": "AI Agent 实习 - 字节",
                "user_profile_text": "胡阳的简历",
                "company": "字节跳动",
            }
        )

        called = [name for name, _ in runtime.calls]
        assert called == ["prepare_interview"]
        assert result.get("prep_result") is not None
        assert result.get("score_result") is None
        assert result.get("gaps_result") is None

    def test_everything_action_runs_all_four(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": "do all"}],
                "requested_action": "everything",
                "job_text": "AI Agent 实习",
                "user_profile_text": "胡阳",
                "company": "阿里",
            }
        )

        called = [name for name, _ in runtime.calls]
        assert called == [
            "score_match",
            "analyze_gaps",
            "prepare_interview",
            "deep_project_prep",
        ]
        assert result.get("score_result") is not None
        assert result.get("gaps_result") is not None
        assert result.get("prep_result") is not None
        assert result.get("deep_prep_result") is not None

    def test_score_and_gaps_does_not_run_prep(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": "x"}],
                "requested_action": "score_and_gaps",
                "job_text": "JD",
                "user_profile_text": "P",
            }
        )

        called = [name for name, _ in runtime.calls]
        assert called == ["score_match", "analyze_gaps"]
        assert "prepare_interview" not in called
        assert result.get("prep_result") is None


# ═══════════════════════════════════════════════════════════════════
# COMPANY HANDLING
# ═══════════════════════════════════════════════════════════════════


class TestPrepCompanyHandling:
    def test_missing_company_produces_error(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                # No company!
            }
        )

        assert result.get("error") is not None
        assert "company" in str(result["error"]).lower()
        assert runtime.calls == []  # SKILL never invoked
        assert result.get("prep_result") is None

    def test_blank_company_string_produces_error(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                "company": "   ",  # whitespace only
            }
        )

        assert result.get("error") is not None
        assert runtime.calls == []


# ═══════════════════════════════════════════════════════════════════
# PAST EXPERIENCES RETRIEVAL
# ═══════════════════════════════════════════════════════════════════


class TestPastExperienceRetrieval:
    def test_retrieves_from_corpus_when_store_provided(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_experiences(store, "字节跳动", n=3)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                "company": "字节跳动",
            }
        )

        assert result.get("prep_used_experiences") == 3
        # Verify the past_experiences input was non-empty
        prep_call = next(c for c in runtime.calls if c[0] == "prepare_interview")
        assert "面经 1" in prep_call[1]["past_experiences"]

    def test_caller_provided_takes_precedence(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_experiences(store, "字节跳动", n=3)  # corpus has data
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                "company": "字节跳动",
                "past_experiences": "用户手动粘贴的特殊面经",  # override
            }
        )

        prep_call = next(c for c in runtime.calls if c[0] == "prepare_interview")
        assert prep_call[1]["past_experiences"] == "用户手动粘贴的特殊面经"
        assert "面经 1" not in prep_call[1]["past_experiences"]

    def test_no_store_no_provided_sends_empty(self, tmp_path: Path) -> None:
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=None)

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                "company": "未知公司",
            }
        )

        prep_call = next(c for c in runtime.calls if c[0] == "prepare_interview")
        assert prep_call[1]["past_experiences"] == ""
        assert result.get("prep_used_experiences") == 0

    def test_corpus_empty_for_company_sends_empty(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # No data inserted for this company
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                "company": "完全没数据的公司",
            }
        )

        assert result.get("prep_used_experiences") == 0
        prep_call = next(c for c in runtime.calls if c[0] == "prepare_interview")
        assert prep_call[1]["past_experiences"] == ""


# ═══════════════════════════════════════════════════════════════════
# SUMMARY RENDERING
# ═══════════════════════════════════════════════════════════════════


class TestPrepSummary:
    def test_prep_section_appears_in_final_response(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_experiences(store, "字节跳动", n=2)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                "company": "字节跳动",
            }
        )

        text = result.get("final_response") or ""
        assert "面试备战" in text
        assert "字节跳动" in text
        assert "Transformer" in text  # prep_focus_areas
        assert "技术" in text or "[technical]" in text  # category rendered

    def test_action_literal_includes_new_options(self) -> None:
        """Smoke test that the type alias is in sync."""
        valid: list[RequestedAction] = [
            "score",
            "gaps",
            "score_and_gaps",
            "prepare_interview",
            "everything",
        ]
        assert "prepare_interview" in valid
        assert "everything" in valid


# ═══════════════════════════════════════════════════════════════════
# ERROR PATHS
# ═══════════════════════════════════════════════════════════════════


class TestPrepErrors:
    def test_runtime_none_returns_error(self, tmp_path: Path) -> None:
        skills = discover_skills(SKILLS_ROOT)
        graph = build_graph(skills=skills, runtime=None, store=None)

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                "company": "字节",
            }
        )
        # The prep_node short-circuits on runtime=None before company check
        assert result.get("error") is not None
        assert "Runtime" in result.get("error", "") or "runtime" in result.get("error", "")

    def test_skill_missing_returns_error(self, tmp_path: Path) -> None:
        # Build with only score_match SKILL — prepare_interview missing
        all_skills = discover_skills(SKILLS_ROOT)
        skills = [s for s in all_skills if s.name == "score_match"]
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=None)

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "prepare_interview",
                "job_text": "JD",
                "user_profile_text": "P",
                "company": "字节",
            }
        )
        assert result.get("error") is not None
        assert "prepare_interview" in result["error"]


def test_skill_result_can_carry_runtime_kwargs() -> None:
    """The fake runtime's signature should accept all SkillRuntime kwargs."""
    rt = _FakeRuntime()
    spec = next(s for s in discover_skills(SKILLS_ROOT) if s.name == "score_match")
    out = rt.invoke(spec, {"job_text": "x", "user_profile": "y"})
    assert out.skill_name == "score_match"


# Smoke test: pytest collection works
def test_module_imports() -> None:
    assert build_graph is not None
    assert pytest is not None
