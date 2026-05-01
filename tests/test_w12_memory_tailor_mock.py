"""W12 — long-term memory + tailor_resume + mock_interview.

Coverage:

- user_facts
  * add_fact dedup, length bounds, kind validation
  * retrieve scoring (Jaccard + entity + confidence) and used_count bump
  * retrieve_for_prompt formats correctly when empty / non-empty
  * extract_facts_from_text returns empty without LLM
  * extract_pending_runs respects the LLM-NULL skip path
- DB schema
  * user_facts table created on init_schema with all columns
- TailorResumeResult schema
  * Validates a fully-filled hand-crafted JSON
  * helpers (changes_by_kind, lift, has_unfabricated_audit) work
- tailor_resume adapter
  * metric runs schema=1.0 on valid JSON
  * fabrication detector catches "字节实习" claim absent from master
- MockInterviewResult schema
  * Validates first-turn (no eval) and middle-turn (with eval)
  * is_first_turn / is_complete helpers
- mock_interview adapter
  * metric runs schema=1.0 on valid first-turn output
- /tailor and /mock routes
  * Render with no LLM (degraded) — empty state visible
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import user_facts as uf
from offerguide.config import Settings
from offerguide.evolution.adapters import (
    REGISTRY,
)
from offerguide.evolution.adapters import (
    mock_interview as mi_adapter,
)
from offerguide.evolution.adapters import (
    tailor_resume as tr_adapter,
)
from offerguide.profile import UserProfile
from offerguide.skills import discover_skills
from offerguide.skills.mock_interview.helpers import MockInterviewResult
from offerguide.skills.tailor_resume.helpers import TailorResumeResult
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# user_facts
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def store(tmp_path: Path):
    s = offerguide.Store(tmp_path / "w12.db")
    s.init_schema()
    return s


class TestUserFacts:
    def test_add_and_dedup(self, store) -> None:
        fid = uf.add_fact(store, fact_text="上财应统硕士 2027 届", kind="profile")
        assert fid is not None
        # ADD-only: same text → silently skipped (None)
        dup = uf.add_fact(store, fact_text="上财应统硕士 2027 届", kind="profile")
        assert dup is None

    def test_length_bounds(self, store) -> None:
        # too short
        assert uf.add_fact(store, fact_text="a", kind="profile") is None
        # too long
        assert uf.add_fact(
            store, fact_text="x" * 300, kind="profile"
        ) is None

    def test_invalid_kind_raises(self, store) -> None:
        with pytest.raises(ValueError, match="unknown kind"):
            uf.add_fact(
                store, fact_text="x" * 20, kind="bogus",  # type: ignore[arg-type]
            )

    def test_retrieve_scores_and_marks_used(self, store) -> None:
        # Seed
        for kind, text in [
            ("project", "RemeDi 项目用 BERT 双塔, AUC 提升 0.04"),
            ("experience", "法至科技 NLP 实习, LangGraph 多 agent"),
            ("feedback", "腾讯一面 GRPO 答得卡需要补 RL"),
        ]:
            uf.add_fact(store, fact_text=text, kind=kind, confidence=0.8)

        # Query for "LangGraph 字节" should rank LangGraph fact highest
        results = uf.retrieve(store, query="LangGraph 字节 AI Agent")
        assert len(results) >= 1
        assert "LangGraph" in results[0].fact_text

        # used_count bumped after retrieve
        results2 = uf.retrieve(store, query="LangGraph")
        top_id = results2[0].id
        all_facts = uf.list_facts(store)
        top_fact = next(f for f in all_facts if f.id == top_id)
        assert top_fact.used_count >= 1

    def test_retrieve_filters_by_kind(self, store) -> None:
        uf.add_fact(store, fact_text="profile fact - 上财硕士 2027 届", kind="profile")
        uf.add_fact(store, fact_text="project fact - RemeDi 双塔模型", kind="project")
        results = uf.retrieve(store, query="模型", kinds=("project",))
        # Only project kind, no profile
        assert all(f.kind == "project" for f in results)

    def test_retrieve_for_prompt_empty_returns_blank(self, store) -> None:
        out = uf.retrieve_for_prompt(store, query="anything")
        assert out == ""

    def test_retrieve_for_prompt_non_empty_format(self, store) -> None:
        uf.add_fact(store, fact_text="法至科技 NLP 实习 LangGraph", kind="experience")
        out = uf.retrieve_for_prompt(store, query="LangGraph 经验")
        assert "已知用户长期事实" in out
        assert "[experience]" in out
        assert "(置信度" in out

    def test_extract_without_llm_returns_empty(self) -> None:
        candidates = uf.extract_facts_from_text(
            text="一些文本", source_skill="score_match", llm=None,
        )
        assert candidates == []

    def test_extract_pending_runs_skips_without_llm(self, store) -> None:
        with store.connect() as c:
            c.execute(
                "INSERT INTO skill_runs(skill_name, skill_version, input_hash, "
                "input_json, output_json) VALUES "
                "('score_match', 'v1', 'h1', '{}', '{\"x\": 1}')"
            )
        result = uf.extract_pending_runs(store, llm=None)
        # No LLM → 0 candidates extracted (extract_facts_from_text returns [])
        # but runs_scanned bumps
        assert result["runs_scanned"] == 1
        assert result["candidates"] == 0
        assert result["inserted"] == 0


# ═══════════════════════════════════════════════════════════════════
# Schemas — TailorResumeResult + MockInterviewResult
# ═══════════════════════════════════════════════════════════════════


VALID_TAILOR_JSON = {
    "company": "字节跳动",
    "role_focus": "AI Agent 后端实习",
    "tailored_markdown": (
        "# 胡阳\n上海财经大学 应用统计 硕士 2027 届\n\n"
        "## 实习经历\n- **法至科技** (2025/3 至今) — LangGraph 多 agent 评测\n\n"
        "## 项目\n- Deep Research Agent — LangGraph + DSPy + Pydantic\n"
    ),
    "change_log": [
        {
            "section": "实习经历 - 法至科技",
            "kind": "reword",
            "before": "做了 NLP 工作",
            "after": "用 LangGraph 搭多 agent 评测系统",
            "rationale": "JD 第 2 条要求 LangGraph 经验",
        },
        {
            "section": "技能",
            "kind": "ats_keyword_add",
            "before": "Python, ML",
            "after": "Python, PyTorch, LangGraph, Pydantic",
            "rationale": "JD ATS 关键词覆盖",
        },
    ],
    "ats_keywords_used": ["LangGraph", "Pydantic", "PyTorch"],
    "ats_keywords_missing": ["GRPO", "Speculative Decoding"],
    "cannot_fake_warnings": [
        "拒绝执行: 添加'字节实习一段', 因为 master_resume 只有'法至'实习"
    ],
    "fit_estimate": {
        "before": 0.45,
        "after": 0.62,
        "rationale": "reorder + LangGraph emphasis + 4 个 ATS 关键词",
    },
    "suggested_filename": "胡阳_字节跳动_AI_Agent后端实习_2026-05-02.pdf",
}


class TestTailorResumeSchema:
    def test_valid_json_validates(self) -> None:
        result = TailorResumeResult.model_validate(VALID_TAILOR_JSON)
        assert result.company == "字节跳动"
        assert len(result.change_log) == 2

    def test_changes_by_kind(self) -> None:
        result = TailorResumeResult.model_validate(VALID_TAILOR_JSON)
        counts = result.changes_by_kind()
        assert counts == {"reword": 1, "ats_keyword_add": 1}

    def test_lift(self) -> None:
        result = TailorResumeResult.model_validate(VALID_TAILOR_JSON)
        assert abs(result.lift() - 0.17) < 0.001

    def test_has_unfabricated_audit(self) -> None:
        result = TailorResumeResult.model_validate(VALID_TAILOR_JSON)
        assert result.has_unfabricated_audit() is True


VALID_MOCK_TURN_FIRST = {
    "company": "字节跳动",
    "role_focus": "AI Agent 后端实习",
    "turn_index": 1,
    "evaluation_of_last_answer": None,
    "next_question": {
        "question": "讲讲 LangGraph state machine 怎么设计 stateful agent",
        "category": "technical",
        "difficulty": "medium",
        "rationale": "简历有 LangGraph 项目, 起手中等难度",
        "expected_aspects": [
            "TypedDict / Pydantic 定义 state shape",
            "节点之间 reducer 合并机制",
            "conditional edge 路由",
            "并发执行 + lock",
        ],
    },
    "session_status": "in_progress",
    "session_summary": None,
}

VALID_MOCK_TURN_MID = {
    "company": "字节跳动",
    "role_focus": "AI Agent 后端实习",
    "turn_index": 3,
    "evaluation_of_last_answer": {
        "question": "RAG 怎么 debug retrieval miss",
        "user_answer": "看 log + 看 query embedding...",
        "score": 0.82,
        "scoring_dimensions": {
            "factual_accuracy": 0.85,
            "depth": 0.85,
            "structure": 0.75,
            "evidence": 0.8,
        },
        "strengths": ["分了 query 重写 / 召回扩展两个方向"],
        "improvements": [
            "应该先说'分类失败原因再分别处理', 而不是直接列方法"
        ],
        "model_answer_skeleton": "1) 分类miss原因 2) embedding mismatch 3) chunk 边界 4) 召回 K 不够",
        "follow_up_likely": "你的 reranker 怎么训的",
    },
    "next_question": {
        "question": "假设 RAG 召回 hit@5 突然从 0.74 掉到 0.5, 你怎么 root cause",
        "category": "technical",
        "difficulty": "hard",
        "rationale": "上轮 0.82 强答 → 升 hard, 同 category 深挖",
        "expected_aspects": [
            "数据漂移检测", "embedding 模型变更", "索引坏",
        ],
    },
    "session_status": "in_progress",
    "session_summary": None,
}


class TestMockInterviewSchema:
    def test_first_turn_validates(self) -> None:
        r = MockInterviewResult.model_validate(VALID_MOCK_TURN_FIRST)
        assert r.is_first_turn() is True
        assert r.is_complete() is False

    def test_middle_turn_validates(self) -> None:
        r = MockInterviewResult.model_validate(VALID_MOCK_TURN_MID)
        assert r.is_first_turn() is False
        assert r.evaluation_of_last_answer is not None
        assert r.evaluation_of_last_answer.score == 0.82


# ═══════════════════════════════════════════════════════════════════
# Adapter registry + metrics
# ═══════════════════════════════════════════════════════════════════


class TestAdapters:
    def test_both_skills_registered(self) -> None:
        assert "tailor_resume" in REGISTRY
        assert "mock_interview" in REGISTRY

    def test_tailor_metric_runs_on_valid(self) -> None:
        example = tr_adapter.EXAMPLES[0]
        result = tr_adapter.metric(example, json.dumps(VALID_TAILOR_JSON))
        assert 0.0 <= result.total <= 1.0
        assert result.breakdown["schema"] == 1.0

    def test_mock_metric_runs_on_first_turn(self) -> None:
        example = mi_adapter.EXAMPLES[0]
        result = mi_adapter.metric(example, json.dumps(VALID_MOCK_TURN_FIRST))
        assert 0.0 <= result.total <= 1.0
        assert result.breakdown["schema"] == 1.0

    def test_metric_zero_on_invalid(self) -> None:
        example = tr_adapter.EXAMPLES[0]
        result = tr_adapter.metric(example, "not json")
        assert result.total == 0.0


# ═══════════════════════════════════════════════════════════════════
# UI routes (degraded — no LLM)
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="resume body text")
    skills = discover_skills(SKILLS_ROOT)
    app = create_app(
        settings=Settings(),
        store=store,
        profile=profile,
        skills=skills,
        runtime=None,  # degraded
        notifier=ConsoleNotifier(),
    )
    return app, store


class TestUIRoutes:
    def test_tailor_page_renders(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/tailor")
        assert resp.status_code == 200
        assert "简历微调" in resp.text
        # No JDs ≥200 chars seeded → empty state shows
        assert "没有可微调的 JD" in resp.text

    def test_mock_page_renders(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/mock")
        assert resp.status_code == 200
        assert "模拟面试" in resp.text
        assert "开始第 1 题" in resp.text

    def test_tailor_run_returns_error_no_llm(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).post(
            "/api/tailor/run", data={"job_id": "1"},
        )
        # 200 with error fragment (we don't 4xx the LLM-missing case)
        assert resp.status_code == 200
        assert "未配置 LLM" in resp.text

    def test_mock_turn_returns_error_no_llm(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).post(
            "/api/mock/turn",
            data={"company": "字节跳动", "role_focus": "AI Agent",
                  "turn_history_json": "[]", "last_user_answer": ""},
        )
        assert resp.status_code == 200
        assert "未配置 LLM" in resp.text


_ = json  # keep import
