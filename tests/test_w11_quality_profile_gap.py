"""W11 — Corpus quality classifier + successful_profile + profile_resume_gap.

Coverage:

- corpus_quality
  * deterministic pre-filter catches blatant marketer posts
  * classify_one returns sane signals for real-content (no LLM path)
  * classify_pending updates DB rows with verdicts
  * fetch_high_quality respects min_score + content_kind filters
  * fetch_high_quality falls back when role match is empty
- DB schema
  * interview_experiences gains content_kind/quality_score/quality_signals_json
    /quality_classified_at on init_schema (and on _migrate of older DB)
- SuccessfulProfileResult schema
  * confidence() heuristic — small samples → low score
  * model_validate accepts a fully-filled profile JSON
- ProfileResumeGapResult schema
  * model_validate accepts Chinese-keyed JSON via aliases
  * total_gaps + short_term_total_hours + has_unfakeable_blocker work
- Adapter registry
  * successful_profile + profile_resume_gap registered
  * Each metric runs without crash on a hand-crafted valid output
- /profile/{company} route
  * empty samples → friendly empty state with helpful instructions
  * profile rendered when LLM-not-configured but samples exist
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import corpus_quality
from offerguide.config import Settings
from offerguide.evolution.adapters import (
    REGISTRY,
)
from offerguide.evolution.adapters import (
    profile_resume_gap as gap_adapter,
)
from offerguide.evolution.adapters import (
    successful_profile as sp_adapter,
)
from offerguide.profile import UserProfile
from offerguide.skills import discover_skills
from offerguide.skills.profile_resume_gap.helpers import ProfileResumeGapResult
from offerguide.skills.successful_profile.helpers import SuccessfulProfileResult
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# corpus_quality
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def store(tmp_path: Path):
    s = offerguide.Store(tmp_path / "w11.db")
    s.init_schema()
    return s


def _seed_corpus(store, *, raw: str, source: str, company: str = "X",
                 role_hint: str | None = None) -> int:
    h = hashlib.sha256(raw.encode()).hexdigest()
    with store.connect() as c:
        cur = c.execute(
            "INSERT INTO interview_experiences"
            "(company, role_hint, raw_text, source, content_hash) "
            "VALUES (?,?,?,?,?) RETURNING id",
            (company, role_hint, raw, source, h),
        )
        return int(cur.fetchone()[0])


class TestObviousMarketerFilter:
    def test_blatant_marketer_skips_llm(self) -> None:
        text = (
            "加微信 wx123 训练营 包过 大厂内推 资料包 限免领取 "
            "1v1 辅导 课程 价格 999 仅限今日 加我 dm"
        )
        verdict = corpus_quality.classify_one(text=text, llm=None)
        assert verdict.signals.kind == "marketer"
        assert verdict.signals.quality_score == 0.0
        assert verdict.skipped_llm is True

    def test_real_content_with_marketer_words_falls_through(self) -> None:
        # Has both marketer-ish and real content — should NOT be auto-marketer
        text = (
            "腾讯一面: 1) 手撕反转链表 2) 深挖 RemeDi 项目, "
            "面试官追问 grad clip 怎么算的我没答上来 二面挂了。"
            "如果想看更多 面经 关注公众号 (一句话引流但内容真实)"
        )
        verdict = corpus_quality.classify_one(text=text, llm=None)
        # No LLM, falls through to "other" with default 0.5 (gray-zone case)
        assert verdict.signals.kind == "other"
        assert verdict.skipped_llm is True

    def test_too_short_to_classify(self) -> None:
        text = "加 V 包过"
        verdict = corpus_quality.classify_one(text=text, llm=None)
        assert verdict.signals.kind == "other"  # too short → no marketer call


class TestClassifyPending:
    def test_persists_verdict_to_row(self, store) -> None:
        marketer_id = _seed_corpus(
            store,
            raw=(
                "加微信 wx123 训练营 包过 大厂内推 资料包 限免领取 "
                "1v1 辅导 课程 价格 999 仅限今日 加我 dm"
            ),
            source="m1",
        )
        _seed_corpus(
            store,
            raw="腾讯 一面 30min: 1) leetcode 206 反转链表 2) RemeDi 项目深挖, "
                "我训练崩溃用 grad clip 缓解 二面挂了 投递周期 3 周",
            source="r1",
        )
        result = corpus_quality.classify_pending(store, llm=None)
        assert result["processed"] == 2
        assert result["marketer"] == 1

        with store.connect() as c:
            rows = c.execute(
                "SELECT id, content_kind, quality_score, quality_classified_at "
                "FROM interview_experiences ORDER BY id"
            ).fetchall()
        ids = {r[0]: r for r in rows}
        assert ids[marketer_id][1] == "marketer"
        assert ids[marketer_id][2] == 0.0
        assert ids[marketer_id][3] is not None  # classified_at set

    def test_classified_rows_skipped_on_rerun(self, store) -> None:
        _seed_corpus(
            store,
            raw=(
                "加微信 wx123 训练营 包过 大厂内推 资料包 限免领取 "
                "1v1 辅导 课程 价格 999 加我 dm"
            ),
            source="m1",
        )
        first = corpus_quality.classify_pending(store, llm=None)
        second = corpus_quality.classify_pending(store, llm=None)
        assert first["processed"] == 1
        assert second["processed"] == 0


class TestFetchHighQuality:
    def test_filters_by_min_score(self, store) -> None:
        _seed_corpus(
            store,
            raw=(
                "加微信 wx123 训练营 包过 大厂内推 资料包 限免领取 "
                "1v1 辅导 课程 价格 999 加我 dm"
            ),
            source="bad", company="字节跳动",
        )
        good = _seed_corpus(
            store,
            raw=("字节跳动 AI Agent offer 复盘: 项目深度被表扬, "
                 "leetcode 206 一面手撕通过, system design 答得不错。"
                 "教育背景：985 计算机硕士。投递周期 3 周。"),
            source="good", company="字节跳动",
        )
        # Manually set good's score above threshold
        with store.connect() as c:
            c.execute(
                "UPDATE interview_experiences "
                "SET quality_score = 0.8, content_kind = 'offer_post', "
                "    quality_classified_at = julianday('now') WHERE id = ?",
                (good,),
            )
        # Run classify on remaining (the marketer row)
        corpus_quality.classify_pending(store, llm=None)

        results = corpus_quality.fetch_high_quality(store, company="字节跳动")
        assert len(results) == 1
        assert results[0]["id"] == good
        assert results[0]["quality_score"] == 0.8

    def test_role_filter_falls_back_when_empty(self, store) -> None:
        _id = _seed_corpus(
            store, raw="腾讯 后端 一面经历", source="t1",
            company="腾讯", role_hint="后端",
        )
        with store.connect() as c:
            c.execute(
                "UPDATE interview_experiences SET quality_score = 0.7, "
                "content_kind = 'interview', "
                "quality_classified_at = julianday('now') WHERE id = ?",
                (_id,),
            )
        # Strict role filter: 算法 (no match) — falls back to no role
        results = corpus_quality.fetch_high_quality(
            store, company="腾讯", role_hint="算法",
        )
        # Should fall back to 1 result (the 后端 one)
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════════


VALID_PROFILE_JSON = {
    "company": "字节跳动",
    "role_focus": "AI Agent 后端实习",
    "evidence_count": 3,
    "evidence_kinds": ["offer_post", "interview", "project_share"],
    "background_pattern": {
        "education_level": "硕士",
        "school_tier": "985 头部",
        "majors": ["计算机"],
        "internships": ["美团"],
        "competitions": [],
        "publications": [],
    },
    "skill_pattern": {
        "must_have": ["Python", "PyTorch", "LangGraph"],
        "highly_valued": ["GRPO"],
        "differentiators": [],
    },
    "project_pattern": {
        "typical_project_themes": ["LLM agent"],
        "common_tech_stacks": ["LangGraph"],
        "scale_signals": ["200+ 测试用例"],
        "outcome_signals": [],
    },
    "interview_pattern": {
        "common_questions": [
            {"question": "LangGraph state 设计", "category": "technical",
             "evidence_count": 2}
        ],
        "behavioral_themes": [],
        "decision_factors": ["项目深度"],
    },
    "why_they_passed": [
        "项目深度被多次表扬 (来自 2 条 offer_post)",
        "对 RAG 工程取舍熟悉 (来自 2 条 interview)",
    ],
    "evidence_sources": [],
    "uncertainty_notes": [],
}


class TestSuccessfulProfileSchema:
    def test_valid_json_validates(self) -> None:
        result = SuccessfulProfileResult.model_validate(VALID_PROFILE_JSON)
        assert result.company == "字节跳动"
        assert len(result.why_they_passed) == 2

    def test_confidence_grows_with_evidence(self) -> None:
        small = VALID_PROFILE_JSON | {"evidence_count": 1, "uncertainty_notes": []}
        big = VALID_PROFILE_JSON | {"evidence_count": 8, "uncertainty_notes": []}
        s = SuccessfulProfileResult.model_validate(small)
        b = SuccessfulProfileResult.model_validate(big)
        assert b.confidence() > s.confidence()

    def test_uncertainty_notes_lower_confidence(self) -> None:
        clean = VALID_PROFILE_JSON | {"evidence_count": 5}
        worried = clean | {
            "uncertainty_notes": ["a", "b", "c", "d", "e"]
        }
        c = SuccessfulProfileResult.model_validate(clean)
        w = SuccessfulProfileResult.model_validate(worried)
        assert c.confidence() > w.confidence()


VALID_GAP_JSON = {
    "company": "字节跳动",
    "role_focus": "AI Agent 后端实习",
    "已具备": [
        {"topic": "LangGraph", "evidence_in_resume": "Deep Research Agent 项目",
         "evidence_in_profile": "must_have: LangGraph", "strength": "strong"},
    ],
    "短期能补 (≤2周)": [
        {"topic": "GRPO 推导", "why_missing": "简历未提",
         "concrete_action": "读 paper + 实现 minimal demo, 预计 8 小时",
         "estimated_hours": 8,
         "skill_signal_after": "实现 GRPO minimal demo 并开源"},
    ],
    "短期补不了": [],
    "不能编": [
        {"topic": "本科非 985", "why_unfakeable": "学信网必查",
         "reframe_strategy": "强调 ACM 银 + 论文复现"},
    ],
    "投递建议": {
        "verdict": "go",
        "rationale_chinese": "must_have 已具备 80%，短期能补一项即可投递",
        "top_3_pre_apply_actions": [
            "实现 GRPO demo 并加到简历",
            "刷 leetcode hot100 后端类题",
            "整理 cover letter 并审 AI 风险",
        ],
    },
    "calibration": {
        "covered_profile_fields": 4,
        "skipped_due_to_low_evidence": [],
    },
}


class TestProfileResumeGapSchema:
    def test_valid_json_with_chinese_keys_validates(self) -> None:
        # by_alias=True needed because keys are Chinese
        result = ProfileResumeGapResult.model_validate(VALID_GAP_JSON)
        assert result.apply_advice.verdict == "go"
        assert len(result.have) == 1

    def test_helper_methods(self) -> None:
        result = ProfileResumeGapResult.model_validate(VALID_GAP_JSON)
        assert result.total_gaps() == 2
        assert result.short_term_total_hours() == 8
        assert result.has_unfakeable_blocker() is True


# ═══════════════════════════════════════════════════════════════════
# Adapters
# ═══════════════════════════════════════════════════════════════════


class TestAdapters:
    def test_both_skills_in_registry(self) -> None:
        assert "successful_profile" in REGISTRY
        assert "profile_resume_gap" in REGISTRY

    def test_successful_profile_metric_runs(self) -> None:
        example = sp_adapter.EXAMPLES[0]
        result = sp_adapter.metric(example, json.dumps(VALID_PROFILE_JSON))
        assert 0.0 <= result.total <= 1.0
        # should at least pass schema
        assert result.breakdown["schema"] == 1.0

    def test_profile_gap_metric_runs(self) -> None:
        example = gap_adapter.EXAMPLES[0]
        result = gap_adapter.metric(example, json.dumps(VALID_GAP_JSON))
        assert 0.0 <= result.total <= 1.0
        assert result.breakdown["schema"] == 1.0

    def test_metric_zero_on_invalid_json(self) -> None:
        example = sp_adapter.EXAMPLES[0]
        result = sp_adapter.metric(example, "not json at all")
        assert result.total == 0.0


# ═══════════════════════════════════════════════════════════════════
# /profile/{company} route
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="resume text")
    skills = discover_skills(SKILLS_ROOT)
    app = create_app(
        settings=Settings(),
        store=store,
        profile=profile,
        skills=skills,
        runtime=None,  # explicitly no LLM → degraded mode
        notifier=ConsoleNotifier(),
    )
    return app, store


class TestProfileRoute:
    def test_empty_samples_renders_empty_state(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/profile/某某公司")
        assert resp.status_code == 200
        assert "样本不足" in resp.text or "还没有该公司" in resp.text

    def test_with_samples_no_llm_renders_sample_list(self, app_setup) -> None:
        app, store = app_setup
        # Seed one quality-classified sample
        with store.connect() as c:
            h = hashlib.sha256(b"x").hexdigest()
            c.execute(
                "INSERT INTO interview_experiences"
                "(company, role_hint, raw_text, source, content_hash, "
                " content_kind, quality_score, quality_classified_at) "
                "VALUES (?,?,?,?,?, 'offer_post', 0.8, julianday('now'))",
                ("字节跳动", "AI Agent", "字节 AI Agent offer 复盘 ...",
                 "manual_paste", h),
            )
        resp = TestClient(app).get("/profile/字节跳动")
        assert resp.status_code == 200
        # Sample card visible even without LLM
        assert "未配置简历或 LLM" in resp.text or "未配置 LLM" in resp.text
        assert "字节跳动" in resp.text
