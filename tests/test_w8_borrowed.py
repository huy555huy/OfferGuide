"""W8 +2 — Tests for Career-Ops / Resume-Matcher inspired features.

Story bank (Career-Ops STAR+Reflection pattern) + write_cover_letter
SKILL (Career-Ops + Resume-Matcher inspiration).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import offerguide
from offerguide import story_bank
from offerguide.config import Settings
from offerguide.evolution.adapters import get_adapter
from offerguide.evolution.adapters import write_cover_letter as wcl_adapter
from offerguide.profile import UserProfile
from offerguide.skills import discover_skills, load_skill
from offerguide.skills.write_cover_letter.helpers import CoverLetterResult
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def _make_store(tmp_path: Path) -> offerguide.Store:
    store = offerguide.Store(tmp_path / "borrowed.db")
    store.init_schema()
    return store


# ═══════════════════════════════════════════════════════════════════
# STORY BANK
# ═══════════════════════════════════════════════════════════════════


class TestStoryBankCRUD:
    def test_insert_round_trips(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        s = story_bank.insert(
            store,
            title="法至产品分歧",
            situation="实习期间产品要 ChatGPT 即时响应",
            task="说服产品保留 Deep Research 多步推理",
            action="拉数据 + 出方案: 双模式切换",
            result="产品同意保留 + 用户切换率 35%",
            reflection="技术决策也得讲故事讲到决策者",
            tags=["collaboration", "conflict"],
            confidence=0.7,
        )
        assert s.id > 0
        assert s.confidence == 0.7
        assert s.tags == ["collaboration", "conflict"]

        loaded = story_bank.get(store, s.id)
        assert loaded is not None
        assert loaded.title == "法至产品分歧"
        assert loaded.reflection is not None

    def test_required_fields(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        with pytest.raises(ValueError):
            story_bank.insert(
                store, title="", situation="x", task="y", action="z", result="r",
            )

    def test_search_by_tag_bumps_used_count(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        s = story_bank.insert(
            store, title="x", situation="s", task="t", action="a", result="r",
            tags=["collaboration"],
        )
        assert s.used_count == 0

        results = story_bank.search_by_tag(store, "collaboration")
        assert len(results) == 1
        # After retrieval, used_count is bumped
        loaded = story_bank.get(store, s.id)
        assert loaded is not None
        assert loaded.used_count == 1

    def test_search_returns_least_used_first(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        story_bank.insert(
            store, title="A", situation="s", task="t", action="a", result="r",
            tags=["learning"],
        )
        s2 = story_bank.insert(
            store, title="B", situation="s", task="t", action="a", result="r",
            tags=["learning"],
        )
        # Bump s1's used_count manually
        story_bank.search_by_tag(store, "learning")  # both bumped to 1
        story_bank.search_by_tag(store, "learning")  # both bumped to 2
        # Reset s2 to 0 by direct DB op
        with store.connect() as conn:
            conn.execute("UPDATE behavioral_stories SET used_count = 0 WHERE id = ?", (s2.id,))
        results = story_bank.search_by_tag(store, "learning")
        # s2 should come first (lower used_count)
        assert results[0].id == s2.id

    def test_render_for_skill(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        story_bank.insert(
            store, title="X", situation="S text",
            task="T text", action="A text", result="R text",
            reflection="Ref text", tags=["learning"],
        )
        rendered = story_bank.render_for_skill(
            story_bank.list_all(store), max_chars=500
        )
        assert "S: S text" in rendered
        assert "T: T text" in rendered
        assert "+R: Ref text" in rendered

    def test_recommended_tags_exposed(self) -> None:
        assert "collaboration" in story_bank.RECOMMENDED_TAGS
        assert "conflict" in story_bank.RECOMMENDED_TAGS
        assert "tradeoff" in story_bank.RECOMMENDED_TAGS


# ═══════════════════════════════════════════════════════════════════
# /stories endpoint
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_setup(tmp_path: Path):
    store = _make_store(tmp_path)
    profile = UserProfile(raw_resume_text="x")
    skills = discover_skills(SKILLS_ROOT)
    app = create_app(
        settings=Settings(), store=store, profile=profile,
        skills=skills, runtime=None, notifier=ConsoleNotifier(),
    )
    return app, store


class TestStoriesEndpoint:
    def test_get_stories_empty(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/stories")
        assert resp.status_code == 200
        assert "STAR + Reflection 故事库" in resp.text
        assert "故事库还空着" in resp.text

    def test_post_inserts_story(self, app_setup) -> None:
        app, store = app_setup
        resp = TestClient(app).post(
            "/api/stories/insert",
            data={
                "title": "RemeDi 训练崩溃",
                "situation": "训到 30B token 突然 loss spike",
                "task": "找出原因并恢复",
                "action": "对比 wandb logs + 二分回退",
                "result": "定位到 LR scheduler bug，修后正常",
                "reflection": "训练前先冻结实验跑短跑",
                "tags": "failure, learning",
                "confidence": "0.8",
            },
        )
        assert resp.status_code == 200
        rows = story_bank.list_all(store)
        assert len(rows) == 1
        assert rows[0].title == "RemeDi 训练崩溃"
        assert rows[0].tags == ["failure", "learning"]

    def test_topbar_has_stories_link(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/")
        assert 'href="/stories"' in resp.text


# ═══════════════════════════════════════════════════════════════════
# write_cover_letter SKILL — Pydantic schema
# ═══════════════════════════════════════════════════════════════════


def _good_cover_letter() -> dict:
    return {
        "opening_hook": "看到 RemeDi 双流架构 + GRPO 这条路径已经落到 Doubao 后训练 pipeline 上",
        "narrative_body": [
            "我去年在法至科技实习期间也踩过类似 trade-off，最终选择 evidence-centric 上下文管理。",
            "RemeDi 项目用 GRPO 训扩散语言模型，遇到 reward hacking 问题，最终通过 ... 解决。",
        ],
        "closing_call_to_action": "可即刻入职 2026 年 5 月开始 8 周实习，方便本周三前完成笔试。",
        "customization_signals": [
            "JD 第 3 条要求 DeepSpeed 经验，简历里 ZeRO-2 项目对得上",
            "提到 Seed 实验室最近在做 SFT/DPO/GRPO 多线程对比",
        ],
        "ats_keywords_used": ["PyTorch", "GRPO", "DeepSpeed", "Transformer", "RLHF"],
        "ai_risk_warnings": [],
        "suggested_tone": "warm_concise",
        "personalization_score": 0.78,
        "overall_word_count": 220,
    }


class TestCoverLetterSchema:
    def test_validates_well_formed(self) -> None:
        result = CoverLetterResult.model_validate(_good_cover_letter())
        assert result.suggested_tone == "warm_concise"
        assert len(result.narrative_body) == 2
        assert result.personalization_score == 0.78

    def test_rejects_extra_keys(self) -> None:
        bad = _good_cover_letter()
        bad["surprise_field"] = "x"
        with pytest.raises(ValidationError):
            CoverLetterResult.model_validate(bad)

    def test_rejects_invalid_tone(self) -> None:
        bad = _good_cover_letter()
        bad["suggested_tone"] = "casual_emoji"
        with pytest.raises(ValidationError):
            CoverLetterResult.model_validate(bad)

    def test_rejects_score_oor(self) -> None:
        bad = _good_cover_letter()
        bad["personalization_score"] = 1.2
        with pytest.raises(ValidationError):
            CoverLetterResult.model_validate(bad)

    def test_word_count_min(self) -> None:
        bad = _good_cover_letter()
        bad["overall_word_count"] = 30  # min is 50
        with pytest.raises(ValidationError):
            CoverLetterResult.model_validate(bad)

    def test_render_plain(self) -> None:
        r = CoverLetterResult.model_validate(_good_cover_letter())
        text = r.render_plain()
        # Opening hook + 2 paragraphs + closing = 4 blocks separated by blank lines
        assert text.count("\n\n") == 3
        assert "RemeDi" in text

    def test_has_high_ai_risk(self) -> None:
        good = _good_cover_letter()
        good["ai_risk_warnings"] = ["a", "b", "c"]
        r = CoverLetterResult.model_validate(good)
        assert r.has_high_ai_risk() is True

        good["ai_risk_warnings"] = ["a"]
        r = CoverLetterResult.model_validate(good)
        assert r.has_high_ai_risk() is False


# ═══════════════════════════════════════════════════════════════════
# write_cover_letter SKILL — adapter & metric
# ═══════════════════════════════════════════════════════════════════


def _example():
    return next(e for e in wcl_adapter.EXAMPLES if e.name == "bytedance_seed_intern")


class TestCoverLetterAdapter:
    def test_skill_loads(self) -> None:
        spec = load_skill(SKILLS_ROOT / "write_cover_letter")
        assert spec.name == "write_cover_letter"
        assert spec.version == "0.1.0"

    def test_registered(self) -> None:
        assert get_adapter("write_cover_letter") is wcl_adapter

    def test_invalid_json_zero(self) -> None:
        ex = _example()
        result = wcl_adapter.metric(ex, "not json")
        assert result.total == 0.0

    def test_good_letter_scores_high(self) -> None:
        ex = _example()
        result = wcl_adapter.metric(ex, json.dumps(_good_cover_letter(), ensure_ascii=False))
        assert result.breakdown["schema"] == 1.0
        assert result.total > 0.6

    def test_ai_giveaway_phrases_penalized(self) -> None:
        ex = _example()
        bad = _good_cover_letter()
        bad["narrative_body"] = [
            "我谨此致函向贵公司表达由衷的兴趣。本人热情饱满，乐于挑战。",
            "RemeDi 项目用 GRPO 训扩散语言模型...",
        ]
        result = wcl_adapter.metric(ex, json.dumps(bad, ensure_ascii=False))
        assert result.breakdown["ai_risk_clean"] < 1.0

    def test_too_few_keywords_penalized(self) -> None:
        ex = _example()
        bad = _good_cover_letter()
        bad["ats_keywords_used"] = ["PyTorch"]  # only 1
        result = wcl_adapter.metric(ex, json.dumps(bad, ensure_ascii=False))
        assert result.breakdown["ats_density"] < 1.0

    def test_too_long_penalized(self) -> None:
        ex = _example()
        bad = _good_cover_letter()
        bad["overall_word_count"] = 600  # max for this example is 350
        result = wcl_adapter.metric(ex, json.dumps(bad, ensure_ascii=False))
        assert result.breakdown["length_in_band"] < 1.0

    def test_overclaimed_personalization_penalized(self) -> None:
        ex = _example()
        bad = _good_cover_letter()
        bad["customization_signals"] = []  # 0 signals
        bad["ats_keywords_used"] = []  # 0 keywords
        bad["personalization_score"] = 0.95  # claimed very high
        result = wcl_adapter.metric(ex, json.dumps(bad, ensure_ascii=False))
        assert result.breakdown["personalization_realism"] < 0.5

    def test_split_train_val_deterministic(self) -> None:
        a1, b1 = wcl_adapter.split_train_val(seed=0)
        a2, b2 = wcl_adapter.split_train_val(seed=0)
        assert [e.name for e in a1] == [e.name for e in a2]


# ═══════════════════════════════════════════════════════════════════
# Agent integration: cover_letter action runs through graph
# ═══════════════════════════════════════════════════════════════════


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._next_id = 700

    def invoke(self, spec, inputs, **_):
        from offerguide.skills import SkillResult
        self._next_id += 1
        self.calls.append((spec.name, dict(inputs)))
        parsed = (
            _good_cover_letter()
            if spec.name == "write_cover_letter"
            else {"_": "_"}
        )
        return SkillResult(
            raw_text=json.dumps(parsed, ensure_ascii=False),
            parsed=parsed, skill_name=spec.name, skill_version=spec.version,
            skill_run_id=self._next_id, input_hash="x",
            cost_usd=0.0001, latency_ms=42,
        )


class TestCoverLetterAgentIntegration:
    def test_cover_letter_action_runs_only_cover_letter(self, tmp_path: Path) -> None:
        from offerguide.agent import build_graph

        store = _make_store(tmp_path)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)  # type: ignore[arg-type]

        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": "x"}],
                "requested_action": "cover_letter",
                "job_text": "AI Agent 实习",
                "user_profile_text": "胡阳",
                "company": "字节跳动",
            }
        )
        called = [c[0] for c in runtime.calls]
        assert called == ["write_cover_letter"]
        assert result.get("cover_letter_result") is not None

    def test_missing_company_errors(self, tmp_path: Path) -> None:
        from offerguide.agent import build_graph

        store = _make_store(tmp_path)
        skills = discover_skills(SKILLS_ROOT)
        runtime = _FakeRuntime()
        graph = build_graph(skills=skills, runtime=runtime, store=store)  # type: ignore[arg-type]

        result = graph.invoke(
            {
                "messages": [],
                "requested_action": "cover_letter",
                "job_text": "JD",
                "user_profile_text": "P",
                # no company
            }
        )
        assert result.get("error") is not None
        assert "company" in str(result["error"]).lower()
        assert runtime.calls == []
