"""W13.4 apply_assistant SKILL + /apply UI + lifecycle tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import offerguide
from offerguide.config import Settings
from offerguide.llm import LLMResponse
from offerguide.profile import UserProfile
from offerguide.skills import SkillRuntime, discover_skills, load_skill
from offerguide.skills.apply_assistant.helpers import (
    ApplyPackage,
)
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# SKILL frontmatter + helpers
# ═══════════════════════════════════════════════════════════════════


class TestSkillLoading:
    def test_skill_md_loads(self):
        spec = load_skill(SKILLS_ROOT / "apply_assistant")
        assert spec.name == "apply_assistant"
        assert spec.version
        assert "apply" in spec.tags or "application" in spec.tags
        assert "company" in spec.inputs
        assert "job_text" in spec.inputs
        assert "user_profile" in spec.inputs

    def test_discover_picks_it_up(self):
        names = {s.name for s in discover_skills(SKILLS_ROOT)}
        assert "apply_assistant" in names


# ═══════════════════════════════════════════════════════════════════
# Pydantic schema validation
# ═══════════════════════════════════════════════════════════════════


VALID_PACKAGE = {
    "company": "字节跳动",
    "role_focus": "AI Agent 后端实习",
    "self_intro_snippet": {
        "platform_hint": "boss_zhipin",
        "text": "看到 LLM Agent 实习, 想投。我用 LangGraph 做了 Deep Research Agent (双层 state machine), "
                "跟你们 JD 里 'agent runtime' 那条契合。",
        "rationale": "用 1 个具体项目对齐 JD 关键词",
    },
    "qa_templates": [
        {
            "question": "你为什么选我们?",
            "category": "motivation",
            "answer": "字节 AI Lab 的 OpenSora 是我跟最久的开源项目, 来贡献感觉是顺其自然的事。"
                      "另外我用过你们的 Doubao API, 觉得 latency 控制很厉害。" * 2,
            "anti_patterns": ["别说想学习成长"],
            "personalization_score": 0.85,
        }
    ],
    "submission_strategy": {
        "best_time_window": "周一早 10-11 点",
        "platform_specific_tips": ["Boss 直聊为主", "字节有内推码值得找"],
        "follow_up_plan": "5-7 天没回再 Boss 二次发, 不要 '在吗'",
        "expected_response_window_days": 7,
    },
    "pre_submit_checklist": [
        "简历已 tailor 过且文件名格式正确",
        "附件 < 5MB",
    ],
    "skip_reasons": [],
    "confidence": 0.85,
}


class TestPydanticSchema:
    def test_valid_package_parses(self):
        pkg = ApplyPackage(**VALID_PACKAGE)
        assert pkg.company == "字节跳动"
        assert pkg.confidence == 0.85
        assert not pkg.should_skip

    def test_skip_package(self):
        bad = dict(VALID_PACKAGE)
        bad["skip_reasons"] = ["JD 要求 5 年经验, 应届不符合"]
        pkg = ApplyPackage(**bad)
        assert pkg.should_skip
        assert "5 年经验" in pkg.skip_reasons[0]

    def test_extra_field_rejected(self):
        bad = dict(VALID_PACKAGE)
        bad["secret_extra"] = "shouldn't be here"
        with pytest.raises(ValidationError):
            ApplyPackage(**bad)

    def test_self_intro_too_short_rejected(self):
        bad_intro = dict(VALID_PACKAGE)
        bad_intro["self_intro_snippet"] = {
            "platform_hint": "boss_zhipin",
            "text": "短",  # < 20 chars
            "rationale": "x",
        }
        with pytest.raises(ValidationError):
            ApplyPackage(**bad_intro)

    def test_personalization_score_clamped(self):
        bad_qa = dict(VALID_PACKAGE)
        bad_qa["qa_templates"] = [{
            "question": "x",
            "category": "motivation",
            "answer": "x" * 50,
            "anti_patterns": [],
            "personalization_score": 1.5,  # > 1.0
        }]
        with pytest.raises(ValidationError):
            ApplyPackage(**bad_qa)


# ═══════════════════════════════════════════════════════════════════
# /apply route + /api/apply endpoints
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_client(tmp_path):
    store = offerguide.Store(tmp_path / "apply.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)
    s = Settings(deepseek_api_key="dummy", deepseek_base_url="x", default_model="m")

    class _SkillStubLLM:
        def chat(self, messages, **kw):
            return LLMResponse(
                content=json.dumps(VALID_PACKAGE, ensure_ascii=False),
                model="stub",
            )
    runtime = SkillRuntime(llm=_SkillStubLLM(), store=store)

    profile = UserProfile(
        raw_resume_text="TestUser, AI 方向, LangGraph 经验",
        source_pdf="/tmp/fake.pdf",
    )
    app = create_app(
        settings=s, store=store, profile=profile,
        skills=skills, runtime=runtime, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestApplyRoute:
    def test_view_404_for_missing_job(self, app_client):
        client, _ = app_client
        resp = client.get("/apply/99999")
        assert resp.status_code == 404

    def test_view_renders_for_existing_job(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, "
                "  raw_text, content_hash) VALUES (?,?,?,?,?,?,?)",
                ("manual", "x", "https://example.com",
                 "AI Agent 后端", "字节跳动",
                 "完整 JD 全文" * 30, "h1"),
            )
        resp = client.get("/apply/1")
        assert resp.status_code == 200
        assert "字节跳动" in resp.text
        assert "AI Agent 后端" in resp.text

    def test_generate_endpoint_invokes_skill(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, "
                "  raw_text, content_hash) VALUES (?,?,?,?,?,?,?)",
                ("manual", "x", "x", "AI Agent", "字节",
                 "完整 JD 全文" * 30, "h1"),
            )
        resp = client.post("/api/apply/1/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill_run_id"] is not None
        assert data["package"]["company"] == "字节跳动"
        assert "self_intro_snippet" in data["package"]

    def test_generate_rejects_thin_jd(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, "
                "  raw_text, content_hash) VALUES (?,?,?,?,?,?,?)",
                ("manual", "thin", "x", "x", "x", "短", "h_thin"),
            )
        resp = client.post("/api/apply/1/generate")
        assert resp.status_code == 400
        assert "thin" in resp.text or "raw_text too thin" in resp.text


class TestApplyMark:
    def test_mark_submitted_creates_app(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) "
                "VALUES ('m', ?, 'h1')", ("x" * 250,),
            )
        resp = client.post("/api/apply/1/mark", data={"status": "submitted"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "submitted"
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE job_id = 1"
            ).fetchone()
            event = conn.execute(
                "SELECT kind FROM application_events WHERE application_id = ?",
                (data["app_id"],),
            ).fetchone()
        assert row[0] == "submitted"
        assert event[0] == "submitted"

    def test_mark_offer_writes_app_outcome_signal(self, app_client):
        from offerguide.evolution.signals import fetch_signals

        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) "
                "VALUES ('m', ?, 'h1')", ("x" * 250,),
            )
        client.post("/api/apply/1/mark", data={"status": "offer"})
        signals = fetch_signals(store, skill_name="apply_assistant")
        assert any(s.signal_kind == "app_outcome" and s.signal_value == 1.0 for s in signals)

    def test_mark_rejected_writes_negative_signal(self, app_client):
        from offerguide.evolution.signals import fetch_signals

        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) "
                "VALUES ('m', ?, 'h1')", ("x" * 250,),
            )
        client.post("/api/apply/1/mark", data={"status": "rejected"})
        signals = fetch_signals(store, skill_name="apply_assistant")
        assert any(s.signal_kind == "app_outcome" and s.signal_value == 0.0 for s in signals)

    def test_mark_invalid_status(self, app_client):
        client, _ = app_client
        resp = client.post("/api/apply/1/mark", data={"status": "wat"})
        assert resp.status_code == 400

    def test_mark_idempotent_on_subsequent(self, app_client):
        """Second mark on same job UPDATEs the existing application, not duplicates."""
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) "
                "VALUES ('m', ?, 'h1')", ("x" * 250,),
            )
        r1 = client.post("/api/apply/1/mark", data={"status": "submitted"})
        r2 = client.post("/api/apply/1/mark", data={"status": "hr_viewed"})
        # Same app_id (the second call updates, not creates)
        assert r1.json()["app_id"] == r2.json()["app_id"]
        with store.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM applications WHERE job_id = 1"
            ).fetchone()[0]
        assert n == 1
