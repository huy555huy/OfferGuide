"""W13.5 home page rewrite — agent-driven (not stat-dashboard-driven)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.llm import LLMResponse
from offerguide.profile import UserProfile
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def app_client(tmp_path):
    store = offerguide.Store(tmp_path / "home.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)
    s = Settings(deepseek_api_key="dummy", deepseek_base_url="x", default_model="m")

    class _SkillStubLLM:
        def chat(self, messages, **kw):
            return LLMResponse(content="{}", model="stub")
    runtime = SkillRuntime(llm=_SkillStubLLM(), store=store)
    profile = UserProfile(raw_resume_text="x", source_pdf="/tmp/y.pdf")

    app = create_app(
        settings=s, store=store, profile=profile,
        skills=skills, runtime=runtime, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestHomeRendering:
    def test_empty_db_renders_no_run_state(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Agent Chat" in resp.text
        assert "主工作台" in resp.text

    def test_home_shows_latest_agent_run(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, iterations, "
                "  final_answer, critic_score, latency_ms, ended_at) "
                "VALUES ('cron_wake', 'check', 'ok', 2, "
                "        'agent says all clear, lay low', 0.9, 8500, julianday('now'))"
            )
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Agent Chat" in resp.text
        assert "最近 Agent 运行" in resp.text

    def test_home_does_not_show_running_or_failed_runs_in_hero(self, app_client):
        """Only ok runs land in the hero (failed/running runs would be confusing)."""
        client, store = app_client
        with store.connect() as conn:
            # Failed run, more recent
            conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, iterations, "
                "  final_answer, started_at) "
                "VALUES ('cron_wake', 'x', 'error', 1, NULL, julianday('now'))"
            )
            # Successful run, older
            conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, iterations, "
                "  final_answer, started_at) "
                "VALUES ('cron_wake', 'older check', 'ok', 1, "
                "        'older agent answer', julianday('now') - 0.1)"
            )
        resp = client.get("/today")
        assert resp.status_code == 200
        assert "Mission Control" in resp.text
        assert "older agent answer" not in resp.text

    def test_home_lists_pending_agent_suggestions(self, app_client):
        from offerguide import inbox as inbox_mod
        client, store = app_client
        # Add 2 agent_suggestion items + 1 legacy consider_jd (should not appear in suggestions)
        inbox_mod.enqueue_agent_suggestion(
            store, title="建议 tailor 字节简历", body="x",
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        inbox_mod.enqueue_agent_suggestion(
            store, title="silent 14 天的字节申请要不要 follow up",
            body="y", source_skill_name="score_match",
            source_skill_version="0.1.0",
        )
        inbox_mod.enqueue(
            store, kind="consider_jd", title="legacy item", body="z",
        )
        resp = client.get("/today")
        assert resp.status_code == 200
        assert "建议 tailor 字节简历" in resp.text
        assert "silent 14 天" in resp.text
        # legacy item not shown (it's not agent_suggestion kind)
        assert "legacy item" not in resp.text

    def test_home_quick_links_present(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        for link in ["/", "/today", "/applications", "/tailor", "/mock", "/evolution", "/inbox"]:
            assert f'href="{link}"' in resp.text


class TestWakeAgentEndpoint:
    def test_wake_agent_requires_runtime(self, tmp_path):
        from fastapi.testclient import TestClient
        store = offerguide.Store(tmp_path / "x.db")
        store.init_schema()
        s = Settings(deepseek_api_key="", db_path=str(tmp_path / "x.db"))
        app = create_app(
            settings=s, store=store, profile=None, skills=[],
            runtime=None, notifier=ConsoleNotifier(),
        )
        client = TestClient(app)
        resp = client.post("/api/home/wake-agent")
        assert resp.status_code == 400
        assert "agent 不可用" in resp.text or "OFFERGUIDE_LLM_API_KEY" in resp.text


class TestNoOldDashboardLanguage:
    """W13.5 regression: the home page text should NOT use cron-style 'recommend X tasks' language."""

    def test_home_no_4_stat_card_language(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        # Old hero text was "今日驾驶舱" and "件事建议你今天处理"
        assert "今日驾驶舱" not in resp.text
        assert "件事建议你今天处理" not in resp.text

    def test_home_centerpiece_is_agent_assessment(self, app_client):
        """Even on empty DB, the hero is about agent (not about stats)."""
        client, _ = app_client
        resp = client.get("/")
        assert "Agent Chat" in resp.text
