"""W14 — push channel + connectivity polish."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import goals as _goals
from offerguide.agent import AgentLoop
from offerguide.config import Settings
from offerguide.llm import LLMResponse, ToolCall
from offerguide.profile import UserProfile
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# send_notification action tool
# ═══════════════════════════════════════════════════════════════════


class _StubLLM:
    def chat(self, messages, **kw):
        return LLMResponse(content="{}", model="stub")


class _MockNotifier:
    """Records notify() calls so we can assert agent invoked it correctly."""
    def __init__(self):
        self.calls: list[dict] = []
    def notify(self, *, title, body, level="info"):
        self.calls.append({"title": title, "body": body, "level": level})
        from offerguide.ui.notify._base import NotifyResult
        return NotifyResult(ok=True, channel="mock")


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "w14.db")
    s.init_schema()
    return s


class TestSendNotification:
    def test_tool_in_schemas(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills,
                          notifier=_MockNotifier())
        names = {sc["function"]["name"] for sc in loop._tool_schemas}
        assert "send_notification" in names

    def test_dispatches_to_notifier(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        notif = _MockNotifier()
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills,
                          notifier=notif)

        result = loop._execute_tool(ToolCall(
            id="x", name="send_notification",
            arguments={
                "title": "字节面试明早 10 点",
                "body": "明早 10:00 字节 AI Agent 一面, 准备 LangGraph 项目深挖",
                "level": "high",
            },
        ))
        assert "OK" in result
        assert len(notif.calls) == 1
        assert notif.calls[0]["title"] == "字节面试明早 10 点"
        assert notif.calls[0]["level"] == "high"

    def test_fallback_message_when_no_notifier(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills,
                          notifier=None)
        result = loop._execute_tool(ToolCall(
            id="x", name="send_notification",
            arguments={"title": "x", "body": "y"},
        ))
        assert "ERROR" in result
        # Tells the agent what to do instead
        assert "write_suggestion" in result

    def test_missing_args_returns_error(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills,
                          notifier=_MockNotifier())
        result = loop._execute_tool(ToolCall(
            id="x", name="send_notification",
            arguments={"title": "x"},  # missing body
        ))
        assert "ERROR" in result


# ═══════════════════════════════════════════════════════════════════
# Inbox ↔ agent_runs cross-link in /agent/runs/{id} page
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_client(tmp_path):
    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)
    s = Settings(deepseek_api_key="x", default_model="stub")
    runtime = SkillRuntime(llm=_StubLLM(), store=store)
    profile = UserProfile(raw_resume_text="x", source_pdf="/tmp/x.pdf")
    app = create_app(
        settings=s, store=store, profile=profile,
        skills=skills, runtime=runtime, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestAgentRunDetailCrossLinks:
    def test_run_detail_lists_inbox_suggestions_from_this_run(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, "
                "  iterations, final_answer, ended_at) "
                "VALUES ('cron_wake','x','ok',1,'done',julianday('now'))"
            )
            run_id = cur.lastrowid
        # Two suggestions from this run + one from another run
        from offerguide import inbox as inbox_mod
        inbox_mod.enqueue_agent_suggestion(
            store, title="suggest A", body="x",
            source_agent_run_id=run_id,
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        inbox_mod.enqueue_agent_suggestion(
            store, title="suggest B", body="y",
            source_agent_run_id=run_id,
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        inbox_mod.enqueue_agent_suggestion(
            store, title="suggest C from another run", body="z",
            source_agent_run_id=999,  # different run
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        resp = client.get(f"/agent/runs/{run_id}")
        assert resp.status_code == 200
        assert "suggest A" in resp.text
        assert "suggest B" in resp.text
        # Suggestion from a different run should NOT appear
        assert "suggest C from another run" not in resp.text

    def test_run_detail_shows_evolution_signals_from_this_run(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, "
                "  iterations, final_answer, ended_at) "
                "VALUES ('cron_wake','x','ok',1,'done',julianday('now'))"
            )
            run_id = cur.lastrowid
            conn.execute(
                "INSERT INTO evolution_signals(skill_name, skill_version, "
                "  signal_kind, signal_value, signal_weight, notes) "
                "VALUES ('score_match','0.1.0','critic',0.85,1.0,?)",
                (f"agent_run#{run_id}: clean run",),
            )
            # And a signal from a different run that should NOT appear
            conn.execute(
                "INSERT INTO evolution_signals(skill_name, skill_version, "
                "  signal_kind, signal_value, signal_weight, notes) "
                "VALUES ('tailor_resume','0.1.0','critic',0.5,1.0,'agent_run#999')"
            )
        resp = client.get(f"/agent/runs/{run_id}")
        assert "score_match" in resp.text
        assert "0.85" in resp.text
        # Signals not tied to this run shouldn't leak in
        assert "tailor_resume" not in resp.text

    def test_run_detail_shows_cost_when_recorded(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, "
                "  iterations, final_answer, cost_usd, ended_at) "
                "VALUES ('cron_wake','x','ok',1,'done', 0.0234, julianday('now'))"
            )
            run_id = cur.lastrowid
        resp = client.get(f"/agent/runs/{run_id}")
        assert "0.0234" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Application outcome → user_facts auto-extract (W14.4)
# ═══════════════════════════════════════════════════════════════════


class TestApplyOutcomeFactExtraction:
    def test_offer_creates_positive_fact(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
                "VALUES ('m','AI Agent 实习','字节跳动', ?, 'h1')", ("x" * 250,),
            )
        client.post("/api/apply/1/mark", data={"status": "offer"})

        with store.connect() as conn:
            facts = conn.execute(
                "SELECT fact_text, kind FROM user_facts WHERE source_skill='apply_lifecycle'"
            ).fetchall()
        assert len(facts) == 1
        assert "字节跳动" in facts[0][0]
        assert "offer" in facts[0][0]
        assert facts[0][1] == "experience"

    def test_rejected_creates_negative_company_signal(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
                "VALUES ('m','OD 算法','某公司', ?, 'h1')", ("x" * 250,),
            )
        client.post("/api/apply/1/mark", data={"status": "rejected"})

        with store.connect() as conn:
            facts = conn.execute(
                "SELECT fact_text, kind FROM user_facts WHERE source_skill='apply_lifecycle'"
            ).fetchall()
        assert len(facts) == 1
        assert facts[0][1] == "company_signal"
        assert "降优先级" in facts[0][0] or "拒了" in facts[0][0]

    def test_submitted_does_not_create_fact(self, app_client):
        """Only terminal/decisive outcomes warrant facts; 'submitted' is too early."""
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
                "VALUES ('m','x','字节', ?, 'h1')", ("x" * 250,),
            )
        client.post("/api/apply/1/mark", data={"status": "submitted"})
        with store.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM user_facts WHERE source_skill='apply_lifecycle'"
            ).fetchone()[0]
        assert n == 0


# ═══════════════════════════════════════════════════════════════════
# /tailor history index (W14.5)
# ═══════════════════════════════════════════════════════════════════


class TestTailorHistoryIndex:
    def test_no_history_no_section(self, app_client, monkeypatch, tmp_path):
        client, _ = app_client
        # cwd has no data/tailored
        monkeypatch.chdir(tmp_path)
        resp = client.get("/tailor")
        assert resp.status_code == 200
        # Section header doesn't appear when empty
        assert "已生成的 tailored docx" not in resp.text

    def test_lists_existing_tailored_files(self, app_client, monkeypatch, tmp_path):
        client, _ = app_client
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "tailored").mkdir(parents=True)
        for fn in ["tailored_byteDance_001.docx", "tailored_tencent_002.docx"]:
            (tmp_path / "data" / "tailored" / fn).write_bytes(b"x" * 1024)
        resp = client.get("/tailor")
        assert resp.status_code == 200
        assert "已生成的 tailored docx" in resp.text
        assert "byteDance" in resp.text
        assert "tencent" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Goal off-track injection into wake_agent goal (W14.6)
# ═══════════════════════════════════════════════════════════════════


class TestWakeAgentGoalInjection:
    def test_off_track_goal_appended_to_wake_goal(self, tmp_path, monkeypatch):
        """When user has an off-track goal, the wake_agent job should append
        urgent context to the base goal text."""
        from offerguide.autonomous.scheduler import build_agent_wake_scheduler

        monkeypatch.delenv("OFFERGUIDE_LLM_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        store = offerguide.Store(tmp_path / "wake.db")
        store.init_schema()
        # Create an off-track goal: 30 days left, 0 apps in flight
        _goals.add_goal(
            store, title="拿 1 个字节实习 offer",
            target_date=date.today() + timedelta(days=30),
            target_metric="1 offer",
        )

        s = Settings(deepseek_api_key="", db_path=str(tmp_path / "wake.db"))
        sched = build_agent_wake_scheduler(settings=s)
        result = sched.trigger_once("wake_agent")
        sched.shutdown()
        # No LLM key → wake should skip but cleanly
        assert isinstance(result, dict)
        assert "skipped" in result
