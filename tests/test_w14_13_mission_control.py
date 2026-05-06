"""W14.13 — Mission Control: home shows what agent CAN do + IS DOING.

User feedback: "你得让人清楚, 你能干什么, 在干什么啊" — the W14.12
home (4 numeric cards) failed at this because all numbers were 0 on a
fresh setup, and it didn't surface daemon status, schedule, or a way to
trigger anything manually.

W14.13 adds:
1. Mission Control: per-daemon card with icon / what / last_run / next
   schedule / 24h count / status pulse / ▶ trigger button
2. Activity timeline (collapsed by default): last 8 daemon_runs
3. POST /api/scheduler/trigger/{name}: manual trigger endpoint with
   daemon_runs row recording
"""

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
    store = offerguide.Store(tmp_path / "mc.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)
    s = Settings(deepseek_api_key="x", default_model="stub")

    class _StubLLM:
        def chat(self, messages, **kw):
            return LLMResponse(content="{}", model="stub")

    runtime = SkillRuntime(llm=_StubLLM(), store=store)
    profile = UserProfile(raw_resume_text="x", source_pdf="/tmp/x.pdf")
    app = create_app(
        settings=s, store=store, profile=profile,
        skills=skills, runtime=runtime, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


# ═══════════════════════════════════════════════════════════════════
# Home shows Mission Control with all 3 daemons
# ═══════════════════════════════════════════════════════════════════


class TestMissionControlOnHome:
    def test_mission_control_renders_3_cards(self, app_client):
        """W14.18: cards now reflect new architecture — wake_agent is the
        only cron, discover/score are agent-tools also exposed for manual
        trigger. The 3 cards stay (wake_agent + 2 manual triggers) but
        labels changed."""
        client, _ = app_client
        resp = client.get("/")
        # Header
        assert "Mission Control" in resp.text
        # New canonical names per W14.18 (discover_new_jobs/score_unscored_jobs)
        assert "discover_new_jobs" in resp.text
        assert "score_unscored_jobs" in resp.text
        assert "wake_agent" in resp.text
        # wake_agent is the only one with a real cron schedule now
        assert "每小时" in resp.text or "心跳" in resp.text
        # The other 2 are explicitly described as agent-driven, not crony
        assert "中央 agent 自主决定" in resp.text or "agent 自己调" in resp.text

    def test_each_daemon_has_trigger_button(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        # The triggerDaemon JS function and the "▶ 立刻跑一次" button text
        assert "triggerDaemon" in resp.text
        assert "立刻跑一次" in resp.text
        # W14.18: card names migrated to new canonical names
        for name in ["discover_new_jobs", "score_unscored_jobs", "wake_agent"]:
            assert f"triggerDaemon('{name}'" in resp.text

    def test_no_daemon_runs_shows_never_run(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        assert "从未跑过" in resp.text

    def test_with_daemon_runs_shows_last_status(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO daemon_runs(job_name, status, summary_json, "
                "  started_at, ended_at) "
                "VALUES ('discover_jobs_via_search', 'ok', "
                "        '{\"inserted\":3,\"hits_evaluated\":7}', "
                "        julianday('now') - 0.001, julianday('now'))"
            )
        resp = client.get("/")
        # Status pill + summary key=value
        assert "inserted=3" in resp.text or "inserted" in resp.text
        # The "ok" status pill should appear in some form
        assert ">ok<" in resp.text or "status-ok" in resp.text or "ok</span>" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Activity timeline
# ═══════════════════════════════════════════════════════════════════


class TestActivityTimeline:
    def test_recent_daemon_runs_appear_in_timeline(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            for _i in range(3):
                conn.execute(
                    "INSERT INTO daemon_runs(job_name, status, summary_json) "
                    "VALUES (?, 'ok', '{}')",
                    ("discover_jobs_via_search",),
                )
            conn.execute(
                "INSERT INTO daemon_runs(job_name, status, error_text, summary_json) "
                "VALUES ('auto_score_new_jobs', 'error', "
                "        'LLM rate limited', '{}')"
            )
        resp = client.get("/")
        # Timeline section appears
        assert "活动时间线" in resp.text or "recent activity" in resp.text.lower()
        # Error event surfaces with the error text
        assert "LLM rate limited" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Manual trigger API: /api/scheduler/trigger/{name}
# ═══════════════════════════════════════════════════════════════════


class TestManualTriggerAPI:
    def test_unknown_daemon_returns_404(self, app_client):
        client, _ = app_client
        resp = client.post("/api/scheduler/trigger/no_such_thing")
        assert resp.status_code == 404

    def test_no_llm_returns_400(self, tmp_path):
        from offerguide.ui.web import create_app

        store = offerguide.Store(tmp_path / "mc2.db")
        store.init_schema()
        s = Settings(deepseek_api_key="", default_model="stub")
        app = create_app(
            settings=s, store=store, profile=None, skills=[],
            runtime=None, notifier=ConsoleNotifier(),
        )
        client = TestClient(app)
        resp = client.post("/api/scheduler/trigger/discover_jobs_via_search")
        assert resp.status_code == 400

    def test_trigger_records_daemon_run(self, app_client, monkeypatch):
        """Manual trigger should write a daemon_runs row (status=ok or
        error) so the timeline reflects the manual run alongside cron runs.

        W15.7: endpoint now routes through harness.run_one — patch that
        instead of the deleted W14 daemon helpers."""
        client, store = app_client
        from offerguide.harness import RunResult
        from offerguide.ui import web as web_mod

        def _fake_harness_run(*, trigger, deps, max_iterations=20):
            return RunResult(
                run_id=999, iterations=2,
                final_text="(stub) discovered 2 new jobs",
                tool_call_log=["iter1.discover_jobs(criteria='...')"],
                cost_usd=0.0, latency_ms=10,
                finish_reason="end_turn",
            )

        # Patch the module-level alias the route imports lazily inside
        # the handler. Easiest: patch on the harness module itself.
        from offerguide import harness as harness_mod
        monkeypatch.setattr(harness_mod, "run", _fake_harness_run)
        # Some lazy imports go through harness.loop.run too — patch both
        from offerguide.harness import loop as harness_loop_mod
        monkeypatch.setattr(harness_loop_mod, "run", _fake_harness_run)
        # And the bound name in ui.web (was imported as `harness_run`)
        if hasattr(web_mod, "harness_run"):
            monkeypatch.setattr(web_mod, "harness_run", _fake_harness_run)

        resp = client.post("/api/scheduler/trigger/discover_jobs_via_search")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job"] == "discover_jobs_via_search"
        assert "run_id" in data
        # New result schema has harness_run_id + iterations + finish + cost_usd
        assert "harness_run_id" in data["result"]
        assert data["result"]["iterations"] == 2

        # daemon_runs row should be present with status=ok
        with store.connect() as conn:
            row = conn.execute(
                "SELECT job_name, status, summary_json FROM daemon_runs "
                "WHERE id = ?", (data["run_id"],),
            ).fetchone()
        assert row[0] == "discover_jobs_via_search"
        assert row[1] == "ok"
        assert "harness_run_id" in row[2]

    def test_trigger_failure_records_error(self, app_client, monkeypatch):
        """W15.7: route through harness; assert error path records to daemon_runs."""
        client, store = app_client

        def _broken(*, trigger, deps, max_iterations=20):
            raise RuntimeError("simulated daemon crash")

        from offerguide import harness as harness_mod
        from offerguide.harness import loop as harness_loop_mod
        from offerguide.ui import web as web_mod
        monkeypatch.setattr(harness_mod, "run", _broken)
        monkeypatch.setattr(harness_loop_mod, "run", _broken)
        if hasattr(web_mod, "harness_run"):
            monkeypatch.setattr(web_mod, "harness_run", _broken)

        resp = client.post("/api/scheduler/trigger/discover_jobs_via_search")
        assert resp.status_code == 500
        # daemon_runs row should be 'error' with the message captured
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status, error_text FROM daemon_runs "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "error"
        assert "simulated daemon crash" in row[1]


# ═══════════════════════════════════════════════════════════════════
# Each daemon has user-facing description ("能干什么")
# ═══════════════════════════════════════════════════════════════════


class TestDaemonCapabilityDescriptions:
    def test_what_each_daemon_does_is_visible(self, app_client):
        """Per the user feedback "让人清楚你能干什么", each daemon card
        must spell out what it does in human-readable Chinese, not just
        cron schedules."""
        client, _ = app_client
        resp = client.get("/")
        # discover_jobs_via_search description
        assert "Tavily" in resp.text
        # auto_score_new_jobs description
        assert "score_match" in resp.text
        # wake_agent description
        assert "中央 agent" in resp.text or "off-track" in resp.text
