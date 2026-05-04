"""W13.6 long-horizon goals + meta-reflection (真 agent 必备)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import goals as _goals
from offerguide.agent import AgentLoop, snapshot_state
from offerguide.config import Settings
from offerguide.llm import LLMResponse, ToolCall
from offerguide.profile import UserProfile
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "goals.db")
    s.init_schema()
    return s


# ═══════════════════════════════════════════════════════════════════
# Schema migration
# ═══════════════════════════════════════════════════════════════════


class TestSchemaMigration:
    def test_user_goals_table_exists(self, store):
        with store.connect() as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(user_goals)").fetchall()}
        for c in ("title", "description", "target_date", "target_metric",
                  "status", "achieved_at", "notes"):
            assert c in cols

    def test_agent_self_observations_table_exists(self, store):
        with store.connect() as conn:
            cols = {row[1] for row in conn.execute(
                "PRAGMA table_info(agent_self_observations)"
            ).fetchall()}
        for c in ("observation", "pattern_kind", "evidence_json",
                  "valid_until", "superseded_by"):
            assert c in cols


# ═══════════════════════════════════════════════════════════════════
# Goals CRUD
# ═══════════════════════════════════════════════════════════════════


class TestGoalsCRUD:
    def test_add_active_goal(self, store):
        g = _goals.add_goal(
            store,
            title="拿到 1 个 AI Agent 暑期实习 offer",
            target_date=date.today() + timedelta(days=60),
            target_metric="1 offer",
            description="字节/腾讯/小红书优先",
        )
        assert g.id > 0
        assert g.status == "active"
        assert g.target_date == date.today() + timedelta(days=60)

    def test_list_active_goals(self, store):
        _goals.add_goal(store, title="g1")
        _goals.add_goal(store, title="g2")
        goals = _goals.list_active_goals(store)
        assert len(goals) == 2

    def test_target_date_string_accepted(self, store):
        g = _goals.add_goal(store, title="x", target_date="2026-07-15")
        assert g.target_date == date(2026, 7, 15)

    def test_update_status_to_achieved(self, store):
        g = _goals.add_goal(store, title="x")
        assert _goals.update_goal_status(store, g.id, status="achieved")
        re_fetched = _goals.get_goal(store, g.id)
        assert re_fetched.status == "achieved"
        assert re_fetched.achieved_at is not None

    def test_achieved_goals_excluded_from_active_list(self, store):
        g1 = _goals.add_goal(store, title="g1")
        _goals.add_goal(store, title="g2")
        _goals.update_goal_status(store, g1.id, status="achieved")
        active = _goals.list_active_goals(store)
        assert len(active) == 1
        assert active[0].title == "g2"


# ═══════════════════════════════════════════════════════════════════
# Progress computation (the funnel)
# ═══════════════════════════════════════════════════════════════════


class TestGoalProgress:
    def test_empty_funnel(self, store):
        g = _goals.add_goal(store, title="x", target_date=date.today() + timedelta(days=30))
        p = _goals.compute_progress(store, g)
        assert p.apps_total == 0
        assert p.offers == 0
        assert p.days_left == 30

    def test_funnel_counts_real_apps(self, store):
        g = _goals.add_goal(store, title="x", target_date=date.today() + timedelta(days=30))
        with store.connect() as conn:
            conn.execute("INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m','x','h1')")
            conn.execute("INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m','y','h2')")
            conn.execute(
                "INSERT INTO applications(job_id, status, applied_at) "
                "VALUES (1, 'applied', julianday('now'))"
            )
            conn.execute(
                "INSERT INTO applications(job_id, status, applied_at) "
                "VALUES (2, 'offer', julianday('now'))"
            )
        p = _goals.compute_progress(store, g)
        assert p.apps_total == 2
        assert p.apps_active == 1  # only 'applied' is non-terminal
        assert p.offers == 1

    def test_silent_apps_categorized(self, store):
        g = _goals.add_goal(store, title="x")
        with store.connect() as conn:
            conn.execute("INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m','x','h1')")
            # 14+ days silent
            conn.execute(
                "INSERT INTO applications(job_id, status, applied_at, last_status_change) "
                "VALUES (1, 'applied', julianday('now') - 20, julianday('now') - 20)"
            )
        p = _goals.compute_progress(store, g)
        assert p.apps_silent_14d == 1
        assert p.apps_silent_7d == 1  # 7+ includes 14+

    def test_render_for_prompt_human_readable(self, store):
        g = _goals.add_goal(
            store, title="拿 1 offer", target_date=date.today() + timedelta(days=60),
            target_metric="1 offer", description="ByteDance preferred",
        )
        p = _goals.compute_progress(store, g)
        out = p.render_for_prompt()
        assert "拿 1 offer" in out
        assert "60" in out  # days left
        assert "1 offer" in out  # metric
        assert "ByteDance" in out  # description excerpt

    def test_overdue_goal_render(self, store):
        g = _goals.add_goal(
            store, title="x", target_date=date.today() - timedelta(days=5),
        )
        p = _goals.compute_progress(store, g)
        out = p.render_for_prompt()
        assert "已过期" in out

    def test_on_track_with_offer(self, store):
        g = _goals.add_goal(store, title="x",
                             target_date=date.today() + timedelta(days=10))
        with store.connect() as conn:
            conn.execute("INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m','x','h1')")
            conn.execute(
                "INSERT INTO applications(job_id, status, applied_at) "
                "VALUES (1, 'offer', julianday('now'))"
            )
        p = _goals.compute_progress(store, g)
        assert p.is_on_track is True

    def test_off_track_too_few_apps(self, store):
        g = _goals.add_goal(store, title="x",
                             target_date=date.today() + timedelta(days=30))
        # 0 apps in 30 days = ratio 0 < 0.3
        p = _goals.compute_progress(store, g)
        assert p.is_on_track is False


# ═══════════════════════════════════════════════════════════════════
# Self-observations
# ═══════════════════════════════════════════════════════════════════


class TestSelfObservations:
    def test_write_then_read(self, store):
        oid = _goals.write_self_observation(
            store, observation="I keep enriching JDs the user dismisses",
            pattern_kind="overreach",
            evidence={"sample_runs": [1, 2, 3]},
        )
        assert oid > 0
        obs = _goals.list_active_self_observations(store)
        assert len(obs) == 1
        assert obs[0].pattern_kind == "overreach"
        assert obs[0].evidence == {"sample_runs": [1, 2, 3]}

    def test_expiry_filter(self, store):
        # Insert one expired + one valid
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO agent_self_observations(observation, pattern_kind, valid_until) "
                "VALUES ('expired', 'tone', julianday('now') - 5)"
            )
            conn.execute(
                "INSERT INTO agent_self_observations(observation, pattern_kind, valid_until) "
                "VALUES ('valid', 'tone', julianday('now') + 5)"
            )
            conn.execute(
                "INSERT INTO agent_self_observations(observation, pattern_kind) "
                "VALUES ('forever', 'success_pattern')"
            )
        obs = _goals.list_active_self_observations(store)
        observations = {o.observation for o in obs}
        assert "valid" in observations
        assert "forever" in observations
        assert "expired" not in observations

    def test_supersede_chain(self, store):
        old = _goals.write_self_observation(
            store, observation="old wrong observation", pattern_kind="tone",
        )
        new = _goals.write_self_observation(
            store, observation="actually it's the opposite", pattern_kind="tone",
        )
        _goals.supersede_observation(store, old_id=old, new_id=new)
        active = _goals.list_active_self_observations(store)
        # old is superseded → not active anymore
        assert all(o.id != old for o in active)
        assert any(o.id == new for o in active)


# ═══════════════════════════════════════════════════════════════════
# Snapshot includes goals + self_obs
# ═══════════════════════════════════════════════════════════════════


class TestSnapshotIncludesNorthStar:
    def test_no_goal_no_section(self, store):
        snap = snapshot_state(store)
        # No North Star section when no active goals
        assert "🎯 North Star" not in snap

    def test_active_goal_appears_in_snapshot(self, store):
        _goals.add_goal(
            store, title="拿 1 个字节实习",
            target_date=date.today() + timedelta(days=30),
        )
        snap = snapshot_state(store)
        assert "🎯 North Star" in snap
        assert "拿 1 个字节实习" in snap

    def test_self_observations_appear_in_snapshot(self, store):
        _goals.write_self_observation(
            store, observation="overreach pattern noticed",
            pattern_kind="overreach",
        )
        snap = snapshot_state(store)
        assert "📓 Agent 自我观察" in snap
        assert "overreach pattern noticed" in snap


# ═══════════════════════════════════════════════════════════════════
# meta_reflect AgentLoop tool
# ═══════════════════════════════════════════════════════════════════


class _StubLLM:
    def chat(self, messages, **kw):
        return LLMResponse(content="{}", model="stub")


class TestMetaReflectTool:
    def test_tool_in_schemas(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills)
        names = {sc["function"]["name"] for sc in loop._tool_schemas}
        assert "meta_reflect" in names

    def test_writes_self_observation(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills)
        loop._current_run_id = 7

        result = loop._execute_tool(ToolCall(
            id="x", name="meta_reflect",
            arguments={
                "observation": "我连续 3 次唤醒都建议 enrich 但用户都没点过批准",
                "pattern_kind": "overreach",
            },
        ))
        assert "OK" in result
        assert "self_observation" in result

        obs = _goals.list_active_self_observations(store)
        assert len(obs) == 1
        assert obs[0].pattern_kind == "overreach"
        assert obs[0].evidence.get("from_agent_run") == 7

    def test_invalid_pattern_kind_rejected(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills)

        result = loop._execute_tool(ToolCall(
            id="x", name="meta_reflect",
            arguments={"observation": "x", "pattern_kind": "bogus_kind"},
        ))
        assert "ERROR" in result

    def test_valid_for_days_supports_expiry(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills)
        loop._current_run_id = 1

        loop._execute_tool(ToolCall(
            id="x", name="meta_reflect",
            arguments={
                "observation": "user is busy this week",
                "pattern_kind": "tone",
                "valid_for_days": 7,
            },
        ))
        obs = _goals.list_active_self_observations(store)
        assert obs[0].valid_until is not None


# ═══════════════════════════════════════════════════════════════════
# /goals UI
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


class TestGoalsUI:
    def test_empty_renders(self, app_client):
        client, _ = app_client
        resp = client.get("/goals")
        assert resp.status_code == 200
        assert "North Star" in resp.text
        assert "还没设任何 goal" in resp.text

    def test_add_goal_endpoint(self, app_client):
        client, _ = app_client
        resp = client.post("/api/goals/add", data={
            "title": "拿字节 offer",
            "target_date": "2026-07-15",
            "target_metric": "1 offer",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "拿字节 offer"
        assert data["status"] == "active"

    def test_add_goal_invalid_date(self, app_client):
        client, _ = app_client
        resp = client.post("/api/goals/add", data={
            "title": "x",
            "target_date": "not-a-date",
        })
        assert resp.status_code == 400

    def test_add_goal_no_title(self, app_client):
        client, _ = app_client
        resp = client.post("/api/goals/add", data={"title": "  "})
        assert resp.status_code == 400

    def test_set_status_endpoint(self, app_client):
        client, store = app_client
        g = _goals.add_goal(store, title="x")
        resp = client.post(f"/api/goals/{g.id}/status", data={"status": "achieved"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "achieved"

    def test_set_status_invalid_value(self, app_client):
        client, store = app_client
        g = _goals.add_goal(store, title="x")
        resp = client.post(f"/api/goals/{g.id}/status", data={"status": "bogus"})
        assert resp.status_code == 400

    def test_set_status_not_found(self, app_client):
        client, _ = app_client
        resp = client.post("/api/goals/9999/status", data={"status": "paused"})
        assert resp.status_code == 404

    def test_view_renders_active_goal_with_funnel(self, app_client):
        client, store = app_client
        _goals.add_goal(
            store, title="拿字节 offer",
            target_date=date.today() + timedelta(days=45),
            target_metric="1 offer",
            description="目标公司: 字节",
        )
        resp = client.get("/goals")
        assert resp.status_code == 200
        assert "拿字节 offer" in resp.text
        assert "投递" in resp.text or "funnel" in resp.text.lower()
        assert "目标公司: 字节" in resp.text
