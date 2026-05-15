"""W14.5 — UI polish: nav badges, status colors, empty states, collapsibles."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import goals as _goals
from offerguide.config import Settings
from offerguide.llm import LLMResponse
from offerguide.profile import UserProfile
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def app_client(tmp_path):
    store = offerguide.Store(tmp_path / "polish.db")
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
# Nav badges (P1)
# ═══════════════════════════════════════════════════════════════════


class TestNavBadges:
    def test_inbox_badge_shows_pending_count(self, app_client):
        client, store = app_client
        from offerguide import inbox as inbox_mod
        # 3 pending items
        for i in range(3):
            inbox_mod.enqueue_agent_suggestion(
                store, title=f"sug {i}", body="x",
                source_skill_name="score_match", source_skill_version="0.1.0",
            )
        # 1 already approved (should not count)
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="approved", body="x",
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        inbox_mod.decide(store, item.id, decision="approved")

        resp = client.get("/")
        assert resp.status_code == 200
        # Badge "3" should appear next to inbox link
        assert 'class="nav-badge medium">3</span>' in resp.text

    def test_no_inbox_badge_when_empty(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        # No "nav-badge medium" should appear (no pending)
        assert 'nav-badge medium' not in resp.text

    def test_goals_warning_badge_when_off_track(self, app_client):
        client, store = app_client
        # Off-track goal: deadline soon, no apps
        _goals.add_goal(
            store, title="off-track goal",
            target_date=date.today() + timedelta(days=10),
        )
        resp = client.get("/")
        assert 'nav-badge high' in resp.text  # warning class
        # W15.16 — tooltip changed from "off-track" English → "偏离轨道" 中文
        assert ('off-track' in resp.text or '偏离轨道' in resp.text)

    # Removed: agent_critic_badge — W13 LLM-self-critique was retired in
    # the W21 refactor (CLAUDE.md rule D: LLM can't self-eval). nav badge
    # for last_critic is now always None; real signal lives in
    # evolution_signals (user_thumbs / app_outcome / follow_through),
    # surfaced on the /evolution page.


# ═══════════════════════════════════════════════════════════════════
# Empty states (P3)
# ═══════════════════════════════════════════════════════════════════


class TestEmptyStates:
    def test_inbox_empty_has_actionable_hint(self, app_client):
        client, _ = app_client
        resp = client.get("/inbox")
        assert resp.status_code == 200
        # New empty state shows example suggestions and a CTA button
        assert "暂无 inbox 项" in resp.text
        # Button to /agent
        assert "▶ 跑一次 agent 让它看现状" in resp.text or 'href="/agent"' in resp.text

    def test_funnel_empty_has_explanation(self, app_client):
        client, _ = app_client
        resp = client.get("/funnel")
        assert resp.status_code == 200
        # New empty state explains the funnel + flow
        assert "还没投递记录" in resp.text or "投递" in resp.text

    def test_goals_empty_has_example_text(self, app_client):
        client, _ = app_client
        resp = client.get("/goals")
        assert resp.status_code == 200
        assert "还没设任何 goal" in resp.text
        # Example should be in the empty hint
        assert "AI Agent" in resp.text  # part of example or form placeholder

    def test_portfolio_zero_data_shows_seeding_hint(self, app_client):
        client, _ = app_client
        resp = client.get("/portfolio")
        assert resp.status_code == 200
        # New seeding hint when 0 runs
        assert "数据还很少" in resp.text or "agent run 一次" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Status colors (P2)
# ═══════════════════════════════════════════════════════════════════


class TestStatusColors:
    def test_inbox_approved_row_has_ok_class(self, app_client):
        client, store = app_client
        from offerguide import inbox as inbox_mod
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="approve me", body="x",
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        inbox_mod.decide(store, item.id, decision="approved")
        resp = client.get("/inbox")
        # Approved row gets row-status-ok class
        assert "row-status-ok" in resp.text

    def test_inbox_rejected_row_has_bad_class(self, app_client):
        client, store = app_client
        from offerguide import inbox as inbox_mod
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="reject me", body="x",
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        inbox_mod.decide(store, item.id, decision="rejected")
        resp = client.get("/inbox")
        assert "row-status-bad" in resp.text

    def test_agent_critic_in_recent_runs_color_coded(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            # High critic
            conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, "
                "  iterations, critic_score, ended_at) "
                "VALUES ('t','high','ok',1,0.9, julianday('now'))"
            )
            # Low critic
            conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, "
                "  iterations, critic_score, ended_at) "
                "VALUES ('t','low','ok',1,0.2, julianday('now') - 0.001)"
            )
        resp = client.get("/agent")
        # The recent-runs table should color-code critics
        assert "var(--success)" in resp.text  # high score in green
        assert "var(--danger)" in resp.text   # low score in red


# ═══════════════════════════════════════════════════════════════════
# Trajectory collapsible (P4)
# ═══════════════════════════════════════════════════════════════════


class TestTrajectoryCollapse:
    def test_harness_run_detail_renders_tool_calls(self, app_client):
        """harness_runs.tool_calls_json renders as a trajectory list on
        /agent/runs/{id}. (Pre-W21 this tested 'state_snapshot' inside a
        collapsible <details>; AgentLoop wrote richly typed trajectory_json
        events. The harness records a flat tool_call_log instead — simpler
        + cheaper, and matches Claude Code's design.)
        """
        client, store = app_client
        import json
        from offerguide.agent_runtime import _schema as _hs
        _hs.init_agent_runtime_schema(store)
        payload = {
            "calls": [
                "iter1.memory(command='view') → OK MEMORY.md (16 lines)",
                "iter2.score_match(job_id=1) → probability=0.65",
            ],
            "sub_agent_cost_usd": 0.0,
        }
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO harness_runs(trigger_kind, trigger_detail, status, "
                "  iterations, final_text, tool_calls_json, ended_at) "
                "VALUES ('user_input','{\"message\":\"x\"}','ok',2,'done',?, julianday('now'))",
                (json.dumps(payload, ensure_ascii=False),),
            )
            run_id = cur.lastrowid
        resp = client.get(f"/agent/runs/{run_id}")
        assert resp.status_code == 200
        # tool call summaries surfaced in the detail page
        assert "memory" in resp.text or "score_match" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Loading affordances (P7)
# ═══════════════════════════════════════════════════════════════════


class TestLoadingStates:
    def test_home_wake_agent_button_has_loading_text(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        # W14.13: home now uses Mission Control "▶ 立刻跑一次 (不等 cron)"
        # daemon trigger buttons (one of which is wake_agent) instead of a
        # single hero "唤醒 agent" button. Either loading-text pattern is fine.
        assert (
            'data-loading-text="agent 思考中..."' in resp.text
            or 'data-loading-text="跑中… (10-60s)"' in resp.text
        )

    def test_apply_mark_buttons_have_loading_text(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, 'h1')",
                ("x" * 250,),
            )
        resp = client.get("/apply/1")
        assert 'data-loading-text="标记中..."' in resp.text

    def test_global_toast_helper_present(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        # The global toast/btnLoading/btnReset helpers should be in base.html
        assert "window.toast" in resp.text
        assert "window.btnLoading" in resp.text
        assert "og-toast-container" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Lifecycle button grouping (P5)
# ═══════════════════════════════════════════════════════════════════


class TestLifecycleGrouping:
    def test_apply_buttons_split_into_three_groups(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, 'h1')",
                ("x" * 250,),
            )
        resp = client.get("/apply/1")
        assert "第一阶 — 投递动作" in resp.text
        assert "第二阶 — 推进信号" in resp.text
        assert "终结 — 已定结果" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Goals form polish (P6)
# ═══════════════════════════════════════════════════════════════════


class TestGoalsFormPolish:
    def test_form_has_inline_error_target(self, app_client):
        client, _ = app_client
        resp = client.get("/goals")
        # inline error span for title
        assert 'id="title-err"' in resp.text
        # submit button has loading text
        assert 'data-loading-text="保存中..."' in resp.text

    def test_target_metric_placeholder_helpful(self, app_client):
        client, _ = app_client
        resp = client.get("/goals")
        # Friendlier placeholder mentions multiple metric examples
        assert "1 offer" in resp.text and "interview" in resp.text
