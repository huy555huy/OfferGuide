"""W14.20 — ambient agent redesign: working memory + ask_user_question +
home as agent-driven view (not user-managed).

Pinning the 4 architectural shifts the user demanded:
  1. agent_self_notes table for cross-wake working memory
  2. inbox kind='question' so agent can ask user (not just push suggestions)
  3. Snapshot surfaces self_notes + pending questions at top
  4. Home leads with "今日重点" / agent 内心独白 / pending questions —
     Mission Control demoted to bottom collapsible debug
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import inbox as inbox_mod
from offerguide.agent.loop import ToolCall
from offerguide.config import Settings
from offerguide.llm import LLMResponse
from offerguide.profile import UserProfile
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def app_client(tmp_path):
    store = offerguide.Store(tmp_path / "w1420.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)
    s = Settings(deepseek_api_key="x", default_model="stub")

    class _StubLLM:
        def chat(self, messages, **kw):
            return LLMResponse(content="{}", model="stub")
        def chat_with_tools(self, messages, tools, **kw):
            return LLMResponse(content="", model="stub")
        def close(self): pass

    runtime = SkillRuntime(llm=_StubLLM(), store=store)
    profile = UserProfile(raw_resume_text="x", source_pdf="/tmp/x.pdf")
    app = create_app(
        settings=s, store=store, profile=profile,
        skills=skills, runtime=runtime, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


# ═══════════════════════════════════════════════════════════════════
# agent_self_notes — working memory across wakes
# ═══════════════════════════════════════════════════════════════════


class TestSelfNotes:
    def test_table_exists(self, tmp_path):
        store = offerguide.Store(tmp_path / "n.db")
        store.init_schema()
        with store.connect() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(agent_self_notes)")}
        # Must have the working-memory shape
        assert {"id", "body", "note_kind", "valid_until", "cleared_at",
                "cleared_reason", "related_run_id", "created_at"} <= cols

    def test_write_note_via_agent_tool(self, app_client):
        from offerguide.agent.loop import AgentLoop
        client, store = app_client
        skills = discover_skills(SKILLS_ROOT)

        class _StubLLM:
            def chat(self, *a, **kw): return LLMResponse(content="{}", model="stub")
            def chat_with_tools(self, *a, **kw): return LLMResponse(content="", model="stub")
            def close(self): pass

        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        loop = AgentLoop(
            llm=_StubLLM(), runtime=runtime, store=store, skills=skills,
            master_resume_text="x",
        )
        loop._current_run_id = 1

        tc = ToolCall(
            id="t1", name="write_note_to_self",
            arguments={
                "body": "下次 wake 看 4 个 JD score 出来了没",
                "kind": "todo",
                "valid_for_hours": 24,
            },
        )
        result = loop._execute_action_tool(tc)
        assert "OK:" in result and "self_note" in result
        # Verify row landed
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT body, note_kind, related_run_id "
                "FROM agent_self_notes WHERE cleared_at IS NULL"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "下次 wake 看 4 个 JD score 出来了没"
        assert rows[0][1] == "todo"
        assert rows[0][2] == 1  # related_run_id

    def test_clear_note_via_agent_tool(self, app_client):
        from offerguide.agent.loop import AgentLoop
        client, store = app_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO agent_self_notes(body, note_kind) "
                "VALUES ('todo: do X', 'todo')"
            )
            note_id = cur.lastrowid

        skills = discover_skills(SKILLS_ROOT)

        class _StubLLM:
            def chat(self, *a, **kw): return LLMResponse(content="{}", model="stub")
            def chat_with_tools(self, *a, **kw): return LLMResponse(content="", model="stub")
            def close(self): pass

        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        loop = AgentLoop(
            llm=_StubLLM(), runtime=runtime, store=store, skills=skills,
            master_resume_text="x",
        )
        loop._current_run_id = 99

        tc = ToolCall(
            id="t2", name="clear_self_note",
            arguments={"note_id": note_id, "reason": "did it"},
        )
        result = loop._execute_action_tool(tc)
        assert "OK:" in result
        with store.connect() as conn:
            row = conn.execute(
                "SELECT cleared_at, cleared_reason FROM agent_self_notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        assert row[0] is not None
        assert row[1] == "did it"

    def test_clear_already_cleared_warns(self, app_client):
        from offerguide.agent.loop import AgentLoop
        client, store = app_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO agent_self_notes(body, note_kind, cleared_at, cleared_reason) "
                "VALUES ('done', 'todo', julianday('now'), 'old')"
            )
            note_id = cur.lastrowid

        skills = discover_skills(SKILLS_ROOT)

        class _StubLLM:
            def chat(self, *a, **kw): return LLMResponse(content="{}", model="stub")
            def chat_with_tools(self, *a, **kw): return LLMResponse(content="", model="stub")
            def close(self): pass

        loop = AgentLoop(
            llm=_StubLLM(), runtime=SkillRuntime(llm=_StubLLM(), store=store),
            store=store, skills=skills, master_resume_text="x",
        )
        loop._current_run_id = 1
        result = loop._execute_action_tool(
            ToolCall(id="t", name="clear_self_note",
                     arguments={"note_id": note_id, "reason": "x"}),
        )
        assert "WARN:" in result or "已经 cleared" in result


# ═══════════════════════════════════════════════════════════════════
# Snapshot surfaces self_notes + pending questions at top
# ═══════════════════════════════════════════════════════════════════


class TestSnapshotSurfaces:
    def test_snapshot_shows_active_self_notes(self, app_client):
        from offerguide.agent.loop import snapshot_state
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO agent_self_notes(body, note_kind) "
                "VALUES ('look at 字节 application tomorrow', 'todo')"
            )
            conn.execute(
                "INSERT INTO agent_self_notes(body, note_kind, cleared_at, cleared_reason) "
                "VALUES ('did this', 'todo', julianday('now'), 'done')"
            )
        snap = snapshot_state(store)
        assert "📝" in snap
        assert "look at 字节 application tomorrow" in snap
        # Cleared notes should NOT appear
        assert "did this" not in snap

    def test_snapshot_shows_pending_questions(self, app_client):
        from offerguide.agent.loop import snapshot_state
        client, store = app_client
        inbox_mod.enqueue_question(
            store,
            question="你 north star AI Agent 但简历偏数据分析, 改哪个?",
            context="...",
            options=[{"id": "a", "label": "改 north star"}, {"id": "b", "label": "改简历"}],
        )
        snap = snapshot_state(store)
        assert "❓" in snap
        assert "AI Agent 但简历偏数据分析" in snap


# ═══════════════════════════════════════════════════════════════════
# inbox.kind='question' + answer_question
# ═══════════════════════════════════════════════════════════════════


class TestQuestionInbox:
    def test_enqueue_question_creates_kind_question_item(self, app_client):
        client, store = app_client
        item = inbox_mod.enqueue_question(
            store, question="你想投哪家?",
            context="agent 找了 3 家匹配的, 你挑",
            options=[
                {"id": "anthropic", "label": "Anthropic"},
                {"id": "cresta", "label": "Cresta"},
                {"id": "axiomatic", "label": "Axiomatic"},
            ],
        )
        assert item.kind == "question"
        assert item.question_options is not None
        assert len(item.question_options) == 3
        assert item.question_options[0]["id"] == "anthropic"

    def test_answer_writes_user_facts(self, app_client):
        client, store = app_client
        item = inbox_mod.enqueue_question(
            store, question="想不想改方向?", context="...",
            options=[
                {"id": "stay", "label": "保持 AI Agent 不变"},
                {"id": "broader", "label": "扩到 LLM 应用都行"},
            ],
        )
        result = inbox_mod.answer_question(
            store, item.id, option_id="broader",
        )
        assert result.status == "approved"
        # user_facts should have a row about this answer
        with store.connect() as conn:
            facts = conn.execute(
                "SELECT fact_text FROM user_facts WHERE source_skill = 'ask_user_question'"
            ).fetchall()
        assert len(facts) == 1
        assert "扩到 LLM 应用都行" in facts[0][0]
        assert "想不想改方向" in facts[0][0]

    def test_answer_question_idempotent(self, app_client):
        client, store = app_client
        item = inbox_mod.enqueue_question(
            store, question="?", context="",
            options=[{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
        )
        inbox_mod.answer_question(store, item.id, option_id="a")
        # Second answer should fail (already answered)
        with pytest.raises(ValueError):
            inbox_mod.answer_question(store, item.id, option_id="b")

    def test_answer_endpoint_writes_user_facts(self, app_client):
        client, store = app_client
        item = inbox_mod.enqueue_question(
            store, question="?", context="",
            options=[{"id": "a", "label": "A"}],
        )
        resp = client.post(
            f"/inbox/{item.id}/answer",
            data={"option_id": "a"},
            follow_redirects=False,
        )
        assert resp.status_code == 303
        with store.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM user_facts WHERE source_skill='ask_user_question'"
            ).fetchone()[0]
        assert n == 1


# ═══════════════════════════════════════════════════════════════════
# Home — ambient redesign
# ═══════════════════════════════════════════════════════════════════


class TestHomeAmbientRedesign:
    def test_pending_questions_render_at_top_with_buttons(self, app_client):
        client, store = app_client
        item = inbox_mod.enqueue_question(
            store, question="你想投哪家公司?",
            context="agent 找了 3 家",
            options=[
                {"id": "anthropic", "label": "Anthropic"},
                {"id": "cresta", "label": "Cresta"},
            ],
        )
        resp = client.get("/")
        assert "Agent 在等你回答" in resp.text
        assert "你想投哪家公司?" in resp.text
        # Both option buttons rendered (text with surrounding whitespace)
        assert "Anthropic" in resp.text
        assert "Cresta" in resp.text
        assert 'value="anthropic"' in resp.text  # button submits the option_id
        # Form posts to /inbox/{id}/answer
        assert f'/inbox/{item.id}/answer' in resp.text

    def test_agent_inner_monologue_card_renders(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, "
                " iterations, final_answer, started_at, ended_at) "
                "VALUES ('cron_wake', 'x', 'ok', 3, "
                " '我看了 state, 决定 lay low — 深夜了', "
                " julianday('now'), julianday('now'))"
            )
        resp = client.get("/")
        # W15.16 — copy de-jargonized: "当前在想" → "上次跑完说"
        assert "Agent 上次跑完说" in resp.text or "Agent 当前在想" in resp.text
        assert "我看了 state, 决定 lay low" in resp.text

    def test_mission_control_demoted_to_collapsed(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        # Mission Control still exists but inside <details>, not the hero
        assert "Mission Control" in resp.text
        # Look for "Mission Control" inside a <details>
        # The new placement: collapsed, with debug label
        assert "(debug" in resp.text or "(folded" in resp.text or "details" in resp.text.lower()

    def test_no_pending_questions_card_when_none(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        assert "Agent 在等你回答" not in resp.text


# ═══════════════════════════════════════════════════════════════════
# ask_user_question agent tool
# ═══════════════════════════════════════════════════════════════════


class TestAskUserQuestionTool:
    def test_creates_question_inbox_item(self, app_client):
        from offerguide.agent.loop import AgentLoop
        client, store = app_client
        skills = discover_skills(SKILLS_ROOT)

        class _StubLLM:
            def chat(self, *a, **kw): return LLMResponse(content="{}", model="stub")
            def chat_with_tools(self, *a, **kw): return LLMResponse(content="", model="stub")
            def close(self): pass

        loop = AgentLoop(
            llm=_StubLLM(), runtime=SkillRuntime(llm=_StubLLM(), store=store),
            store=store, skills=skills, master_resume_text="x",
        )
        loop._current_run_id = 42

        tc = ToolCall(
            id="t", name="ask_user_question",
            arguments={
                "question": "改 north star 还是改简历重点?",
                "context": "你 north star AI Agent 但简历偏数据分析",
                "options": [
                    {"id": "change_ns", "label": "改 north star 到数据科学"},
                    {"id": "rewrite_resume", "label": "重写简历突出 agent 倾向"},
                ],
            },
        )
        result = loop._execute_action_tool(tc)
        assert "OK:" in result and "kind=question" in result

        # Verify inbox row
        with store.connect() as conn:
            row = conn.execute(
                "SELECT kind, source_agent_run_id FROM inbox_items WHERE id = "
                "(SELECT MAX(id) FROM inbox_items)"
            ).fetchone()
        assert row[0] == "question"
        assert row[1] == 42

    def test_rejects_lt_2_options(self, app_client):
        from offerguide.agent.loop import AgentLoop
        client, store = app_client
        skills = discover_skills(SKILLS_ROOT)

        class _StubLLM:
            def chat(self, *a, **kw): return LLMResponse(content="{}", model="stub")
            def chat_with_tools(self, *a, **kw): return LLMResponse(content="", model="stub")
            def close(self): pass

        loop = AgentLoop(
            llm=_StubLLM(), runtime=SkillRuntime(llm=_StubLLM(), store=store),
            store=store, skills=skills, master_resume_text="x",
        )
        loop._current_run_id = 1
        result = loop._execute_action_tool(
            ToolCall(id="t", name="ask_user_question",
                     arguments={
                         "question": "?", "context": "",
                         "options": [{"id": "a", "label": "A"}],
                     }),
        )
        assert "ERROR" in result and "2 options" in result
