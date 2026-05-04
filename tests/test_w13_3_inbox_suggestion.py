"""W13.3 inbox → agent_suggestion → user_thumbs → evolution_signals loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import offerguide
from offerguide import inbox as inbox_mod
from offerguide.agent import AgentLoop
from offerguide.evolution.signals import fetch_signals
from offerguide.llm import LLMResponse, ToolCall
from offerguide.skills import SkillRuntime, discover_skills

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "inbox_sug.db")
    s.init_schema()
    return s


# ═══════════════════════════════════════════════════════════════════
# Schema migration
# ═══════════════════════════════════════════════════════════════════


class TestInboxSchemaMigration:
    def test_new_columns_added(self, store):
        with store.connect() as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(inbox_items)").fetchall()}
        assert "source_agent_run_id" in cols
        assert "source_skill_name" in cols
        assert "source_skill_version" in cols
        assert "proposed_action_json" in cols

    def test_legacy_inbox_items_still_load(self, store):
        """Old enqueue() with no attribution still works (backward compat)."""
        item = inbox_mod.enqueue(
            store, kind="consider_jd", title="legacy item",
            body="x", payload={"y": 1},
        )
        assert item.kind == "consider_jd"
        assert item.source_agent_run_id is None
        assert item.proposed_action is None


# ═══════════════════════════════════════════════════════════════════
# enqueue_agent_suggestion
# ═══════════════════════════════════════════════════════════════════


class TestEnqueueAgentSuggestion:
    def test_basic_enqueue_with_attribution(self, store):
        item = inbox_mod.enqueue_agent_suggestion(
            store,
            title="建议 tailor 字节 AI Agent 简历",
            body="snapshot 显示这个 job 还没 tailor, score 0.78 高优",
            source_agent_run_id=42,
            source_skill_name="score_match",
            source_skill_version="0.1.0",
            source_skill_run_id=7,
            proposed_action={"tool": "tailor_resume", "args": {"job_id": 1}},
        )
        assert item.kind == "agent_suggestion"
        assert item.source_agent_run_id == 42
        assert item.source_skill_name == "score_match"
        assert item.source_skill_version == "0.1.0"
        assert item.proposed_action == {"tool": "tailor_resume", "args": {"job_id": 1}}
        # source_skill_run_id stored in payload (heuristic)
        assert item.payload.get("source_skill_run_id") == 7

    def test_no_proposed_action_is_optional(self, store):
        item = inbox_mod.enqueue_agent_suggestion(
            store,
            title="只是通知", body="没具体 action",
            source_skill_name="score_match",
            source_skill_version="0.1.0",
        )
        assert item.proposed_action is None


# ═══════════════════════════════════════════════════════════════════
# decide() fans out user_thumbs to evolution_signals (W13.3 core)
# ═══════════════════════════════════════════════════════════════════


class TestDecideFansOutSignals:
    def test_approve_writes_positive_thumbs(self, store):
        item = inbox_mod.enqueue_agent_suggestion(
            store,
            title="试试 tailor 这个 job", body="...",
            source_skill_name="tailor_resume",
            source_skill_version="0.2.0",
            source_skill_run_id=99,
        )
        inbox_mod.decide(store, item.id, decision="approved")

        signals = fetch_signals(store, skill_name="tailor_resume")
        assert len(signals) == 1
        assert signals[0].signal_kind == "user_thumbs"
        assert signals[0].signal_value == 1.0  # approved → +1
        assert signals[0].skill_run_id == 99
        assert signals[0].skill_version == "0.2.0"

    def test_reject_writes_negative_thumbs(self, store):
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="x", body="x",
            source_skill_name="score_match",
            source_skill_version="0.1.0",
        )
        inbox_mod.decide(store, item.id, decision="rejected")
        signals = fetch_signals(store, skill_name="score_match")
        assert len(signals) == 1
        assert signals[0].signal_value == -1.0

    def test_dismiss_writes_no_signal(self, store):
        """Dismissed = user said neither yes nor no, no signal."""
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="x", body="x",
            source_skill_name="score_match",
            source_skill_version="0.1.0",
        )
        inbox_mod.decide(store, item.id, decision="dismissed")
        signals = fetch_signals(store, skill_name="score_match")
        assert len(signals) == 0

    def test_legacy_kind_no_signal(self, store):
        """Pre-W13.3 'consider_jd' items don't have skill attribution → no signal."""
        item = inbox_mod.enqueue(
            store, kind="consider_jd", title="legacy", body="x",
        )
        inbox_mod.decide(store, item.id, decision="approved")
        signals = fetch_signals(store, skill_name="score_match")
        assert len(signals) == 0

    def test_decision_note_persisted(self, store):
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="x", body="x",
            source_skill_name="score_match",
            source_skill_version="0.1.0",
        )
        inbox_mod.decide(
            store, item.id, decision="rejected", note="JD 是 OD 岗位",
        )
        re_fetched = inbox_mod.get(store, item.id)
        assert re_fetched.decision_note == "JD 是 OD 岗位"

    def test_already_decided_raises(self, store):
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="x", body="x",
            source_skill_name="score_match",
            source_skill_version="0.1.0",
        )
        inbox_mod.decide(store, item.id, decision="approved")
        with pytest.raises(ValueError):
            inbox_mod.decide(store, item.id, decision="rejected")


# ═══════════════════════════════════════════════════════════════════
# AgentLoop write_suggestion action tool
# ═══════════════════════════════════════════════════════════════════


class _StubLLM:
    def chat(self, messages, **kw):
        return LLMResponse(content="{}", model="stub")


class TestWriteSuggestionTool:
    def test_tool_in_schemas(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills)
        names = {sc["function"]["name"] for sc in loop._tool_schemas}
        assert "write_suggestion" in names

    def test_write_suggestion_creates_inbox_item(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills)
        # Simulate being in the middle of a run
        loop._current_run_id = 7
        loop._current_skill_invocations = {
            "tc_x": {
                "skill_name": "score_match",
                "skill_version": "0.2.0",
                "skill_run_id": 99,
            }
        }

        result = loop._execute_tool(ToolCall(
            id="x", name="write_suggestion",
            arguments={
                "title": "试试 tailor 字节 AI Agent",
                "body": "score=0.85 值得 tailor",
                "skill_to_call": "tailor_resume",
                "skill_args_json": '{"job_id": 42}',
            },
        ))
        assert "OK" in result
        assert "score_match" in result  # attribution shown
        assert "skill_run#99" in result

        items = inbox_mod.list_items(store)
        assert len(items) == 1
        i = items[0]
        assert i.kind == "agent_suggestion"
        assert i.source_skill_name == "score_match"
        assert i.source_agent_run_id == 7
        assert i.proposed_action == {"tool": "tailor_resume", "args": {"job_id": 42}}

    def test_missing_title_returns_error(self, store):
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills)

        result = loop._execute_tool(ToolCall(
            id="x", name="write_suggestion",
            arguments={"body": "missing title"},
        ))
        assert "ERROR" in result

    def test_no_skill_attribution_when_loop_state_empty(self, store):
        """If no SKILL was invoked yet in this run, suggestion has no attribution."""
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        class _A:
            def chat(self, messages, **kw): return LLMResponse(content="x", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")
        loop = AgentLoop(llm=_A(), runtime=runtime, store=store, skills=skills)
        loop._current_run_id = 7
        loop._current_skill_invocations = {}  # empty

        loop._execute_tool(ToolCall(
            id="x", name="write_suggestion",
            arguments={"title": "x", "body": "y"},
        ))
        items = inbox_mod.list_items(store)
        assert len(items) == 1
        assert items[0].source_skill_name is None  # no SKILL to attribute


# ═══════════════════════════════════════════════════════════════════
# End-to-end loop
# ═══════════════════════════════════════════════════════════════════


class TestEndToEndSuggestionLoop:
    def test_agent_writes_suggestion_then_user_approves_then_signal(self, store):
        """Full loop: agent writes suggestion → user approves → signal lands in evolution_signals."""
        # 1. Simulate agent writing a suggestion
        item = inbox_mod.enqueue_agent_suggestion(
            store,
            title="建议 tailor 字节 AI Agent 简历",
            body="snapshot 显示 score 0.85 但还没 tailor, 这是高优 job",
            source_agent_run_id=42,
            source_skill_name="score_match",
            source_skill_version="0.2.0",
            source_skill_run_id=99,
            proposed_action={"tool": "tailor_resume", "args": {"job_id": 1}},
        )

        # 2. User approves via UI
        decided = inbox_mod.decide(store, item.id, decision="approved",
                                    note="是高优, 同意 tailor")
        assert decided.status == "approved"

        # 3. Signal landed
        signals = fetch_signals(store, skill_name="score_match")
        assert len(signals) == 1
        s = signals[0]
        assert s.signal_kind == "user_thumbs"
        assert s.signal_value == 1.0
        assert s.signal_weight == 2.0  # default thumbs weight
        assert s.skill_run_id == 99
        assert "agent_run#42" in (s.notes or "")
        assert "approved" in (s.notes or "")

    def test_fitness_includes_new_thumbs_signal(self, store):
        """After user approves a suggestion, compute_fitness should reflect it."""
        from offerguide.evolution.fitness import compute_fitness

        # Add 5 mediocre critic scores
        from offerguide.evolution.signals import record_critic_signal
        for _ in range(5):
            record_critic_signal(
                store, skill_name="score_match", skill_version="0.1.0",
                skill_run_id=None, score=0.5,
            )

        # Baseline fitness ≈ 0.5
        before = compute_fitness(store, skill_name="score_match")
        assert before.fitness == pytest.approx(0.5)

        # Agent writes a suggestion, user approves → +1 thumbs (weight 2.0, normalized to 1.0)
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="x", body="y",
            source_skill_name="score_match",
            source_skill_version="0.1.0",
        )
        inbox_mod.decide(store, item.id, decision="approved")

        after = compute_fitness(store, skill_name="score_match")
        # weighted: (5*0.5*1.0 + 1*1.0*2.0) / (5 + 2) = 4.5/7 ≈ 0.643
        assert after.fitness > before.fitness
        assert after.fitness == pytest.approx(4.5 / 7, abs=0.01)
