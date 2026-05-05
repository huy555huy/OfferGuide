"""W13.2 maintenance — daemon jobs as agent-callable action tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import offerguide
from offerguide.agent import AgentLoop, snapshot_state
from offerguide.agent.maintenance import (
    MAINTENANCE_TOOL_NAMES,
    MAINTENANCE_TOOL_SCHEMAS,
    MaintenanceCtx,
    execute_maintenance_tool,
)
from offerguide.llm import LLMResponse, ToolCall
from offerguide.skills import SkillRuntime, discover_skills

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "maint.db")
    s.init_schema()
    return s


# ═══════════════════════════════════════════════════════════════════
# Schema sanity
# ═══════════════════════════════════════════════════════════════════


class TestMaintenanceSchemas:
    def test_maintenance_tools_declared(self):
        names = MAINTENANCE_TOOL_NAMES
        # Original W13.2 tools
        assert "discover_new_jobs" in names
        assert "enrich_thin_jds" in names
        assert "classify_corpus" in names
        assert "check_silent_applications" in names
        assert "refresh_company_corpus" in names
        assert "extract_facts_from_runs" in names
        assert "regenerate_company_brief" in names
        # W14.18: score_unscored_jobs added so the central agent can
        # decide when to score (instead of having a separate cron daemon).
        assert "score_unscored_jobs" in names
        assert len(names) >= 8

    def test_schemas_are_openai_compliant(self):
        for sc in MAINTENANCE_TOOL_SCHEMAS:
            assert sc["type"] == "function"
            fn = sc["function"]
            assert isinstance(fn["name"], str) and fn["name"]
            assert isinstance(fn["description"], str)
            params = fn["parameters"]
            assert params["type"] == "object"
            # Tools that need args specify required; agent will see them
            if "company" in params["properties"]:
                assert "company" in params["required"]


# ═══════════════════════════════════════════════════════════════════
# Maintenance tools wired to AgentLoop
# ═══════════════════════════════════════════════════════════════════


class TestAgentLoopWiring:
    def test_maintenance_tools_appear_in_loop_schemas(self, store):
        skills = discover_skills(SKILLS_ROOT)
        class _StubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_StubLLM(), store=store)

        class _AgentStub:
            def chat(self, messages, **kw): return LLMResponse(content="", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="done", model="stub")
        loop = AgentLoop(
            llm=_AgentStub(), runtime=runtime, store=store, skills=skills,
        )
        names = {sc["function"]["name"] for sc in loop._tool_schemas}
        # All 7 maintenance tools should be available
        for name in MAINTENANCE_TOOL_NAMES:
            assert name in names

    def test_unknown_maintenance_tool_returns_error(self, store):
        skills = discover_skills(SKILLS_ROOT)
        class _StubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_StubLLM(), store=store)

        class _AgentStub:
            def chat(self, messages, **kw): return LLMResponse(content="", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="x", model="stub")
        loop = AgentLoop(
            llm=_AgentStub(), runtime=runtime, store=store, skills=skills,
        )
        # Direct dispatch of a name that's NOT in MAINTENANCE_TOOL_NAMES but
        # also not a SKILL → should return ERROR not raise
        result = loop._execute_tool(ToolCall(
            id="x", name="not_a_real_tool", arguments={},
        ))
        assert result.startswith("ERROR")


# ═══════════════════════════════════════════════════════════════════
# execute_maintenance_tool dispatch
# ═══════════════════════════════════════════════════════════════════


class TestMaintenanceDispatch:
    def test_missing_company_returns_error(self, store):
        skills = discover_skills(SKILLS_ROOT)
        class _StubLLM:
            def chat(self, messages, **kw): return LLMResponse(content="{}", model="stub")
        ctx = MaintenanceCtx(
            store=store, llm=_StubLLM(), runtime=SkillRuntime(llm=_StubLLM(), store=store),
            skills=skills,
        )
        result = execute_maintenance_tool("refresh_company_corpus", {}, ctx)
        assert "ERROR" in result
        assert "company" in result

    def test_unknown_name_returns_error(self, store):
        skills = discover_skills(SKILLS_ROOT)
        class _StubLLM:
            def chat(self, messages, **kw): return LLMResponse(content="{}", model="stub")
        ctx = MaintenanceCtx(
            store=store, llm=_StubLLM(), runtime=SkillRuntime(llm=_StubLLM(), store=store),
            skills=skills,
        )
        result = execute_maintenance_tool("nonexistent", {}, ctx)
        assert result.startswith("ERROR")


# ═══════════════════════════════════════════════════════════════════
# Snapshot maintenance hints (W13.2)
# ═══════════════════════════════════════════════════════════════════


class TestSnapshotObservations:
    """Snapshot now surfaces FACTS not RULES.

    Pre-W13.x snapshots had lines like 'thin JDs: 4 → 调 enrich_thin_jds 如果 > 0'
    which embedded the decision tree in the prompt. Current shape: just states
    the count + lets the agent decide whether to act."""

    def test_empty_db_shows_zero_observations(self, store):
        snap = snapshot_state(store)
        # New section names: '系统观察' instead of '维护待办 hints'
        assert "系统观察" in snap or "情境" in snap
        assert "raw_text < 200" in snap
        assert "0 个" in snap  # zero counts spelled out

    def test_thin_jd_count_in_observations(self, store):
        with store.connect() as conn:
            for i in range(3):
                conn.execute(
                    "INSERT INTO jobs(source, source_id, url, title, company, "
                    "  raw_text, content_hash) VALUES (?,?,?,?,?,?,?)",
                    ("manual", f"j{i}", "http://x", "t", "c", "x" * 50, f"h{i}"),
                )
        snap = snapshot_state(store)
        assert "raw_text < 200 字: 3 个" in snap

    def test_silent_apps_count_in_observations(self, store):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, "
                "  raw_text, content_hash) VALUES ('m','j1','x','t','c',?,'h1')",
                ("x" * 250,),
            )
            # 14+ day silent app
            conn.execute(
                "INSERT INTO applications(job_id, status, last_status_change) "
                "VALUES (1, 'applied', julianday('now') - 14)"
            )
        snap = snapshot_state(store)
        # Either 7-14 or 14+ category should appear
        assert "申请" in snap and "未推进" in snap

    def test_unclassified_corpus_count(self, store):
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO interview_experiences(company, raw_text, source, "
                "  content_hash) VALUES ('字节', 'sample text', 'manual', 'h1')"
            )
        snap = snapshot_state(store)
        assert "interview_experiences 未分类: 1 个" in snap

    def test_no_decision_rules_leaked_into_snapshot(self, store):
        """Regression: snapshot must NOT contain '→ 调 X' rule-style hints.
        The agent should infer what to do, not be told."""
        snap = snapshot_state(store)
        # Old rule-style language gone
        assert "→ 调 enrich_thin_jds" not in snap
        assert "→ 调 classify_corpus" not in snap
        assert "→ 调 check_silent_applications" not in snap


# ═══════════════════════════════════════════════════════════════════
# scheduler factory
# ═══════════════════════════════════════════════════════════════════


class TestSchedulerFactory:
    def test_build_agent_wake_scheduler_registers_jobs(self):
        from offerguide.autonomous.scheduler import build_agent_wake_scheduler
        from offerguide.config import Settings

        sched = build_agent_wake_scheduler(
            settings=Settings(deepseek_api_key="", db_path=":memory:")
        )
        names = sched.list_jobs()
        # W14.18: collapsed back to ONE cron — the central agent's
        # heartbeat. discover_new_jobs / score_unscored_jobs are now
        # tools the agent calls itself, not independent cron daemons.
        assert names == ["wake_agent"]
        sched.shutdown()

    def test_wake_agent_skips_when_no_llm(self, tmp_path, monkeypatch):
        from offerguide.autonomous.scheduler import build_agent_wake_scheduler
        from offerguide.config import Settings

        monkeypatch.delenv("OFFERGUIDE_LLM_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        sched = build_agent_wake_scheduler(
            settings=Settings(deepseek_api_key="", db_path=str(tmp_path / "x.db"))
        )
        result = sched.trigger_once("wake_agent")
        sched.shutdown()
        assert isinstance(result, dict)
        assert "skipped" in result or result.get("agent_run_id") is None
