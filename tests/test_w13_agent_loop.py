"""W13 — central agent loop with model in the driver's seat.

This test pack covers the new ``agent/loop.py`` paradigm where the model
decides which SKILL to call (via OpenAI tool-calling), when to stop, and
how to compose tools — replacing the W4 hardcoded LangGraph routing.

Tests use a fully scripted ``_StubLLMWithTools`` so the agent loop's
control flow is exercised without any network / token cost. The real-LLM
end-to-end smoke is in ``test_w13_agent_loop_real_llm.py`` (gated on env).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import offerguide
from offerguide.agent import (
    AgentEvent,
    AgentLoop,
    AgentRunResult,
    build_tool_schemas,
    snapshot_state,
)
from offerguide.llm import LLMResponse, ToolCall
from offerguide.skills import SkillRuntime, discover_skills

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# Stub LLM with scripted tool-call behavior
# ═══════════════════════════════════════════════════════════════════


class _StubLLMWithTools:
    """Returns scripted responses on each chat_with_tools / chat call.

    Construct with two queues:
      - ``tool_responses``: list of LLMResponse for chat_with_tools (the
         agent loop). Each can have empty or non-empty ``tool_calls``.
      - ``critic_responses``: list of LLMResponse for chat (the critic).
    """

    def __init__(
        self,
        *,
        tool_responses: list[LLMResponse] | None = None,
        critic_responses: list[LLMResponse] | None = None,
    ):
        self._tool_q = list(tool_responses or [])
        self._critic_q = list(critic_responses or [])
        self.chat_with_tools_calls: list[dict] = []
        self.chat_calls: list[dict] = []

    def chat_with_tools(self, messages, *, tools, temperature=0.4, **kw):
        self.chat_with_tools_calls.append({
            "messages_count": len(messages),
            "tools_count": len(tools),
            "last_message_role": messages[-1].get("role") if messages else None,
        })
        if not self._tool_q:
            # Fallback: tell agent we're done
            return LLMResponse(
                content="(stub: out of scripted responses, ending)",
                model="stub",
                finish_reason="stop",
            )
        return self._tool_q.pop(0)

    def chat(self, messages, **kw):
        self.chat_calls.append({"messages_count": len(messages)})
        if not self._critic_q:
            # Default critic — no signal
            return LLMResponse(
                content=json.dumps({
                    "overall": 0.7,
                    "dimensions": {},
                    "notes": "(stub critic default)",
                    "improvement_hint": "n/a",
                }),
                model="stub",
            )
        return self._critic_q.pop(0)


def _stub_resp_with_tool(name: str, args: dict, content: str = "") -> LLMResponse:
    """Build a scripted LLMResponse that triggers one tool call."""
    return LLMResponse(
        content=content,
        model="stub",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(
                id=f"call_{name}",
                name=name,
                arguments=args,
                arguments_raw=json.dumps(args, ensure_ascii=False),
            )
        ],
    )


def _stub_resp_final(content: str) -> LLMResponse:
    """Build a scripted LLMResponse where the model says no-more-tools."""
    return LLMResponse(content=content, model="stub", finish_reason="stop", tool_calls=[])


# ═══════════════════════════════════════════════════════════════════
# Tool-schema generation
# ═══════════════════════════════════════════════════════════════════


class TestToolSchemas:
    def test_all_skills_become_tools(self):
        skills = discover_skills(SKILLS_ROOT)
        schemas = build_tool_schemas(skills)
        assert len(schemas) == len(skills)
        names = {s["function"]["name"] for s in schemas}
        # Spot-check key SKILLs are exposed as tools
        assert "tailor_resume" in names
        assert "score_match" in names
        assert "mock_interview" in names

    def test_tool_schema_shape(self):
        skills = discover_skills(SKILLS_ROOT)
        schemas = build_tool_schemas(skills)
        for sc in schemas:
            assert sc["type"] == "function"
            fn = sc["function"]
            assert isinstance(fn["name"], str) and fn["name"]
            assert isinstance(fn["description"], str)
            assert len(fn["description"]) <= 1024
            params = fn["parameters"]
            assert params["type"] == "object"
            assert isinstance(params["properties"], dict)
            assert isinstance(params["required"], list)
            assert params.get("additionalProperties") is False
            # required list should match properties keys (declared inputs)
            assert set(params["required"]) == set(params["properties"].keys())

    def test_input_hints_are_human_readable(self):
        """Common input names should get useful Chinese hints, not f-string fallbacks."""
        skills = [s for s in discover_skills(SKILLS_ROOT) if s.name == "tailor_resume"]
        assert skills, "tailor_resume should exist"
        sc = build_tool_schemas(skills)[0]
        props = sc["function"]["parameters"]["properties"]
        # company should have a curated hint, not the f-string fallback
        assert "字节" in props["company"]["description"] or "公司" in props["company"]["description"]
        # master_resume should mention 'ground truth' or '不能编造'
        assert any(
            kw in props["master_resume"]["description"]
            for kw in ("ground truth", "编造", "主简历")
        )


# ═══════════════════════════════════════════════════════════════════
# State snapshot
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def empty_store(tmp_path):
    s = offerguide.Store(tmp_path / "agent_loop.db")
    s.init_schema()
    return s


@pytest.fixture
def populated_store(tmp_path):
    s = offerguide.Store(tmp_path / "agent_loop_pop.db")
    s.init_schema()
    with s.connect() as conn:
        # Insert 2 jobs (one with raw_text >= 200, one too short)
        conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, company, location, "
            "raw_text, content_hash) VALUES (?,?,?,?,?,?,?,?)",
            (
                "boss", "j1", "https://x", "AI Agent 实习",
                "字节跳动", "北京", "x" * 800, "h1",
            ),
        )
        conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, company, location, "
            "raw_text, content_hash) VALUES (?,?,?,?,?,?,?,?)",
            (
                "boss", "j2", "https://y", "前端实习",
                "腾讯", "深圳", "y" * 50, "h2",
            ),
        )
        # 1 application
        conn.execute(
            "INSERT INTO applications(job_id, status, applied_at) VALUES (1, 'applied', julianday('now'))"
        )
        # 1 SKILL run
        conn.execute(
            "INSERT INTO skill_runs(skill_name, skill_version, input_hash, "
            "input_json, output_json, latency_ms) VALUES (?,?,?,?,?,?)",
            ("score_match", "0.1.0", "h", "{}", '{"probability":0.7}', 1234),
        )
        # 2 user_facts
        conn.execute(
            "INSERT INTO user_facts(fact_text, kind, confidence, used_count) "
            "VALUES (?,?,?,?)",
            ("用户优先 AI Agent 后端岗位", "preference", 0.9, 5),
        )
        conn.execute(
            "INSERT INTO user_facts(fact_text, kind, confidence, used_count) "
            "VALUES (?,?,?,?)",
            ("用户对纯算法刷题型 OD 岗不感兴趣", "preference", 0.7, 3),
        )
    return s


class TestSnapshot:
    def test_empty_store_snapshot_doesnt_crash(self, empty_store):
        snap = snapshot_state(empty_store)
        assert "数据库当前为空" in snap or "Snapshot" in snap

    def test_populated_snapshot_includes_each_section(self, populated_store):
        snap = snapshot_state(populated_store)
        # Should mention the long-raw_text job, NOT the too-short one
        assert "字节跳动" in snap
        assert "腾讯" not in snap  # raw_text=50 chars, filtered out
        assert "applied" in snap
        assert "score_match" in snap
        assert "AI Agent 后端岗位" in snap

    def test_snapshot_marks_jobs_with_existing_apps(self, populated_store):
        snap = snapshot_state(populated_store)
        # job#1 has an application, should show APP marker
        assert "[APP=applied]" in snap


# ═══════════════════════════════════════════════════════════════════
# AgentLoop end-to-end (with stub LLM)
# ═══════════════════════════════════════════════════════════════════


class TestAgentLoopE2E:
    def test_loop_with_zero_tool_calls_just_finalizes(self, empty_store):
        """Model can decide there's nothing to do — emits final immediately.

        W20.3: trivial run (≤1 iter + 0 SKILL invoked) → critic SKIPPED via
        gating. We assert 'critique_skipped' event with reason='trivial_run'
        instead of 'critique'. Critic is no longer wasted on empty trajectories.
        """
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLMWithTools(), store=empty_store)
        stub = _StubLLMWithTools(
            tool_responses=[
                _stub_resp_final("数据库为空, 暂无可做事项, 收工。"),
            ],
        )
        loop = AgentLoop(
            llm=stub, runtime=runtime, store=empty_store,
            skills=skills, max_iterations=4,
            critic_enabled=True,  # W20.4 — explicit since default is now OFF
        )
        result = loop.run(goal="今日例行检查", trigger_kind="test")
        assert result.error is None
        assert result.iterations == 1
        assert "暂无" in result.final_answer
        kinds = [e.kind for e in result.events]
        assert "state_snapshot" in kinds
        assert "thinking" in kinds
        assert "final" in kinds
        # W20.3: trivial run → critic gated out (skip event instead of run event)
        assert "critique_skipped" in kinds
        assert "critique" not in kinds
        skip_event = next(e for e in result.events if e.kind == "critique_skipped")
        assert skip_event.payload.get("reason") == "trivial_run"

    def test_loop_executes_one_tool_then_finalizes(self, populated_store):
        """Model calls 1 tool, sees result, then issues final."""
        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(
            llm=_StubLLMWithTools(  # SkillRuntime needs an LLM for SKILL invocation
                tool_responses=[
                    LLMResponse(content='{"probability": 0.85, "reasoning": "good fit"}',
                                model="stub"),
                ],
            ),
            store=populated_store,
        )
        # Override SkillRuntime's LLM with one that responds to SKILL invocations
        # via plain chat (not chat_with_tools)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(
                    content='{"probability": 0.85, "reasoning": "matches AI Agent 偏好"}',
                    model="stub",
                )
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)

        agent_stub = _StubLLMWithTools(
            tool_responses=[
                _stub_resp_with_tool(
                    "score_match",
                    {"job_text": "字节 AI Agent 实习" * 30, "user_profile": "TestUser, AI 方向"},
                    content="我看到 job#1 (字节) 没 score, 先 score 一下。",
                ),
                _stub_resp_final("score 完成, probability=0.85, 是高优, 通知用户。"),
            ],
        )
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=populated_store,
            skills=skills, max_iterations=4,
        )
        result = loop.run(goal="找今日要 score 的新 job", trigger_kind="test")
        assert result.error is None
        assert result.iterations == 2
        # We expect: state_snapshot, thinking#0, tool_call, tool_result, thinking#1, final, critique
        kinds = [e.kind for e in result.events]
        assert kinds.count("tool_call") == 1
        assert kinds.count("tool_result") == 1
        assert "final" in kinds
        # The tool call should reference the actual SKILL we scripted
        tc_event = next(e for e in result.events if e.kind == "tool_call")
        assert tc_event.payload["name"] == "score_match"

    def test_loop_persists_agent_run_with_trajectory(self, empty_store):
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)

        agent_stub = _StubLLMWithTools(
            tool_responses=[_stub_resp_final("done")],
        )
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=empty_store, skills=skills,
        )
        result = loop.run(goal="test goal", trigger_kind="unit_test")
        assert result.run_id is not None

        with empty_store.connect() as conn:
            row = conn.execute(
                "SELECT trigger_kind, goal, status, iterations, final_answer, "
                "       trajectory_json, critic_score "
                "FROM agent_runs WHERE id = ?",
                (result.run_id,),
            ).fetchone()
        assert row is not None
        trigger, goal, status, iters, final_ans, traj_json, critic = row
        assert trigger == "unit_test"
        assert goal == "test goal"
        assert status == "ok"
        assert iters == 1
        assert "done" in final_ans
        traj = json.loads(traj_json)
        assert isinstance(traj, list)
        assert len(traj) >= 2  # at least state_snapshot + final
        assert any(e["kind"] == "state_snapshot" for e in traj)
        # W20.3 — trigger_kind="unit_test" + 0 SKILL + 1 iter = trivial,
        # critic skipped → critic_score=None persisted. Test validates the
        # row was still written with the trivial trajectory; just no critic.
        assert critic is None

    def test_loop_handles_unknown_tool_gracefully(self, empty_store):
        """When model calls a tool that doesn't exist, error flows through to model."""
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)

        agent_stub = _StubLLMWithTools(
            tool_responses=[
                _stub_resp_with_tool(
                    "nonexistent_tool", {"x": "y"},
                    content="trying a fake tool",
                ),
                _stub_resp_final("got error, giving up"),
            ],
        )
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=empty_store, skills=skills,
        )
        result = loop.run(goal="test", trigger_kind="test")
        # Loop should NOT crash; it should surface the error in tool_result
        assert result.iterations == 2
        tool_result_event = next(e for e in result.events if e.kind == "tool_result")
        assert "ERROR" in tool_result_event.payload["result_preview"]

    def test_loop_respects_max_iterations(self, empty_store):
        """Non-repeating tool calls should hit max_iter cap (not the
        circuit breaker — that fires only on REPEAT pattern)."""
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)

        # Each iter calls a DIFFERENT args (so circuit breaker doesn't fire)
        infinite_tool_calls = [
            _stub_resp_with_tool("score_match",
                                  {"job_text": f"unique_{i}" * 50, "user_profile": "y"})
            for i in range(10)
        ]
        agent_stub = _StubLLMWithTools(tool_responses=infinite_tool_calls)
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=empty_store,
            skills=skills, max_iterations=3,
        )
        result = loop.run(goal="never-ending test", trigger_kind="test")
        assert result.iterations == 3
        kinds = [e.kind for e in result.events]
        assert "error" in kinds
        assert any("max_iter" in str(e.payload) for e in result.events if e.kind == "error")

    def test_circuit_breaker_aborts_on_repeated_tool_calls(self, empty_store):
        """W13.7: same (tool, args) 3 times in a row → abort, force final."""
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)

        # Same tool call IDENTICAL args 5 times — circuit breaker should fire on 3rd
        repeat_calls = [
            _stub_resp_with_tool("score_match",
                                  {"job_text": "same" * 100, "user_profile": "same"})
            for _ in range(5)
        ]
        agent_stub = _StubLLMWithTools(tool_responses=repeat_calls)
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=empty_store,
            skills=skills, max_iterations=10,  # high max so we can prove breaker fired first
        )
        result = loop.run(goal="repeat test", trigger_kind="test")
        # Breaker fired on iter 3 (1-based: 1st, 2nd, 3rd identical → abort)
        assert result.iterations <= 3
        # An 'error' event with circuit_breaker mention should appear
        assert any(
            "circuit_breaker" in str(e.payload)
            for e in result.events if e.kind == "error"
        )
        # And a final event should follow (forced final)
        assert any(e.kind == "final" for e in result.events)
        # Final answer should mention the circuit breaker
        final = next(e for e in result.events if e.kind == "final")
        assert "circuit_breaker" in str(final.payload)

    def test_circuit_breaker_doesnt_fire_on_distinct_calls(self, empty_store):
        """Different args each iter should not trip the breaker."""
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)

        # 5 calls with different args, then a final
        diverse_calls = [
            _stub_resp_with_tool("score_match",
                                  {"job_text": f"diff_{i}" * 100, "user_profile": "y"})
            for i in range(5)
        ] + [_stub_resp_final("done")]
        agent_stub = _StubLLMWithTools(tool_responses=diverse_calls)
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=empty_store,
            skills=skills, max_iterations=10,
        )
        result = loop.run(goal="diverse calls", trigger_kind="test")
        assert "done" in result.final_answer
        # No circuit_breaker error
        assert not any(
            "circuit_breaker" in str(e.payload)
            for e in result.events if e.kind == "error"
        )

    def test_event_callback_streams_in_real_time(self, empty_store):
        """on_event callback must fire for each event, in order."""
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)

        agent_stub = _StubLLMWithTools(
            tool_responses=[_stub_resp_final("done")],
        )
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=empty_store, skills=skills,
        )

        streamed: list[dict] = []
        def cb(ev):
            streamed.append(dict(ev))

        result = loop.run(goal="x", trigger_kind="test", on_event=cb)
        assert len(streamed) == len(result.events)
        # Each callback dict has at least kind + at
        for ev in streamed:
            assert "kind" in ev
            assert "at" in ev

    def test_event_callback_exception_doesnt_break_loop(self, empty_store):
        """If on_event raises, loop continues — UI bugs shouldn't kill agent."""
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)

        agent_stub = _StubLLMWithTools(
            tool_responses=[_stub_resp_final("done")],
        )
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=empty_store, skills=skills,
        )

        def evil_cb(ev):
            raise RuntimeError("oops UI is broken")

        # Should NOT raise
        result = loop.run(goal="x", trigger_kind="test", on_event=evil_cb)
        assert result.error is None
        assert "done" in result.final_answer


# ═══════════════════════════════════════════════════════════════════
# AgentRunResult shape
# ═══════════════════════════════════════════════════════════════════


class TestCriticSignalAutoWrite:
    """W13.1: AgentLoop should write the critic_score into evolution_signals
    for every SKILL that ran during the trajectory. Without this, GEPA
    evolution has no quality signal to evolve against."""

    def test_critic_score_fans_out_to_each_invoked_skill(self, populated_store):
        """One agent run that invokes 1 SKILL should produce 1 critic signal."""
        from offerguide.evolution.signals import fetch_signals

        skills = discover_skills(SKILLS_ROOT)
        # SKILL stub returns a fixed JSON
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(
                    content='{"probability": 0.85, "reasoning": "ok"}',
                    model="stub",
                )
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)

        # Agent stub: 1 tool call (score_match) then final
        agent_stub = _StubLLMWithTools(
            tool_responses=[
                _stub_resp_with_tool(
                    "score_match",
                    {"job_text": "x" * 250, "user_profile": "TestUser statistics"},
                    content="scoring",
                ),
                _stub_resp_final("score=0.85, recommend"),
            ],
            critic_responses=[
                LLMResponse(
                    content=json.dumps({
                        "overall": 0.78,
                        "dimensions": {},
                        "notes": "good run",
                        "improvement_hint": "n/a",
                    }),
                    model="stub",
                ),
            ],
        )
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=populated_store,
            skills=skills, max_iterations=4,
            critic_enabled=True,  # W20.4 — explicit since default is now OFF
        )
        result = loop.run(goal="score the bytedance job", trigger_kind="test")
        assert result.critic_score == pytest.approx(0.78)

        # The critic_score should have been written to evolution_signals
        # tagged with skill_name=score_match
        signals = fetch_signals(populated_store, skill_name="score_match")
        assert len(signals) == 1, f"expected 1 signal, got {len(signals)}"
        sig = signals[0]
        assert sig.signal_kind == "critic"
        assert sig.signal_value == pytest.approx(0.78)
        assert sig.skill_run_id is not None
        assert sig.notes and "agent_run#" in sig.notes

    def test_lookup_tools_dont_get_signals(self, populated_store):
        """read_job / read_user_resume are stateless lookups — no SKILL
        version to evolve, so they shouldn't appear in evolution_signals."""

        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content='{}', model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)

        agent_stub = _StubLLMWithTools(
            tool_responses=[
                _stub_resp_with_tool("read_job", {"job_id": 1}, content="reading"),
                _stub_resp_final("done"),
            ],
        )
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=populated_store,
            skills=skills, master_resume_text="x",
        )
        loop.run(goal="just read the job", trigger_kind="test")

        # No SKILL was actually invoked — only a lookup. No signal expected.
        with populated_store.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM evolution_signals"
            ).fetchone()[0]
        assert count == 0

    def test_no_signals_when_critic_disabled(self, populated_store):
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content='{"x":1}', model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)

        agent_stub = _StubLLMWithTools(
            tool_responses=[
                _stub_resp_with_tool("score_match",
                                      {"job_text": "x" * 250, "user_profile": "y"}),
                _stub_resp_final("done"),
            ],
        )
        loop = AgentLoop(
            llm=agent_stub, runtime=runtime, store=populated_store,
            skills=skills, critic_enabled=False,
        )
        loop.run(goal="x", trigger_kind="test")

        with populated_store.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM evolution_signals"
            ).fetchone()[0]
        assert count == 0


class TestActionTools:
    """W13.1: agent can call evolve_skill and detect_evolution_candidates as
    action tools (separate from lookup + SKILL tools)."""

    def test_action_tools_appear_in_schemas(self, populated_store):
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=populated_store,
            skills=skills, master_resume_text="x" * 200,
        )
        names = {sc["function"]["name"] for sc in loop._tool_schemas}
        assert "evolve_skill" in names
        assert "detect_evolution_candidates" in names

    def test_detect_with_no_candidates_returns_friendly(self, empty_store):
        from offerguide.llm import ToolCall
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=empty_store,
            skills=skills,
        )
        result = loop._execute_tool(ToolCall(
            id="x", name="detect_evolution_candidates", arguments={},
        ))
        assert "暂无符合进化条件" in result

    def test_detect_finds_low_fitness_skill(self, populated_store):
        from offerguide.evolution.signals import record_critic_signal
        from offerguide.llm import ToolCall

        skills = discover_skills(SKILLS_ROOT)
        # Inject 12 low signals + a live variant
        for _ in range(12):
            record_critic_signal(
                populated_store, skill_name="score_match",
                skill_version="0.1.0", skill_run_id=None, score=0.3,
            )
        with populated_store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status, "
                "  promoted_at) VALUES (?,?,?,?, julianday('now'))",
                ("score_match", "0.1.0", "seed", "live"),
            )
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=populated_store,
            skills=skills,
        )
        result = loop._execute_tool(ToolCall(
            id="x", name="detect_evolution_candidates", arguments={},
        ))
        assert "score_match" in result
        assert "fitness=0.30" in result

    def test_evolve_skill_missing_skill_name_returns_error(self, empty_store):
        from offerguide.llm import ToolCall
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=empty_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=empty_store,
            skills=skills,
        )
        result = loop._execute_tool(ToolCall(
            id="x", name="evolve_skill", arguments={},
        ))
        assert result.startswith("ERROR")
        assert "skill_name" in result

    def test_evolve_skill_returns_summary(self, empty_store):
        """End-to-end: evolve_skill on a real SKILL, with stub LLM returning variants."""
        from offerguide.llm import ToolCall
        skills = discover_skills(SKILLS_ROOT)
        # Stub LLM that returns 2 variant bodies
        variants_payload = {
            "variants": [
                {
                    "version_label": "tighten constraints",
                    "body": "你是 OfferGuide 的简历评分员 (W13.1 evolved v1, 更严格的约束)。" * 20,
                    "rationale": "原 prompt 太松, 给低质量输入也给高分",
                },
                {
                    "version_label": "add CoT",
                    "body": "你是 OfferGuide 的简历评分员 (W13.1 evolved v2, 加 chain-of-thought)。" * 20,
                    "rationale": "原 prompt 直接出 JSON, 没思考过程",
                },
            ]
        }
        class _LLM:
            def chat(self, messages, **kw):
                return LLMResponse(content=json.dumps(variants_payload, ensure_ascii=False),
                                    model="stub")
        runtime = SkillRuntime(llm=_LLM(), store=empty_store)
        loop = AgentLoop(
            llm=_LLM(), runtime=runtime, store=empty_store, skills=skills,
        )
        result = loop._execute_tool(ToolCall(
            id="x", name="evolve_skill",
            arguments={"skill_name": "score_match", "num_variants": 2},
        ))
        assert "score_match" in result
        assert "持久化" in result
        # Verify variants landed in skill_variants
        with empty_store.connect() as conn:
            rows = conn.execute(
                "SELECT version, status FROM skill_variants WHERE skill_name = ?",
                ("score_match",),
            ).fetchall()
        assert len(rows) >= 1
        assert all(status == "shadow" for _, status in rows)


class TestArgumentParser:
    """Regression test for the ccvibe concatenated-JSON proxy bug.

    Caught in W13 dogfood #2: the proxy returned ``arguments=\"{}{...}\"``
    which strict ``json.loads`` rejects, causing tool calls to land with
    empty {} args 100% of the time. The robust parser handles this and
    other quirks."""

    def test_clean_json_passes_through(self):
        from offerguide.llm.client import _parse_tool_arguments
        assert _parse_tool_arguments('{"city": "北京"}') == {"city": "北京"}

    def test_concatenated_empty_then_real(self):
        """The actual ccvibe bug shape."""
        from offerguide.llm.client import _parse_tool_arguments
        result = _parse_tool_arguments('{}{"city": "北京"}')
        assert result == {"city": "北京"}

    def test_concatenated_all_empty(self):
        from offerguide.llm.client import _parse_tool_arguments
        # Should still return a dict (the last one), not raise
        result = _parse_tool_arguments('{}{}')
        assert result == {}

    def test_concatenated_real_then_empty(self):
        from offerguide.llm.client import _parse_tool_arguments
        result = _parse_tool_arguments('{"job_id": 5}{}')
        assert result == {"job_id": 5}

    def test_empty_string_returns_empty_dict(self):
        from offerguide.llm.client import _parse_tool_arguments
        assert _parse_tool_arguments("") == {}

    def test_garbage_returns_empty_dict_not_raise(self):
        from offerguide.llm.client import _parse_tool_arguments
        assert _parse_tool_arguments("not json at all") == {}

    def test_nested_dict_preserved(self):
        from offerguide.llm.client import _parse_tool_arguments
        s = '{"a": {"b": 1}, "c": [2, 3]}'
        result = _parse_tool_arguments(s)
        assert result == {"a": {"b": 1}, "c": [2, 3]}


class TestLookupTools:
    """Lookup tools fix the W13-first-dogfood bug: model couldn't pass job_text
    to SKILLs because snapshot only listed jobs by id+title, not raw_text.
    With read_job + read_user_resume, model can fetch full data on demand."""

    def test_lookup_tools_appear_in_schemas(self, populated_store):
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=populated_store,
            skills=skills, master_resume_text="x" * 200,
        )
        names = {sc["function"]["name"] for sc in loop._tool_schemas}
        assert "read_job" in names
        assert "read_user_resume" in names
        # Schema lookup tools come BEFORE SKILL tools (gentle ordering bias)
        assert loop._tool_schemas[0]["function"]["name"] == "read_job"

    def test_read_job_returns_full_raw_text(self, populated_store):
        from offerguide.llm import ToolCall
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=populated_store,
            skills=skills,
        )
        # job#1 was inserted with raw_text = "x" * 800 + company "字节跳动"
        result = loop._execute_tool(ToolCall(
            id="x", name="read_job", arguments={"job_id": 1},
        ))
        assert "字节跳动" in result
        assert "AI Agent" in result  # title
        # Should include the full raw_text (or at least most of it)
        assert "x" * 200 in result  # part of the raw_text body

    def test_read_job_for_missing_id_returns_error(self, populated_store):
        from offerguide.llm import ToolCall
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=populated_store,
            skills=skills,
        )
        result = loop._execute_tool(ToolCall(
            id="x", name="read_job", arguments={"job_id": 99999},
        ))
        assert result.startswith("ERROR")
        assert "not found" in result

    def test_read_user_resume_when_set(self, populated_store):
        from offerguide.llm import ToolCall
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=populated_store,
            skills=skills, master_resume_text="# TestUser\n某高校",
        )
        result = loop._execute_tool(ToolCall(
            id="x", name="read_user_resume", arguments={},
        ))
        assert "TestUser" in result
        assert "某高校" in result

    def test_read_user_resume_when_unset_returns_error(self, populated_store):
        from offerguide.llm import ToolCall
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=populated_store,
            skills=skills,  # no master_resume_text
        )
        result = loop._execute_tool(ToolCall(
            id="x", name="read_user_resume", arguments={},
        ))
        assert result.startswith("ERROR")
        assert "master_resume_text" in result

    def test_read_job_with_invalid_id_type(self, populated_store):
        from offerguide.llm import ToolCall
        skills = discover_skills(SKILLS_ROOT)
        class _SkillStubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
        runtime = SkillRuntime(llm=_SkillStubLLM(), store=populated_store)
        loop = AgentLoop(
            llm=_StubLLMWithTools(), runtime=runtime, store=populated_store,
            skills=skills,
        )
        result = loop._execute_tool(ToolCall(
            id="x", name="read_job", arguments={"job_id": "not_a_number"},
        ))
        assert result.startswith("ERROR")
        assert "job_id" in result


class TestAgentRunResult:
    def test_to_dict_shape(self, empty_store):
        ev = AgentEvent(kind="thinking", at="2026-05-02T00:00:00+00:00", payload={"text": "x"})
        r = AgentRunResult(
            run_id=42, goal="g", trigger_kind="t", final_answer="f",
            events=[ev], iterations=1, cost_usd=0.0, latency_ms=100,
        )
        d = r.to_dict()
        assert d["run_id"] == 42
        assert d["goal"] == "g"
        assert d["trigger_kind"] == "t"
        assert isinstance(d["events"], list)
        assert d["events"][0]["kind"] == "thinking"


# ═══════════════════════════════════════════════════════════════════
# UI routes for /agent
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def agent_app_no_key(tmp_path):
    """App with no API key — agent endpoint should return error frame, not crash."""
    from fastapi.testclient import TestClient

    from offerguide.config import Settings
    from offerguide.ui.notify import ConsoleNotifier
    from offerguide.ui.web import create_app

    store = offerguide.Store(tmp_path / "ui_agent.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)

    # Settings without an API key
    s = Settings(
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        default_model="stub",
    )
    app = create_app(
        settings=s, store=store, profile=None,
        skills=skills, runtime=None, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


@pytest.fixture
def agent_app_with_runtime(tmp_path):
    """App with a SkillRuntime + stub LLM — full agent loop callable."""
    from fastapi.testclient import TestClient

    from offerguide.config import Settings
    from offerguide.ui.notify import ConsoleNotifier
    from offerguide.ui.web import create_app

    store = offerguide.Store(tmp_path / "ui_agent2.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)

    class _SkillStubLLM:
        def chat(self, messages, **kw):
            return LLMResponse(content='{"ok": true}', model="stub")

    runtime = SkillRuntime(llm=_SkillStubLLM(), store=store)
    s = Settings(
        deepseek_api_key="dummy",
        deepseek_base_url="https://api.deepseek.com",
        default_model="stub",
    )
    app = create_app(
        settings=s, store=store, profile=None,
        skills=skills, runtime=runtime, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestAgentUIRoutes:
    def test_agent_page_renders(self, agent_app_no_key):
        client, _ = agent_app_no_key
        resp = client.get("/agent")
        assert resp.status_code == 200
        assert "模型在主位" in resp.text
        # No runtime + no key -> agent should be disabled (W14.6 polish: friendlier msg)
        assert "Agent 还没接通 LLM" in resp.text or "Agent 不可用" in resp.text

    def test_agent_page_includes_recent_runs_section(self, agent_app_with_runtime):
        client, store = agent_app_with_runtime
        # Insert a fake completed run
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, iterations, "
                "                       critic_score, latency_ms, ended_at) "
                "VALUES (?,?,?,?,?,?, julianday('now'))",
                ("test", "test goal x", "ok", 2, 0.85, 1500),
            )
        resp = client.get("/agent")
        assert resp.status_code == 200
        assert "test goal x" in resp.text
        assert "0.85" in resp.text

    def test_agent_run_detail_page(self, agent_app_with_runtime):
        client, store = agent_app_with_runtime
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO agent_runs(trigger_kind, goal, status, iterations, "
                "                       final_answer, trajectory_json, critic_score, "
                "                       critic_notes, latency_ms, ended_at) "
                "VALUES (?,?,?,?,?,?,?,?, ?, julianday('now'))",
                (
                    "user_button", "test detail goal", "ok", 3,
                    "agent done", json.dumps([
                        {"kind": "state_snapshot", "at": "2026-05-02T00:00:00Z",
                         "payload": {"snapshot": "DB empty"}},
                        {"kind": "final", "at": "2026-05-02T00:00:01Z",
                         "payload": {"text": "agent done"}},
                    ]), 0.9, "good", 2200,
                ),
            )
            run_id = cur.lastrowid
        resp = client.get(f"/agent/runs/{run_id}")
        assert resp.status_code == 200
        assert "test detail goal" in resp.text
        assert "agent done" in resp.text
        assert "0.9" in resp.text
        assert "good" in resp.text

    def test_agent_run_detail_404_when_missing(self, agent_app_with_runtime):
        client, _ = agent_app_with_runtime
        resp = client.get("/agent/runs/999999")
        assert resp.status_code == 404

    def test_agent_stream_returns_error_when_no_api_key(self, agent_app_no_key):
        client, _ = agent_app_no_key
        # SSE endpoint with no API key configured should yield one error frame,
        # not crash. We read the streamed body.
        with client.stream("GET", "/api/agent/stream?goal=test") as resp:
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("text/event-stream")
            body = "".join(resp.iter_text())
        assert "OFFERGUIDE_LLM_API_KEY" in body or "error" in body
