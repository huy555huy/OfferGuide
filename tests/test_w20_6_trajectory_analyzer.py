"""W20.6 — Deterministic trajectory analyzer tests.

User's framing (verbatim): "正确且处理好 agent，才是最关键，也最能拿得出手的"

The agent should LEARN from its mistakes WITHOUT another LLM judging it.
Pure Python pattern detection from event stream → agent_self_observations
table → next snapshot reads them → behavior loop closes.

This is the alternative to the (W20.4-killed) critic LLM. It's defensible
because:
- 0 LLM calls (deterministic)
- Detects SPECIFIC bad patterns, not generic 'goodness score'
- Outputs are factual descriptions, not opinions
- Closes a real learning loop (cross-run memory)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import offerguide
from offerguide.agent.loop import AgentEvent, AgentLoop


@pytest.fixture
def loop_with_store(tmp_path):
    store = offerguide.Store(tmp_path / "w206.db")
    store.init_schema()
    loop = AgentLoop(
        llm=MagicMock(), runtime=MagicMock(),
        store=store, skills=[], max_iterations=4,
    )
    return loop, store


def _ev(kind, **payload):
    return AgentEvent(kind=kind, at="2026-05-11T00:00:00Z", payload=payload)


# ── D1: max_iter overshoot ─────────────────────────────────────────


class TestMaxIterDetector:
    def test_max_iter_error_without_final_triggers_overreach(self, loop_with_store):
        loop, _ = loop_with_store
        events = [
            _ev("state_snapshot", snapshot="x"),
            _ev("tool_call", name="read_job", arguments={"job_id": 1}),
            _ev("tool_result", name="read_job", result_preview="OK ..."),
            _ev("error", message="max_iter (4) reached"),
        ]
        patterns = loop._detect_patterns(
            goal="找今日要 score 的新 job",
            events=events,
            final_iteration=4,
            skill_invocations={"x": {"skill_name": "score_match"}},
        )
        kinds = [p[0] for p in patterns]
        texts = [p[1] for p in patterns]
        assert "overreach" in kinds
        assert any("max_iter" in t and "早点判断收手" in t for t in texts)

    def test_max_iter_message_but_with_final_doesnt_trigger(self, loop_with_store):
        """If agent did reach a final answer, max_iter mention isn't an
        overreach pattern (might be benign mid-loop msg)."""
        loop, _ = loop_with_store
        events = [
            _ev("error", message="max_iter (4) reached"),  # weird, but
            _ev("final", text="done", iteration=4),
        ]
        patterns = loop._detect_patterns(
            goal="g", events=events, final_iteration=4, skill_invocations={},
        )
        assert not any(p[0] == "overreach" for p in patterns)


# ── D2: circuit-breaker / repeated same (tool, args) ───────────────


class TestRepeatedToolArgsDetector:
    def test_same_tool_same_args_3x_triggers(self, loop_with_store):
        loop, _ = loop_with_store
        events = []
        for _ in range(3):
            events.append(_ev(
                "tool_call", name="read_job",
                arguments={"job_id": 5},
            ))
            events.append(_ev(
                "tool_result", name="read_job",
                result_preview="OK ...",
            ))
        patterns = loop._detect_patterns(
            goal="g", events=events, final_iteration=3, skill_invocations={},
        )
        rm = [p for p in patterns if p[0] == "repeated_mistake"]
        assert rm
        assert any("read_job" in p[1] and "3 次" in p[1] for p in rm)

    def test_same_tool_different_args_doesnt_trigger(self, loop_with_store):
        """Calling read_job(1), read_job(2), read_job(3) is healthy iteration."""
        loop, _ = loop_with_store
        events = [
            _ev("tool_call", name="read_job", arguments={"job_id": 1}),
            _ev("tool_result", name="read_job", result_preview="OK"),
            _ev("tool_call", name="read_job", arguments={"job_id": 2}),
            _ev("tool_result", name="read_job", result_preview="OK"),
            _ev("tool_call", name="read_job", arguments={"job_id": 3}),
            _ev("tool_result", name="read_job", result_preview="OK"),
        ]
        patterns = loop._detect_patterns(
            goal="g", events=events, final_iteration=3, skill_invocations={},
        )
        # No repeated-same-args mistake; could still emit other patterns
        # but at least not THIS one
        repeated_msgs = [p[1] for p in patterns if p[0] == "repeated_mistake"]
        # Make sure no "同一组 args" message about read_job specifically
        assert not any("read_job 同一组 args" in m for m in repeated_msgs)

    def test_2x_repeat_doesnt_trigger_under_threshold(self, loop_with_store):
        loop, _ = loop_with_store
        events = [
            _ev("tool_call", name="x", arguments={"a": 1}),
            _ev("tool_result", name="x", result_preview="OK"),
            _ev("tool_call", name="x", arguments={"a": 1}),
            _ev("tool_result", name="x", result_preview="OK"),
        ]
        patterns = loop._detect_patterns(
            goal="g", events=events, final_iteration=2, skill_invocations={},
        )
        assert not any(p[0] == "repeated_mistake" and "同一组 args" in p[1]
                       for p in patterns)


# ── D3: tool returned ERROR ≥2 times ───────────────────────────────


class TestRepeatedToolErrorDetector:
    def test_tool_error_twice_triggers(self, loop_with_store):
        loop, _ = loop_with_store
        events = [
            _ev("tool_call", name="score_match", arguments={"job_id": 1}),
            _ev("tool_result", name="score_match",
                result_preview="ERROR: invalid inputs for score_match: missing job_text"),
            _ev("tool_call", name="score_match", arguments={"job_id": 2}),
            _ev("tool_result", name="score_match",
                result_preview="ERROR: invalid inputs for score_match: missing job_text"),
        ]
        patterns = loop._detect_patterns(
            goal="g", events=events, final_iteration=2,
            skill_invocations={},
        )
        msgs = [p[1] for p in patterns if p[0] == "repeated_mistake"]
        assert any("score_match" in m and "ERROR" in m for m in msgs)

    def test_tool_success_once_then_error_doesnt_trigger(self, loop_with_store):
        loop, _ = loop_with_store
        events = [
            _ev("tool_call", name="score_match", arguments={"job_id": 1}),
            _ev("tool_result", name="score_match", result_preview="OK ..."),
            _ev("tool_call", name="score_match", arguments={"job_id": 2}),
            _ev("tool_result", name="score_match", result_preview="ERROR: bad"),
        ]
        patterns = loop._detect_patterns(
            goal="g", events=events, final_iteration=2, skill_invocations={},
        )
        # 1 error not 2 → no detector fires
        assert not any(
            p[0] == "repeated_mistake" and "score_match" in p[1] and "ERROR" in p[1]
            for p in patterns
        )


# ── D4: zero-action on action-keyword goal ─────────────────────────


class TestZeroActionDetector:
    def test_action_goal_no_tools_triggers_underreach(self, loop_with_store):
        loop, _ = loop_with_store
        events = [
            _ev("state_snapshot", snapshot="empty"),
            _ev("final", text="无可做", iteration=1),
        ]
        patterns = loop._detect_patterns(
            goal="评估这个 JD 是否值得投",
            events=events, final_iteration=1, skill_invocations={},
        )
        msgs = [p[1] for p in patterns if p[0] == "underreach"]
        assert msgs
        assert any("0 tool_call" in m or "空判断" in m for m in msgs)

    def test_non_action_goal_no_tools_doesnt_trigger(self, loop_with_store):
        """Goals like '今日例行检查' may legitimately need no action."""
        loop, _ = loop_with_store
        events = [
            _ev("state_snapshot", snapshot="empty"),
            _ev("final", text="今日无紧急事项", iteration=1),
        ]
        patterns = loop._detect_patterns(
            goal="今日例行检查", events=events,
            final_iteration=1, skill_invocations={},
        )
        # No action keywords in goal → no underreach pattern
        assert not any(p[0] == "underreach" for p in patterns)

    def test_action_goal_with_tools_doesnt_trigger(self, loop_with_store):
        loop, _ = loop_with_store
        events = [
            _ev("tool_call", name="read_job", arguments={"job_id": 1}),
            _ev("tool_result", name="read_job", result_preview="OK"),
            _ev("final", text="ok", iteration=2),
        ]
        patterns = loop._detect_patterns(
            goal="评估 job#1", events=events,
            final_iteration=2, skill_invocations={},
        )
        # Tools were called → not zero-action
        assert not any(p[0] == "underreach" for p in patterns)


# ── Dedup + persistence ─────────────────────────────────────────────


class TestPersistenceAndDedup:
    def test_persist_writes_to_agent_self_observations(self, loop_with_store):
        loop, store = loop_with_store
        # Trigger D1 via simple events
        events = [
            _ev("tool_call", name="x", arguments={}),
            _ev("error", message="max_iter (4) reached"),
        ]
        persisted = loop._analyze_and_persist_observations(
            run_id=1, goal="some action goal", events=events,
            final_iteration=4, skill_invocations={"a": {}},
        )
        assert any(p[0] == "overreach" for p in persisted)

        with store.connect() as conn:
            rows = conn.execute(
                "SELECT pattern_kind, observation, evidence_json, valid_until "
                "FROM agent_self_observations"
            ).fetchall()
        assert len(rows) >= 1
        # valid_until populated (14-day TTL)
        assert all(r[3] is not None for r in rows)

    def test_dedup_skips_recent_same_observation(self, loop_with_store):
        """Same exact text + pattern_kind written twice in a row should
        persist once."""
        loop, store = loop_with_store
        events = [
            _ev("tool_call", name="x", arguments={}),
            _ev("error", message="max_iter (4) reached"),
        ]
        # Round 1
        first = loop._analyze_and_persist_observations(
            run_id=1, goal="g action", events=events,
            final_iteration=4, skill_invocations={"a": {}},
        )
        # Round 2 with identical events
        second = loop._analyze_and_persist_observations(
            run_id=2, goal="g action", events=events,
            final_iteration=4, skill_invocations={"a": {}},
        )
        # The overreach pattern should appear in first, NOT in second
        assert any(p[0] == "overreach" for p in first)
        assert not any(p[0] == "overreach" for p in second)

        with store.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM agent_self_observations "
                "WHERE pattern_kind = 'overreach'"
            ).fetchone()[0]
        assert count == 1

    def test_dedup_allows_different_observation_text(self, loop_with_store):
        """If trajectory is different enough to produce a different observation
        text, both should be persisted."""
        loop, store = loop_with_store
        # First trajectory: max_iter overshoot
        events1 = [_ev("error", message="max_iter (4) reached")]
        loop._analyze_and_persist_observations(
            run_id=1, goal="g action", events=events1,
            final_iteration=4, skill_invocations={"a": {}},
        )
        # Second: a repeated_mistake (different observation text)
        events2 = [
            _ev("tool_call", name="badtool", arguments={"k": 1}),
            _ev("tool_call", name="badtool", arguments={"k": 1}),
            _ev("tool_call", name="badtool", arguments={"k": 1}),
            _ev("final", text="ok"),
        ]
        loop._analyze_and_persist_observations(
            run_id=2, goal="g", events=events2,
            final_iteration=3, skill_invocations={"a": {}},
        )

        with store.connect() as conn:
            rows = conn.execute(
                "SELECT pattern_kind FROM agent_self_observations"
            ).fetchall()
        kinds = {r[0] for r in rows}
        assert "overreach" in kinds
        assert "repeated_mistake" in kinds


# ── Integration: run loop wires analyzer correctly ──────────────────


class TestRunLoopIntegration:
    """Verify that run() actually calls _analyze_and_persist_observations
    and skips it for trivial runs (≤1 iter + 0 SKILL)."""

    def test_trivial_run_skips_analyzer(self, loop_with_store):
        """If iterations ≤ 1 AND no SKILL invoked, analyzer should not fire."""
        loop, store = loop_with_store
        # Track if analyzer was called
        analyzer_calls = []
        original = loop._analyze_and_persist_observations

        def _spy(**kwargs):
            analyzer_calls.append(kwargs)
            return original(**kwargs)

        loop._analyze_and_persist_observations = _spy  # type: ignore[assignment]

        # Stub LLM to immediately return a final without tool calls
        from offerguide.llm import LLMResponse

        class _StubLLM:
            def chat_with_tools(self, **kw):
                return LLMResponse(
                    content="无事可做", model="stub",
                    tool_calls=[],  # no tool calls
                )

        loop._llm = _StubLLM()  # type: ignore[assignment]

        result = loop.run(goal="trivial check", trigger_kind="test")

        # Run should succeed but analyzer never called (trivial gate)
        assert result.error is None
        assert analyzer_calls == []

    def test_non_trivial_run_invokes_analyzer(self, loop_with_store):
        """If iterations > 1 OR SKILL invoked, analyzer should fire."""
        loop, store = loop_with_store
        analyzer_calls = []
        original = loop._analyze_and_persist_observations

        def _spy(**kwargs):
            analyzer_calls.append(kwargs)
            return original(**kwargs)

        loop._analyze_and_persist_observations = _spy  # type: ignore[assignment]

        # Stub: model calls a fake lookup tool, then finals
        from offerguide.llm import LLMResponse, ToolCall

        class _StubLLM:
            def __init__(self):
                self._n = 0
            def chat_with_tools(self, **kw):
                self._n += 1
                if self._n == 1:
                    return LLMResponse(
                        content="looking",
                        model="stub",
                        tool_calls=[ToolCall(
                            id="t1", name="read_user_resume", arguments={},
                        )],
                    )
                return LLMResponse(
                    content="done", model="stub", tool_calls=[],
                )

        loop._llm = _StubLLM()  # type: ignore[assignment]

        result = loop.run(goal="评估今日", trigger_kind="test")
        assert result.error is None
        # Analyzer must have been called once
        assert len(analyzer_calls) == 1
