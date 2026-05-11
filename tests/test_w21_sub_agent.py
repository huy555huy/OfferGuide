"""W21 Phase 2 — SubAgent base class tests.

Sub-agent = bounded ReAct loop scoped to one tool group. Tests:
- runs to natural completion when LLM calls done()
- runs to max_iter when LLM never terminates
- circuit breaker fires on repeated same-args
- tool dispatch passes through registry
- cost accumulated correctly
- timeout enforced
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

import offerguide
from offerguide.agents.base import SubAgent, SubAgentResult, register_universal_tools
from offerguide.llm.client import LLMResponse, ToolCall
from offerguide.tools.registry import ToolRegistry, tool_result


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "sub.db")
    s.init_schema()
    return s


@pytest.fixture
def reg():
    r = ToolRegistry()
    register_universal_tools(r)
    return r


def _stub_llm(*responses: LLMResponse):
    """Mock LLM client that returns successive responses."""
    mock = MagicMock()
    mock.chat_with_tools = MagicMock(side_effect=list(responses))
    return mock


def _final_resp(text: str = "done", cost: float = 0.001) -> LLMResponse:
    return LLMResponse(content=text, model="stub", tool_calls=[], cost_usd=cost)


def _tool_resp(
    name: str, args: dict, cost: float = 0.001, content: str = "thinking",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        cost_usd=cost,
        tool_calls=[ToolCall(
            id=f"tc_{name}", name=name, arguments=args,
            arguments_raw=json.dumps(args),
        )],
    )


class _TestSubAgent(SubAgent):
    SYSTEM_PROMPT = "test sub-agent"
    GROUP = "shared"
    NAME = "test_sub"


# ── Natural completion via done() ─────────────────────────────────


def test_sub_agent_done_call_terminates_cleanly(store, reg):
    llm = _stub_llm(
        _tool_resp("done", {"summary": "已完成 X"}, cost=0.001),
    )
    sub = _TestSubAgent(llm=llm, registry=reg, store=store, max_iter=4)
    result = sub.run(goal="跑个简单的事")
    assert result.final_answer == "已完成 X"
    assert result.iterations == 1
    assert result.tool_calls_made == 1
    assert result.cost_usd == pytest.approx(0.001)
    assert result.error is None
    kinds = [e.kind for e in result.events]
    assert "tool_call" in kinds
    assert "final" in kinds


def test_sub_agent_natural_stop_no_tool_call(store, reg):
    """LLM returns content but no tool_call → treat as graceful stop."""
    llm = _stub_llm(
        _final_resp("我想了想, 数据太少, 没事做.", cost=0.0005),
    )
    sub = _TestSubAgent(llm=llm, registry=reg, store=store, max_iter=4)
    result = sub.run(goal="x")
    assert "数据太少" in result.final_answer
    assert result.tool_calls_made == 0
    assert result.iterations == 1


# ── max_iter without final ───────────────────────────────────────


def test_sub_agent_max_iter_returns_error(store, reg):
    """LLM keeps calling tools without done() → loop terminates at cap."""
    def _h(args, **kw):
        return tool_result(ok=True, iter=time.monotonic())
    reg.register(
        name="endless", group="shared",
        schema={"name": "endless", "parameters": {"type": "object", "properties": {}}},
        handler=_h,
    )
    # 4 tool_resp + would-be 5th never reached
    llm = _stub_llm(
        _tool_resp("endless", {"i": 1}, cost=0.001),
        _tool_resp("endless", {"i": 2}, cost=0.001),
        _tool_resp("endless", {"i": 3}, cost=0.001),
        _tool_resp("endless", {"i": 4}, cost=0.001),
    )
    sub = _TestSubAgent(llm=llm, registry=reg, store=store, max_iter=4)
    result = sub.run(goal="x")
    assert "max_iter" in result.final_answer
    assert result.iterations == 4
    assert any(e.kind == "error" for e in result.events)


# ── Circuit breaker ──────────────────────────────────────────────


def test_circuit_breaker_aborts_on_repeated_same_args(store, reg):
    """Same (tool, args) repeated 3x → abort with breaker message."""
    def _h(args, **kw): return tool_result(stuck=True)
    reg.register(
        name="stuck", group="shared",
        schema={"name": "stuck", "parameters": {"type": "object", "properties": {}}},
        handler=_h,
    )
    # 3 identical calls
    llm = _stub_llm(
        _tool_resp("stuck", {"k": 1}),
        _tool_resp("stuck", {"k": 1}),
        _tool_resp("stuck", {"k": 1}),
    )
    sub = _TestSubAgent(llm=llm, registry=reg, store=store, max_iter=10)
    result = sub.run(goal="x")
    assert "circuit breaker" in result.final_answer.lower() or "3x" in result.final_answer
    # Aborts inside iter 3, before iter 4
    assert result.iterations <= 3


def test_circuit_breaker_does_not_fire_on_different_args(store, reg):
    def _h(args, **kw): return tool_result(ok=True)
    reg.register(
        name="varies", group="shared",
        schema={"name": "varies", "parameters": {"type": "object", "properties": {}}},
        handler=_h,
    )
    # Same tool, DIFFERENT args, then done
    llm = _stub_llm(
        _tool_resp("varies", {"k": 1}),
        _tool_resp("varies", {"k": 2}),
        _tool_resp("varies", {"k": 3}),
        _tool_resp("done", {"summary": "ok"}),
    )
    sub = _TestSubAgent(llm=llm, registry=reg, store=store, max_iter=5)
    result = sub.run(goal="x")
    # No breaker; cleanly terminates via done
    assert result.error is None
    assert "circuit breaker" not in result.final_answer.lower()


# ── Tool dispatch through registry ───────────────────────────────


def test_sub_agent_dispatches_via_registry_with_runtime_kwargs(store, reg):
    captured = {}
    def _h(args, *, store=None, **kw):
        captured["got_store"] = store is not None
        captured["got_args"] = args
        return tool_result(seen=True)
    reg.register(
        name="needs_store", group="shared",
        schema={"name": "needs_store", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}},
        handler=_h,
    )
    llm = _stub_llm(
        _tool_resp("needs_store", {"x": 42}),
        _tool_resp("done", {"summary": "ok"}),
    )
    sub = _TestSubAgent(llm=llm, registry=reg, store=store, max_iter=4)
    result = sub.run(goal="x")
    assert result.error is None
    assert captured.get("got_store") is True
    assert captured.get("got_args") == {"x": 42}


def test_sub_agent_handles_unknown_tool_gracefully(store, reg):
    """LLM hallucinates a tool name — should get error JSON, not crash."""
    llm = _stub_llm(
        _tool_resp("nonexistent_tool", {"x": 1}),
        _tool_resp("done", {"summary": "recovered"}),
    )
    sub = _TestSubAgent(llm=llm, registry=reg, store=store, max_iter=4)
    result = sub.run(goal="x")
    assert result.error is None
    # The unknown tool's result_preview should mention error
    tool_results = [e for e in result.events if e.kind == "tool_result"]
    assert any("error" in e.payload["result_preview"].lower() for e in tool_results)


# ── Tool group scoping ───────────────────────────────────────────


def test_sub_agent_only_sees_own_group_plus_shared(store):
    """Discovery sub-agent shouldn't see evaluation tools."""
    r = ToolRegistry()
    register_universal_tools(r)
    def _h(args, **kw): return "{}"
    r.register(
        name="fetch_x", group="discovery",
        schema={"name": "fetch_x", "parameters": {"type": "object", "properties": {}}},
        handler=_h,
    )
    r.register(
        name="score_x", group="evaluation",
        schema={"name": "score_x", "parameters": {"type": "object", "properties": {}}},
        handler=_h,
    )

    class _Discovery(SubAgent):
        GROUP = "discovery"
        NAME = "disc"

    # Capture which tool schemas the LLM is given
    captured = {}
    def _spy_chat(messages, tools, temperature):
        captured["tools"] = tools
        return _final_resp("ok")
    llm = MagicMock()
    llm.chat_with_tools = _spy_chat
    sub = _Discovery(llm=llm, registry=r, store=store, max_iter=2)
    sub.run(goal="x")
    tool_names = {t["function"]["name"] for t in captured["tools"]}
    assert "fetch_x" in tool_names
    assert "done" in tool_names
    assert "score_x" not in tool_names


# ── Cost accumulation ────────────────────────────────────────────


def test_sub_agent_accumulates_cost_across_iterations(store, reg):
    def _h(args, **kw): return tool_result(ok=True)
    reg.register(
        name="cheap", group="shared",
        schema={"name": "cheap", "parameters": {"type": "object", "properties": {}}},
        handler=_h,
    )
    llm = _stub_llm(
        _tool_resp("cheap", {"k": 1}, cost=0.001),
        _tool_resp("cheap", {"k": 2}, cost=0.002),
        _tool_resp("done", {"summary": "ok"}, cost=0.0005),
    )
    sub = _TestSubAgent(llm=llm, registry=reg, store=store, max_iter=5)
    result = sub.run(goal="x")
    assert result.cost_usd == pytest.approx(0.0035)


# ── SubAgentResult.summary() shape ───────────────────────────────


def test_result_summary_for_main_agent(store, reg):
    llm = _stub_llm(_tool_resp("done", {"summary": "找到 3 个岗"}, cost=0.002))
    sub = _TestSubAgent(llm=llm, registry=reg, store=store)
    result = sub.run(goal="x")
    s = result.summary()
    assert s["final_answer"] == "找到 3 个岗"
    assert s["cost_usd"] == 0.002
    assert "iterations" in s
    assert "duration_ms" in s
