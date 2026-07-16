from __future__ import annotations

import threading
import time
from dataclasses import replace

from offerguide.memory import Store
from offerguide.research_agents.coordinator import ResearchAgentCoordinator
from offerguide.research_agents.runner import AgentRunResult, AgentRunStatus


def _result(
    status: AgentRunStatus = AgentRunStatus.PUBLISHED,
    *,
    terminal_output=None,
) -> AgentRunResult:
    return AgentRunResult(
        run_id="domain-run",
        agent_name="JobDiscoveryAgent",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=2,
        starting_result_revision=1,
        status=status,
        iterations=3,
        final_text="",
        terminal_tool="publish_job_selection",
        tool_calls=("search_sources", "publish_job_selection"),
        cost_usd=0.0,
        latency_ms=20,
        reason="terminal_tool_succeeded",
        terminal_output=terminal_output,
    )


def _wait_for_status(
    coordinator: ResearchAgentCoordinator,
    invocation_id: str,
    expected: set[str],
) -> str:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        current = coordinator.get(invocation_id)
        assert current is not None
        if current.status in expected:
            return current.status
        time.sleep(0.01)
    raise AssertionError(f"invocation did not reach {expected}")


def test_invocation_runs_outside_the_request_and_persists_result(tmp_path) -> None:
    store = Store(tmp_path / "coordinator.db")
    store.init_schema()
    coordinator = ResearchAgentCoordinator(store, max_workers=1)
    gate = threading.Event()

    def call() -> AgentRunResult:
        gate.wait(2)
        return _result()

    invocation, created = coordinator.enqueue(
        agent_name="JobDiscoveryAgent",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=2,
        trigger_reason="user requested refresh",
        call=call,
    )
    assert created is True
    assert _wait_for_status(coordinator, invocation.id, {"running"}) == "running"

    gate.set()
    assert _wait_for_status(coordinator, invocation.id, {"published"}) == "published"
    saved = coordinator.get(invocation.id)
    assert saved is not None
    assert saved.domain_run_id == "domain-run"
    coordinator.shutdown(wait=True)


def test_same_subject_revision_reuses_active_invocation(tmp_path) -> None:
    store = Store(tmp_path / "dedup.db")
    store.init_schema()
    coordinator = ResearchAgentCoordinator(store, max_workers=1)
    gate = threading.Event()

    def call() -> AgentRunResult:
        gate.wait(2)
        return _result(AgentRunStatus.UNCHANGED)

    first, created = coordinator.enqueue(
        agent_name="JobDiscoveryAgent",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=2,
        trigger_reason="manual refresh",
        call=call,
    )
    second, created_again = coordinator.enqueue(
        agent_name="JobDiscoveryAgent",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=2,
        trigger_reason="duplicate click",
        call=call,
    )
    assert created is True
    assert created_again is False
    assert second.id == first.id

    gate.set()
    assert _wait_for_status(coordinator, first.id, {"unchanged"}) == "unchanged"
    coordinator.shutdown(wait=True)


def test_blocked_terminal_details_reach_the_persisted_invocation(tmp_path) -> None:
    store = Store(tmp_path / "blocked-details.db")
    store.init_schema()
    coordinator = ResearchAgentCoordinator(store, max_workers=1)

    invocation, created = coordinator.enqueue(
        agent_name="JobDiscoveryAgent",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=2,
        trigger_reason="manual refresh",
        call=lambda: _result(
            AgentRunStatus.BLOCKED,
            terminal_output={
                "reason": "两个岗位页均要求登录",
                "checks_completed": ["打开官方列表", "读取两个详情页"],
                "evidence_gaps": ["缺少可读取的完整 JD"],
            },
        ),
    )

    assert created is True
    assert _wait_for_status(coordinator, invocation.id, {"blocked"}) == "blocked"
    saved = coordinator.get(invocation.id)
    assert saved is not None
    assert "两个岗位页均要求登录" in saved.message
    assert "读取两个详情页" in saved.message
    assert saved.unresolved == ("缺少可读取的完整 JD",)
    coordinator.shutdown(wait=True)


def test_blocked_run_without_terminal_tool_keeps_protocol_failure_reason(tmp_path) -> None:
    store = Store(tmp_path / "blocked-protocol.db")
    store.init_schema()
    coordinator = ResearchAgentCoordinator(store, max_workers=1)

    result = replace(
        _result(AgentRunStatus.BLOCKED),
        terminal_tool=None,
        terminal_output=None,
        reason="model_stopped_without_terminal_tool",
        tool_calls=("search_sources", "read_source"),
        tool_errors=("publish_selection: 引用不在当前来源中",),
    )
    invocation, created = coordinator.enqueue(
        agent_name="JobDiscoveryAgent",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=2,
        trigger_reason="manual refresh",
        call=lambda: result,
    )

    assert created is True
    assert _wait_for_status(coordinator, invocation.id, {"blocked"}) == "blocked"
    saved = coordinator.get(invocation.id)
    assert saved is not None
    assert "未按完成协议结束" in saved.message
    assert "search_sources" in saved.message
    assert "引用不在当前来源中" in saved.message
    assert saved.unresolved == (
        "模型停止前未调用发布、确认无变化或阻塞报告工具",
        "publish_selection: 引用不在当前来源中",
    )
    coordinator.shutdown(wait=True)


def test_new_coordinator_marks_abandoned_process_run_failed(tmp_path) -> None:
    store = Store(tmp_path / "restart.db")
    store.init_schema()
    first = ResearchAgentCoordinator(store)
    first.shutdown(wait=True)
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO research_agent_invocations("
            "id, agent_name, subject_kind, subject_id, subject_revision, "
            "status, trigger_reason"
            ") VALUES ('old', 'InterviewResearchAgent', 'workspace', '4', 1, "
            "'running', 'submit')"
        )

    second = ResearchAgentCoordinator(store)
    abandoned = second.get("old")
    assert abandoned is not None
    assert abandoned.status == "failed"
    assert abandoned.error_text == "owner process is no longer running"
    second.shutdown(wait=True)


def test_second_live_coordinator_does_not_invalidate_or_duplicate_active_run(
    tmp_path,
) -> None:
    store = Store(tmp_path / "two-processes.db")
    store.init_schema()
    first = ResearchAgentCoordinator(store, max_workers=1)
    gate = threading.Event()

    def call() -> AgentRunResult:
        gate.wait(2)
        return _result()

    invocation, created = first.enqueue(
        agent_name="JobDiscoveryAgent",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=2,
        trigger_reason="first process",
        call=call,
    )
    assert created is True
    assert _wait_for_status(first, invocation.id, {"running"}) == "running"

    second = ResearchAgentCoordinator(store, max_workers=1)
    still_running = second.get(invocation.id)
    assert still_running is not None
    assert still_running.status == "running"
    duplicate, duplicate_created = second.enqueue(
        agent_name="JobDiscoveryAgent",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=2,
        trigger_reason="second process",
        call=call,
    )
    assert duplicate_created is False
    assert duplicate.id == invocation.id

    gate.set()
    assert _wait_for_status(second, invocation.id, {"published"}) == "published"
    second.shutdown(wait=True)
    first.shutdown(wait=True)
