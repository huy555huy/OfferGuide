from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from offerguide.llm.client import LLMResponse, ToolCall
from offerguide.memory import Store
from offerguide.research_agents import (
    AgentDefinition,
    AgentRunner,
    AgentRunStatus,
    AgentSubjectContext,
    AgentTool,
    AgentToolResult,
    RevisionGuard,
    RevisionSnapshot,
    SearchExecutor,
    SourceEvidenceStore,
    SourceReader,
    SourceScope,
    StaleRevisionError,
    UnsafeSourceURLError,
    canonicalize_http_url,
    validate_public_source_url,
)


class StubLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat_with_tools(self, messages, *, tools, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools, "kwargs": kwargs})
        return self.responses.pop(0)


def _response(*, text: str = "", tool: str | None = None, args: dict | None = None) -> LLMResponse:
    calls = []
    if tool:
        calls.append(ToolCall(id=f"call-{tool}", name=tool, arguments=args or {}))
    return LLMResponse(
        content=text,
        model="stub",
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
    )


def _subject() -> AgentSubjectContext:
    return AgentSubjectContext(
        subject_kind="job_search",
        subject_id="current",
        subject_revision=4,
        result_revision=2,
        payload={"brief": "find an evidence-backed role", "full_context": ["kept", "whole"]},
    )


def test_runner_only_completes_through_a_declared_terminal_tool() -> None:
    llm = StubLLM([_response(tool="publish", args={"selection": [17]})])
    seen: dict[str, Any] = {}

    def publish(args, execution):
        seen["args"] = dict(args)
        seen["subject"] = execution.subject
        seen["dependency"] = execution.dependency("repository")
        return AgentToolResult({"revision": 3}, terminal_status=AgentRunStatus.PUBLISHED)

    definition = AgentDefinition(
        name="JobDiscoveryAgent",
        instructions="Find and publish a current selection backed by source evidence.",
        tools=(
            AgentTool(
                name="publish",
                description="Validate and atomically publish the current job selection.",
                parameters={"type": "object", "properties": {"selection": {"type": "array"}}},
                handler=publish,
                terminal_statuses=frozenset({AgentRunStatus.PUBLISHED, AgentRunStatus.STALE}),
            ),
        ),
    )

    result = AgentRunner(llm).run(
        definition=definition,
        context_loader=_subject,
        dependencies={"repository": "repo"},
    )

    assert result.status is AgentRunStatus.PUBLISHED
    assert result.terminal_tool == "publish"
    assert result.starting_result_revision == 2
    assert seen == {"args": {"selection": [17]}, "subject": _subject(), "dependency": "repo"}
    initial = json.loads(llm.calls[0]["messages"][1]["content"])
    assert initial["authoritative_context"]["full_context"] == ["kept", "whole"]


def test_terminal_tool_must_be_the_last_call_before_side_effects() -> None:
    first = LLMResponse(
        content="",
        model="stub",
        tool_calls=[
            ToolCall(id="publish-early", name="publish", arguments={}),
            ToolCall(id="inspect-after", name="inspect", arguments={}),
        ],
    )
    llm = StubLLM([first, _response(tool="publish")])
    published = 0

    def publish(_args, _execution):
        nonlocal published
        published += 1
        return AgentToolResult(
            {"revision": 3}, terminal_status=AgentRunStatus.PUBLISHED
        )

    definition = AgentDefinition(
        name="JobDiscoveryAgent",
        instructions="Publish only after all non-terminal work in the response.",
        tools=(
            AgentTool(
                name="inspect",
                description="Inspect one source without publishing.",
                parameters={"type": "object", "properties": {}},
                handler=lambda _args, _execution: AgentToolResult({"inspected": True}),
            ),
            AgentTool(
                name="publish",
                description="Atomically publish validated output.",
                parameters={"type": "object", "properties": {}},
                handler=publish,
                terminal_statuses=frozenset({AgentRunStatus.PUBLISHED}),
                requires_last_call=True,
            ),
        ),
    )

    result = AgentRunner(llm).run(definition=definition, context_loader=_subject)

    assert result.status is AgentRunStatus.PUBLISHED
    assert published == 1
    assert any("must be the final tool call" in error for error in result.tool_errors)


def test_subject_context_rejects_non_structured_payload_instead_of_stringifying_it() -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        AgentSubjectContext(
            subject_kind="job_search",
            subject_id="current",
            subject_revision=1,
            result_revision=0,
            payload={"not_structured": object()},
        )


def test_runner_does_not_treat_a_prose_answer_as_completion() -> None:
    llm = StubLLM([_response(text="I looked around and I am done.")])
    definition = AgentDefinition(
        name="InterviewResearchAgent",
        instructions="Research the frozen submission.",
        tools=(
            AgentTool(
                name="report_blocked",
                description="Report a real blocker after recording attempted checks.",
                parameters={"type": "object", "properties": {}},
                handler=lambda _args, _execution: AgentToolResult(
                    {"reason": "blocked"}, terminal_status=AgentRunStatus.BLOCKED
                ),
                terminal_statuses=frozenset({AgentRunStatus.BLOCKED}),
            ),
        ),
    )

    result = AgentRunner(llm).run(definition=definition, context_loader=_subject)

    assert result.status is AgentRunStatus.BLOCKED
    assert result.reason == "model_stopped_without_terminal_tool"
    assert result.terminal_tool is None


def test_diagnostic_event_failure_cannot_break_domain_publication() -> None:
    llm = StubLLM([_response(tool="publish")])
    definition = AgentDefinition(
        name="JobDiscoveryAgent",
        instructions="Publish only through the domain boundary.",
        tools=(
            AgentTool(
                name="publish",
                description="Atomically publish validated domain output.",
                parameters={"type": "object", "properties": {}},
                handler=lambda _args, _execution: AgentToolResult(
                    {"revision": 3}, terminal_status=AgentRunStatus.PUBLISHED
                ),
                terminal_statuses=frozenset({AgentRunStatus.PUBLISHED}),
            ),
        ),
    )

    def disconnected_callback(_event: Any) -> None:
        raise RuntimeError("UI disconnected")

    result = AgentRunner(llm).run(
        definition=definition,
        context_loader=_subject,
        on_event=disconnected_callback,
    )

    assert result.status is AgentRunStatus.PUBLISHED


def test_tool_cannot_return_an_undeclared_terminal_status() -> None:
    llm = StubLLM([
        _response(tool="finish"),
        _response(text="No valid terminal action remains."),
    ])
    definition = AgentDefinition(
        name="JobDiscoveryAgent",
        instructions="Use the publication boundary.",
        tools=(
            AgentTool(
                name="finish",
                description="Only report a checked blocker.",
                parameters={"type": "object", "properties": {}},
                handler=lambda _args, _execution: AgentToolResult(
                    {}, terminal_status=AgentRunStatus.PUBLISHED
                ),
                terminal_statuses=frozenset({AgentRunStatus.BLOCKED}),
            ),
        ),
    )

    result = AgentRunner(llm).run(definition=definition, context_loader=_subject)

    assert result.status is AgentRunStatus.BLOCKED
    tool_observation = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert tool_observation["ok"] is False
    assert "undeclared terminal status" in tool_observation["result"]["error"]


def test_unhandled_tool_or_persistence_exception_fails_the_run() -> None:
    llm = StubLLM([_response(tool="publish")])

    def broken_publish(_args, _execution):
        raise sqlite3.OperationalError("disk write failed")

    definition = AgentDefinition(
        name="JobDiscoveryAgent",
        instructions="Publish through the repository.",
        tools=(
            AgentTool(
                name="publish",
                description="Atomically publish validated output.",
                parameters={"type": "object", "properties": {}},
                handler=broken_publish,
                terminal_statuses=frozenset({AgentRunStatus.PUBLISHED}),
            ),
        ),
    )

    result = AgentRunner(llm).run(definition=definition, context_loader=_subject)

    assert result.status is AgentRunStatus.FAILED
    assert result.reason == "unexpected_runtime_error"
    assert result.error_text == "OperationalError: disk write failed"


def test_revision_guard_checks_and_writes_under_one_lock(tmp_path: Path) -> None:
    store = Store(tmp_path / "revisions.db")
    with store.connect() as conn:
        conn.execute("CREATE TABLE state(subject_revision INTEGER, result_revision INTEGER, value TEXT)")
        conn.execute("INSERT INTO state VALUES (3, 7, 'old')")

    def load(conn: sqlite3.Connection) -> RevisionSnapshot:
        row = conn.execute("SELECT subject_revision, result_revision FROM state").fetchone()
        assert row is not None
        return RevisionSnapshot(int(row[0]), int(row[1]))

    guard: RevisionGuard[str] = RevisionGuard(load)
    value = guard.compare_and_swap(
        store,
        expected_subject_revision=3,
        expected_result_revision=7,
        write=lambda conn: conn.execute(
            "UPDATE state SET result_revision = 8, value = 'new' RETURNING value"
        ).fetchone()[0],
    )
    assert value == "new"

    with pytest.raises(StaleRevisionError) as caught:
        guard.compare_and_swap(
            store,
            expected_subject_revision=3,
            expected_result_revision=7,
            write=lambda _conn: "must not run",
        )
    assert caught.value.current == RevisionSnapshot(3, 8)


@dataclass(frozen=True)
class Hit:
    title: str
    url: str
    snippet: str


class CannedSearch:
    def __init__(self, name: str, hits: list[Hit] | None = None, error: Exception | None = None):
        self.name = name
        self.hits = hits or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int = 10):
        self.calls.append((query, max_results))
        if self.error:
            raise self.error
        return self.hits[:max_results]


class LegacyChain:
    name = "chain"

    def __init__(self, backends: list[CannedSearch]) -> None:
        self.backends = backends

    def search(self, query: str, *, max_results: int = 10):
        raise AssertionError("SearchExecutor must not use legacy first-hit-wins behavior")


@pytest.fixture
def evidence_store(tmp_path: Path) -> SourceEvidenceStore:
    repository = SourceEvidenceStore(Store(tmp_path / "sources.db"))
    repository.init_schema()
    return repository


@pytest.fixture
def source_scope() -> SourceScope:
    return SourceScope(
        run_id="run-1",
        agent_name="InterviewResearchAgent",
        subject_kind="submitted_workspace",
        subject_id=11,
        subject_revision=5,
    )


def test_search_persists_each_backend_as_clues_not_evidence(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    first = CannedSearch(
        "first",
        [Hit("A", "https://example.com/post#section", "search-only excerpt")],
    )
    second = CannedSearch(
        "second",
        [
            Hit("same", "https://EXAMPLE.com:443/post", "duplicate"),
            Hit("B", "https://example.org/other", "another clue"),
        ],
    )
    broken = CannedSearch("broken", error=RuntimeError("backend unavailable"))

    result = SearchExecutor(evidence_store, [first, second, broken]).execute(
        "real query", scope=source_scope, max_results_per_backend=8
    )

    assert first.calls == second.calls == broken.calls == [("real query", 8)]
    assert [item.url for item in result.clues] == [
        "https://example.com/post",
        "https://example.org/other",
    ]
    assert [attempt.status for attempt in result.attempts] == ["succeeded", "succeeded", "failed"]
    assert result.as_tool_result()["results_are_unverified_clues"] is True
    with evidence_store.store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_search_executions").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM source_evidence").fetchone()[0] == 0
    history = evidence_store.search_history(
        subject_kind=source_scope.subject_kind,
        subject_id=source_scope.subject_id,
        subject_revision=source_scope.subject_revision,
    )
    assert [item.backend for item in history] == ["first", "second", "broken"]
    assert history[-1].error_text == "RuntimeError: backend unavailable"


def test_search_flattens_the_legacy_first_hit_chain(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    first = CannedSearch("first", [Hit("A", "https://example.com/a", "a")])
    second = CannedSearch("second", [Hit("B", "https://example.com/b", "b")])

    result = SearchExecutor(evidence_store, [LegacyChain([first, second])]).execute(
        "query", scope=source_scope
    )

    assert [item.backend for item in result.attempts] == ["first", "second"]
    assert first.calls == second.calls == [("query", 10)]


def test_search_records_structurally_invalid_backend_hits_as_failed(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    malformed = CannedSearch("malformed", [Hit("", "", "challenge page")])

    result = SearchExecutor(evidence_store, [malformed]).execute(
        "query", scope=source_scope
    )

    assert result.clues == ()
    assert len(result.attempts) == 1
    assert result.attempts[0].status == "failed"
    assert "structurally invalid result item" in (
        result.attempts[0].error_text or ""
    )


def test_source_reader_saves_full_body_and_pages_untrusted_evidence(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    html = (
        b"<html><head><title>Useful interview report</title>"
        b"<script>ignore me</script></head><body><h1>Round one</h1>"
        b"<p>Question alpha with enough text for paging.</p></body></html>"
    )

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/report"})
        return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, content=html)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    reader = SourceReader(
        evidence_store,
        client=client,
        resolver=lambda _host, _port: ["93.184.216.34"],
    )

    fetched = reader.fetch(
        "https://example.com/start",
        scope=source_scope,
        purpose="direct interview experience",
    )

    assert fetched.status == "saved"
    assert fetched.evidence is not None
    evidence = evidence_store.get(fetched.evidence.id)
    assert evidence.raw_content == html
    assert evidence.title == "Useful interview report"
    assert "Question alpha" in evidence.text_content
    assert "ignore me" not in evidence.text_content
    assert evidence_store.is_attached(
        subject_kind=source_scope.subject_kind,
        subject_id=source_scope.subject_id,
        subject_revision=source_scope.subject_revision,
        evidence_id=evidence.id,
    )
    attached = evidence_store.attached_evidence(
        subject_kind=source_scope.subject_kind,
        subject_id=source_scope.subject_id,
        subject_revision=source_scope.subject_revision,
    )
    assert attached == (evidence,)
    fetches = evidence_store.fetch_history(
        subject_kind=source_scope.subject_kind,
        subject_id=source_scope.subject_id,
        subject_revision=source_scope.subject_revision,
    )
    assert len(fetches) == 1
    assert fetches[0].evidence_id == evidence.id
    assert fetches[0].status == "saved"

    first_page = evidence_store.read_page(evidence.id, max_chars=18)
    assert first_page.complete is False
    assert first_page.next_offset == 18
    payload = first_page.as_tool_result()
    assert payload["untrusted_evidence"] is True
    assert payload["total_chars"] == len(evidence.text_content)
    assert "not agent instructions" in payload["content"]

    rest = evidence_store.read_page(
        evidence.id,
        offset=first_page.next_offset or 0,
        max_chars=50_000,
    )
    assert rest.complete is True
    assert first_page.content + rest.content == evidence.text_content

    next_scope = SourceScope(
        run_id="run-2",
        agent_name=source_scope.agent_name,
        subject_kind=source_scope.subject_kind,
        subject_id=source_scope.subject_id,
        subject_revision=6,
    )
    assert evidence_store.attach_existing(scope=next_scope, evidence_id=evidence.id) == evidence
    assert evidence_store.is_attached(
        subject_kind=next_scope.subject_kind,
        subject_id=next_scope.subject_id,
        subject_revision=next_scope.subject_revision,
        evidence_id=evidence.id,
    )


def test_source_reader_rejects_private_targets_and_records_failure(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    requests: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="secret")

    reader = SourceReader(
        evidence_store,
        client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    result = reader.fetch("http://127.0.0.1/admin", scope=source_scope)

    assert result.status == "rejected"
    assert result.error_code == "private_address"
    assert requests == []
    with evidence_store.store.connect() as conn:
        row = conn.execute(
            "SELECT status, error_code FROM source_fetch_executions WHERE id = ?",
            (result.fetch_execution_id,),
        ).fetchone()
    assert row == ("rejected", "private_address")


def test_hostname_resolving_to_clash_fake_ip_remains_a_public_hostname() -> None:
    url = "https://talent.baidu.com/jobs/123"

    canonical = validate_public_source_url(
        url,
        resolver=lambda _host, _port: ["198.18.23.9"],
    )

    assert canonical == url


def test_source_reader_keeps_public_hostname_when_dns_returns_fake_ip(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    requested_hosts: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(str(request.url.host))
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain; charset=utf-8"},
            text="公开招聘页面完整正文",
        )

    reader = SourceReader(
        evidence_store,
        client=httpx.Client(transport=httpx.MockTransport(handle)),
        resolver=lambda _host, _port: ["198.18.23.9"],
    )

    result = reader.fetch(
        "https://talent.baidu.com/jobs/123",
        scope=source_scope,
    )

    assert result.status == "saved"
    assert requested_hosts == ["talent.baidu.com"]
    assert result.evidence is not None
    assert result.evidence.requested_url == "https://talent.baidu.com/jobs/123"


def test_literal_benchmark_fake_ip_is_still_rejected() -> None:
    with pytest.raises(UnsafeSourceURLError) as exc_info:
        validate_public_source_url(
            "https://198.18.23.9/jobs/123",
            resolver=lambda _host, _port: ["8.8.8.8"],
        )

    assert exc_info.value.code == "private_address"


@pytest.mark.parametrize("resolved", ["127.0.0.1", "10.2.3.4", "169.254.4.5"])
def test_hostname_resolving_to_local_network_is_rejected(resolved: str) -> None:
    with pytest.raises(UnsafeSourceURLError) as exc_info:
        validate_public_source_url(
            "https://join.qq.com/post/123",
            resolver=lambda _host, _port: [resolved],
        )

    assert exc_info.value.code == "private_address"


def test_mixed_global_and_private_dns_results_are_rejected() -> None:
    with pytest.raises(UnsafeSourceURLError) as exc_info:
        validate_public_source_url(
            "https://www.zhipin.com/job_detail/123",
            resolver=lambda _host, _port: ["8.8.8.8", "10.2.3.4"],
        )

    assert exc_info.value.code == "private_address"


def test_user_paste_is_evidence_without_claiming_a_network_fetch(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    evidence = evidence_store.save_user_provided(
        scope=source_scope,
        text="一面：围绕项目边界和失败恢复追问。",
        title="用户粘贴的真实面试记录",
        source_url="https://example.com/shared-note",
        purpose="user supplied interview experience",
    )

    assert evidence.provenance == "user_provided"
    assert evidence.http_status is None
    assert evidence.raw_content.decode() == evidence.text_content
    assert evidence_store.read_page(evidence.id).as_tool_result()["untrusted_evidence"] is True
    with evidence_store.store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_fetch_executions").fetchone()[0] == 0


def test_latest_url_follows_the_latest_fetch_even_when_body_reverts(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    common = {
        "scope": source_scope,
        "requested_url": "https://example.com/changing",
        "final_url": "https://example.com/changing",
        "title": "Changing source",
        "media_type": "text/plain",
        "charset": "utf-8",
        "http_status": 200,
    }
    first, _, _ = evidence_store.save_response(
        **common,
        raw_content=b"version A",
        text_content="version A",
    )
    second, _, _ = evidence_store.save_response(
        **common,
        raw_content=b"version B",
        text_content="version B",
    )
    reverted, status, _ = evidence_store.save_response(
        **common,
        raw_content=b"version A",
        text_content="version A",
    )

    assert first.id != second.id
    assert reverted.id == first.id
    assert status == "reused"
    assert evidence_store.get_latest_by_url("https://example.com/changing") == first


def test_oversized_source_is_rejected_explicitly_without_partial_evidence(
    evidence_store: SourceEvidenceStore,
    source_scope: SourceScope,
) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(
        200,
        headers={"Content-Type": "text/plain"},
        content=b"123456789",
    )))
    reader = SourceReader(
        evidence_store,
        client=client,
        resolver=lambda _host, _port: ["93.184.216.34"],
        max_response_bytes=8,
    )

    result = reader.fetch("https://example.com/large", scope=source_scope)

    assert result.status == "failed"
    assert result.error_code == "source_too_large"
    with evidence_store.store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_evidence").fetchone()[0] == 0


def test_canonical_url_keeps_query_but_removes_fragment_and_default_port() -> None:
    assert canonicalize_http_url(" HTTPS://Example.COM:443/a?b=2&a=1#part ") == (
        "https://example.com/a?b=2&a=1"
    )
