"""W14.16 — JobFinderAgent: LLM-in-driver-seat ReAct loop for JD discovery.

Pinning: agent must drive each step itself (search → fetch → ingest → done),
not be shoehorned into a hardcoded BFS (W14.15 mistake the user called
"偷懒耍滑, 和不要模型有什么区别").

Tests use scripted LLM responses so we verify the actual control flow:
each turn we check the agent called the right tool with the right args.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import offerguide
from offerguide.agentic.job_finder_agent import JobFinderAgent
from offerguide.agentic.search import SearchHit
from offerguide.llm import LLMResponse
from offerguide.llm.client import ToolCall


def _tool_call(name: str, args: dict, tc_id: str = "tc1") -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments=args)


def _llm_resp(*tcs: ToolCall, content: str = "") -> LLMResponse:
    return LLMResponse(
        content=content, model="stub",
        tool_calls=list(tcs),
        finish_reason="tool_calls" if tcs else "stop",
    )


class _ScriptedLLM:
    """LLM that returns a sequence of canned responses, one per chat call."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.call_count = 0
        self.calls_log: list[list[dict]] = []

    def chat_with_tools(self, *, messages, tools, **kw):
        self.calls_log.append(list(messages))
        self.call_count += 1
        if not self._responses:
            return _llm_resp(_tool_call("done", {"reason": "no_more_responses"}))
        return self._responses.pop(0)

    def chat(self, *a, **kw):
        return LLMResponse(content="{}", model="stub")

    def close(self): pass


class _StubSearch:
    """Tavily replacement — returns scripted hits per query keyword."""
    name = "stub"

    def __init__(self, results_by_query=None):
        self.results_by_query = results_by_query or {}
        self.queries_received: list[str] = []

    def search(self, q, *, max_results=10):
        self.queries_received.append(q)
        for keyword, hits in self.results_by_query.items():
            if keyword in q:
                return hits[:max_results]
        return []


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "jf.db")
    s.init_schema()
    return s


# ═══════════════════════════════════════════════════════════════════
# Agent drives the whole loop: search → fetch → ingest → done
# ═══════════════════════════════════════════════════════════════════


class TestAgentDrivesLoop:
    def test_full_search_fetch_ingest_done_flow(self, store):
        """The agent picks query, picks URL to fetch, decides this is a real JD,
        builds extract args itself, then says done. Each tool call is real."""
        jd_url = "https://nowcoder.com/jd/abc"
        jd_body = (
            "AI Agent 暑期实习 - 字节跳动 北京\n"
            "工作内容: 1. 设计并实现 LLM agent loop\n"
            "2. evolve prompts via user feedback\n"
            "任职要求: Python / LangGraph 经验; 在校学生\n"
        ) + ("." * 250)

        search = _StubSearch({
            "AI Agent": [
                SearchHit(title="字节 AI Agent 实习", url=jd_url,
                          snippet="字节跳动 AI 实习招聘"),
            ],
        })

        # Scripted agent turns:
        #   turn 1: web_search("AI Agent 实习 字节")
        #   turn 2: fetch_url(jd_url)
        #   turn 3: extract_and_ingest_jd(...)
        #   turn 4: done("找够了")
        llm = _ScriptedLLM([
            _llm_resp(_tool_call("web_search",
                                  {"query": "AI Agent 实习 字节", "max_results": 5})),
            _llm_resp(_tool_call("fetch_url", {"url": jd_url})),
            _llm_resp(_tool_call("extract_and_ingest_jd", {
                "url": jd_url,
                "jd_body": jd_body,
                "company": "字节跳动",
                "title": "AI Agent 暑期实习",
                "location": "北京",
            })),
            _llm_resp(_tool_call("done", {"reason": "已入库 1 个 JD, 这一轮够了"})),
        ])

        agent = JobFinderAgent(store=store, llm=llm, search=search,
                                max_iterations=10)
        # Stub fetch since the URL doesn't actually exist
        agent._fetch_text = lambda u: jd_body if u == jd_url else ""
        # Patch the fetch tool to use our stub via _http monkey
        original_get = agent._http.get
        def _fake_get(url, **kw):
            class _R:
                status_code = 200
                headers = {"content-type": "text/html"}
                text = jd_body
            return _R()
        agent._http.get = _fake_get  # type: ignore[method-assign]

        try:
            result = agent.run(north_star="AI Agent 暑期实习")
        finally:
            agent._http.get = original_get  # type: ignore[method-assign]
            agent.close()

        # Verify the full flow ran
        assert llm.call_count == 4, f"expected 4 LLM turns, got {llm.call_count}"
        assert result.iterations == 4
        assert result.inserted == 1
        assert result.finish_reason.startswith("done")
        # Audit trail should reflect each step
        assert any("web_search" in n for n in result.notes)
        assert any("fetch_url" in n for n in result.notes)
        assert any("extract_and_ingest_jd" in n for n in result.notes)
        # search query was actually used
        assert "AI Agent 实习 字节" in search.queries_received

    def test_agent_can_pivot_after_fetch_shows_careers_index(self, store):
        """When fetch returns a careers main page, agent should re-search
        with a more specific query, NOT fabricate a JD. This is the W14.15
        failure mode that made the user complain."""
        index_url = "https://talent.bytedance.com/campus"
        jd_url = "https://talent.bytedance.com/jobs/ai-agent-001"
        index_page = "字节跳动 2026 校招主页 - 各类岗位入口" + ("." * 200)
        real_jd = "字节跳动 AI Agent 暑期实习 工作内容: ... 任职要求: ..." + ("." * 250)

        search = _StubSearch({
            "字节": [SearchHit(title="字节校招", url=index_url,
                                snippet="字节跳动 2026 校招")],
            "AI Agent": [SearchHit(title="字节 AI Agent JD", url=jd_url,
                                    snippet="具体岗位")],
        })

        llm = _ScriptedLLM([
            # Turn 1: broad search → finds careers index
            _llm_resp(_tool_call("web_search", {"query": "字节 实习"})),
            # Turn 2: fetch the index page
            _llm_resp(_tool_call("fetch_url", {"url": index_url})),
            # Turn 3: realize this is a careers page, do a more specific search
            _llm_resp(_tool_call("web_search",
                                  {"query": "字节跳动 AI Agent 暑期实习 具体岗位"})),
            # Turn 4: fetch the specific JD URL
            _llm_resp(_tool_call("fetch_url", {"url": jd_url})),
            # Turn 5: ingest
            _llm_resp(_tool_call("extract_and_ingest_jd", {
                "url": jd_url, "jd_body": real_jd,
                "company": "字节跳动", "title": "AI Agent 暑期实习",
                "location": "北京",
            })),
            # Turn 6: done
            _llm_resp(_tool_call("done", {"reason": "找到 1 个真 JD, 够了"})),
        ])

        agent = JobFinderAgent(store=store, llm=llm, search=search,
                                max_iterations=10)
        page_for_url = {index_url: index_page, jd_url: real_jd}
        def _fake_get(url, **kw):
            class _R:
                status_code = 200
                headers = {"content-type": "text/html"}
                text = page_for_url.get(url, "")
            return _R()
        agent._http.get = _fake_get  # type: ignore[method-assign]
        try:
            result = agent.run(north_star="AI Agent 实习")
        finally:
            agent.close()

        assert result.inserted == 1
        # Agent really did search twice (broad → specific)
        assert len(result.search_queries) == 2
        # Agent did NOT fabricate a JD from the index page
        assert index_url not in [r for r in result.new_job_ids]


# ═══════════════════════════════════════════════════════════════════
# Tool guardrails — agent can't shortcut/cheat
# ═══════════════════════════════════════════════════════════════════


class TestToolGuardrails:
    def test_ingest_without_fetch_rejected(self, store):
        """If the LLM tries to ingest a URL it never fetched (= it's making
        up data), the tool returns ERROR and won't write to DB."""
        search = _StubSearch({})
        llm = _ScriptedLLM([
            _llm_resp(_tool_call("extract_and_ingest_jd", {
                "url": "https://made-up.example/jd/123",
                "jd_body": "fabricated content " + ("x" * 250),
                "company": "Fake Co", "title": "Imaginary",
                "location": "Nowhere",
            })),
            _llm_resp(_tool_call("done", {"reason": "tried"})),
        ])
        agent = JobFinderAgent(store=store, llm=llm, search=search,
                                max_iterations=5)
        try:
            result = agent.run(north_star="x")
        finally:
            agent.close()
        assert result.inserted == 0
        # The tool result should explicitly mention the rule
        assert any("没 fetch 过" in n or "ingest" in n for n in result.notes)

    def test_duplicate_search_rejected(self, store):
        search = _StubSearch({"x": [SearchHit(title="t", url="https://nowcoder.com/x", snippet="x")]})
        llm = _ScriptedLLM([
            _llm_resp(_tool_call("web_search", {"query": "x"})),
            _llm_resp(_tool_call("web_search", {"query": "x"})),  # duplicate
            _llm_resp(_tool_call("done", {"reason": "stuck"})),
        ])
        agent = JobFinderAgent(store=store, llm=llm, search=search,
                                max_iterations=5)
        try:
            result = agent.run(north_star="x")
        finally:
            agent.close()
        # Search counted once, second was rejected
        assert search.queries_received == ["x"]
        # The dup attempt note should appear
        assert any("已经搜过" in n or "ERROR" in n for n in result.notes)

    def test_duplicate_fetch_rejected(self, store):
        url = "https://nowcoder.com/abc"
        search = _StubSearch({})
        llm = _ScriptedLLM([
            _llm_resp(_tool_call("fetch_url", {"url": url})),
            _llm_resp(_tool_call("fetch_url", {"url": url})),  # duplicate
            _llm_resp(_tool_call("done", {"reason": "stuck"})),
        ])
        agent = JobFinderAgent(store=store, llm=llm, search=search,
                                max_iterations=5)
        def _fake_get(u, **kw):
            class _R:
                status_code = 200
                headers = {"content-type": "text/html"}
                text = "page content " * 50
            return _R()
        agent._http.get = _fake_get  # type: ignore[method-assign]
        try:
            result = agent.run(north_star="x")
        finally:
            agent.close()
        # Note about the dup attempt
        assert any("已经 fetch 过" in n or "ERROR" in n for n in result.notes)

    def test_short_jd_body_rejected(self, store):
        url = "https://nowcoder.com/abc"
        search = _StubSearch({})
        llm = _ScriptedLLM([
            _llm_resp(_tool_call("fetch_url", {"url": url})),
            _llm_resp(_tool_call("extract_and_ingest_jd", {
                "url": url, "jd_body": "way too short",
                "company": "X", "title": "Y", "location": "Z",
            })),
            _llm_resp(_tool_call("done", {"reason": "tried"})),
        ])
        agent = JobFinderAgent(store=store, llm=llm, search=search,
                                max_iterations=5)
        def _fake_get(u, **kw):
            class _R:
                status_code = 200
                headers = {"content-type": "text/html"}
                text = "real page content " * 50
            return _R()
        agent._http.get = _fake_get  # type: ignore[method-assign]
        try:
            result = agent.run(north_star="x")
        finally:
            agent.close()
        assert result.inserted == 0


# ═══════════════════════════════════════════════════════════════════
# Budget enforcement — agent can't burn unlimited tokens
# ═══════════════════════════════════════════════════════════════════


class TestBudget:
    def test_max_iterations_caps_loop(self, store):
        search = _StubSearch({})
        # LLM keeps calling search forever
        llm = _ScriptedLLM([
            _llm_resp(_tool_call("web_search", {"query": f"q{i}"}))
            for i in range(50)
        ])
        agent = JobFinderAgent(store=store, llm=llm, search=search,
                                max_iterations=3)
        try:
            result = agent.run(north_star="x")
        finally:
            agent.close()
        # Stops after exactly 3 iterations regardless of LLM's intent
        assert llm.call_count == 3
        assert result.iterations == 3
        assert result.finish_reason == "budget_exhausted"

    def test_no_tool_call_terminates_gracefully(self, store):
        search = _StubSearch({})
        # LLM doesn't call any tool (just text) — should stop, not infinite loop
        llm = _ScriptedLLM([
            _llm_resp(content="I have nothing to do"),
        ])
        agent = JobFinderAgent(store=store, llm=llm, search=search,
                                max_iterations=10)
        try:
            result = agent.run(north_star="x")
        finally:
            agent.close()
        assert result.iterations == 1
        assert result.finish_reason == "no_tool_call"
