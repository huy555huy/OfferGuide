from __future__ import annotations

import json

import offerguide
from offerguide.agentic.job_finder_agent import JobFinderAgent
from offerguide.agentic.search import SearchHit
from offerguide.llm import LLMResponse
from offerguide.llm.client import ToolCall
from offerguide.platforms import RawJob
from offerguide.platforms.official_jobs import (
    SourceSearchResult,
    company_key,
    parse_baidu_campus_jobs,
    raw_job_from_tencent_campus,
    raw_job_from_tencent_social,
)


def _tool_call(name: str, args: dict, tc_id: str = "tc1") -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments=args)


def _llm_resp(*tcs: ToolCall, content: str = "") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="stub",
        tool_calls=list(tcs),
        finish_reason="tool_calls" if tcs else "stop",
    )


class _ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls_log: list[list[dict]] = []

    def chat_with_tools(self, *, messages, tools, **kw):
        self.calls_log.append(list(messages))
        if not self._responses:
            return _llm_resp(_tool_call("done", {"reason": "no_more"}))
        return self._responses.pop(0)


class _StubSearch:
    name = "stub"

    def search(self, q: str, *, max_results: int = 10) -> list[SearchHit]:
        return []


def test_company_key_reports_unverified_and_login_wall_sources() -> None:
    assert company_key("腾讯") == "tencent"
    assert company_key("百度") == "baidu"
    assert company_key("字节跳动") == "bytedance"
    assert company_key("美团") == "meituan"
    assert company_key("互联网大厂") is None


def test_parse_baidu_campus_ssr_jobs() -> None:
    row = {
        "education": "",
        "name": "数据平台-Data Agent 研发工程师-2026AIDU(J93348)",
        "postId": "a4f02a9a-242d-49c0-ab10-7da4764f7502",
        "jobId": "5f4aad46-3691-4293-852a-e1e8683fc816",
        "postType": "技术",
        "publishDate": "2025-09-08",
        "updateDate": "2026-05-01",
        "serviceCondition": "-有AI Agent或智能对话系统开发经验者优先",
        "workContent": "-负责AI Agent及相关智能应用的设计、开发和优化",
        "workPlace": "北京市",
        "projectType": "AIDU项目",
    }
    data = {"listData": {"recruitType": "GRADUATE", "listDetailData": [row]}}
    html = (
        "<html><script>window.__INITIAL_DATA__ ="
        + json.dumps(data, ensure_ascii=False)
        + ";</script></html>"
    )

    jobs = parse_baidu_campus_jobs(
        html,
        evidence_url="https://talent.baidu.com/jobs/list?search=Agent",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.source == "baidu_campus"
    assert job.company == "百度"
    assert job.source_id == "a4f02a9a-242d-49c0-ab10-7da4764f7502"
    assert "Data Agent" in job.title
    assert job.extras["source_verified"] is True
    assert job.extras["job_code"] == "J93348"
    assert job.url == (
        "https://talent.baidu.com/jobs/detail/GRADUATE/"
        "a4f02a9a-242d-49c0-ab10-7da4764f7502"
    )


def test_tencent_raw_jobs_mark_official_json_evidence() -> None:
    campus = raw_job_from_tencent_campus(
        {
            "postId": "1216462959547938818",
            "positionTitle": "AI-HR培训生（技术&应用方向）",
            "projectName": "应届实习",
            "bgs": "S3",
        },
        {
            "postId": "1216462959547938818",
            "title": "AI-HR培训生（技术&应用方向）",
            "desc": "项目介绍" * 30,
            "request": "任职要求" * 30,
            "workCityList": ["深圳总部", "成都"],
            "recruitCityList": ["远程面试"],
            "tidName": "职能",
        },
    )
    social = raw_job_from_tencent_social(
        {
            "PostId": "2035224441180553216",
            "RecruitPostName": "AI Agent 应用架构工程师",
            "LocationName": "深圳",
            "BGName": "TEG",
            "ProductName": "云架构平台",
            "CategoryName": "技术",
            "Responsibility": "负责边缘 AI Agent 系统架构设计与演进",
            "Requirement": "熟悉 Python、Go、C++ 和 Agent 基本技术原理",
        }
    )

    assert campus.source == "tencent_campus"
    assert campus.company == "腾讯"
    assert campus.extras["source_kind"] == "official_json_api"
    assert "join.qq.com/jobdesc.html" in (campus.url or "")
    assert social.source == "tencent_social"
    assert social.extras["source_verified"] is True
    assert "careers.tencent.com/jobdesc.html" in (social.url or "")


def test_job_finder_can_ingest_verified_official_tool(monkeypatch, tmp_path) -> None:
    store = offerguide.Store(tmp_path / "official-agent.db")
    store.init_schema()
    raw_job = RawJob(
        source="baidu_campus",
        source_id="p1",
        url="https://talent.baidu.com/jobs/detail/GRADUATE/p1",
        title="数据平台-Data Agent 研发工程师-2026AIDU(J93348)",
        company="百度",
        location="北京市",
        raw_text="负责AI Agent及相关智能应用的设计、开发和优化。" * 20,
        extras={"source_verified": True, "source_kind": "official_ssr"},
    )

    def _fake_search_verified_official_jobs(**kwargs):
        assert kwargs["company"] == "百度"
        assert kwargs["keyword"] == "AI Agent"
        return [
            SourceSearchResult(
                source="baidu_campus",
                status="ok",
                evidence_url="https://talent.baidu.com/jobs/list?search=AI+Agent",
                jobs=[raw_job],
                note="fixture",
            )
        ]

    monkeypatch.setattr(
        "offerguide.platforms.official_jobs.search_verified_official_jobs",
        _fake_search_verified_official_jobs,
    )
    llm = _ScriptedLLM(
        [
            _llm_resp(
                _tool_call(
                    "search_verified_official_jobs",
                    {"company": "百度", "keyword": "AI Agent", "limit": 3},
                )
            ),
            _llm_resp(_tool_call("done", {"reason": "已查官方源"})),
        ]
    )
    agent = JobFinderAgent(store=store, llm=llm, search=_StubSearch(), max_iterations=5)

    try:
        result = agent.run(north_star="AI Agent 校招")
    finally:
        agent.close()

    assert result.inserted == 1
    with store.connect() as conn:
        row = conn.execute(
            "SELECT source, company, title, extras_json FROM jobs"
        ).fetchone()
    assert row[0] == "baidu_campus"
    assert row[1] == "百度"
    assert "Data Agent" in row[2]
    assert json.loads(row[3])["source_verified"] is True


def test_job_finder_preserves_reasoning_content_between_tool_turns(
    monkeypatch,
    tmp_path,
) -> None:
    store = offerguide.Store(tmp_path / "reasoning.db")
    store.init_schema()

    monkeypatch.setattr(
        "offerguide.platforms.official_jobs.search_verified_official_jobs",
        lambda **kw: [
            SourceSearchResult(
                source="bytedance",
                status="unverified_js_shell",
                evidence_url="https://jobs.bytedance.com/campus/",
                jobs=[],
                note="fixture",
            )
        ],
    )
    tc = _tool_call(
        "search_verified_official_jobs",
        {"company": "字节", "keyword": "AI Agent"},
        tc_id="call_reasoning",
    )
    assistant_message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should probe official sources first.",
        "tool_calls": [
            {
                "id": "call_reasoning",
                "type": "function",
                "function": {
                    "name": "search_verified_official_jobs",
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
        ],
    }
    llm = _ScriptedLLM(
        [
            LLMResponse(
                content="",
                model="stub",
                tool_calls=[tc],
                finish_reason="tool_calls",
                assistant_message=assistant_message,
            ),
            _llm_resp(_tool_call("done", {"reason": "confirmed limitation"})),
        ]
    )
    agent = JobFinderAgent(store=store, llm=llm, search=_StubSearch(), max_iterations=3)

    try:
        agent.run(north_star="AI Agent 官方源")
    finally:
        agent.close()

    second_call_messages = llm.calls_log[1]
    assistant_history = next(m for m in second_call_messages if m.get("role") == "assistant")
    assert assistant_history["reasoning_content"] == "I should probe official sources first."
