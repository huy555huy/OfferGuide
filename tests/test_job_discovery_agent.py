from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from offerguide import Store
from offerguide.agentic.search import SearchHit
from offerguide.llm.client import LLMResponse, ToolCall
from offerguide.research_agents.job_discovery import (
    CandidateEvidence,
    CandidateEvidenceDocument,
    JobDiscoveryAgent,
    JobDiscoveryRepository,
    JobPostingEvidence,
    JobSelectionItem,
    StoredJobSourceEvidenceVerifier,
    build_job_source_tools,
    init_job_discovery_schema,
    platform_job_reference,
    standard_source_dependencies,
)
from offerguide.research_agents.runner import (
    AgentRunner,
    AgentRunStatus,
    AgentTool,
    AgentToolResult,
)
from offerguide.research_agents.sources import (
    SearchExecutor,
    SourceEvidenceStore,
    SourceReader,
)


@dataclass
class _SearchBackend:
    name: str = "test-search"

    def search(self, query: str, *, max_results: int = 10):
        assert query == "真实 AI 产品岗位"
        return [
            SearchHit(
                title="示例公司 AI 产品经理",
                url="https://jobs.example.com/positions/42",
                snippet="这只是搜索摘要，不能进入推荐。",
            )
        ][:max_results]


@pytest.fixture
def agent_setup(tmp_path):
    store = Store(tmp_path / "agent.db")
    store.init_schema()
    source_store = SourceEvidenceStore(store)
    source_store.init_schema()
    init_job_discovery_schema(store)
    repository = JobDiscoveryRepository(store)
    context = repository.replace_search_context(
        intent="寻找能参与真实 AI 产品交付的岗位",
        hard_constraints=["用户明确要求工作地点在上海或允许远程"],
        feedback=["更关注能接触用户反馈的团队"],
        expected_revision=None,
    )
    return store, source_store, repository, context


def _candidate(text: str = "确认后的完整候选人材料") -> CandidateEvidence:
    return CandidateEvidence(
        documents=[
            CandidateEvidenceDocument(
                reference="resume:confirmed",
                kind="master_resume",
                title="确认后的主简历",
                text=text,
            )
        ]
    )


def _tool_response(name: str, args: dict, call_id: str) -> LLMResponse:
    return LLMResponse(
        content="",
        model="stub",
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=args,
                arguments_raw=json.dumps(args, ensure_ascii=False),
            )
        ],
    )


def _job_args() -> dict:
    return {
        "source_evidence_id": 1,
        "source_name": "official-example",
        "source_job_id": "position-42",
        "canonical_url": "https://jobs.example.com/positions/42",
        "company": "示例公司",
        "title": "AI 产品经理",
        "location": "上海",
        "recruitment_type": "实习",
        "page_time_information": ["页面显示岗位仍开放"],
        "jd_text": (
            "负责从真实用户问题发现、方案设计到上线复盘的完整产品交付。\n"
            "需要能够分析复杂需求，与设计和工程团队协作并根据反馈持续改进。"
        ),
        "source_status": "unknown",
    }


def _record_args() -> dict:
    facts = _job_args()
    return {
        key: facts[key]
        for key in (
            "source_evidence_id",
            "company",
            "title",
            "location",
            "recruitment_type",
            "page_time_information",
            "source_status",
        )
    }


def _grounding() -> list[Any]:
    return [
        {
            "reference": "search-context:intent",
            "quote": "寻找能参与真实 AI 产品交付的岗位",
        }
    ]


def _generic_saved_job() -> JobPostingEvidence:
    return JobPostingEvidence.model_validate(_job_args()).model_copy(
        update={
            "source_name": "jobs.example.com",
            "source_job_id": None,
            "jd_text": _http_client_body_text(),
        }
    )


def _http_client_body_text() -> str:
    facts = _job_args()
    return "\n".join(
        [
            facts["company"],
            facts["title"],
            facts["location"],
            facts["recruitment_type"],
            *facts["page_time_information"],
            facts["jd_text"],
        ]
    )


def _http_client(*, status_code: int = 200) -> httpx.Client:
    body = _http_client_body_text().encode("utf-8")

    def _handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://jobs.example.com/positions/42"
        return httpx.Response(
            status_code,
            headers={"content-type": "text/plain; charset=utf-8"},
            content=body,
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(_handler))


def _agent(
    *,
    llm,
    source_store: SourceEvidenceStore,
    repository: JobDiscoveryRepository,
    candidate_text: str = "确认后的完整候选人材料",
    source_status_code: int = 200,
) -> JobDiscoveryAgent:
    search_executor = SearchExecutor(source_store, [_SearchBackend()])
    reader = SourceReader(
        source_store,
        client=_http_client(status_code=source_status_code),
        resolver=lambda host, port: ["8.8.8.8"],
    )
    return JobDiscoveryAgent(
        repository=repository,
        runner=AgentRunner(llm),
        candidate_evidence_loader=lambda: _candidate(candidate_text),
        source_tools=build_job_source_tools(),
        source_evidence_verifier=StoredJobSourceEvidenceVerifier(source_store),
        dependencies=standard_source_dependencies(
            evidence_store=source_store,
            search_executor=search_executor,
            source_reader=reader,
        ),
    )


def test_agent_searches_reads_records_and_publishes_one_ordered_selection(agent_setup):
    store, source_store, repository, _ = agent_setup
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "search_job_sources",
                {"query": "真实 AI 产品岗位", "max_results_per_backend": 5},
                "search",
            ),
            _tool_response(
                "fetch_job_source",
                {
                    "url": "https://jobs.example.com/positions/42",
                    "purpose": "job_detail",
                },
                "fetch",
            ),
            _tool_response(
                "read_job_source",
                {"evidence_id": 1, "offset": 0, "max_chars": 50_000},
                "read",
            ),
            _tool_response("record_verified_web_job", _record_args(), "record"),
            _tool_response(
                "publish_job_selection",
                {
                    "items": [
                        {
                            "job_evidence_id": 1,
                                "why_worth_attention": (
                                    "寻找能参与真实 AI 产品交付的岗位；该职责覆盖研究到复盘。"
                                ),
                                "grounding_quotes": _grounding(),
                            "concerns": ["页面没有说明团队规模"],
                            "unknowns": ["页面没有说明转正安排"],
                        }
                    ],
                    "coverage_summary": "搜索并读取了示例公司的完整官方岗位页。",
                    "evidence_gaps": ["团队规模仍未知"],
                },
                "publish",
            ),
        ]
    )
    agent = _agent(llm=llm, source_store=source_store, repository=repository)

    result = agent.run(trigger_reason="用户要求立即刷新岗位")

    assert result.status == AgentRunStatus.PUBLISHED
    assert result.terminal_tool == "publish_job_selection"
    assert result.tool_calls == (
        "search_job_sources",
        "fetch_job_source",
        "read_job_source",
        "record_verified_web_job",
        "publish_job_selection",
    )
    selection = repository.get_current_selection()
    assert selection is not None
    assert len(selection.items) == 1
    saved_job = repository.get_job_evidence(selection.items[0].job_evidence_id)
    assert saved_job is not None
    assert saved_job.job_id is not None
    with store.connect() as conn:
        assert conn.execute(
            "SELECT raw_text FROM jobs WHERE id = ?", (saved_job.job_id,)
        ).fetchone()[0] == _http_client_body_text()


def test_complete_long_context_is_given_to_model_without_character_slicing(agent_setup):
    _, source_store, repository, _ = agent_setup
    candidate_text = "完整候选人证据" * 6_000
    captured: dict = {}

    def _stop(messages, **kwargs):
        captured["messages"] = messages
        return LLMResponse(content="我先不调用工具。", model="stub", tool_calls=[])

    llm = MagicMock()
    llm.chat_with_tools = _stop
    result = _agent(
        llm=llm,
        source_store=source_store,
        repository=repository,
        candidate_text=candidate_text,
    ).run(trigger_reason="后台刷新")

    assert result.status == AgentRunStatus.BLOCKED
    assert result.reason == "model_stopped_without_terminal_tool"
    assert candidate_text in captured["messages"][1]["content"]
    assert repository.get_current_selection() is None


def test_model_cannot_publish_without_using_a_real_source_tool(agent_setup):
    _, source_store, repository, _ = agent_setup
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "publish_job_selection",
                {
                    "items": [],
                    "coverage_summary": "模型没有实际搜索。",
                    "evidence_gaps": ["没有执行来源请求"],
                },
                "publish-too-early",
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "尚未取得真实来源",
                    "checks_completed": [],
                    "evidence_gaps": ["需要执行真实搜索"],
                },
                "blocked",
            ),
        ]
    )
    result = _agent(
        llm=llm, source_store=source_store, repository=repository
    ).run(trigger_reason="立即刷新")

    assert result.status == AgentRunStatus.BLOCKED
    assert result.terminal_tool == "report_job_discovery_blocked"
    assert repository.get_current_selection() is None


def test_rejected_publication_returns_actionable_contract_hints(agent_setup):
    _, source_store, repository, _ = agent_setup
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "search_job_sources",
                {"query": "真实 AI 产品岗位", "max_results_per_backend": 5},
                "search",
            ),
            _tool_response(
                "fetch_job_source",
                {"url": "https://jobs.example.com/positions/42"},
                "fetch",
            ),
            _tool_response(
                "read_job_source",
                {"evidence_id": 1, "offset": 0, "max_chars": 50_000},
                "read",
            ),
            _tool_response("record_verified_web_job", _record_args(), "record"),
            _tool_response(
                "publish_job_selection",
                {
                    "items": [
                        {
                            "job_evidence_id": 1,
                            "why_worth_attention": "岗位正文说明了完整产品职责。",
                            "grounding_quotes": [
                                {
                                    "reference": "job_evidence_id:1",
                                    "quote": "完整产品交付",
                                }
                            ],
                            "concerns": [],
                            "unknowns": [],
                        }
                    ],
                    "coverage_summary": "已读取岗位。",
                    "evidence_gaps": [],
                },
                "invalid-publish",
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "测试结束",
                    "checks_completed": ["已取得发布纠错信息"],
                    "evidence_gaps": ["需要按契约修正引用"],
                },
                "blocked",
            ),
        ]
    )

    result = _agent(
        llm=llm,
        source_store=source_store,
        repository=repository,
    ).run(trigger_reason="验证发布纠错信息")

    assert result.status == AgentRunStatus.BLOCKED
    final_messages = llm.chat_with_tools.call_args_list[-1].args[0]
    publish_message = next(
        message
        for message in reversed(final_messages)
        if message.get("tool_call_id") == "invalid-publish"
    )
    publish_result = json.loads(publish_message["content"])["result"]
    contract = publish_result["publication_contract"]
    assert contract["recorded_job_evidence_ids_this_run"] == [1]
    assert contract["allowed_grounding_references"] == [
        "search-context:intent",
        "search-context:hard-constraint:1",
        "search-context:feedback:1",
        "resume:confirmed",
    ]
    assert "job-evidence grounding reference" in contract["grounding_note"]


def test_failed_search_backend_cannot_unlock_empty_publication(agent_setup):
    _, source_store, repository, _ = agent_setup

    class _FailedBackend:
        name = "failed-search"

        def search(self, query: str, *, max_results: int = 10):
            raise RuntimeError("HTTP 200 challenge page was rejected")

    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "search_job_sources", {"query": "真实 AI 产品岗位"}, "search"
            ),
            _tool_response(
                "publish_job_selection",
                {
                    "items": [],
                    "coverage_summary": "搜索没有返回岗位。",
                    "evidence_gaps": ["来源不可用"],
                },
                "invalid-empty",
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "搜索来源失败",
                    "checks_completed": ["尝试了真实搜索请求"],
                    "evidence_gaps": ["没有成功完成的搜索"],
                },
                "blocked",
            ),
        ]
    )
    agent = JobDiscoveryAgent(
        repository=repository,
        runner=AgentRunner(llm),
        candidate_evidence_loader=_candidate,
        source_tools=build_job_source_tools(),
        source_evidence_verifier=StoredJobSourceEvidenceVerifier(source_store),
        dependencies=standard_source_dependencies(
            evidence_store=source_store,
            search_executor=SearchExecutor(source_store, [_FailedBackend()]),
            source_reader=SourceReader(
                source_store,
                client=_http_client(),
                resolver=lambda host, port: ["8.8.8.8"],
            ),
        ),
    )

    result = agent.run(trigger_reason="刷新岗位")

    assert result.status == AgentRunStatus.BLOCKED
    assert repository.get_current_selection() is None
    history = source_store.search_history(
        subject_kind="job_search", subject_id="current", subject_revision=1
    )
    assert [(item.backend, item.status) for item in history] == [
        ("failed-search", "failed")
    ]


def test_model_cannot_record_a_job_before_reading_the_complete_source(agent_setup):
    _, source_store, repository, _ = agent_setup
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "fetch_job_source",
                {"url": _job_args()["canonical_url"], "purpose": "job_detail"},
                "fetch",
            ),
            _tool_response(
                "record_verified_web_job", _record_args(), "record-too-early"
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "岗位页尚未完整读取",
                    "checks_completed": ["已取得详情页但尚未完成读取"],
                    "evidence_gaps": ["需要读取完整来源正文"],
                },
                "blocked",
            ),
        ]
    )

    result = _agent(
        llm=llm, source_store=source_store, repository=repository
    ).run(trigger_reason="立即刷新")

    assert result.status == AgentRunStatus.BLOCKED
    assert repository.list_job_evidence() == []


def test_web_record_requires_source_evidence_id_even_after_a_complete_read(agent_setup):
    _, source_store, repository, _ = agent_setup
    missing_id = _record_args()
    del missing_id["source_evidence_id"]
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "fetch_job_source",
                {"url": _job_args()["canonical_url"], "purpose": "job_detail"},
                "fetch",
            ),
            _tool_response(
                "read_job_source",
                {"evidence_id": 1, "offset": 0, "max_chars": 50_000},
                "read",
            ),
            _tool_response("record_verified_web_job", missing_id, "record-without-id"),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "普通网页录入缺少 evidence id",
                    "checks_completed": ["完整读取了来源"],
                    "evidence_gaps": ["需要绑定实际保存的来源证据"],
                },
                "blocked",
            ),
        ]
    )

    result = _agent(
        llm=llm, source_store=source_store, repository=repository
    ).run(trigger_reason="刷新岗位")

    assert result.status == AgentRunStatus.BLOCKED
    assert repository.list_job_evidence() == []


def test_generic_web_job_rejects_model_identity_and_short_jd_substring(agent_setup):
    _, source_store, repository, _ = agent_setup
    malicious = {
        **_record_args(),
        "source_name": "attacker-controlled-label",
        "source_job_id": "invented-id",
        "canonical_url": _job_args()["canonical_url"],
        "jd_text": _job_args()["jd_text"],
    }
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "fetch_job_source",
                {"url": _job_args()["canonical_url"], "purpose": "job_detail"},
                "fetch",
            ),
            _tool_response(
                "read_job_source",
                {"evidence_id": 1, "offset": 0, "max_chars": 50_000},
                "read",
            ),
            _tool_response(
                "record_verified_web_job", malicious, "malicious-record"
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "模型提交了不受信任的岗位身份和截断 JD",
                    "checks_completed": ["完整读取了来源"],
                    "evidence_gaps": ["需要按证据契约重新登记"],
                },
                "blocked",
            ),
        ]
    )

    result = _agent(
        llm=llm, source_store=source_store, repository=repository
    ).run(trigger_reason="刷新岗位")

    assert result.status == AgentRunStatus.BLOCKED
    assert repository.list_job_evidence() == []


def test_generic_collision_cannot_republish_preserved_platform_evidence(agent_setup):
    _, source_store, repository, _ = agent_setup
    platform_jd = (
        "PLATFORM_SENTINEL：负责从用户调研到产品上线和效果复盘的完整交付。\n"
        "需要分析复杂需求，与多个团队协作验证方案并根据反馈持续改进。"
    )
    platform = repository.save_job_evidence(
        JobPostingEvidence.model_validate(_job_args()).model_copy(
            update={
                "evidence_kind": "platform_adapter",
                "source_evidence_id": 999,
                "source_name": "official-example",
                "source_job_id": "position-42",
                "jd_text": platform_jd,
                "source_status": "open",
            }
        )
    )
    assert platform.id is not None
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "fetch_job_source",
                {"url": _job_args()["canonical_url"], "purpose": "job_detail"},
                "fetch",
            ),
            _tool_response(
                "read_job_source",
                {"evidence_id": 1, "offset": 0, "max_chars": 50_000},
                "read",
            ),
            _tool_response("record_verified_web_job", _record_args(), "collision"),
            _tool_response(
                "publish_job_selection",
                {
                    "items": [
                        {
                            "job_evidence_id": platform.id,
                            "why_worth_attention": (
                                "寻找能参与真实 AI 产品交付的岗位，旧证据不能冒充刷新。"
                            ),
                            "grounding_quotes": _grounding(),
                            "concerns": [],
                            "unknowns": [],
                        }
                    ],
                    "coverage_summary": "只读取了 generic 页面。",
                    "evidence_gaps": ["需要平台适配器刷新"],
                },
                "publish",
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "generic 页面不能刷新平台证据",
                    "checks_completed": ["完整读取了 generic 页面"],
                    "evidence_gaps": ["需要平台适配器重新核验"],
                },
                "blocked",
            ),
        ]
    )

    result = _agent(
        llm=llm, source_store=source_store, repository=repository
    ).run(trigger_reason="刷新岗位")

    assert result.status == AgentRunStatus.BLOCKED
    assert repository.get_current_selection() is None
    record_observation = json.dumps(
        llm.chat_with_tools.call_args_list[3].args[0], ensure_ascii=False
    )
    assert "generic_collision_preserved_platform_evidence" in record_observation
    live = repository.get_job_evidence(platform.id)
    assert live is not None
    assert live.evidence_kind == "platform_adapter"
    assert live.source_evidence_id == 999
    assert live.jd_text == platform_jd


def test_old_run_is_stale_when_search_context_changes_before_publish(agent_setup):
    _, source_store, repository, context = agent_setup

    class _MutatingSearch(_SearchBackend):
        def search(self, query: str, *, max_results: int = 10):
            repository.replace_search_context(
                intent="用户在运行期间修改后的找岗方向",
                hard_constraints=[],
                feedback=[],
                expected_revision=context.revision,
            )
            return super().search(query, max_results=max_results)

    search_executor = SearchExecutor(source_store, [_MutatingSearch()])
    reader = SourceReader(
        source_store,
        client=_http_client(),
        resolver=lambda host, port: ["8.8.8.8"],
    )
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "search_job_sources", {"query": "真实 AI 产品岗位"}, "search"
            ),
            _tool_response(
                "publish_job_selection",
                {
                    "items": [],
                    "coverage_summary": "基于旧上下文完成的搜索。",
                    "evidence_gaps": ["搜索结果已因用户修改方向而过期"],
                },
                "publish",
            ),
        ]
    )
    agent = JobDiscoveryAgent(
        repository=repository,
        runner=AgentRunner(llm),
        candidate_evidence_loader=_candidate,
        source_tools=build_job_source_tools(),
        source_evidence_verifier=StoredJobSourceEvidenceVerifier(source_store),
        dependencies=standard_source_dependencies(
            evidence_store=source_store,
            search_executor=search_executor,
            source_reader=reader,
        ),
    )

    result = agent.run(trigger_reason="刷新岗位")
    assert result.status == AgentRunStatus.STALE
    assert repository.get_current_selection() is None


def test_unchanged_rejects_a_search_that_did_not_refresh_the_selected_job(agent_setup):
    _, source_store, repository, context = agent_setup
    saved = repository.save_job_evidence(_generic_saved_job())
    repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=saved.id,
                why_worth_attention=(
                    "寻找能参与真实 AI 产品交付的岗位，当前岗位仍符合该方向。"
                ),
                grounding_quotes=_grounding(),
            )
        ],
        candidate_evidence=_candidate(),
        coverage_summary="此前已经读取完整岗位页。",
        evidence_gaps=["团队规模尚未公开"],
    )
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "search_job_sources", {"query": "真实 AI 产品岗位"}, "search"
            ),
            _tool_response(
                "confirm_job_selection_unchanged",
                {
                    "reason": "当前查询未发现需要替换或重排的证据。",
                    "checks_completed": ["重新执行了真实岗位搜索"],
                },
                "unchanged",
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "当前选择中的岗位尚未逐一重新读取",
                    "checks_completed": ["执行了市场搜索"],
                    "evidence_gaps": ["需要重新读取并登记当前岗位"],
                },
                "blocked",
            ),
        ]
    )
    result = _agent(
        llm=llm, source_store=source_store, repository=repository
    ).run(trigger_reason="后台定期刷新")

    assert result.status == AgentRunStatus.BLOCKED
    assert result.terminal_tool == "report_job_discovery_blocked"
    assert repository.current_result_revision() == 1


def test_unchanged_requires_rereading_and_recording_every_selected_job(agent_setup):
    _, source_store, repository, context = agent_setup
    saved = repository.save_job_evidence(_generic_saved_job())
    repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=saved.id,
                why_worth_attention=(
                    "寻找能参与真实 AI 产品交付的岗位，当前岗位仍符合该方向。"
                ),
                grounding_quotes=_grounding(),
            )
        ],
        candidate_evidence=_candidate(),
        coverage_summary="此前已经读取完整岗位页。",
        evidence_gaps=["团队规模尚未公开"],
    )
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "fetch_job_source",
                {"url": _job_args()["canonical_url"], "purpose": "refresh current job"},
                "fetch",
            ),
            _tool_response(
                "read_job_source",
                {"evidence_id": 1, "offset": 0, "max_chars": 50_000},
                "read",
            ),
            _tool_response("record_verified_web_job", _record_args(), "record"),
            _tool_response(
                "confirm_job_selection_unchanged",
                {
                    "reason": "当前岗位正文和开放状态均未变化。",
                    "checks_completed": ["重新读取并登记了当前岗位完整来源"],
                },
                "unchanged",
            ),
        ]
    )

    result = _agent(
        llm=llm, source_store=source_store, repository=repository
    ).run(trigger_reason="后台定期刷新")

    assert result.status == AgentRunStatus.UNCHANGED
    assert result.terminal_tool == "confirm_job_selection_unchanged"
    assert repository.current_result_revision() == 1


def test_recorded_job_history_is_read_on_demand_instead_of_injected(agent_setup):
    _, source_store, repository, _ = agent_setup
    sentinel = "HISTORICAL_COMPLETE_JD_SENTINEL " + ("历史正文 " * 200)
    recorded = repository.save_job_evidence(
        JobPostingEvidence.model_validate(_job_args()).model_copy(
            update={"jd_text": sentinel}
        )
    )
    responses = iter(
        [
            _tool_response(
                "read_recorded_job",
                {"job_evidence_id": recorded.id},
                "read-recorded",
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "本轮只核对历史记录",
                    "checks_completed": ["按需读取了一份完整历史 JD"],
                    "evidence_gaps": ["尚未执行新的来源搜索"],
                },
                "blocked",
            ),
        ]
    )
    captured_messages: list[list[dict]] = []

    def _respond(messages, **_kwargs):
        captured_messages.append(json.loads(json.dumps(messages, ensure_ascii=False)))
        return next(responses)

    llm = MagicMock()
    llm.chat_with_tools = _respond

    result = _agent(
        llm=llm, source_store=source_store, repository=repository
    ).run(trigger_reason="检查历史岗位")

    assert result.status == AgentRunStatus.BLOCKED
    first_messages, second_messages = captured_messages
    assert sentinel not in json.dumps(first_messages, ensure_ascii=False)
    assert sentinel in json.dumps(second_messages, ensure_ascii=False)


def test_platform_results_record_with_only_the_returned_opaque_ref(agent_setup):
    _, _source_store, repository, _ = agent_setup
    first = {
        **_job_args(),
        "evidence_kind": "platform_adapter",
        "source_evidence_id": 77,
        "source_job_id": "shared-1",
        "canonical_url": "https://jobs.example.com/shared/1",
        "title": "Agent 产品实习生",
    }
    second = {
        **_job_args(),
        "evidence_kind": "platform_adapter",
        "source_evidence_id": 77,
        "source_job_id": "shared-2",
        "canonical_url": "https://jobs.example.com/shared/2",
        "title": "大模型平台实习生",
    }
    for item in (first, second):
        item["platform_job_ref"] = platform_job_reference(
            source_evidence_id=item["source_evidence_id"],
            source_name=item["source_name"],
            source_job_id=item["source_job_id"],
            canonical_url=item["canonical_url"],
        )
    platform_tool = AgentTool(
        name="search_verified_official_jobs",
        description="test shared response",
        parameters={"type": "object", "properties": {}},
        handler=lambda _args, _execution: AgentToolResult(
            {"verified_jobs": [first, second]}
        ),
    )
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response("search_verified_official_jobs", {}, "platform"),
            _tool_response(
                "record_verified_platform_job",
                {
                    "platform_job_ref": first["platform_job_ref"],
                    "source_evidence_id": 999,
                    "company": "attacker override",
                },
                "malicious-platform-record",
            ),
            _tool_response(
                "record_verified_platform_job",
                {"platform_job_ref": first["platform_job_ref"]},
                "record-1",
            ),
            _tool_response(
                "record_verified_platform_job",
                {"platform_job_ref": second["platform_job_ref"]},
                "record-2",
            ),
            _tool_response(
                "publish_job_selection",
                {
                    "items": [
                        {
                            "job_evidence_id": 1,
                            "why_worth_attention": (
                                "寻找能参与真实 AI 产品交付的岗位，产品方向相关。"
                            ),
                            "grounding_quotes": _grounding(),
                            "concerns": [],
                            "unknowns": [],
                        },
                        {
                            "job_evidence_id": 2,
                            "why_worth_attention": (
                                "寻找能参与真实 AI 产品交付的岗位，平台方向相关。"
                            ),
                            "grounding_quotes": _grounding(),
                            "concerns": [],
                            "unknowns": [],
                        },
                    ],
                    "coverage_summary": "读取同一平台响应中的两个完整岗位。",
                    "evidence_gaps": [],
                },
                "publish",
            ),
        ]
    )
    verifier = MagicMock()
    agent = JobDiscoveryAgent(
        repository=repository,
        runner=AgentRunner(llm),
        candidate_evidence_loader=_candidate,
        source_tools=(platform_tool,),
        source_evidence_verifier=verifier,
    )

    result = agent.run(trigger_reason="读取共享平台响应")

    assert result.status == AgentRunStatus.PUBLISHED
    rejected_override = json.dumps(
        llm.chat_with_tools.call_args_list[2].args[0], ensure_ascii=False
    )
    assert "accepts only platform_job_ref" in rejected_override
    selection = repository.get_current_selection()
    assert selection is not None
    assert [item.job_evidence.title for item in selection.items] == [
        "Agent 产品实习生",
        "大模型平台实习生",
    ]
    assert verifier.verify_platform_job_page.call_count == 2


def test_generic_verifier_uses_complete_saved_body_and_rejects_unproved_metadata(
    agent_setup,
):
    _, source_store, _, context = agent_setup
    from offerguide.research_agents.sources import SourceScope

    evidence, _, _ = source_store.save_response(
        scope=SourceScope(
            run_id="run",
            agent_name="JobDiscoveryAgent",
            subject_kind="job_search",
            subject_id="current",
            subject_revision=context.revision,
        ),
        requested_url="https://jobs.example.com/positions/42",
        final_url="https://jobs.example.com/positions/42",
        title="岗位",
        media_type="text/plain",
        charset="utf-8",
        raw_content=("示例公司 AI 产品经理 上海 实习 " + _job_args()["jd_text"]).encode(),
        text_content="示例公司 AI 产品经理 上海 实习 " + _job_args()["jd_text"],
        http_status=200,
        purpose="job_detail",
    )
    verifier = StoredJobSourceEvidenceVerifier(source_store)
    resolved = verifier.resolve_generic_job_page(
        source_evidence_id=evidence.id,
        company="示例公司",
        title="AI 产品经理",
        location="上海",
        recruitment_type="实习",
        page_time_information=[],
        source_status="unknown",
        subject_kind="job_search",
        subject_id="current",
        subject_revision=context.revision,
    )
    assert resolved.jd_text == evidence.text_content
    assert resolved.canonical_url == evidence.canonical_url
    assert resolved.source_name == "jobs.example.com"
    assert resolved.source_job_id is None
    with pytest.raises(ValueError, match="not attached"):
        verifier.resolve_generic_job_page(
            source_evidence_id=evidence.id,
            company="示例公司",
            title="AI 产品经理",
            location="上海",
            recruitment_type="实习",
            page_time_information=[],
            source_status="unknown",
            subject_kind="job_search",
            subject_id="current",
            subject_revision=context.revision + 1,
        )
    with pytest.raises(ValueError, match="company"):
        verifier.resolve_generic_job_page(
            source_evidence_id=evidence.id,
            company="模型虚构的公司",
            title="AI 产品经理",
            location="上海",
            recruitment_type="实习",
            page_time_information=[],
            source_status="unknown",
            subject_kind="job_search",
            subject_id="current",
            subject_revision=context.revision,
        )
    with pytest.raises(ValueError, match="source_status='unknown'"):
        verifier.resolve_generic_job_page(
            source_evidence_id=evidence.id,
            company="示例公司",
            title="AI 产品经理",
            location="上海",
            recruitment_type="实习",
            page_time_information=[],
            source_status="open",
            subject_kind="job_search",
            subject_id="current",
            subject_revision=context.revision,
        )


def test_generic_verifier_rejects_a_search_card_as_complete_jd(agent_setup):
    _, source_store, _, context = agent_setup
    from offerguide.research_agents.sources import SourceScope

    evidence, _, _ = source_store.save_response(
        scope=SourceScope(
            run_id="thin-card",
            agent_name="JobDiscoveryAgent",
            subject_kind="job_search",
            subject_id="current",
            subject_revision=context.revision,
        ),
        requested_url="https://jobs.example.com/search",
        final_url="https://jobs.example.com/search",
        title="岗位列表",
        media_type="text/plain",
        charset="utf-8",
        raw_content=b"Example Company Data Role Shanghai",
        text_content="Example Company Data Role Shanghai",
        http_status=200,
        purpose="job_detail",
    )

    with pytest.raises(ValueError, match="identity/card text"):
        StoredJobSourceEvidenceVerifier(source_store).resolve_generic_job_page(
            source_evidence_id=evidence.id,
            company="Example Company",
            title="Data Role",
            location="Shanghai",
            recruitment_type=None,
            page_time_information=[],
            source_status="unknown",
            subject_kind="job_search",
            subject_id="current",
            subject_revision=context.revision,
        )
def test_failed_refresh_cannot_republish_history_and_404_marks_live_source_closed(
    agent_setup,
) -> None:
    _store, source_store, repository, context = agent_setup
    saved = repository.save_job_evidence(
        JobPostingEvidence.model_validate(_job_args()).model_copy(
            update={"source_status": "open"}
        )
    )
    assert saved.id is not None
    current = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=saved.id,
                why_worth_attention=(
                    "寻找能参与真实 AI 产品交付的岗位，此前核验时值得关注。"
                ),
                grounding_quotes=_grounding(),
            )
        ],
        candidate_evidence=_candidate(),
        coverage_summary="此前已经核验岗位。",
        evidence_gaps=["团队信息未知"],
    )
    llm = MagicMock()
    llm.chat_with_tools = MagicMock(
        side_effect=[
            _tool_response(
                "fetch_job_source",
                {"url": saved.canonical_url, "purpose": "refresh current job"},
                "fetch-404",
            ),
            _tool_response(
                "publish_job_selection",
                {
                    "items": [
                        {
                            "job_evidence_id": saved.id,
                            "why_worth_attention": "试图复用旧证据。",
                            "grounding_quotes": _grounding(),
                            "concerns": [],
                            "unknowns": [],
                        }
                    ],
                    "coverage_summary": "错误地声称已刷新。",
                    "evidence_gaps": [],
                },
                "invalid-publish",
            ),
            _tool_response(
                "report_job_discovery_blocked",
                {
                    "reason": "岗位详情页返回 404",
                    "checks_completed": ["重新请求岗位详情页"],
                    "evidence_gaps": ["无法重新读取完整 JD"],
                },
                "blocked",
            ),
        ]
    )

    result = _agent(
        llm=llm,
        source_store=source_store,
        repository=repository,
        source_status_code=404,
    ).run(trigger_reason="后台刷新")

    assert result.status == AgentRunStatus.BLOCKED
    assert repository.get_current_selection() == current
    live = repository.get_job_evidence(saved.id)
    assert live is not None
    assert live.source_status == "closed"
    assert live.checked_at >= saved.checked_at
    assert live.last_seen_at == saved.last_seen_at
