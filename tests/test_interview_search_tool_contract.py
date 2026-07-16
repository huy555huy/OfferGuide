from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from offerguide.agent_runtime.tools import ALL_TOOL_SCHEMAS
from offerguide.interview_research import agent as interview_agent
from offerguide.research_agents import AgentExecutionContext, AgentSubjectContext


@dataclass
class _SearchResult:
    def as_tool_result(self) -> dict[str, Any]:
        return {"clues": [], "results_are_unverified_clues": True}


class _SearchProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, query: str, **kwargs: Any) -> _SearchResult:
        self.calls.append({"query": query, **kwargs})
        return _SearchResult()


class _SavedSourceStore:
    def __init__(self) -> None:
        self.record = SimpleNamespace(
            id=42,
            provenance="web",
            title="已保存的公开面经",
            final_url="https://example.com/interview/42",
            text_content="完整公开正文",
        )
        self.attached: list[int] = []

    def get_latest_by_url(self, url: str) -> Any:
        return self.record if url == self.record.final_url else None

    def attach_existing(self, *, evidence_id: int, **_kwargs: Any) -> Any:
        self.attached.append(evidence_id)
        return self.record


def _execution(provider: _SearchProvider) -> AgentExecutionContext:
    return AgentExecutionContext(
        run_id="interview-search-contract",
        agent_name="InterviewResearchAgent",
        subject=AgentSubjectContext(
            subject_kind="interview_research",
            subject_id="application:1:workspace:1",
            subject_revision=1,
            result_revision=0,
            payload={},
        ),
        dependencies={
            "source_provider": provider,
            "session": SimpleNamespace(completed_searches=0, failed_searches=0),
        },
        started_at=1.0,
        iteration=1,
        tool_call_id="search-call",
    )


def _research_tool(name: str):
    return next(tool for tool in interview_agent._research_definition().tools if tool.name == name)


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "大模型推理实习 面经"},
        {"query": "大模型推理实习 面经", "domains": []},
    ],
)
def test_interview_search_allows_whole_web_without_domain_hints(
    arguments: dict[str, Any],
) -> None:
    provider = _SearchProvider()
    execution = _execution(provider)

    result = _research_tool("search_public_interview_experiences").handler(arguments, execution)

    assert not result.is_error
    assert provider.calls[0]["domains"] == []
    assert execution.dependencies["session"].completed_searches == 1
    assert "complete saved page body" in result.content["instruction"]
    assert "isolated original post" not in result.content["instruction"]


def test_interview_research_tools_leave_source_judgment_to_agent() -> None:
    search = _research_tool("search_public_interview_experiences")
    fetch = _research_tool("fetch_public_interview_experience")
    assess = _research_tool("assess_interview_experience")

    assert search.parameters["required"] == ["query"]
    domain_schema = search.parameters["properties"]["domains"]
    assert domain_schema["default"] == []
    assert "minItems" not in domain_schema
    assert "site, URL path, or page layout" in fetch.description
    assert "site, URL path, page layout, or company identity" in assess.description

    instructions = interview_agent._RESEARCH_INSTRUCTIONS.lower()
    assert "direct or adjacent" not in instructions
    assert "firsthand" not in instructions
    assert "isolates and saves" not in instructions
    assert "same company is only a search priority" in instructions
    assert "same\nquestion pool" in instructions


def test_fetch_result_reports_actual_content_scope() -> None:
    result = SimpleNamespace(
        status="saved",
        content_scope="unscoped_page",
        evidence=SimpleNamespace(
            id=7,
            title="公开页面",
            final_url="https://example.com/interview",
            text_content="完整正文",
        ),
    )
    session = SimpleNamespace(
        references=SimpleNamespace(source_ref=lambda _evidence_id: "source-1")
    )

    payload = interview_agent._fetch_result_for_model(result, session)

    assert payload["content_scope"] == "unscoped_page"


def test_rediscovered_public_body_can_be_reused_but_must_be_reassessed() -> None:
    store = _SavedSourceStore()
    session = interview_agent._RunState(references=interview_agent.InterviewRunReferences())
    execution = _execution(_SearchProvider())
    execution.dependencies.update({"source_store": store, "session": session})
    payload = {"clues": [{"url": store.record.final_url}]}

    interview_agent._expose_matching_saved_sources(payload, execution, session)
    source_ref = payload["clues"][0]["saved_source_ref"]
    result = interview_agent._reuse_source({"source_ref": source_ref}, execution)

    assert not result.is_error
    assert store.attached == [42]
    assert "assessing" in result.content["instruction"]
    assert "No earlier acceptance decision" in payload["clues"][0]["saved_source_instruction"]


def test_main_agent_describes_one_cross_company_similar_role_pool() -> None:
    schema = next(
        item["function"]
        for item in ALL_TOOL_SCHEMAS
        if item["function"]["name"] == "research_interview"
    )

    description = schema["description"]
    assert "similar roles across companies" in description
    assert "same company affects search priority only" in description.lower()
    assert "firsthand" not in description.lower()


def test_writer_does_not_treat_jd_or_word_overlap_as_candidate_evidence() -> None:
    instructions = interview_agent._WRITER_INSTRUCTIONS
    grounding_schema = interview_agent._ModelGroundingReference.model_json_schema()
    answer_schema = interview_agent._ModelQuestionAnswer.model_json_schema()

    assert "target_job\ndescribes the employer and role" in instructions
    assert "Word overlap is not entailment" in instructions
    assert "school date does not prove course" in instructions
    assert "Never fill the gap with a\nplausible interview persona" in instructions
    assert "can never prove a fact about the candidate" in grounding_schema["properties"][
        "kind"
    ]["description"]
    assert "explicit request for the user" in answer_schema["properties"]["answer"][
        "description"
    ]


def test_rejected_source_error_does_not_offer_its_questions_to_writer() -> None:
    repository = SimpleNamespace(
        load_subject=lambda *_args: SimpleNamespace(
            source_assessments=[
                {
                    "evidence_id": "7",
                    "is_current": True,
                    "assessment": {
                        "decision": "rejected",
                        "source_kind": "repost",
                        "rationale": "正文没有足够证据证明这些题来自真实面试",
                        "actual_questions": ["不应重新暴露给 writer 的题"],
                    },
                }
            ]
        )
    )
    references = SimpleNamespace(resolve_source=lambda source_ref: 7)
    execution = AgentExecutionContext(
        run_id="rejected-source-contract",
        agent_name="InterviewResearchAgent",
        subject=AgentSubjectContext(
            subject_kind="interview_research",
            subject_id="application:1:workspace:1",
            subject_revision=1,
            result_revision=0,
            payload={},
        ),
        dependencies={
            "repository": repository,
            "application_id": 1,
            "submitted_workspace_id": 1,
            "session": SimpleNamespace(references=references, staged_answers=[]),
        },
        started_at=1.0,
        iteration=1,
        tool_call_id="stage-call",
    )

    result = interview_agent._stage_validation_error(
        ValueError("source is not accepted for this application's interview question pool"),
        {
            "item": {
                "source_citations": [
                    {
                        "source_ref": "source-1",
                        "quote": "不应重新暴露给 writer 的题",
                    }
                ]
            }
        },
        execution,
    )

    assert result.content["error"] == "source is not accepted for this application"
    assert result.content["source_contracts"] == [
        {
            "source_ref": "source-1",
            "assessment_status": "rejected",
            "source_type": "repost",
        }
    ]
