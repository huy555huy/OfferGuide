from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from offerguide.interview_research import (
    EvidenceDocument,
    GroundingReference,
    InterviewAnswerSet,
    InterviewEvidenceError,
    InterviewQuestionAnswer,
    InterviewResearchAgent,
    InterviewResearchConflictError,
    InterviewResearchRepository,
    InterviewSourceAssessment,
    SourceCitation,
    SubmittedWorkspaceRequiredError,
    init_interview_research_schema,
)
from offerguide.interview_research.context import (
    InterviewRunReferences,
    build_writing_packet,
)
from offerguide.interview_research.repository import (
    CURRENT_INTERVIEW_EVIDENCE_CONTRACT_VERSION,
)
from offerguide.llm.client import LLMResponse, ToolCall
from offerguide.memory import Store
from offerguide.project_vault import insert as insert_project
from offerguide.research_agents import AgentRunStatus, SourceEvidenceStore, SourceScope

QUESTION = "为什么选择当前检索方案？"
SECOND_QUESTION = "如何判断这个功能值得继续投入？"


@pytest.fixture
def research_store(tmp_path) -> Store:
    store = Store(tmp_path / "interview-research.db")
    store.init_schema()
    SourceEvidenceStore(store).init_schema()
    init_interview_research_schema(store)
    return store


def _submission(
    store: Store,
    *,
    submitted: bool = True,
) -> tuple[int, int, int]:
    project = insert_project(
        store,
        title="检索增强项目",
        mainstream_direction="LLM 应用",
        project_task="让用户从真实文档检索答案",
        my_work="完成数据接入、检索和回答链路",
        evidence="仓库和演示记录",
        do_not_claim="没有线上用户规模",
    )
    job_snapshot = {
        "job_id": 0,
        "company": "示例科技",
        "title": "AI 产品实习生",
        "raw_text": "完整 JD：负责模型应用产品与用户研究。",
        "url": "https://jobs.example.com/role/1",
    }
    frozen_context = {
        "job": {
            "job_id": 0,
            "company": "示例科技",
            "title": "AI 产品实习生",
            "jd_text": job_snapshot["raw_text"],
        },
        "project_facts": [
            {
                "project_id": project.id,
                "title": project.title,
                "facts": ["我的真实工作: 完成数据接入、检索和回答链路"],
                "do_not_claim": ["没有线上用户规模"],
            }
        ],
    }
    resume_document = {
        "header": {"name": "候选人", "phone": "13800000000"},
        "sections": [
            {
                "title": "项目经历",
                "entries": [
                    {
                        "rows": [{"left": "检索增强项目", "right": "2025"}],
                        "blocks": [
                            {
                                "kind": "bullet",
                                "content": "完成数据接入、检索和回答链路",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    apply_pack = {
        "assistant": {"message": "您好，我希望申请该岗位。"},
        "preparation_notes": [{"claim": "了解 LangChain", "note": "准备组件边界与替代方案"}],
    }
    with store.connect() as conn:
        job_id = int(
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, location, "
                "raw_text, content_hash) VALUES ('manual', 'role-1', ?, ?, ?, '', ?, ?) "
                "RETURNING id",
                (
                    job_snapshot["url"],
                    job_snapshot["title"],
                    job_snapshot["company"],
                    job_snapshot["raw_text"],
                    "job-content-hash",
                ),
            ).fetchone()[0]
        )
        job_snapshot["job_id"] = job_id
        frozen_context["job"]["job_id"] = job_id
        application_id = int(
            conn.execute(
                "INSERT INTO applications(job_id, status, applied_at) VALUES (?, ?, ?) "
                "RETURNING id",
                (
                    job_id,
                    "applied" if submitted else "considered",
                    2461234.5 if submitted else None,
                ),
            ).fetchone()[0]
        )
        workspace_id = int(
            conn.execute(
                "INSERT INTO resume_workspaces("
                "application_id, status, job_snapshot_json, master_source_sha256, "
                "context_json, resume_document_json, pdf_path, pdf_sha256, "
                "apply_pack_json, submitted_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    application_id,
                    "submitted" if submitted else "draft",
                    json.dumps(job_snapshot, ensure_ascii=False),
                    "a" * 64,
                    json.dumps(frozen_context, ensure_ascii=False),
                    json.dumps(resume_document, ensure_ascii=False),
                    "/tmp/submitted-resume.pdf",
                    "b" * 64,
                    json.dumps(apply_pack, ensure_ascii=False),
                    2461234.5 if submitted else None,
                ),
            ).fetchone()[0]
        )
        if submitted:
            conn.execute(
                "INSERT INTO application_events(application_id, kind, source, payload_json) "
                "VALUES (?, 'submitted', 'manual', ?)",
                (application_id, json.dumps({"workspace_id": workspace_id})),
            )
    return application_id, workspace_id, project.id


def _scope(application_id: int, workspace_id: int, revision: int, *, run: str) -> SourceScope:
    return SourceScope(
        run_id=run,
        agent_name="interview_research_agent",
        subject_kind="interview_research",
        subject_id=f"application:{application_id}:workspace:{workspace_id}",
        subject_revision=revision,
    )


def _save_source(
    store: Store,
    *,
    application_id: int,
    workspace_id: int,
    subject_revision: int,
    text: str,
    title: str = "真实面经",
    source_url: str | None = None,
    run: str = "saved-source",
) -> tuple[Any, EvidenceDocument]:
    source_store = SourceEvidenceStore(store)
    record = source_store.save_user_provided(
        scope=_scope(application_id, workspace_id, subject_revision, run=run),
        text=text,
        title=title,
        source_url=source_url,
        purpose="test interview experience",
    )
    return record, EvidenceDocument(
        evidence_id=str(record.id),
        url=record.final_url,
        title=record.title,
        text=record.text_content,
        fetched_at=record.fetched_at,
        complete=True,
        attached_to_subject=True,
    )


def _assess_source(
    repository: InterviewResearchRepository,
    *,
    application_id: int,
    workspace_id: int,
    subject_token: str,
    evidence: EvidenceDocument,
    question: str = QUESTION,
    decision: str = "accepted",
) -> InterviewSourceAssessment:
    assessment = InterviewSourceAssessment(
        decision=decision,
        source_kind="community_interview_post",
        rationale="原帖明确记录了与当前岗位相似的真实面试",
        actual_questions=[question],
    )
    repository.assess_source(
        application_id,
        workspace_id,
        evidence=evidence,
        assessment=assessment,
        expected_subject_token=subject_token,
    )
    return assessment


def _answer_set(
    evidence_id: int,
    *,
    question: str = QUESTION,
    citation_quote: str | None = None,
) -> InterviewAnswerSet:
    return InterviewAnswerSet(
        answers=[
            InterviewQuestionAnswer(
                question=question,
                answer="我会先说明目标用户和约束，再用检索质量与失败案例解释方案取舍。",
                source_citations=[
                    SourceCitation(
                        evidence_id=str(evidence_id),
                        quote=citation_quote or question,
                    )
                ],
                grounding=[
                    GroundingReference(
                        kind="submitted_resume",
                        quote="检索增强项目",
                    )
                ],
            )
        ]
    )


def _publish_accepted_answer(
    store: Store,
    *,
    application_id: int,
    workspace_id: int,
    question: str = QUESTION,
) -> tuple[InterviewResearchRepository, EvidenceDocument, InterviewAnswerSet]:
    repository = InterviewResearchRepository(store)
    subject = repository.ensure_subject(application_id, workspace_id)
    record, evidence = _save_source(
        store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"这是一份真实面试记录。面试官问：{question}",
        source_url="https://example.com/interview/published",
    )
    _assess_source(
        repository,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_token=subject.subject_token,
        evidence=evidence,
        question=question,
    )
    answer_set = _answer_set(record.id, question=question)
    repository.publish(
        application_id,
        workspace_id,
        answer_set=answer_set,
        expected_subject_token=subject.subject_token,
        expected_result_revision=0,
        evidence_loader=lambda evidence_id: evidence if evidence_id == str(record.id) else None,
    )
    return repository, evidence, answer_set


def test_answer_set_has_only_answered_or_not_found_results() -> None:
    not_found = InterviewAnswerSet(status="not_found")
    assert not_found.answers == []

    with pytest.raises(ValidationError, match="requires at least one"):
        InterviewAnswerSet(status="answered")
    with pytest.raises(ValidationError, match="cannot contain generated questions"):
        InterviewAnswerSet(
            status="not_found",
            answers=[
                InterviewQuestionAnswer(
                    question=QUESTION,
                    answer="回答",
                    source_citations=[SourceCitation(evidence_id="1", quote=QUESTION)],
                )
            ],
        )


@pytest.mark.parametrize("legacy_field", ["blocks", "research_gaps"])
def test_answer_set_rejects_legacy_material_fields(legacy_field: str) -> None:
    payload: dict[str, Any] = {"status": "not_found", legacy_field: []}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        InterviewAnswerSet.model_validate(payload)


@pytest.mark.parametrize(
    "source_kind",
    ["community_interview_post", "repost_with_original_questions", "new_kind"],
)
def test_source_kind_is_open_metadata_not_a_question_pool_gate(
    source_kind: str,
) -> None:
    assessment = InterviewSourceAssessment(
        decision="accepted",
        source_kind=source_kind,
        rationale="Agent 完整读取正文后确认这是相似岗位的真实面经",
        actual_questions=[QUESTION],
    )

    assert assessment.source_kind == source_kind
    assert assessment.decision == "accepted"


def test_source_assessment_schema_has_decision_without_closed_source_kinds() -> None:
    properties = InterviewSourceAssessment.model_json_schema()["properties"]

    assert properties["decision"]["enum"] == ["accepted", "rejected"]
    assert "relationship" not in properties
    assert "enum" not in properties["source_kind"]


def test_rejected_source_can_preserve_questions_for_traceability() -> None:
    assessment = InterviewSourceAssessment(
        decision="rejected",
        source_kind="mixed_page",
        rationale="正文有问题文本，但 Agent 判断它不是可用于当前题池的真实面经",
        actual_questions=[QUESTION],
    )

    assert assessment.actual_questions == [QUESTION]


@pytest.mark.parametrize(
    ("legacy_relationship", "expected_decision"),
    [
        ("direct", "accepted"),
        ("adjacent", "accepted"),
        ("accepted", "accepted"),
        ("rejected", "rejected"),
    ],
)
def test_legacy_source_relationship_is_read_as_decision(
    legacy_relationship: str,
    expected_decision: str,
) -> None:
    assessment = InterviewSourceAssessment.model_validate(
        {
            "relationship": legacy_relationship,
            "source_kind": "legacy_source_kind",
            "rationale": "旧版本已接受的相似岗位面经",
            "actual_questions": [QUESTION],
        }
    )

    assert assessment.decision == expected_decision
    assert "relationship" not in assessment.model_dump()


def test_grounding_reference_keeps_candidate_identity_explicit() -> None:
    with pytest.raises(ValidationError, match="requires a reference"):
        GroundingReference(kind="project_vault", quote="完成检索链路")
    with pytest.raises(ValidationError, match="does not accept"):
        GroundingReference(
            kind="submitted_resume",
            reference="project-1",
            quote="检索增强项目",
        )


def test_research_requires_the_matching_submitted_workspace(research_store: Store) -> None:
    application_id, workspace_id, _ = _submission(research_store, submitted=False)
    repository = InterviewResearchRepository(research_store)
    with pytest.raises(SubmittedWorkspaceRequiredError):
        repository.ensure_subject(application_id, workspace_id)


def test_source_assessment_quotes_must_exist_in_complete_attached_body(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    _record, evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"面试官问：{QUESTION}",
    )
    assessment = InterviewSourceAssessment(
        decision="accepted",
        source_kind="community_interview_post",
        rationale="真实面试原帖",
        actual_questions=["原文不存在的问题？"],
    )

    with pytest.raises(InterviewEvidenceError, match="not in the source body"):
        repository.assess_source(
            application_id,
            workspace_id,
            evidence=evidence,
            assessment=assessment,
            expected_subject_token=subject.subject_token,
        )
    with pytest.raises(InterviewEvidenceError, match="partially read"):
        repository.assess_source(
            application_id,
            workspace_id,
            evidence=evidence.model_copy(update={"complete": False}),
            assessment=assessment.model_copy(update={"actual_questions": [QUESTION]}),
            expected_subject_token=subject.subject_token,
        )
    with pytest.raises(InterviewEvidenceError, match="not attached"):
        repository.assess_source(
            application_id,
            workspace_id,
            evidence=evidence.model_copy(update={"attached_to_subject": False}),
            assessment=assessment.model_copy(update={"actual_questions": [QUESTION]}),
            expected_subject_token=subject.subject_token,
        )


def test_repository_allows_minimal_display_normalization_but_keeps_exact_citation(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    record, evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"面试官问：{QUESTION} 随后继续讨论检索召回率。",
        source_url="https://example.com/interview/exact-question",
    )
    _assess_source(
        repository,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_token=subject.subject_token,
        evidence=evidence,
    )

    non_question_quote = _answer_set(
        record.id,
        question="继续讨论检索召回率",
        citation_quote="继续讨论检索召回率",
    )
    with pytest.raises(InterviewEvidenceError, match="not an actual question"):
        repository.publish(
            application_id,
            workspace_id,
            answer_set=non_question_quote,
            expected_subject_token=subject.subject_token,
            expected_result_revision=0,
            evidence_loader=lambda _evidence_id: evidence,
        )

    normalized_display = _answer_set(
        record.id,
        question="为什么选择该岗位所需的检索方案？",
        citation_quote=QUESTION,
    )
    published = repository.publish(
        application_id,
        workspace_id,
        answer_set=normalized_display,
        expected_subject_token=subject.subject_token,
        expected_result_revision=0,
        evidence_loader=lambda _evidence_id: evidence,
    )
    assert published.answer_set.answers[0].question == "为什么选择该岗位所需的检索方案？"
    assert published.answer_set.answers[0].source_citations[0].quote == QUESTION
    assert published.used_evidence_ids == [str(record.id)]


def test_similar_role_company_questions_can_share_one_normalized_answer(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    target_company_question = "为什么选择示例科技？"
    other_company_question = "为什么选择另一家公司？"
    target_record, target_evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"示例科技相似岗位面试官问：{target_company_question}",
        source_url="https://example.com/interview/target-company-question",
        run="target-company-question",
    )
    other_record, other_evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"另一家公司相似岗位面试官问：{other_company_question}",
        source_url="https://example.com/interview/other-company-question",
        run="other-company-question",
    )
    _assess_source(
        repository,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_token=subject.subject_token,
        evidence=target_evidence,
        question=target_company_question,
        decision="accepted",
    )
    _assess_source(
        repository,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_token=subject.subject_token,
        evidence=other_evidence,
        question=other_company_question,
        decision="accepted",
    )
    pooled = InterviewAnswerSet(
        answers=[
            InterviewQuestionAnswer(
                question="为什么选择当前目标公司？",
                answer="我会结合岗位职责、团队方向与自己的项目证据说明选择。",
                source_citations=[
                    SourceCitation(
                        evidence_id=str(target_record.id), quote=target_company_question
                    ),
                    SourceCitation(evidence_id=str(other_record.id), quote=other_company_question),
                ],
                grounding=[GroundingReference(kind="submitted_resume", quote="检索增强项目")],
            )
        ]
    )
    evidence_by_id = {
        str(target_record.id): target_evidence,
        str(other_record.id): other_evidence,
    }

    published = repository.publish(
        application_id,
        workspace_id,
        answer_set=pooled,
        expected_subject_token=subject.subject_token,
        expected_result_revision=0,
        evidence_loader=evidence_by_id.get,
    )

    answer = published.answer_set.answers[0]
    assert answer.question == "为什么选择当前目标公司？"
    assert {citation.quote for citation in answer.source_citations} == {
        target_company_question,
        other_company_question,
    }
    assert set(published.used_evidence_ids) == {
        str(target_record.id),
        str(other_record.id),
    }


def test_rejected_job_posting_cannot_be_published_as_interview_question(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    record, evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"岗位说明：{QUESTION}",
        title="岗位招聘页",
        source_url="https://example.com/jobs/not-an-interview",
    )
    repository.assess_source(
        application_id,
        workspace_id,
        evidence=evidence,
        assessment=InterviewSourceAssessment(
            decision="rejected",
            source_kind="job_posting",
            rationale="这是岗位说明，不是面试经历",
            actual_questions=[QUESTION],
        ),
        expected_subject_token=subject.subject_token,
    )

    current_subject = repository.load_subject(application_id, workspace_id)
    writer_packet = build_writing_packet(
        current_subject,
        InterviewRunReferences.for_subject(current_subject),
        current_evidence_contract_version=(CURRENT_INTERVIEW_EVIDENCE_CONTRACT_VERSION),
    )
    assert writer_packet["accepted_interview_experiences"] == []

    with pytest.raises(InterviewEvidenceError, match="not accepted"):
        repository.publish(
            application_id,
            workspace_id,
            answer_set=_answer_set(record.id),
            expected_subject_token=subject.subject_token,
            expected_result_revision=0,
            evidence_loader=lambda _evidence_id: evidence,
        )


def test_legacy_answer_relationship_is_ignored_not_used_as_a_pool_gate(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    record, evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"另一家公司相似岗位面试官问：{QUESTION}",
        source_url="https://example.com/interview/adjacent",
    )
    _assess_source(
        repository,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_token=subject.subject_token,
        evidence=evidence,
        decision="accepted",
    )

    legacy_payload = _answer_set(record.id).model_dump(mode="json")
    legacy_payload["answers"][0]["relationship"] = "direct"
    compatible = InterviewAnswerSet.model_validate(legacy_payload)
    assert "relationship" not in compatible.answers[0].model_dump()

    published = repository.publish(
        application_id,
        workspace_id,
        answer_set=compatible,
        expected_subject_token=subject.subject_token,
        expected_result_revision=0,
        evidence_loader=lambda _evidence_id: evidence,
    )
    assert published.answer_set.answers[0].question == QUESTION


def test_candidate_grounding_must_be_exact_frozen_evidence(research_store: Store) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    record, evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"面试官问：{QUESTION}",
        source_url="https://example.com/interview/grounding",
    )
    _assess_source(
        repository,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_token=subject.subject_token,
        evidence=evidence,
    )
    invalid = _answer_set(record.id)
    invalid.answers[0].grounding = [
        GroundingReference(kind="submitted_resume", quote="不存在的候选人经历")
    ]

    with pytest.raises(InterviewEvidenceError, match="quote is not present"):
        repository.publish(
            application_id,
            workspace_id,
            answer_set=invalid,
            expected_subject_token=subject.subject_token,
            expected_result_revision=0,
            evidence_loader=lambda _evidence_id: evidence,
        )


def test_context_refresh_keeps_previous_answer_set_until_atomic_replacement(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository, _evidence, answer_set = _publish_accepted_answer(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
    )
    before = repository.load_subject(application_id, workspace_id)

    refreshed = repository.add_context_update(
        application_id,
        workspace_id,
        kind="user_provided_source",
        content={"evidence_id": 999, "title": "刷新中的新面经"},
        expected_revision=before.context_revision,
    )

    assert refreshed.current_material_is_stale is True
    assert refreshed.current_material is not None
    assert refreshed.current_material.answer_set == answer_set
    assert refreshed.current_material.result_revision == 1


def test_concurrent_result_revision_cannot_overwrite_current_answer_set(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository, evidence, answer_set = _publish_accepted_answer(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
    )
    current = repository.load_subject(application_id, workspace_id)

    with pytest.raises(InterviewResearchConflictError, match="current interview material"):
        repository.publish(
            application_id,
            workspace_id,
            answer_set=answer_set,
            expected_subject_token=current.subject_token,
            expected_result_revision=0,
            evidence_loader=lambda _evidence_id: evidence,
        )


@dataclass(frozen=True)
class _SearchResult:
    query: str
    domains: tuple[str, ...]

    def as_tool_result(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "domains": list(self.domains),
            "results_are_unverified_clues": True,
            "clues": [],
        }


class _StubPublicProvider:
    def __init__(
        self,
        evidence_store: SourceEvidenceStore,
        outcomes: list[_SearchResult | Exception] | None = None,
    ) -> None:
        self.evidence_store = evidence_store
        self.outcomes = list(outcomes or [])
        self.search_calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        scope: SourceScope,
        domains: list[str],
        max_results: int,
    ) -> _SearchResult:
        self.search_calls.append(
            {
                "query": query,
                "scope": scope,
                "domains": list(domains),
                "max_results": max_results,
            }
        )
        if not self.outcomes:
            raise AssertionError("unexpected public search")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def fetch(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unexpected public fetch")


class _SequenceLLM:
    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        review_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.responses = iter(responses)
        self.review_results = iter(review_results or [])
        self.calls: list[dict[str, Any]] = []
        self.review_calls: list[dict[str, Any]] = []

    def chat_with_tools(self, *args: Any, **kwargs: Any) -> LLMResponse:
        messages = args[0] if args else kwargs.get("messages")
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        return next(self.responses)

    def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        messages = args[0] if args else kwargs.get("messages")
        self.review_calls.append({"messages": copy.deepcopy(messages), **kwargs})
        try:
            result = next(self.review_results)
        except StopIteration:
            result = {"approved": True, "issues": []}
        return LLMResponse(
            content=json.dumps(result, ensure_ascii=False),
            model="test-reviewer",
        )


def _tool_response(name: str, arguments: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content="",
        model="test",
        tool_calls=[
            ToolCall(
                id=f"call-{name}",
                name=name,
                arguments=arguments,
                arguments_raw=json.dumps(arguments, ensure_ascii=False),
            )
        ],
    )


def _tool_names(call: dict[str, Any]) -> set[str]:
    return {str(tool["function"]["name"]) for tool in call["tools"]}


def _model_answer(
    source_ref: str,
    *,
    question: str = QUESTION,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": "我会先定义目标和约束，再结合检索增强项目解释方案取舍。",
        "source_citations": [{"source_ref": source_ref, "quote": question}],
        "grounding": [{"kind": "submitted_resume", "quote": "检索增强项目"}],
    }


def test_agent_research_is_private_and_writer_publishes_only_real_question(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    record, _evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"真实面试记录。面试官问：{QUESTION}",
        source_url="https://example.com/interview/agent-happy-path",
    )
    assessment = {
        "decision": "accepted",
        "source_kind": "community_interview_post_v2",
        "rationale": "原帖记录了当前岗位的真实面试",
        "actual_questions": [QUESTION],
    }
    llm = _SequenceLLM(
        [
            _tool_response("read_interview_experience", {"source_ref": "source-1"}),
            _tool_response(
                "assess_interview_experience",
                {"source_ref": "source-1", "assessment": assessment},
            ),
            _tool_response("write_answers_to_real_interview_questions", {}),
            _tool_response(
                "stage_interview_question_answers",
                {"items": [_model_answer("source-1")], "replace_all": True},
            ),
            _tool_response("review_interview_question_answers", {}),
            _tool_response("publish_interview_question_answers", {}),
        ]
    )
    source_store = SourceEvidenceStore(research_store)
    agent = InterviewResearchAgent(
        store=research_store,
        llm=llm,
        source_provider=_StubPublicProvider(source_store),  # type: ignore[arg-type]
        source_store=source_store,
    )

    result = agent.run(
        application_id=application_id,
        submitted_workspace_id=workspace_id,
    )

    assert result.status == AgentRunStatus.PUBLISHED
    current = repository.get_current(application_id, workspace_id)
    assert current is not None
    assert current.answer_set.status == "answered"
    assert current.answer_set.answers[0].question == QUESTION
    assert current.answer_set.answers[0].source_citations[0].evidence_id == str(record.id)

    research_context = json.loads(llm.calls[0]["messages"][1]["content"])
    assert set(research_context["authoritative_context"]) == {
        "target_job",
        "saved_sources",
    }
    assert research_context["authoritative_context"]["saved_sources"][0]["source_ref"] == "source-1"
    assert "reference" not in research_context["authoritative_context"]["saved_sources"][0]
    research_serialized = json.dumps(research_context, ensure_ascii=False)
    for private_value in (
        "检索增强项目",
        "完成数据接入、检索和回答链路",
        "您好，我希望申请该岗位。",
        "LangChain",
        "13800000000",
    ):
        assert private_value not in research_serialized

    research_tools = _tool_names(llm.calls[0])
    assert "fetch_authenticated_interview_source" not in research_tools
    assert all("authenticated" not in name for name in research_tools)

    writer_context = json.loads(llm.calls[3]["messages"][1]["content"])
    writer_serialized = json.dumps(writer_context, ensure_ascii=False)
    assert "检索增强项目" in writer_serialized
    assert QUESTION in writer_serialized
    assert "source-1" in writer_serialized
    assert "community_interview_post_v2" in writer_serialized


def test_independent_grounding_review_blocks_draft_until_corrected(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    _record, _evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"真实面试记录。面试官问：{QUESTION}",
        source_url="https://example.com/interview/review-gate",
    )
    unsupported = _model_answer("source-1")
    unsupported["answer"] = "我已经确定可以长期在北京工作。"
    corrected = _model_answer("source-1")
    corrected["answer"] = "现有候选人材料没有说明工作地点意愿，需要你确认后再补充。"
    corrected["grounding"] = []
    assessment = {
        "decision": "accepted",
        "source_kind": "community_interview_post",
        "rationale": "真实面试记录",
        "actual_questions": [QUESTION],
    }
    llm = _SequenceLLM(
        [
            _tool_response("read_interview_experience", {"source_ref": "source-1"}),
            _tool_response(
                "assess_interview_experience",
                {"source_ref": "source-1", "assessment": assessment},
            ),
            _tool_response("write_answers_to_real_interview_questions", {}),
            _tool_response(
                "stage_interview_question_answers",
                {"items": [unsupported], "replace_all": True},
            ),
            _tool_response("review_interview_question_answers", {}),
            _tool_response(
                "stage_interview_question_answers",
                {"items": [corrected], "positions": [0]},
            ),
            _tool_response("review_interview_question_answers", {}),
            _tool_response("publish_interview_question_answers", {}),
        ],
        review_results=[
            {
                "approved": False,
                "issues": [
                    {
                        "position": 0,
                        "unsupported_claims": ["我已经确定可以长期在北京工作"],
                        "required_action": "请用户确认地点意愿，不要从 JD 推断。",
                    }
                ],
            },
            {"approved": True, "issues": []},
        ],
    )
    source_store = SourceEvidenceStore(research_store)
    agent = InterviewResearchAgent(
        store=research_store,
        llm=llm,
        source_provider=_StubPublicProvider(source_store),  # type: ignore[arg-type]
        source_store=source_store,
    )

    result = agent.run(
        application_id=application_id,
        submitted_workspace_id=workspace_id,
    )

    assert result.status == AgentRunStatus.PUBLISHED
    published = repository.get_current(application_id, workspace_id)
    assert published is not None
    assert published.answer_set.answers[0].answer == corrected["answer"]
    assert len(llm.review_calls) == 2
    review_packet = json.loads(llm.review_calls[0]["messages"][1]["content"])
    assert "complete_candidate_evidence" in review_packet
    correction_turn = json.dumps(llm.calls[5]["messages"], ensure_ascii=False)
    assert "我已经确定可以长期在北京工作" in correction_turn
    assert "matching zero-based positions" in correction_turn


def test_duplicate_source_titles_use_stable_source_refs(research_store: Store) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    repository = InterviewResearchRepository(research_store)
    subject = repository.ensure_subject(application_id, workspace_id)
    first_record, first_evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"第一份真实面试。面试官问：{QUESTION}",
        title="AI 产品面经",
        source_url="https://example.com/interview/duplicate-title-1",
        run="duplicate-one",
    )
    second_record, second_evidence = _save_source(
        research_store,
        application_id=application_id,
        workspace_id=workspace_id,
        subject_revision=subject.agent_subject_revision,
        text=f"第二份真实面试。面试官问：{SECOND_QUESTION}",
        title="AI 产品面经",
        source_url="https://example.com/interview/duplicate-title-2",
        run="duplicate-two",
    )
    for evidence, question in (
        (first_evidence, QUESTION),
        (second_evidence, SECOND_QUESTION),
    ):
        _assess_source(
            repository,
            application_id=application_id,
            workspace_id=workspace_id,
            subject_token=subject.subject_token,
            evidence=evidence,
            question=question,
        )

    current_subject = repository.load_subject(application_id, workspace_id)
    references = InterviewRunReferences.for_subject(current_subject)
    first_ref = references.existing_source_ref(first_record.id)
    second_ref = references.existing_source_ref(second_record.id)
    assert first_ref is not None and second_ref is not None and first_ref != second_ref

    provider = _StubPublicProvider(
        SourceEvidenceStore(research_store),
        [_SearchResult("AI 产品实习生 面经", ("example.com",))],
    )
    llm = _SequenceLLM(
        [
            _tool_response(
                "search_public_interview_experiences",
                {
                    "query": "AI 产品实习生 面经",
                    "domains": ["example.com"],
                },
            ),
            _tool_response("write_answers_to_real_interview_questions", {}),
            _tool_response(
                "stage_interview_question_answers",
                {
                    "items": [
                        _model_answer(
                            second_ref,
                            question=SECOND_QUESTION,
                        )
                    ],
                    "replace_all": True,
                },
            ),
            _tool_response("review_interview_question_answers", {}),
            _tool_response("publish_interview_question_answers", {}),
        ]
    )
    agent = InterviewResearchAgent(
        store=research_store,
        llm=llm,
        source_provider=provider,  # type: ignore[arg-type]
        source_store=provider.evidence_store,
    )

    result = agent.run(
        application_id=application_id,
        submitted_workspace_id=workspace_id,
    )

    assert result.status == AgentRunStatus.PUBLISHED
    assert len(provider.search_calls) == 1
    published = repository.get_current(application_id, workspace_id)
    assert published is not None
    citation = published.answer_set.answers[0].source_citations[0]
    assert citation.evidence_id == str(second_record.id)
    assert citation.evidence_id != str(first_record.id)
    writer_context = json.loads(llm.calls[2]["messages"][1]["content"])
    accepted = writer_context["authoritative_context"]["accepted_interview_experiences"]
    assert {item["source_ref"] for item in accepted} == {first_ref, second_ref}
    assert {item["title"] for item in accepted} == {"AI 产品面经"}
    research_sources = json.loads(llm.calls[0]["messages"][1]["content"])["authoritative_context"][
        "saved_sources"
    ]
    assert all("accepted_interview_experience" in item for item in research_sources)
    assert all(
        "relationship" not in item["accepted_interview_experience"] for item in research_sources
    )
    research_instructions = str(llm.calls[0]["messages"][0]["content"])
    assert "One accepted real question is enough" not in research_instructions
    assert "same company is only a search priority" in research_instructions.lower()
    writer_instructions = str(llm.calls[2]["messages"][0]["content"])
    assert "one similar-role\nquestion pool" in writer_instructions


def test_not_found_requires_successful_search_and_is_persisted(
    research_store: Store,
) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    provider = _StubPublicProvider(
        SourceEvidenceStore(research_store),
        [_SearchResult("示例科技 AI 产品实习生 面经", ("nowcoder.com",))],
    )
    llm = _SequenceLLM(
        [
            _tool_response("publish_interview_experiences_not_found", {}),
            _tool_response(
                "search_public_interview_experiences",
                {
                    "query": "示例科技 AI 产品实习生 面经",
                    "domains": ["nowcoder.com"],
                },
            ),
            _tool_response("publish_interview_experiences_not_found", {}),
        ]
    )
    agent = InterviewResearchAgent(
        store=research_store,
        llm=llm,
        source_provider=provider,  # type: ignore[arg-type]
        source_store=provider.evidence_store,
    )

    result = agent.run(
        application_id=application_id,
        submitted_workspace_id=workspace_id,
    )

    assert result.status == AgentRunStatus.PUBLISHED
    assert any("complete at least one public search" in item for item in result.tool_errors)
    current = agent.repository.get_current(application_id, workspace_id)
    assert current is not None
    assert current.answer_set.status == "not_found"
    assert current.answer_set.answers == []
    assert current.used_evidence_ids == []
    assert len(provider.search_calls) == 1


def test_all_search_failures_are_blocked_not_not_found(research_store: Store) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    provider = _StubPublicProvider(
        SourceEvidenceStore(research_store),
        [RuntimeError("search transport unavailable")],
    )
    llm = _SequenceLLM(
        [
            _tool_response(
                "search_public_interview_experiences",
                {
                    "query": "示例科技 AI 产品实习生 面经",
                    "domains": ["nowcoder.com"],
                },
            ),
            _tool_response(
                "report_public_search_unavailable",
                {"reason": "公开搜索服务不可用"},
            ),
        ]
    )
    agent = InterviewResearchAgent(
        store=research_store,
        llm=llm,
        source_provider=provider,  # type: ignore[arg-type]
        source_store=provider.evidence_store,
    )

    result = agent.run(
        application_id=application_id,
        submitted_workspace_id=workspace_id,
    )

    assert result.status == AgentRunStatus.BLOCKED
    assert result.terminal_tool == "report_public_search_unavailable"
    assert agent.repository.get_current(application_id, workspace_id) is None


def test_completed_search_cannot_be_reported_as_unavailable(research_store: Store) -> None:
    application_id, workspace_id, _ = _submission(research_store)
    provider = _StubPublicProvider(
        SourceEvidenceStore(research_store),
        [_SearchResult("相似岗位 面经", ("nowcoder.com",))],
    )
    llm = _SequenceLLM(
        [
            _tool_response(
                "search_public_interview_experiences",
                {"query": "相似岗位 面经", "domains": ["nowcoder.com"]},
            ),
            _tool_response(
                "report_public_search_unavailable",
                {"reason": "错误地把空结果当故障"},
            ),
            _tool_response("publish_interview_experiences_not_found", {}),
        ]
    )
    agent = InterviewResearchAgent(
        store=research_store,
        llm=llm,
        source_provider=provider,  # type: ignore[arg-type]
        source_store=provider.evidence_store,
    )

    result = agent.run(
        application_id=application_id,
        submitted_workspace_id=workspace_id,
    )

    assert result.status == AgentRunStatus.PUBLISHED
    assert any("publish not_found" in item for item in result.tool_errors)
    current = agent.repository.get_current(application_id, workspace_id)
    assert current is not None and current.answer_set.status == "not_found"
