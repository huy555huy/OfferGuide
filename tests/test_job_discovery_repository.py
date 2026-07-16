from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from offerguide import Store
from offerguide.research_agents.job_discovery.models import (
    CandidateEvidence,
    CandidateEvidenceDocument,
    JobPostingEvidence,
    JobSelectionItem,
    JobSelectionSet,
    SelectionGroundingQuote,
)
from offerguide.research_agents.job_discovery.repository import (
    JobDiscoveryEvidenceError,
    JobDiscoveryRepository,
    JobDiscoveryRevisionConflict,
    canonicalize_job_url,
    init_job_discovery_schema,
)


@pytest.fixture
def repository(tmp_path) -> JobDiscoveryRepository:
    store = Store(tmp_path / "job-discovery.db")
    store.init_schema()
    init_job_discovery_schema(store)
    return JobDiscoveryRepository(store)


def _candidate_evidence(text: str = "完整简历内容") -> CandidateEvidence:
    return CandidateEvidence(
        documents=[
            CandidateEvidenceDocument(
                reference="master-resume:confirmed",
                kind="master_resume",
                title="确认后的主简历",
                text=text,
            )
        ]
    )


def _grounding() -> list[Any]:
    return [
        {"reference": "master-resume:confirmed", "quote": "完整简历内容"}
    ]


def _complete_jd(marker: str = "当前版本") -> str:
    return (
        f"{marker}：负责理解真实用户问题，和产品、工程团队共同完成方案交付。\n"
        "岗位需要能够分析复杂需求，推动实现、验证效果并根据反馈持续改进。"
    )


def test_selection_grounding_rejects_punctuation_as_evidence() -> None:
    with pytest.raises(ValueError, match="too weak"):
        SelectionGroundingQuote(reference="search-context:intent", quote="。")


def _job(
    *,
    jd_text: str = _complete_jd(),
    source_evidence_id: int = 11,
) -> JobPostingEvidence:
    return JobPostingEvidence(
        evidence_kind="platform_adapter",
        source_evidence_id=source_evidence_id,
        source_name="official-example",
        source_job_id="position-42",
        canonical_url="https://jobs.example.com/positions/42?utm_source=search",
        company="示例公司",
        title="AI 产品经理",
        location="上海",
        recruitment_type="实习",
        page_time_information=["页面显示 2026-07-10 更新"],
        jd_text=jd_text,
        source_status="open",
    )


def test_context_is_one_user_visible_revision_with_compare_and_swap(repository):
    first = repository.replace_search_context(
        intent="寻找能够真正参与 AI 产品交付的岗位",
        hard_constraints=["用户明确要求在上海或远程"],
        feedback=[],
        expected_revision=None,
    )
    assert first.revision == 1

    second = repository.replace_search_context(
        intent="更关注有真实用户反馈闭环的 AI 产品岗位",
        hard_constraints=["用户明确要求在上海或远程"],
        feedback=["少推荐纯运营岗位"],
        expected_revision=1,
    )
    assert second.revision == 2
    assert repository.get_search_context() == second

    with pytest.raises(JobDiscoveryRevisionConflict):
        repository.replace_search_context(
            intent="旧页面的覆盖写入",
            expected_revision=1,
        )


def test_same_source_job_updates_in_place_when_jd_changes(repository):
    first = repository.save_job_evidence(_job())
    updated_jd = _complete_jd("职责已更新")
    updated = repository.save_job_evidence(
        _job(
            jd_text=updated_jd,
            source_evidence_id=12,
        )
    )

    assert updated.id == first.id
    assert updated.job_id == first.job_id
    assert updated.source_evidence_id == 12
    assert updated.jd_text.startswith("职责已更新")
    assert updated.content_sha256 != first.content_sha256
    assert len(repository.list_job_evidence()) == 1
    with repository.store.connect() as conn:
        application_job = conn.execute(
            "SELECT source_id, raw_text FROM jobs WHERE id = ?", (updated.job_id,)
        ).fetchone()
    assert application_job == (
        "position-42",
        updated_jd,
    )


def test_source_id_is_primary_identity_even_if_tracking_url_changes(repository):
    first = repository.save_job_evidence(_job())
    changed_url = _job().model_copy(
        update={"canonical_url": "https://jobs.example.com/new-route/42?from=homepage"}
    )
    second = repository.save_job_evidence(changed_url)
    assert second.id == first.id
    assert second.job_id == first.job_id


def test_url_identity_is_promoted_to_source_id_without_creating_a_second_job(
    repository,
):
    without_source_id = _job().model_copy(update={"source_job_id": None})
    first = repository.save_job_evidence(without_source_id)

    promoted = repository.save_job_evidence(_job(source_evidence_id=12))
    missing_again = repository.save_job_evidence(
        without_source_id.model_copy(update={"source_evidence_id": 13})
    )

    assert promoted.id == first.id
    assert promoted.job_id == first.job_id
    assert promoted.source_job_id == "position-42"
    assert missing_again.id == first.id
    assert missing_again.job_id == first.job_id
    assert missing_again.source_job_id == "position-42"
    with repository.store.connect() as conn:
        evidence_rows = conn.execute(
            "SELECT stable_key, source_job_id, job_id FROM job_posting_evidence"
        ).fetchall()
        job_rows = conn.execute(
            "SELECT id, source_id FROM jobs WHERE source = 'official-example'"
        ).fetchall()
    assert evidence_rows == [
        ("source-id:official-example:position-42", "position-42", first.job_id)
    ]
    assert job_rows == [(first.job_id, "position-42")]


def test_existing_application_job_is_reused_by_stable_source_id(repository):
    with repository.store.connect() as conn:
        unrelated_job_id = int(
            conn.execute(
                "INSERT INTO jobs("
                "source, source_id, url, title, company, raw_text, content_hash"
                ") VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    "manual",
                    "unrelated",
                    "https://example.test/unrelated",
                    "无关岗位",
                    "其他公司",
                    "无关 JD",
                    "unrelated-hash",
                ),
            ).fetchone()[0]
        )
        existing_job_id = int(
            conn.execute(
                "INSERT INTO jobs("
                "source, source_id, url, title, company, location, raw_text, content_hash"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    "official-example",
                    "position-42",
                    "https://jobs.example.com/old-route/42",
                    "旧标题",
                    "示例公司",
                    "上海",
                    "旧 JD",
                    "legacy-hash",
                ),
            ).fetchone()[0]
        )

    evidence = repository.save_job_evidence(_job())

    assert evidence.id != existing_job_id
    assert evidence.job_id == existing_job_id
    with repository.store.connect() as conn:
        row = conn.execute(
            "SELECT url, title, raw_text FROM jobs WHERE id = ?", (existing_job_id,)
        ).fetchone()
    assert row == (
        "https://jobs.example.com/positions/42",
        "AI 产品经理",
        _job().jd_text,
    )
    with repository.store.connect() as conn:
        unrelated = conn.execute(
            "SELECT title, raw_text FROM jobs WHERE id = ?", (unrelated_job_id,)
        ).fetchone()
    assert unrelated == ("无关岗位", "无关 JD")


def test_url_identity_removes_fragment_and_utm_without_removing_meaningful_query():
    value = canonicalize_job_url(
        "HTTPS://Jobs.Example.COM:443/position/?id=7&utm_campaign=x#details"
    )
    assert value == "https://jobs.example.com/position?id=7"


def test_generic_web_identity_ignores_model_source_labels_and_source_ids(repository):
    first = _job().model_copy(
        update={
            "evidence_kind": "web",
            "source_name": "model-label-a",
            "source_job_id": "invented-a",
        }
    )
    second = first.model_copy(
        update={
            "source_evidence_id": 12,
            "source_name": "model-label-b",
            "source_job_id": "invented-b",
            "canonical_url": (
                "https://JOBS.EXAMPLE.COM:443/positions/42?utm_source=another#top"
            ),
        }
    )

    saved_first = repository.save_job_evidence(first)
    saved_second = repository.save_job_evidence(second)

    assert saved_second.id == saved_first.id
    assert saved_second.job_id == saved_first.job_id
    assert saved_second.source_name == "jobs.example.com"
    assert saved_second.source_job_id is None
    assert len(repository.list_job_evidence()) == 1
    with repository.store.connect() as conn:
        rows = conn.execute(
            "SELECT stable_key FROM job_posting_evidence"
        ).fetchall()
    assert rows == [("web-url:https://jobs.example.com/positions/42",)]


def test_generic_fetch_cannot_demote_or_rewrite_platform_evidence(repository):
    platform = repository.save_job_evidence(_job())
    assert platform.id is not None and platform.job_id is not None

    generic = repository.save_job_evidence(
        _job(source_evidence_id=12).model_copy(
            update={
                "evidence_kind": "web",
                "source_name": "model-controlled",
                "source_job_id": "invented",
                "jd_text": _complete_jd("generic page supplied different text"),
                "source_status": "unknown",
            }
        )
    )
    refreshed = repository.save_job_evidence(
        _job(source_evidence_id=13).model_copy(
            update={"jd_text": _complete_jd("platform refresh")}
        )
    )

    assert generic.id == refreshed.id == platform.id
    assert generic.job_id == refreshed.job_id == platform.job_id
    assert generic.evidence_kind == "platform_adapter"
    assert generic.source_name == "official-example"
    assert generic.source_job_id == "position-42"
    assert generic.source_evidence_id == platform.source_evidence_id
    assert generic.jd_text == platform.jd_text
    assert refreshed.source_evidence_id == 13
    assert len(repository.list_job_evidence()) == 1
    with repository.store.connect() as conn:
        jobs = conn.execute(
            "SELECT source, source_id FROM jobs WHERE id = ?", (platform.job_id,)
        ).fetchall()
    assert jobs == [("official-example", "position-42")]


def test_repository_rejects_title_or_card_text_as_complete_jd(repository):
    with pytest.raises(ValueError, match="identity/card text"):
        repository.save_job_evidence(
            _job(jd_text="示例公司 AI 产品经理，上海实习。")
        )


def test_snapshot_contains_full_context_and_a_lightweight_recorded_job_catalog(repository):
    context = repository.replace_search_context(
        intent="寻找适合当前阶段的岗位",
        expected_revision=None,
    )
    job = repository.save_job_evidence(_job(jd_text="JD" * 10_000))
    published = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=job.id,
                why_worth_attention="完整简历内容显示的方向与岗位交付环节相关。",
                grounding_quotes=_grounding(),
                concerns=["团队边界尚未公开"],
                unknowns=["页面没有说明转正安排"],
            )
        ],
        candidate_evidence=_candidate_evidence(),
        coverage_summary="检查了该公司的官方岗位详情页。",
        evidence_gaps=[],
    )
    long_resume = "候选人证据" * 8_000
    snapshot = repository.load_snapshot(_candidate_evidence(long_resume))

    assert snapshot.search_context == context
    assert snapshot.candidate_evidence.documents[0].text == long_resume
    assert snapshot.recorded_job_catalog.total == 1
    assert snapshot.recorded_job_catalog.items[0].job_evidence_id == job.id
    assert "jd_text" not in snapshot.recorded_job_catalog.items[0].model_dump()
    assert snapshot.current_selection == published
    assert snapshot.current_selection.items[0].job_evidence is not None
    assert snapshot.current_selection.items[0].job_evidence.jd_text == "JD" * 10_000
    assert snapshot.current_result_revision == 1


def test_recorded_job_catalog_is_pageable_without_losing_complete_evidence(repository):
    saved = []
    for number in range(3):
        saved.append(
            repository.save_job_evidence(
                _job(source_evidence_id=20 + number).model_copy(
                    update={
                        "source_job_id": f"position-{number}",
                        "canonical_url": f"https://jobs.example.com/positions/{number}",
                        "title": f"岗位 {number}",
                        "jd_text": f"COMPLETE_RECORDED_JD_{number} " + ("正文 " * 30),
                    }
                )
            )
        )

    first = repository.list_job_evidence_catalog(offset=0, limit=2)
    second = repository.list_job_evidence_catalog(offset=first.next_offset, limit=2)

    assert first.total == 3
    assert len(first.items) == 2
    assert first.next_offset == 2
    assert len(second.items) == 1
    assert second.next_offset is None
    catalog_ids = {item.job_evidence_id for item in [*first.items, *second.items]}
    assert catalog_ids == {item.id for item in saved}
    for item in saved:
        complete = repository.get_job_evidence(item.id)
        assert complete is not None
        assert f"COMPLETE_RECORDED_JD_{item.source_evidence_id - 20}" in complete.jd_text


def test_published_selection_keeps_its_evidence_after_live_updates_and_failed_runs(
    repository,
):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    first_jd = _complete_jd("首次发布时读取")
    first = repository.save_job_evidence(_job(jd_text=first_jd))
    published = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=first.id,
                why_worth_attention="完整简历内容与当前岗位证据一致。",
                grounding_quotes=_grounding(),
            )
        ],
        candidate_evidence=_candidate_evidence(),
        coverage_summary="读取了岗位的官方详情页。",
        evidence_gaps=["团队信息尚未公开"],
    )

    updated_jd = _complete_jd("后来刷新到")
    updated = repository.save_job_evidence(
        _job(
            jd_text=updated_jd,
            source_evidence_id=12,
        )
    )
    with pytest.raises(JobDiscoveryRevisionConflict):
        repository.publish_selection(
            expected_context_revision=context.revision,
            expected_result_revision=0,
            items=[],
            candidate_evidence=_candidate_evidence(),
            coverage_summary="过期运行。",
            evidence_gaps=["已经存在更新的结果"],
        )
    with pytest.raises(JobDiscoveryEvidenceError):
        repository.publish_selection(
            expected_context_revision=context.revision,
            expected_result_revision=1,
            items=[
                JobSelectionItem(
                    job_evidence_id=999,
                    why_worth_attention="失败运行引用了不存在的岗位。",
                    grounding_quotes=_grounding(),
                )
            ],
            candidate_evidence=_candidate_evidence(),
            coverage_summary="失败运行。",
            evidence_gaps=[],
        )

    live = repository.get_job_evidence(updated.id)
    current = repository.get_current_selection()
    assert live is not None
    assert live.jd_text == updated_jd
    assert current is not None
    assert current.result_revision == 1
    assert current.items[0].job_evidence is not None
    assert current.items[0].job_evidence.jd_text == first_jd
    assert current == published


def test_each_published_selection_revision_remains_actionable(repository) -> None:
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    first_jd = _complete_jd("第一版")
    second_jd = _complete_jd("第二版")
    original = repository.save_job_evidence(_job(jd_text=first_jd))
    first = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=original.id,
                why_worth_attention="完整简历内容是第一轮选择的候选人依据。",
                grounding_quotes=_grounding(),
            )
        ],
        candidate_evidence=_candidate_evidence(),
        coverage_summary="第一轮发布。",
        evidence_gaps=["团队信息未知"],
    )
    repository.save_job_evidence(
        original.model_copy(
            update={
                "source_evidence_id": 99,
                "title": "更新后的岗位名",
                "jd_text": second_jd,
            }
        )
    )
    second = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=first.result_revision,
        items=[
            JobSelectionItem(
                job_evidence_id=original.id,
                why_worth_attention="完整简历内容是第二轮选择的候选人依据。",
                grounding_quotes=_grounding(),
            )
        ],
        candidate_evidence=_candidate_evidence(),
        coverage_summary="第二轮发布。",
        evidence_gaps=["团队信息未知"],
    )

    saved_first = repository.get_selection_revision(first.result_revision)
    saved_second = repository.get_selection_revision(second.result_revision)

    assert saved_first is not None
    assert saved_second is not None
    assert saved_first.items[0].job_evidence is not None
    assert saved_second.items[0].job_evidence is not None
    assert saved_first.items[0].job_evidence.title != "更新后的岗位名"
    assert saved_first.items[0].job_evidence.jd_text == first_jd
    assert saved_second.items[0].job_evidence.title == "更新后的岗位名"
    assert saved_second.items[0].job_evidence.jd_text == second_jd


def test_publication_is_ordered_and_revision_safe(repository):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    first_job = repository.save_job_evidence(_job())
    second_job = repository.save_job_evidence(
        _job(source_evidence_id=22).model_copy(
            update={
                "source_job_id": "position-43",
                "canonical_url": "https://jobs.example.com/positions/43",
                "title": "Agent 产品实习生",
            }
        )
    )
    selection = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=second_job.id,
                why_worth_attention="完整简历内容与该岗位方向更接近。",
                grounding_quotes=_grounding(),
            ),
            JobSelectionItem(
                job_evidence_id=first_job.id,
                why_worth_attention="完整简历内容也可迁移到这一相邻方向。",
                grounding_quotes=_grounding(),
            ),
        ],
        candidate_evidence=_candidate_evidence(),
        coverage_summary="读取并比较了两份完整官方 JD。",
        evidence_gaps=["两页均未公开团队规模"],
    )
    assert [item.job_evidence_id for item in selection.items] == [
        second_job.id,
        first_job.id,
    ]

    with pytest.raises(JobDiscoveryRevisionConflict):
        repository.publish_selection(
            expected_context_revision=context.revision,
            expected_result_revision=0,
            items=[],
            candidate_evidence=_candidate_evidence(),
            coverage_summary="旧运行试图覆盖新结果。",
            evidence_gaps=["无权发布"],
        )
    assert repository.get_current_selection() == selection


def test_context_change_makes_older_agent_run_stale(repository):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    repository.replace_search_context(
        intent="用户已经修改了方向",
        expected_revision=context.revision,
    )
    with pytest.raises(JobDiscoveryRevisionConflict):
        repository.publish_selection(
            expected_context_revision=context.revision,
            expected_result_revision=0,
            items=[],
            candidate_evidence=_candidate_evidence(),
            coverage_summary="基于旧方向的搜索。",
            evidence_gaps=["结果已过期"],
        )


def test_selection_cannot_reference_missing_job_evidence(repository):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    with pytest.raises(JobDiscoveryEvidenceError):
        repository.publish_selection(
            expected_context_revision=context.revision,
            expected_result_revision=0,
            items=[
                JobSelectionItem(
                    job_evidence_id=999,
                    why_worth_attention="模型声称存在但数据库没有的岗位。",
                    grounding_quotes=_grounding(),
                )
            ],
            candidate_evidence=_candidate_evidence(),
            coverage_summary="无可验证来源。",
            evidence_gaps=[],
        )


def test_selection_rejects_unsupported_candidate_quote_and_openclaw_claim(repository):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    job = repository.save_job_evidence(_job())

    with pytest.raises(JobDiscoveryEvidenceError, match="not a copied passage"):
        repository.publish_selection(
            expected_context_revision=context.revision,
            expected_result_revision=0,
            items=[
                JobSelectionItem(
                    job_evidence_id=job.id,
                    why_worth_attention="候选人简历中提及 OpenClaw。",
                    grounding_quotes=[
                        SelectionGroundingQuote(
                            reference="master-resume:confirmed",
                            quote="OpenClaw",
                        )
                    ],
                )
            ],
            candidate_evidence=_candidate_evidence(
                "候选人做过 Agent 产品，但没有该工具声明。"
            ),
            coverage_summary="读取了岗位详情。",
            evidence_gaps=[],
        )


def test_selection_rejects_unrelated_quote_for_invented_specific_claim(repository):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    job = repository.save_job_evidence(_job())

    with pytest.raises(JobDiscoveryEvidenceError, match="absent from"):
        repository.publish_selection(
            expected_context_revision=context.revision,
            expected_result_revision=0,
            items=[
                JobSelectionItem(
                    job_evidence_id=job.id,
                    why_worth_attention=(
                        "候选人做过 Agent 产品，并有 OpenClaw 大规模生产落地经验。"
                    ),
                    grounding_quotes=[
                        SelectionGroundingQuote(
                            reference="master-resume:confirmed",
                            quote="Agent 产品",
                        )
                    ],
                )
            ],
            candidate_evidence=_candidate_evidence("候选人做过 Agent 产品。"),
            coverage_summary="读取了岗位详情。",
            evidence_gaps=[],
        )


def test_selection_accepts_pdf_line_wraps_without_pasting_quote_into_reason(repository):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    job = repository.save_job_evidence(_job())
    wrapped_resume = (
        "提出 \u201c语义层 + 工作区层\u201d 双层架构：语义层以 AgentState 作为当前研究理解的结构化锚点，承载问题\n"
        "解释、关键未知与闭环条件。"
    )

    selection = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=job.id,
                why_worth_attention="候选人的研究系统设计经历与岗位职责方向相关。",
                grounding_quotes=[
                    SelectionGroundingQuote(
                        reference="master-resume:confirmed",
                        quote=(
                            '提出 "语义层 + 工作区层" 双层架构：语义层以 AgentState '
                            "作为当前研究理解的结构化锚点，承载问题解释、关键未知与闭环条件"
                        ),
                    )
                ],
            )
        ],
        candidate_evidence=_candidate_evidence(wrapped_resume),
        coverage_summary="读取了岗位详情。",
        evidence_gaps=[],
    )

    assert selection.items[0].why_worth_attention == (
        "候选人的研究系统设计经历与岗位职责方向相关。"
    )


def test_selection_accepts_ordered_ellipsis_and_literals_from_referenced_document(
    repository,
):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    job = repository.save_job_evidence(_job())
    resume = (
        "提出语义层与工作区层双层架构，负责研究动作执行、持久化与恢复。\n"
        "使用 DeepSpeed、LoRA 与 ZeRO-2 完成训练优化。"
    )

    selection = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=job.id,
                why_worth_attention=(
                    "双层研究架构经验与岗位相关，且具备 DeepSpeed、LoRA 与 ZeRO-2 实践。"
                ),
                grounding_quotes=[
                    SelectionGroundingQuote(
                        reference="master-resume:confirmed",
                        quote="提出语义层与工作区层双层架构……负责研究动作执行、持久化与恢复",
                    )
                ],
            )
        ],
        candidate_evidence=_candidate_evidence(resume),
        coverage_summary="读取了岗位详情。",
        evidence_gaps=[],
    )

    assert selection.items[0].grounding_quotes[0].quote.endswith("持久化与恢复")


def test_selection_rejects_reordered_ellipsis_fragments(repository):
    context = repository.replace_search_context(intent="寻找岗位", expected_revision=None)
    job = repository.save_job_evidence(_job())

    with pytest.raises(JobDiscoveryEvidenceError, match="not a copied passage"):
        repository.publish_selection(
            expected_context_revision=context.revision,
            expected_result_revision=0,
            items=[
                JobSelectionItem(
                    job_evidence_id=job.id,
                    why_worth_attention="候选人的研究架构经历与岗位相关。",
                    grounding_quotes=[
                        SelectionGroundingQuote(
                            reference="master-resume:confirmed",
                            quote="负责研究动作执行……提出语义层与工作区层双层架构",
                        )
                    ],
                )
            ],
            candidate_evidence=_candidate_evidence(
                "提出语义层与工作区层双层架构，负责研究动作执行。"
            ),
            coverage_summary="读取了岗位详情。",
            evidence_gaps=[],
        )


def test_hard_constraints_and_feedback_are_first_class_grounding_sources(repository):
    context = repository.replace_search_context(
        intent="寻找岗位",
        hard_constraints=["地点必须在上海或支持远程"],
        feedback=["更关注能接触真实用户反馈的团队"],
        expected_revision=None,
    )
    job = repository.save_job_evidence(_job())

    selection = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=job.id,
                why_worth_attention=(
                    "地点必须在上海或支持远程；同时更关注能接触真实用户反馈的团队。"
                ),
                grounding_quotes=[
                    SelectionGroundingQuote(
                        reference="search-context:hard-constraint:1",
                        quote="地点必须在上海或支持远程",
                    ),
                    SelectionGroundingQuote(
                        reference="search-context:feedback:1",
                        quote="更关注能接触真实用户反馈的团队",
                    ),
                ],
            )
        ],
        candidate_evidence=CandidateEvidence(),
        coverage_summary="读取了岗位详情。",
        evidence_gaps=[],
    )

    assert selection.items[0].grounding_quotes[0].reference.endswith(":1")


def test_legacy_ungrounded_selection_is_not_kept_actionable(tmp_path):
    store = Store(tmp_path / "legacy-job-discovery.db")
    store.init_schema()
    old_selection = {
        "context_revision": 1,
        "result_revision": 1,
        "items": [
            {
                "job_evidence_id": 1,
                "why_worth_attention": "旧结果仍然引用这份完整岗位证据。",
                "concerns": [],
                "unknowns": ["团队信息未知"],
            }
        ],
        "coverage_summary": "旧版本已经读取岗位详情。",
        "evidence_gaps": ["团队信息未知"],
    }
    with store.connect() as conn:
        unrelated_job_id = int(
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, raw_text, "
                "content_hash) VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    "manual",
                    "unrelated",
                    "https://example.test/unrelated",
                    "无关岗位",
                    "其他公司",
                    "无关 JD",
                    "unrelated-hash",
                ),
            ).fetchone()[0]
        )
        conn.execute(
            """
            CREATE TABLE job_posting_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stable_key TEXT NOT NULL UNIQUE,
                source_evidence_id INTEGER NOT NULL,
                source_name TEXT NOT NULL,
                source_job_id TEXT,
                canonical_url TEXT NOT NULL,
                company TEXT NOT NULL,
                title TEXT NOT NULL,
                location TEXT,
                recruitment_type TEXT,
                page_time_json TEXT NOT NULL DEFAULT '[]',
                jd_text TEXT NOT NULL,
                source_status TEXT NOT NULL DEFAULT 'unknown',
                content_sha256 TEXT NOT NULL,
                checked_at REAL NOT NULL,
                last_seen_at REAL NOT NULL,
                created_at REAL NOT NULL DEFAULT (julianday('now')),
                updated_at REAL NOT NULL DEFAULT (julianday('now'))
            )
            """
        )
        conn.execute(
            "INSERT INTO job_posting_evidence("
            "stable_key, source_evidence_id, source_name, source_job_id, canonical_url, "
            "company, title, location, recruitment_type, page_time_json, jd_text, "
            "source_status, content_sha256, checked_at, last_seen_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "source-url:official-example:https://jobs.example.com/positions/42",
                11,
                "official-example",
                None,
                "https://jobs.example.com/positions/42",
                "示例公司",
                "AI 产品经理",
                "上海",
                "实习",
                "[]",
                "旧 schema 保存的完整 JD。",
                "open",
                "a" * 64,
                2461234.5,
                2461234.5,
            ),
        )
        conn.execute(
            """
            CREATE TABLE job_discovery_current (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                context_revision INTEGER NOT NULL,
                result_revision INTEGER NOT NULL,
                selection_json TEXT NOT NULL,
                published_at REAL NOT NULL DEFAULT (julianday('now'))
            )
            """
        )
        conn.execute(
            "INSERT INTO job_discovery_current("
            "singleton, context_revision, result_revision, selection_json"
            ") VALUES (1, 1, 1, ?)",
            (json.dumps(old_selection, ensure_ascii=False),),
        )

    init_job_discovery_schema(store)
    repository = JobDiscoveryRepository(store)
    migrated = repository.get_job_evidence(1)
    selection = repository.get_current_selection()

    assert migrated is not None
    assert migrated.job_id is not None
    assert migrated.job_id != migrated.id
    assert migrated.job_id != unrelated_job_id
    assert selection is None
    assert repository.get_selection_revision(1) is None
    assert repository.current_result_revision() == 1
    with store.connect() as conn:
        columns = {
            row[1]: row for row in conn.execute(
                "PRAGMA table_info(job_posting_evidence)"
            ).fetchall()
        }
    assert columns["job_id"][3] == 1

    with pytest.raises(sqlite3.IntegrityError), store.connect() as conn:
        conn.execute(
            "UPDATE job_posting_evidence SET job_id = NULL WHERE id = ?",
            (migrated.id,),
        )

    second = repository.save_job_evidence(
        _job(source_evidence_id=22).model_copy(
            update={
                "source_job_id": "position-43",
                "canonical_url": "https://jobs.example.com/positions/43",
            }
        )
    )
    with pytest.raises(sqlite3.IntegrityError), store.connect() as conn:
        conn.execute(
            "UPDATE job_posting_evidence SET job_id = ? WHERE id = ?",
            (migrated.job_id, second.id),
        )


def test_empty_selection_requires_a_real_gap_explanation():
    with pytest.raises(ValueError):
        JobSelectionSet(
            context_revision=1,
            result_revision=1,
            items=[],
            coverage_summary="完成了本轮检查。",
            evidence_gaps=[],
        )
