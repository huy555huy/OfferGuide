from __future__ import annotations

import pytest
from pydantic import ValidationError

from offerguide.resume import (
    MasterResumeDocument,
    MasterResumeSource,
    ProjectFact,
    ResumeJobContext,
    build_resume_context,
)

SHA = "b" * 64


def _master(*, confirmed: bool = True, document_sha: str = SHA):
    source = MasterResumeSource(
        source_path="/resume/master.pdf",
        sha256=SHA,
        extracted_text="PDF 原始文本",
    )
    document = MasterResumeDocument(
        source_sha256=document_sha,
        semantic_text="用户确认后的 master 语义文本",
        confirmed_by_user=confirmed,
    )
    return source, document


def test_context_keeps_every_source_separate_and_complete() -> None:
    tail = "TAIL_SENTINEL"
    source, master = _master()
    source = source.model_copy(update={"extracted_text": "原始简历" + "甲" * 12_000 + tail})
    master = master.model_copy(update={"semantic_text": "语义简历" + "乙" * 12_000 + tail})
    job = ResumeJobContext(
        job_id=10,
        company="百度",
        title="Agent 策略算法实习生",
        jd_text="完整 JD" + "丙" * 12_000 + tail,
        verified_information=["官网岗位仍在招聘" + tail],
    )
    project = ProjectFact(
        project_id=7,
        title="OfferGuide",
        facts=["实现完整求职信息流" + tail],
        do_not_claim=["没有线上用户规模" + tail],
    )
    context = build_resume_context(
        job=job,
        master_source=source,
        master_document=master,
        project_facts=[project],
        feedback=["第一页这里太空" + tail],
    )

    assert context.job.jd_text.endswith(tail)
    assert context.job.verified_information[0].endswith(tail)
    assert context.master_source.extracted_text.endswith(tail)
    assert context.master_document.semantic_text.endswith(tail)
    assert context.project_facts[0].facts[0].endswith(tail)
    assert context.project_facts[0].do_not_claim[0].endswith(tail)
    assert context.feedback[0].endswith(tail)


def test_context_keeps_project_boundaries_explicit() -> None:
    source, master = _master()
    context = build_resume_context(
        job=ResumeJobContext(
            job_id=1,
            company="公司",
            title="岗位",
            jd_text="完整岗位描述",
        ),
        master_source=source,
        master_document=master,
        project_facts=[
            ProjectFact(
                project_id=1,
                title="候选人项目",
                facts=["完成本地原型"],
                do_not_claim=["没有准确率数据"],
            )
        ],
    )

    assert context.project_facts[0].facts == ["完成本地原型"]
    assert context.project_facts[0].do_not_claim == ["没有准确率数据"]


def test_context_requires_confirmed_semantics_for_the_same_pdf() -> None:
    job = ResumeJobContext(job_id=1, company="公司", title="岗位", jd_text="完整 JD")
    source, unconfirmed = _master(confirmed=False)
    with pytest.raises(ValidationError, match="confirmed by the user"):
        build_resume_context(
            job=job,
            master_source=source,
            master_document=unconfirmed,
        )

    source, wrong_source = _master(document_sha="c" * 64)
    with pytest.raises(ValidationError, match="does not match"):
        build_resume_context(
            job=job,
            master_source=source,
            master_document=wrong_source,
        )


def test_context_has_no_unconnected_placeholder_collections() -> None:
    source, master = _master()
    context = build_resume_context(
        job=ResumeJobContext(job_id=1, company="公司", title="岗位", jd_text="完整 JD"),
        master_source=source,
        master_document=master,
    )

    dumped = context.model_dump()
    assert "user_supplements" not in dumped
    assert "references" not in dumped
    assert "omitted_materials" not in dumped


def test_context_rejects_accidentally_passing_one_string_as_a_list() -> None:
    source, master = _master()
    with pytest.raises(TypeError, match="sequence of strings"):
        build_resume_context(
            job=ResumeJobContext(job_id=1, company="公司", title="岗位", jd_text="完整 JD"),
            master_source=source,
            master_document=master,
            feedback="不要把我拆成单个字符",  # type: ignore[arg-type]
        )
