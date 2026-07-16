from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from offerguide.llm import LLMError
from offerguide.resume import (
    ArtifactFile,
    MasterResumeDocument,
    MasterResumeSource,
    ProjectFact,
    ResumeBlock,
    ResumeContext,
    ResumeDocument,
    ResumeEditor,
    ResumeEditorResult,
    ResumeEntry,
    ResumeEntryRow,
    ResumeHeader,
    ResumeJobContext,
    ResumePage,
    ResumeRenderResult,
    ResumeSection,
    RichText,
    RichTextSpan,
    VisualReviewUnavailable,
)

SHA = "a" * 64


def _text(value: str, *, emphasis: bool = False) -> RichText:
    return RichText(spans=[RichTextSpan(text=value, emphasis=emphasis)])


def _document() -> ResumeDocument:
    return ResumeDocument(
        header=ResumeHeader(name=_text("胡阳"), lines=[_text("huy@example.com")]),
        sections=[
            ResumeSection(
                title=_text("任意栏目"),
                entries=[
                    ResumeEntry(
                        rows=[ResumeEntryRow(left=_text("OfferGuide", emphasis=True))],
                        blocks=[
                            ResumeBlock(
                                kind="bullet",
                                content=RichText(
                                    spans=[
                                        RichTextSpan(text="实现"),
                                        RichTextSpan(text="完整信息流", emphasis=True),
                                        RichTextSpan(text="并保留事实边界"),
                                    ]
                                ),
                            )
                        ],
                    )
                ],
            )
        ],
    )


def _context(*, current: ResumeDocument | None = None) -> ResumeContext:
    return ResumeContext(
        job=ResumeJobContext(
            job_id=10,
            company="百度",
            title="Agent 策略算法实习生",
            jd_text="完整 JD 尾部标记 JD_TAIL",
            source_url="https://example.test/job/10",
            verified_information=["岗位仍在招聘 VERIFIED_TAIL"],
        ),
        master_source=MasterResumeSource(
            source_path="/resume/master.pdf",
            sha256=SHA,
            extracted_text="PDF 原始文本 MASTER_SOURCE_TAIL",
        ),
        master_document=MasterResumeDocument(
            source_sha256=SHA,
            semantic_text="用户确认后的语义文档 MASTER_DOCUMENT_TAIL",
            confirmed_by_user=True,
        ),
        project_facts=[
            ProjectFact(
                project_id=7,
                title="OfferGuide",
                facts=["完成完整求职链路 PROJECT_FACT_TAIL"],
                do_not_claim=["没有线上用户规模 DO_NOT_CLAIM_TAIL"],
            )
        ],
        feedback=["历史反馈 HISTORY_FEEDBACK_TAIL"],
        current_resume=current,
    )


def _result_json(document: ResumeDocument, note: str = "完成当前编辑") -> str:
    return json.dumps(
        {"document": document.model_dump(mode="json"), "editor_note": note},
        ensure_ascii=False,
    )


@dataclass
class _Response:
    content: str


class _RecordingLLM:
    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Response:
        self.calls.append((messages, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _Response(content=response)


def _render_result(tmp_path: Path, page_count: int = 2) -> ResumeRenderResult:
    pages: list[ResumePage] = []
    for page_number in range(1, page_count + 1):
        image_path = tmp_path / f"page-{page_number}.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([page_number]) * 128)
        pages.append(
            ResumePage(
                page_number=page_number,
                image=ArtifactFile(
                    path=image_path,
                    sha256=f"{page_number:064x}",
                    size_bytes=image_path.stat().st_size,
                ),
                extracted_text=f"第 {page_number} 页提取文字 PAGE_TEXT_{page_number}",
            )
        )
    return ResumeRenderResult(
        generated_on=date(2026, 7, 13),
        document_sha256="b" * 64,
        template_sha256="c" * 64,
        portrait_sha256=None,
        render_key="d" * 64,
        visible_characters=100,
        pdf=ArtifactFile(path=tmp_path / "resume.pdf", sha256="e" * 64, size_bytes=100),
        pages=tuple(pages),
    )


def test_edit_sends_complete_context_current_resume_and_feedback() -> None:
    document = _document()
    context = _context(current=document)
    llm = _RecordingLLM(_result_json(document))

    result = ResumeEditor(llm).edit(context, user_feedback="本次反馈 CURRENT_FEEDBACK_TAIL")

    assert result.document == document
    assert result.editor_note == "完成当前编辑"
    assert result.preparation_notes == []
    assert len(llm.calls) == 1
    messages, kwargs = llm.calls[0]
    system_prompt = messages[0]["content"]
    request = json.loads(messages[1]["content"])
    assert "顶层栏目默认沿用已确认 master 的阅读顺序" in system_prompt
    assert "不能用于栏目首条" in system_prompt
    assert "RichText.spans 只用于同一段连续文字中的局部强调" in system_prompt
    assert "source_path" not in request["resume_context"]["master_source"]
    assert request["resume_context"]["master_source"]["extracted_text"].endswith(
        "MASTER_SOURCE_TAIL"
    )
    assert request["resume_context"]["current_resume"] == document.model_dump(
        mode="json",
        exclude_none=True,
    )
    assert request["user_feedback_for_this_edit"] == "本次反馈 CURRENT_FEEDBACK_TAIL"
    assert request["resume_context"]["feedback"] == ["历史反馈 HISTORY_FEEDBACK_TAIL"]
    assert kwargs == {"temperature": 0.25, "json_mode": True}


def test_malformed_json_triggers_exactly_one_model_repair() -> None:
    document = _document()
    llm = _RecordingLLM("not-json", _result_json(document, "修复后可用"))

    result = ResumeEditor(llm).edit(_context())

    assert result.editor_note == "修复后可用"
    assert len(llm.calls) == 2
    repair_messages, repair_kwargs = llm.calls[1]
    repair_request = json.loads(repair_messages[1]["content"])
    assert repair_request["invalid_output"] == "not-json"
    assert "validation_error" in repair_request
    assert "output_schema" in repair_request
    assert repair_kwargs == {"temperature": 0.0, "json_mode": True}


def test_review_pages_attaches_an_image_url_for_every_rendered_page(tmp_path: Path) -> None:
    document = _document()
    current = ResumeEditorResult(document=document, editor_note="初稿")
    render = _render_result(tmp_path, page_count=2)
    llm = _RecordingLLM(_result_json(document, "页面已审阅"))

    result = ResumeEditor(llm).review_pages(
        context=_context(current=document),
        current=current,
        render=render,
        user_feedback="第一页空白太大",
    )

    assert result.editor_note == "页面已审阅"
    messages, kwargs = llm.calls[0]
    content = messages[1]["content"]
    assert isinstance(content, list)
    image_parts = [part for part in content if part.get("type") == "image_url"]
    assert len(image_parts) == len(render.pages) == 2
    for image_part, page in zip(image_parts, render.pages, strict=True):
        data_url = image_part["image_url"]["url"]
        prefix, encoded = data_url.split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded) == page.image.path.read_bytes()
        assert image_part["image_url"]["detail"] == "high"
    text_payload = json.loads(content[0]["text"])
    assert [page["extracted_text"] for page in text_payload["rendered_pages"]] == [
        "第 1 页提取文字 PAGE_TEXT_1",
        "第 2 页提取文字 PAGE_TEXT_2",
    ]
    assert text_payload["user_feedback_for_this_review"] == "第一页空白太大"
    assert kwargs == {"temperature": 0.2, "json_mode": True}


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 400: image_url content is not supported",
        "HTTP 400: expected `text` content part",
    ],
)
def test_text_only_endpoint_error_becomes_visual_review_unavailable(
    tmp_path: Path,
    message: str,
) -> None:
    document = _document()
    llm = _RecordingLLM(LLMError(message))

    with pytest.raises(VisualReviewUnavailable, match="text only"):
        ResumeEditor(llm).review_pages(
            context=_context(current=document),
            current=ResumeEditorResult(document=document, editor_note="初稿"),
            render=_render_result(tmp_path, page_count=1),
        )

    assert len(llm.calls) == 1
