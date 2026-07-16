from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

import offerguide
from offerguide.resume import (
    MasterResumeDocument,
    MasterResumeSource,
    ResumeBlock,
    ResumeDocument,
    ResumeEntry,
    ResumeEntryRow,
    ResumeHeader,
    ResumeSection,
    RichText,
    RichTextSpan,
    load_resume_pdf,
)

SHA = "a" * 64


def _text(value: str, *, emphasis: bool = False) -> RichText:
    return RichText(spans=[RichTextSpan(text=value, emphasis=emphasis)])


def test_master_source_requires_real_extracted_text_and_normalizes_hash() -> None:
    source = MasterResumeSource(
        source_path="/tmp/master.pdf",
        sha256="A" * 64,
        extracted_text="第一页\n\n第二页",
    )

    assert source.sha256 == SHA
    assert source.extracted_text == "第一页\n\n第二页"
    with pytest.raises(ValidationError, match="no extractable text"):
        MasterResumeSource(
            source_path="/tmp/blank.pdf",
            sha256=SHA,
            extracted_text=" \n\t ",
        )


def test_loader_returns_source_identity_and_rejects_empty_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "master.pdf"
    pdf.write_bytes(b"%PDF-test-content")
    monkeypatch.setattr(
        "offerguide.resume.master._extract_pdf_text",
        lambda _path: "完整简历文本",
    )

    source = load_resume_pdf(pdf)

    assert source.source_path == str(pdf.resolve())
    assert source.sha256 == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert source.extracted_text == "完整简历文本"

    monkeypatch.setattr("offerguide.resume.master._extract_pdf_text", lambda _path: "  ")
    with pytest.raises(ValueError, match="no extractable text"):
        load_resume_pdf(pdf)


def test_master_semantic_document_records_explicit_user_confirmation() -> None:
    document = MasterResumeDocument(
        source_sha256=SHA,
        semantic_text="用户检查并修正后的完整语义文本",
        confirmed_by_user=True,
    )

    assert document.confirmed_by_user is True
    assert document.semantic_text == "用户检查并修正后的完整语义文本"


def test_resume_document_preserves_explicit_rich_text_and_layout_decisions() -> None:
    content = RichText(
        spans=[
            RichTextSpan(text="实现 "),
            RichTextSpan(text="状态恢复", emphasis=True),
            RichTextSpan(text="，并保留证据。"),
        ]
    )
    document = ResumeDocument(
        header=ResumeHeader(name=_text("胡阳"), lines=[_text("huy@example.com")]),
        sections=[
            ResumeSection(
                title=_text("项目经历"),
                page_break_before=True,
                keep_title_with_first_entry=True,
                entries=[
                    ResumeEntry(
                        rows=[
                            ResumeEntryRow(
                                left=_text("OfferGuide", emphasis=True),
                                right=_text("2026.01-至今"),
                            ),
                            ResumeEntryRow(left=_text("完整求职链路")),
                        ],
                        page_break_before=True,
                        keep_header_with_first_block=True,
                        blocks=[ResumeBlock(kind="bullet", content=content)],
                    )
                ],
            )
        ],
    )

    restored = ResumeDocument.model_validate_json(document.model_dump_json())
    entry = restored.sections[0].entries[0]
    assert entry.blocks[0].content.plain_text == "实现 状态恢复，并保留证据。"
    assert entry.blocks[0].content.spans[1].emphasis is True
    assert restored.sections[0].page_break_before is True
    assert restored.sections[0].keep_title_with_first_entry is True
    assert entry.page_break_before is True
    assert entry.keep_header_with_first_block is True
    assert entry.rows[0].right is not None
    assert entry.rows[0].right.plain_text == "2026.01-至今"
    assert entry.rows[1].left.plain_text == "完整求职链路"


def test_resume_tree_accepts_content_shape_without_global_quotas() -> None:
    entries = [
        ResumeEntry(
            rows=[ResumeEntryRow(left=_text(f"项目 {index}"))],
            blocks=[
                ResumeBlock(kind="paragraph", content=_text("背景说明")),
                *[
                    ResumeBlock(kind="bullet", content=_text(f"事实 {bullet}"))
                    for bullet in range(12)
                ],
            ],
        )
        for index in range(8)
    ]
    document = ResumeDocument(
        header=ResumeHeader(name=_text("候选人")),
        sections=[ResumeSection(title=_text("任意栏目名"), entries=entries)],
    )

    assert len(document.sections[0].entries) == 8
    assert len(document.sections[0].entries[0].blocks) == 13


def test_pre_row_document_is_upgraded_without_exposing_legacy_fields_in_schema() -> None:
    legacy = {
        "header": {"name": {"spans": [{"text": "候选人"}]}},
        "sections": [
            {
                "title": {"spans": [{"text": "经历"}]},
                "keep_heading_with_first_block": True,
                "entries": [
                    {
                        "heading": {"spans": [{"text": "公司"}]},
                        "aside": {"spans": [{"text": "2026"}]},
                        "keep_heading_with_first_block": True,
                    }
                ],
            }
        ],
    }

    document = ResumeDocument.model_validate(legacy)
    dumped = document.model_dump(mode="json")
    entry = document.sections[0].entries[0]

    assert entry.rows[0].left.plain_text == "公司"
    assert entry.rows[0].right is not None
    assert entry.rows[0].right.plain_text == "2026"
    assert entry.keep_header_with_first_block is True
    assert document.sections[0].keep_title_with_first_entry is True
    assert "heading" not in dumped["sections"][0]["entries"][0]
    assert "aside" not in dumped["sections"][0]["entries"][0]
    schema = ResumeDocument.model_json_schema()
    entry_schema = schema["$defs"]["ResumeEntry"]["properties"]
    assert "rows" in entry_schema
    assert "heading" not in entry_schema
    assert "aside" not in entry_schema


def test_old_half_populated_user_profile_is_not_public_api() -> None:
    assert not hasattr(offerguide, "UserProfile")
