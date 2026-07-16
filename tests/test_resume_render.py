from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from pypdf import PdfReader

from offerguide.resume import (
    ResumeBlock,
    ResumeDocument,
    ResumeEntry,
    ResumeEntryRow,
    ResumeHeader,
    ResumeSection,
    RichText,
    RichTextSpan,
    document_text,
    render_resume,
)
from offerguide.resume import render as resume_render


def _text(value: str, *, emphasis: bool = False) -> RichText:
    return RichText(spans=[RichTextSpan(text=value, emphasis=emphasis)])


def _document(
    *,
    section_title: str = "任意栏目名",
    bullets: int = 1,
    page_break_before: bool = False,
) -> ResumeDocument:
    blocks = [
        ResumeBlock(
            kind="bullet",
            content=RichText(
                spans=[
                    RichTextSpan(text=f"第 {index + 1} 条采用"),
                    RichTextSpan(text="双层架构", emphasis=True),
                    RichTextSpan(text="解决完整信息流问题，并保留可核验的事实边界。"),
                ]
            ),
        )
        for index in range(bullets)
    ]
    return ResumeDocument(
        header=ResumeHeader(name=_text("胡阳"), lines=[_text("huy@example.com")]),
        sections=[
            ResumeSection(
                title=_text(section_title),
                page_break_before=page_break_before,
                entries=[
                    ResumeEntry(
                        rows=[
                            ResumeEntryRow(
                                left=_text("未带公司后缀的自由抬头", emphasis=True),
                                right=_text("2026.01-至今"),
                            ),
                            ResumeEntryRow(left=_text("招商银行 FinTech 训练营推理优化第一名")),
                        ],
                        blocks=blocks,
                    )
                ],
            )
        ],
    )


@pytest.fixture
def render_tools() -> None:
    if resume_render._find_typst() is None:
        pytest.skip("typst CLI is not installed")
    if shutil.which("pdftoppm") is None:
        pytest.skip("pdftoppm is not installed")


def test_arbitrary_section_name_is_not_inferred_and_emphasis_stays_adjacent() -> None:
    document = _document(section_title="火星档案 / 自定义栏目")

    source = resume_render._render_typst_source(
        document,
        portrait_path=None,
    )

    assert document_text(document).splitlines() == [
        "胡阳",
        "huy@example.com",
        "火星档案 / 自定义栏目",
        "未带公司后缀的自由抬头",
        "2026.01-至今",
        "招商银行 FinTech 训练营推理优化第一名",
        "第 1 条采用双层架构解决完整信息流问题，并保留可核验的事实边界。",
    ]
    assert '#resume-section([#text("火星档案 / 自定义栏目")]' in source
    assert "#resume-entry(" in source
    entry_call = next(line for line in source.splitlines() if line.startswith("#resume-entry("))
    assert "#resume-bullet(" in entry_call
    assert "rows:" in source
    assert "2026.01-至今" in source
    assert '#strong[#text("未带公司后缀的自由抬头")]' in source
    assert '#text("第 1 条采用")#strong[#text("双层架构")]#text("解决完整信息流问题' in source
    assert "采用 双层架构" not in source
    assert "教育背景" not in source
    assert "项目经历" not in source
    assert not any(line.startswith("#resume-bullet(") for line in source.splitlines())


def test_renderer_passes_through_any_number_of_bullets_without_fixed_page_rules() -> None:
    document = _document(bullets=17)

    source = resume_render._render_typst_source(
        document,
        portrait_path=None,
    )

    assert source.count("#resume-bullet(") == 17
    assert "#pagebreak(" not in source
    assert "scale(" not in source
    assert "not final" not in source


def test_first_entry_cannot_duplicate_the_section_rule() -> None:
    document = _document()
    first = document.sections[0].entries[0]
    second = first.model_copy(
        update={
            "rows": [ResumeEntryRow(left=_text("第二条经历"))],
            "divider_before": True,
        }
    )
    document = document.model_copy(
        update={
            "sections": [
                document.sections[0].model_copy(
                    update={
                        "entries": [
                            first.model_copy(update={"divider_before": True}),
                            second,
                        ]
                    }
                )
            ]
        }
    )

    source = resume_render._render_typst_source(document, portrait_path=None)
    entry_lines = [line for line in source.splitlines() if line.startswith("#resume-entry(")]

    assert "divider-before: false" in entry_lines[0]
    assert "divider-before: true" in entry_lines[1]


def test_natural_pagination_produces_one_png_per_actual_page_and_content_addressed_path(
    tmp_path: Path,
    render_tools: None,
) -> None:
    document = _document(bullets=80)

    result = render_resume(
        document=document,
        company="百度",
        role="Agent 策略算法实习生",
        output_dir=tmp_path,
    )

    assert len(result.pages) >= 2
    assert len(PdfReader(str(result.pdf.path)).pages) == len(result.pages)
    assert [page.page_number for page in result.pages] == list(range(1, len(result.pages) + 1))
    for page in result.pages:
        assert page.image.path.is_file()
        assert page.image.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert page.image.path.parent.name == result.render_key[:16]
    canonical = document.model_dump_json(exclude_none=True)
    document_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected_key = hashlib.sha256(
        json.dumps(
            {
                "document_sha256": document_sha,
                "portrait_sha256": None,
                "template_sha256": result.template_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert result.document_sha256 == document_sha
    assert result.portrait_sha256 is None
    assert result.render_key == expected_key
    assert result.render_key[:10] in result.pdf.path.stem
    assert result.pdf.sha256 == hashlib.sha256(result.pdf.path.read_bytes()).hexdigest()


def test_same_render_identity_reuses_the_verified_artifact(
    tmp_path: Path,
    render_tools: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document()
    first = render_resume(
        document=document,
        company="Example",
        role="Agent Intern",
        output_dir=tmp_path,
    )

    def unexpected_compile(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an existing render identity must not be recompiled")

    monkeypatch.setattr(resume_render, "_compile_typst_pdf", unexpected_compile)
    second = render_resume(
        document=document,
        company="Example",
        role="Agent Intern",
        output_dir=tmp_path,
    )

    assert second.render_key == first.render_key
    assert second.pdf == first.pdf
    assert second.pages == first.pages


def test_render_identity_changes_with_the_actual_portrait_content() -> None:
    document_sha = "a" * 64
    template_sha = "b" * 64

    without_portrait = resume_render._render_identity(document_sha, template_sha, None)
    first_portrait = resume_render._render_identity(document_sha, template_sha, "c" * 64)
    second_portrait = resume_render._render_identity(document_sha, template_sha, "d" * 64)

    assert len({without_portrait, first_portrait, second_portrait}) == 3


def test_explicit_page_break_is_applied_only_at_the_requested_semantic_boundary(
    tmp_path: Path,
    render_tools: None,
) -> None:
    first = _document(section_title="第一页内容").sections[0]
    second = _document(section_title="第二页显式开始", page_break_before=True).sections[0]
    document = ResumeDocument(
        header=ResumeHeader(name=_text("胡阳")),
        sections=[first, second],
    )

    source = resume_render._render_typst_source(
        document,
        portrait_path=None,
    )
    result = render_resume(
        document=document,
        company="测试公司",
        role="测试岗位",
        output_dir=tmp_path,
    )

    assert source.count("#pagebreak(weak: false)") == 1
    assert len(result.pages) == 2
    assert "第一页内容" in result.pages[0].extracted_text
    assert "第二页显式开始" in result.pages[1].extracted_text


def test_template_and_binding_do_not_restore_old_sticky_or_section_alias_rules() -> None:
    template = resume_render.resume_template_path().read_text(encoding="utf-8")
    binding = Path(resume_render.__file__).read_text(encoding="utf-8")
    combined = template + "\n" + binding

    assert "sticky: not final" not in template
    assert "#pagebreak" not in template
    assert "section_alias" not in combined.casefold()
    assert "SECTION_ALIASES" not in combined
    assert "_section_kind" not in combined
    assert "教育背景" not in combined
    assert "项目经历" not in combined
    assert "bullet_count" not in combined
    assert "page_count" not in combined
    assert "columns: (1fr, 39mm)" in template
    assert 'weight: "bold", primary' not in template
    assert "columns: (22mm, 1fr, 22mm)" in template
    assert "place(" not in template


def test_pdf_layout_allows_shared_line_edges_but_rejects_real_overlap() -> None:
    touching = """\
<pdf2xml>
  <page number="1">
    <text top="378" height="17">first line</text>
    <text top="395" height="17">second line</text>
  </page>
</pdf2xml>
"""
    overlapping = """\
<pdf2xml>
  <page number="1">
    <text top="378" height="17">first line</text>
    <text top="394" height="17">second line</text>
  </page>
</pdf2xml>
"""

    assert resume_render._overlapping_text_lines(touching) == []
    assert resume_render._overlapping_text_lines(overlapping) == [
        (1, 378, 395, 394, 411)
    ]
