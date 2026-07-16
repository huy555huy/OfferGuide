"""The single model editor for initial drafting, feedback, and visual review."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..llm import LLMError
from .models import ResumeContext, ResumeDocument
from .render import ResumeRenderResult


class ResumeEditorError(RuntimeError):
    """The editing model did not return a usable semantic document."""


class VisualReviewUnavailable(ResumeEditorError):
    """The configured model endpoint cannot inspect rendered page images."""


class PreparationNote(BaseModel):
    """A newly added, interview-preparable capability and what the user must cover."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    note: str

    @field_validator("claim", "note")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("preparation note fields must not be blank")
        return value


class ResumeEditorResult(BaseModel):
    """Only the current document, a short explanation, and genuine prep needs."""

    model_config = ConfigDict(extra="forbid")

    document: ResumeDocument
    editor_note: str
    preparation_notes: list[PreparationNote] = Field(default_factory=list)

    @field_validator("editor_note")
    @classmethod
    def _editor_note_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("editor_note must not be blank")
        return value


class _ChatClient(Protocol):
    def chat(
        self,
        messages: Any,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> Any: ...


class ResumeEditor:
    """Edit the one current document with all candidate and job information present."""

    def __init__(self, llm: _ChatClient) -> None:
        self.llm = llm

    def edit(
        self,
        context: ResumeContext,
        *,
        user_feedback: str | None = None,
    ) -> ResumeEditorResult:
        feedback = str(user_feedback or "").strip()
        request = {
            "resume_context": _context_for_model(context, include_current_resume=True),
            "user_feedback_for_this_edit": feedback or None,
            "output_schema": ResumeEditorResult.model_json_schema(),
        }
        response = self.llm.chat(
            [
                {"role": "system", "content": _EDITOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False),
                },
            ],
            temperature=0.25,
            json_mode=True,
        )
        return self._parse_with_repair(response.content)

    def review_pages(
        self,
        *,
        context: ResumeContext,
        current: ResumeEditorResult,
        render: ResumeRenderResult,
        user_feedback: str | None = None,
    ) -> ResumeEditorResult:
        """Let a multimodal model inspect actual pages and revise the same document."""
        text_payload = {
            "resume_context": _context_for_model(context, include_current_resume=False),
            "current_editor_result": current.model_dump(mode="json"),
            "rendered_pages": [
                {
                    "page_number": page.page_number,
                    "extracted_text": page.extracted_text,
                    "image_label": f"rendered-page-{page.page_number}",
                }
                for page in render.pages
            ],
            "user_feedback_for_this_review": str(user_feedback or "").strip() or None,
            "output_schema": ResumeEditorResult.model_json_schema(),
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(text_payload, ensure_ascii=False),
            }
        ]
        for page in render.pages:
            content.append({"type": "text", "text": f"rendered-page-{page.page_number}"})
            content.append(_image_content(page.image.path))
        try:
            response = self.llm.chat(
                [
                    {"role": "system", "content": _VISUAL_REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.2,
                json_mode=True,
            )
        except LLMError as exc:
            message = str(exc)
            if "image_url" in message or "expected `text`" in message:
                raise VisualReviewUnavailable(
                    "the configured visual-review endpoint accepts text only; "
                    "configure a multimodal OpenAI-compatible model"
                ) from exc
            raise ResumeEditorError(f"visual review failed: {exc}") from exc
        return self._parse_with_repair(response.content)

    def _parse_with_repair(self, raw: str) -> ResumeEditorResult:
        try:
            return _parse_editor_result(raw)
        except ResumeEditorError as first_error:
            repair_request = {
                "invalid_output": str(raw or ""),
                "validation_error": str(first_error),
                "output_schema": ResumeEditorResult.model_json_schema(),
            }
            response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "修复下面的简历编辑器输出，使它成为严格符合 output_schema 的 JSON。"
                            "保留原有简历内容和决定，不要借修复机会改写、总结或增加内容。"
                            "只返回 JSON，不要代码块。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(repair_request, ensure_ascii=False),
                    },
                ],
                temperature=0.0,
                json_mode=True,
            )
            try:
                return _parse_editor_result(response.content)
            except ResumeEditorError as second_error:
                raise ResumeEditorError(
                    f"resume editor JSON remained invalid after one repair: {second_error}"
                ) from second_error


def _parse_editor_result(raw: str) -> ResumeEditorResult:
    try:
        value = json.loads(str(raw or ""))
        return ResumeEditorResult.model_validate(value)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ResumeEditorError(f"resume editor returned invalid JSON: {exc}") from exc


def _context_for_model(
    context: ResumeContext,
    *,
    include_current_resume: bool,
) -> dict[str, Any]:
    """Remove storage-only identity and exact duplicate master text from model input."""
    payload = context.model_dump(mode="json", exclude_none=True)
    source = payload.get("master_source")
    if isinstance(source, dict):
        source.pop("source_path", None)
        if context.master_source.extracted_text == context.master_document.semantic_text:
            source.pop("extracted_text", None)
    if not include_current_resume:
        payload.pop("current_resume", None)
    return payload


def _image_content(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ResumeEditorError(f"visual review image does not exist: {source}")
    suffix = source.suffix.casefold()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"},
    }


_EDITOR_SYSTEM_PROMPT = """你是 OfferGuide 唯一的简历编辑器。你的任务是针对当前 JD，编辑候选人当前唯一的一份简历，而不是给建议、打分或生成第二个版本。

你会收到完整 Resume Context，其中 master 简历证据、Project Facts、当前 JD、当前草稿和用户反馈已经分开标注。

工作原则：
- 从当前岗位真正筛选候选人的信息出发，决定内容取舍、栏目内顺序、展开程度和强调。顶层栏目默认沿用已确认 master 的阅读顺序；除非用户明确要求，不要把实习经历、项目经历、教育背景等顶层栏目互相调换。不要套固定 bullet 数量、字数、页数或写作公式。
- 学校、公司、岗位、时间、项目、奖项和数字结果等硬事实只能来自 master 或 Project Facts，不能改写成不存在的经历。
- 可以加入与已有能力相邻、面试前能准备到可解释程度的基础知识或工具声明，但不能伪装成做过的项目、业绩或上线结果；每一条这类新增声明必须进入 preparation_notes。
- 明确决定每一处局部加粗。不要按冒号、开头关键词或固定数量自动加粗，也不要让强调切断正常阅读节奏。
- RichText.spans 只用于同一段连续文字中的局部强调，不代表多个 bullet。多个独立事实或成果需要分别表达时，使用多个 ResumeBlock；数量由当前内容决定，不套固定公式。
- 每个 ResumeEntry 的 rows 就是最终可见的条目抬头行。left 是主要阅读列；right 是同一行右对齐的紧凑元信息。日期、地点等适合 right，较长的学位、岗位、项目说明或奖项应由你放在下一条 left-only row，不能把多个语义挤进一个含糊字段。行数、文字和强调均由你根据当前内容决定。
- ResumeDocument 的 page_break_before、keep_title_with_first_entry、keep_header_with_first_block 和 divider_before 只是你明确作出的当前页面编排决定，不是全局规则。divider_before 只表示同一栏目中本条与前一条之间的分隔，不能用于栏目首条。需要增强相邻经历/项目的分隔时可以显式使用；首次编辑仍可保持自然排版。
- 如果提供了 current_resume 或用户反馈，修改它本身，保持同一份简历。

只返回符合 output_schema 的 JSON。不要 Markdown 代码块，不要 change log、ATS 字段、建议文件名、固定 warnings 或额外字段。editor_note 用几句话说明本次最重要的内容决定。"""


_VISUAL_REVIEW_SYSTEM_PROMPT = """你是 OfferGuide 简历编辑器的页面审阅阶段。你必须查看消息中按顺序附带的每一张实际渲染页面图片；坐标、页数或测试通过不能替代视觉判断。

同时使用 Resume Context、当前 ResumeDocument、页面提取文字和用户反馈，检查：
- 信息层级、阅读顺序、段落节奏和页面之间是否自然；
- 是否有大块无意义空白、孤立标题、内容断裂、过密或过疏；
- 局部加粗是否准确、连续、克制且没有造成字形或断行问题；
- 当前内容组织是否真正服务 JD，而不是技术细节越多越好；
你可以修改内容、rows 的左右行编排、顺序、强调、page_break_before、keep_title_with_first_entry、keep_header_with_first_block 和 divider_before，但仍然是在更新同一份简历。不要通过缩小字体、强塞固定页数或删除重要事实来解决页面问题。若页面已经合适，原样返回 document。

只返回符合 output_schema 的 JSON，不要额外字段或 Markdown。editor_note 说明这次页面审阅后实际调整了什么，或明确说明页面无需再改。"""


__all__ = [
    "PreparationNote",
    "ResumeEditor",
    "ResumeEditorError",
    "ResumeEditorResult",
    "VisualReviewUnavailable",
]
