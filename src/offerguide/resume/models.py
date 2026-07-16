"""Semantic models for the one current resume and all of its input context."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .master import MasterResumeDocument, MasterResumeSource


class RichTextSpan(BaseModel):
    """A text fragment whose semantic emphasis is chosen explicitly by the model."""

    model_config = ConfigDict(extra="forbid")

    text: str
    emphasis: bool = False

    @field_validator("text")
    @classmethod
    def _text_must_not_be_empty(cls, value: str) -> str:
        if value == "":
            raise ValueError("rich text span must not be empty")
        return value


class RichText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spans: list[RichTextSpan] = Field(min_length=1)

    @model_validator(mode="after")
    def _content_must_not_be_blank(self) -> RichText:
        if not self.plain_text.strip():
            raise ValueError("rich text must contain visible content")
        return self

    @property
    def plain_text(self) -> str:
        return "".join(span.text for span in self.spans)


class ResumeHeader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: RichText
    lines: list[RichText] = Field(default_factory=list)


class ResumeBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["paragraph", "bullet"] = Field(
        description="一个完整段落或一个完整 bullet；多个独立事实应使用多个 block。"
    )
    content: RichText


class ResumeEntryRow(BaseModel):
    """One model-authored row in an entry header."""

    model_config = ConfigDict(extra="forbid")

    left: RichText = Field(
        description=("该行的主要阅读内容，按正常阅读顺序完整书写；模型决定其中每一处强调。")
    )
    right: RichText | None = Field(
        default=None,
        description=(
            "同一行靠右对齐的紧凑元信息，适合日期或地点。较长且需要连续阅读的"
            "学位、岗位、项目说明或奖项应放在另一条只有 left 的行中。"
        ),
    )


class ResumeEntry(BaseModel):
    """One explicitly structured experience, project, education item, or flat item."""

    model_config = ConfigDict(extra="forbid")

    rows: list[ResumeEntryRow] = Field(
        default_factory=list,
        description=(
            "条目开头按视觉顺序排列的行；行数由当前内容决定。每行可只使用左列，"
            "也可把一项紧凑元信息放在右列。"
        ),
    )
    blocks: list[ResumeBlock] = Field(
        default_factory=list,
        description="紧随条目抬头的正文段落或项目符号，顺序即最终阅读顺序。",
    )
    page_break_before: bool = False
    keep_header_with_first_block: bool = False
    divider_before: bool = Field(
        default=False,
        description=(
            "是否在同一栏目中本条与前一条之间显示分隔线；栏目首条没有前一条，"
            "因此该值对首条不生效。"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_heading_pair(cls, value: object) -> object:
        """Read pre-row documents without exposing their fields in the current schema."""
        if not isinstance(value, dict) or "rows" in value:
            return value
        if "heading" not in value and "aside" not in value:
            return value
        upgraded = dict(value)
        heading = upgraded.pop("heading", None)
        aside = upgraded.pop("aside", None)
        if heading is not None:
            upgraded["rows"] = [{"left": heading, "right": aside}]
        elif aside is not None:
            upgraded["rows"] = [{"left": aside}]
        else:
            upgraded["rows"] = []
        if "keep_header_with_first_block" not in upgraded:
            upgraded["keep_header_with_first_block"] = upgraded.pop(
                "keep_heading_with_first_block", False
            )
        else:
            upgraded.pop("keep_heading_with_first_block", None)
        return upgraded

    @model_validator(mode="after")
    def _entry_must_have_content(self) -> ResumeEntry:
        if not self.rows and not self.blocks:
            raise ValueError("resume entry requires at least one header row or block")
        return self


class ResumeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: RichText
    entries: list[ResumeEntry] = Field(min_length=1)
    page_break_before: bool = False
    keep_title_with_first_entry: bool = False

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_keep_field(cls, value: object) -> object:
        if not isinstance(value, dict) or "keep_heading_with_first_block" not in value:
            return value
        upgraded = dict(value)
        upgraded.setdefault(
            "keep_title_with_first_entry",
            upgraded.pop("keep_heading_with_first_block"),
        )
        return upgraded


class ResumeDocument(BaseModel):
    """The complete semantic resume; the renderer must not infer missing meaning."""

    model_config = ConfigDict(extra="forbid")

    header: ResumeHeader
    sections: list[ResumeSection] = Field(min_length=1)


class ApplicationFormAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str

    @field_validator("question", "answer")
    @classmethod
    def _must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("form answer fields must not be blank")
        return cleaned


class ApplicationPackage(BaseModel):
    """The complete, minimal application copy paired with one resume draft."""

    model_config = ConfigDict(extra="forbid")

    message: str | None = None
    form_answers: list[ApplicationFormAnswer] = Field(default_factory=list)
    pre_submit_checks: list[str] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _normalize_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("pre_submit_checks")
    @classmethod
    def _checks_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("pre-submit checks must not be blank")
        return cleaned


class ResumeJobContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: int | str
    company: str
    title: str
    jd_text: str
    source_url: str | None = None
    verified_information: list[str] = Field(default_factory=list)

    @field_validator("company", "title", "jd_text")
    @classmethod
    def _required_job_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("job company, title, and jd_text must not be blank")
        return value


class ProjectFact(BaseModel):
    """Structured candidate evidence supplied by the Project Vault."""

    model_config = ConfigDict(extra="forbid")

    project_id: int | str
    title: str
    facts: list[str] = Field(min_length=1)
    do_not_claim: list[str] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def _title_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("project title must not be blank")
        return value


class ResumeContext(BaseModel):
    """The single, provenance-preserving input assembled for resume editing."""

    model_config = ConfigDict(extra="forbid")

    job: ResumeJobContext
    master_source: MasterResumeSource
    master_document: MasterResumeDocument
    project_facts: list[ProjectFact] = Field(default_factory=list)
    feedback: list[str] = Field(default_factory=list)
    current_resume: ResumeDocument | None = None

    @model_validator(mode="after")
    def _master_document_must_match_confirmed_source(self) -> ResumeContext:
        if not self.master_document.confirmed_by_user:
            raise ValueError("master semantic document must be confirmed by the user")
        if self.master_source.sha256 != self.master_document.source_sha256:
            raise ValueError("master semantic document does not match the source PDF")
        return self


__all__ = [
    "ApplicationFormAnswer",
    "ApplicationPackage",
    "ProjectFact",
    "ResumeBlock",
    "ResumeContext",
    "ResumeDocument",
    "ResumeEntry",
    "ResumeEntryRow",
    "ResumeHeader",
    "ResumeJobContext",
    "ResumeSection",
    "RichText",
    "RichTextSpan",
]
