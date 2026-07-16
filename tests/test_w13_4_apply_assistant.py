"""Minimal apply-assistant contract used by the one resume workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import offerguide
from offerguide.llm import LLMResponse
from offerguide.skills import SkillRuntime, discover_skills, load_skill
from offerguide.skills.apply_assistant.helpers import ApplyPackage

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def _valid_package() -> dict:
    return {
        "message": (
            "应用统计硕士在读，最近完成了带证据闭环的研究 Agent，"
            "与岗位中的 Agent runtime 工作直接相关。"
        ),
        "form_answers": [
            {
                "question": "为什么申请这个岗位？",
                "answer": (
                    "我现有项目中的状态管理、工具路由和证据闭环，"
                    "与该岗位负责的 Agent runtime 工作直接相关。"
                ),
            }
        ],
        "pre_submit_checks": ["确认联系方式和当前定制 PDF 附件。"],
    }


def test_apply_assistant_skill_is_current_and_discoverable() -> None:
    spec = load_skill(SKILLS_ROOT / "apply_assistant")
    assert spec.name == "apply_assistant"
    assert spec.version == "0.3.0"
    assert set(spec.inputs) == {"company", "role_focus", "job_text", "user_profile"}
    assert "application-package" in spec.tags
    assert spec.name in {skill.name for skill in discover_skills(SKILLS_ROOT)}


def test_apply_package_accepts_message_answers_and_checks() -> None:
    package = ApplyPackage.model_validate(_valid_package())

    assert package.message
    assert package.form_answers[0].question == "为什么申请这个岗位？"
    assert package.pre_submit_checks == ["确认联系方式和当前定制 PDF 附件。"]


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "直接沟通文案", "form_answers": [], "pre_submit_checks": []},
        {
            "message": None,
            "form_answers": [{"question": "可入职时间？", "answer": "待用户确认"}],
            "pre_submit_checks": [],
        },
    ],
)
def test_apply_package_accepts_either_kind_of_usable_material(payload: dict) -> None:
    ApplyPackage.model_validate(payload)


def test_apply_package_allows_no_extra_copy_when_channel_has_no_known_fields() -> None:
    package = ApplyPackage.model_validate(
        {"message": None, "form_answers": [], "pre_submit_checks": []}
    )

    assert package.message is None
    assert package.form_answers == []


@pytest.mark.parametrize(
    ("payload", "error_fragment"),
    [
        (
            {"message": "x", "form_answers": [], "pre_submit_checks": [], "legacy": True},
            "legacy",
        ),
        (
            {
                "message": None,
                "form_answers": [{"question": "问题", "answer": "   "}],
                "pre_submit_checks": [],
            },
            "form answer fields must not be blank",
        ),
        (
            {"message": "x", "form_answers": [], "pre_submit_checks": ["   "]},
            "pre-submit checks must not be blank",
        ),
    ],
)
def test_apply_package_rejects_invalid_or_legacy_output(
    payload: dict,
    error_fragment: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        ApplyPackage.model_validate(payload)

    assert error_fragment in str(exc_info.value)


def test_apply_assistant_runtime_output_round_trips_through_current_schema(tmp_path) -> None:
    payload = _valid_package()

    class StubLLM:
        def chat(self, messages, **_kwargs):
            return LLMResponse(
                content=json.dumps(payload, ensure_ascii=False),
                model="stub",
            )

    store = offerguide.Store(tmp_path / "apply-assistant.db")
    store.init_schema()
    runtime = SkillRuntime(StubLLM(), store)  # type: ignore[arg-type]
    spec = load_skill(SKILLS_ROOT / "apply_assistant")

    result = runtime.invoke(
        spec,
        {
            "company": "字节跳动",
            "role_focus": "AI Agent 后端实习",
            "job_text": "负责 Agent runtime、工具调用和状态管理。",
            "user_profile": json.dumps(
                {
                    "final_resume": "候选人当前定制简历全文",
                    "preparation_notes": [],
                },
                ensure_ascii=False,
            ),
        },
    )

    package = ApplyPackage.model_validate(result.parsed)
    assert package.message == payload["message"]
    with store.connect() as conn:
        row = conn.execute(
            "SELECT skill_name, output_json FROM skill_runs WHERE id = ?",
            (result.skill_run_id,),
        ).fetchone()
    assert row[0] == "apply_assistant"
    assert ApplyPackage.model_validate_json(row[1]) == package
