from __future__ import annotations

from pathlib import Path

from offerguide.agent_runtime.context import EVIDENCE_FIRST_POLICY
from offerguide.agent_runtime.tools import ALL_TOOL_SCHEMAS


ROOT = Path(__file__).parent.parent


def test_harness_instructions_make_goal_and_user_state_unknown_until_evidenced():
    instructions = (ROOT / "src/offerguide/agent_runtime/instructions.md").read_text(
        encoding="utf-8"
    )
    assert "## 证据优先" in instructions
    assert "运行时会注入共享的证据政策" in instructions
    assert "事实、推断、未知分开" in instructions
    assert "帮用户拿到 2026 暑期 AI Agent / LLM 应用岗 offer" not in instructions


def test_shared_evidence_policy_is_injected_into_runtime_prompts():
    assert "证据优先" in EVIDENCE_FIRST_POLICY
    assert "未知优先" in EVIDENCE_FIRST_POLICY
    assert "不能驱动批量投递" in EVIDENCE_FIRST_POLICY


def test_harness_tool_descriptions_require_grounded_actions():
    schemas = {s["function"]["name"]: s["function"] for s in ALL_TOOL_SCHEMAS}

    discover = schemas["discover_jobs"]
    discover_text = (
        discover["description"]
        + " "
        + discover["parameters"]["properties"]["criteria"]["description"]
    )
    assert "explicit evidence" in discover_text
    assert "Do not invent missing preferences" in discover_text

    schedule = schemas["schedule_next_wake"]["description"]
    assert "concrete event" in schedule
    assert "guessed user state" in schedule

    assert "capture_project" in schemas
    assert "save_project_record" in schemas
    assert "read_artifact" in schemas
    save_project = schemas["save_project_record"]["description"]
    assert "Do not invent missing metrics" in save_project
    read_artifact = schemas["read_artifact"]["description"]
    assert "without sending the user to hunt through pages" in read_artifact


def test_goals_template_does_not_treat_silence_as_rejection():
    template = (ROOT / "src/offerguide/ui/templates/goals.html").read_text(
        encoding="utf-8"
    )
    assert "未记录新进展" in template
    assert "大概率挂" not in template


def test_harness_instructions_are_chat_first_and_result_oriented():
    instructions = (ROOT / "src/offerguide/agent_runtime/instructions.md").read_text(
        encoding="utf-8"
    )
    assert "主入口是 Agent Chat" in instructions
    assert "页面只是展示层" in instructions
    assert "capture_project" in instructions
    assert "save_project_record" in instructions
    assert "read_artifact" in instructions
    assert "skill_run_id" in instructions
