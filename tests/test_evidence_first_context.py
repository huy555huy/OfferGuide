from __future__ import annotations

from pathlib import Path

from offerguide.agent_runtime.context import EVIDENCE_FIRST_POLICY
from offerguide.agent_runtime.tools import ALL_TOOL_SCHEMAS

ROOT = Path(__file__).parent.parent


def test_runtime_instructions_center_the_three_real_job_search_tasks():
    instructions = (ROOT / "src/offerguide/agent_runtime/instructions.md").read_text(
        encoding="utf-8"
    )
    assert "## 证据优先" in instructions
    assert "运行时会注入共享的证据政策" in instructions
    assert "事实、推断、未知" in instructions
    assert "投递前：找到值得投的岗位" in instructions
    assert "投递中：完成一份可用的投递包" in instructions
    assert "投递后：查找真实面经并回答原题" in instructions
    assert "不生成替代题、预测题或泛化准备内容" in instructions
    assert "内容取舍、展开程度和篇幅服从当前 JD" in instructions


def test_shared_evidence_policy_is_injected_into_runtime_prompts():
    assert "证据优先" in EVIDENCE_FIRST_POLICY
    assert "未知优先" in EVIDENCE_FIRST_POLICY
    assert "不能驱动批量投递" in EVIDENCE_FIRST_POLICY


def test_main_tools_stay_on_job_search_instead_of_self_evolution():
    schemas = {s["function"]["name"]: s["function"] for s in ALL_TOOL_SCHEMAS}

    research_jobs = schemas["research_jobs"]
    research_text = (
        research_jobs["description"]
        + " "
        + research_jobs["parameters"]["properties"]["intent"]["description"]
    )
    assert "sole JobDiscoveryAgent" in research_text
    assert "Do not turn assumptions" in research_text

    assert "research_interview" in schemas
    assert "prepare_application" in schemas
    assert "revise_application" in schemas
    assert "rerender_application" in schemas
    assert "read_artifact" in schemas
    assert "detect_evolution_candidates" not in schemas
    assert "evolve_skill" not in schemas
    assert "run_release_cycle" not in schemas


def test_goal_progress_does_not_treat_silence_as_rejection():
    goals_source = (ROOT / "src/offerguide/goals.py").read_text(encoding="utf-8")
    assert "apps_silent_7d" not in goals_source
    assert "apps_silent_14d" not in goals_source
    assert "大概率挂" not in goals_source


def test_runtime_instructions_do_not_turn_internal_state_into_the_goal():
    instructions = (ROOT / "src/offerguide/agent_runtime/instructions.md").read_text(
        encoding="utf-8"
    )
    assert (
        "worldview、goals、work items 和 events 是可选的持久化手段，不是目标"
    ) in instructions
    assert "不要求每次结束写 reflection" in instructions
    assert "用户当前明确请求优先于过期的 agenda" in instructions
    assert "不要主动运行 Skill 自进化" in instructions
