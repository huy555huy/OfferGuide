"""Evaluation sub-agent — bounded ReAct loop scoped to evaluation tools.

Same minimal pattern as DiscoverySubAgent. The "evaluation philosophy"
is NOT baked here — it's what the main agent writes when it calls
`delegate_evaluation(goal, instructions)`. Sub-agent just executes the
ask using available tools.
"""

from __future__ import annotations

import logging

from .base import SubAgent

log = logging.getLogger(__name__)


DISCOVERY_SYSTEM_PROMPT_NOTE = "see DiscoverySubAgent — same pattern"

EVALUATION_SYSTEM_PROMPT = """你是一个 sub-agent. 主 agent 已经在 user message 里写好了**这次要评估什么** (某个 job 打分 / 写投递包 / 对比几个 / 看简历缺口) + 必要背景.

你只做这一次 user message 要的事:
- 看可用工具 (score_job / generate_apply_pack / generate_interview_prep / tailor_resume / find_resume_gaps / compare_jobs / read_job), 选合适的调
- 每次 LLM-backed tool 有 cost — 别盲目链调
- 完成后 `done(summary="...")` 上报: 用户能 act 的判断, 不是 "我调了哪个 SKILL"

不要扩大 scope. 用户没要 tailor_resume 就别 tailor.
"""


class EvaluationSubAgent(SubAgent):
    """Evaluation sub-agent — runs with evaluation tool group."""
    SYSTEM_PROMPT = EVALUATION_SYSTEM_PROMPT
    GROUP = "evaluation"
    NAME = "evaluation"
