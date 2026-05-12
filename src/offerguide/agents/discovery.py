"""Discovery sub-agent — bounded ReAct loop scoped to discovery tools.

This file does NOT contain a 100-line "你这个用户怎么想" persona blob.
That would be the developer (me) putting opinions in the user's mouth.

Sub-agent system prompt is minimal + generic. The actual task framing
(what to look for / what to skip / specific strategy for this call) is
authored by the **main agent at delegation time** and passed as the user
message via the `delegate_discovery(goal, instructions)` tool.

Same pattern as Claude Code's Agent tool: caller writes the prompt for
the spawned sub-agent. Tool exists, sub-agent runtime exists, but the
intent comes from the calling agent (or the human) at call time, not
baked into the file.
"""

from __future__ import annotations

import logging

from .base import SubAgent

log = logging.getLogger(__name__)


# Minimal, generic. No opinions about job hunting, the user, or strategy.
# Those come from the main agent's delegate_discovery(...) instructions arg.
DISCOVERY_SYSTEM_PROMPT = """你是一个 sub-agent. 主 agent 已经在 user message 里写好了**这次具体要干什么** + 必要背景.

你只做这一次 user message 描述的事:
- 看可用工具 (在 tool list 里), 选合适的调
- 调完看返回值, 决定下一步
- 完成后调 `done(summary="...")` 上报

不要做 user message 没要求的额外事. 不要扩大 scope.
"""


class DiscoverySubAgent(SubAgent):
    """Discovery sub-agent — runs with discovery tool group."""
    SYSTEM_PROMPT = DISCOVERY_SYSTEM_PROMPT
    GROUP = "discovery"
    NAME = "discovery"
