"""Sub-agents — domain-specialized ReAct loops the main agent delegates to.

The main agent (in `offerguide.agent.loop`) keeps a small tool list. When
a task falls into a specialized domain (discovery / evaluation / outcome
review), the main agent calls `delegate_*(goal)`, which spawns a sub-agent
of the appropriate kind. The sub-agent has its own tool subset + system
prompt + iteration cap, runs to completion, and returns a structured
result.

This mirrors hermes-agent's `delegate_tool.py` pattern: sub-agent is a
fresh AIAgent with restricted tools + bounded iterations.

LOCK 1 (from user, 2026-05-11):
  同 domain 多于 3-4 个 tool → 升级 sub-agent. 主 agent tool ≤ 15.

LOCK 2 (from user):
  Evolution 是核心架构, 不是 optional.
"""

from __future__ import annotations

from .base import SubAgent, SubAgentResult

__all__ = ["SubAgent", "SubAgentResult"]
