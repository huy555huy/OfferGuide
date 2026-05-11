"""SubAgent base — a bounded ReAct loop with restricted tool group.

A sub-agent is a fresh LLM conversation focused on one domain (discovery,
evaluation, outcome review). The main agent triggers it via a
``delegate_*`` tool, the sub-agent runs to completion, returns a single
text answer + structured trajectory.

Why sub-agents (not flat tools on main):
- LOCK 1: when one domain has more than 3-4 tools, the main agent's tool
  list grows past "I can see all my options". Sub-agent encapsulates.
- Bounded iter cap (≤ 6 by default) — sub-agent can't run away.
- Separate cost accounting — main agent gets cost back as one number.
- Isolated context — sub-agent doesn't see main agent's whole snapshot.

Inspired by hermes-agent's ``tools/delegate_tool.py`` (which spawns a
fresh ``AIAgent`` with restricted toolset + iter cap). Trimmed for our
needs (single-process, no ACP/credential pool sharing).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..llm.client import _parse_tool_arguments

if TYPE_CHECKING:
    from ..llm import LLMClient
    from ..memory import Store
    from ..tools.registry import ToolGroup, ToolRegistry

log = logging.getLogger(__name__)


# Per-sub-agent caps. Sub-agents should converge fast or fail visibly.
DEFAULT_SUB_AGENT_MAX_ITER = 6
DEFAULT_SUB_AGENT_TIMEOUT_S = 120.0

# Circuit breaker: same (tool, args) repeated this many times = abort.
# Anthropic agent guide #1 production failure mode.
SUB_AGENT_REPEAT_THRESHOLD = 3

# Tool result cap fed back to the model per turn. Larger results would
# bloat the next decision's prompt.
SUB_AGENT_TOOL_RESULT_CAP = 4000


@dataclass
class SubAgentEvent:
    """One observable step in the sub-agent's trajectory."""
    kind: str  # 'tool_call' | 'tool_result' | 'thinking' | 'final' | 'error'
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentResult:
    """Return shape for SubAgent.run."""
    final_answer: str
    iterations: int
    tool_calls_made: int
    cost_usd: float
    duration_ms: int
    events: list[SubAgentEvent] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        """Compact dict for main agent's tool_result payload."""
        return {
            "final_answer": self.final_answer,
            "iterations": self.iterations,
            "tool_calls_made": self.tool_calls_made,
            "cost_usd": round(self.cost_usd, 5),
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class SubAgent:
    """Bounded ReAct loop scoped to one tool group.

    Subclasses override ``SYSTEM_PROMPT`` and (optionally) ``group``.
    Construction takes the shared LLM client + tool registry + store.
    Tool dispatch goes through the registry — every tool the sub-agent
    can see is in its group + shared.

    Usage:
        sub = DiscoverySubAgent(llm=..., registry=..., store=..., settings=...)
        result = sub.run(goal="找今天 3 个 AI Agent 暑期实习")
        # result.final_answer / result.events / result.cost_usd
    """

    # Override in subclass.
    SYSTEM_PROMPT: str = "你是一个专门的 sub-agent. 完成主 agent 委托的任务后调 done()."
    GROUP: "ToolGroup" = "shared"  # type: ignore[assignment]
    NAME: str = "sub_agent"

    def __init__(
        self,
        *,
        llm: "LLMClient",
        registry: "ToolRegistry",
        store: "Store",
        settings: Any = None,
        runtime: Any = None,
        skills: list[Any] | None = None,
        user_profile_text: str | None = None,
        max_iter: int = DEFAULT_SUB_AGENT_MAX_ITER,
        timeout_s: float = DEFAULT_SUB_AGENT_TIMEOUT_S,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.store = store
        self.settings = settings
        self.runtime = runtime
        self.skills = skills or []
        self.user_profile_text = user_profile_text
        self.max_iter = max(1, int(max_iter))
        self.timeout_s = float(timeout_s)

    # ── Public ────────────────────────────────────────────────

    def run(self, *, goal: str, context: str = "") -> SubAgentResult:
        """Execute one bounded ReAct loop.

        ``goal`` — the high-level task from the main agent.
        ``context`` — optional extra info (e.g. recent state snapshot)
                      the sub-agent should see in its first user message.
        """
        t0 = time.monotonic()
        events: list[SubAgentEvent] = []
        total_cost = 0.0
        tool_calls_made = 0

        tool_schemas = self.registry.get_schemas(self.GROUP, include_shared=True)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(goal, context)},
        ]

        # Circuit breaker tracking
        recent_calls: list[tuple[str, str]] = []

        iteration = 0
        final_answer = ""
        for iteration in range(1, self.max_iter + 1):
            if time.monotonic() - t0 > self.timeout_s:
                events.append(SubAgentEvent(
                    "error", {"message": f"timeout > {self.timeout_s}s"},
                ))
                final_answer = f"(sub-agent {self.NAME} timed out at {self.timeout_s}s)"
                break

            try:
                resp = self.llm.chat_with_tools(
                    messages=messages, tools=tool_schemas, temperature=0.3,
                )
            except Exception as e:
                events.append(SubAgentEvent(
                    "error", {"message": f"LLM call failed: {e}"},
                ))
                final_answer = f"(LLM error in sub-agent {self.NAME}: {e})"
                break

            total_cost += resp.cost_usd or 0.0

            assistant_msg = (
                dict(resp.assistant_message)
                if resp.assistant_message is not None
                else {"role": "assistant", "content": resp.content or ""}
            )
            if resp.tool_calls and "tool_calls" not in assistant_msg:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id, "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments_raw or json.dumps(
                                tc.arguments, ensure_ascii=False,
                            ),
                        },
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            if resp.content:
                events.append(SubAgentEvent("thinking", {"text": resp.content[:500]}))

            if not resp.tool_calls:
                final_answer = resp.content or "(sub-agent 直接 final, 无 tool call)"
                events.append(SubAgentEvent(
                    "final", {"text": final_answer[:500], "iteration": iteration},
                ))
                break

            # Dispatch each tool call
            done_called = False
            for tc in resp.tool_calls:
                tool_calls_made += 1

                # Circuit breaker: same (name, args_sig) ≥ threshold = abort
                args_sig = json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)[:200]
                recent_calls.append((tc.name, args_sig))
                if recent_calls.count((tc.name, args_sig)) >= SUB_AGENT_REPEAT_THRESHOLD:
                    events.append(SubAgentEvent("error", {
                        "message": f"circuit breaker: {tc.name} repeated {SUB_AGENT_REPEAT_THRESHOLD}x",
                    }))
                    final_answer = (
                        f"(sub-agent aborted: {tc.name} called {SUB_AGENT_REPEAT_THRESHOLD}x with same args)"
                    )
                    done_called = True
                    break

                events.append(SubAgentEvent("tool_call", {
                    "name": tc.name, "arguments": tc.arguments,
                }))

                if tc.name == "done":
                    final_answer = str(tc.arguments.get("summary") or "").strip()
                    if not final_answer:
                        final_answer = "(sub-agent done; no summary provided)"
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps({"ack": True}, ensure_ascii=False),
                    })
                    events.append(SubAgentEvent("final", {
                        "text": final_answer[:500], "iteration": iteration,
                    }))
                    done_called = True
                    break

                tool_result = self._dispatch_tool(tc.name, tc.arguments)
                events.append(SubAgentEvent("tool_result", {
                    "name": tc.name,
                    "result_preview": tool_result[:300],
                }))
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": tool_result[:SUB_AGENT_TOOL_RESULT_CAP],
                })

            if done_called:
                break

        if not final_answer:
            final_answer = f"(sub-agent {self.NAME} hit max_iter={self.max_iter} 没自然 final)"
            events.append(SubAgentEvent("error", {
                "message": f"max_iter ({self.max_iter}) reached",
            }))

        return SubAgentResult(
            final_answer=final_answer,
            iterations=iteration,
            tool_calls_made=tool_calls_made,
            cost_usd=round(total_cost, 5),
            duration_ms=int((time.monotonic() - t0) * 1000),
            events=events,
        )

    # ── Internals ────────────────────────────────────────────

    def _build_user_prompt(self, goal: str, context: str) -> str:
        parts = [f"# 任务 (goal)\n{goal}"]
        if context:
            parts.append(f"\n## 背景信息\n{context}")
        parts.append("\n按工具完成任务. 完成后调 done(summary='...') 上报.")
        return "\n".join(parts)

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool through the registry. Always returns JSON str."""
        return self.registry.dispatch(
            name, args,
            store=self.store,
            settings=self.settings,
            runtime=self.runtime,
            skills=self.skills,
            user_profile_text=self.user_profile_text,
        )


# ── Universal 'done' tool (every sub-agent must be able to declare done) ──
#
# Each sub-agent's tool group should include 'done' (we register it as
# 'shared' so it's universally available). Sub-agent code above handles
# done specially — it terminates the loop with the provided summary.


def _done_handler(args: dict[str, Any], **runtime_kwargs: Any) -> str:
    """Sentinel tool — handled inline by SubAgent.run, never actually
    dispatched through the registry. But registered for schema discovery."""
    # If we ever DO get here (e.g. main agent calls done), echo back.
    summary = args.get("summary") or "(no summary)"
    return json.dumps({"ack": True, "summary": summary}, ensure_ascii=False)


def register_universal_tools(registry: "ToolRegistry") -> None:
    """Register the 'done' tool. Called by the tools package loader."""
    registry.register(
        name="done",
        group="shared",
        schema={
            "name": "done",
            "description": (
                "Signal that the sub-agent has finished its delegated task. "
                "Pass a brief 'summary' describing what was accomplished. "
                "After calling done(), the loop terminates and the summary "
                "is returned to the main agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "1-3 sentence summary of what you did + key findings.",
                    },
                },
                "required": ["summary"],
            },
        },
        handler=_done_handler,
        description="Sub-agent terminator: declare task done with a summary.",
    )
