"""Single-threaded master loop for OfferGuide's application agent.

Anthropic's published agent loop pattern (verbatim from research):

    WHILE NOT task_complete:
      1. Gather context (history + memory + tool results)
      2. response = model.messages.create(messages, tools, context_management)
      3. IF stop_reason == 'tool_use':
           execute tools sequentially → ToolResult blocks → append
         ELSE:
           return final text → mark complete

That's it. No planner-executor-reflector chain. No "if-then-else what to do
next" hardcoded in the loop. The model decides moment-by-moment what to call,
when to ask user, when to update memory, when to stop.

Our loop adds two runtime responsibilities (because OpenAI-compat
DeepSeek doesn't do them server-side):
- Token estimation + compaction (in context.py)
- Tool-result clearing on long runs (in context.py)

Plus persistence:
- Insert harness_runs row at start, update at end (id flows into deps so
  tools can record references)
- Track tool calls and supporting-tool cost for /debug telemetry
"""

from __future__ import annotations

import json as _json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..llm import BudgetExceeded, LLMError, enforce_daily_budget
from . import _schema
from .context import ContextManager, SystemFacts
from .tools import ALL_TOOL_SCHEMAS, AgentRuntimeDeps, dispatch

EventCallback = Callable[[Mapping[str, Any]], None]

log = logging.getLogger(__name__)


# Per-run hard cap so a runaway agent doesn't loop forever. Tuned higher
# than W14 (8 iter) because the new agent is fully in driver seat —
# multi-step reasoning is expected.
DEFAULT_MAX_ITERATIONS = 20

# Per-tool-call timeout safety net — if a tool blocks > N seconds we
# don't have a way to interrupt cleanly with sync httpx, but we log it.
SOFT_TOOL_TIMEOUT_S = 90

# Status values written to harness_runs.status. 'truncated' (W15-review fix)
# distinguishes "agent hit budget cap mid-thought" from "agent finished
# cleanly" — Mission Control / /debug should colour these differently.
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_TRUNCATED = "truncated"


@dataclass
class TriggerEvent:
    """Input that frames one conversation-agent run."""

    kind: str
    """Production requests use ``user_input``; tests may use another label."""

    detail: dict[str, Any] = field(default_factory=dict)
    """Arbitrary structured context, normally including ``message``."""

    def render_initial_user_message(self) -> str:
        """One sentence the loop puts as the first user message of the run."""
        if self.kind == "user_input":
            msg = self.detail.get("message", "")
            return f"用户消息: {msg}"
        return f"未知触发: {self.kind}"


@dataclass
class RunResult:
    """What one loop run produced. Consumed by trigger system + UI."""

    run_id: int | None
    iterations: int
    final_text: str
    tool_call_log: list[str]
    cost_usd: float
    latency_ms: int
    finish_reason: str
    error_text: str | None = None


DEFAULT_TEMPERATURE = 0.4
"""Q1 (W15.13 review answer): exposed as a parameter so callers can
override per-trigger if they want creativity vs determinism trade-offs.
We don't bake automatic per-trigger logic here — that's editorial
encoding the runtime shouldn't do. Callers (chat endpoint, scheduler)
decide based on their context."""


def run(
    *, trigger: TriggerEvent, deps: AgentRuntimeDeps,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    system_facts: SystemFacts | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    on_event: EventCallback | None = None,
    cancel_event: Any = None,
) -> RunResult:
    """One agent run. Returns when the model stops calling tools or hits
    max_iterations.

    Side effects: inserts harness_runs row; writes tool results to
    domain tables via dispatched tools.

    Guarantees (W15.12 review fixes):
    - try/finally around the loop ensures ``_end_run`` ALWAYS commits a
      terminal status to harness_runs (Bug 3). Otherwise an exception in
      ctx mgmt or tool dispatch leaves the row in 'running' forever.
    - status reflects ``finish_reason`` accurately (Bug 1):
      end_turn → 'ok', max_iterations → 'truncated', llm_error/crash → 'error'
    - ``final_text`` accumulates ``resp.content`` from EVERY iteration
      (Bug 2), so reasoning produced alongside tool_calls isn't lost.
    - supporting-tool cost accumulated in ``deps.extra_cost_usd`` is added to
      harness_runs.cost_usd.

    Args:
        temperature: LLM sampling temperature. Defaults to 0.4.
    """
    if deps.llm is None:
        return RunResult(
            run_id=None, iterations=0, final_text="",
            tool_call_log=[], cost_usd=0.0, latency_ms=0,
            finish_reason="no_llm", error_text="LLMClient is None",
        )

    # Ensure runtime telemetry tables exist (idempotent)
    _schema.init_agent_runtime_schema(deps.store)

    # W15.15 — daily budget guard. Cheap (1 SQL aggregate); refuses to
    # start a run if today's LLM spend already exceeded the cap.
    try:
        enforce_daily_budget(deps.store)
    except BudgetExceeded as e:
        log.warning("loop: refusing run, %s", e)
        return RunResult(
            run_id=None, iterations=0, final_text="",
            tool_call_log=[], cost_usd=0.0, latency_ms=0,
            finish_reason="budget_exceeded", error_text=str(e),
        )

    work_item_ids = _schema.prepare_trigger_work_items(
        deps.store, kind=trigger.kind, detail=trigger.detail,
    )

    # Insert harness_runs row early so tools can reference deps.current_run_id
    run_id = _start_run(deps, trigger)
    deps.current_run_id = run_id
    _schema.attach_work_items_to_run(
        deps.store, run_id=run_id, work_item_ids=work_item_ids,
    )
    # Reset cost sink for this run (in case deps was reused across runs)
    deps.extra_cost_usd = 0.0

    tool_call_log: list[str] = []
    total_cost_usd = 0.0
    final_text_parts: list[str] = []
    finish_reason = "max_iterations"
    error_text: str | None = None
    iteration = 0
    crash_text: str | None = None
    t0 = time.monotonic()

    def _emit(kind: str, **payload: Any) -> None:
        if on_event is None:
            return
        try:
            on_event({"kind": kind, "run_id": run_id, **payload})
        except Exception as e:
            log.debug("on_event callback raised: %s", e)

    try:
        # Build initial messages — also wrapped in try so a corrupt
        # worldview file doesn't strand the harness_runs row in 'running'.
        ctx_mgr = ContextManager(
            llm=deps.llm,
            memory=deps.memory_store,
            store=deps.store,
        )
        system_msg_text = ctx_mgr.build_initial_system(system_facts=system_facts)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg_text},
            {"role": "user", "content": trigger.render_initial_user_message()},
        ]

        for iteration in range(1, max_iterations + 1):
            if cancel_event is not None and cancel_event.is_set():
                _emit("_cancelled", iteration=iteration)
                finish_reason = "cancelled"
                break

            # Pre-call context management
            messages, cleared = ctx_mgr.maybe_clear_tool_results(messages)
            if cleared > 0:
                log.info("loop iter %d: cleared %d tool results", iteration, cleared)
            messages, compacted = ctx_mgr.maybe_compact(messages)
            if compacted:
                log.info("loop iter %d: compaction ran", iteration)

            # Call the model
            try:
                resp = deps.llm.chat_with_tools(
                    messages=messages,
                    tools=ALL_TOOL_SCHEMAS,
                    temperature=temperature,
                    tool_choice="auto",
                )
            except LLMError as e:
                log.warning("loop iter %d: LLM error: %s", iteration, e)
                finish_reason = "llm_error"
                error_text = str(e)
                _emit("error", iteration=iteration, message=str(e))
                break

            ctx_mgr.last_prompt_tokens = resp.prompt_tokens
            total_cost_usd += resp.cost_usd or 0.0

            _emit(
                "thinking",
                iteration=iteration,
                text=resp.content or "",
                will_call_tools=bool(resp.tool_calls),
                tool_call_names=[tc.name for tc in resp.tool_calls],
            )

            # Bug 2 fix: accumulate ANY content from this iteration —
            # some models return reasoning text alongside tool_calls. If the
            # loop ends at max_iterations, this preserves the trail.
            if resp.content:
                final_text_parts.append(resp.content)

            # Append assistant message. Prefer LLMClient's prepared history
            # message so provider-specific reasoning fields survive tool turns.
            assistant_msg: dict[str, Any] = (
                dict(resp.assistant_message)
                if resp.assistant_message is not None
                else {
                    "role": "assistant",
                    "content": resp.content or "",
                }
            )
            if resp.tool_calls and "tool_calls" not in assistant_msg:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            if not resp.tool_calls:
                # Model stopped calling tools → end of run
                finish_reason = "end_turn"
                _emit("final", iteration=iteration, text=resp.content or "")
                break

            # Execute each tool call (sequential, like Claude Code)
            for tc in resp.tool_calls:
                _emit(
                    "tool_call", iteration=iteration,
                    call_id=tc.id, name=tc.name, arguments=tc.arguments,
                )
                tool_t0 = time.monotonic()
                tool_result = dispatch(tc.name, tc.arguments, deps)
                tool_dt = time.monotonic() - tool_t0
                if tool_dt > SOFT_TOOL_TIMEOUT_S:
                    log.warning(
                        "loop iter %d: tool %s took %.1fs (slow)",
                        iteration, tc.name, tool_dt,
                    )
                tool_call_log.append(
                    f"iter{iteration}.{tc.name}({_brief_args(tc.arguments)})"
                    f" → {tool_result[:60]}"
                )
                _emit(
                    "tool_result", iteration=iteration,
                    call_id=tc.id, name=tc.name,
                    result_preview=tool_result[:800],
                    result_full_len=len(tool_result),
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
    except Exception as e:
        # Bug 3 fix: any non-LLMError exception (context mgmt crash, OOM,
        # DB lock, tool dispatch unhandled) — record as 'error' status,
        # don't leave harness_runs stuck in 'running'.
        log.exception("loop crashed at iter %d: %s", iteration, e)
        crash_text = f"{type(e).__name__}: {e}"
        finish_reason = "loop_crash"

    # Pick up supporting-tool cost accumulated by tools.
    sub_agent_cost = float(deps.extra_cost_usd or 0.0)
    total_cost_usd += sub_agent_cost

    # Combine all collected reasoning text into one final_text
    final_text = "\n".join(p for p in final_text_parts if p).strip()

    # Bug 1 fix: status reflects finish_reason
    if error_text is not None or crash_text is not None:
        status = STATUS_ERROR
    elif finish_reason == "max_iterations":
        status = STATUS_TRUNCATED
    else:
        status = STATUS_OK

    # Promote crash text to error_text for telemetry
    if crash_text and not error_text:
        error_text = crash_text

    latency_ms = int((time.monotonic() - t0) * 1000)
    # Bug 3 fix: _end_run is always called via try/finally semantics —
    # a top-level failure in _end_run itself shouldn't recurse, so we
    # protect it independently.
    try:
        _end_run(
            deps, run_id=run_id, iterations=iteration,
            tool_calls=tool_call_log, final_text=final_text,
            cost_usd=total_cost_usd, status=status,
            error_text=error_text,
            sub_agent_cost_usd=sub_agent_cost,
        )
    except Exception:
        log.exception("_end_run failed for run %s — telemetry incomplete", run_id)

    return RunResult(
        run_id=run_id,
        iterations=iteration,
        final_text=final_text,
        tool_call_log=tool_call_log,
        cost_usd=total_cost_usd,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        error_text=error_text,
    )


# ── Persistence helpers ──────────────────────────────────────────────


def _start_run(deps: AgentRuntimeDeps, trigger: TriggerEvent) -> int:
    detail_json = _json.dumps(trigger.detail, ensure_ascii=False, default=str)
    with deps.store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO harness_runs(trigger_kind, trigger_detail) "
            "VALUES (?, ?) RETURNING id",
            (trigger.kind, detail_json[:2000]),
        )
        return int(cur.fetchone()[0])


def _end_run(
    deps: AgentRuntimeDeps, *, run_id: int, iterations: int,
    tool_calls: list[str], final_text: str, cost_usd: float,
    status: str, error_text: str | None,
    sub_agent_cost_usd: float = 0.0,
) -> None:
    """Commit terminal state to harness_runs.

    ``sub_agent_cost_usd`` is a historical storage key. Supporting-tool LLM
    cost is already included in ``cost_usd`` by the caller; the JSON field
    preserves the existing telemetry schema.
    """
    with deps.store.connect() as conn:
        # Pack tool_calls + sub_agent breakdown into one JSON blob
        payload = {
            "calls": [t[:200] for t in tool_calls],
            "sub_agent_cost_usd": round(sub_agent_cost_usd, 6),
        }
        conn.execute(
            "UPDATE harness_runs SET ended_at = julianday('now'), "
            "iterations = ?, tool_calls_json = ?, final_text = ?, "
            "cost_usd = ?, status = ?, error_text = ? WHERE id = ?",
            (
                iterations,
                _json.dumps(payload, ensure_ascii=False)[:8000],
                final_text[:4000],
                round(cost_usd, 6),
                status,
                error_text[:500] if error_text else None,
                run_id,
            ),
        )


def _brief_args(args: dict[str, Any]) -> str:
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        s = str(v)
        parts.append(f"{k}={s[:30]!r}" if len(s) > 30 else f"{k}={v!r}")
    return ", ".join(parts)[:80]


# Re-export common types so callers can `from offerguide.agent_runtime.loop import ...`
__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "STATUS_ERROR",
    "STATUS_OK",
    "STATUS_TRUNCATED",
    "RunResult",
    "TriggerEvent",
    "run",
]
# Smell 6 fix (W15.12 review): removed `_ = tools` / `_ = _dt` dead code —
# `tools.py` has no import-time side effects, and `_dt` was unused.
