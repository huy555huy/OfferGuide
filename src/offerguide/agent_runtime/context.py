"""Context assembly + self-implemented compaction & tool-result clearing.

Anthropic API natively supports `context_management` (compaction at 150K,
``clear_tool_uses_20250919`` at 30K). We use DeepSeek (OpenAI-compatible)
which doesn't — so we implement the same **idea** at the harness layer.

3-layer strategy (verbatim from Anthropic engineering docs):

1. **Compaction** at ~150K input_tokens — model summarizes conversation,
   summary replaces old history. Preserves "architectural decisions and
   unresolved bugs"; discards "redundant tool outputs and messages".
2. **Tool-result clearing** at ~30K — old tool result blocks replaced
   with placeholders. Tool *call* records remain so the model knows
   it ran the tool; *output* is dropped (re-fetch if needed).
3. **Memory tool** for cross-session knowledge — implemented in memory.py.

Plus: every wake auto-injects ``MEMORY.md`` first 200 lines and
``agenda.md`` into the system context, so the agent always sees both its
home page and open-loop ledger without having to call ``view`` explicitly.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._markdown import MarkdownBlock, render_block, render_markdown_document
from ..llm import LLMClient, LLMError
from . import _schema
from .memory import MemoryStore

if TYPE_CHECKING:
    from ..memory import Store

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Thresholds for the compaction / tool-result-clearing loops."""

    compaction_trigger_tokens: int = 30_000
    clear_tool_results_trigger_tokens: int = 12_000
    keep_recent_tool_results: int = 4
    keep_recent_messages_after_compact: int = 6


DEFAULT_CONTEXT_POLICY = ContextPolicy()

COMPACTION_TRIGGER_TOKENS = DEFAULT_CONTEXT_POLICY.compaction_trigger_tokens
"""Above this estimated input_tokens, run compaction (model summarizes
older messages). Tuned for DeepSeek's ~64K window."""

CLEAR_TOOL_RESULTS_TRIGGER_TOKENS = (
    DEFAULT_CONTEXT_POLICY.clear_tool_results_trigger_tokens
)
"""Above this, replace old tool output blocks with placeholders.
Cheaper than compaction (no LLM call) so tries this first. Below the
compaction threshold so the cheap path runs FIRST and may obviate compaction."""

KEEP_RECENT_TOOL_RESULTS = DEFAULT_CONTEXT_POLICY.keep_recent_tool_results
"""How many most-recent tool results to keep after clearing."""

KEEP_RECENT_MESSAGES_AFTER_COMPACT = (
    DEFAULT_CONTEXT_POLICY.keep_recent_messages_after_compact
)
"""After compaction, keep N most-recent messages alongside the summary
(to preserve immediate working context)."""


EVIDENCE_FIRST_POLICY = render_markdown_document(
    MarkdownBlock(
        heading="证据优先",
        lines=(
            "所有全局判断先看证据，再做推断。",
            "不能把推理伪装成事实。",
            "把事实、推断、未知分开；证据不够就继续查、问用户，或保持 unknown。",
            "不要把时间、沉默、低活跃度直接解释成用户状态或意图。",
        ),
    ),
    MarkdownBlock(
        heading="未知优先",
        lines=(
            "会影响全局策略的决定必须有证据链。",
            "假设只能作为假设，不能驱动批量投递、改目标、强推通知等动作。",
        ),
    ),
)


# Compaction prompt — mirror Anthropic's published default.
_COMPACTION_INSTRUCTIONS = """\
你是一个对话压缩器. 把下面的对话历史压缩成 1 段紧凑 summary.

**保留**:
- 用户的关键决定与目标
- 你（agent）做出的架构判断和当前 worldview 状态
- 未解决的问题、pending 任务、需要追踪的状态
- 用户给过的反馈（接受 / 拒绝 / 偏好信号）

**丢掉**:
- 工具的具体输出文本
- 中间步骤的 think-aloud
- 重复信息

输出格式: 一段 markdown, 用 ## 小标题分块. 控制在 800 字内."""


@dataclass
class SystemFacts:
    """Dynamic facts injected into the system message every wake.

    These are *facts* (today's date, calendar position) — not rules.
    The agent reasons about them; the harness doesn't decide based on them.
    """

    today: _dt.date = field(default_factory=_dt.date.today)
    calendar_phase: str = ""

    def render(self) -> str:
        return self.to_block().render()

    def to_block(self) -> MarkdownBlock:
        """Return the facts as a reusable markdown block."""
        return render_block(
            "系统事实 (每次 wake 注入)",
            lines=(
                f"- 今天: {self.today.isoformat()} ({_weekday_zh(self.today)})",
                f"- 校招阶段: {self.calendar_phase or _infer_phase(self.today)}",
            ),
        )


def load_instructions(harness_dir: Path | None = None) -> str:
    """Read instructions.md verbatim. This is the agent's 'soul' prompt."""
    base = harness_dir or Path(__file__).parent
    path = base / "instructions.md"
    if not path.exists():
        log.warning("instructions.md missing at %s", path)
        return ""
    return path.read_text(encoding="utf-8")


@dataclass
class ContextManager:
    """Stateful context wrangler for one agent run.

    Owns: token estimation, compaction trigger, tool-result clearing.
    Stateless about the actual conversation — that lives in `messages`
    on the loop. ContextManager just transforms the list in place when
    triggers hit.
    """

    llm: LLMClient
    memory: MemoryStore
    store: "Store | None" = None
    policy: ContextPolicy = field(default_factory=ContextPolicy)
    last_prompt_tokens: int = 0
    """Updated after each LLM response. Best estimate of next call's
    input_tokens (which equals previous input_tokens + new content)."""

    def build_initial_system(
        self, *, system_facts: SystemFacts | None = None,
    ) -> str:
        """Assemble the system message: instructions + facts + MEMORY.md."""
        blocks: list[MarkdownBlock | str] = [
            EVIDENCE_FIRST_POLICY,
            load_instructions(),
        ]
        facts = system_facts or SystemFacts()
        blocks.append(facts.to_block())

        goal_snapshot = self._build_goal_snapshot()
        if goal_snapshot:
            blocks.append(goal_snapshot)

        work_snapshot = self._build_work_item_snapshot()
        if work_snapshot:
            blocks.append(work_snapshot)

        blocks.append(self._build_agenda_block())

        memory_dump = self.memory.auto_load_text(max_lines=200)
        if memory_dump.strip():
            blocks.append(
                render_block(
                    "你脑子里的当前状态 (worldview/MEMORY.md 前 200 行)",
                    body=memory_dump,
                    lines=(
                        "---",
                        "想看 worldview 其它文件 → 调 memory(command='view', path='...').",
                        "想更新 → memory(command='str_replace' / 'insert' / 'create').",
                    ),
                )
            )
        else:
            blocks.append(
                render_block(
                    "你的 worldview 是空的",
                    body=(
                        "这是你第一次 wake (或者 .offerguide/worldview/ 被清过). "
                        "应该先 ask_user 拿基础信息 (cv / 偏好 / 雷区), 写进 candidate.md."
                    ),
                )
            )
        return render_markdown_document(*blocks)

    def _build_agenda_block(self) -> MarkdownBlock:
        """Render the agent's persistent open-loop ledger every wake.

        The runtime does not decide the next action. It only keeps the
        responsibility ledger in view so the model can reason from its own
        commitments instead of treating each wake as a one-shot request.
        """
        path = self.memory.root / "agenda.md"
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as e:
                return render_block(
                    "Agent Agenda (开放回路与责任账本)",
                    lines=(
                        "agenda.md 读取失败; 这不是没有开放回路。",
                        f"ERROR: {type(e).__name__}: {e}",
                    ),
                )
            kept = lines[:120]
            if len(lines) > 120:
                kept.append(f"... (truncated; {len(lines) - 120} more lines)")
            body = "\n".join(kept).strip()
            if body:
                return render_block(
                    "Agent Agenda (开放回路与责任账本)",
                    body=body,
                    lines=(
                        "---",
                        "每次 wake 先用它判断 act / ask / notify / sleep; "
                        "收束前更新关闭/新增的开放回路。",
                    ),
                )

        return render_block(
            "Agent Agenda (开放回路与责任账本)",
            body=(
                "agenda.md 不存在。先创建它, 记录开放回路、阻塞、机会和安静等待项; "
                "不要把这次 wake 当成一次性请求。"
            ),
        )

    def _build_work_item_snapshot(self) -> MarkdownBlock | None:
        """Render durable agent-owned work items into every wake."""
        if self.store is None:
            return None
        try:
            items = _schema.list_active_work_items(self.store, limit=8)
        except Exception as e:
            log.warning("context: failed to build work item snapshot: %s", e)
            return render_block(
                "Agent Work Items (真实开放工作)",
                lines=(
                    "工作项读取失败; 这不是没有工作。",
                    f"ERROR: {type(e).__name__}: {e}",
                ),
            )
        if not items:
            return render_block(
                "Agent Work Items (真实开放工作)",
                lines=(
                    "- 当前没有 open/in_progress/blocked/waiting 工作项。",
                    "- 如果这次 trigger 是用户输入或事件, runtime 会先创建对应工作项。",
                ),
            )

        lines: list[str] = []
        for item in items:
            due = f"; due_at={item.due_at}" if item.due_at is not None else ""
            job = f"; job_id={item.job_id}" if item.job_id is not None else ""
            lines.append(
                f"- WorkItem #{item.id} [{item.status}; p={item.priority}{job}{due}] "
                f"{item.title}"
            )
            if item.summary:
                lines.append(f"  summary: {item.summary[:220]}")
            if item.next_action:
                lines.append(f"  next_action: {item.next_action[:220]}")
            if item.source_ref:
                lines.append(f"  source: {item.source_kind}:{item.source_ref}")
        return render_block(
            "Agent Work Items (真实开放工作)",
            lines=tuple(lines),
        )

    def _build_goal_snapshot(self) -> MarkdownBlock | None:
        """Render active goals + factual progress for every wake.

        The runtime should not decide what the agent does with the goal; it
        only supplies fresh state so the model can reason against the user's
        north star instead of re-deriving priorities from MEMORY.md alone.
        """
        if self.store is None:
            return None

        try:
            from .. import goals as _goals

            active = _goals.list_active_goals(self.store)
            lines: list[str] = []
            if active:
                for goal in active[:5]:
                    progress = _goals.compute_progress(self.store, goal)
                    assessment = progress.assess()
                    target = goal.target_date.isoformat() if goal.target_date else "未设置"
                    lines.extend((
                        f"- Goal #{goal.id}: {goal.title}",
                        f"  target_date: {target}; target_metric: {goal.target_metric or '未设置'}",
                        (
                            "  funnel: "
                            f"active_apps={progress.apps_active}, "
                            f"interviews_scheduled={progress.interviews_scheduled}, "
                            f"offers={progress.offers}, rejects={progress.rejects}"
                        ),
                        (
                            "  heuristic: "
                            f"{assessment.state} / {assessment.summary} "
                            f"(confidence={assessment.confidence:.2f}; 这是启发式判断, 不是事实)"
                        ),
                    ))
                if len(active) > 5:
                    lines.append(f"- 还有 {len(active) - 5} 个 active goals 未展开; 需要时查数据库/页面。")
            else:
                lines.append("- 当前没有 active goals。若 worldview/用户输入也没有明确目标, 先问用户确认。")

            observations = _goals.list_active_self_observations(self.store, limit=5)
            if observations:
                lines.append("- Agent 自我观察:")
                for obs in observations:
                    lines.append(
                        f"  - [{obs.pattern_kind}] {obs.observation}"
                    )

            return render_block(
                "活跃目标与进度快照 (事实 + 明示启发式)",
                lines=tuple(lines),
            )
        except Exception as e:
            log.warning("context: failed to build goal snapshot: %s", e)
            return render_block(
                "活跃目标与进度快照",
                lines=(
                    "目标快照读取失败; 这不是没有目标。",
                    f"ERROR: {type(e).__name__}: {e}",
                ),
            )

    def estimate_input_tokens(
        self, messages: list[dict[str, Any]],
    ) -> int:
        """Rough estimate of input_tokens for next LLM call.

        Uses last call's `prompt_tokens` as anchor (best signal we get
        from OpenAI-compat API), plus a char/4 estimate for new content
        appended since.
        """
        # Conservative fallback when we have no anchor: char/4 for all
        if self.last_prompt_tokens == 0:
            chars = sum(_char_count_of_message(m) for m in messages)
            return chars // 4
        # Otherwise: anchor + delta of "new" messages we can detect
        # (not great, but the trigger is conservative anyway)
        chars = sum(_char_count_of_message(m) for m in messages)
        return max(self.last_prompt_tokens, chars // 4)

    def maybe_clear_tool_results(
        self, messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Replace old tool result content with placeholders if over budget.

        Returns (new messages, count cleared).

        Tool *call* announcements (assistant messages with tool_calls)
        are kept so the model still sees what it ran — only the *result*
        content of role='tool' messages get replaced.
        """
        est = self.estimate_input_tokens(messages)
        if est < self.policy.clear_tool_results_trigger_tokens:
            return messages, 0

        # Find indices of all tool result messages
        tool_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "tool"
        ]
        if len(tool_indices) <= self.policy.keep_recent_tool_results:
            return messages, 0

        # Keep the last N; clear the rest
        keep_from = tool_indices[-self.policy.keep_recent_tool_results]
        cleared = 0
        new_messages: list[dict[str, Any]] = []
        for i, m in enumerate(messages):
            if m.get("role") == "tool" and i < keep_from:
                # Replace content with placeholder, keep tool_call_id
                placeholder = (
                    "(tool result cleared by harness — re-call the tool "
                    "if you need to see this output again)"
                )
                new_messages.append({
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id"),
                    "content": placeholder,
                })
                cleared += 1
            else:
                new_messages.append(m)
        log.info(
            "context: cleared %d old tool results (est_tokens %d > %d trigger)",
            cleared, est, self.policy.clear_tool_results_trigger_tokens,
        )
        return new_messages, cleared

    def maybe_compact(
        self, messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Run compaction if over the COMPACTION trigger.

        Returns (new messages, compacted?).
        Strategy: ask LLM to summarize messages [1:-K]; replace them with
        a single 'summary' assistant message. Keeps system message at
        index 0, plus K most-recent messages at the tail.
        """
        est = self.estimate_input_tokens(messages)
        if est < self.policy.compaction_trigger_tokens:
            return messages, False

        if len(messages) < 6:
            # Too short to compact safely
            return messages, False

        # System at index 0; compact middle; keep last K
        system_msg = messages[0]
        tail = messages[-self.policy.keep_recent_messages_after_compact:]
        middle = messages[1:-self.policy.keep_recent_messages_after_compact]
        if not middle:
            return messages, False

        # Build a summarization prompt. We render middle as plain text.
        rendered = _render_messages_for_summary(middle)
        # Smell 2 fix (W15.12 review): pass extra={"timeout": ...} where
        # supported. LLMClient.chat doesn't expose a per-call timeout
        # parameter directly — protection here is defense in depth via
        # the underlying httpx client's default 180s. If compaction
        # crashes for any reason (timeout, network, rate limit), we
        # fall back to NOT compacting and let the next iteration's
        # tool-result clearing handle headroom.
        try:
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": _COMPACTION_INSTRUCTIONS},
                    {"role": "user", "content": rendered},
                ],
                temperature=0.2,
            )
        except LLMError as e:
            log.warning("compaction LLM call failed: %s", e)
            return messages, False
        except Exception as e:
            # Bug 3 cousin: compaction hang / unexpected crash should not
            # take down the whole agent loop. Log and skip compaction.
            log.exception("compaction crashed (non-LLMError): %s", e)
            return messages, False

        summary_text = resp.content.strip()
        summary_msg = {
            "role": "user",
            "content": (
                "# (前情提要 — harness 压缩了更早的对话)\n\n"
                f"{summary_text}\n\n"
                "---\n上面是历史摘要; 下面是最近的对话."
            ),
        }
        new_messages = [system_msg, summary_msg, *tail]
        log.info(
            "context: compacted %d middle messages (est_tokens %d > %d trigger) "
            "→ kept system + summary + last %d",
            len(middle), est, self.policy.compaction_trigger_tokens,
            self.policy.keep_recent_messages_after_compact,
        )
        # Reset anchor since the conversation is now smaller
        self.last_prompt_tokens = 0
        return new_messages, True


# ── helpers ─────────────────────────────────────────────────────────


_WEEKDAYS_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _weekday_zh(d: _dt.date) -> str:
    return _WEEKDAYS_ZH[d.weekday()]


def _infer_phase(today: _dt.date) -> str:
    """Heuristic: where in the 国内校招 calendar are we?

    This is a *fact* injection, not a rule. Agent reads it, reasons.
    """
    m = today.month
    if m in (4, 5):
        return "暑期实习投递高峰末期 / 部分公司面试季已开始"
    if m == 6:
        return "暑期实习面试季 + 6 月底前出 offer 高峰"
    if m == 7:
        return "暑期 offer 季 + 实习入职窗口"
    if m == 8:
        return "暑期实习进行中 + 秋招提前批开始"
    if m in (9, 10, 11):
        return f"秋招正式批 ({m} 月)"
    if m == 12:
        return "秋招收尾 / 春招准备"
    if m in (1, 2, 3):
        return "春招 + 暑期实习提前批 (1-3 月)"
    return ""


def _char_count_of_message(msg: dict[str, Any]) -> int:
    """Approximate token-budget contribution of one message (in chars).

    For OpenAI-style messages: role + content + tool_calls JSON + tool_call_id.
    """
    n = len(msg.get("role", ""))
    content = msg.get("content")
    if isinstance(content, str):
        n += len(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                n += len(str(block))
    if "tool_calls" in msg:
        n += len(str(msg["tool_calls"]))
    if "tool_call_id" in msg:
        n += len(str(msg["tool_call_id"]))
    return n


def _render_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    """Render a slice of messages as plain text for the summarizer LLM."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue  # never include system in summary input
        content = m.get("content")
        if isinstance(content, str):
            body = content
        elif isinstance(content, list):
            body = " ".join(str(b) for b in content)
        else:
            body = str(content)
        if "tool_calls" in m:
            tcs = m["tool_calls"]
            body += "\n[tool_calls: " + ", ".join(
                tc.get("function", {}).get("name", "?") for tc in tcs
            ) + "]"
        # Truncate per-message to keep summary input tractable
        if len(body) > 1500:
            body = body[:1500] + " ...[truncated]"
        lines.append(f"## [{role}]\n{body}")
    return "\n\n".join(lines)
