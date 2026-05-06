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

Plus: every wake auto-injects ``MEMORY.md`` first 200 lines into the
system context, so the agent always sees its 'home page' without having
to call ``view`` explicitly.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm import LLMClient, LLMError
from .memory import MemoryStore

log = logging.getLogger(__name__)


# Anthropic's published triggers. Tuned for OpenAI-compat (DeepSeek ~64K
# context) — we use lower thresholds because total context budget is
# smaller than Claude's.
#
# Q4 (W15.13 review answer): 40K → 30K. Realistic worst case at 8 iter:
# system prompt 3K + 5 tool results × 6K = 33K alone. Old 40K trigger
# left only 24K for the next call's output + new content → too tight,
# would hit DeepSeek's 64K hard limit and crash with context_length_exceeded.
# 30K trigger leaves comfortable headroom.
COMPACTION_TRIGGER_TOKENS = 30_000
"""Above this estimated input_tokens, run compaction (model summarizes
older messages). Tuned for DeepSeek's ~64K window."""

CLEAR_TOOL_RESULTS_TRIGGER_TOKENS = 12_000
"""Above this, replace old tool output blocks with placeholders.
Cheaper than compaction (no LLM call) so tries this first.
Q4: 15K → 12K to keep clearing well below compaction threshold (30K)
so the cheap path runs FIRST and may obviate compaction."""

KEEP_RECENT_TOOL_RESULTS = 4
"""How many most-recent tool results to keep after clearing."""

KEEP_RECENT_MESSAGES_AFTER_COMPACT = 6
"""After compaction, keep N most-recent messages alongside the summary
(to preserve immediate working context)."""


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
        return (
            f"# 系统事实 (每次 wake 注入)\n"
            f"- 今天: {self.today.isoformat()} ({_weekday_zh(self.today)})\n"
            f"- 校招阶段: {self.calendar_phase or _infer_phase(self.today)}\n"
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
    last_prompt_tokens: int = 0
    """Updated after each LLM response. Best estimate of next call's
    input_tokens (which equals previous input_tokens + new content)."""

    def build_initial_system(
        self, *, system_facts: SystemFacts | None = None,
    ) -> str:
        """Assemble the system message: instructions + facts + MEMORY.md."""
        parts = [load_instructions()]
        facts = system_facts or SystemFacts()
        parts.append(facts.render())

        memory_dump = self.memory.auto_load_text(max_lines=200)
        if memory_dump.strip():
            parts.append(
                f"# 你脑子里的当前状态 (worldview/MEMORY.md 前 200 行)\n\n"
                f"{memory_dump}\n\n"
                "---\n"
                "想看 worldview 其它文件 → 调 memory(command='view', path='...').\n"
                "想更新 → memory(command='str_replace' / 'insert' / 'create')."
            )
        else:
            parts.append(
                "# 你的 worldview 是空的\n\n"
                "这是你第一次 wake (或者 .offerguide/worldview/ 被清过). "
                "应该先 ask_user 拿基础信息 (cv / 偏好 / 雷区), 写进 candidate.md."
            )
        return "\n\n".join(parts)

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
        if est < CLEAR_TOOL_RESULTS_TRIGGER_TOKENS:
            return messages, 0

        # Find indices of all tool result messages
        tool_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "tool"
        ]
        if len(tool_indices) <= KEEP_RECENT_TOOL_RESULTS:
            return messages, 0

        # Keep the last N; clear the rest
        keep_from = tool_indices[-KEEP_RECENT_TOOL_RESULTS]
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
            cleared, est, CLEAR_TOOL_RESULTS_TRIGGER_TOKENS,
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
        if est < COMPACTION_TRIGGER_TOKENS:
            return messages, False

        if len(messages) < 6:
            # Too short to compact safely
            return messages, False

        # System at index 0; compact middle; keep last K
        system_msg = messages[0]
        tail = messages[-KEEP_RECENT_MESSAGES_AFTER_COMPACT:]
        middle = messages[1:-KEEP_RECENT_MESSAGES_AFTER_COMPACT]
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
            len(middle), est, COMPACTION_TRIGGER_TOKENS,
            KEEP_RECENT_MESSAGES_AFTER_COMPACT,
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
