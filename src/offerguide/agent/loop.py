"""W13 — central agent loop. **Model in the driver's seat.**

This is the new execution paradigm for OfferGuide: instead of cron jobs
calling individual SKILLs in hardcoded order (the W4 LangGraph pattern in
``agent/graph.py``), one model sits in the middle, reads the system state,
decides which SKILL to call (via OpenAI tool-calling), looks at the result,
self-critiques, and decides whether to keep going or stop.

Why this matters (per the W13 design conversation):
- The 11 SKILLs were 11 isolated islands — daemon called them on cron, no
  cross-SKILL learning. The model only saw individual prompts.
- Now the model sees the whole state (jobs / applications / user_facts /
  recent runs) and gets to **decide** what's worth doing — not Python.
- This is the difference between "model is a JSON formatter" and "model
  has agency." Per Anthropic's long-running-agent guide, this single-agent
  loop is the recommended starting topology (multi-agent is unproven).

Architecture:
    trigger (cron / user button / event)
        │
        ▼
    AgentLoop.run(goal)
        │
        ├─── 1. snapshot_state()  → read jobs/apps/facts/recent_runs
        │                         → persist as event 'state_snapshot'
        ├─── 2. build_messages(goal, snapshot, tool_schemas)
        │
        ├─── 3. iteration loop (≤ max_iterations):
        │       │
        │       ├── llm.chat_with_tools(messages, tools)
        │       │
        │       ├── if response.tool_calls:
        │       │     for each tool_call:
        │       │       emit 'thinking' (the assistant's preamble text)
        │       │       emit 'tool_call' (name + args)
        │       │       run SkillRuntime.invoke(spec, args)
        │       │       emit 'tool_result' (preview)
        │       │     append assistant + tool_result messages, loop
        │       │
        │       └── else:
        │             emit 'final' — model decided no more tools needed
        │             break
        │
        ├─── 4. self_critique(goal, final_answer, trajectory)
        │       → critic LLM scores trajectory 0..1 + writes notes
        │
        └─── 5. persist agent_runs row + return AgentRunResult

Streaming UX:
    The ``on_event`` callback fires for every event the loop produces.
    The /agent/run UI route uses Server-Sent Events to forward each event
    to the browser in real time, so the user sees the model think and act.

Tool schemas:
    Each SkillSpec → one OpenAI function tool. Inputs are typed as strings
    (we don't have richer type info in SKILL.md frontmatter today). The
    description is the SKILL's `description` field. The body of the SKILL
    runs only when the model calls the tool; the loop never injects SKILL
    bodies into its own system prompt.

Non-goals (deliberately):
- No streaming of token-level model output yet (event-level streaming is
  enough for "agent is thinking" UX). Add later if needed.
- No multi-agent / sub-agent personas. Anthropic's own guide says it's
  unproven; we're starting single-agent.
- No automatic GEPA from agent_runs.critic_score yet — separate W14 task.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..llm import LLMClient, LLMError, ToolCall
from ..llm.client import _parse_tool_arguments  # robust JSON parser for ccvibe quirks
from ..memory import Store
from ..skills import SkillRuntime, SkillSpec

log = logging.getLogger(__name__)

EventCallback = Callable[[Mapping[str, Any]], None]

# Cap the number of LLM <-> tool round-trips. The model can hit ``stop`` on
# its own and usually does in 1-3 iterations; this is the safety net.
DEFAULT_MAX_ITERATIONS = 8

# Cap how much of each tool result we feed back to the model. Tool outputs
# can be 10kb+ markdown blobs (tailor_resume returns the whole rewrite);
# truncating prevents context bloat for the next decision.
TOOL_RESULT_CONTEXT_CAP = 6000

# Truncation cap when we surface tool result PREVIEWs to the UI (events are
# stored verbatim in trajectory_json so nothing is lost).
TOOL_RESULT_UI_PREVIEW_CAP = 800

# W13.7 — circuit breaker: if the model repeats the SAME tool with the SAME
# arguments this many times in a row, abort the loop. Catches the runaway
# pattern that ate 80s of dogfood time pre-W13.0 (read_job(1) → result →
# read_job(1) → result → ...). Anthropic explicitly calls this out as the
# #1 production agent failure mode.
CIRCUIT_BREAKER_REPEAT_THRESHOLD = 3


SYSTEM_PROMPT = """你是用户的求职 copilot。
关于用户的具体信息（姓名、学校、专业、目标方向、过往项目细节、偏好）都在 user_facts 里——
那些事实在每次唤醒时会自动注入到你的上下文上方，你直接看就能知道。
**这个 prompt 里不写任何用户身份信息**，因为如果哪天换了用户、user_facts 内容变了，
你应该照样能 work。

# 你不是巡检员, 是 copilot

一个真正懂用户的 copilot 大部分时候**不动**：
- 用户没新需求, 系统没紧急事 → 给一句话现状, 收工不打扰
- 用户有焦虑信号 (拒投了一堆 / 面试挂了) → 别推新岗位, 等等他
- 用户即将面试 → 哪怕系统里 thin JDs 一堆, 优先准备面试不是去 enrich

**忙不停 ≠ 好 agent**。多数唤醒应该是 0-1 个 tool call + 一段简短判断。
只有当你看到真值得做的事 (比如某个 silent 14 天的字节申请该催了 / 用户标 ⭐ 的岗位还没 tailor)
才动手。

# 三个该问自己的问题

每次唤醒, 先想 (不用说出来):
1. **此刻用户最该 focus 啥？** 看时间 + 用户活跃度 + user_facts 里最近变化
2. **当下行动是不是朝 north star 走？** snapshot 顶部 🎯 是用户的 active goals,
   每件事先问"这能让用户离 target_date 近一步吗" — 不能就别做
3. **我能给用户留下什么有价值的痕迹？** 一份 tailored 简历 / 一个准确的判断, 而不是"调了 3 个工具"

# 北极星 (north star) 优先

snapshot 的 `## 🎯 North Star` 段是用户的目标 + 当前 funnel 进度 (投了几个 / 面试几个 / offer 几个)。
**这是你的真上下文** —— 系统观察 / maintenance hints 都是次级。

行为模式:
- progress 显示 "已过期 N 天" → 用户可能需要调整目标范围或加速, 你**主动**提醒
- progress.is_on_track=False → 优先做能提高 funnel 的事 (找新 jobs / tailor 简历 / 写投递包)
- 没 active goal → 提醒用户去 /goals 设一个; 没 goal 的 agent 是个无目标的执行器

# 元认知 (meta-cognition)

snapshot 的 `## 📓 Agent 自我观察` 段是你**之前给自己写的笔记** —— "我反复犯过这个错"。
这些是你应该敬畏的——别再犯。

如果你这次又看到自己重复一种 pattern (例如 "snapshot 显示我连续 5 次唤醒都建议 X 但 user 都拒了"),
调 **meta_reflect** 工具把这个观察写进去。这是 agent 学习 ABOUT 自己, 不只是关于用户。

# 你能用的工具

**Lookup（免费, 先用这些拿数据）**
- `read_job(job_id)`: 返回完整 raw_text
- `read_user_resume()`: 返回 master 简历全文

**Action（元认知 + 跟用户沟通）**
- `detect_evolution_candidates()`, `evolve_skill(name, n)`, `run_gray_release()` — SKILL 进化
- `meta_reflect(observation, pattern_kind, ...)`: **agent 学习关于 agent 自己的事**。
  当你看到自己反复某种 pattern (好的或坏的), 写进去。下次唤醒会读到。
- `write_suggestion(title, body, ...)`: 跟用户的**主要沟通通道**。用户 approve/reject 是
  evolution 里**最高权重的反馈信号**。

**Maintenance（系统巡检, 由你判断要不要做）**
- `discover_new_jobs`, `enrich_thin_jds`, `classify_corpus`,
  `check_silent_applications`, `refresh_company_corpus`, `extract_facts_from_runs`,
  `regenerate_company_brief`

**SKILL（实质工作）**
11 个 (score_match / analyze_gaps / tailor_resume / mock_interview / prepare_interview / ...)
inputs 通常是完整文本 — 调之前先 lookup。

# 一些必须避免的坏行为

1. **看到数字 > 0 就反射式调对应工具** ← 这是僵硬的 cron 模式
   snapshot 上 "thin JDs: 4" 不代表你必须立即 enrich。先想用户此刻在意啥。
2. **重复调刚刚返回的 lookup**: read_job(1) 已经返回过 → 数据在历史里, 直接用别再调一次
3. **同一工具失败 2 次还是同样错** → 停下来 final 报错, 别无脑重试
4. **不要为了"看起来在做事"而强行调 SKILL** → 没事可做就直接 final

# 输出

每次回复要么是 tool_call 要么是 final, 别混合。
- tool_call 时, content 写一句**判断句**: "我看到 X 让我想到 Y, 所以做 Z"
- final 时, content 写人话总结: 看到了什么 / 做了 (或没做) 什么 / 给用户的建议
"""


CRITIC_PROMPT = """你是用户求职 copilot 的 critic。

# 你不是打分的, 是看一次 agent 行动对用户实际有没有帮助

旧版本的 critic 用 5 个维度 (goal_aligned / tool_choice / iteration_efficiency / transparency / honesty)
打分。问题: 一个 agent 可以"维度全满分"但**对用户毫无价值** ——
比如完美执行了"巡检 maintenance" 但当时用户根本不需要巡检。

你看这一次 agent run 的 trajectory + final answer, 问自己:

1. **这次行动给用户留下了什么真实价值？**
   - 一份 tailored 简历? 一个准确的 score? 一个让用户能 act 的判断?
   - 还是只是"我跑了 3 个工具" 这种 process 上的忙碌?

2. **agent 有没有正确判断"此刻该不该动"？**
   - 用户没新需求 + 系统不紧急 → 该 lay low
   - 反过来, 紧急的事 (silent 14 天 / 面试逼近) 没处理 → 失职
   - 把这两种判断错了, 就是僵硬

3. **agent 有没有为了"看起来在做事"硬调工具？**
   - 看到数字 > 0 就反射式 call 对应工具 = cron 模式
   - 应该是先想"这个数字代表用户的什么需求", 再决定动不动

4. **final answer 是不是诚实 + 用户能 act？**
   - 别夸大 "完成 3 项任务" 当其中 1 项失败了
   - 给用户的建议要可执行, 不是空话

# 输出 (严格 JSON, 不要 markdown 代码块)

{
  "value_delivered": <0..1>, // 给用户的真实价值 0=没有 1=显著
  "judgment_quality": <0..1>, // 该不该动的判断准不准
  "honesty": <0..1>, // final 是否如实反映, 不夸大不编造
  "overall": <0..1>, // 综合 (跟上面 3 个不必算术平均, 你自己加权)
  "notes": <一句话点评, 必须包含"对用户的实际帮助"判断, 100 字内>,
  "improvement_hint": <一句话: 下次类似情境怎么做更好, 100 字内>
}
"""


# ───────────────────────── lookup tools (system) ─────────────────────────
#
# These are agent-loop-internal tools (NOT SKILLs). They're free (no LLM
# call) and let the model fetch full data from the DB on demand. Without
# these, the model only sees the snapshot's job summaries and can't pass
# `job_text` / `user_profile` strings to the SKILLs that need them.
# Discovered necessary in the W13 first dogfood — the agent caught the
# bug perfectly via critic_score=0.0, "完全没从错误中学习".

LOOKUP_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_job",
            "description": (
                "读取某个 job 的完整信息（raw_text 全文 + company / title / location / source）。"
                "在调任何需要 job_text 字符串参数的 SKILL 之前, 先用这个工具拿到 job_text。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "integer",
                        "description": "jobs 表里的 id (snapshot 里以 'job#N' 形式列出)",
                    },
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_user_resume",
            "description": (
                "读取用户的 master 简历全文（markdown）。在调 score_match / analyze_gaps / "
                "tailor_resume 等需要 user_profile / user_resume / master_resume "
                "字符串参数的 SKILL 之前, 用这个工具拿全文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]

# ─── Action tools (W13.1) — write-side helpers, separate from lookups ───
# These DO real work (call LLMs, write DB rows) but aren't SKILLs (no
# SKILL.md, not in skill_runs, not in critic loop). Use them for
# meta-cognitive operations the agent should be able to drive.

ACTION_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "evolve_skill",
            "description": (
                "为表现欠佳的 SKILL 生成 N 个候选 prompt 变种 (W13.1 evolution)。"
                "调用前: 先用 detect_evolution_candidates 看哪些 SKILL 该进化 "
                "(or just check the snapshot's '该进化的 SKILL' 段)。"
                "执行: 把当前 SKILL prompt + 最近表现欠佳的样本喂给一个 meta-LLM, "
                "让它生成改进版本, 持久化到 skill_variants 表 (status=shadow)。"
                "灰度发布逻辑会另外把 shadow 变种 promote 到 canary, 这里只生成不放量。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "要进化的 SKILL 名 (e.g. 'score_match')",
                    },
                    "num_variants": {
                        "type": "integer",
                        "description": "生成多少个候选 (推荐 3, 不超过 5)",
                    },
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_evolution_candidates",
            "description": (
                "查看当前哪些 SKILL 满足进化条件 (信号数 >= 10 + fitness < 0.55 + 距上次进化 > 7 天)。"
                "用于 agent 决定要不要调 evolve_skill 时先看一眼有哪些 candidate。返回简洁列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_gray_release",
            "description": (
                "推进 SKILL 进化的灰度发布状态机一步: shadow → canary → live (或 failed)。"
                "对每个 SKILL 最多做一个动作: 把 shadow promote 到 canary 20%, "
                "或者 canary 收够 8 个 signal 后跟 live 比 fitness 决定 promote/fail。"
                "agent 通常每天调一次 (一次完整巡检)。dry_run=True 只看 plan 不真改。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "True = 只 plan, 不真动 skill_variants 表。默认 False。",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": (
                "推送一条通知到用户的飞书 / Telegram (取决于 settings.notify_channel)。"
                "**这是给用户主动联系的高级通道**——比 inbox suggestion 更打扰, 留给真急事:"
                "面试日 24h 内 / silent app 14+ 天必须跟进 / 高匹配 JD 第一时间通知。"
                "\n\n何时**别**用: (a) 一般 maintenance 事项 (用 write_suggestion 进 inbox 就行); "
                "(b) 短时间内已经推过类似的 (会让用户嫌烦, 自己跟踪上下文); "
                "(c) 用户深夜 (除非真 24h-critical)。"
                "\n\n推 push 是**有成本的**——用户每收到一条都要分心看。一周 push 超过 3 次就太多了。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "推送标题 (≤ 50 字, 用户收到第一眼看到的)",
                    },
                    "body": {
                        "type": "string",
                        "description": "推送正文 (中文, 1-3 句话, 用户能 act 的内容)",
                    },
                    "level": {
                        "type": "string",
                        "enum": ["info", "warn", "high"],
                        "description": "info=一般通知 / warn=该处理 / high=紧急(只用于真紧急事)",
                    },
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meta_reflect",
            "description": (
                "看你自己最近的 N 次 agent_runs + 用户 thumbs 反馈, 思考你自己的行为 pattern, "
                "把发现写进 agent_self_observations 表。"
                "\n\n这是**元认知**操作——agent 学习关于 agent 自己的事 (跟 user_facts 学习关于用户的不同)。"
                "\n\n何时该用: (a) snapshot 显示你最近 5+ 次 cron_wake 都建议同类事但用户都拒了 → "
                "你应该意识到 'overreach' pattern; (b) 你看到自己反复跑 maintenance 但每次都被 critic 评低 → "
                "可能是 'wrong_priority' pattern; (c) 用户长时间没用某条建议链 → 可能 'tone' / 风格不对。"
                "\n\n何时**别**用: 你只跑过 1-2 次 agent run, 数据不够 (N < 5)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "observation": {
                        "type": "string",
                        "description": "你对自己行为的一句话观察 (中文, ≤ 100 字)",
                    },
                    "pattern_kind": {
                        "type": "string",
                        "enum": ["overreach", "underreach", "tone", "wrong_priority", "repeated_mistake", "success_pattern"],
                        "description": (
                            "overreach=你做太多用户嫌烦; underreach=你该做没做; "
                            "tone=语气/风格不对; wrong_priority=优先级错; "
                            "repeated_mistake=同样错误反复犯; success_pattern=做得好的事可继续"
                        ),
                    },
                    "valid_for_days": {
                        "type": "integer",
                        "description": "(可选) 多少天后这条观察过期 (例如 'tone' 类 30 天, 'success_pattern' 通常没期限)",
                    },
                },
                "required": ["observation", "pattern_kind"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_suggestion",
            "description": (
                "写一张推荐卡片到 inbox, 让用户决定批准/拒绝/忽略。这是你跟用户沟通的"
                "**核心通道**——用户不在线时, 你看到值得让用户知道的事就写一张。"
                "用户后续 approve / reject 会作为 user_thumbs (权重最高) 反馈到 evolution_signals。"
                "\n\n何时该用: (a) 你建议跑一个有副作用的 SKILL 但用户没要求 (例如 tailor_resume); "
                "(b) 你看到一个值得用户注意的状态变化 (silent 14 天 / 新高匹配 JD); "
                "(c) 你做完一件事想让用户审阅结果。"
                "\n\n何时**别**用: (a) 已经在最近几小时内推过类似建议; "
                "(b) 用户没问的鸡毛蒜皮事 (会让 inbox 变垃圾邮件)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "卡片标题, 用户看到的第一行 (≤ 60 字, 一句话讲清楚)",
                    },
                    "body": {
                        "type": "string",
                        "description": "推荐理由 + 如果用户批准你打算做啥 (中文 markdown, 几句话)",
                    },
                    "skill_to_call": {
                        "type": "string",
                        "description": (
                            "(可选) 如果建议批准就调这个 SKILL/工具, 例如 'tailor_resume'。"
                            "用户在 UI 上能看到这个意图。"
                        ),
                    },
                    "skill_args_json": {
                        "type": "string",
                        "description": "(可选) skill_to_call 的参数, JSON 字符串 e.g. '{\"job_id\": 42}'",
                    },
                    "source_skill_name": {
                        "type": "string",
                        "description": (
                            "(可选, 但强烈建议) 这条 suggestion 的判断主要基于"
                            "**哪个** SKILL 的输出 (例如 'score_match' / 'analyze_gaps')。"
                            "用户后续 approve / reject 的反馈会归到这个 SKILL 的"
                            "evolution_signals。不传则系统启发式归到本次 trajectory "
                            "里最后一次调过的 SKILL — 多 SKILL 跑过时启发式可能归错。"
                        ),
                    },
                },
                "required": ["title", "body"],
                "additionalProperties": False,
            },
        },
    },
    # ────── W14.20 working memory tools ──────
    {
        "type": "function",
        "function": {
            "name": "write_note_to_self",
            "description": (
                "给**未来的自己**写一条 todo / observation. 下次 wake 你会在 "
                "snapshot 顶部看到这条 note. 用来跨 wake 接力 — 例如:\n"
                "- '下次 wake 看这 4 个 JD score 出来了没, 高分的写 suggestion'\n"
                "- '用户对 Anthropic 兴趣还不确定, 等用户 approve/reject 那条 suggestion 后再调 deeper'\n"
                "- '今天 23:54, 用户在睡觉, 明天 8:00 后再 follow up'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "body": {
                        "type": "string",
                        "description": "1-2 句话的 todo / observation. 用第一人称.",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["todo", "observation", "context"],
                        "description": "todo=具体下次该做; observation=观察; context=session 接力上下文",
                    },
                    "valid_for_hours": {
                        "type": "integer",
                        "description": "(可选) N 小时后这条 note 自动 stale. 默认 48h.",
                    },
                },
                "required": ["body", "kind"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_self_note",
            "description": (
                "把一条 self_note 标记为 done (从 snapshot 移除). 当你完成上次留的 todo 时调."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note_id": {"type": "integer"},
                    "reason": {
                        "type": "string",
                        "description": "1 句话: 为啥可以 clear (做完了 / 没必要了 / ...)",
                    },
                },
                "required": ["note_id", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user_question",
            "description": (
                "**主动问用户一个问题** (不是 suggestion). 当你看 state 后判断"
                "需要用户决定一件事再继续 (而不是单方面推荐) 时调.\n\n"
                "典型场景:\n"
                "- 用户 north star 跟简历方向不一致 — 问用户想改 north star 还是改简历重点\n"
                "- 用户多次 reject 同类 suggestion — 问是不是方向变了\n"
                "- 多个备选 (3 家公司你都觉得不错), 让用户挑哪几家先投\n\n"
                "用户在 inbox 看到这条, 选一个 option, 答案会写到 user_facts 让你下次看到."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "你的问题 (≤ 80 字, 一句话直接问)",
                    },
                    "context": {
                        "type": "string",
                        "description": "为啥问这个 — 给用户一些背景 (markdown, 几句话)",
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["id", "label"],
                        },
                        "description": "2-4 个选项. id 是回答的 short code (e.g. 'change_ns'), label 是 user 看到的文字 (e.g. '改 north star')",
                    },
                },
                "required": ["question", "context", "options"],
                "additionalProperties": False,
            },
        },
    },
]

LOOKUP_TOOL_NAMES = {sc["function"]["name"] for sc in LOOKUP_TOOL_SCHEMAS}
ACTION_TOOL_NAMES = {sc["function"]["name"] for sc in ACTION_TOOL_SCHEMAS}

# W13.2 maintenance tools (the 7 daemon jobs as agent-callable actions).
# These DO real work (spider, classify, web search) — agent decides which to
# fire on each wake based on the snapshot. Replaces hardcoded cron schedule.
from .maintenance import MAINTENANCE_TOOL_NAMES, MAINTENANCE_TOOL_SCHEMAS  # noqa: E402

SYSTEM_TOOL_NAMES = LOOKUP_TOOL_NAMES | ACTION_TOOL_NAMES | MAINTENANCE_TOOL_NAMES


# ───────────────────────── tool schema generation ─────────────────────────


def build_tool_schemas(skills: Iterable[SkillSpec]) -> list[dict[str, Any]]:
    """Convert a set of SkillSpecs into OpenAI function-tool schemas.

    Each SKILL becomes one function. Inputs are all typed as strings —
    SKILL.md doesn't carry per-input types today, and OpenAI's tool spec
    accepts JSON-schema for parameters but we keep it simple: the model
    is good enough to figure out the right value from the input name +
    the tool description (which carries the SKILL's full description).

    Returns a list ready to pass as ``tools=`` to ``LLMClient.chat_with_tools``.
    """
    tools: list[dict[str, Any]] = []
    for spec in skills:
        # Even SKILLs with empty `inputs` get a valid (empty-properties) schema,
        # so the model can call them with no args.
        properties: dict[str, dict[str, str]] = {}
        for input_name in spec.inputs:
            properties[input_name] = {
                "type": "string",
                "description": _input_hint(spec, input_name),
            }
        # OpenAI caps function descriptions; trim to keep request lean.
        description = (spec.description or spec.name)[:1024]
        tools.append({
            "type": "function",
            "function": {
                "name": spec.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(spec.inputs),
                    "additionalProperties": False,
                },
            },
        })
    return tools


def _input_hint(spec: SkillSpec, input_name: str) -> str:
    """Best-effort one-liner help for an input parameter.

    SKILL.md doesn't carry per-input docstrings so we use heuristics on the
    common input names used across the SKILL pack.
    """
    common = {
        "company": "目标公司中文名 (字节/腾讯/小红书/...)",
        "job_text": "JD 全文 (>= 200 字)",
        "user_profile": "用户画像或简历 markdown 文本",
        "user_resume": "用户当前 master 简历 markdown 文本",
        "master_resume": "用户主简历 markdown 全文 (ground truth, 不能编造)",
        "role_focus": "目标岗位类型, e.g. 'AI Agent 后端' / '推荐算法'",
        "role_hint": "岗位简称, e.g. 'AI 算法'",
        "role": "岗位名称",
        "past_experiences": "已渲染好的过去面经片段",
        "successful_profile_json": "successful_profile SKILL 的输出 JSON (可空)",
        "interview_questions": "已问过的面试题 list",
        "user_answer": "用户的回答文本",
        "candidate_jobs_json": "候选 jobs 的 list (id+title+company)",
    }
    if input_name in common:
        return common[input_name]
    return f"参数 {input_name}, 见 SKILL '{spec.name}' 的 description"


# ───────────────────────── state snapshot reader ─────────────────────────


def snapshot_state(
    store: Store,
    *,
    max_jobs: int = 8,
    max_apps: int = 6,
    max_facts: int = 8,
    max_runs: int = 5,
) -> str:
    """Render the current OfferGuide DB state as a single prompt block.

    The agent reads this once per ``run()`` and uses it to decide what to
    do. Keep it terse — the model has a finite context window. We surface:

    - **Recent jobs** that have raw_text >= 200 chars (eligible for SKILLs)
      with their score (if scored) and whether an application exists
    - **Recent applications** with current status (last 14 days)
    - **Recent SKILL runs** (last N) with skill_name + latency + success
    - **Top user_facts** by used_count (most-relied-on memory items)

    Empty sections are omitted so the prompt scales down on a fresh DB.
    """
    parts: list[str] = ["# 当前系统状态 (Snapshot)"]

    # ---- W14.20 working memory: notes you wrote to your future self ----
    # 上次 wake 你给自己留的 todo / observation. 先看, 决定: 完成 + clear,
    # 还是接着做, 还是已经不 relevant 了 (clear with reason).
    try:
        with store.connect() as conn:
            notes = conn.execute(
                "SELECT id, body, note_kind, created_at, valid_until "
                "FROM agent_self_notes "
                "WHERE cleared_at IS NULL "
                "  AND (valid_until IS NULL OR valid_until > julianday('now')) "
                "ORDER BY created_at DESC LIMIT 8"
            ).fetchall()
        if notes:
            parts.append("\n## 📝 你给自己留的 notes (working memory — 跨 wake 接力)")
            for nid, body, kind, _ca, _vu in notes:
                parts.append(f"- [#{nid} {kind}] {body}")
            parts.append(
                "  ↑ 检查: 这些是上次你给自己的 todo. 完成的调 clear_self_note(note_id, reason). "
                "还要做的就接着做. 不再 relevant 的也 clear 掉别让 snapshot 越积越多."
            )
    except Exception as e:
        log.warning("snapshot_state: self_notes read failed: %s", e)

    # ---- W14.20 pending questions: things you asked user, awaiting answer ----
    # If you have unanswered questions sitting in inbox, DO NOT ask the same
    # thing again. Wait for answer. (When user answers, written to user_facts.)
    try:
        with store.connect() as conn:
            pending_qs = conn.execute(
                "SELECT id, title, created_at FROM inbox_items "
                "WHERE kind = 'question' AND status = 'pending' "
                "ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
        if pending_qs:
            parts.append("\n## ❓ 你已经问用户的问题 (等回答, 别重复问)")
            for qid, qtitle, _ca in pending_qs:
                parts.append(f"- inbox#{qid}: {qtitle}")
    except Exception as e:
        log.warning("snapshot_state: pending_questions read failed: %s", e)

    # ---- North star: active goals + progress (W13.6) ----
    # 一个真 agent 有 north star, 不是只看眼前数字。每次唤醒先想:
    # "我们朝什么目标走? 距离目标多近多远? 当下行动该怎么对齐?"
    try:
        from .. import goals as _goals
        active_goals = _goals.list_active_goals(store)
        if active_goals:
            parts.append("\n## 🎯 North Star (用户当前目标)")
            for g in active_goals[:3]:
                progress = _goals.compute_progress(store, g)
                parts.append(progress.render_for_prompt())
                if not progress.is_on_track:
                    parts.append(
                        "  ⚠ 启发式判断: 当前进度可能跟不上 target_date "
                        "(应主动想想: 是不是该多投 / 改简历 / 调整目标范围)"
                    )
    except Exception as e:
        log.warning("snapshot_state: goals read failed: %s", e)

    # ---- Agent's own learned behavior patterns (W13.6 meta-cognition) ----
    # 这是 agent 看自己之前的行为, 记下"我做错了什么 / 该改什么"——
    # 比 user_facts (关于用户) 更高一层: 关于 agent 自己。
    try:
        from .. import goals as _goals
        self_obs = _goals.list_active_self_observations(store, limit=6)
        if self_obs:
            parts.append("\n## 📓 Agent 自我观察 (你之前注意到的关于自己的事)")
            for obs in self_obs:
                parts.append(f"- [{obs.pattern_kind}] {obs.observation}")
            parts.append("  ↑ 别违反这些。如果你发现新的 pattern, 调 meta_reflect 写进来。")
    except Exception as e:
        log.warning("snapshot_state: self_obs read failed: %s", e)

    # ---- jobs (top N most recent with raw_text >= 200) ----
    try:
        with store.connect() as conn:
            jobs = conn.execute(
                "SELECT id, company, title, location, source, "
                "       length(raw_text) AS rtl, fetched_at "
                "FROM jobs WHERE length(raw_text) >= 200 "
                "ORDER BY fetched_at DESC LIMIT ?",
                (max_jobs,),
            ).fetchall()
    except Exception as e:
        log.warning("snapshot_state: jobs read failed: %s", e)
        jobs = []
    if jobs:
        parts.append(f"\n## 最近 {len(jobs)} 个待处理 jobs (raw_text>=200)")
        # W14.8: dropped the latent score_map (declared but never read in
        # the loop below — only app_map is consumed); the comment said
        # "latest_score_prob" but the SELECT below only ever pulled status.
        job_ids = [j[0] for j in jobs]
        app_map: dict[int, str | None] = {jid: None for jid in job_ids}
        try:
            with store.connect() as conn:
                # latest score_match run per job — match input_json by job_id-tagged path
                # (simple proxy: any skill_run that references this job_id verbatim)
                # We just check applications table for has_app status:
                placeholders = ",".join("?" * len(job_ids))
                if job_ids:
                    rows = conn.execute(
                        f"SELECT job_id, status FROM applications "
                        f"WHERE job_id IN ({placeholders}) "
                        f"ORDER BY last_status_change DESC",
                        job_ids,
                    ).fetchall()
                    for jid, status in rows:
                        if app_map.get(jid) is None:
                            app_map[jid] = status
        except Exception:
            pass
        for jid, company, title, loc, source, rtl, _ts in jobs:
            app_marker = f" [APP={app_map[jid]}]" if app_map.get(jid) else ""
            parts.append(
                f"- job#{jid} | {company or '?'} | {(title or '?')[:40]}"
                f" | {loc or '?'} | src={source} | jd={rtl}字{app_marker}"
            )

    # ---- recent applications (last 14 days, regardless of status) ----
    try:
        with store.connect() as conn:
            apps = conn.execute(
                "SELECT a.id, a.job_id, a.status, j.company, j.title, "
                "       a.last_status_change "
                "FROM applications a LEFT JOIN jobs j ON j.id = a.job_id "
                "WHERE a.last_status_change >= julianday('now') - 14 "
                "ORDER BY a.last_status_change DESC LIMIT ?",
                (max_apps,),
            ).fetchall()
    except Exception as e:
        log.warning("snapshot_state: applications read failed: %s", e)
        apps = []
    if apps:
        parts.append(f"\n## 近 14 天 applications ({len(apps)} 条)")
        for aid, jid, status, company, title, _ts in apps:
            parts.append(
                f"- app#{aid} | job#{jid} | {company or '?'}"
                f" | {(title or '?')[:30]} | status={status}"
            )

    # ---- recent SKILL runs (last N) ----
    try:
        with store.connect() as conn:
            runs = conn.execute(
                "SELECT id, skill_name, skill_version, latency_ms, "
                "       length(output_json) AS out_len, created_at "
                "FROM skill_runs ORDER BY created_at DESC LIMIT ?",
                (max_runs,),
            ).fetchall()
    except Exception as e:
        log.warning("snapshot_state: skill_runs read failed: %s", e)
        runs = []
    if runs:
        parts.append(f"\n## 最近 {len(runs)} 个 SKILL 运行")
        for rid, name, ver, lat, out_len, _ts in runs:
            parts.append(
                f"- run#{rid} | {name} v{ver}"
                f" | {lat or 0}ms | output {out_len}字"
            )

    # ---- top user_facts by used_count ----
    try:
        with store.connect() as conn:
            facts = conn.execute(
                "SELECT fact_text, kind, confidence, used_count "
                "FROM user_facts "
                "ORDER BY used_count DESC, confidence DESC LIMIT ?",
                (max_facts,),
            ).fetchall()
    except Exception as e:
        log.warning("snapshot_state: user_facts read failed: %s", e)
        facts = []
    if facts:
        parts.append(f"\n## 高复用 user_facts (top {len(facts)})")
        for fact, kind, conf, used in facts:
            fact_short = fact[:140] + ("…" if len(fact) > 140 else "")
            parts.append(f"- [{kind} conf={conf:.1f} used={used}] {fact_short}")

    # ---- 系统观察 (事实, 不是规则) ----
    # 早期版本这块叫"维护待办 hints", 每行带"→ 调 X 如果 > 0"——那是把决策树
    # 写在 prompt 里, 模型只是执行 if-else, 不算 agent。现在改成纯事实陈述,
    # 让 agent 自己判断这些事实意味着什么 / 此刻该不该处理。
    try:
        with store.connect() as conn:
            thin_jds = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE length(raw_text) < 200"
            ).fetchone()[0]
            unclassified = conn.execute(
                "SELECT COUNT(*) FROM interview_experiences "
                "WHERE quality_classified_at IS NULL"
            ).fetchone()[0]
            silent_apps_7d = conn.execute(
                "SELECT COUNT(DISTINCT a.id) FROM applications a "
                "WHERE a.status NOT IN ('offer', 'rejected', 'withdrawn') "
                "  AND a.last_status_change < julianday('now') - 7"
            ).fetchone()[0]
            silent_apps_14d = conn.execute(
                "SELECT COUNT(DISTINCT a.id) FROM applications a "
                "WHERE a.status NOT IN ('offer', 'rejected', 'withdrawn') "
                "  AND a.last_status_change < julianday('now') - 14"
            ).fetchone()[0]
            today_discoveries = conn.execute(
                "SELECT COUNT(*) FROM jobs "
                "WHERE fetched_at >= julianday('now', '-1 day')"
            ).fetchone()[0]
            recent_runs_24h = conn.execute(
                "SELECT COUNT(*) FROM skill_runs "
                "WHERE created_at >= julianday('now') - 1"
            ).fetchone()[0]
            wakes_last_6h = conn.execute(
                "SELECT COUNT(*) FROM agent_runs "
                "WHERE trigger_kind = 'cron_wake' "
                "  AND started_at >= julianday('now') - 0.25"
            ).fetchone()[0]
            user_actions_24h = conn.execute(
                "SELECT COUNT(*) FROM agent_runs "
                "WHERE trigger_kind IN ('user_button', 'manual') "
                "  AND started_at >= julianday('now') - 1"
            ).fetchone()[0]
            interviews_within_7d = conn.execute(
                "SELECT COUNT(*) FROM interviews "
                "WHERE scheduled_at IS NOT NULL "
                "  AND scheduled_at BETWEEN julianday('now') AND julianday('now') + 7"
            ).fetchone()[0]
    except Exception as e:
        log.warning("snapshot_state: observations failed: %s", e)
        thin_jds = unclassified = silent_apps_7d = silent_apps_14d = 0
        today_discoveries = recent_runs_24h = wakes_last_6h = 0
        user_actions_24h = interviews_within_7d = 0

    # 时间情境 (周几 / 早晚) — 让 agent 知道现在是什么时间, 用户大概在干啥
    from datetime import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        now = _dt.now(ZoneInfo("Asia/Shanghai"))
    except Exception:
        now = _dt.now()
    weekday_cn = "一二三四五六日"[now.weekday()]
    time_of_day = (
        "凌晨" if now.hour < 6 else
        "早上" if now.hour < 12 else
        "下午" if now.hour < 18 else "晚上"
    )

    parts.append("\n## 当下情境")
    parts.append(f"- 现在: 周{weekday_cn} {time_of_day} {now.strftime('%H:%M')} (Asia/Shanghai)")
    parts.append(f"- 用户最近 24h 主动发起 agent runs: {user_actions_24h} 次")
    parts.append(f"- 近 6h cron_wake 已跑: {wakes_last_6h} 次")
    if interviews_within_7d > 0:
        parts.append(f"- ⚠ 未来 7 天内有 {interviews_within_7d} 个面试已排期")

    parts.append("\n## 系统观察 (事实)")
    parts.append(f"- jobs 中 raw_text < 200 字: {thin_jds} 个")
    parts.append(f"- interview_experiences 未分类: {unclassified} 个")
    if silent_apps_14d > 0:
        parts.append(f"- 申请超 14 天未推进: {silent_apps_14d} 个 (非常旧, 多半已挂)")
    if silent_apps_7d > silent_apps_14d:
        parts.append(f"- 申请 7-14 天未推进: {silent_apps_7d - silent_apps_14d} 个")
    parts.append(f"- 过去 24h 新入库 jobs: {today_discoveries} 个")
    parts.append(f"- 过去 24h SKILL 调用: {recent_runs_24h} 次")

    if len(parts) == 1:
        parts.append("\n(数据库空 — 没有 jobs / applications / SKILL 运行 / facts)")

    return "\n".join(parts)


# ───────────────────────── data classes ─────────────────────────


@dataclass
class AgentEvent:
    """One step in the agent's trajectory.

    Persisted into agent_runs.trajectory_json as a JSON list. UI streams
    these via SSE so user sees the agent think + act in real time.
    """
    kind: str
    """One of: 'state_snapshot' | 'thinking' | 'tool_call' | 'tool_result'
       | 'critique' | 'final' | 'error'"""

    at: str
    """ISO-8601 UTC timestamp."""

    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRunResult:
    run_id: int | None
    goal: str
    trigger_kind: str
    final_answer: str
    events: list[AgentEvent]
    iterations: int
    cost_usd: float
    latency_ms: int
    critic_score: float | None = None
    critic_notes: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "trigger_kind": self.trigger_kind,
            "final_answer": self.final_answer,
            "iterations": self.iterations,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "critic_score": self.critic_score,
            "critic_notes": self.critic_notes,
            "error": self.error,
            "events": [asdict(e) for e in self.events],
        }


# ───────────────────────── the loop itself ─────────────────────────


class AgentLoop:
    """Central agent loop — model decides which SKILL to call, when to stop.

    Stateless across runs; all persistence is via the injected ``store``.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        runtime: SkillRuntime,
        store: Store,
        skills: Iterable[SkillSpec],
        master_resume_text: str | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        critic_enabled: bool = True,
        critic_model: str | None = None,
        notifier: Any = None,
    ) -> None:
        self._llm = llm
        self._runtime = runtime
        self._store = store
        # W14: optional notifier for send_notification action tool.
        # When None, send_notification returns "no notifier configured" — agent
        # sees this and adapts (probably writes a write_suggestion instead).
        self._notifier = notifier
        self._skills: dict[str, SkillSpec] = {s.name: s for s in skills}
        # Order: lookups first (cheap reads), then actions (write helpers like
        # evolve_skill), then maintenance (W13.2: spider/classifier/etc), then
        # SKILLs. Gentle bias: read before write, plan before invoke expensive
        # SKILLs.
        self._tool_schemas = (
            LOOKUP_TOOL_SCHEMAS
            + ACTION_TOOL_SCHEMAS
            + MAINTENANCE_TOOL_SCHEMAS
            + build_tool_schemas(self._skills.values())
        )
        self._master_resume_text = master_resume_text or ""
        self._max_iter = max(1, int(max_iterations))
        self._critic_enabled = bool(critic_enabled)
        self._critic_model = critic_model
        # Per-run state (set during run(), cleared after) — exposed to action
        # tools that need attribution context (write_suggestion needs run_id
        # + last invoked SKILL to route user_thumbs feedback correctly).
        self._current_run_id: int | None = None
        self._current_skill_invocations: dict[str, dict] = {}

    # -------- public API --------

    def run(
        self,
        *,
        goal: str,
        trigger_kind: str = "manual",
        on_event: EventCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AgentRunResult:
        """Execute one agent loop. Persists trajectory + result to agent_runs.

        ``goal`` is the natural-language objective the model is given.
        ``trigger_kind`` records who woke the agent (cron / user_button / ...).
        ``on_event`` fires for every event — used by SSE streaming UI.
        ``cancel_event`` is checked at every iteration boundary; when set,
        the run aborts gracefully (emits a ``_cancelled`` event, persists
        agent_runs.status='cancelled', and returns early). Used by the SSE
        endpoint to stop the agent when the client disconnects so we don't
        keep burning LLM credits on a connection nobody's reading. Note:
        cancellation is **cooperative** — the LLM call already in flight
        runs to completion (httpx timeout caps that at 180s). If you need
        a hard cap, layer `httpx.Client(timeout=...)` instead.
        """
        events: list[AgentEvent] = []
        t0 = time.monotonic()
        run_id = self._record_start(goal=goal, trigger_kind=trigger_kind)
        # Track which SKILL invocations happened during this trajectory, so
        # the critique step can write evolution_signals attributing the
        # critic_score to the SKILLs that ran. Keyed by tool_call.id.
        skill_invocations: dict[str, dict] = {}
        # W13.7 circuit breaker — track recent (tool_name, args_signature) so
        # we can detect "model is stuck repeating itself" pattern.
        recent_tool_signatures: list[str] = []
        # Expose to action tools (write_suggestion needs run_id + last skill
        # for attribution). Cleared at end of run() so subsequent invocations
        # of run() get a clean slate.
        self._current_run_id = run_id
        self._current_skill_invocations = skill_invocations
        # W13.7 — accumulate token + $ cost across all LLM calls in this run
        self._current_total_cost_usd: float = 0.0
        self._current_total_prompt_tokens: int = 0
        self._current_total_completion_tokens: int = 0

        def emit(kind: str, **payload: Any) -> AgentEvent:
            ev = AgentEvent(
                kind=kind,
                at=datetime.now(UTC).isoformat(),
                payload=payload,
            )
            events.append(ev)
            if on_event:
                try:
                    on_event({"kind": kind, "at": ev.at, **payload})
                except Exception as e:  # never let the UI break the loop
                    log.debug("on_event callback raised: %s", e)
            return ev

        # 1. State snapshot
        try:
            snapshot = snapshot_state(self._store)
        except Exception as e:
            emit("error", message=f"state snapshot failed: {e}")
            return self._finalize(
                run_id=run_id, goal=goal, trigger_kind=trigger_kind,
                events=events, t0=t0, final_answer="(状态读取失败)",
                error=str(e),
            )
        emit("state_snapshot", snapshot=snapshot)

        # 2. Build initial messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"# 当前任务 (goal)\n{goal}\n\n"
                f"{snapshot}\n\n"
                f"开始决策。"
            )},
        ]

        # 3. Iteration loop
        final_answer = ""
        final_iteration = 0
        for iteration in range(self._max_iter):
            final_iteration = iteration + 1
            # W14.9: cooperative cancellation. Checked at the iteration
            # head so the in-flight LLM call (if any) is allowed to finish
            # — interrupting an httpx call mid-flight has no clean recovery
            # and the answer would be wasted spend anyway. Worst-case extra
            # latency is one LLM call (~3-30s).
            if cancel_event is not None and cancel_event.is_set():
                emit("_cancelled", iteration=iteration,
                     reason="cancel_event set (likely client disconnect)")
                return self._finalize(
                    run_id=run_id, goal=goal, trigger_kind=trigger_kind,
                    events=events, t0=t0, iterations=final_iteration - 1,
                    final_answer="(cancelled)",
                    status="cancelled",
                )
            try:
                resp = self._llm.chat_with_tools(
                    messages=messages,
                    tools=self._tool_schemas,
                    temperature=0.4,
                )
                # Accumulate cost for this iteration's LLM call
                self._current_total_cost_usd += float(getattr(resp, "cost_usd", 0.0) or 0.0)
                self._current_total_prompt_tokens += int(getattr(resp, "prompt_tokens", 0) or 0)
                self._current_total_completion_tokens += int(getattr(resp, "completion_tokens", 0) or 0)
            except LLMError as e:
                emit("error", message=f"LLM call failed at iter {iteration}: {e}")
                return self._finalize(
                    run_id=run_id, goal=goal, trigger_kind=trigger_kind,
                    events=events, t0=t0, iterations=final_iteration,
                    final_answer=f"(LLM 调用失败: {e})", error=str(e),
                )

            # Always emit thinking even if empty — UI knows model spoke
            emit(
                "thinking",
                text=resp.content or "",
                iteration=iteration,
                finish_reason=resp.finish_reason,
                will_call_tools=bool(resp.tool_calls),
                tool_call_names=[tc.name for tc in resp.tool_calls],
            )

            if not resp.tool_calls:
                # Model decided: no more tools, this is the final answer
                final_answer = resp.content or "(model returned empty content)"
                emit("final", text=final_answer, iteration=iteration)
                break

            # Append the assistant message that announced the tool calls.
            # LLMClient preserves provider-specific fields such as DeepSeek
            # reasoning_content while still re-serializing parsed tool args
            # into valid JSON, which keeps both reasoning models and ccvibe's
            # malformed-arguments quirk happy.
            assistant_msg: dict[str, Any] = (
                dict(resp.assistant_message)
                if resp.assistant_message is not None
                else {"role": "assistant", "content": resp.content or ""}
            )
            if "tool_calls" not in assistant_msg:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(
                                tc.arguments, ensure_ascii=False,
                            ),
                        },
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            # ── W13.7 circuit breaker: detect repeat-loop ──
            # Compute a signature for each tool call this iteration. If
            # ALL of this iteration's calls have appeared identically in
            # the last CIRCUIT_BREAKER_REPEAT_THRESHOLD iterations, stop.
            this_iter_signatures = [
                f"{tc.name}({json.dumps(tc.arguments, ensure_ascii=False, sort_keys=True)})"
                for tc in resp.tool_calls
            ]
            recent_window = recent_tool_signatures[
                -CIRCUIT_BREAKER_REPEAT_THRESHOLD * len(this_iter_signatures):
            ] if this_iter_signatures else []
            # Count how many times the exact set has appeared
            same_pattern_count = 0
            window_size = len(this_iter_signatures)
            if window_size > 0:
                # Walk backward through recent_tool_signatures in chunks of window_size
                for chunk_start in range(
                    len(recent_window) - window_size,
                    -1,
                    -window_size,
                ):
                    chunk = recent_window[chunk_start:chunk_start + window_size]
                    if chunk == this_iter_signatures:
                        same_pattern_count += 1
                    else:
                        break
            if same_pattern_count >= CIRCUIT_BREAKER_REPEAT_THRESHOLD - 1:
                # Already happened N-1 times before this iter — this iter
                # would make N. Trip the breaker, abort, force final.
                emit(
                    "error",
                    message=(
                        f"circuit_breaker: same tool pattern repeated "
                        f"{same_pattern_count + 1} times in a row "
                        f"({this_iter_signatures}); aborting to prevent runaway"
                    ),
                )
                final_answer = (
                    f"(circuit_breaker fired — agent stuck repeating "
                    f"{this_iter_signatures} {same_pattern_count + 1} times. "
                    f"Trajectory shows the loop; check skill output to fix root cause.)"
                )
                emit("final", text=final_answer, iteration=iteration)
                break
            # Append this iter's signatures to history before executing
            recent_tool_signatures.extend(this_iter_signatures)

            # Execute each tool sequentially
            for tc in resp.tool_calls:
                emit(
                    "tool_call",
                    call_id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    iteration=iteration,
                )
                tool_text = self._execute_tool(
                    tc, skill_invocations=skill_invocations,
                )
                emit(
                    "tool_result",
                    call_id=tc.id,
                    name=tc.name,
                    result_preview=tool_text[:TOOL_RESULT_UI_PREVIEW_CAP],
                    result_full_len=len(tool_text),
                    iteration=iteration,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_text[:TOOL_RESULT_CONTEXT_CAP],
                })
            # Loop back — let the model decide what to do next
        else:
            # Hit max_iter without a final answer
            final_answer = (
                f"(达到最大迭代数 {self._max_iter} 还未收敛 — agent 可能"
                "在反复调工具或卡住, 看 trajectory 排查)"
            )
            emit("error", message=f"max_iter ({self._max_iter}) reached")

        # 4. Self-critique
        critic_score: float | None = None
        critic_notes: str | None = None
        if self._critic_enabled and final_answer:
            try:
                critic_score, critic_notes = self._self_critique(
                    goal=goal,
                    final_answer=final_answer,
                    events=events,
                )
                emit("critique", score=critic_score, notes=critic_notes)
            except Exception as e:
                log.info("critic pass failed (non-fatal): %s", e)
                emit("critique", score=None, notes=f"(critic failed: {e})")

        # 4b. Write evolution_signals — attribute the critic score to every
        # SKILL that ran during this trajectory. The whole-trajectory score
        # is a coarse attribution (a 5-tool run all gets the same score)
        # but it's the best signal we have for now. fitness.compute_fitness
        # will weight + aggregate across many runs to wash out noise.
        if critic_score is not None and skill_invocations:
            self._write_critic_signals(
                run_id=run_id,
                critic_score=critic_score,
                critic_notes=critic_notes,
                skill_invocations=skill_invocations,
            )

        # 5. Persist + return
        try:
            return self._finalize(
                run_id=run_id, goal=goal, trigger_kind=trigger_kind,
                events=events, t0=t0, iterations=final_iteration,
                final_answer=final_answer,
                critic_score=critic_score, critic_notes=critic_notes,
                total_cost_usd=self._current_total_cost_usd,
            )
        finally:
            # Clear per-run state so subsequent run() calls don't leak attribution
            self._current_run_id = None
            self._current_skill_invocations = {}
            self._current_total_cost_usd = 0.0
            self._current_total_prompt_tokens = 0
            self._current_total_completion_tokens = 0

    # -------- internals --------

    def _execute_tool(
        self, tc: ToolCall, *, skill_invocations: dict[str, dict] | None = None,
    ) -> str:
        """Run a tool (lookup or SKILL) by name. Returns the raw text result.

        Errors are returned as a string starting with ``ERROR:`` rather than
        raised — the model needs to see what happened so it can decide
        whether to retry, switch tools, or stop.

        ``skill_invocations`` (when provided): the loop tracks which
        skill_runs happened during the trajectory by call_id, so the
        post-critique step can write evolution_signals for each.
        """
        # Lookup tools — system-side, free, just DB reads
        if tc.name in LOOKUP_TOOL_NAMES:
            return self._execute_lookup_tool(tc)

        # Action tools — system-side write operations (evolve_skill etc.)
        if tc.name in ACTION_TOOL_NAMES:
            return self._execute_action_tool(tc)

        # Maintenance tools (W13.2) — daemon jobs the agent can drive
        if tc.name in MAINTENANCE_TOOL_NAMES:
            return self._execute_maintenance_tool(tc)

        # SKILL tools
        spec = self._skills.get(tc.name)
        if spec is None:
            available = (
                sorted(SYSTEM_TOOL_NAMES) + sorted(self._skills.keys())
            )
            return f"ERROR: tool '{tc.name}' is not registered. Available: {available}"
        try:
            result = self._runtime.invoke(
                spec, tc.arguments,
                strict_inputs=False,  # model may pass extra context fields
                inject_long_term_memory=True,
            )
            # Track which skill_runs were created during the trajectory
            # so the post-critique signal-writer can attribute scores correctly.
            # W14.7-fix: must use result.skill_version (the version that ACTUALLY
            # ran — could be a canary/live variant) not spec.version (always seed).
            # Otherwise canary/live feedback is mis-attributed to the seed and
            # fitness/promote/compare_versions all see corrupted data.
            if skill_invocations is not None:
                skill_invocations[tc.id] = {
                    "skill_name": spec.name,
                    "skill_version": result.skill_version,
                    "skill_run_id": result.skill_run_id,
                }
            return result.raw_text or "(SKILL returned empty content)"
        except ValueError as e:
            # Missing-required-input or similar — let the model see the fix
            return f"ERROR: invalid inputs for {tc.name}: {e}"
        except Exception as e:
            log.exception("tool execution crashed: %s(%s)", tc.name, tc.arguments)
            return f"ERROR: tool '{tc.name}' raised {type(e).__name__}: {e}"

    def _execute_maintenance_tool(self, tc: ToolCall) -> str:
        """Run a W13.2 maintenance tool (spider, classifier, silence-check, etc).

        Builds a MaintenanceCtx from the loop's resources and dispatches via
        ``maintenance.execute_maintenance_tool``. Failures return an ERROR
        string so the model sees what went wrong and can decide next step.
        """
        from .maintenance import MaintenanceCtx, execute_maintenance_tool
        ctx = MaintenanceCtx(
            store=self._store,
            llm=self._llm,
            runtime=self._runtime,
            skills=list(self._skills.values()),
            user_profile_text=self._master_resume_text or None,
        )
        return execute_maintenance_tool(tc.name, tc.arguments, ctx)

    def _execute_action_tool(self, tc: ToolCall) -> str:
        """Execute a write-side action tool (evolve_skill, detect_evolution_candidates).

        These are NOT SKILLs (no SKILL.md, no critic, no skill_runs row).
        They're meta-cognitive operations the agent can drive: 'check what
        needs evolving', 'evolve this SKILL'. Result text is returned to the
        agent so it can decide next action.
        """
        if tc.name == "detect_evolution_candidates":
            try:
                from ..evolution.fitness import detect_evolution_candidates as _detect
                triggers = _detect(self._store)
            except Exception as e:
                return f"ERROR: detect_evolution_candidates failed: {e}"
            if not triggers:
                return (
                    "暂无符合进化条件的 SKILL "
                    "(条件: signals >= 10 + fitness < 0.55 + 距上次进化 > 7 天)。"
                )
            lines = ["该进化的 SKILL (按 fitness 升序):"]
            for t in triggers[:10]:
                lines.append(
                    f"  - {t.skill_name} v{t.current_version}: "
                    f"fitness={t.fitness:.2f}, samples={t.sample_count}, {t.reason}"
                )
            return "\n".join(lines)

        if tc.name == "evolve_skill":
            skill_name = (tc.arguments.get("skill_name") or "").strip()
            if not skill_name:
                return "ERROR: evolve_skill requires skill_name (string)"
            try:
                num_variants = int(tc.arguments.get("num_variants") or 3)
            except (TypeError, ValueError):
                num_variants = 3
            num_variants = max(1, min(num_variants, 5))
            try:
                from ..evolution.evolve import evolve_skill as _evolve
                result = _evolve(
                    store=self._store, llm=self._llm,
                    skill_name=skill_name, num_variants=num_variants,
                )
            except Exception as e:
                log.exception("evolve_skill action crashed")
                return f"ERROR: evolve_skill raised {type(e).__name__}: {e}"
            return (
                f"# evolve_skill('{skill_name}') 完成\n"
                f"父版本: {result.parent_version}\n"
                f"生成候选: {result.candidates_generated}\n"
                f"持久化 shadow: {result.candidates_persisted}\n"
                f"shadow 版本号: {', '.join(result.variant_versions) or '(无)'}\n"
                f"备注: {result.notes}"
            )

        if tc.name == "run_gray_release":
            dry_run = bool(tc.arguments.get("dry_run", False))
            try:
                from ..evolution.release import run_release_cycle
                cycle = run_release_cycle(self._store, dry_run=dry_run)
            except Exception as e:
                log.exception("run_gray_release crashed")
                return f"ERROR: run_gray_release raised {type(e).__name__}: {e}"
            mode = "[DRY RUN] " if dry_run else ""
            return f"# run_gray_release {mode}\n\n{cycle.render_summary()}"

        if tc.name == "write_suggestion":
            return self._execute_write_suggestion(tc)

        if tc.name == "meta_reflect":
            return self._execute_meta_reflect(tc)

        if tc.name == "send_notification":
            return self._execute_send_notification(tc)

        # ── W14.20 working memory + question tools ──
        if tc.name == "write_note_to_self":
            return self._execute_write_note_to_self(tc)
        if tc.name == "clear_self_note":
            return self._execute_clear_self_note(tc)
        if tc.name == "ask_user_question":
            return self._execute_ask_user_question(tc)

        return f"ERROR: action tool '{tc.name}' is declared but not implemented"

    # ── W14.20 self-notes (working memory) executors ──

    def _execute_write_note_to_self(self, tc: ToolCall) -> str:
        body = (tc.arguments.get("body") or "").strip()
        kind = (tc.arguments.get("kind") or "todo").strip()
        if not body:
            return "ERROR: write_note_to_self requires non-empty body"
        if kind not in {"todo", "observation", "context"}:
            return f"ERROR: invalid kind '{kind}', must be todo/observation/context"
        valid_for_h = tc.arguments.get("valid_for_hours")
        try:
            valid_for_hours = (
                int(valid_for_h) if valid_for_h is not None else 48
            )
        except (TypeError, ValueError):
            valid_for_hours = 48
        try:
            with self._store.connect() as conn:
                cur = conn.execute(
                    "INSERT INTO agent_self_notes("
                    "  body, note_kind, valid_until, related_run_id"
                    ") VALUES (?, ?, julianday('now') + ?, ?)",
                    (body, kind, valid_for_hours / 24.0, self._current_run_id),
                )
                note_id = int(cur.lastrowid or 0)
            return f"OK: 写入 self_note#{note_id}, 下次 wake 你会在 snapshot 顶部看到."
        except Exception as e:
            log.exception("write_note_to_self crashed")
            return f"ERROR: write_note_to_self raised {type(e).__name__}: {e}"

    def _execute_clear_self_note(self, tc: ToolCall) -> str:
        try:
            note_id = int(tc.arguments.get("note_id"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "ERROR: clear_self_note requires note_id (integer)"
        reason = (tc.arguments.get("reason") or "").strip()
        if not reason:
            return "ERROR: clear_self_note requires reason"
        try:
            with self._store.connect() as conn:
                row = conn.execute(
                    "UPDATE agent_self_notes "
                    "SET cleared_at = julianday('now'), cleared_reason = ? "
                    "WHERE id = ? AND cleared_at IS NULL "
                    "RETURNING id",
                    (reason[:200], note_id),
                ).fetchone()
            if row is None:
                return f"WARN: self_note#{note_id} 已经 cleared 或不存在"
            return f"OK: cleared self_note#{note_id} (reason={reason[:80]})"
        except Exception as e:
            log.exception("clear_self_note crashed")
            return f"ERROR: clear_self_note raised {type(e).__name__}: {e}"

    def _execute_ask_user_question(self, tc: ToolCall) -> str:
        question = (tc.arguments.get("question") or "").strip()
        context = (tc.arguments.get("context") or "").strip()
        options = tc.arguments.get("options") or []
        if not question:
            return "ERROR: ask_user_question requires question"
        if not isinstance(options, list) or len(options) < 2:
            return "ERROR: need at least 2 options"
        if len(options) > 4:
            options = options[:4]
        # Validate option shape
        valid_options = []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            opt_id = (opt.get("id") or "").strip()
            opt_label = (opt.get("label") or "").strip()
            if opt_id and opt_label:
                valid_options.append({"id": opt_id[:40], "label": opt_label[:120]})
        if len(valid_options) < 2:
            return "ERROR: at least 2 well-formed options required (id+label)"

        try:
            from .. import inbox as _inbox
            item = _inbox.enqueue_question(
                self._store,
                question=question,
                context=context,
                options=valid_options,
                source_agent_run_id=self._current_run_id,
            )
            return (
                f"OK: 问题入 inbox#{item.id} (kind=question, "
                f"{len(valid_options)} 个选项). 用户答了后会写到 user_facts, "
                f"你下次 wake 在 snapshot 里能看到答案."
            )
        except Exception as e:
            log.exception("ask_user_question crashed")
            return f"ERROR: ask_user_question raised {type(e).__name__}: {e}"

    def _execute_send_notification(self, tc: ToolCall) -> str:
        """W14 — agent decides to push a notification to user (Feishu/Telegram).

        When notifier is None (no channel configured), returns informative
        ERROR so agent knows to use write_suggestion as fallback.
        """
        if self._notifier is None:
            return (
                "ERROR: send_notification 没配 notifier. 用户没设 Feishu / Telegram。"
                "改用 write_suggestion 写到 inbox 让用户下次访问看到。"
            )
        title = (tc.arguments.get("title") or "").strip()
        body = (tc.arguments.get("body") or "").strip()
        level = tc.arguments.get("level") or "info"
        if not title or not body:
            return "ERROR: send_notification requires title + body"
        if level not in ("info", "warn", "high"):
            level = "info"

        try:
            result = self._notifier.notify(title=title, body=body, level=level)
        except Exception as e:
            log.exception("send_notification dispatch crashed")
            return f"ERROR: notifier raised {type(e).__name__}: {e}"

        if not getattr(result, "ok", False):
            err = getattr(result, "error", "unknown")
            channel = getattr(result, "channel", "?")
            return f"ERROR: {channel} push failed: {err}"
        channel = getattr(result, "channel", "?")
        return f"OK: 已通过 {channel} 推送 [{level}] '{title}'"

    def _execute_meta_reflect(self, tc: ToolCall) -> str:
        """Agent records an observation about its OWN behavior pattern (W13.6).

        This is the meta-cognition primitive — the thing that makes
        ``agent_self_observations`` populated, which then flows back into
        future snapshots and changes future agent behavior.
        """
        observation = (tc.arguments.get("observation") or "").strip()
        pattern_kind = (tc.arguments.get("pattern_kind") or "").strip()
        if not observation or pattern_kind not in (
            "overreach", "underreach", "tone",
            "wrong_priority", "repeated_mistake", "success_pattern",
        ):
            return (
                "ERROR: meta_reflect requires observation (string) + "
                "pattern_kind (overreach/underreach/tone/wrong_priority/"
                "repeated_mistake/success_pattern)"
            )
        valid_for_days_raw = tc.arguments.get("valid_for_days")
        valid_for_days: int | None = None
        if valid_for_days_raw is not None:
            try:
                valid_for_days = int(valid_for_days_raw)
                if valid_for_days <= 0:
                    valid_for_days = None
            except (TypeError, ValueError):
                valid_for_days = None
        try:
            from .. import goals as _goals
            obs_id = _goals.write_self_observation(
                self._store,
                observation=observation,
                pattern_kind=pattern_kind,  # type: ignore[arg-type]
                evidence={
                    "from_agent_run": self._current_run_id,
                    "trajectory_skill_count": len(self._current_skill_invocations),
                },
                valid_for_days=valid_for_days,
            )
            valid_str = f", valid {valid_for_days}d" if valid_for_days else ", no expiry"
            return (
                f"OK: 写入 self_observation #{obs_id} [{pattern_kind}{valid_str}]\n"
                f"  '{observation}'\n"
                f"将来 cron_wake 看 snapshot 时会读到, 影响行为。"
            )
        except Exception as e:
            log.exception("meta_reflect crashed")
            return f"ERROR: meta_reflect raised {type(e).__name__}: {e}"

    def _execute_write_suggestion(self, tc: ToolCall) -> str:
        """Write an agent_suggestion inbox item (W13.3).

        The suggestion is attributed to the current agent_run (via
        ``self._current_run_id``) so when the user later approves/rejects,
        the user_thumbs signal flows back to the right agent_run for
        attribution.

        W14.9: SKILL-level attribution now prefers an explicit
        ``source_skill_name`` arg from the model. Previously we used
        "trajectory's most-recent SKILL" — which mis-attributes when the
        model called several SKILLs and the suggestion is logically about
        an earlier one (e.g. score_match → tailor_resume → suggestion that's
        really *about* score_match). The heuristic remains as fallback.
        """
        title = (tc.arguments.get("title") or "").strip()
        body = (tc.arguments.get("body") or "").strip()
        if not title or not body:
            return "ERROR: write_suggestion requires title + body"

        skill_to_call = (tc.arguments.get("skill_to_call") or "").strip() or None
        skill_args_raw = tc.arguments.get("skill_args_json") or ""
        proposed_action = None
        if skill_to_call:
            args = {}
            if skill_args_raw:
                try:
                    args = json.loads(skill_args_raw)
                    if not isinstance(args, dict):
                        args = {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
            proposed_action = {"tool": skill_to_call, "args": args}

        # W14.9: explicit > heuristic. The model can name the SKILL whose
        # output this suggestion is about; only fall back to "most recent in
        # trajectory" when not given. This uses the actual recorded
        # skill_invocations dict so version + run_id stay consistent with
        # what runtime actually invoked.
        sk_name = sk_ver = None
        sr_id = None
        explicit_name = (tc.arguments.get("source_skill_name") or "").strip() or None
        if explicit_name and self._current_skill_invocations:
            # Find the most recent invocation of THIS skill (so version +
            # skill_run_id are real, not made-up).
            for inv in reversed(list(self._current_skill_invocations.values())):
                if inv.get("skill_name") == explicit_name:
                    sk_name = inv.get("skill_name")
                    sk_ver = inv.get("skill_version")
                    sr_id = inv.get("skill_run_id")
                    break
        if sk_name is None and explicit_name:
            # Model named a SKILL we never invoked this trajectory.
            # Honor the name (the model knows what the suggestion's about)
            # but mark version "?" so fitness doesn't bucket it under a fake
            # version, and skip skill_run_id (no real run to point at).
            sk_name = explicit_name
            sk_ver = "?"
        if sk_name is None and self._current_skill_invocations:
            # Fall back to old heuristic only when model didn't tell us.
            most_recent = list(self._current_skill_invocations.values())[-1]
            sk_name = most_recent.get("skill_name")
            sk_ver = most_recent.get("skill_version")
            sr_id = most_recent.get("skill_run_id")

        try:
            from .. import inbox as _inbox
            item = _inbox.enqueue_agent_suggestion(
                self._store,
                title=title, body=body,
                source_agent_run_id=self._current_run_id,
                source_skill_name=sk_name,
                source_skill_version=sk_ver,
                source_skill_run_id=sr_id,
                proposed_action=proposed_action,
            )
            attribution = (
                f" (attributed to {sk_name} v{sk_ver}, skill_run#{sr_id})"
                if sk_name else " (no SKILL attribution)"
            )
            return (
                f"OK: 写入 inbox#{item.id} '{title}'{attribution}\n"
                f"用户在 /inbox 决定 approve/reject 后, 会作为 user_thumbs 信号流回 evolution。"
            )
        except Exception as e:
            log.exception("write_suggestion crashed")
            return f"ERROR: write_suggestion raised {type(e).__name__}: {e}"

    def _execute_lookup_tool(self, tc: ToolCall) -> str:
        """Execute one of the system-side lookup tools (read_job / read_user_resume)."""
        if tc.name == "read_job":
            try:
                # tc.arguments.get returns Any|None; int(None) → TypeError caught below.
                job_id = int(tc.arguments.get("job_id"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return (
                    "ERROR: read_job requires job_id as integer, "
                    f"got {tc.arguments!r}"
                )
            try:
                with self._store.connect() as conn:
                    row = conn.execute(
                        "SELECT company, title, location, source, raw_text "
                        "FROM jobs WHERE id = ?", (job_id,),
                    ).fetchone()
            except Exception as e:
                return f"ERROR: read_job DB error: {e}"
            if row is None:
                return f"ERROR: job#{job_id} not found"
            company, title, loc, source, raw_text = row
            return (
                f"# job#{job_id}\n"
                f"company: {company or '?'}\n"
                f"title: {title or '?'}\n"
                f"location: {loc or '?'}\n"
                f"source: {source}\n\n"
                f"## raw_text (字数 {len(raw_text or '')})\n{raw_text or '(空)'}"
            )

        if tc.name == "read_user_resume":
            if not self._master_resume_text:
                return (
                    "ERROR: master_resume_text 没传给 AgentLoop "
                    "(create_app 里 profile=None, 或者构造时 master_resume_text='')"
                )
            return f"# 用户 master 简历\n\n{self._master_resume_text}"

        # Schema-listed but not implemented — programming error
        return f"ERROR: lookup tool '{tc.name}' is declared but not implemented"

    def _self_critique(
        self, *, goal: str, final_answer: str, events: list[AgentEvent],
    ) -> tuple[float | None, str | None]:
        """Run a critic LLM that scores the trajectory.

        Returns (score, notes). Score is None if the critic call or parse
        fails — non-fatal, just no signal for this run.
        """
        # Render trajectory compactly for the critic
        traj_lines: list[str] = []
        for ev in events:
            if ev.kind == "tool_call":
                traj_lines.append(
                    f"[tool_call] {ev.payload.get('name')}({ev.payload.get('arguments')})"
                )
            elif ev.kind == "tool_result":
                preview = (ev.payload.get("result_preview") or "")[:200]
                traj_lines.append(
                    f"[tool_result#{ev.payload.get('name')}] {preview}"
                )
            elif ev.kind == "thinking" and ev.payload.get("text"):
                traj_lines.append(f"[thinking] {ev.payload['text'][:200]}")
            elif ev.kind == "error":
                traj_lines.append(f"[error] {ev.payload.get('message')}")
        trajectory_render = "\n".join(traj_lines) or "(empty trajectory)"

        user_msg = (
            f"## goal\n{goal}\n\n"
            f"## trajectory\n{trajectory_render}\n\n"
            f"## final_answer\n{final_answer[:2000]}\n\n"
            f"评估上面这次 agent run, 严格输出 JSON。"
        )
        resp = self._llm.chat(
            messages=[
                {"role": "system", "content": CRITIC_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=self._critic_model,
            temperature=0.0,
            json_mode=True,
        )
        # Use the robust parser — ccvibe sometimes emits concatenated JSON
        # objects in regular chat responses too (W13 dogfood #5). json.loads
        # would raise; _parse_tool_arguments handles it gracefully.
        data = _parse_tool_arguments(resp.content)
        if not data:
            # If the parser returned empty, try one more bare json.loads in
            # case it was a valid singleton object the parser missed
            try:
                fallback = json.loads(resp.content.strip())
                if isinstance(fallback, dict) and fallback:
                    data = fallback
            except (json.JSONDecodeError, AttributeError):
                pass
        if not data:
            return None, f"(critic returned non-parseable JSON: {resp.content[:400]})"
        if not isinstance(data, dict):
            return None, f"(critic returned non-object JSON: {type(data).__name__})"
        try:
            # data.get returns Any|None; float(None) → TypeError caught below.
            score = float(data.get("overall"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            score = None
        notes = str(data.get("notes") or "")[:500]
        return score, notes

    def _write_critic_signals(
        self,
        *,
        run_id: int | None,
        critic_score: float,
        critic_notes: str | None,
        skill_invocations: dict[str, dict],
    ) -> None:
        """Fan the critic_score out to every SKILL that ran during the trajectory.

        Each unique (skill_name, version, run_id) tuple gets one row in
        evolution_signals. Lookup tools (read_job/read_user_resume) don't
        appear in skill_invocations so they don't get signals — they're
        cheap and stateless, no point evolving their prompts.
        """
        from ..evolution import (
            signals as _signals,  # local import: keeps loop.py loadable when evolution/ is mid-refactor
        )

        notes = (critic_notes or "")[:300]
        seen_runs: set[int] = set()
        for inv in skill_invocations.values():
            srid = inv.get("skill_run_id")
            if srid in seen_runs:
                continue
            if srid is not None:
                seen_runs.add(srid)
            try:
                _signals.record_critic_signal(
                    self._store,
                    skill_name=inv["skill_name"],
                    skill_version=inv["skill_version"],
                    skill_run_id=srid,
                    score=critic_score,
                    notes=f"agent_run#{run_id}: {notes}" if run_id else notes,
                )
            except Exception as e:
                log.debug("evolution_signals fan-out failed for %s: %s",
                          inv.get("skill_name"), e)

    def _record_start(self, *, goal: str, trigger_kind: str) -> int | None:
        """Insert a 'running' row into agent_runs and return its id."""
        try:
            with self._store.connect() as conn:
                cur = conn.execute(
                    "INSERT INTO agent_runs(trigger_kind, goal, status) "
                    "VALUES (?, ?, 'running')",
                    (trigger_kind, goal),
                )
                return int(cur.lastrowid or 0)
        except Exception as e:
            log.warning("agent_runs INSERT failed (non-fatal): %s", e)
            return None

    def _finalize(
        self,
        *,
        run_id: int | None,
        goal: str,
        trigger_kind: str,
        events: list[AgentEvent],
        t0: float,
        final_answer: str,
        iterations: int = 0,
        critic_score: float | None = None,
        critic_notes: str | None = None,
        total_cost_usd: float = 0.0,
        error: str | None = None,
        status: str | None = None,
    ) -> AgentRunResult:
        latency_ms = int((time.monotonic() - t0) * 1000)
        # W14.9: explicit status arg lets callers (notably the cancellation
        # branch) record agent_runs.status='cancelled' without faking an
        # error. Default behavior preserved: error → 'error', else 'ok'.
        if status is None:
            status = "error" if error else "ok"
        if run_id is not None:
            try:
                with self._store.connect() as conn:
                    conn.execute(
                        "UPDATE agent_runs SET ended_at = julianday('now'), "
                        "  status = ?, iterations = ?, final_answer = ?, "
                        "  trajectory_json = ?, critic_score = ?, critic_notes = ?, "
                        "  latency_ms = ?, cost_usd = ?, error_text = ? "
                        "WHERE id = ?",
                        (
                            status, iterations, final_answer,
                            json.dumps(
                                [asdict(e) for e in events],
                                ensure_ascii=False,
                                default=str,
                            ),
                            critic_score, critic_notes,
                            latency_ms, float(total_cost_usd), error,
                            run_id,
                        ),
                    )
            except Exception as e:
                log.warning("agent_runs UPDATE failed (non-fatal): %s", e)
        return AgentRunResult(
            run_id=run_id,
            goal=goal,
            trigger_kind=trigger_kind,
            final_answer=final_answer,
            events=events,
            iterations=iterations,
            cost_usd=float(total_cost_usd),
            latency_ms=latency_ms,
            critic_score=critic_score,
            critic_notes=critic_notes,
            error=error,
        )
