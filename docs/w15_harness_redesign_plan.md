# W15 — Harness Redesign Plan

> 状态：方向已定，待 Anthropic harness research 回填具体设计
>
> 最后更新：2026-05-06
>
> 触发：用户对比了 santifer/career-ops（42.8k★, 2026-04 创建），指出 OfferGuide
> "使用逻辑都不对"。3 轮深挖后定下方向。

---

## 0. TL;DR

**W14 系列做错了什么**：把 ambient agent 当成了"cron + ReAct + self_notes 小抄 + 工具 guardrails + 链式 prompt"——本质是 reactive agent 套了 ambient 的壳子，没有真心智。

**W15 要做什么**：极简 harness + 模型自主心智 + 求职特化 schema。所有"心智行为"
（planning / reflection / 通信判断 / 因果推理）**全部由模型 in-context 做**，不切成独立子系统；harness 只负责模型自己做不到的事（capabilities / memory / context / scheduling / feedback）。

---

## 1. 核心原则（不可让步）

### 1.1 Harness vs Agency 的硬边界

| Harness（**工程编码**） | Agency（**模型自做，工程不碰**） |
|---|---|
| 持久记忆（外部存储） | 反思 |
| 工具（capabilities） | 目标分解 / planning |
| 触发（cron / event） | 通信判断（说不说、何时说、说什么） |
| 反馈通道（GEPA） | 因果推理（用户为啥没回我） |
| Context 拼装 | 进化方向（哪些 SKILL 该改） |
| UI（心智窗口） | 决策（投不投、改不改） |

> "不能堆叠过程式逻辑暴力模拟智能 — 庞大的规则树、节点图、链式提示词瀑布流 —
> 然后祈祷胶水代码涌现自主行为。Agency 是学出来的，不是编出来的。
> 我们做的是 harness，心智是模型自己本身。"  — 用户原话，2026-05-06

### 1.2 Agent 主动性原则

**Agent 应主动做**：用户嫌烦、容易拖延、对结果重要的事
- 找岗位（持续扫、按 worldview 筛、主动推荐）
- 改简历建议（针对具体 JD 给定向修改建议）
- 沉默 followup 提醒（投了 N 天没回应）

**Agent 应等用户来**：用户自己有动力、不需要 push 的事
- 面试备战（用户带"我要面 X"来）
- 复盘（用户面完愿意聊）
- 最终决策（投不投、改不改、用哪份简历）

**判定边界**：用户**自己愿意干吗**？愿意 → 等。不愿意但重要 → 主动。

> 原则写进 system prompt 给 agent framing，**不**做成 if-else 规则。

### 1.3 节制靠学，不靠规则

"agent 该多主动 / 多频繁推送" 的尺度由 **GEPA 学习用户反馈**决定：
- 用户 ⊘ 一个推荐 → "下次少推这类" 进 SKILL evolution buffer
- 用户长时间不读 inbox → "推得太频" 信号
- 用户接受 + 投了 → 正信号

**不**做"每天最多推 N 个 / 每 X 天最多 followup 1 次"硬规则。

---

## 2. 求职 agent ≠ 通用 agent + 求职 prompt

求职这一面**决定了 harness 的 schema、tool 边界、context 优先级**。求职 agent
跟通用 agent 的差别：

| 维度 | 通用 agent | 求职 agent |
|---|---|---|
| Worldview 里**什么是真**的 | 任意话题 | candidate / tracked jobs / events / 校招日历 |
| **关心的时间尺度** | 任意 | 5-8 周内卷窗口 + 单场面试 24h 备战 |
| **关键事件** | 任意 trigger | 用户带 JD / 标记投了 / 收面试 / 面完 / 挂了 |
| **节奏感**从哪来 | 无 | 校招日历 |
| **失败定义** | 任务失败 | 错过 deadline / 投错岗 / 没准备挂掉 |
| **成功定义** | 任务完成 | 拿 offer |

**结论**：求职 agent 的心智不是空的。它生下来就知道"我在乎什么、我知道什么、
我能做什么"——这些是 **harness 里 hard-baked** 的（worldview schema 模板、tool
集合、校招日历常识），**不违反**"agency 不能编"原则，因为编的是 capabilities
+ context + memory，不是 behavior。

---

## 3. Worldview Schema（求职生命周期，agent 自维护的 markdown）

形态：单一 markdown 文件，agent 自己当主人，schema 模板是 harness 给的初始结构：

```markdown
# 我对 H 的理解
[agent 自由文本：cv 摘要、性格、偏好、雷区、目标演化]

# 当前阶段
[校招日历坐标 + 我观察到的用户节奏]

# 跟进中的岗位
- 字节 / Algo Intern / 投了 5 天 / 我建议 followup
- 阿里 / NLP Intern / 待评估
- ...

# 即将到来的事件
- 字节明天 14:00 一面

# 最近的复盘
[过去 N 次 action / 用户反应 / 我学到啥]

# 当前我的策略
[agent 自己定的，可改]

# 未解疑问
[agent 想找时机问的事]
```

Agent 自己往里写、自己改结构。Harness 只约定**文件位置 + 启动时模板**。

---

## 4. Tool 集合（求职活动的 capabilities，不是通用）

```
# Agent 主动调用
discover_jobs(criteria)        # 主动找岗位（W14.16 ReAct loop 升级版，是核心）
tailor_advice(job, cv)         # 找到值得投的岗位时配套出
notify_user(message)           # 主动推（节制由 GEPA 学）

# 用户来才被动用
interview_prep(job)            # 用户带"我要面 X"才用
reflect_outcome(job, outcome)  # 用户愿意聊面试结果才用

# 通用 capabilities
fetch_jd(url_or_text)          # 把岗位拉进来
score_match(job, candidate)    # 评估匹配
record_event(event)            # 求职事件入账（投了/约面/挂了）
update_worldview(diff)         # 改自己的脑
schedule_next_wake(when, why)  # 自决何时再醒
ask_user(question)             # 问用户
web_search(q) / fetch_url(u)   # 原始能力
```

**没有** `discover_pdf / batch_evaluate / generate_pdf` — 国内校招用 .docx 不用 PDF。

**Tool 边界 = 求职 agent 该有的肌肉边界**。多一个都是引诱跑偏的方向。

---

## 5. 触发系统（求职生命周期事件 + cron fallback）

| 事件 | 触发源 | Agent 反应 |
|---|---|---|
| 用户 paste 一个 JD | 用户主动 | 立即 wake |
| 用户标记"投了 X" | 用户主动 | 立即 wake，**agent 自己 schedule_next_wake(7d, "看字节回没回")** |
| 用户说"收到字节面试" | 用户主动 | 立即 wake，**但 agent 不主动凑过去 push 备战** |
| 用户面完 | 用户主动 | 立即 wake，复盘 + 学习 |
| 跟进岗位沉默 N 天 | Agent 自己之前 schedule 的 | wake 后看情况 |
| 用户长时间没出现 | 系统观察 | **不打扰**，agent 减少推送频率 |
| Cron heartbeat | 兜底 fallback | 弱触发，agent 看完没事就接着睡 |

**关键修正**：cron 心跳从 W14.18 的"主路径"退化为 **fallback safety net**。
**主路径**是 agent 自己 `schedule_next_wake(when, why)` 决策。

---

## 6. 校招日历感（决策 2：选 C — Hybrid）

**Hard-bake**（注入 context 给 agent 当事实，不是规则）：
- 国内大厂校招大日历常识（5 月 / 6 月 / 7 月 各阶段调性）
- 当前日期、距各阶段截止天数

**Agent 学**（动态、写进 worldview）：
- 具体公司具体截止（agent 通过 web_search 实时获取）
- 用户告诉 agent 的具体 deadline

> 理由：日历常识是事实不是规则，注入是 harness 本职；具体公司截止是动态的让
> agent 学。Cold start 不能太弱。

---

## 7. 找岗位策略（决策 1：选 C — Hybrid）

**主路径**（agent 主动）：
- W14.16 ReAct loop 升级版 — agent 用 Tavily web_search 主动找
- Criteria 从 worldview 里的 candidate model 来（不写死）
- 找到后用 SKILL 学习用户的 ⊘ 反馈

**辅助信号**（浏览器扩展，轻动作）：
- 用户在 BOSS 收藏一个岗 → 扩展同步进 OfferGuide
- 用户在 BOSS/牛客 已经表达过偏好的岗位池作为强信号
- **不**爬池子（反爬 + 工程量大）

---

## 8. W14 系列代码处置

| W14.x | 内容 | 处置 | 原因 |
|---|---|---|---|
| W14.16 | ReAct loop 找岗位 | ✅ **核心**，留并升级 | Agent 主动找岗位是主路径 |
| W14.17 | 砍硬编码 prompt 流程 | ✅ 方向对，再砍透 | 让模型主导 |
| W14.18 | 1 心跳 vs 3 daemon | ⚠️ **改**：cron 退化为 fallback；主路径是 agent 自决 schedule_next_wake | "事件驱动 + 自决" 才是真 ambient |
| W14.19 | Mission Control 真数据 | ✅ 保留为 /debug 视角 | 不上首屏 |
| W14.20 self_notes | self_notes 表 | ⚠️ **升级**为 worldview.md（agent 自维护 markdown） | Schema 化只保留最少元数据 |
| W14.20 inbox question | agent → user 双向通道 | ✅ 留 | 是 harness 必需的反馈通道 |
| W14.20 home 改 | "agent 在做啥" 视角 | ⚠️ **改内容**：主推 agent 找的岗位 + 简历建议 + followup 提醒，不是 inner monologue | 心智窗口要展示**有用**的脑内状态 |
| W14.18 _AGENT_WAKE_GOAL prompt 3-tier 工具组织 | 工具分类 prompt | ❌ **砍**，是工程师在替模型规划 | 让 agent 自己组织 |
| W14.16 tool guardrails ("不能 ingest 没 fetch 的 URL") | hard-coded 防错 | ❌ **砍**，让模型通过 GEPA 学避坑 | 不在 hard-coded layer 加 guardrail，让模型学 |

---

## 9. 极简 harness 骨架（待 Anthropic research 细化）

```
forever:
  1. 外部触发（cron fallback / event / user input）
  2. context = load_worldview() + recent_events + new_input + 校招日历事实
  3. response = model(context, tools=[...])  # 模型 in-context 做所有心智
  4. 执行 response 里的 tool calls
       ← 包括 update_worldview / notify_user / schedule_next_wake
  5. 持久化新事件 → 回到 1
```

工程师只编：
- worldview 文件读写 + 起手模板
- tool 集合（10 个左右）
- 触发系统 + cron fallback
- context 拼装（worldview + 最近事件 + 校招日历事实 + pending input）
- GEPA 反馈管道
- UI（心智窗口）

工程师**不编**：
- 反思层 / planning 层 / 通信 gate / hypothesis log
- 任何"什么时候该 X"的 if-else 规则

---

## 10. 待研究：Anthropic harness 设计（research agent 进行中）

需要回答的关键问题：

1. Claude Code 的 agent loop 长啥样？turn 之间持久化什么？
2. Context 管理（compaction、CLAUDE.md、TodoWrite、system reminder）
3. Tool 暴露方式 + guardrail 在 schema-level 还是 system-prompt-level？
4. 持久记忆（CLAUDE.md / memory / hooks / settings.json 各自定位）
5. Anthropic 关于 "context engineering" 的明确主张
6. Long-horizon 处理（TodoWrite 哲学、plan mode、sub-agent）
7. Anti-patterns（Anthropic 明确反对的设计）
8. Communication 判断（何时问 vs 继续）
9. Ambient / persistent agent SDK 例子
10. What goes in harness vs left to model 的明确边界原则

Research 完成后回到本文档，补 §11 "具体 harness 设计（参照 Anthropic）"。

---

## 11. 具体 harness 设计（参照 Anthropic）

### 11.1 Anthropic 的核心立场（research 回来的硬证据）

来源：[Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)、
[Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)、
Claude Code docs、Claude Cookbook context engineering recipe。

> "The model functions as a stateless token predictor providing raw cognitive
> reasoning, while the harness comprises the Runtime Software Infrastructure
> that coordinates tool dispatch, context management, and safety enforcement."

> "Context engineering is the art and science of curating what will go into
> the limited context window from that constantly evolving universe of
> possible information."

**一句话**：Anthropic 的 harness **是故意笨的**——它不做决策，只协调 Claude
的决策。所有 agency in-context via model。所有 persistence file-based。所有
safety 在 boundary。

这跟我之前提的"7 个 sub-system" 完全反向，**用户骂得对**。

### 11.2 Anthropic 反对的 anti-patterns（逐条对应我之前的错）

| Anti-pattern | 我之前提的方案 |
|---|---|
| ❌ Planner-Executor-Reflector chain | "反思 wake 独立于 action wake / planning 层 / 通信 gate" |
| ❌ Hard-coded guardrails in prompt | W14.16 tool guardrails ("不能 ingest 没 fetch 的 URL") |
| ❌ Tool overload (50 tools) | 还好我控制在 10 个左右 |
| ❌ Context bloat | W14 snapshot 一直在加东西 |
| ❌ Chain-of-thought as audit trail (model misrepresents) | Mission Control 把 inner_monologue 当真相展示 |

### 11.3 Anthropic 的核心 building blocks（直接照搬）

#### A. Single-threaded master loop

```
WHILE not done:
  1. Gather context (history + memory + tool results)
  2. response = client.messages.create(messages, tools, context_management)
  3. IF response.stop_reason == "tool_use":
       execute tools sequentially → ToolResult blocks → append to messages
     ELSE:
       return final text → mark done
```

**不是** planner → executor → reflector chain。**就一个 loop**。

#### B. 3 层 context management（Anthropic API 原生支持）

| 层 | 触发 | 行为 | OfferGuide 用法 |
|---|---|---|---|
| **Compaction** (`compact_20260112`) | input_tokens > 150K | 模型自己 summarize history，summary 替换 history | 长会话省钱，保留 architectural decisions |
| **Tool-result clearing** (`clear_tool_uses_20250919`) | input_tokens > 30K | 移除老 tool output blocks，保留 tool call 记录 | 大量 fetch/search 输出别堆积 |
| **Memory tool** | 模型主动调用 | 读写 memory 文件 | 跨会话知识 |

#### C. Memory 三层（Claude Code 模式）

| 层 | 位置 | 谁写 | 何时载入 |
|---|---|---|---|
| **CLAUDE.md** | 项目根 / 用户 home | 工程师手写 | 每次 session 开始全文 |
| **Auto MEMORY.md** | `~/.claude/projects/<proj>/memory/` | Agent 自动写 | 每次 session 开始前 200 行 |
| **Memory tool** | 同上目录的其他 .md 文件 | Agent 用 tool 调用 | On-demand |

#### D. Hard-coded guardrails 全部在 boundary，零在 prompt

| 位置 | 用途 | OfferGuide 对应 |
|---|---|---|
| `.claude/settings.json` permissions | Allowlist/denylist tool calls | 暂不需要 |
| Hooks (PreToolUse) | Pre-execution 验证 / 拒绝 | 已有的 cost_tracking / circuit_breaker 类似机制 |
| System prompt + CLAUDE.md | "soft" guidance, 模型学 | 主路径 |

**关键**：Anthropic **零** hard-coded guardrail in tool schema。"agent 不能在没
fetch URL 的时候 ingest" 这种规则——**让 agent 自己学**。失败一次 → 写进
memory → 下次自己避开。这才是 agency。

### 11.4 OfferGuide W15 harness 文件结构（照搬 Claude Code）

```
src/offerguide/harness/                # 新模块
├── loop.py                            # Single-threaded master loop
├── context.py                         # Context assembly + 3 层 management
├── memory.py                          # Memory tool 实现 (read/write worldview)
├── tools.py                           # Tool 定义 (10 个左右，求职特化)
├── triggers.py                        # 事件 + cron fallback
├── instructions.md                    # = Claude Code 的 CLAUDE.md
│                                       # 求职 agent 的"how it works"
│                                       # - 求职 agent 的目标
│                                       # - 校招日历常识 (hard-bake §6 的 C)
│                                       # - tool 用法
│                                       # - 主动 vs 被动原则 (§1.2)
│                                       # - 通信判断 framing (§11.5 D)
│                                       # - 当前 SKILLs 列表
└── feedback.py                        # GEPA 反馈管道

.offerguide/                           # 用户/agent 数据 (gitignored)
├── worldview/
│   ├── MEMORY.md                      # = Claude Code 的 auto MEMORY.md
│   │                                   # Agent 自维护 markdown，§3 的 schema
│   ├── candidate.md                   # CV + 偏好 + 雷区
│   ├── tracked-jobs.md                # 跟进岗位 + 状态
│   ├── upcoming-events.md             # 面试 / deadline
│   ├── reflections.md                 # 复盘
│   └── strategy.md                    # 当前策略 + 未解疑问
└── sessions/
    └── session-<id>.jsonl             # 完整对话 + tool results
```

### 11.5 关键映射（Anthropic 模式 → OfferGuide）

#### A. Loop（替换 W14.16 + W14.18）

```python
# src/offerguide/harness/loop.py
async def run(trigger_context: TriggerContext) -> RunResult:
    messages = build_initial_messages(trigger_context)
    while True:
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            system=load_instructions(),       # instructions.md
            messages=messages,
            tools=ALL_TOOLS,                  # 10 个左右
            context_management={
                "edits": [
                    {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 150_000}},
                    {"type": "clear_tool_uses_20250919", "trigger": {"type": "input_tokens", "value": 30_000}, "keep": {"type": "tool_uses", "value": 4}},
                ]
            },
            max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            return RunResult(messages=messages, ...)
        # tool_use
        tool_results = await execute_tool_calls(response.content)
        messages.append({"role": "user", "content": tool_results})
```

#### B. Worldview = Auto MEMORY.md（替换 W14.20 self_notes 表）

- `agent_self_notes` SQL 表 → `.offerguide/worldview/MEMORY.md` markdown
- 起手模板 = §3 的 schema
- Memory tool: `view`, `str_replace`, `insert`, `create`（按 Anthropic memory tool spec）
- 每次 wake：`load_first_n_lines(MEMORY.md, 200)` → 注入 system context
- Agent 想看具体 → 自己调 memory tool

#### C. Tool 集合（求职特化 10 个）

```python
ALL_TOOLS = [
    # Memory (harness layer)
    Memory,                # view / str_replace / insert / create on worldview/*.md
    
    # 求职 capabilities
    discover_jobs,         # ReAct 找岗位 (W14.16 升级)
    fetch_jd,              # paste/url → 入库
    score_match,           # 评估匹配
    tailor_advice,         # 简历定向修改建议
    interview_prep,        # 面试备战 (用户带"我要面 X"才用)
    reflect_outcome,       # 面试后复盘
    record_event,          # 求职事件入账
    
    # Communication
    notify_user,           # agent 主动推 (节制由 GEPA 学)
    ask_user,              # agent 问问题
    
    # Self-management
    schedule_next_wake,    # agent 自决何时再醒
    
    # 原始能力
    web_search,
    fetch_url,
]
```

**砍**：W14 的 `write_note_to_self / clear_self_note`（用 Memory tool 替代）、
maintenance dispatch 的 `discover_new_jobs / score_unscored_jobs`（不再有
maintenance daemon）。

#### D. Instructions.md（替换 W14.18 _AGENT_WAKE_GOAL）

按 §1.2 主动性原则 + §6 校招日历常识 + tool 用法 + 通信判断 framing 写。**单段
陈述目标 + 给工具 + 给原则**，不**是** "3-tier 工具组织"。

伪结构：
```markdown
# OfferGuide 求职 Agent

## 你帮谁
[从 worldview/candidate.md 学，启动时为空]

## 你的目标
帮 H 拿到 2026 暑期 AI Agent / LLM 应用岗 offer。

## 求职日历常识
- 2026-05: 大厂暑期投递高峰收尾，部分公司已截止
- 2026-06: 面试季，1 面 / 2 面 / HR 面
- 2026-07: offer 季

## 你的工具
[简洁列表 + 用法 hint，不写"3-tier 分类"]

## 主动 vs 被动原则
**应主动做**：找岗位、改简历建议、followup 提醒
**应等用户来**：面试备战、复盘、最终决策

## 何时通知用户
- 找到高匹配岗位
- 投递 N 天没回应
- 用户 worldview 里写过的 deadline 临近

## 何时不通知
- 你在 worldview 里小修小补
- 你刚 web_search 没结果
- 你已经推过类似的（看 reflections.md）

## 节制
不重复推已经推过的岗位（看 tracked-jobs.md）。如果用户 ⊘ 过类似的，少推。
```

#### E. Triggers（替换 W14.18 cron 主路径）

```python
# 主路径：求职生命周期事件
on_event("user_paste_jd")       → wake
on_event("user_marked_applied") → wake (agent 自己 schedule_next_wake)
on_event("user_received_interview") → wake
on_event("user_finished_interview") → wake
on_event("scheduled_wake_due")  → wake (agent 之前自己定的)

# Fallback：cron 心跳
cron("0 */6 * * *") → wake_check  # 每 6h, agent 看看有没事，没事接着睡
```

#### F. UI 心智窗口（替换 W14.20 home）

主体内容（不是 Mission Control debug）：
- Agent 现在对你的理解（worldview/candidate.md 摘要）
- 你在跟进的岗位（worldview/tracked-jobs.md 摘要）
- Agent 给你找的新岗位 + 简历建议 + followup 提醒（high-priority inbox）
- Agent 想问你的问题（pending question）
- Chat 入口（你想跟 agent 说啥）

**砍**：home 上的 inner_monologue / Mission Control（移到 /debug）。

### 11.6 Anthropic 安全栈（OfferGuide 现状对比）

| Anthropic | OfferGuide 已有 | 缺 |
|---|---|---|
| Permissions allowlist | 无（个人项目暂不需要） | - |
| Hooks (PreToolUse) | 有 cost_tracking / circuit_breaker 类似机制 | 接到 harness loop |
| Sandboxing | 无 | 后续可加 (subprocess + restricted env) |

---

## 12. 砍 / 留 / 新增清单（带文件路径，待跟用户对齐再动）

### 砍

| 文件 / 模块 | 理由 |
|---|---|
| `src/offerguide/agentic/job_finder_agent.py` 的 4 个 hard-coded tool guardrail | Anthropic 反对；让 agent 学 |
| `src/offerguide/autonomous/scheduler.py` 的 `_AGENT_WAKE_GOAL` 的 3-tier 工具分类 | 工程师替模型规划 |
| `src/offerguide/agent/maintenance.py` 的 `discover_new_jobs` / `score_unscored_jobs` 工具 dispatch | 不再有 maintenance daemon 概念 |
| `src/offerguide/memory/db.py` `agent_self_notes` 表 | 升级为 worldview/*.md 文件 |
| `src/offerguide/agent/loop.py` `write_note_to_self` / `clear_self_note` tool | 用 Memory tool 替代 |
| `src/offerguide/ui/templates/home.html` 的 inner monologue 大卡片 | 移到 /debug，home 改成"agent 给你的东西" |
| Mission Control 上首屏 | 移到 /debug |

### 留并升级

| 文件 / 模块 | 升级方向 |
|---|---|
| `src/offerguide/agentic/job_finder_agent.py` ReAct loop | 抽到 `harness/loop.py` 通用版 |
| `src/offerguide/inbox.py` kind=question | 接进新 ask_user tool |
| `src/offerguide/evolution/` GEPA 系统 | 接通用户反馈管道 (feedback.py) |
| `src/offerguide/skills/*/` SKILL 体系 | 保留，作为 tool 内部实现 |
| `src/offerguide/ui/web.py` FastAPI 路由 | 加 /debug；改 / 的内容 |

### 新增

| 新文件 | 用途 |
|---|---|
| `src/offerguide/harness/loop.py` | Single-threaded master loop |
| `src/offerguide/harness/context.py` | Context assembly + 3 层 management |
| `src/offerguide/harness/memory.py` | Memory tool 实现 (worldview/*.md 读写) |
| `src/offerguide/harness/tools.py` | 10 个 tool 定义 |
| `src/offerguide/harness/triggers.py` | 事件 + cron fallback |
| `src/offerguide/harness/feedback.py` | GEPA 反馈管道 |
| `src/offerguide/harness/instructions.md` | "how this agent works"（CLAUDE.md 等价） |
| `.offerguide/worldview/MEMORY.md` 模板 | 起手 schema |

---

## 13. 完成日志

| W15.x | 内容 | 文件 |
|---|---|---|
| W15.0 | 摸清 W14 现状 | (无产出，5 个文件 grep) |
| W15.1 | Harness 骨架 8 文件 | `src/offerguide/harness/{__init__,_schema,context,feedback,instructions.md,loop,memory,tools,triggers}.py` |
| W15.2 | Memory tool 6 动作 + worldview 模板 | `harness/memory.py`（MemoryStore + MEMORY_TOOL_SCHEMA + 6 bootstrap files） |
| W15.3 | 求职特化 13 tools + dispatch | `harness/tools.py`（memory + discover_jobs / fetch_jd / score_match / tailor_advice / interview_prep / reflect_outcome / record_event / notify_user / ask_user / schedule_next_wake / web_search / fetch_url） |
| W15.4 | 数据迁移脚本 | `scripts/migrate_w14_to_w15.py`（dry-run / apply 模式） |
| W15.5 | Triggers 系统 | `harness/triggers.py`（fire_event / poll_pending / scheduled wake cleanup / cron heartbeat） |
| W15.6 | instructions.md | `harness/instructions.md`（求职 agent 灵魂 + 校招日历常识 + 主动性原则 + 通信判断 framing） |
| W15.7 | 砍 W14 dead code | `scheduler.py` 砍 `_AGENT_WAKE_GOAL`/`_discover_via_search_job`/`_auto_score_via_daemon`；改 `_wake_agent_job` 走 harness; `web.py` 的 `/api/scheduler/trigger/{job_name}` 改走 harness |
| W15.8 | GEPA 反馈管道 | `harness/feedback.py`（接 W6 evolution_signals schema） |
| W15.9 | UI 心智窗口 | `home.html` 加 worldview 摘要 + chat 输入框 + harness_runs strip; `web.py` 加 `/api/home/chat` + `/debug` 路由; `debug.html` 新模板; `base.html` nav 加 /debug 链接 |
| W15.10 | 新 W15 tests + 全 pass | `tests/test_w15_harness.py`（55 tests: 13 memory + 5 context + 11 tools + 5 loop + 6 triggers + 5 feedback + 5 UI） |
| W15.11 | inbox decide/answer 接进 evolution_signals | `web.py` 的 `/inbox/{id}/decide` + `/inbox/{id}/answer` 调 `harness_feedback.on_inbox_*` / `on_question_answered`; +7 tests |

## 14. 最终验证

### 静态
- ✅ src/ ruff: 0 errors
- ✅ harness/ pyright: 0 errors
- ✅ pytest: **780 passed + 12 skipped** (had 718 before W15; +62 new W15 tests, 0 failures)
- ✅ Memory tool 6 动作 smoke test + path traversal 阻拦
- ✅ /debug + home 心智窗口 渲染验证
- ✅ Migration script idempotent, 检测 fresh DB 不报错

### 真实 server end-to-end (2026-05-06)
跑活的 server (uv run --extra ui python -m offerguide.ui.web), 用真 DeepSeek API:

- ✅ 启动 → worldview/ 自动 bootstrap 6 个 markdown 文件 (MEMORY/candidate/tracked-jobs/upcoming-events/reflections/strategy)
- ✅ Home: 心智窗口卡片展示 worldview MEMORY.md 前 60 行; chat 输入框; harness_runs 摘要 strip
- ✅ /debug: 4 section 渲染 (Harness Runs / Scheduled Wakes / Harness Events / Daemon Runs)
- ✅ POST /api/home/chat ("帮我看看现在的状态") → 200 OK
- ✅ Harness loop 跑了 8 iter, $0.0835 cost, status=ok
- ✅ Agent 自主调 memory tool 7 次 (view 每个 worldview 文件) — **真心智在 in-context 决策**, 不是 hardcoded plan
- ✅ Agent 给出诚实评估: "worldview 是全新的，我不了解你"+ 列出系统事实 (日期 2026-05-06 周三, 校招阶段判断) + 主动问 3 个具体问题填 candidate.md
- ✅ 重新进 home → harness_runs strip 真实展示这一次 wake (status=ok, 8 步, $0.0835)

**关键观察**: agent 没瞎调 discover_jobs 找岗位 — 它先意识到"我不了解用户" → 优先 ask_user. 这正是 instructions.md 里"主动 vs 被动原则"在 in-context 起作用, 不是 if-else 编出来.

## 15. 给 GEPA / 后续工作的留白

- ✅ ~~evolution_signals 接通了，但目前 inbox decide 路由还没把 user 反馈调 `feedback.on_*`~~ → W15.11 已接通
- worldview/MEMORY.md 的进化（agent 自己往里写）需要真 LLM 跑才能验证；当前 stub 测试覆盖 schema
- W14 central agent (agent/loop.py + maintenance.py) 仍存在（deprecated path），未删；后续可清

**爆破半径实际**：1 个新模块 (harness/, 8 文件) + 2 个改文件 (scheduler.py, web.py) + 2 个新模板 (debug.html + home.html 修改) + 1 个迁移脚本 + 1 个新 tests 文件 (55 tests)。**没有破坏任何现有 tests（718 → 773 全 pass）。**

---

## 16. W15.12 — review 撞出 7 bug + 6 smell, 都修了

正式 code review 后发现 7 个会在生产坏的 bug 和 6 个 design smell. 全部修了 + 加 13 个 regression test.

### Bug 修复

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | loop.py:223 | max_iterations 命中后 status='ok' 误导 telemetry | 加 STATUS_TRUNCATED, 按 finish_reason 分 status |
| 2 | loop.py:175-197 | 模型同时返 content + tool_calls 时 final_text 丢失 | 每 iter 累积 resp.content 到 final_text_parts |
| 3 | loop.py:148-225 | 无 try/finally — context mgmt / dispatch crash 让 harness_runs 永远 'running' | 整个 loop 包 try/finally, 异常路径走 'error' status |
| 4 | tools.py:487 | paste:// URL 用 builtin hash() (process-randomized) → dedup 失效 | 改用 hashlib.sha256(raw).hexdigest()[:16] |
| 5 | tools.py:430 + job_finder_agent.py | discover_jobs sub-agent cost 不进 telemetry — 一次 sweep $0.10-0.20 漏记 | JobFinderResult 加 total_cost_usd; HarnessDeps.extra_cost_usd 流回主 harness_runs |
| 6 | feedback.py:91 | notes JSON 截断到 2000 字会切到字符串中间 → 无效 JSON → GEPA 崩 | 先按 user_text/metadata 内部缩, 保 dump 出来一定 valid |
| 7 | scheduler.py:362 | pt.cleanup() 不在 finally — harness 失败时 scheduled_wake 永不 fire → 无限循环 | cleanup 移到 finally, 失败也跑 |

### Smell 修复

| # | 问题 | 修复 |
|---|---|---|
| 1 | Agent 第一次 wake 浪费 7 iter view 每个 worldview 文件 — auto-load 没告诉 agent 其它文件状态 | auto_load 加文件索引 (每文件 1 行: 行数 + 第一个 heading); instructions.md 明确"不要每次 wake 都 view 所有文件" |
| 2 | compaction LLM call 没 timeout 防御 | 加 broader Exception catch, 失败走 noop 不卡死 |
| 3 | SKILL 输出 JSON 解析失败时返 "{}" 让 agent 蒙圈 | fallback 显示 raw_text 前 500 字让 agent 自纠 |
| 5 | build_deps 静默吞 search/skills/profile/notifier 初始化错 | 全部 log.warning, 不静默 |
| 6 | loop.py 死代码 _ = tools / _ = _dt | 砍了 |
| 8 | memory view 大文件不截断 | 默认 500 行截断 + 提示 view_range 看更多 |

### 验证

- ✅ src/ ruff: 0 errors
- ✅ harness/ pyright: 0 errors
- ✅ pytest: **780 → 793 passed + 12 skipped** (+13 review-fix tests, 0 既有失败)

### 关键 review 教训

我 W15 一边写一边觉得"测试都过了应该没事". 这次正式 review 才发现：测试 pass ≠ 代码对.
- Bug 1 测试不会失败因为没人查 status='ok' 是不是该是 'truncated'
- Bug 5 测试不会失败因为我们 stub LLM 永远 cost=0
- Bug 3 测试不会失败因为没人故意制造 ctx mgmt 异常
- Bug 7 测试不会失败因为没人故意让 harness_run 抛错

**面试如果被问"你怎么知道你的代码没问题"**: "代码 review 是和测试不同的活. 测试验证我以为该测的, review 验证我没想到该测的. 这次发现 7 个 bug 都不在原 test set 覆盖范围."
