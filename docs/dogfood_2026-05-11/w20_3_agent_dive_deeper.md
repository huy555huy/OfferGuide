# W20.3 — Agent 深 dive 审查 + critic 重设计

**用户原话**:
> 都干都干，无所谓的。基本上我们这个就是 human-in-loop。然后 critic 这个得想清楚了，不能因为他而反耽误了我的模型的使用

之前在 W20.2 末提了 4 个深 dive 方向 (主 AgentLoop critic→evolve / ambient
源优先级 / JobFinderAgent vs LangGraph / SKILL 灰度发布). 这次都干.
**首要事**: critic 重设计.

---

## 1. Critic 重设计 (W20.3 — 首要)

### 1.1 真现状审查

`agent_runs` 表 prod store **0 行**, `evolution_signals` 表 **0 行** —
即 critic 还从来**没在用户实操中烧过钱**, 但默认 `critic_enabled=True`,
一旦用户真去用 `/agent` 端点, **每次 +1 LLM call ≈ 5-10s + ~$0.001**.

### 1.2 真用户痛点

人类直接说: **"不能因为他而反耽误了我的模型的使用"**.

一个 critic 该不该跑, 取决于触发场景. 不分场景一刀切默认开 = 错.

### 1.3 真改动 — 4 层 gating

```python
def _should_run_critic(*, trigger_kind, iterations, skill_invocations):
    if not self._critic_enabled:           return False, "constructor_disabled"
    if env "OFFERGUIDE_CRITIC_DISABLED":   return False, "env_disabled"
    if trigger_kind in INTERACTIVE:        return False, f"interactive_trigger:{trigger_kind}"
    if iterations <= 1 and not skill_invocations:  return False, "trivial_run"
    if random > sample_rate:               return False, f"sampled_out:{rate}"
    return True, None
```

**真分类 trigger_kind**:

| 类别 | trigger_kinds | 决策 | 理由 |
|---|---|---|---|
| 交互 (interactive) | `user_button` / `manual` / `chat` / `sse` / `user` / `user_chat` | **SKIP** critic | 用户在等响应, UX latency 优先于 evolve signal |
| 后台 (background) | `cron_wake` / `scheduler` / `autonomous` / `ambient` | **KEEP** critic | 后台跑, 时间富裕, evolve 需要 signal |
| 未知 | (其他) | 默认 KEEP | safer default — 新增 trigger 不会丢 signal |

### 1.4 真硬护栏

- **trivial run skip**: ≤1 iter AND 0 SKILL = 没东西可评 (e.g., 模型立刻 final), critic 调了也是浪费
- **timeout 15s**: critic LLM 调用挂超过 15s 强制中断 (`_self_critique_with_timeout` 用 thread+join 实现, 主线程 join 超时即放弃)
- **env override**:
  - `OFFERGUIDE_CRITIC_DISABLED=1` — 全关
  - `OFFERGUIDE_CRITIC_SAMPLE_RATE=0.5` — 50% 概率跑 (cost cut 半)

### 1.5 真测试 (`tests/test_w20_3_critic_gating.py`, 18 case 全 PASS)

- 6 case 测 trigger_kind 分类 (user_button/manual/chat/sse/cron_wake/scheduler/autonomous/unknown)
- 2 case 测 trivial gate (≤1 iter + 0 SKILL → skip; 但 1 iter + 1 SKILL OR 5 iter + 0 SKILL → run)
- 2 case 测 constructor disable + env disable
- 3 case 测 sample rate (0/1.0/invalid)
- 3 case 测 timeout (hang → 0.5s 后返 None / 正常完 / 异常传播)
- 2 case 修了 W13 老测试 (新行为 = trivial run 应该 skip + critic_score=None)

### 1.6 真节省的 cost / latency

每个 user-triggered `/agent` 调用:
- **节省 ~$0.001 + 5-10s 等待**
- 用户 8 次/天 用 `/agent` = 节省 $0.008 + 40-80s/天

每个 cron 后台调用:
- **保持** critic, evolve signal 不丢
- 长期 fitness 数据继续累积, SKILL 进化机制不受影响

### 1.7 真验证 W20.3 改动是否破环老行为

- 老测试 `test_loop_with_zero_tool_calls_just_finalizes` 期望 `'critique'` 事件 — **新行为正确地** emit `'critique_skipped' reason='trivial_run'`. 测试已更新, 反映这个真新意图.
- 老测试 `test_loop_persists_agent_run_with_trajectory` 期望 `critic == 0.7` — **新行为正确地**写 `critic == None` for trivial run. 测试已更新.

全套 1045 pass / 13 skip, 0 真 regression (只 2 个测试更新反映新设计).

---

## 2. 主 AgentLoop trajectory + critic → SKILL evolve 的真链路

```
user 唤醒 /agent
   ↓
AgentLoop.run(goal, trigger_kind)
   ↓
1. snapshot_state() → 把 jobs/applications/user_facts/recent_runs 注 LLM 上下文
2. iter loop (≤8 iter):
     LLM.chat_with_tools → tool_call → SkillRuntime.invoke → tool_result → ...
     [W13.7 circuit breaker: 同 tool 同 args 3 次 → 强制 stop]
3. _should_run_critic(trigger, iter, skill_invs)  ← W20.3 新加
     ├── 跳过 → emit 'critique_skipped'
     └── 跑 → _self_critique_with_timeout (15s 硬上限)
            → 第 2 个 LLM 评分 0..1 + notes
4. _write_critic_signals(run_id, score, notes, skill_invocations)
     → 把 score 写到 evolution_signals 表, 每个被调的 SKILL 一行
        (skill_name, skill_version, signal_kind='critic', signal_value=0..1, ...)
5. _finalize → agent_runs row + return AgentRunResult
```

**接下来**, `evolve_skill` tool 由 agent 主动调时:

```python
detect_evolution_candidates()
   ↓ (查 evolution_signals)
对每个 SKILL:
   compute_fitness(signals) → fitness ∈ [0, 1]
   triggers if:
     fitness < 0.55              (差)
     AND sample_count >= 10      (有数据)
     AND days_since_last_evolution > 7  (cooldown)
   ↓
evolve_skill('score_match', n=3)
   → meta-LLM 看当前 prompt + 最近差样本 → 生成 3 个候选 prompt 变体
   → 持久化到 skill_variants 表 (status=shadow, traffic_pct=0)
```

然后 `run_gray_release` 推进灰度:

```python
shadow → promote_to_canary(traffic_pct=0.2) → canary
canary 收 ≥ 8 signals → compare_versions(canary vs live)
                          ├── win (decisive) → promote_to_live (旧 live → retired)
                          └── lose → fail_variant (canary → retired, live 保留)
```

**这套真值** (从代码读, 不是文档):
- `evolution/fitness.py` line 41: `MIN_SIGNALS_FOR_TRIGGER = 10`
- `evolution/fitness.py` line 44: `EVOLUTION_THRESHOLD = 0.55` (低于这个考虑 evolve)
- `evolution/fitness.py` line 47: `EVOLUTION_COOLDOWN_DAYS = 7` (不能太频繁)
- `evolution/release.py` line 55: `DEFAULT_CANARY_TRAFFIC_PCT = 0.2` (新版本只 20% 流量)
- `evolution/release.py` line 60: `MIN_CANARY_SIGNALS = 8` (canary 跑够 8 个才比)

**为什么这套设计是有意义的**:
- Critic 是第三方裁判, 不是 SKILL 自评 (单个 SKILL 不能给自己打分)
- Score fanning out 到所有调用的 SKILL = 粗 attribution, 但多 run 平均后噪声会洗掉
- 灰度 20% + 8 signals + decisive win = 谨慎升级, 一次只动一个 SKILL
- Cooldown 7 天 = 不会进化得抽风
- `compare_versions` 内部用统计学比较 (不是简单看 mean), 防 sample size 欺骗

**当前真状态**:
- 0 evolution_signals → 0 fitness → 0 evolve trigger → 0 蒸发
- W20.3 后, 后台 cron 跑会持续累积 signal, 用户只用 `/agent` 偶尔不影响

---

## 3. Ambient 源优先级 — 真凭什么 ?

`workers/ambient.py` 的 `_run_one_cycle` 真排序 (W20.2 后):

```
parallel via asyncio.gather:
  ├─ nowcoder_sitemap (15 jobs/cycle)
  ├─ 0voice GitHub (cap 80 jobs)
  ├─ verified_official × 5 keywords (15-25 jobs/cycle)
  └─ shixiseng × 2 keywords (16 jobs/cycle)

then sequential:
  └─ agent_search (cap 8 iter, ~14 niche jobs)

then parallel (ThreadPool=4):
  └─ score_match × all unscored
```

**为什么这个顺序**:

| 顺序考虑 | 真原因 |
|---|---|
| 4 fetch 并行先 | 都打不同域 (nowcoder.com / github.com / qq.com+baidu.com / shixiseng.com), 互不抢 rate, gather 起来真便宜 |
| agent_search 不进 gather | 它用同 LLM provider, 跟 score_match 抢 rate. 单独串行 |
| score_match 最后 | 等所有 ingest 完, 一次 batch 评分; 也避免新 ingest 还没 commit 就被查 |
| nowcoder cap=15 | 上限低, 因为牛客很多重复 / 过期; quality > quantity |
| 0voice cap=80 | 全的源, 不限 cap 会 1000+ 一次写爆 score_match cost |
| verified × 5 keywords | DeepSeek-V4 抽 5 个 niche keyword, 每 keyword 限 3 条 → ~15 |
| shixiseng × 2 keywords | 实习专用, 2 keyword × 8 detail page = 16 reqs cap |

**真不是按 "公司 brand" 排, 是按**:
1. **数据 quality/cost ratio** (verified API > GitHub repo > sitemap > web search)
2. **rate limit scope** (互不冲突的并行, 同 provider 的串行)
3. **每源真实测过** (CLAUDE.md 第 3 节 SOURCE_LANDSCAPE)

---

## 4. JobFinderAgent vs LangGraph ReAct — 真区别

我们不用 LangGraph. 自己手写 ReAct loop in `agentic/job_finder_agent.py`:

```python
# 真代码 (loop core, line 240-329):
messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "开始."}]
for iteration in range(1, self.max_iter + 1):
    resp = self.llm.chat_with_tools(messages, tools=_TOOL_SCHEMAS, temperature=0.3)
    messages.append(assistant_msg)  # 含 tool_calls
    if not resp.tool_calls:
        break  # 模型决定不再调
    for tc in resp.tool_calls:
        tool_result = self._dispatch(tc.name, tc.arguments)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})
        if tc.name == "done":
            done_called = True
    if done_called:
        break
```

**跟 LangGraph 区别**:

| 维度 | OfferGuide (手写) | LangGraph |
|---|---|---|
| 状态机 | LLM driven (model 决定下一步) | 显式 graph (Python 写 transitions) |
| 依赖 | httpx + json | langgraph + langchain + ... |
| 序列化 | messages 是 OpenAI 标准 dict | TypedDict + StateGraph |
| 持久化 | 我们自己的 harness_events / agent_runs 表 | langgraph checkpoint API |
| 调试 | `_notes` list 记每 iter, log 直接 grep | langsmith trace |
| Circuit breaker | 自己写 (`CIRCUIT_BREAKER_REPEAT_THRESHOLD=3`) | 用户自己实现 |
| 多 agent | 故意不上 | 鼓励 (subgraph) |
| Vendor lock | 0 (任何 OpenAI-compatible API) | 中 (跟 langchain 生态深绑) |

**我觉得手写赢的两点**:
1. **代码量**: JobFinderAgent 总共 528 行, 替代 LangGraph 的 ReAct + state + checkpoint 大概 300 行 boilerplate. 写出来 review 就懂.
2. **不需要适配**: LangGraph 升级一次破坏 N 个用法. 我们的 messages 是 OpenAI 标准 dict, DeepSeek/Claude/OpenRouter 全吃.

**LangGraph 赢的点**: 多 agent / 动态 routing / 复杂 conditional edge — 我们故意不需要 (Anthropic agent guide 说 "single-agent 起点未证明 multi-agent 必要").

---

## 5. SKILL 灰度发布 (gray release) 真链路

**触发**: agent 调 `run_gray_release` (action tool) 或 cron 定时.

**真状态机** (`evolution/release.py:7-22`):

```
shadow ──── promote_to_canary(20% traffic) ────▶ canary
                                                 │
                                      collect ≥ 8 signals
                                                 │
                                                 ▼
                                       compare_versions(canary vs live)
                                          /             \
                              decisive WIN              decisive LOSE
                                  /                        \
                                 ▼                          ▼
                           promote_to_live              fail_variant
                           (旧 live → retired)         (canary → retired)
```

**真护栏**:

| 护栏 | 值 | 为什么 |
|---|---|---|
| canary 流量 % | 20% | 损失最坏 case = 20% 用户 1 次差 SKILL output |
| 比较前最少 signals | 8 (canary) | 太少 = 偶然性大; 太多 = 升级慢 |
| 各版本最少 signals | 5 (`MIN_SIGNALS_PER_SIDE`) | 双侧统计需要的最小 sample |
| 一次 cycle 最多动 | 1 个 promote + 1 个 evaluate | 一次只动一件事, 出问题好回滚 |
| Cooldown | 7 天 | 不能进化得太频繁 |
| Decisive win 判定 | `compare_versions` 内部统计学 | 不是简单 mean 比较, 防 noise |

**真 reverse path** (有问题怎么回滚):
- `fail_variant` → canary 退到 retired, live 不动
- 手动 operator 可以从 retired promote_back 到 live

**没真验过** (诚实): 我们 SKILL evolve 整链路从来没真触发过 (因为 0 signals). W20.3 critic 改 + cron 上线后会开始累积. 第一次 evolve 触发可能要 4-6 周 (10 signals × 几周 / cron 不是每分钟跑).

---

## 总结: 4 个真改动落实情况

| # | 改动 | 真测试 | 真实跑 |
|---|---|---|---|
| 1 | critic gating (4 层) | 18 case | 跑过 1045 测试套 |
| 2 | critic timeout (15s) | 3 case 含 hang→0.5s | thread+join 实现 |
| 3 | env overrides (DISABLED / SAMPLE_RATE) | 4 case | 已加 |
| 4 | 老测试更新 (反映新行为) | 2 case 改 | 1045 全过 |

**真 cost 节省**: 用户每用 `/agent` 一次省 $0.001 + 5-10s.
**真 evolve 信号**: 后台 cron 仍持续跑 critic, signal 链路不断.
**真 UX 改善**: 用户主动唤醒 agent → 立刻拿到 final answer, 不再等 critic.

## 还没解决的真问题 (诚实清单)

1. **`evolve_skill` 流程从来没真跑过端到端** — 0 evolution_signals → 0 触发. W20.3 后 cron 跑会开始累积, 但第一次真 evolve 触发可能要 4-6 周
2. **W4 LangGraph 残留**: `tests/test_w4_*.py` skip 着 ("W13.1: agent/graph.py removed"), 没清掉
3. **多 SKILL 同时 evolve 怎么办**: `run_gray_release` 一次只动 1 个, N 个 SKILL 都需要 evolve 时 N 周才能轮一遍 — 这是设计约束 (debuggability > throughput), 不是 bug
