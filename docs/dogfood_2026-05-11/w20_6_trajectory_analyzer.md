# W20.6 — Deterministic Trajectory Analyzer (agent 真学)

> 用户原话: "我现在说一个 ai agent 的工程师，那么对我来说最紧要的是 agent，
> 你知道不。我们的项目不同于别人的也是 agent，所以得正确且处理好 agent，
> 才是最关键，也最能拿得出手的"

W20.4 杀了 critic LLM (LLM 评 LLM = epistemo 弱). W20.5 wire 了真信号
(follow_through / app_outcome multi-SKILL). 但**agent 自身的学习闭环还
是空的**:

- `agent_self_observations` 表存在 ✅
- `meta_reflect` tool 暴露给 agent ✅
- snapshot_state 读这表喂下次 wake ✅
- **agent 几乎不主动调 meta_reflect** ❌ ← 学习闭环空着

W20.6 真改: **Python 在每次 non-trivial agent run 结束时, 自动 detect
event 流里的 bad pattern, 写 self-observation**. 下次 wake snapshot 看到
真 lesson, 行为闭环.

## 为什么这**不是**另一个 critic

| 维度 | 旧 critic LLM (W20.4 killed) | W20.6 trajectory analyzer |
|---|---|---|
| 实现 | 第 2 个 LLM call | **Python 函数** |
| 输出 | overall score 0..1 | 文字描述 (事实, 非评分) |
| 信号源 | LLM 判断 | event 流 (真发生的事) |
| 用途 | 喂 evolution_signals (评分 → evolve SKILL) | 喂 agent_self_observations (agent 自己读) |
| Cost | $0.001/run, 5-10s | 0, 微秒 |
| Bias | self-confirmation (同 model 判同 model) | 0 (deterministic) |

**核心区别**: critic 是**外部评价者** (我评你好不好), trajectory analyzer
是**内部审计员** (我看你这次发生了啥). 类比: critic 像老板年终打分, analyzer
像你自己复盘.

不违反 CLAUDE.md 规则 D (LLM 不能自评): **0 LLM call**, pure deterministic.

## 真 4 个 detector

每个都看 event 流的**真**特征. 不是猜的, 是看 `kind=='tool_call'` 等
真 emit.

### D1: max_iter overshoot (PatternKind=overreach)

**触发**: events 含 `kind=error, message contains "max_iter"` 且**无** `kind=final`.

**含义**: agent 跑满了所有 iter 还没自然 final, 卡了/堆 tool 堆死.

**写入 observation**: "上次跑到了 max_iter (N) 才停, 自然 final 没触发.
下次 goal 类似时早点判断收手, 别堆 tool call."

### D2: circuit-breaker 模式 (PatternKind=repeated_mistake)

**触发**: 同 (tool_name, args_signature) 调用 ≥3 次. signature 是 args 的
sorted JSON, identical args → identical sig.

**含义**: agent 反复调同一个 tool 同一组 args, 期待结果会变. 不会变 (W13.7
的 CIRCUIT_BREAKER_REPEAT_THRESHOLD=3 在 production 已经 abort 了, 但 self
note 让 agent 下次别开始这条路).

**写入 observation**: "上次反复调 {tool} 同一组 args ({n} 次). 如果第一次
结果没用, 检查 args 是否对而不是再调一次."

### D3: 同 tool 重复 ERROR (PatternKind=repeated_mistake)

**触发**: 同 tool_name 的 `tool_result.result_preview` 以 "ERROR" 开头出现
≥2 次.

**含义**: tool 在当前状态下不可用 (e.g. SKILL 输入不对, 表不存在). agent
应该先 verify 前提.

**写入 observation**: "上次 {tool} 调用 {n} 次都返回 ERROR. 调它前先验证
input 假设 (e.g. job_id 存在 / 表已初始化)."

### D4: zero action on action-keyword goal (PatternKind=underreach)

**触发**: 有 final, 0 个 tool_call, 且 goal 含 action 关键词 (评估/投递/
面试/推荐/分析/score/tailor/evaluate/...).

**含义**: agent 看到一个需要做事的 goal, 但**直接 final 没调任何工具**.
没看数据就回答 = 空判断.

**写入 observation**: "上次 goal 含 action 词 (e.g. 评估/找/分析) 但
0 tool_call 直接 final. 下次先 read 数据再回答, 别空判断."

**注意**: goal 不含 action 词 (e.g. "今日例行检查") → 不触发. 维护类
goal 真可能 0 action 是对的 ("没事可做就别瞎动"). 之前测试真发现 "查"
1 char 太松, 砍了, 只留 ≥2 char specific keyword.

## 真 dedup + TTL

不能让同样的 lesson 反复写, snapshot 会被同一 lesson 淹没.

- **Dedup**: 写之前查 `agent_self_observations WHERE observation=? AND
  pattern_kind=? AND superseded_by IS NULL AND (valid_until IS NULL OR
  valid_until > now)`. 已存在跳过, 不重复写.
- **TTL**: 写入时设 `valid_for_days=14`. 14 天后自动过期 (agent 学完不
  让它纠缠永远).

## 真 gating

trivial run (`iterations ≤ 1 AND skill_invocations empty`) **不触发**
analyzer. 没东西可分析.

非 trivial → 跑. **不分 trigger_kind** (跟 critic 不一样): pure Python,
无 LLM cost, 跑遍 cron / user_button / sse 都行. 真 zero overhead.

## 真闭环

```
agent run N
   ↓
[iterations 跑 tool / SKILL ...]
   ↓
[run 结束]
   ↓
_analyze_and_persist_observations  ← W20.6 新加
   ↓
agent_self_observations 表 +1 行 (e.g. "上次反复调 X")
   ↓
... 时间过去, agent run N+1 ...
   ↓
snapshot_state(store) 读 agent_self_observations
   ↓
agent LLM 看到 "📓 Agent 自我观察" 段, 含上次的 lesson
   ↓
agent 这次行为基于真历史 lesson 调整 (希望)
```

这是 agent 真**跨 run 学**的, 不是评分 → evolve 那条慢链路 (后者需要
≥10 signal + fitness <0.55 + 7天 cooldown, 通常 4-6 周才触发一次).

## 真测试 (`tests/test_w20_6_trajectory_analyzer.py`, 15 case 全 PASS)

- D1 max_iter: 2 case (触发 / 不触发)
- D2 repeated args: 3 case (3x 触发 / 不同 args 不触发 / 2x 不触发)
- D3 repeated error: 2 case (2x 触发 / 1 错不触发)
- D4 zero action: 3 case (action goal 触发 / 维护 goal 不触发 / 有 tool 不触发)
- Dedup + 持久化: 3 case (写入 / 同文 dedup / 不同文都写)
- 集成 (run loop 真 wire): 2 case (trivial skip / 非 trivial 跑)

全套 1070 pass / 13 skip / 0 regression.

## 为什么这是 "拿得出手" 的真 agent 工程

不是 "我又加了个 feature". 是:

1. **明确的 agent 学习 architecture**: trajectory event 流 → 决定性 pattern
   分析 → 跨 run 持久化 lesson → 下次 wake 真读
2. **明确避开 LLM-as-judge 反模式**: 不让 LLM 评 LLM 装高级, 用 Python
   做 pattern detect — 这是 epistemo 清晰
3. **真 dedup + TTL**: 不让 self-observations 失控膨胀
4. **真测试覆盖**: 每 detector 测了正反两面 (触发 + 不触发), 不是"应该 work"
5. **真闭环验证**: snapshot_state 已有读 self_observations 段, 不需要新搭

可以跟领导讲:

> 我们的 agent 不只是个 ReAct loop. 它有真**跨 run 记忆 + 学习闭环**:
> Python 在每次 run 结束自动分析 event 流, 检测 4 类 bad pattern (max_iter
> 卡死 / circuit_breaker 模式 / 重复 tool 错 / 该 act 不 act), 写 fact-based
> self-observation 进 DB. 下次 wake 的 snapshot 自动带这些 lesson, agent
> 真 behavioral adjustment.
>
> 关键是: 我们**没用 LLM-judge-LLM 假高级**. 评分驱动的 critic 我前一阵
> 杀掉了, 因为 same-model judging same-model 是 self-confirmation bias.
> 这版是 deterministic pattern detection, **0 LLM call, 微秒 overhead,
> 跨 run 真闭环**.

## 还没做的 (诚实)

1. **更多 detector**: 当前 4 个. 真用一阵后会发现新 pattern (e.g. "agent
   总是用 tool A 但其实 tool B 更直接") — W20.7 候选, 按真数据加
2. **PatternKind 'success_pattern' 没用**: 我故意没写 success detector
   (snapshot 不应该自夸). 但如果发现 "用户对某 trajectory 真满意" 信号
   可以补
3. **没真用 agent_runs 跑过端到端 e2e**: prod store agent_runs 还是 0 行.
   真要等用户用 /agent 才能验 W20.6 在真 trajectory 上的 false-positive
   rate
