# W20.4 — Critic LLM 默认关掉, 改用真用户行动信号

**用户原话** (W20.3 之后):

> 烧钱什么的还好，不太在乎，因为 deepseek 本身就便宜，但是你不能耽误我的时间。
> 不能使用一个纯系统代码的 critic 来掩饰，这是没有意义的，因为他凭什么能评判你？

我之前 W20.3 加了 4 层 gating + timeout, 但本质上还在用 LLM 评 LLM. 用户
直接戳穿: **这事不是优化的问题, 是合法性的问题**.

## 1. 真问题: critic LLM 凭什么能评判 agent LLM?

不能. 而且代码里**原作者自己就承认了** (`evolution/signals.py:46-58`):

```python
# critic 是 mid 因为 it's a model judging another model (good but biased);
DEFAULT_WEIGHTS: dict[SignalKind, float] = {
    "critic":         1.0,   ← LLM 评 LLM, 中位 weight, 自承偏见
    "user_thumbs":    2.0,   ← 真用户点 👍/👎, 最高 weight
    "app_outcome":    1.5,   ← 真投了 → 收回复 / 面试 / offer
    "follow_through": 0.8,   ← 用户真执行 agent 建议没?
    "eval_synthetic": 0.6,   ← 离线 eval, 最低
}
```

5 种 signal, **3 种是真用户行动 (user_thumbs / app_outcome / follow_through),
sum weight 4.3**. critic LLM weight 1.0, 不到 1/4. 设计意图本来就是真行动
为主, critic 是补充.

但实际 wiring 状态:

| Signal | 真 wired? | 真有 row? |
|---|---|---|
| `critic` | ✅ (agent/loop.py 自动写) | 0 (agent_runs 0 行 → 0 critic 写) |
| `user_thumbs` | ✅ (inbox.py 用户点 👍/👎) | 待 verify |
| `app_outcome` | ✅ (web.py 用户改 status 时) | 待 verify |
| `follow_through` | **❌ 没 wired** (W20.4 之前) | 0 |
| `eval_synthetic` | ❌ 没 wired | 0 |

**最讽刺**: 设计了真信号优先, 实操 wiring 反而漏了 follow_through; 而最弱的
critic 倒被默认开启 + 跑得飞起 (一旦用户用 /agent).

## 2. 真改动 (W20.4)

### 2.1 critic 默认 OFF

`AgentLoop.__init__(critic_enabled=False)` (之前 True).

env opt-in 留着: `OFFERGUIDE_CRITIC_ENABLED=1` → 强制开. 想烧钱 + 想拿 critic
signal 的人可以开. 默认不开.

W20.3 的 gating 层 (trigger_kind / trivial / timeout / sample_rate) 全留着 —
不是删, 是降级到"如果你 opt-in critic, 这些 gating 还会保护你不被坑". 但
**默认路径不再走 critic**, 所以 gating 也不需要了.

### 2.2 真信号 wire: "我投了" → follow_through

之前 `/api/jobs/N/applied` (用户点"我投了") 只:
- INSERT applications row
- 排 7 天 wake
- INSERT harness_event 'user_marked_applied'

**漏的**: 没写 evolution_signal. 即用户真采纳了 score_match 的推荐, 这个事实
没回流到 SKILL 评估系统.

W20.4 加了 (web.py 真改的代码):

```python
# 找 score_match 推荐这个 job 的那次 skill_run_id
row = conn.execute(
    "SELECT json_extract(note, '$.skill_run_id') as srid "
    "FROM harness_events WHERE kind='scored' AND job_id=? "
    "ORDER BY id DESC LIMIT 1", (job_id,)
).fetchone()
if row and row[0]:
    srid = int(row[0])
    sr = conn.execute(
        "SELECT skill_version FROM skill_runs WHERE id=?", (srid,)
    ).fetchone()
    if sr and sr[0]:
        _evo.record_follow_through(
            store, skill_name="score_match", skill_version=str(sr[0]),
            skill_run_id=srid, executed=True,
            notes=f"user clicked '我投了' on job#{job_id}",
        )
```

**为什么这是真信号**:
- 用户真**评估**了 score_match 给的概率
- 用户真**决定**这岗值得投
- 用户真**点了** "我投了"
- 整个链路是真行动, 不是 LLM 自评

每次"我投了" → 1 条 `evolution_signals(kind='follow_through', value=1.0,
weight=0.8)` for the responsible score_match SKILL run.

### 2.3 没 wire 的 (诚实清单)

- **eval_synthetic** 还是 0 wired. 这本来就是 offline eval, 等用户跑 eval
  flow 再说
- **app_outcome → 多 SKILL 归属**: 现在只 attribute 到 apply_assistant
  (web.py:2769 hardcoded). 一个 application 经过 score_match → tailor_resume
  → apply_assistant → prepare_interview, outcome 应该按 weight 分给所有
  涉及的 SKILL. 这是 follow-up

## 3. 信号层级 — 真该怎么思考

| 层 | 信号 | 真权威 | 真延迟 | 真 wire |
|---|---|---|---|---|
| **Ground Truth** | offer 拿了 | 100% — 没法 fake | 周/月 | app_outcome via status='offer' |
| **强行动** | 收到 HR 回复 / 面试 | 90% — 客观事件 | 天/周 | app_outcome via status='replied'/'interview' |
| **行动信号** | 用户点 "我投了" / 👍 | 80% — 用户花时间评估了 | 即时 | follow_through (W20.4 加) + user_thumbs |
| **意见信号** | critic LLM 评分 | **30% — LLM 自己评 LLM** | 即时 (但贵) | critic (W20.4 默认关) |
| **离线信号** | offline eval 跑 | 50% — 跟分布 drift 有关 | hours | eval_synthetic (没 wire) |

W20.4 后的真等级:
1. **首要**: 用户真行动 (follow_through ✅ 已 wire / user_thumbs ✅ 已 wire / app_outcome ✅ 部分 wire)
2. **次要**: critic LLM (默认关; 想要可 env opt-in)
3. **未来**: eval_synthetic (offline)

## 4. 真测试 (`tests/test_w20_4_follow_through_signal.py`, 4 case 全 PASS)

- `test_marking_applied_writes_follow_through_signal`: TestClient 真 POST
  `/api/jobs/N/applied`, 真验 evolution_signals 表写入了 1 行 (skill=score_match,
  value=1.0, weight=0.8, notes 含 "我投了")
- `test_marking_applied_without_prior_score_match_doesnt_crash`: 没 score_match
  的 job 也能标已投, 只是不写 signal
- `test_marking_applied_emits_user_marked_applied_event`: 不 regression 老
  harness_event
- `test_signal_is_real_user_action_not_llm_judgment`: assert 真行动总 weight
  > 4 × critic weight (设计意图代码里就这么写的)

加上 W20.3 的 gating tests 重测 (含 default OFF + env opt-in):
- `test_critic_default_OFF_for_user_concerns`: bare AgentLoop 默认 critic_enabled=False
- `test_env_force_on_overrides_default_off`: OFFERGUIDE_CRITIC_ENABLED=1 强制开

**24 case 全 PASS**, 全套 1051 pass / 13 skip / 0 regression.

## 5. 实测节省

| 场景 | W20.3 (gating) | W20.4 (默认关) |
|---|---|---|
| 用户用 /agent 一次 | gating 跳过 | 不跑, 0 LLM call |
| cron 后台跑 | 跑 critic, 5-10s + $0.001 | **不跑**, 0 LLM call |
| 想要 evolve signal | 等 critic | 用 follow_through (用户真投了即记) |

cost 不是关键 (你说了不在乎), **关键是没浪费时间在没意义的 LLM 自评上**.

## 6. 对你 evolve 系统的真影响

**短期** (1-4 周): 
- critic signal 不再来 → fitness.compute_fitness 只看 user_thumbs / app_outcome / follow_through
- 用户每"我投了" 1 次 = score_match SKILL +1 follow_through signal
- 每收到 HR 回复 / 面试 = apply_assistant SKILL +1 app_outcome signal
- evolve 触发条件 (≥10 signals + fitness < 0.55 + cooldown 7天) 还是要 ≥10 signal,
  现在用真行动累积更慢但更真

**长期效果**:
- evolve 决策基于真用户 outcome, 不是 "LLM 觉得 agent 干得不错"
- 减少 self-confirmation bias (LLM 自评常给自家 trajectory 高分)
- SKILL 进化方向更对齐用户真需要

## 7. CLAUDE.md 应该加一条

CLAUDE.md 第 0 节 "最高优先级铁律" 加一条 D:

> **D. LLM 不能自评.** 凡是"用一个 LLM 评判另一个 LLM 输出"的设计 (含同一
> model 跑两次扮演不同角色), 默认假设它是弱信号. 真信号必须来自用户行动 /
> 真世界 outcome / 用户标注. 想用 LLM-as-judge, 必须有独立 ground truth
> 校准过 (e.g. judge 跟人工标注的 agreement > 0.8).
