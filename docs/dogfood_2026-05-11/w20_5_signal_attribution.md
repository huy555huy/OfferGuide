# W20.5 — 修真 SKILL→job→outcome attribution

W20.4 杀了 critic, 真信号靠用户行动. 但**真 wiring 有结构 bug**:
- score_match 写 `harness_event` 链 skill_run_id ↔ job_id (W15.22 加的)
- apply_assistant / prepare_interview **没写过这种 link**
- 没 link → app_outcome 只能 hardcode attribute 给 apply_assistant 一个 SKILL
- 没 link → /apply-pack 看了 N 次都不能写 follow_through, 因为找不到 skill_run_id

W20.5 修了这个结构 bug.

## 改的 4 件事

### 1. SKILL→job link helper (`_link_skill_to_job_and_signal`)

每次 view 调 SKILL 后 emit 1 个 `harness_event` 把 skill_run_id 跟 job_id 链
起来 + record 1 个 `follow_through` signal.

```python
def _link_skill_to_job_and_signal(*, skill_name, skill_run_id, job_id, signal_kind):
    # 1. INSERT harness_event(kind=signal_kind, job_id, note={skill_run_id, name, version})
    # 2. record_follow_through(skill_name, skill_version, executed=True)
```

### 2. 用到 2 个 view

- `/jobs/N/apply-pack` → emit `apply_pack_generated` event + follow_through for `apply_assistant`
- `/jobs/N/post-apply-pack` → emit `post_apply_pack_generated` event + follow_through for `prepare_interview`

每次用户打开这俩页面 = 在用 SKILL 的输出 = 真 follow_through (用户花时间
评估了 + 决定继续).

### 3. `/api/apply/N/mark` 改成多 SKILL outcome 分

之前: hardcoded `record_app_outcome(skill_name='apply_assistant', skill_version='0.1.0', ...)`. 1 个 SKILL 拿 credit.

W20.5: 查这个 job 的所有 harness_events (`scored` / `apply_pack_generated` /
`post_apply_pack_generated`), 找出所有真参与的 skill_runs, 每个 SKILL 各
record 1 个 `app_outcome` signal. 现在:

- `score_match` 拿到信号 (recommended this job)
- `apply_assistant` 拿到信号 (wrote materials)
- `prepare_interview` 拿到信号 (if user opened post-apply-pack)
- `tailor_resume` 拿到信号 (if user opened tailor view — 待 wire)

**每条 signal weight=1.0** (直接 attribution). Fallback (job 没真 SKILL chain):
单条给 apply_assistant, weight=0.5 (低权重表示 attribution 模糊).

### 4. 修了 1 个 silent bug

**真 bug**: `harness_events` 表不被 `Store.init_schema()` 创建, 只 `harness._schema.init_harness_schema()` 创建. 我新代码假设它在 → SQL 报错 → 我外层 try/except 静默吞了 → 老 W13.4 测试 (没 init harness_events) 信号写不进 → 测试失败.

**真 fix**: apply_mark handler 里调 `init_harness_schema(store)` (idempotent), 保证表存在.

CLAUDE.md 规则 C 又一次出现 (我假设表存在, 真不在). 这次没用 grep 假设, 而是
真假设. 这条规则该补一句: "新写的 SQL 之前先确认表 schema 真存在, 不能依赖
caller 已 init".

## 真测试 (`tests/test_w20_5_signal_attribution.py`, 4 case 全 PASS)

1. `test_apply_pack_view_writes_link_and_signal_when_skill_runs`: 没 LLM key
   时 SKILL 不 invoke, 没 link/signal 写 (defensive 不 crash)
2. `test_apply_pack_view_writes_signal_when_skill_succeeds`: 真 stub SKILL
   返 fake skill_run_id, GET /apply-pack → 真验 link harness_event 写了 +
   follow_through signal 写了
3. `test_apply_mark_status_interview_fans_to_all_involved_skills`: setup 1 个
   job + 2 个 SKILL events (score_match + apply_assistant), POST /mark
   status=interview, 真验 **2 SKILL 都拿到 app_outcome signal** (之前只 1 个)
4. `test_apply_mark_with_no_skill_chain_falls_back`: 用户 paste 进的 job 没
   SKILL chain, fallback 给 apply_assistant weight=0.5

加上 W20.4 的 4 follow_through tests + 18 critic gating tests = **26 case**.

全套 1055 pass / 13 skip / 0 regression.

## 真信号现在的样子

用户走完一个 application 流程, evolution_signals 表会进多少行?

| 用户动作 | 信号 (真写) | weight |
|---|---|---|
| 看 /recommended | (无 — 浏览不算) | - |
| 打开 /jobs/N/apply-pack | follow_through for apply_assistant | 0.8 |
| 点 "我投了" | follow_through for score_match | 0.8 |
| 后端跳 /post-apply-pack | follow_through for prepare_interview | 0.8 |
| 7 天后, status=replied | app_outcome × N (score_match + apply_assistant + prepare_interview), value=1.0 each | 1.0 each |
| 后续 status=interview | 同上, +1 round | 1.0 each |
| 拿到 offer (status=offer) | 同上, +1 round | 1.0 each |

**完整一个 successful application 流程 ≈ 12-15 真行动 signal** 进 evolve 系统.
你在 5-9 月暑期实习窗口投 30+ 个, evolve 数据池就 360+ 真信号. 触发条件
(≥10 signal + fitness < 0.55) 1-2 个月就能命中.

## 还没做的 (诚实)

1. **tailor_resume 的 view 还没 wire** — 没 wire 是因为 tailor_resume 现在
   是嵌在 /apply-pack 里的子 component, 没单独 view. W20.6 候选.
2. **7-day no-action negative follow_through** (我之前说要做的): 还没做.
   要写一个 daily-cron 任务扫 inbox_items WHERE created_at < now-7d AND
   status='pending' → record_follow_through(executed=False). 留 W20.6.
3. **eval_synthetic** 还是 0 wired. offline batch eval, 单独 task.

## 真节省 + 真效果

- 现在每个 application 真累积 12-15 个真信号 (W20.5 之前只 1 个 hardcoded
  app_outcome)
- 信号纯来自用户真行动 (浏览 / 投递 / outcome update), 0 LLM 自评
- 跟 evolve 的真链路通了: 真行动 → 真 signal → fitness → 触发 evolve

## 用户该做的 (P1, 你昨天答应了的)

跑 server, 真用一周:
1. `python -m offerguide.ui.web` 启 daemon
2. `/recommended` 看真匹配岗 (W20.2 真 dogfood ≥30% 的 13 个, 5 个 ≥60%)
3. **真挑 3-5 个真打算投的**, 每个都点 /apply-pack → copy 自我介绍 + QA
4. **真投后真点 "我投了"** → score_match 拿 follow_through
5. 7 天后真 status update (替换 silent / replied / interview / rejected) →
   3 个 SKILL 都拿 app_outcome
6. 最后 1 周再来一次, 我看 evolution_signals 表数据, 跟你聊真该 evolve 哪个
   SKILL prompt
