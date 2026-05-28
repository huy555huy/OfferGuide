# Goal: make OfferGuide a real useful agent (2026-05-16)

## User Ask

用户目标: “优化这个项目，变成一个真实好用的agent，记录所有思路与过程”。

本记录只写可复盘的决策、依据、过程和验证，不记录不可审计的隐藏推理链。

## Working Definition

OfferGuide 不是一个功能按钮集合，也不是替用户乱投的 auto-applier。真实好用的 agent
应该持续维护自己的世界状态、目标状态、证据状态、未决条件和下一次可继续的动作条件。

成功标准:

1. 目标明确: 每次 wake 都能看到 active goals、截止日期、目标指标和当前 funnel。
2. 证据优先: 找岗位、评分、简历建议、面试准备都区分事实、推断、未知。
3. 主动但不烦: 主动做找岗位、投后准备、deadline/followup 提醒；最终投不投、改不改由用户决定。
4. 状态可继续: 每次 wake 后, agent 知道哪些事实变了、哪些未知仍在、哪个 work item 该继续、等待或停止。
5. 能持续: 有 worldview、harness_runs、harness_events、scheduled_wakes、PROGRESS.md 和测试结果可接续。
6. 能学习: 用户 thumbs、application outcome、follow-through 进入 evolution signals，而不是 LLM 自评。
7. 有自己的议程: 每次 wake 都能看到开放回路、阻塞、机会和安静等待项，并基于它决定下一步。
8. 主动发现优先: 用户手动输入 JD 是 fallback 和监督信号, 说明 agent 没先找到或没找全;
   处理该 JD 的同时必须反查 discovery 覆盖缺口。

非目标:

- 不自动投递，不绕过用户确认。
- 不编简历经历、指标、学历、公司名或项目成果。
- 不把“未记录新状态”解释成“失败/挂了”。
- 不为了显得智能而频繁通知用户。
- 不把“粘 JD”包装成主路径成功; 用户自己找到机会通常意味着 agent 应该改进搜索覆盖。

## Current Diagnosis

现有基础已经有 agent 的骨架:

- `src/offerguide/agent_runtime/loop.py`: single master loop + tool calls + telemetry。
- `.offerguide/worldview/`: markdown memory，跨 wake 保存用户、策略、复盘。
- `src/offerguide/goals.py`: active goals、progress、agent self-observations。
- `src/offerguide/agent_runtime/tools.py`: job discovery、score、tailor、interview、project vault、schedule wake。
- `PROGRESS.md`: 长任务 handoff。

主要缺口:

- active goals/progress 存在，但主 agent 的 system context 之前只稳定注入 `MEMORY.md`，目标不一定进入每次 wake。
- `instructions.md` 虽然写了“没有固定流程”，但旧的 chat-first 段落仍用
  `fetch_jd -> score_match -> tailor_advice` 这类例子暗示固定工具链，容易退回 workflow automation。
- worldview 没有一份每次自动注入的 responsibility ledger；agent 可能知道记忆，但不知道“我还欠用户什么”。
- `PROGRESS.md` 滞后，仍写着 full suite 有 9 个 legacy failure；上一轮实际已验证为 `858 passed, 2 skipped`。
- 项目级 north star 没有正式文档化，后续优化容易变成 UI 或功能点堆砌。

## Slice 1 Implemented

本次第一块不做大重构，先把 north star 接到 agent 的主循环:

- `ContextManager` 新增可选 `store`。
- `build_initial_system()` 自动注入 “活跃目标与进度快照”。
- 快照包含 active goal、target date、target metric、funnel 事实、启发式评估，以及 agent self-observations。
- `loop.run()` 把 `deps.store` 传给 `ContextManager`。
- 新增测试覆盖 active goal progress 出现在 system prompt 中。

为什么先做这个:

- 真实 agent 首先要知道自己在长期推进什么。
- 这是低风险切片，不改变工具执行策略，不替模型硬编码决策。
- 后续 UI、推荐、tailor、schedule 都可以围绕同一个 active goal snapshot 演进。

## Slice 2 Implemented

用户指出“一个固定链路不是 agent”，因此第二块修的不是页面，也不是继续加工具，而是把 agent 的
自主决策责任接进运行时:

- `instructions.md` 删除 workflow-ish 的 “Chat-first loop” 形态，改成每次 wake 的决策契约:
  Observe -> Agenda -> Decide -> Act -> Verify -> Sleep。
- 明确每次只能选择一个主动作方向: `act / ask / notify / sleep`，由证据和 agenda 决定。
- 将投后准备包从“投了就固定三步”改成一个开放回路: agent 先判断岗位价值、用户意愿、证据缺口和是否已有 schedule。
- 新增 `agenda.md` 作为 worldview 的责任账本，记录开放回路、阻塞、机会、安静等待。
- `ContextManager.build_initial_system()` 每次 wake 自动注入 `agenda.md`，缺失时提醒 agent 创建；这让议程进入系统 prompt，而不是停留在文档概念。
- 当前项目的 `.offerguide/worldview/agenda.md` 已补齐真实开放回路，供后续 wake 接续。
- 新增 regression tests 覆盖 bootstrap 创建 `agenda.md`、system prompt 注入 agenda、instructions 包含决策契约。

依据:

- Anthropic 对 agents/workflows 的区分: workflows 是预定义代码路径；agents 是模型动态指导自己的流程和工具使用。
- OpenAI Agents 文档里的工程组成可作为约束面: instructions、tools、guardrails、handoffs、tracing；本项目当前先补 instructions + persistent agenda + tests。

为什么这块比继续做 “JD 一句话固定链路” 更优先:

- 用户明确指出固定链路本身是错的。
- 真实 agent 需要先拥有开放回路和判断权，再谈某个 JD 场景如何执行。
- 这仍是小切片: 不重写 loop、不新增数据库 schema、不碰 UI，只让现有 agent runtime 每次醒来带着责任账本工作。

## Slice 3 Implemented

用户指出上一版仍然是形式主义: `record_decision` 只是让 agent 先填一张表, 并没有让它拥有真实工作状态。
因此第三块撤掉 decision guard, 改成真实的 agent work model:

- 新增 `agent_work_items` 表, 表示 agent 真实拥有的开放工作: open / in_progress / blocked / waiting / done / dismissed。
- 用户输入、生命周期事件、scheduled wake 在 LLM 调用前会被 materialize 成 work item。
- `loop.run()` 会把本次 trigger 对应的 work item 绑定到 run, open 项进入 in_progress。
- `ContextManager.build_initial_system()` 每次注入 active work items, agent 醒来看到的是“我正在推进哪些工作”, 不是一段形式化决策。
- `fire_event()` 不只是写 event log, 还会创建对应 work item, 例如 `user_paste_jd` → `Process pasted JD for job#X`。
- 已撤掉 `record_decision` 工具和 loop guard; 运行时不再强迫 agent 填决策表。
- 新增 regression tests 覆盖: user input / event 会创建 work item, run 会接管 work item, context 会注入 work item。

这才更接近真实 agent: workflow 是一次性链路; agent 拥有持续状态、开放工作和推进责任。

## Slice 4 Implemented

用户继续指出: 不能继续“改了什么算什么”, 要先判断这些功能还有没有存在必要。
因此第四块先做功能必要性审计, 不继续堆 UI 或 agent 仪式:

- 新增 `docs/feature_necessity_audit_2026-05-16.md`, 逐项审查 UI 页面、agent tools、SKILL 是否应该成为一等入口。
- 结论: 当前项目不是缺功能, 而是入口过多; 很多能力有用, 但应被 agent 持有, 只作为状态切面或内部能力露出。
- 保留核心入口: Mission Control `/`, Pipeline `/pipeline`, Tailor `/tailor`, Interviews `/interviews`, Agent run detail。
- 合并/降级: `/recommended`, `/jobs`, `/applications`, `/funnel`, `/compare`, `/mock`, `/reflect`, `/stories`, `/project-vault` 不应继续表现为平级主功能。
- 降级内部: `/evolution`, `/metrics`, `/portfolio`, `/debug` 属于 Lab/Debug, 不应影响普通求职主路径。
- 冻结重复入口: `/apply/{id}` 与 `/jobs/{id}/apply-pack` 后续应收敛为一个 canonical apply pack。

这一步的意义: 真实 agent 不是把所有功能铺给用户点, 而是 agent 维护状态、证据和下一次动作条件; 用户只介入边界决定。

## Slice 5 Implemented

用户继续强调“先是一个 agent”。复查后确认第三块 work item 只完成了一半:
runtime 能把触发变成工作项, 也能把工作项交给 run, 但 agent 没有真实机制把工作项关闭、挂起或标阻塞。
这会让系统仍停在“接活了”的样子, 没有可恢复的状态转换。

第五块补的是 agent ownership 的状态出口, 不是 decision log:

- `_schema.update_work_item()` 可以更新单个 `agent_work_items` 的状态、summary、evidence、next_action、due_at 和 last_run_id。
- `done` / `dismissed` 会写 `closed_at` 并从 active work items 中消失; `waiting` / `blocked` 保持 active, 下一次 wake 继续可见。
- 新增 agent tool `update_work_item`, 用于 agent 在实际推进后标记 done / waiting / blocked / dismissed / in_progress。
- tool 要求 `waiting` / `blocked` / `in_progress` 必须写 `next_action`, 避免把未完成工作伪装成状态更新。
- `instructions.md` 的 Verify 阶段明确: 推进了 work item 就要在结束前写回真实状态; 没推进则不要为了“留痕”调用。
- 新增 regression tests 覆盖 tool schema、dispatch 更新、waiting 必须有 next_action、done 不再进入 active context、loop 中 agent 能关闭触发工作项。

这一步的边界:

- 不做自动 post-run heuristic, 不因为 final_text 存在就自动 mark done。否则又会变成假的状态转换。
- 不恢复 `record_decision`; 工作状态直接写到 agent 拥有的 work item, 后续 wake 从这里接续。

## Roadmap

### Phase A: Goal and Process Discipline

- 每次 runtime wake 注入 active goals/progress/self-observations。
- 每次 runtime wake 注入 agenda.md，让 agent 看到自己的开放回路和阻塞。
- 每次 runtime wake 注入 `agent_work_items`, 让 agent 看到真实待推进工作。
- agent 推进工作后必须能把 `agent_work_items` 标成 done / waiting / blocked / dismissed, 并留下 evidence / next_action。
- prompt 使用 Observe -> Agenda -> Decide -> Act -> Verify -> Sleep，而不是固定工具链。
- `PROGRESS.md` 作为人类 handoff，保持当前真值和下一步。
- `.offerguide/worldview/strategy.md` 记录当前产品策略，不只记录求职策略。

验收:

- 测试能证明 system prompt 里有 active goal 和 funnel 事实。
- 测试能证明 system prompt 里有 agenda 内容和 `act / ask / notify / sleep` 决策契约。
- 测试能证明 user input / event / scheduled wake 进入 durable work items。
- 测试能证明 agent 可以显式关闭/挂起 work item, 且 completed item 不再污染 active context。
- `PROGRESS.md` 与最新测试状态一致。

### Phase B: Useful Agent Loop

- 用户一句话目标能被 agent 判断并推进: 先看 agenda 和证据，再决定是否 fetch JD、score、tailor、ask、notify 或 sleep。
- 主 chat 能读取已有状态引用, 不要求用户自己找页面拼上下文。
- 对信息缺口，agent 优先问最少的问题或自己查证。
- 用户粘 JD 时, agent 先服务眼前岗位, 再把它当成 discovery miss 做覆盖复盘。

验收:

- 至少 3 条端到端 dogfood 记录: 主动发现岗位、用户粘 JD 后的 discovery miss 复盘、投后准备。
- 每条都有 job_id/skill_run_id/run_id 或页面链接。

### Phase C: Discovery Quality

- 主动发现岗位必须来自 verified source 或明确标注 external/unverified。
- 排名解释必须引用 candidate evidence、JD evidence、source attribution。
- 不够了解用户偏好时不批量找岗位，先补 candidate.md。

验收:

- 推荐池里每条有 source attribution、why match、risk/gap。
- 低质/不相关岗位有明确过滤原因。

### Phase D: Application Ownership

- 应用状态、followup、面试、结果进入事件流。
- silence check 只说“未记录新状态”，不做失败推断。
- 投后准备包在用户标记投递后可自动生成或提示。

验收:

- 一个真实 application 从 considered -> applied -> followup/prep -> outcome 全链路可追踪。

### Phase E: Learning Loop

- 用户 thumbs / outcome / follow-through 汇聚到 skill fitness。
- Evolution 只在真实信号足够时触发。
- 每次 prompt variant release 有 shadow/canary/live 记录。

验收:

- 至少一个 SKILL 的 variant 通过真实反馈完成 canary 决策。

## Process Log

- 2026-05-16: 接手 Claude 未完成的新 UI/agent 优化任务，先检查 git dirty 状态，保留既有 UI/test 改动。
- 2026-05-16: 读取 `README.md`、`PROGRESS.md`、`.offerguide/worldview/*`、`agent_runtime/*`、`goals.py`。
- 2026-05-16: 判断最小高价值切片是把 active goals/progress 注入 agent system context。
- 2026-05-16: 修改 `ContextManager` 和 `loop.run()`，新增 active goal prompt 测试。
- 2026-05-16: 写入本目标文档，并同步 `PROGRESS.md` / worldview。
- 2026-05-16: 验证 `uv run pytest tests/test_w15_harness.py::TestContextManager -q`，结果 `7 passed`。
- 2026-05-16: 验证 `uv run pytest tests/test_w15_harness.py tests/test_evidence_first_context.py -q`，结果 `142 passed`。
- 2026-05-16: 验证 `uv run pytest -q`，结果 `859 passed, 2 skipped`。
- 2026-05-16: 用户指出新 UI “没接好”、与 `offer.zip` 差距大且有旧版残留；复查后确认 zip 资源已在
  `src/offerguide/ui/static/redesign/*`，问题是路由和模板接线没有收口。
- 2026-05-16: 将旧 top nav 收敛为 redesign 左 rail + topbar；核心 IA 映射为
  Mission Control / Pipeline / Tailor / Interview / Lab / Settings。
- 2026-05-16: 修正 `/funnel`、`/portfolio`、`/apply`、`/recommended`、`/applications`、
  `/jobs/*/apply-pack`、`/jobs/*/post-apply-pack` 等页面的 active rail、breadcrumb 和可见命名。
- 2026-05-16: 清理核心路径旧版残留，包括 `Workspace` 兜底、旧 “待点列表” 命名、emoji-heavy 操作标签、
  以及 user-facing agent run 详情里的 `critic` 表述。
- 2026-05-16: 用新进程 `http://127.0.0.1:8770/` 浏览器验收 `/`、`/pipeline`、`/tailor`、
  `/recommended`、`/portfolio`、`/interviews`、`/evolution`、`/applications`，确认 active rail 与 breadcrumb
  正确，旧顶部导航不再出现。
- 2026-05-16: 验证 focused UI 集合 `136 passed`，再验证 `uv run pytest -q`，结果
  `859 passed, 2 skipped`。
- 2026-05-16: 用户指出固定“一句话链路”不是 agent；重新学习并对齐 agents vs workflows 的区分。
- 2026-05-16: 将 `instructions.md` 改为 Observe -> Agenda -> Decide -> Act -> Verify -> Sleep 决策契约。
- 2026-05-16: 新增 worldview `agenda.md` bootstrap 和当前项目真实 agenda，并在 runtime system context 自动注入。
- 2026-05-16: 新增 tests 覆盖 agenda bootstrap、agenda system prompt 注入、决策契约 prompt。
- 2026-05-16: 验证 focused agenda tests `4 passed`。
- 2026-05-16: 验证 runtime/context/evidence 集合
  `uv run pytest tests/test_w15_harness.py::TestMemoryTool tests/test_w15_harness.py::TestContextManager tests/test_evidence_first_context.py -q`，
  结果 `29 passed`。
- 2026-05-16: 用户指出 `record_decision` 路线仍是形式主义, 要重新构造; 确认应撤掉“填决策表”式 guard。
- 2026-05-16: 新增 `agent_work_items` 表和 `_schema.prepare_trigger_work_items()`, 将 user input / event / scheduled wake 变成 durable work items。
- 2026-05-16: `loop.run()` 在模型调用前创建 work items, 并将本 run 接管的项标记为 in_progress。
- 2026-05-16: `ContextManager` 每次注入 Agent Work Items, 让 agent 面对真实开放工作。
- 2026-05-16: 撤掉 `record_decision` 工具、decision guard 和 `/agent/runs` 上的 agent_decision 展示。
- 2026-05-16: 验证 focused runtime/work-item 集合
  `uv run pytest tests/test_w15_harness.py::TestToolDispatch::test_memory_tool_schema_first tests/test_w15_harness.py::TestHarnessLoop tests/test_w15_harness.py::TestTriggers tests/test_w15_harness.py::TestContextManager -q`，
  结果 `21 passed`。
- 2026-05-16: 用户指出不能继续盲改, 必须先审查现有功能是否还有必要; 开始 feature necessity audit。
- 2026-05-16: 盘点 `web.py` 路由、templates、agent tools、skills, 将功能按 keep / merge / demote / freeze 分类。
- 2026-05-16: 新增 `docs/feature_necessity_audit_2026-05-16.md`, 明确下一步应先做 IA 收敛, 而不是再新增 agent 行为。
- 2026-05-16: 做最小 IA 收敛切片: 将首页 “新进推荐池” 改为 “新进候选”, Pipeline 中 `/recommended` / `/applications`
  表达为 “候选视图” / “应用时间线”, `/funnel` 表达为 Pipeline 转化概览。
- 2026-05-16: 投递包和投后备战包的返回入口统一指向 Pipeline; 投后备战包不再直接把 `/mock` 作为一等 CTA,
  改为进入 Interview 工作台。
- 2026-05-16: 验证 IA 收敛聚焦测试
  `uv run pytest tests/test_w15_harness.py::TestHomeWithW15 tests/test_w15_harness.py::TestReviewFixes::test_recommended_page_renders_empty tests/test_w15_harness.py::TestReviewFixes::test_navbar_includes_recommended_link tests/test_w15_harness.py::TestReviewFixes::test_recommended_card_links_to_apply_pack tests/test_w8_applications_page.py tests/test_w13_8_extension_funnel_portfolio.py::TestFunnelView tests/test_w17_recruit_type.py::TestRecommendedFilter tests/test_w18_match_keywords.py::TestRecommendedShowsW18 -q`,
  结果 `32 passed`。
- 2026-05-16: 用户再次强调“先是一个 agent”; 复查确认当前 work item 模型缺少关闭/挂起/阻塞出口。
- 2026-05-16: 新增 `_schema.update_work_item()` 和 agent tool `update_work_item`, 让 agent 实际推进后能把 durable work item 标成 done / waiting / blocked / dismissed。
- 2026-05-16: 更新 `instructions.md` Verify 契约: 推进了 work item 必须写回真实状态; 没推进不得为了形式调用。
- 2026-05-16: 新增测试覆盖 tool schema/dispatch、waiting next_action 校验、done 从 active context 移除、loop 内关闭 trigger work item。
- 2026-05-16: 验证 focused runtime tests
  `uv run pytest tests/test_w15_harness.py::TestToolDispatch tests/test_w15_harness.py::TestHarnessLoop tests/test_w15_harness.py::TestTriggers tests/test_w15_harness.py::TestContextManager -q`,
  结果 `38 passed`。
- 2026-05-16: 验证 full suite `uv run pytest -q`, 结果 `866 passed, 2 skipped`。

## Next Verifiable Slice

建议下一块先做真实端到端 dogfood, 再继续 IA 收敛:

1. 用一个真实 JD 样例验证: user_input work item -> agent 最小工具行动 -> state/event reference -> discovery miss 复盘 -> `update_work_item(done/waiting/blocked)` -> 下一次 context 只保留仍需继续的项。
2. 继续收起剩余一等入口: `/jobs`, `/compare`, `/mock`, `/reflect`, `/stories` 从主路径降级或改为 Pipeline / Tailor / Interview 内的动作。
3. Tailor 把 Project Vault 明确为“项目事实档案”, 只服务简历和面试上下文。
4. Interview 把 mock / reflect 绑定到 company/application, 不再作为泛用玩具入口。
