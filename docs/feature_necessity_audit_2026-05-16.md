# Feature Necessity Audit: real agent, not agent-shaped UI (2026-05-16)

## 用户校正

用户指出: 现在不能继续“改了什么算什么”, 必须先判断这些功能还有没有存在必要。

本审计只记录可复盘判断: 哪些功能应保留为核心 agent loop, 哪些应合并成产物/能力, 哪些只应留在 Lab/Debug,
哪些应冻结或移出主路径。

## 判断标准

OfferGuide 的目标不是做一个求职工具箱, 而是做一个能持续推进求职目标的 agent。

一个功能要成为一等入口, 必须满足至少一个条件:

1. 让 agent 更好地维护目标、开放工作项、阻塞和下一步。
2. 让用户更容易确认/否决 agent 的高影响行动。
3. 承载求职链路中的真实状态: 岗位、投递、面试、产物、结果。
4. 作为 agent 产出的可审阅结果页, 而不是要求用户手工跑流程。

反过来, 如果一个功能只是“看起来完整”的工具, 但需要用户自己理解何时使用、如何串联、怎么收尾,
它就不应是一等产品入口。它可以保留为后台能力、产物详情页或内部调试页。

参考依据:

- Anthropic 的 [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents?guides=understanding-tradeoffs)
  对 workflows / agents 的区分: workflow 是预定义代码路径; agent 是模型根据上下文动态指导自己的流程和工具使用。
- OpenAI 的 [Agents SDK guide](https://platform.openai.com/docs/guides/agents-sdk/) 和
  [Agent reference](https://openai.github.io/openai-agents-python/ref/agent/) 把 agent 落到 instructions、tools、handoffs、guardrails、tracing。
  对本项目而言, 这些不是 UI 名词,
  而是运行时契约: agent 有目标和工具, 用户有确认边界, 系统有轨迹和可追溯结果。

## 总体诊断

当前最大问题不是功能不够, 而是入口太多。

页面和 SKILL 都在围绕同一条求职链路重复展开:

发现岗位 -> 评分 -> 决定是否投 -> 调简历 -> 投递话术 -> 跟踪状态 -> 面试准备 -> 复盘 -> 学习。

这条链路应该由 agent 拥有, 页面只负责三类事情:

- Mission Control: 看 agent 正在推进什么、等用户决定什么、最近做了什么。
- Work Surfaces: Pipeline / Tailor / Interview 这几个真实工作台。
- Artifact / Debug: 某次投递包、某次 run、某个 metrics 页面用于审阅和排查。

不应该继续让用户在十几个入口之间像操作后台系统一样拼流程。

用户手动粘 JD 应视为兜底和监督信号, 不是主路径。只要用户需要自己输入一个目标岗位,
通常就说明 agent 没有先找到、没有找全, 或没有及时把机会推到用户面前。产品可以保留
粘 JD 入口来服务眼前需求, 但它应反向驱动 discovery 覆盖改进, 不应被包装成核心成功路径。

## UI 页面审计

### 保留为核心入口

| 功能 | 当前路由 | 结论 | 原因 |
| --- | --- | --- | --- |
| Mission Control / agent workspace | `/` | 保留, 作为唯一首页 | 这是 agent 状态、开放事项、用户确认、近期活动的总入口。真正的 agent 体感应该从这里发生。 |
| Pipeline | `/pipeline` | 保留, 但收敛子入口 | 岗位、投递状态、followup 和 funnel 是真实求职状态, agent 必须读写。 |
| Tailor | `/tailor` | 保留, 但定位为产物工作台 | 简历微调是高价值产物, 但不该要求用户先理解多个 SKILL。agent 应能从 JD 自动带用户到这里审稿。 |
| Interviews | `/interviews` | 保留, 但更像资料/准备工作台 | 面经、面试准备和复盘是投递后的真实状态, 但入口应依附于具体 application/company。 |
| Agent run detail | `/agent/runs/{id}` | 保留为追踪页 | 它不是普通用户主流程, 但对“真实 agent”很重要: 轨迹、工具调用、产物和错误必须可追溯。 |

### 合并进核心入口, 不再当一等导航

| 功能 | 当前路由 | 结论 | 合并方式 |
| --- | --- | --- | --- |
| Recommended | `/recommended` | 合并进 Pipeline | 推荐池是 Pipeline 的一个视图, 不是独立产品入口。用户关心的是“哪些值得推进”, 不是“去推荐池管理”。 |
| Jobs tracked | `/jobs` | 合并进 Pipeline | 与 Pipeline / Applications 重复。应作为 Pipeline 的筛选或详情, 不再暴露成主入口。 |
| Applications | `/applications` | 合并进 Pipeline | 投递记录是真状态, 但应和 Kanban / funnel 同处一个工作台。 |
| Funnel | `/funnel` | 合并进 Pipeline 或 Metrics | 数据少时价值低, 适合作为 Pipeline 统计区, 不应让用户单独进入。 |
| Apply pack | `/jobs/{id}/apply-pack`, `/apply/{id}` | 保留为产物详情, 入口由 agent / Pipeline 引导 | 投递包有用, 但不是“用户去跑一个工具”。agent 判断岗位值得投后生成或建议生成, 用户审阅。两个 apply pack 入口目前重复, 后续应收成一个 canonical route。 |
| Post-apply pack | `/jobs/{id}/post-apply-pack` | 保留为投后产物详情 | 只有在已投/进入面试准备时才出现, 不应常驻导航。 |
| Project Vault | `/project-vault` | 合并进 Tailor / Interview | 项目事实档案是防止简历和面试编造的基础能力, 很重要, 但应作为简历/面试上下文编辑器。 |
| Stories | `/stories` | 合并进 Interview 或 Project Vault | STAR 故事是面试资产, 不应独立成页面让用户主动维护一套系统。 |
| Profile briefing | `/profile/{company}` | 保留为公司/岗位 briefing 产物 | 成功者画像和 gap 分析有用, 但应由 agent 在比较/面试准备时按需生成。 |
| Reflect | `/reflect` | 合并进 Interview | 面试复盘应挂在 application/company timeline 下, 不应作为孤立入口。 |
| Mock | `/mock` | 合并进 Interview, 且默认隐藏 | 模拟面试有价值, 但只有在有具体面试场景和 prep 结果时才像 agent 行为。否则像玩具。 |
| Compare | `/compare` | 合并进 Pipeline/agent command | 多 JD 比较是 agent 在“同公司多个岗位/投递限额”场景下的能力, 不应是用户手动开页面跑。 |
| Goals | `/goals` | 已合并到 `/` 是正确方向 | 用户可以看到目标, 但目标维护应成为 Mission Control 的一部分。 |
| Inbox | `/inbox` | 已合并到 `/` 是正确方向 | 用户待确认事项应该直接出现在 Mission Control。 |
| Agent chat page | `/agent` | 已合并到 `/` 是正确方向 | 单独 agent 页会制造“另一个入口”, 主页就应该是 agent。 |

### 降级为 Lab / Debug / 内部

| 功能 | 当前路由 | 结论 | 原因 |
| --- | --- | --- | --- |
| Evolution | `/evolution` | 降级为 Lab | SKILL 进化是内部能力, 不是求职用户主路径。只有开发/演示时需要。 |
| Metrics | `/metrics` | 降级为 Lab/Debug | dogfood metrics 重要, 但用户不是每天看 agent 指标的人。 |
| Portfolio | `/portfolio` | 暂时降级为 Lab | 作为项目展示/简历材料有价值, 但它是“构建 OfferGuide 的证明”, 不是求职 agent 的核心使用入口。 |
| Debug | `/debug` | 保留为 Debug | 必要, 但不应在普通用户 IA 中显眼。 |
| Scheduler trigger / search test APIs | `/api/scheduler/*`, `/api/search/test` | 保留为内部 API | 对排查和自动化有用, 但不是产品能力。 |

### 冻结或准备删除

| 功能 | 当前形态 | 结论 | 原因 |
| --- | --- | --- | --- |
| 两套投递包入口 | `/apply/{id}` 与 `/jobs/{id}/apply-pack` | 冻结旧入口, 后续收敛 | 二者都在生成投递材料, 会让 UI 和测试长期分叉。应该选一个 canonical route。 |
| 独立 `/jobs` 页面 | tracked jobs list | 准备删除或重定向到 Pipeline | 与 Pipeline / Applications 重叠明显。 |
| 独立 `/funnel` 页面 | standalone funnel | 准备删除或重定向到 Pipeline anchor | 统计价值存在, 但页面级入口过重。 |
| 独立 `/compare` 页面 | manual compare form | 冻结页面入口 | 正确形态应是 agent 在需要时调 compare_jobs, 给出结果卡。 |
| 独立 `/mock` 页面 | generic mock interview | 冻结页面入口 | 没有具体 application/context 时像“功能演示”, 不是 agent 推进。 |
| 独立 `/reflect` 页面 | transcript form | 冻结页面入口 | 应从 Interview/application timeline 进入。 |

## Agent 工具审计

### 保留为核心工具

| 工具 | 结论 | 原因 |
| --- | --- | --- |
| memory | 保留 | 长期偏好、事实、边界和用户反馈必须可持续。 |
| search_official_jobs / discover_jobs / fetch_jd | 保留 | 发现机会是 agent 主动性的基础; fetch_jd 是兜底和补全, 用户粘 JD 时还应触发 discovery 覆盖复盘。 |
| score_match | 保留 | agent 需要判断是否值得推进, 不能每个 JD 都生成一堆材料。 |
| tailor_advice | 保留 | 高价值产物, 但应由 agent 在合适时调用。 |
| interview_prep | 保留 | 投后/面试阶段核心能力。 |
| capture_project / save_project_record | 保留 | 防止编造和支撑简历/面试, 是求职 agent 的事实底座。 |
| read_artifact | 保留 | agent 必须能读回自己生成过的产物, 否则无法持续。 |
| record_event | 保留 | 真实求职状态需要事件流。 |
| ask_user / notify_user / schedule_next_wake | 保留 | agent 不是一次性问答, 必须能问、通知、安静等待。 |
| web_search / fetch_url | 保留但加边界 | 用于现查公司政策、JD、面经来源; 必须标注来源和不确定性。 |

### 保留但降级/加触发条件

| 工具 | 结论 | 原因 |
| --- | --- | --- |
| reflect_outcome | 保留, 但只在 outcome/复盘后触发 | 复盘是学习信号, 不能变成每次流程里的形式动作。 |
| detect_evolution_candidates / evolve_skill / run_release_cycle | 保留为内部工具 | 这属于 agent/skill 自学习, 不应干扰用户主流程。触发必须依赖真实信号, 不是 LLM 自评。 |

### 不应再新增的工具类型

- 不要再加“为了证明 agent 做过决策”的记录工具。`record_decision` 已证明是形式主义。
- 不要加固定流程守卫, 例如强制 `fetch_jd -> score -> tailor -> notify`。这会把 agent 拉回 workflow。
- 如果需要状态, 应落到真实对象: work item、application event、artifact、schedule、user question。

## SKILL 审计

### 核心 SKILL

| SKILL | 结论 | 原因 |
| --- | --- | --- |
| score_match | 保留核心 | 决策前置, 避免无差别产出材料。 |
| analyze_gaps | 保留核心, 可能与 tailor_resume 合并呈现 | 找差距是简历微调的依据, 但用户不需要单独跑。 |
| tailor_resume | 保留核心 | 直接产出可用简历版本。 |
| apply_assistant | 保留核心, 并入 apply pack | 复制即用话术和表单 QA 是真实价值。 |
| prepare_interview | 保留核心 | 已投/面试阶段的主要产物。 |
| post_interview_reflection | 保留核心, 但只由 Interview timeline 触发 | 复盘转成学习信号和故事资产。 |

### 支撑型 SKILL

| SKILL | 结论 | 原因 |
| --- | --- | --- |
| deep_project_prep | 保留为 interview/project 支撑 | 很有价值, 但不应作为独立入口。 |
| compare_jobs | 保留为 agent 工具能力 | 用在同公司多 JD 或投递限额场景, 不应要求用户打开 compare 页。 |
| successful_profile | 保留为 briefing 子能力 | 对公司画像有用, 但必须基于真实样本和来源。 |
| profile_resume_gap | 保留为 briefing 子能力 | 和 successful_profile 成对使用, 作为 gap 产物。 |

### 冻结/降级 SKILL

| SKILL | 结论 | 原因 |
| --- | --- | --- |
| mock_interview | 降级 | 只有在有具体面试、已有 prep、用户明确练习时才有价值。不能作为主线功能显摆。 |
| write_cover_letter | 冻结默认入口 | 国内校招/实习主流程中 cover letter 不是高频刚需。保留工具, 但不要让它扩大主路径。 |

## 推荐的信息架构

下一轮 UI 收敛应按这个方向做:

1. `/` Mission Control: agent 当前工作、开放 work items、需要用户决定的事项、最近产物、输入框。
2. `/pipeline`: 岗位池、推荐/跟踪/投递状态、application timeline、轻量 funnel。
3. `/tailor`: 简历产物、JD 关联、项目事实档案入口。
4. `/interviews`: 公司面经、面试准备、mock、复盘, 但以 application/company 为中心。
5. `/lab`: evolution / metrics / portfolio / debug 的内部集合, 或至少不在主 rail 展示多个入口。

主 rail 不应再暗示十几个平级功能。真正的产品心智应是:

> Agent 拥有流程; 用户审阅关键产物和边界决定。

## 立即可做的最小改动

不急着删代码。先做不破坏数据和测试的收口:

1. 从可见导航和首页弱化 `/jobs`, `/applications`, `/funnel`, `/compare`, `/mock`, `/reflect`, `/stories` 的独立入口。
2. Pipeline 页面内吸收推荐池/投递记录/漏斗链接, 但 label 改成同一工作台下的视图。
3. Tailor 页面把 Project Vault 改名为“项目事实档案”, 明确它是防编造上下文, 不是另一个产品。
4. Interview 页面把 Mock / Reflect 作为具体公司或 application 的动作, 不再作为泛用入口。
5. Lab 只留给开发/演示, 不再影响普通用户主路径。

验收标准:

- 用户从首页出发, 不需要知道 SKILL 名称, 也不需要理解多页面流程。
- 同一个 JD 的路径最多是: Mission Control / Pipeline -> Apply Pack / Tailor artifact -> Application timeline。
- 每个高价值产物都有 run_id / skill_run_id / job_id / application_id 可追踪。
- 没有新增“证明自己像 agent”的表或 prompt 仪式。

## 本轮结论

项目不是缺 agent 功能, 是缺产品收敛。

应该保留能力, 收掉入口; 保留产物, 收掉手动流程; 保留追踪, 收掉形式主义。

下一步如果动代码, 应先做 IA 收敛, 而不是再新增 agent 行为。
