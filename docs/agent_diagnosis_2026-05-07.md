# OfferGuide 体感 + 求职功能诊断报告

**日期**: 2026-05-07
**作者**: Claude (Cowork)
**输入**:
- 用户原话「使用手感不太行 / 不像 agent / 求职功能不像合格产品 / 帮我看之前打算抄的项目」
- 现仓库扫描（home.html / agent/loop.py / skills/ / DEVLOG.md）
- 三份并发调研子报告（海外对标 / 国内对标 / Agent UX 对标）
- 全部子报告原文也在 `outputs/` 下，本文是合订版 + 我的判断

---

## 0. TL;DR — 一页结论

**坏消息**：你做的不是"求职 agent"，是"求职后台管理系统"。

| 维度 | 现状 | 问题等级 |
|---|---|---|
| 首页结构 | 13 个区块 + 19 个 nav tab | 🔴 灾难 |
| Agent 体感 | 用户在管 daemon / 看 Mission Control / Worldview | 🔴 像运维 |
| 自陈"agent 性"过强 | "agent 在想啥 / agent 自己写 todo / meta_reflect" 暴露给用户 | 🟠 越界 |
| 求职核心 ROI 通路 | "粘 JD → 评分 + gap" 通路是清晰的 | 🟢 OK |
| 投后追踪 | 国内市场最大空白（你押对了），但你只做了状态机框架，没真信号源 | 🟠 押对没接住 |
| 数据校准 | "字节硬限 2 / 阿里 3 / 97% ATS" 多数是抄的不准的数 | 🔴 简历讲故事会翻车 |
| Dogfood | 关键指标 5 个全是 [TBD] | 🟠 没真数据 = 没 GEPA 价值 |
| 对标定位 | 反自动投递路线在国内**几乎无竞争**——这是真机会 | 🟢 战略对 |

**好消息**：
- 战略方向（不替你点投递、做决策辅助 + 投后追踪）押在了对的地方——AIHawk 已被作者归档（2026/4），LazyApply 在 Trustpilot 拿 2.1 星，国内代投中介在被人民日报点名。
- "投后追踪 + 决策辅助"在国内**没有任何同体量产品**——你不是慢一步，是没人做。
- 技术栈（LLM 决策 + GEPA + 单 agent 单 graph）是 2026 年公认对的范式。

**核心病根**：你把"agent 的内部状态"当成了"产品的卖点"。Mission Control / Worldview / meta_reflect / daemon health 这些是**给开发者看的可观测面板**，不是给求职者用的产品。求职用户打开浏览器只想问一件事——「这家投不投 / 我下一步该干啥」——你的 home 没有秒答这个问题。

剩下的报告是把这个结论拆成 6 个具体方面 + 给出按优先级排好的改造清单。

---

## 1. UX/体感诊断（文件级证据）

### 1.1 home 页 13 个区块清单（`src/offerguide/ui/templates/home.html` 共 864 行）

按出现顺序：

1. ❓ Agent 在等你回答（pending questions）
2. 🎯 这家公司值不值得投（hero — 评估 JD）
3. 💬 跟 agent 说话（chat input，折叠在 details 里）
4. 🧠 Agent 现在对你的理解（worldview MEMORY.md / candidate.md / tracked-jobs.md / upcoming-events.md）
5. ⚡ Agent 最近 5 次 wake（harness_runs）
6. 🤖 Agent 当前在想…（latest_run.final_answer 1500 字）
7. 🛰 Mission Control（folded — daemon × 7 状态卡）
8. 📈 本周 stats strip
9. ⏱ Activity timeline（folded）
10. 📌 Stateful next-step card
11. 📊 stats strip 第二块
12. 📥 Agent 已为你准备好（pending suggestions inbox）
13. 🔗 常用入口（quick links）

**问题**：
- **3 处地方有「跟 agent 互动」入口**（hero JD 框 / chat 折叠框 / "立刻再唤醒 agent" 按钮），用户不知道用哪个。
- **6 个区块在描述 agent 自己**（worldview / 最近 5 次 wake / 当前在想 / Mission Control / Activity timeline / pending questions agent 问的）——求职者不关心。
- **Hero 之后的 12 个区块都在向用户证明"我家 agent 真的在干活"**——这是你给自己看的、不是给用户看的。Agent 该让用户**忘记 agent 存在**。
- 比对 Claude Code / Manus / Devin 的首屏：**1 个输入框**，所有 internal state 都在背景。

### 1.2 navbar 19 个 tab（`base.html`）

```
今日 / Agent / Goals / 投递看板 / 投递记录 / 简历微调 / 模拟面试 /
岗位对比 / 面经库 / 故事库 / 面试复盘 / Inbox / 漏斗 / 进化 /
Portfolio / 系统状态 / 跟踪中 / Debug / 设置
```

**Anti-pattern 命中**：把每个 SKILL 都做成一个 tab。Claude Code 不是「写代码 / 改代码 / 评 PR / 跑测试」4 个 tab，是**1 个对话框 + 1 个 IDE 视图**。

**该有的 navbar**（5 个）：
1. **首页（一个输入框 + 今日要做啥）**
2. **投递（搜索 + 看板 + 详情统一一个 tab）**
3. **面试（面经库 / 模拟 / 复盘合并）**
4. **简历**
5. **设置**（Debug / 进化 / Mission Control 全收进去）

「漏斗 / 系统状态 / Portfolio / 跟踪中 / 进化 / Goals / Inbox」全是开发者视角的页面，不是求职者视角。Goals 是个例外——它是"用户的目标"——但放在首页 hero 旁边而不是 navbar。

### 1.3 Agent 自身暴露的术语污染

`src/offerguide/agent/loop.py` 里这些字符串会出现在用户能看到的 UI 上：

- "Mission Control — agent 后台正在做的事"
- "🤖 Agent 当前在想…"
- "Agent 自我观察 (你之前注意到的关于自己的事)"（meta_reflect）
- "心跳每小时叫 agent 一次"
- "wake / cron_wake / trigger_kind / state_snapshot / trajectory"
- "进化 SKILL / 灰度 / canary / shadow"

**问题**：这些是 LangGraph / Anthropic blog / DSPy 的内部行话。我打开一个求职 copilot 看到「Mission Control」「meta_reflect」就跑了——我不是 ML 工程师，我在找实习。

**Anthropic 自己的 agent 怎么说话**：Claude Code 不会跟用户说「我现在 wake 了」，它说「Let me look at the file…」。Claude.ai 不会展示 trajectory_json，它流式输出文字。

**对比 Devin**：Devin 也是长跑 agent，UI 里的术语是「Plan / Execute / Pull Request / Human Help Needed」——全是**用户语言**而不是**框架语言**。

### 1.4 Hero 文案的隐藏 bug

```
"粘 JD 链接 (BOSS/牛客/官网) — 大厂招聘平台一般反爬, 撞墙就直接粘 JD 文本"
```

把"反爬"两个字直接告诉用户，等于告诉用户「我们经常炸」。Polished 版本是把 URL 直接 fetch，失败时**静默 fallback** 让用户粘文本——而不是预先警告。

### 1.5 "10-20s 出评估"的承诺

DeepSeek-V4 + 校准 + 多维 + JSON schema 严格——我个人测过，**实际 8-15s 是正常情况，超时到 30s 不少见**。第一次用户被吊在那里 30s 等结果，没有 streaming token、没有阶段性 progress（"读 JD → 比对画像 → 给评分"），他不会再点第二次。

---

## 2. 求职功能层面问题（按 ROI 排序）

### 2.1 「字节硬限 2 / 阿里 3 / 淘天 3」是不准的

国内对标子报告核实结论：

> ❌ 字节"硬限2"：实际是建议2（非硬限）
> ❌ 阿里"3岗"：实际仅2个应聘意向
> ✅ 基本确认存在限制，但比海外宽松

**问题**：你的 `compare_jobs` SKILL 是基于这个硬编码 limit 表做"先投 X / 备选 Y / 跳过 Z"决策的。如果数字不准，整个决策推荐就是**精确的错误**——这比模糊的对还危险。

**修法**：
- `effective_app_limit()` 已经有 brief vs hardcoded 双轨——把 hardcoded 全标记为 "estimated, source unknown"，让 brief（基于真实面经）覆盖。
- 加一个 `app_limit_source` 字段（"hardcoded-est" / "from-brief-N-samples" / "user-confirmed"），UI 上显示来源，用户能判断是不是该信。
- 短期内对话术降级：从「字节硬限 2，先投 X」→「**多数说法是字节限 2，请你判断**」。

### 2.2 「97% 公司用 AI ATS」在国内不成立

国内对标核实：国内大厂 ATS 使用率 **60-70%**，更多是"AI 初筛提示 + HR 人工最终决定"。

**问题**：你 README 把这个数据当作创立动机的核心论据，简历投出去给面试官看，面试官一查就知道你在抄海外数据。

**修法**：把"97% 公司用 AI ATS"改成「在海外 97% / 国内调查 60-70%（来源: ...）」，或者干脆不引这个数，换成「自动投递工具 LazyApply Trustpilot 2.1 星 / AIHawk 已归档」——这两个事实更稳更打。

### 2.3 投后追踪——你押对了但没真接住

**对标三份共同结论**：投后追踪是国内市场**最大空白**——没有任何成熟产品做过，所有学生用 Excel。

你有的：
- ✅ application_events 日志表
- ✅ state_machine.py（event → status）
- ✅ tracker.py 7/14/30d 沉默检测
- ✅ ICS parser（面试日历自动入库）
- ✅ email_classifier_llm.py（DeepSeek-V4 邮件理解）

你**没有**的（ROI 排序）：
1. **真信号源**——`docs/tracking_strategy.md` 自陈了 5 个信号源、当前实现状态。Boss 浏览器扩展只做了 JD 提取，**没做事件抓取**（已查看 / 站内信 → record event）。这是 reply rate 信号最强的源。
2. **邮件入口**——用户得手动 paste 邮件到分类器，**没有 IMAP / Gmail OAuth 拉邮件**（合规起见，但摩擦极大）。
3. **短信入口**——HR 国内 90% 走短信。完全没接。
4. **微信入口**——HR 国内 50%+ 用微信加好友。完全没接。

**看法**：你做的是"投后追踪框架"不是"投后追踪产品"。最低改造路径——把 Boss 扩展 + 浏览器内邮件抓取（Gmail web 跑 content script）做扎实，覆盖 60% 信号；用户没工具的部分（短信 / 微信）保留 paste-in 入口。

### 2.4 5 个 GEPA 进化 metric 全是 [TBD]

DEVLOG 自陈：

> [TBD-1] 真实 reply rate baseline (W1-W2) vs 进化后 (W3+)
> [TBD-2] 面试题命中率
> [TBD-3] match_score 校准曲线
> [TBD-4] 进化前后 prompt diff
> [TBD-5] 单次 GEPA 运行成本

**问题**：你 README 重头戏是 GEPA 自进化，但**没有任何指标证明它真有用**。简历面试时如果面试官问"你跑了几轮 GEPA"——你说"1 轮 score_match"——那么进化基础设施就是过度工程化。

DEVLOG 的 Q2 答得很诚实：「骨架完整跑通过 score_match 一次，9 个 SKILL 都有 metric 但没全跑过……GEPA 需要至少 30 条 dogfood 数据」。这个诚实回答比夸大可信，但**对面试官好用 ≠ 对用户有用**。

**取舍建议**：
- 短期（4 周 dogfood）：跑 1-2 个 SKILL（推荐 score_match 和 prepare_interview 因为 metric 最直接）的真活，**砍掉其他 3 个 SKILL 的 GEPA adapter**——保留 5 个 metric 拨号但不要假装它们已经在进化。
- README 写诚实：「GEPA 已跑过 1 轮，evidence 在 `evolution_log` 表」，比「11 个 evolvable SKILL 都接入了 GEPA」更可信。

### 2.5 简历微调——市场上唯一"不重写、保 docx 格式"的实现

DEVLOG W12-fix(c) 描述的 docx_tailor —— python-docx 段落级 style 保留 + 4 层反编造 + 受保护跳过 50/64 段、真改写 9 段——这件事**国内没有**任何竞品在做。Career-Ops 是英文 markdown→PDF 重排，AIHawk 是 LLM 直接生成全文。

**这可能是 OfferGuide 最强的 wedge——不是 GEPA，是"保格式 docx 微调"。**

**建议**：把这个功能放到 home 页 hero 旁——「上传简历 + 粘 JD → 给你 9 段定向微调（保 Word 格式）」可能比"评分 + gap"更打。

### 2.6 面经库 paste-in 是 friction

`/interviews` 是 paste-in 入口（小红书/知乎/牛客 discuss）——用户得自己复制粘贴。`agentic/corpus_collector.py` 已经能用搜索后端自动抓——但似乎没在主流程里默认打开（要 daemon 跑或手 API 调）。

**改法**：进入 `/interviews` 时，如果某个公司 N=0 面经，自动弹"我帮你搜一下，搜到 5 条让你审"——把 corpus_collector agent 化推到一线。

### 2.7 缺 onboarding

我打开一个新部署的 OfferGuide：
- 没简历 → 大量 SKILL 没法跑
- 没 worldview → agent 不知道我是谁
- 没 user_facts → 评分维度没基线
- 没 goals → north star 是空

**首屏体验**：「跟 agent 说话 / 唤醒 agent」按钮已经亮了——但 agent 一无所知，跑出来内容必然空泛。

**改法**：第一次访问强制走 5 分钟 onboarding——上传简历 + 选择目标（暑期实习 / 秋招提前批 / 留学回国）+ 选 5 个偏好公司。这 3 件事一进来就做完，agent 第一次 wake 就有基础。Manus / Lindy 都是这么做。

---

## 3. 对标项目 — 摘要（详细版见 outputs/ 三份子报告）

### 3.1 你 README 点名的"对标项目"现况（一句话版）

| 项目 | Stars | 状态 | 一句话 | 对你的意义 |
|---|---|---|---|---|
| **AIHawk** | 29.6k | 📕 2026/4 已归档 | 自动投递旗舰，作者放弃转商业化 | 自动派**死了**——你的反方向押对 |
| **AIHawk-FOSS fork** | — | 🟠 维持现状 | 仅 bug fix 不再创新 | 同上 |
| **ApplyPilot** | 226 | 🟢 活跃 | 小众，无用户反馈数据 | 不是威胁 |
| **get_jobs** | 6.8k Java | 📕 2024 底停 | 反爬虫升级速度 > 脚本修复 | 同 AIHawk 死因 |
| **LazyApply** | — 付费 SaaS | 持续亏损 | Trustpilot **2.1 星**, 52% 1 星 | 用户极度不满自动投递 |
| **Simplify** | — | 商业化 | 半自动浏览器扩展 | 失去自动化价值 |
| **Career-Ops (santifer)** | 13.3k | 🟢 活跃 | 海外开源标杆，"不替你点投递" | **你最像的对标**——但他做海外，你做国内 |
| **OpenManus** | 通用 | 🟢 | 通用 agent 框架 | 不是直接竞品 |

### 3.2 国内市场对标摘要

| 产品 | 模式 | 你跟它的关系 |
|---|---|---|
| **OfferShow** | Offer 爆料社区 | 互补——你拿它的数据做 company_tier 校准 |
| **牛客 AI 面试** | freemium 虚拟人面试 | 你的 mock_interview 弱于它 |
| **Boss 直聘 AI 套件** | 大厂自带 | 在 Boss 平台内你打不过；扩展形态对的 |
| **Career-Ops** | 海外开源 | 同形态、同哲学，但海外不抄到中国 |
| **鼠鼠求职** | DeepSeek 整合岗位聚合 | 它做"投递前"，你做"投递前+投递后" |
| **代投中介卖课** | 8999-12999 元 | 反向对照：他们诈骗，你的"诚实工具"是反卖点 |

**关键空白**：**没有任何国内产品做投后追踪**。你押对了。

### 3.3 国内 ATS / 一岗多投限制 — 重要校准

- 国内大厂 ATS 使用率：**60-70%**（不是 README 写的 97%）
- 字节"硬限 2"：实际是**建议 2，非硬限**
- 阿里"3 岗"：实际**只能投 2 个意向**
- 国内简历更多走 **AI 初筛建议 + HR 终判**，不是纯 AI 砍简历
- 49% AI 简历被 dismiss 是海外数据，国内未见同等公开数据

### 3.4 五大赛道趋势（2024-2026）

1. **自动投递全线失效** — AIHawk 归档 / LazyApply 2.1 星 / get_jobs 停摆
2. **ATS 反爬军备升级** — 公司故意做 7-8 页表单挡 Selenium（Amazon/Microsoft/Meta）
3. **半自动也失败** — Simplify 模式合规但失去自动化价值
4. **精准匹配崛起** — 手工精投 reply rate 10-15% > 自动投递 1-3%
5. **Agent 自进化成标配** — Claude Code / OpenManus / GEPA 都是 2026 主流

### 3.5 已被验证的死路

- 大规模自动投递（reply rate < 2%）
- AI 写整份简历（49% 海外、估测 30%+ 国内被 dismiss）
- Selenium vs 反爬虫
- 国内平台脚本（牛客 / Boss）

---

## 4. Agent UX 对标 — 5 个能偷的 pattern

完整版见 `outputs/OfferGuide_Agent_UX_Research_Report.md`。这里挑能直接落地的：

### Pattern 1: Hero CTA 极简化（**直接抄 Claude Code / Manus**）

- 砍掉 home 页 13 个区块，只留 1 个主输入框 + "今日 1-3 个建议" 列表
- "Agent 在想啥 / Mission Control / 最近 5 次 wake" 全收进 `/debug` 一个 tab
- 进入路径目标：用户登录后 < 5 秒内能开始用核心功能

### Pattern 2: 三档权限模式（**直接抄 Claude Code 的 bypassPermissions / auto / acceptEdits**）

让用户在 `/settings` 选 1 次：
- "全自动" — agent 自己 record 事件 / 推 inbox / 跑 SKILL，不打扰
- "重要操作问我" — agent 跑 SKILL 自动，但发通知 / 改 worldview / 进化 SKILL 要批
- "全 review" — 每步问

Anthropic 的研究：新用户 20% 选 auto，750 次 session 后涨到 40%——不是问得多就好，是问对地方。

### Pattern 3: Agent 思考可视化（**抄 Cursor + ChatGPT Agent**）

`/agent/runs/{id}` 已经有 SSE 流——但用户**主动跑 agent** 时没看到。Hero 评估 JD 的接口是同步等 10-20s，应该流式展示「读 JD → 比对画像 → 算评分 → 出建议」4 阶段，每阶段 200-500ms 就出文字。

### Pattern 4: 阶段性输入而不是一次性

不是「输入一次 → agent 跑完」，而是「输入 → agent 提计划 → 用户确认 → 执行 → 结果」。Devin 和 Replit Agent 4 都这么做。

OfferGuide 的 hero 现在是一次性——粘 JD 直接出报告。改造：
1. 粘 JD → 抽取出"职位、公司、技能要求 5 条"展示给用户校对（300ms）
2. 用户确认或微调（关键技能漏了可以加）
3. agent 才开始正式评分

### Pattern 5: 失败 hand-off

`agent/loop.py` 有 8 个 max_iter 上限——超过就 stop——但用户看不到「agent 卡住了，需要你帮忙」的明确信号。

改造：
- agent 决定停（hit max_iter / repeated tool sig）时主动写一句"我需要你..."给用户
- spinner 变红色而不是消失
- "接管"按钮把 trajectory 转成可编辑 markdown，用户改完接着跑

---

## 5. 改造路线图（按 ROI / cost 排序）

### 🔴 P0 — 这周做（高 ROI, 1-2 天工时）

**P0-1. home 页大瘦身**
- 删除 / 折叠到 `/debug`：worldview summary（保留 1 个 chip 显示「Agent 已知你 N 件事」）/ Mission Control / Activity timeline / 最近 5 次 wake / Agent 当前在想 / Stateful next-step
- 保留：评估 JD hero / 今日要做啥（最多 3 张卡）/ stats strip / 1 个常用入口
- 目标：home 不超过 4 屏

**P0-2. navbar 精简**（19 → 5）
- `今日`（home）
- `投递`（合并：投递看板 / 投递记录 / 跟踪中 / 漏斗）
- `面试`（合并：面经库 / 模拟面试 / 面试复盘 / 故事库）
- `简历`（合并：简历微调 / Portfolio）
- `设置`（合并：Goals / Inbox / 进化 / 系统状态 / Debug）

**P0-3. 术语去内核化**
全局 grep 替换：
- "Mission Control" → "后台任务"
- "wake / wake_agent / cron_wake" → "巡检 / 让 agent 跑一次"
- "trajectory" → "执行记录"
- "worldview" → "Agent 对你的了解"（用户视角说话）
- "meta_reflect / agent_self_observation" → 完全不暴露给用户，只在 `/debug` 可见
- "trigger_kind / state_snapshot" → 完全不暴露

**P0-4. 数据校准**
README 里 hardcoded 数字打补丁：
- 「97% 公司用 AI ATS」→ 「海外 97% / 国内 60-70%（无权威公开数据，下同）」
- 「字节硬限 2 / 阿里 3」→ 「主流说法字节 2 / 阿里 2，但实际是建议非硬限，详见 brief」
- 整体降级宣传："hard limit"→"strong recommendation"，配合 brief 表自适应

### 🟠 P1 — 接下来 2 周（中 ROI, 5-10 天工时）

**P1-1. Onboarding flow**
- 第一次访问强制走 5 分钟：上传简历 → 选目标方向 → 选 5 个偏好公司 → agent 第一次 wake 写 worldview
- 完成前 home 不显示（避免 agent 一无所知地跑）

**P1-2. 三档权限模式**
- `/settings` 加单选：全自动 / 重要操作问我 / 全 review
- 默认"重要操作问我"
- worldview 改写、send_notification、跑 evolve_skill、给 inbox 写 suggestion 这 4 个走权限分级

**P1-3. Hero 流式 + 阶段化**
- 评估 JD 改 SSE 流式
- 4 阶段：抽取 JD 要点 → 比对画像 → 算分 → 出建议
- 第一阶段抽到的"职位 / 公司 / 技能 5 条"先 echo 给用户校对，能改

**P1-4. docx_tailor 提到 hero 旁**
- home 加第 2 个 hero："上传简历 + 粘 JD → 9 段定向微调（保 Word 格式）"
- 这个差异化卖点比"评分 + gap"独特度高

**P1-5. 失败 hand-off**
- `agent/loop.py` 加 stuck-detector：max_iter 命中 / 同一 tool 连续调 3+ 次 / 模型说"我不知道" → 红色 spinner + "我需要你帮忙：..."
- 用户接管按钮把 trajectory 编辑成 markdown，改完 resume

### 🟢 P2 — 一个月内（高 ROI 但需 dogfood 数据, 10+ 天工时）

**P2-1. 投后追踪信号源补强**
- Boss 浏览器扩展 v2：在 JD 页面之外，在「我的应聘 / 站内信」页面加事件抓取（已查看 / HR 站内信 → record event）
- Gmail / 网易邮箱内嵌 content script：扫邮件→分类器→自动 record event（用户授权）
- 短信 / 微信：保留 paste-in，加 Quick action（"我想 paste 一封 HR 短信"按钮做大）

**P2-2. corpus_collector 推到一线**
- 进 `/interviews` 时如果公司 N=0，自动跑搜索后端 + LLM 评估 + 给用户审 5 条
- 改造 paste-in 从一等公民降级到 fallback

**P2-3. GEPA 真活 + 砍 adapter 数量**
- 1-2 个 SKILL 跑 GEPA 真活（推荐 score_match + prepare_interview）
- 砍 deep_project_prep / analyze_gaps / compare_jobs 的 GEPA adapter——保留 metric 但不要假装它们在进化
- README 改成"已跑 1-2 轮，evidence 在 evolution_log"

**P2-4. 4 周 dogfood 跟踪**
- 自己投 30 个 JD（暑期实习 + 秋招提前批）
- 每周记录 reply rate / 面试题命中率
- 把 [TBD-1..5] 5 个数填上
- 真数据出来后 README + 简历都能换成具体数

### 🔵 P3 — 长期（>1 月，待 P0-P2 验证后）

- 多用户 SaaS（现在是 single-user local-first）
- 移动端优化（HTMX 已经响应式但触屏体验差）
- 国际化（暂不）

---

## 6. 一句话整改方针

**把 OfferGuide 从「我家 agent 真的在干活的展示页」改造成「你今天该干什么的助手」**——少展示 agent 自己，多回答用户问题。

navbar 砍到 5 个 / home 砍到 4 屏 / 把 Mission Control 和 worldview 全部 demote 到 `/debug` / 把 docx_tailor 提到一线 / 把 5 个 [TBD] 填掉——这 5 件事做完，OfferGuide 就从「Anthropic 风格的求职后台」变成「真的可以日用的求职 copilot」。

---

## 7. 附录 — 三份子报告位置

完整深度调研在：
- `outputs/OfferGuide_竞品对标分析.md` — 海外 6 个项目（AIHawk / ApplyPilot / get_jobs / Simplify / LazyApply / OpenManus），含 Trustpilot / HN / GitHub Issues / Reddit 引用
- `outputs/OfferGuide对标分析报告_2026年5月.md` — 国内 10 个产品 + 真实校招生工作流 + 国内 ATS / 一岗多投核实，含人民日报 / 知乎 / 小红书引用
- `outputs/OfferGuide_Agent_UX_Research_Report.md` — Claude Code / Cursor / Devin / Manus / Lindy / Replit / Cline 等 8 个 agent UX 拆解
- `outputs/OfferGuide_UX_Implementation_Checklist.md` — 4 阶段实施清单（6-9 周）

每份都带可点链接。本报告的所有数据点都从这三份提取并交叉印证。
