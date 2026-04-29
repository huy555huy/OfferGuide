# OfferGuide 项目可行性与方案备忘录

日期：2026-04-28

## 1. 核心结论

OfferGuide 是可行的，但它不应该被定位成“自动海投工具”。更稳、更有长期价值的定位是：

> 本地优先的中国校招求职 Copilot：自动发现岗位，辅助判断投不投，生成克制的投递包，追踪投递结果，并从沉默与反馈中持续校准策略。

这意味着项目的重心不是“替用户投更多”，而是：

- 自动找岗位，减少刷平台的时间。
- 帮用户判断岗位是否值得投入精力。
- 做简历微调，而不是 AI 味整段重写。
- 辅助官网填表、Boss/牛客页面采集和沟通草稿，但关键动作保留人工确认。
- 记录投递、沉默、回复、笔试、面试、拒绝、offer，形成反馈闭环。
- 未来在投递数据足够后，引入生存分析和 prompt evolution。

## 2. 为什么这个方向有现实需求

中国校招压力是真实背景。公开报道显示，2026 届全国普通高校毕业生预计 1270 万人，同比增加 48 万。岗位信息分散在大厂官网、Boss、牛客、智联、前程无忧、猎聘、高校就业办、公众号、内推群等渠道，普通学生需要反复搜索、判断、填表、记录进度。

同时，招聘侧也在 AI 化。海外数据已经显示招聘 AI 常用于写 JD、筛简历、搜索候选人、沟通候选人；候选人和招聘方之间也出现了 AI 信任危机。这个背景下，“自动海投”会进一步制造低质量申请，而“高质量筛选 + 可解释投递策略 + 反馈学习”更有价值。

## 3. 国内类似产品现状

调研到的国内产品大致可以分为几类：

| 类型 | 代表产品 | 主要做法 |
|---|---|---|
| 全流程托管型 | Offer快 | 自动搜索、筛选、沟通、发简历、网申填表，保留人工介入提醒 |
| 校招官网填表型 | 求职方舟AI | AI找工作、AI填简历、AI改简历，浏览器插件自动填写企业官网表单 |
| 校招 Copilot 型 | 效招 Joblytic | 岗位匹配、定制材料、跨平台自动填表、进度看板、AI 求职私教 |
| Boss 批量投递型 | 简历快投、智投简历、Sider 求职大师 | 围绕 Boss/智联等平台做自动筛选、批量打招呼、发送简历图片 |
| 浏览器插件型 | boss-helper、AI求职助手 | 在 Boss 页面解析岗位、生成招呼语、辅助一键投递 |
| 求职工具箱型 | OfferKit | 简历优化、岗位匹配、模拟面试、投递追踪，不太强调真正自动投递 |

现有产品的共同不足：

- 过度强调自动投递、批量打招呼、日投几百，容易滑向低质量海投。
- 平台脆弱性高，Boss/牛客/智联页面变化、验证码、登录态和风控都会影响稳定性。
- 简历“包装”“魔改”容易越过事实边界，生成明显 AI 味内容。
- 多数没有严肃的真实反馈校准，只展示匹配度或宣传回复率。
- 对隐私、招聘账号安全、数据用途的解释不足。
- 更像效率工具，不像长期求职决策系统。

OfferGuide 的机会在于：不拼“投得多”，而拼“判断可信、修改克制、反馈可学、本地安全、中国校招语境深”。

## 4. 产品主流程

建议完整流程如下：

```text
自动发现岗位
→ 去重 / 清洗 / 标准化
→ 根据简历与偏好排序
→ 每日推荐清单
→ 用户选择感兴趣岗位
→ 生成投递包
→ 辅助填表 / 辅助沟通 / 用户确认提交
→ 记录投递状态
→ 7/14/30 天沉默反馈
→ 生存分析与评分校准
→ 反哺下一轮推荐
```

核心不是单次 LLM 调用，而是把“建议 -> 行动 -> 结果 -> 复盘”接起来。

## 5. 自动化边界

自动化需要分层，不要一上来做全自动海投：

| 层级 | 行为 | 可行性 | 风险 | 建议 |
|---|---|---:|---:|---|
| L1 自动找岗位 | 扫描、汇总、去重、排序 | 高 | 低 | 必做 |
| L2 自动准备投递包 | 简历微调、开放题、打招呼语 | 高 | 低 | 必做 |
| L3 自动填表但不提交 | 插件/RPA 填官网表单，停在提交前 | 中高 | 中 | 应做 |
| L4 自动提交/自动沟通 | 自动投递、自动发 Boss 消息、自动跟进 | 中 | 高 | 谨慎、白名单、限频 |

默认模式应该是 Copilot：

- 自动准备材料。
- 自动填写重复字段。
- 自动生成沟通草稿。
- 停在提交/发送前，由用户确认。

未来可以做白名单自动提交，但必须满足：

- 用户明确开启。
- 每日上限。
- 最低匹配分阈值。
- 黑名单公司/外包/猎头过滤。
- 全程记录。
- 一键暂停。

不建议做：

- 绕验证码。
- 绕登录风控。
- 伪装人类大规模骚扰 HR。
- 编造简历经历。

## 6. 大厂官网、牛客、Boss 分别怎么处理

### 大厂官网

大厂官网投递通常是自研系统、第三方 ATS、官网表单、邮箱投递或内推链接。这里不适合一开始强行全自动提交。

建议做 Application Pack：

- 匹配判断。
- 推荐投 / 不投理由。
- 简历微调建议。
- 开放题答案草稿。
- 岗位专属项目亮点。
- 官网投递链接。
- 状态追踪。

浏览器插件负责：

- 读取当前官网页面可见岗位信息。
- 辅助填写表单。
- 停在提交前。

### 牛客

牛客可以支持两种路径：

- 低频、公开、谨慎地使用 sitemap 或公开页面。
- 更优先做浏览器插件：用户打开岗位页或列表页后，从当前页面提取职位名、公司、城市、薪资、JD、反馈率、岗位链接。

不要把牛客动态页面爬虫作为主路径。

### Boss

Boss 强登录、强交互、强风控，不建议后台爬虫或自动登录。

建议：

- 用户打开 Boss 搜索页或岗位页。
- 插件读取当前可见岗位。
- OfferGuide 计算匹配度、生成招呼语。
- 可以复制/填入草稿，但默认不自动发送。
- 若未来自动发送，必须白名单、限频、保留用户接管。

## 7. 技术栈建议

当前技术栈方向基本正确：

| 模块 | 技术 |
|---|---|
| 后端 | Python + FastAPI |
| Agent 编排 | LangGraph |
| 技能系统 | SKILL.md + Pydantic schema |
| 存储 | SQLite + sqlite-vec |
| Web UI | Jinja2 + HTMX |
| LLM | DeepSeek V4 Flash 默认，复杂任务可升 Pro |
| 通知 | Feishu / Telegram / Console |
| 浏览器插件 | Chrome Extension Manifest V3 |
| 调度 | APScheduler，后置 |
| 页面自动化测试 | Playwright，主要用于测试和少量公开页面验证 |

架构建议：

```text
Source Capture 层
手动粘贴 / 当前网页插件抓取 / sitemap / 官网 registry / 邮件导入 / 截图 OCR

Normalization 层
RawJob -> 统一 JD schema -> 去重 -> jobs + extras_json

Decision 层
score_match / analyze_gaps / prepare_interview / survival estimates

Action Assist 层
投递包 / 表单辅助填写 / 消息草稿 / inbox 确认

Learning 层
application_events / silent checks / survival analysis / prompt evolution
```

## 8. 数据模型重点

不要只用一个 `status` 字段表达所有事情。建议保留事件日志：

```text
applications
- id
- job_id
- channel
- status
- applied_at
- first_response_at
- first_response_kind
- last_checked_at
- score_run_id
- gaps_run_id
- resume_version_id
- notes

application_events
- id
- application_id
- kind: submitted / viewed / replied / assessment / interview / rejected / offer / silent_check
- occurred_at
- source: manual / email / platform / calendar
- payload_json
```

这样后续可以从事件日志推导状态、反馈、沉默、转化率和生存分析数据。

之前代码 review 中也确认了两点要修：

- SKILL runtime 要拒绝未声明的 extra inputs，或让 render/hash/persist 使用同一份规范化输入。
- `RawJob.extras` 应该结构化存进 `extras_json`，不要拼到 `raw_text`。

## 9. 沉默也是反馈

真实求职里，大多数投递不会得到明确拒信，而是石沉大海。不能因为没有反馈就无法学习。

建议把沉默建模为延迟反馈 / 右删失数据：

| 状态 | 含义 |
|---|---|
| `submitted` | 已投递 |
| `viewed` | 被查看 |
| `replied` | HR 回复 |
| `assessment` | 笔试/测评 |
| `interview` | 面试 |
| `rejected` | 明确拒绝 |
| `silent_7d` | 7 天无反馈 |
| `silent_14d` | 14 天无反馈 |
| `silent_30d` | 30 天无反馈 |
| `expired_unknown` | 岗位过期但无反馈 |

产品上可以显示沉默节点，但建模时要注意：沉默不是明确失败，而是“截至观察日仍未发生事件”。

## 10. 生存分析设想

如果投递数据多，生存分析非常适合这个项目。

映射关系：

| 生存分析 | 求职场景 |
|---|---|
| 起点 `t0` | 投递时间 |
| 事件 event | HR 回复 / 笔试 / 面试 / 拒信 |
| 时间 `T` | 从投递到事件发生的天数 |
| 右删失 censoring | 到今天还没任何反馈 |
| hazard | 某一天突然收到反馈的即时概率 |
| survival curve | 投递后仍然没有反馈的概率曲线 |
| covariates | 匹配分、渠道、公司、岗位、城市、学历要求、简历版本、投递时间等 |

可以回答的问题：

- 高匹配岗位通常几天内会回？
- 超过多少天没回基本可以降低期待？
- 官网、Boss、牛客哪个渠道反馈更快？
- 大厂和中小厂反馈时间分布有什么差异？
- 简历微调是否不仅提高回复率，也缩短反馈时间？
- 周一上午投递是否比周五晚上更好？

分阶段实现：

| 数据量 | 做法 |
|---:|---|
| 0-30 条 | 只记录事件，不建模 |
| 30-80 条 | 统计 7/14/30 天沉默率、回复率 |
| 80-150 条 | Kaplan-Meier 曲线 |
| 150-300 条 | Cox Proportional Hazards |
| 300+ 条 | 渠道/岗位分层、竞争风险模型 |

未来 `score_match` 可以从单个概率升级为：

```json
{
  "reply_probability_7d": 0.12,
  "reply_probability_14d": 0.21,
  "median_expected_response_days": 9,
  "worth_tailoring": true,
  "reasoning": "官网投递慢，但岗位匹配度高，值得深度定制"
}
```

这会让 OfferGuide 从“LLM 求职顾问”进化成“个人投递策略系统”。

## 11. MVP 路线

### Phase 0：稳定当前原型

- 修复 pytest / pyright / review findings。
- 补上 `extras_json`。
- 规范 SKILL inputs。
- README 更新到真实进度。

### Phase 1：Job Radar

- 支持手动粘贴 JD。
- 支持牛客公开来源或页面抓取。
- 建 company source registry。
- 做岗位去重、清洗、排序。
- 每日推荐清单。

### Phase 2：Application Copilot

- 生成投递包。
- 生成官网开放题草稿。
- 生成 Boss 招呼语草稿。
- 浏览器插件抓当前页面 JD。
- 插件辅助填写官网表单，但默认不提交。

### Phase 3：Tracking & Feedback

- 真正启用 `applications`。
- 增加 `application_events`。
- 记录 submitted / viewed / replied / assessment / interview / rejected / offer / silent_check。
- 做 7/14/30 天沉默提醒。
- dashboard 展示投递数、回复率、沉默率、高低分组差异。

### Phase 4：Interview Prep

- 支持手动粘贴面经。
- 公司/岗位维度 RAG。
- 根据 JD + 简历 gap + 面经生成准备清单。

### Phase 5：Survival & Evolution

- Kaplan-Meier 曲线。
- 分渠道/分岗位反馈时间分析。
- 数据足够后做 Cox 模型。
- 再考虑 GEPA 或 prompt evolution。

## 12. 成功标准

短期不要证明“agent 很聪明”，要证明以下闭环成立：

- 自动找岗位能减少每天刷平台的时间。
- 匹配评分能让用户更快判断值不值得投。
- 简历建议足够具体、克制、不编造，用户愿意采纳。
- 投递状态能被低成本记录。
- 高分岗位的 14/30 天回复率高于低分岗位。
- 沉默反馈能指导下一轮投递策略。

一个现实的 dogfood 目标：

```text
30 天内处理 100 个 JD
实际投递 20-40 个
记录完整 application_events
输出高分组 vs 低分组的回复率/沉默率对比
形成第一版求职策略复盘
```

## 13. 关键取舍

应该自动化：

- 找岗位。
- 去重。
- 过滤垃圾岗位。
- 排优先级。
- 生成投递材料。
- 填重复表单。
- 记录状态。
- 定时提醒。
- 统计沉默率、回复率、反馈时间分布。

谨慎自动化：

- 点提交。
- 发 Boss 招呼。
- follow-up。
- 大规模批量申请。

不应该自动化：

- 绕验证码。
- 绕登录风控。
- 伪装人类进行大规模骚扰。
- 编造简历经历。

## 14. 参考来源

- 教育部 / 央视网：2026 届高校毕业生预计 1270 万人  
  https://news.cctv.com/2025/11/20/ARTI0xYbzeyS5Y6Zky3R3VZg251120.shtml
- SHRM: AI in HR 2025  
  https://www.shrm.org/topics-tools/research/2025-talent-trends/ai-in-hr
- Greenhouse: AI Trust Crisis in Hiring  
  https://www.greenhouse.com/uk/newsroom/an-ai-trust-crisis-70-of-hiring-managers-trust-ai-to-make-faster-and-better-hiring-decisions-only-8-of-job-seekers-call-it-fair
- Gartner: Job applicants' trust in AI hiring  
  https://www.gartner.com/en/newsroom/press-releases/2025-07-31-gartner-survey-shows-just-26-percent-of-job-applicants-trust-ai-will-fairly-evaluate-them
- DeepSeek API pricing  
  https://api-docs.deepseek.com/quick_start/pricing
- LangGraph durable execution  
  https://docs.langchain.com/oss/python/langgraph/durable-execution
- LangGraph human-in-the-loop  
  https://docs.langchain.com/oss/python/langchain/human-in-the-loop
- DSPy GEPA  
  https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA/overview.md
- PIPL Article 24 automated decision-making  
  https://en.spp.gov.cn/2021-12/29/c_948419.htm
- Offer快  
  https://www.offerkuai.com/
- 求职方舟AI  
  https://www.qiuzhifangzhou.com/doc/about/
- 求职方舟AI：AI填简历使用手册  
  https://www.qiuzhifangzhou.com/doc/manual/autofill.html
- 效招 Joblytic  
  https://xzjobs.cn/
- 简历快投  
  https://jobdajob.com.cn/
- 智投简历  
  https://www.zhitoujianli.com/
- OfferKit  
  https://www.offerkit.ai/
