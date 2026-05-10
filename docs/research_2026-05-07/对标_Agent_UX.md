# OfferGuide Agent UX 调研报告
## 对标 8 个顶级 Agent 产品的深度分析

**报告日期**：2026 年 5 月  
**调研范围**：2024-2026 年通用 Agent + 垂类 Agent 产品  
**核心问题**：用户反馈"不够顺畅、不够 agent"，需要找到 UX 设计核心瓶颈

---

## 执行摘要

通过对 **Claude Code、Cursor Composer、Devin 2.0、OpenAI Operator、Manus、Lindy、Replit Agent、Cline/Roo Code** 的调研，我发现**"顺畅的 agent 体感"的核心不在于功能多，而在于三个层面的极简化**：

1. **进入路径极简**：一个输入框、一个明确的 CTA，不是 13 个区块 + 19 个 tab
2. **审批环节隐形**：让 agent 在"安全的默认值"内自主运作，只在真正危险的操作时弹出审批
3. **失败 handoff 流畅**：agent 卡住时有明确的"我搞不定，你来接手"的过渡，而不是让用户懵逼

**OfferGuide 的核心问题**就在这三点都没做好。

---

## 一、对标产品 UX 深度分析

### 1. Claude Code（Anthropic 官方 Agent 参考）

**进入路径**  
- Hero 页面：空白编辑器 + 一个输入框
- 无菜单，无导航栏  
- 权限提示（upfront permission gates）：一次性设置后，不再重复打扰

**Agent 工作时**  
- 左侧流式显示 thinking trace（summarized thinking 模式）
- 右侧实时代码编辑
- **红色 spinner 信号**：权限检查触发时自动变色，用户瞬间知道 agent 在等待  
- Stop/Pause/Cancel 按钮：固定在顶部工具条，非常显眼

**审批环节**  
- 三层权限模式：
  - `bypassPermissions`：信任高的用户，auto-run  
  - `auto`：ML 分类器判断安全/危险  
  - `acceptEdits`：每次都问  
- Anthropic 研究发现：新用户仅 20% 选择 auto-approve，但经过 750 次交互后，这个比例上升到 40%，**说明好的 UX 能让用户逐渐信任**

**失败 / 卡住时**  
- Agent 可以主动说"我需要更多信息"或"这个任务超出我的能力"  
- 用户可以中断并接手（context 完全保留）  
- Claude Code 最长运行时间从 25 分钟翻倍到 45+ 分钟（99.9 百分位），说明用户越来越放心让 agent 长期运作

**历史 / trajectory**  
- 完整的 session 回放  
- 可以在任意时刻"pause and review"

**设计洞察**  
权限审批的关键不在于"问得多"，而在于**问对地方**。Anthropic 发现 93% 的审批率会导致"审批疲劳"，解决方案不是加警告，而是**重构边界**——让 agent 在"安全的默认值"内自由运作。

---

### 2. Cursor Composer Agent

**进入路径**  
- VSCode 侧边栏出现"Composer"面板  
- 输入框 + 预定义的 5-10 个常见任务模板  
- 一键启动"agent mode"

**Agent 工作时**  
- Diff 视图（而非 inline blocks）  
- Known-safe URLs（如 Cursor 文档）自动批准，不提示  
- 可以用 `--yolo` 和 `--force` 标志跳过审批（给高级用户）  
- Sandbox mode：agent 可以安全地探索，删除文件被沙箱挡掉，降低用户心理负担

**审批环节**  
- 权限请求仅在非预期操作时触发  
- 企业管理员可配置团队级别的 git/network 权限

**失败 / 卡住时**  
- Sandbox 限制？用户可选 `--force` 重新运行  
- 清晰的错误消息 + 建议操作

**设计洞察**  
Cursor 的核心策略是**"安全的沙箱 + 信任的视觉信号"**。用户知道 agent 被限制在沙箱内，心理负担大幅降低。

---

### 3. Devin 2.0（Cognition Labs）

**进入路径**  
- Cloud-based IDE  
- 一个大的文本输入框："describe your task"  
- 然后是 Devin 的响应区

**Agent 工作时**  
核心特性：**Interactive Planning**  
- 用户输入 task 后，Devin 在 2-3 秒内回应：relevant files + 初步计划  
- 用户可以修改/确认计划，**确保 alignment 再让 agent 自动运行**  
- 解决的问题：用户不知道 agent 理解的是否正确，导致最后结果偏离预期

**失败 / 卡住时**  
- Multi-instance 架构：用户可以同时打开多个 Devin instances  
- 卡住的 task 丢给另一个 instance，继续并行工作  
- 人工审查 + 编辑 agent 输出的能力

**历史 / trajectory**  
- VSCode 风格的 UI，用户熟悉  
- 完整的工作日志回放

**设计洞察**  
**显式的"规划同步"环节**打破了长流程 agent 最常见的问题：用户发现 agent 理解错误时，往往已经跑了很远。Devin 的做法是**强制在大幅自动运行前做一次对齐检查**。

---

### 4. OpenAI Operator（ChatGPT Agent）

**进入路径**  
- ChatGPT web 界面的工具菜单  
- 选择"agent mode"  
- 输入任务

**Agent 工作时**  
- 浏览器窗口独立弹出（用户可以看到 agent 在干什么）  
- 实时解释：agent 做的每个操作都配有文字说明  
- Agent 可以填表、下单、预约等多步骤任务

**审批环节**  
- 操作前明确告知用户下一步动作（feedforward）  
- 某些操作（如支付）强制暂停等待用户确认

**失败 / 卡住时**  
- 用户可以在浮窗中直接接手  
- 浏览器上下文完全保留

**设计洞察**  
**可见性最大化**。让用户在一个独立窗口里看到 agent 的操作，比信息系统里的日志要直观得多。这是**降低用户焦虑**的有效方法。

---

### 5. Manus AI

**进入路径**  
- 熟悉的聊天机器人界面  
- 空白输入框  
- 用户评价："UX is what so many others promised... but this time it just works"

**Agent 工作时**  
- Agent 响应速度很快  
- 用户可以持续对话（follow-up 支持好）

**核心优势**  
- 简化到极致  
- 没有多余的 UI 元素

**设计洞察**  
**化繁为简的极致**。Manus 的成功不在于新功能，而在于把一个熟悉的界面做得极其流畅。

---

### 6. Lindy AI（工作流 Agent）

**进入路径**  
- 拖拽式工作流编辑器（no-code）  
- 模板库（预定义的常见流程）  
- 创建新工作流的路径清晰：trigger → condition → action

**Agent 工作时**  
- 可视化流程图  
- 每个 action 前都可以插入"human approval"节点  
- Code action 支持（Python / JavaScript）

**审批环节**  
- Human-in-the-loop controls  
- 可以在工作流任意位置插入 Slack/Email 审批

**失败 / 卡住时**  
- 工作流可以暂停 / 重试  
- 错误日志清晰

**设计洞察**  
**显式的审批节点设计**。Lindy 的做法是让用户在设计阶段就明确"哪些步骤需要人工决策"，而非在运行时临时加审批。

---

### 7. Replit Agent 4

**进入路径**  
- 新建项目时：输入框 + "Describe what you want to build"  
- Agent 立即问澄清问题（ideation phase）  
- 然后自动进入 design / build phase

**Agent 工作时**  
三阶段流程：
1. **Ideation**：Agent 问问题并制定计划  
2. **Design**：生成 UI 草稿（可以调整）  
3. **Build**：并行 subagents 构建不同模块

**失败 / 卡住时**  
- Visual Editor + Code 同时可见，用户可以手工调整  
- Figma import：设计 → 代码的自动转换  
- 用户可以在任何时刻"接管"某个子任务

**设计洞察**  
**阶段性的用户输入**而非"一次性输入后就不理"。Replit 让用户在关键阶段（plan 确认、design 审批）有清晰的介入点。

---

### 8. Cline / Roo Code（VSCode 扩展）

**进入路径**  
- VSCode 侧边栏集成  
- 一个聊天框  
- 快速命令：`@cline refactor this file`

**Agent 工作时**  
Roo Code 做了关键改进：
- **Architect mode**：先规划，不动代码  
- **Act mode**：执行计划  
- **Ask mode**：征求意见  
用户可以根据任务需要切换模式

**Diff 视图**  
- 修改被展示为 side-by-side diff  
- 比 Cline 的 inline blocks 清晰得多

**失败 / 卡住时**  
- Context menu 支持（Explain Code, Fix Issues）  
- 用户可以随时接手某个文件

**设计洞察**  
**Mode-based 设计**。Roo Code 让用户显式选择"我想要多少 agent 自由度"，而不是 agent 自己判断。

---

## 二、OfferGuide 的 UX 问题诊断

根据你的描述（home 页 13 个区块、navbar 19 个 tab），OfferGuide 正在犯**三个 anti-pattern**：

### Anti-Pattern 1：信息过载（Information Overload）

**症状**：
- Home 页显示 13 个区块（求职进度、职位库、简历模板、面试准备、salary compare 等）  
- 用户打开 app 时不知道该从哪里开始  
- 每个区块都在争夺注意力

**对标对比**：
- Claude Code：一个空白编辑器  
- Manus：一个输入框  
- Devin：一个大的 task 输入区域  
- OfferGuide：13 个区块

**后果**：
- 用户感觉在"管运维"而非"和 agent 对话"  
- 认知负荷过高

---

### Anti-Pattern 2：导航复杂性（Navigation Complexity）

**症状**：
- 19 个 tab  
- 用户需要多级菜单才能找到功能  
- 像是后台管理系统，不像是 agent 应用

**对标对比**：
- Claude Code：无导航菜单  
- Cursor：一个 Composer 侧边栏  
- OpenAI Operator：一个工具菜单  
- OfferGuide：19 个 tab

**后果**：
- 新用户上手困难  
- Agent 的"顺畅感"被繁琐的导航破坏

---

### Anti-Pattern 3：模糊的审批逻辑（Unclear Approval Flows）

**症状**：
- 用户不清楚"什么时候 agent 会自动执行，什么时候需要我确认"  
- 可能存在隐形的权限检查，让 agent 时而快时而慢

**对标对比**：
- Claude Code：权限模式明确（`bypassPermissions` / `auto` / `acceptEdits`）  
- Cursor：Known-safe URLs 自动批准  
- Devin：Planning 阶段显式对齐  
- OfferGuide：？

**后果**：
- 用户无法预期 agent 的行为  
- 信任度下降

---

## 三、5 个可落地的 UX Pattern（可直接偷用）

### Pattern 1：Hero CTA 极简化（"Golden Path" Design）

**做法**：
- Home 页只有一个主按钮："Start Job Search" 或 "Let me prepare your resume"  
- 其他功能隐藏在二级菜单或侧边栏  
- 首次用户被强制走这个 golden path  

**参考**：
- Claude Code（空白编辑器 + 输入框）  
- Manus（聊天输入框）  
- Devin（task 输入框）

**实现成本**：低（UI 重构）

---

### Pattern 2：权限模式可见化（Permission Mode Selector）

**做法**：
- 在 agent 启动前，让用户选择运作模式：
  - "Fully Autonomous"：Agent 在安全默认值内自由运作（如发送申请表）  
  - "Review Each Step"：Agent 的每个操作都要确认  
  - "Architect Only"：Agent 只制定计划，用户决定执行  

- 用户可以在 settings 中改变模式  
- **关键**：这个选择要显著、易找，让用户感觉在"控制" agent，而非被 agent 控制

**参考**：
- Claude Code（三层权限模式）  
- Roo Code（Architect / Act / Ask 三种模式）  
- Lindy（显式的 human-in-the-loop 节点）

**实现成本**：中（需要改权限系统）

---

### Pattern 3：Agent 思考过程可视化（Thinking Trace）

**做法**：
- 当 agent 在工作时，左侧流式显示它的"思考"（summarized thinking）  
- 用户看到 agent 在干什么，而不是黑盒运行  
- 这大幅**降低用户焦虑**

**参考**：
- Claude Code（thinking trace）  
- OpenAI Operator（实时操作说明）  
- Devin（planning 的可视化）

**实现成本**：中（需要 streaming UI）

---

### Pattern 4：Staged User Input（分阶段的用户决策）

**做法**：
- 不是"用户输入一次 → agent 跑完"  
- 而是在关键阶段强制停顿：
  - 阶段 1：用户给 task  
  - 阶段 2：Agent 提出计划，用户确认  
  - 阶段 3：Agent 执行  
  - 阶段 4：Agent 展示结果，用户可以反馈微调

**参考**：
- Devin 2.0（Interactive Planning）  
- Replit Agent 4（Ideation → Design → Build）

**实现成本**：中（需要改 agent 流程）

---

### Pattern 5：Failure Handoff 设计（优雅的失败转接）

**做法**：
- 当 agent 卡住时，显示明确的信号：
  - 红色 spinner（如 Claude Code）  
  - 或一个 tooltip："I'm stuck on [reason], can you help?"  
- 用户可以一键"接手"这个任务，context 完全保留  
- 不是让用户"重新开始"，而是"继续接力"

**参考**：
- Claude Code（spinner 变红 + user interrupt）  
- OpenAI Operator（浮窗接手）  
- Lindy（工作流暂停 + 人工介入）

**实现成本**：低-中（UI + 工作流改动）

---

## 四、5 个 Anti-Pattern 警告（OfferGuide 要避免）

### Anti-Pattern A：深层菜单结构

问题：用户需要点击 3-4 级菜单才能找到功能  
如何避免：设计原则是"一级菜单最多 5-7 项"，超过则分散到侧边栏或浮动面板  

---

### Anti-Pattern B：模糊的 Agent 状态信号

问题：用户不知道 agent 在干什么，是在思考、等待审批、还是卡住了  
如何避免：  
- 显式的 spinner / 进度条  
- 清晰的状态文字（"Reviewing your resume..." / "Waiting for your approval..."）  
- 声音反馈（可选）

---

### Anti-Pattern C：没有"接手"按钮

问题：Agent 卡住或出错时，用户无法中断并接手  
如何避免：始终保留一个显眼的 Stop / Cancel / Take Over 按钮  

---

### Anti-Pattern D：权限检查太频繁

问题：Approval fatigue —— 用户被问太多次确认，最后就不再看审批框，直接点通过  
如何避免：  
- 只在"真正危险"的操作时问（如发送邮件、删除文件）  
- "安全的默认值"内 agent 自由运作（如浏览职位库）  
- Anthropic 的数据：93% 审批率会导致疲劳；重构边界比加警告有效

---

### Anti-Pattern E：忽视"首次体验"

问题：新用户打开 app 看到 13 个区块，不知道从何开始  
如何避免：  
- 设计"onboarding flow"  
- 首次用户被强制走 golden path  
- 进阶功能隐藏直到用户主动寻找

---

## 五、"Agent-Like 顺畅感"的 3 个核心要素

基于调研数据，顺畅的 agent 体感由以下三个要素共同决定：

### 要素 1：极简的进入路径（Entry Point Minimalism）

**定义**：  
用户从打开应用到发出第一个 agent 指令，最多需要 2-3 步。

**实践数据**：  
- Claude Code：0 步（打开即可用）  
- Manus：1 步（点击输入框）  
- Devin：1 步（输入框）  
- OfferGuide：多步（导航 → 选功能 → 输入）

**Anthropic Autonomy 研究**的启示：  
新用户需要至少 50 个 session 才能习惯 agent 的工作方式。如果进入路径复杂，他们在前 5 个 session 就会放弃。

**OfferGuide 的做法**：  
- 删除 home 页 13 个区块，只保留 1-2 个主要的 CTA  
- 其他功能通过"settings"或侧边栏访问  
- 新用户的 onboarding 就是：输入职位搜索条件 → agent 开始工作

---

### 要素 2：隐形的权限管理（Invisible Permission Management）

**定义**：  
让 agent 在"安全的默认值"内自由运作，只在真正危险的操作时弹出审批。

**实践数据**：  
- Claude Code：三层权限模式，新用户选择最保守的，经过 750 sessions 后逐渐信任 agent  
- Cursor Composer：Known-safe URLs 自动批准，不提示  
- Lindy：用户在设计阶段明确"哪些步骤需要审批"，而非运行时临时加  

**核心原则**（Anthropic 研究）：  
> "93% 的审批率会导致审批疲劳。解决方案不是加警告，而是重构边界，让 agent 在安全的默认值内自由运作。"

**OfferGuide 的做法**：  
- 模式 1（Auto）：Agent 自动浏览职位库、生成申请材料、填表 → 不需要审批  
- 模式 2（Review）：Agent 每步都确认 → 需要审批  
- 模式 3（Plan Only）：Agent 只制定策略，用户决定是否执行  
- 用户可以在 settings 中选择，或针对具体 task 临时调整

---

### 要素 3：优雅的失败过渡（Graceful Failure Transition）

**定义**：  
当 agent 无法完成任务时，有清晰的"移交"机制，让用户无缝接手，而不是重新开始。

**实践数据**：  
- Devin 2.0：Multi-instance 架构，卡住的 task 丢给另一个 instance 继续  
- Claude Code：Context 完全保留，用户可以用自然语言纠正  
- OpenAI Operator：浮窗让用户直接接管浏览器  
- Lindy：工作流暂停 + 人工介入 + 继续运行

**关键洞察**：  
失败的 agent 不是"浪费用户时间"，而是"浪费用户信任"。好的失败设计能把"我需要你帮我"变成"我们一起来"。

**OfferGuide 的做法**：  
- 当 agent 申请职位失败时（如表单有陌生字段），显示：
  - "I couldn't auto-fill the 'Years of Experience' field. Can you provide it?"  
  - 用户输入一个值  
  - Agent 继续执行  
- 当 agent 卡住（如找不到合适职位）时，显示：
  - "I've searched 500+ jobs but couldn't find a good match for [your criteria]. Want to adjust the filters or I'll keep searching?"  
  - 用户可以调整条件或让 agent 继续

---

## 六、关键业界数据（来自 Anthropic 官方研究）

### 信任与自动化的关系

Anthropic 的 ["Measuring AI Agent Autonomy in Practice"](https://www.anthropic.com/research/measuring-agent-autonomy) 研究基于 **Claude Code 数百万次交互**：

| 用户 Session 数 | Auto-Approve 比例 | 含义 |
|---|---|---|
| < 50 sessions | ~20% | 新用户还在观察，需要多确认 |
| 100-200 sessions | ~25-30% | 开始有一些信任 |
| 500-750 sessions | ~35-40% | 明显的信任提升 |

**结论**：好的 UX 能加速信任建立。如果进入路径清晰、权限管理透明，新用户会更快地从"观察者"变成"委托者"。

### Agent 的实际使用场景分布

- 软件工程：49.7%  
- 后台自动化：9.1%  
- 营销文案：4.4%  
- 销售 / CRM：4.3%  
- 财务：4.0%  
- 数据分析：3.5%

**对 OfferGuide 的启示**：求职属于"高度定制化的后台自动化"领域，用户需要明确感知 agent 在做什么（不能黑盒运行）。

---

## 七、总结与落地建议

### 现状问题

OfferGuide 的"不够顺畅、不够 agent"源自三个设计决策：

1. **首页过满**：13 个区块试图展示所有功能，导致信息过载  
2. **导航过深**：19 个 tab 让用户像在用后台管理系统  
3. **权限逻辑不透明**：用户不知道什么时候 agent 会自动执行、什么时候需要确认

### 立即可做的改动（优先级）

**P0（这个月）**：
- 重设计 home 页：只留 1 个主 CTA（如"Start your job search"）  
- 其他功能移到侧边栏或设置  
- 用户看到的第一件事就是一个大的输入框

**P1（下个月）**：
- 实现权限模式选择器（Auto / Review / Plan Only）  
- 让用户在 agent 启动前知道自己选了什么  
- 在运行时用红色 spinner 等视觉信号表示"等待审批"

**P2（两个月）**：
- 加入"Staged User Input"：agent 制定计划后，让用户确认再执行  
- 优化失败 handoff：agent 卡住时有明确的"我需要你帮忙"的说法  
- 加入 thinking trace 流式显示（如预算允许）

### 最终指标

改动成功的标志：
- 新用户首次操作的路径长度：从 5+ 步 → 2-3 步  
- Auto-approve 比例：新用户从 < 20% 提升到 > 30%（说明信任在增加）  
- Session 平均长度：从 X 分钟 → X + 30%（用户更放心让 agent 长期运作）  
- 用户评价：从"不够顺畅"变成"这 UX 做得真不错"

---

## 参考资源

### Anthropic 官方研究

- [Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy)  
- [Claude Code auto mode: a safer way to skip permissions](https://www.anthropic.com/engineering/claude-code-auto-mode)  
- [Trustworthy agents in practice](https://www.anthropic.com/research/trustworthy-agents)  
- [Building Effective AI Agents](https://www.anthropic.com/research/building-effective-agents)

### 业界 Agent UX 设计指南

- [Secrets of Agentic UX: Emerging Design Patterns](https://uxmag.com/articles/secrets-of-agentic-ux-emerging-design-patterns-for-human-interaction-with-ai-agents)  
- [Designing for Autonomy: UX Principles for Agentic AI](https://www.uxmatters.com/mt/archives/2025/12/designing-for-autonomy-ux-principles-for-agentic-ai.php)  
- [Top 10 Agentic AI Design Patterns](https://www.aufaitux.com/blog/agentic-ai-design-patterns-enterprise-guide/)  
- [Designing For Agentic AI: Practical UX Patterns](https://www.smashingmagazine.com/2026/02/designing-agentic-ai-practical-ux-patterns/)

### 产品案例

- [Cognition - Devin 2.0](https://cognition.ai/blog/devin-2)  
- [Agent-Native Development: A Deep Dive into Devin 2.0](https://medium.com/@takafumi.endo/agent-native-development-a-deep-dive-into-devin-2-0s-technical-design-3451587d23c0)  
- [Cursor 2.0: Agent-First Architecture](https://www.digitalapplied.com/blog/cursor-2-0-agent-first-architecture-guide)  
- [OpenAI Operator - Introducing Operator](https://openai.com/index/introducing-operator/)  
- [Manus AI: An Analytical Guide](https://www.baytechconsulting.com/blog/manus-ai-an-analytical-guide-to-the-autonomous-ai-agent-2025)  
- [Lindy AI Review: No-Code Agent Platform](https://skywork.ai/blog/lindy-ai-review-2025-no-code-workflow-automation/)  
- [Replit Agent 4: Features & Use Cases](https://www.analyticsvidhya.com/blog/2025/04/devin-2-0/)  
- [Cline vs Roo Code: Best VS Code Extensions](https://www.qodo.ai/blog/roo-code-vs-cline/)  
- [Continue.dev: The Leading Open-Source AI Code Agent](https://continue.dev/)

### Agent Handoff & Failure Recovery

- [Escalation Design: Why AI Fails at the Handoff](https://www.bucher-suter.com/escalation-design-why-ai-fails-at-the-handoff-not-the-automation/)  
- [Chatbot to Human Handoff: Complete Guide](https://www.spurnow.com/en/blogs/chatbot-to-human-handoff)  
- [OpenAI Agents SDK - Handoffs](https://openai.github.io/openai-agents-python/handoffs/)

### 认知负荷与 UX 设计

- [Ultimate Guide to Cognitive Load Reduction in UX Design](https://developerux.com/2025/04/18/ultimate-guide-to-cognitive-load-reduction-in-ux-design/)  
- [Flows in the Age of Agentic AI](https://www.designative.info/2025/11/20/flows-age-agentic-ai-what-if-our-core-ux-models-no-longer-apply/)

---

**报告完成日期**：2026 年 5 月  
**推荐行动**：基于本报告的 P0 建议在 4-6 周内实施，预期可显著改善"agent-like 顺畅感"。
