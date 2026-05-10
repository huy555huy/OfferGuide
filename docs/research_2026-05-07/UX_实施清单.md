# OfferGuide Agent UX 改进实施清单

基于调研报告，这是一份可立即执行的改进清单。

---

## 第一阶段：快速胜利（1-2 周，UI 重构）

### Home 页面重设计

- [ ] **删除区块过载**
  - 删除首页 13 个区块中的 8-10 个
  - 只保留：
    - 主 CTA："Start Job Search" 或 "Prepare Your Application"
    - 一个次要 CTA："View Recent Activity"（显示最近的 agent 操作）
    - 底部快速链接：Settings / Help / Profile（4-5 项）

- [ ] **强化输入框**
  - 放大首页输入框到 "hero size"（类似 Claude Code / Manus）
  - Placeholder 文案要清晰："e.g., Find marketing jobs in NYC, prepare for interviews"
  - 一键示例：点击预定义的搜索（如"Data Science jobs near me"）

### 导航简化

- [ ] **Navbar 精简（从 19 个 tab → 5-7 个）**
  - Keep：Job Search / Applications / Resume / Interview Prep / Profile
  - Move to Sidebar：Analytics / Messages / Saved Jobs / Settings / Help
  - Or：用 tabbed interface，只显示当前功能的下级菜单

- [ ] **Sidebar 菜单重组**
  - 一级分组：Automation / Learning / Account
  - 每级最多 5 项
  - 用 collapsible 减少视觉混乱

### Agent 状态指示器

- [ ] **添加清晰的 agent 状态信号**
  - Idle：灰色 spinner  
  - Working："Searching jobs..." / "Filling application..."（带 animated icon）  
  - Waiting for approval：**红色 spinner**（关键改动）  
  - Stuck/Error：红色 X + 清晰的错误信息 + "Take Over?" 按钮

---

## 第二阶段：权限管理系统（2-3 周）

### 权限模式实现

- [ ] **在 settings 中添加"Agent Autonomy Level"选择**
  ```
  [ ] Full Auto (自动执行所有安全操作)
  [ ] Review Each Step (每步都问)
  [ ] Plan Only (只制定计划，用户决定)
  [ ] Custom (针对不同 action 自定义)
  ```

- [ ] **权限规则的"安全默认值"定义**
  - Safe（Auto-approve）：
    - 浏览职位库
    - 生成求职文本
    - 修改本地草稿
  - Risky（需要确认）：
    - 发送求职申请
    - 修改个人信息
    - 连接到外部账户
  - Very Risky（强制暂停）：
    - 删除数据
    - 发送邮件
    - 发起任何付款

- [ ] **在 agent 启动时显示权限确认**
  - 对话框："Agent will [action] with [autonomy level]. Confirm?"
  - 给用户一次改变心意的机会

### Approval Fatigue 防治

- [ ] **实现 ML 分类器（可选，进阶）**
  - Classify each agent action：safe / risky / very risky
  - Only ask approval for risky+
  - 参考：Anthropic 的 auto mode 使用 ML 分类器减少审批

- [ ] **Approval UI 最小化**
  - 不是弹出整个对话框，而是一个小的 card：
    ```
    [Agent is about to send job application to Acme Corp]
    [Approve] [Cancel] [Edit]
    ```

---

## 第三阶段：Agent 工作流优化（2-4 周）

### Interactive Planning（Devin 2.0 风格）

- [ ] **在 agent 执行前强制规划同步**
  - Step 1：用户输入搜索条件  
  - Step 2：Agent 返回 3-5 条搜索结果 + 计划  
    ```
    "Based on your input, I'll:
    1. Search for 'marketing jobs in NYC' (estimated: 2-3 min)
    2. Filter by salary > $80k
    3. Generate tailored resume for top 3 matches
    [Looks good?] [Adjust filters] [Cancel]"
    ```
  - Step 3：用户确认 → agent 执行

- [ ] **在重要决策点插入确认**
  - 发送申请前："Send application to [company]? (Expires in 10 min)"
  - 修改简历前："Update resume with [changes]?"

### Thinking Trace 显示（如有时间）

- [ ] **Agent 工作时左侧流式显示思考过程**
  - 不需要完整的 extended thinking（太复杂）
  - 只需要简化版的 thinking trace：
    ```
    Searching job boards...
    Found 127 matches for "marketing"
    Filtering by location...
    Top 3 matches identified
    Generating cover letter...
    ```

### Failure Handoff

- [ ] **Agent 卡住时的处理流程**
  - Detection：agent 运行 > 5 min 无进度 OR 明确的 error
  - Signal：红色 spinner + toast notification
  - Message："I'm stuck trying to [action]. Can you help?"
  - Options：
    - "Take Over"：用户接管，context 保留
    - "Adjust Filters"：让用户改变条件（不重新开始）
    - "Skip This"：跳过当前 action，继续下一个

- [ ] **Context Preservation**
  - 即使用户接管，agent 看到的所有信息都被保存
  - 用户修改后，agent 可以继续执行下一步（不是重新开始）

---

## 第四阶段：用户教育与监测（1-2 周）

### Onboarding Flow

- [ ] **新用户首次打开应该看到**
  1. 3-5 秒的动画：展示 agent 在做什么
  2. 一个简单的问卷：工作类型 / 经验水平 / 目标  
  3. 权限模式选择（默认："Review Each Step"，给新用户信心）
  4. 第一个示例 task（如"Find marketing jobs in NYC"）
  5. Agent 执行 + 用户观看 → agent 完成 → "Ready to try yourself?"

- [ ] **In-App Tips**
  - 首次使用某个功能时，显示 1-2 句的 tip
  - 可关闭，不要烦人

### 数据收集与监测

- [ ] **关键指标埋点**
  - Time to first agent command（从打开 app 到发送第一个指令）
  - Agent autonomy adoption：多少比例的用户选择 "Full Auto"
  - Approval fatigue：用户拒绝了多少 approval  
  - Handoff success rate：Agent 卡住时用户"Take Over"的比例
  - Session retention：3-day, 7-day, 30-day retention

- [ ] **用户反馈渠道**
  - 在 agent 完成一个 task 后，显示"How did that feel?" slider（1-10）
  - 低分时收集反馈："What didn't work?"
  - 高分时鼓励分享

---

## 优先级与预期时间

| Phase | Items | Effort | Timeline | Expected Impact |
|---|---|---|---|---|
| P0 | Home 页重设计 + Navbar 精简 | 1-2 week | Week 1-2 | High（立即减少认知负荷） |
| P1 | 权限模式 + 状态指示器 | 2-3 weeks | Week 3-5 | High（用户信任提升） |
| P2 | Interactive Planning + Handoff | 2-4 weeks | Week 5-9 | Medium（工作流优化） |
| P3 | Thinking Trace（可选） | 1-2 weeks | Week 9-11 | Low（锦上添花） |
| Ongoing | Monitoring + 文案优化 | 1 week | Week 1+ | Medium（持续改进） |

**总计**：约 6-9 周，建议并行执行 P0 + P1。

---

## 设计规范参考

### Color Signals

- **Idle Agent**：Gray (#6B7280)  
- **Working Agent**：Blue (#3B82F6)  
- **Waiting for Approval**：Orange / Red (#F59E0B / #EF4444)  
- **Success**：Green (#10B981)  
- **Error**：Red (#EF4444)

### Typography

- **Hero CTA**：24-28px, bold  
- **Agent Status**：14-16px, medium, animated  
- **Approval Card**：14px, clear hierarchy

### Interactive Elements

- **Stop/Cancel Button**：Always visible at top-right of agent panel  
- **Spinner**：2-3 second rotation, color changes based on state  
- **Progress Indicator**：如果 task 预计 > 1 min，显示 progress bar

---

## 成功指标（6 周后目标）

- [ ] Time to first agent command：从平均 > 3 步 → < 2 步  
- [ ] New user retention（D1）：从 X% → X% + 15%  
- [ ] User satisfaction（"Does agent feel smooth?"）：从平均 < 6/10 → > 7.5/10  
- [ ] Agent autonomy adoption：新用户 "Full Auto" 选择率 > 25%（vs. current < 10%）  
- [ ] Handoff success：Agent 卡住时用户成功接管的比例 > 80%

---

## 风险与缓解

### 风险 1：权限模式过度设计

**缓解**：Start simple（3 modes），不要一开始就做 "Custom" 选项。

### 风险 2：用户在 Interactive Planning 时迷茫

**缓解**：Agent 的计划要用最简单的语言，提供 2-3 个"Edit"选项（改变条件），而不是让用户自由输入。

### 风险 3：Thinking Trace 造成性能问题

**缓解**：可选的，不是 MVP。先做其他 4 个 phase，如有余力再加。

---

## 文案建议

### Agent Idle

```
"Ready to help you land your next role. What would you like me to do?"
```

### Agent Working

```
"Searching for [position] roles in [location]..."
"Generating your cover letter..."
"Submitting applications to 3 matching jobs..."
```

### Agent Waiting for Approval

```
"I'm about to send your application to Acme Corp. Look good?"
"Found 12 marketing roles. Should I review these for fit?"
```

### Agent Stuck

```
"I couldn't find your LinkedIn profile. Can you help me locate it?"
"The application form has a field I'm not sure how to fill: [Field name]"
```

### Agent Success

```
"Done! Applied to 3 jobs, generated 2 cover letters, and updated your resume."
"Want me to continue searching, or would you like to review these results first?"
```

---

**最后的建议**：逐步推出这些改动，每周测试一个 phase。用 A/B test 验证改动的效果（例如，Home 页简化是否真的降低了新用户入门时间）。

祝你们的 OfferGuide 变得更"agent-like"！
