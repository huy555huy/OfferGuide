# OfferGuide — 简历 + 面试 wedge 完整 kit

**用途**: 简历项目栏文案 + 5 个 30s 面试答案 + 1 周 dogfood 跟踪模板.
**目标公司**: 阿里通义 Agent / 字节 AML application / 美团 LLM 应用 / 腾讯混元应用 (hit rate 30-50%).
**别投**: DeepSeek / Minimax / 面壁 / Kimi RL Agent (要论文, 浪费时间).

---

## 1. 简历项目栏 (复制粘贴版)

### 中文版

> **OfferGuide — Ambient 求职 Agent (国内校招版)**
> 个人项目 · Python / FastAPI / SQLite · 25k LOC · 2026.4 - 至今 · GitHub: [...]
>
> - **自实现 Anthropic-style minimal harness**: single master loop + 13 specialized tools + memory tool (6 commands on file-based worldview markdown, agent 自维护). **拒绝 LangChain wrapper**, 借鉴 Cognition Devin / Manus 的 ambient 设计哲学.
> - **12 evolvable SKILLs** (Hermes-format prompts, Anthropic Agent Skills 标准兼容) + GEPA-style evolution scaffold + user feedback bridge (用户 thumbs up/down → evolution_signals → SKILL 长期进化).
> - **DeepSeek-V4 backend**: prompt-cache-aware cost tracking (~10% cache-hit rate, 60% input cost 节省) + per-day budget guardrail (防 agent 跑飞).
> - **Browser extension (Chrome MV3)**: 一键 sync BOSS 推荐池 N 个岗位到本地 agent — 用户在 BOSS 自己刷, agent 后台 score 排序 + 主动通知.
> - **国内独有**: 保 .docx 段落 style 的简历定向修改 (国内 Word 网投 ATS 必须), 不重写, 每条插入 claim 带 5-7 天 prep_plan ("把简历微调从虚假变成真学习").
> - **Dogfood**: 我自己 2026 暑期实习用它, 第 N 周累计评估 X 个 JD / 投 Y 个 / 拿 Z 个面试 (数据每周更新).

### English version (海外公司)

> **OfferGuide — Ambient Job-Hunt Agent for Chinese Campus Recruitment**
> Personal project · Python / FastAPI / SQLite · 25k LOC · GitHub: [...]
>
> - **Self-implemented Anthropic-style minimal harness**: single master loop + 13 specialized tools + memory tool with 6 file-based commands on agent-maintained worldview markdown. **Deliberately not a LangChain wrapper** — design philosophy from Anthropic's "Effective Harnesses for Long-Running Agents" + Cognition Devin / Manus ambient design.
> - **12 evolvable SKILLs** (Hermes-format prompts, Anthropic Agent Skills standard compatible) with GEPA-style evolution scaffold and user-feedback signal bridge.
> - **DeepSeek-V4 backend** with prompt-cache-aware cost tracking + per-day budget guardrail.
> - **Strategic positioning**: anti-auto-applier (the AIHawk → archived 2026/4 / LazyApply → Trustpilot 2.1★ category). Decision-aid + post-application tracking instead. Empty market in 国内 校招.
> - **Dogfooding for my own 2026 summer internship** since May 9 — N JDs evaluated / Y applied / Z interviews so far.

### 关键: 别写啥

❌ "Production-ready" / "Used by N users" — 0 用户
❌ "GEPA self-improving SKILLs evolved N times" — 0 evolutions
❌ "Auto-applied to N jobs" — 你战略放弃了
❌ "Increases reply rate by X%" — 没 baseline
❌ "864 tests passing" — 没人看, 留给面试时主动提

---

## 2. 5 个面试 30s 答案 (高频 + 杀手级)

### Q1: "你为什么不用 LangChain?"

> "LangChain agents 是 reactive turn loops with chains — 一连串 prompt 模板 + 状态管理.
>
> 我要的是 **ambient agent** — model in the driver's seat, file-based persistent memory, agent 自决 schedule_next_wake. 这是 Anthropic 公开 *Effective Harnesses for Long-Running Agents* 那篇博客的设计哲学, 跟 LangChain 完全反向.
>
> LangChain 适合 RAG QA 这种短任务. 求职 Agent 是长跑场景 — 7 天后回来查回应, agent 不能每次重头读 history. 所以我自实现 single master loop + worldview markdown 让 agent 自己写持久化笔记."

### Q2: "为什么不替用户点投递?"

> "AIHawk (29.6k star) 2026/4 被作者归档. LazyApply Trustpilot 2.1 星. **自动投递赛道全线退场**.
>
> 数据: 自动投递 reply rate < 2%, 精投 10-15%. 国内招聘 false-positive 成本不可逆 — 字节一年 2 次投递机会, 你瞎投一个就没了.
>
> 所以我把 AI 用在 search / draft / track, **decision 留给用户**. 这是**风险工程判断**, 不是保守 — false-positive 成本不可逆的场景, 不该把决策让给概率模型."

### Q3: "Anthropic harness 你借鉴了什么改了什么?"

> "**借鉴**:
> - Single master loop (no planner / executor / reflector chain)
> - File-based worldview markdown (agent 自维护) — 比 SQL 表更适合 agent 思考
> - 6-command memory tool (view / create / str_replace / insert / delete / rename) on `.offerguide/worldview/*.md`
> - Agent 自决 schedule_next_wake (`agent.run()` 内部调 tool)
>
> **改**:
> - DeepSeek 没 native context_management, 自实现 compaction (40K trigger) + tool-result clearing (15K trigger)
> - 加 GEPA evolution scaffold — 接 user feedback signals 让 SKILL prompt 长期进化, Anthropic 的 SKILL 是手写的
> - 国内特化校招日历 + docx 段落保格式 tailor"

### Q4: "GEPA 进化跑过几轮?"

诚实答 (don't bullshit):

> "**Scaffold 是搭好的** — schema 接通了, signals 从 inbox decide 流到 evolution_signals, 但**还没攒够 dogfood 数据跑系统进化轮次**.
>
> 我从这周 (May 9) 开始 dogfood, 计划 4 周累 30+ 投递数据后跑第一轮 score_match SKILL evolution, evidence 会进 evolution_log 表.
>
> 这是为什么我项目 status 写 'technical spike currently dogfooding' 而不是 'production'."

(主动 reframe: 不是缺陷, 是 explicit roadmap)

### Q5: "vs Career-Ops / Devin / Manus, 你独特点在哪?"

> "Career-Ops (santifer, 13.3k star) 是海外 — Greenhouse / Lever / Ashby 公开 API 让他 zero-touch 自动找岗. **国内没等价物**, 我得走"用户开 BOSS 我帮 sync"路径.
>
> Manus 复刻 / Devin 复刻在 2026 牛客面经里**已经批量出现** — 滑向 commodity. 我反 commodity 的 3 个点:
> 1. **国内校招特化** — 校招日历 hard-bake, agent 知道 5 月底是字节暑期截止
> 2. **不替用户决策** (上面 Q2 答案)
> 3. **docx 段落保格式微调** — 国内 Word 网投独有, career-ops markdown→PDF 路径在国内场景不适用"

---

## 3. 1 周 Dogfood 跟踪模板 (每天填一行)

把这个粘到 DEVLOG.md 或单独 docs/dogfood_log.md, **明天 (May 10) 开始填**:

```markdown
# Dogfood Log — Week 1 (2026-05-10 to 2026-05-16)

## 总指标 (周末写专栏前算)

- 评估 JD 数: ___
- 投递数 (实际上 BOSS / 牛客点投递的): ___
- BOSS 站内信回复: ___
- HR 邮件回复: ___
- 笔试通知: ___
- 1 面通知: ___
- 拿到 offer: ___
- 累计 LLM cost: $___
- 评估命中率 (我评的"投" 实际投 / 我评的"不投" 没投): __%

## 每日记录

### Day 1 — 2026-05-10
**今天投了**:
- [ ] 公司 X / 岗位 Y → OfferGuide 评分 N → 我决定: 投 / 不投 / 暂存
- [ ] ...

**OfferGuide 帮到的**:
- (这件事 ChatGPT 给不了)

**OfferGuide 没帮到 / 撞坑**:
- (写出来下次能修)

**Agent 主动行为**:
- 推过几次 inbox suggestion?
- 7d schedule_next_wake 触发了吗?
- 我接受 / 拒绝 / 忽略?

### Day 2 — 2026-05-11
...
```

---

## 4. 1 周后写专栏 (May 17 左右) — 标题 + 大纲

**标题候选** (按 click rate 排):

1. "**我自己写了个求职 Agent 找 2026 暑期实习, 第一周拿到 N 个 BOSS 回应**"  ← 最稳
2. "**为什么我不用 LangChain — 自实现 Anthropic-style ambient agent 找暑期实习的 7 天复盘**"
3. "**自动投递死了 — 我用反方向的 AI 求职 agent 找 2026 暑期, 第一周拿了 N 个面试**"

**大纲**:

```
1. 故事钩子 (200 字)
   - 我是上财应统专硕 27 届, 找 AI Agent 实习
   - 看了 career-ops / AIHawk 等项目, 决定反方向走
   - 自己撸了一个 ambient agent

2. 战略 (300 字) — 不投递只决策辅助 (Q2 答案)
   - AIHawk 归档 / LazyApply 2.1 星 数据
   - "false-positive 成本不可逆" 论点

3. 技术 (500 字) — Anthropic harness 借鉴 + 改 (Q3 答案)
   - 单 loop / worldview markdown / GEPA scaffold

4. 第一周数据 (400 字) — 关键, 必须有真数字
   - 评了 X 个 JD, 投了 Y 个, BOSS 回复 Z 个
   - 1 个具体例子: "字节 NLP Algo 实习"
     - OfferGuide 评分 X, 关键 gap: ABC
     - tailor_resume 加了 2 条, prep_plan 让我读 RAG paper
     - 我 5 天后投, BOSS 回 / HR 没回 / 进笔试
   - 总成本: $_

5. 反思 (300 字) — 哪些 work 哪些不 work
   - work: docx 保格式 / agent 主动提醒 followup
   - 不 work: BOSS 反爬, 浏览器扩展 selectors 偶尔漂

6. CTA (100 字) — Github + 招内推
```

发布渠道按优先级:
1. **知乎** (专栏 — 算法/求职 tag)
2. **即刻** (Live 2 条 — 数据 1 条 + 战略 1 条)
3. **小红书** (短版本 + 视频)
4. **V2EX** (展会 — 求职 tag)

---

## 5. 60-90s Dogfood 视频脚本 (B 站 + 小红书)

**钩子 (0-5s)**:
- "我自己写了个求职 Agent 找暑期实习, 来看一下它怎么帮我."

**Demo 流程 (10-70s)**:
1. (5s) 开 BOSS 推荐页
2. (5s) 浏览器扩展点一下 → 抓 N 个岗位
3. (10s) 切回 OfferGuide → /jobs 显示新岗位 + 自动 score 排序
4. (15s) 选一个高匹配的 → 评估报告 (评分 / gap / prep_plan)
5. (15s) /tailor 跑定向修改 → 显示 docx diff
6. (10s) 一键 "✓ 已投" → 7 天后 agent 提醒卡片显示
7. (5s) 看 cost dashboard (今天总花 $_, 比 ChatGPT plus 还便宜)

**结尾 (70-90s)**:
- "代码全开源 GitHub: huy555huy/OfferGuide"
- "在找 2026 暑期 AI Agent / LLM 应用岗内推 — 简介有联系方式"

---

## 6. 内推 + 招聘渠道 (针对 hit rate 30-50% 的目标)

按优先级:

1. **阿里通义 Agent 组** — `taohaoxiang.thx@alibaba-inc.com` (公开邮箱), JD 在 [CSDN 内推帖](https://blog.csdn.net/2301_78285120/article/details/146723727)
2. **字节 AML / 豆包 application 组** — Boss直聘 / 内推码 / TopSeed 计划
3. **美团 LLM 应用** — 牛客内推 / Boss
4. **腾讯混元应用** — 青云计划 / 同上
5. (**月之暗面 Kimi Agent**) — 备选, hit rate 偏低但 worth try, [Superlinear](https://www.superlinear.academy/c/collaborate/kimi-agent-team)

**别投**:
- ❌ DeepSeek 多模态 / 推理优化 (要顶会一作两篇)
- ❌ 智谱 ChatGLM 预训练 / 模型压缩
- ❌ 面壁 / Minimax / 阶跃 model 团队

---

## 一句话内化

**B- 档 → B+ 档**, 1 周内 1 件事:
**录视频 + 写专栏 + 填 dogfood 数据**.

不是攒 star, 不是加更多 SKILL, 不是改 UI.

**叙事 + 真数字 = 拿 offer**.
