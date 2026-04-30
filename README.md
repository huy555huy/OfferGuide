# OfferGuide

**国内校招 Ambient 求职 Copilot + GEPA-style Self-Evolution.**

> 不点投递按钮——做精投决策、投后跟踪、面试备战；用户 dogfood 数据驱动 SKILL 自动进化。

![dashboard](docs/screenshots/dashboard.png)

| Chat (评分 + 差距 + 备战) | Dashboard (进化历史 / 漏斗) |
|---|---|
| ![chat](docs/screenshots/chat_report.png) | ![home](docs/screenshots/home.png) |

---

## Why this, and not another auto-applier

业内的 AI 求职 agent ([AIHawk](https://github.com/feder-cr/Jobs_Applier_AI_Agent_AIHawk) 29.7k★、
[ApplyPilot](https://github.com/Pickle-Pixel/ApplyPilot)、
[get_jobs](https://github.com/loks666/get_jobs) 6.8k★ Java) **全是自动投递派**。
但数据显示自动投递是死路：

- **97% 公司用 AI 驱动 ATS** 过滤简历（[来源](https://boterview.com/a/ai-recruitment-statistics)）
- **49% 自动 dismiss AI 写的简历**（[来源](https://www.gettailor.ai/blog/ai-resume-detection)）
- LazyApply Trustpilot **2.1 星 / 52% 最低分**；某用户投 14000 份只收到几百个 skills-mismatch 拒信
  （[来源](https://www.trustpilot.com/review/lazyapply.com)）

OfferGuide 反方向走：**不点投递**，做真正提高 reply rate 的事——精准匹配、定向微调
（不重写）、投后跟踪、面试备战——并用用户自己的 dogfood 数据通过 GEPA **自进化**
agent 的 SKILL prompt。

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Conversational Agent (LangGraph)               │
│                                                                     │
│   evolvable SKILLs (Hermes-style SKILL.md):                         │
│     ★ score_match       — 校准过的匹配概率 + 多维 reasoning          │
│     ★ analyze_gaps      — 关键词差距 + 微调建议（带 AI 检测风险）      │
│     ★ prepare_interview — 公司画像 + 题目预测 + 备战重点               │
│                                                                     │
│   utility SKILLs:                                                   │
│       update_status     — 应用状态机                                 │
│       query_history     — 历史检索                                   │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
┌──────▼─────┐       ┌───────▼────────┐
│  Workers   │       │  Inbox + UI    │
│            │       │  (HITL queue)  │
│ Scout:     │       │                │
│  · 牛客    │       │ FastAPI + HTMX │
│  · 浏览器  │       │                │
│    扩展    │       │ 飞书 webhook   │
│            │       │ Telegram bot   │
│ Tracker:   │       │                │
│  · 状态机  │       │                │
│  · 沉默检测│       │                │
│  · 7/14/30d│       │                │
└──────┬─────┘       └────────┬───────┘
       │                      │
       └──────────┬───────────┘
                  │
         ┌────────▼────────┐
         │ Memory          │
         │ (local-first)   │
         │                 │
         │ SQLite +        │
         │ sqlite-vec      │
         │                 │
         │ application_    │
         │   events log    │
         │ skill_runs      │
         │ evolution_log   │
         │ inbox_items     │
         └─────────┬───────┘
                   │
        ┌──────────▼─────────────┐
        │ Self-evolution layer   │
        │                        │
        │ DSPy GEPA              │
        │ (ICLR 2026 Oral)       │
        │                        │
        │ Golden trainset        │
        │   + 3-axis metric:     │
        │     · prob in band     │
        │     · keyword recall   │
        │     · anti-FP          │
        │                        │
        │ Writes evolved         │
        │ SKILL.md, parent .bak  │
        │ evolution_log row      │
        └────────────────────────┘
```

### Key design decisions

| Decision | Choice | Why (with source) |
|---|---|---|
| Multi-agent? | **Single LangGraph agent + tools** | [Anthropic](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them): "start simple, multi-agent costs 3-10x tokens" |
| Skill format | **Hermes SKILL.md** (design only, not runtime) | [Hermes Agent](https://github.com/nousresearch/hermes-agent) ICLR 2026 Oral, MIT |
| Self-evolution | **DSPy GEPA** | [GEPA paper](https://arxiv.org/abs/2507.19457) ICLR 2026 Oral; cheap (~$2/run, "auto=light") |
| Vector store | **sqlite-vec** (single-user) | local-first, zero ops; Qdrant overkill at our scale |
| LLM | **DeepSeek V4** main + reflection | OpenAI-compat API, China-friendly |
| Notifier | **飞书 webhook + Telegram bot** dual-rail | Server酱 5 条/天硬限制不够用 |
| HITL | **Inbox queue (SQLite)** rather than `interrupt()` | Async-friendly, easier to reason about |
| Boss 接入 | **浏览器扩展 (Manifest V3, click-to-extract)** | Boss ToS 不允许后台爬，扩展 inject 是合规的；**默认不自动发送** |

---

## Self-evolution loop (the resume pitch in one paragraph)

`score_match / analyze_gaps / prepare_interview` **三个** SKILL 都接入了 GEPA 进化基础设施，
通过 `evolution/adapters/` 的 adapter 模式插入；想加第四个 SKILL，只需新建一个 `adapters/<skill>.py`
+ 注册到 REGISTRY。

进化器是 [DSPy GEPA](https://dspy.ai/api/optimizers/GEPA/overview/) (Genetic-Pareto Prompt
Evolution, ICLR 2026 Oral)——no gradient, no fine-tuning, 比 GRPO 强 6-20%, rollout 减少 35x。

每个 adapter 都提供：

1. **手工 golden trainset**（按 SKILL 不同有不同字段）：
   - `score_match`: 10 例，覆盖 fit/misfit/middle 三档（probability_range + must_mention + must_not_mention）
   - `analyze_gaps`: 7 例，real + edge_case 两档（expected_keywords + ai_risk_floor + count range）
   - `prepare_interview`: 6 例，with_面经/no_面经/edge_case 三档（profile_keywords + jd_keywords）
2. **多轴 metric**，axes 因 SKILL 而异：
   - `score_match`: 0.5 × prob_in_band + 0.3 × recall + 0.2 × anti_FP
   - `analyze_gaps`: 0.40 × keyword_recall + 0.30 × schema_validity + 0.15 × ai_risk_floor + 0.15 × count_range
   - `prepare_interview`: 0.30 × grounded + 0.25 × category_coverage + 0.20 × schema + 0.15 × calibration_spread + 0.10 × count
   每个轴都返回 human-readable feedback 给 GEPA 的 reflection LM
3. **进化产物**——一行 CLI 跑出新版 prompt，写回 `SKILL.md`，旧版自动 `.bak` 备份，
   所有指标 delta 入 `evolution_log` 表

```bash
# 进化任意一个 SKILL
$ DEEPSEEK_API_KEY=sk-... python -m offerguide.evolution evolve score_match
$ DEEPSEEK_API_KEY=sk-... python -m offerguide.evolution evolve analyze_gaps
$ DEEPSEEK_API_KEY=sk-... python -m offerguide.evolution evolve prepare_interview --auto medium

# 看进化前后 prompt diff + 指标对比（适合贴博客 / README）
$ python -m offerguide.evolution diff score_match --markdown > evolution.md
```

`diff` 命令输出长这样（这就是简历里 "GEPA 进化前后对比 [TBD-4]" 的可视化证据）：

```markdown
# `score_match` — GEPA Evolution Report

- **Parent version**: `0.2.0`
- **Evolved version**: `0.2.1`

## Metric — total

| baseline | evolved | Δ |
|---|---|---|
| 0.503 | 0.724 | **↑ +0.221** |

## Per-axis breakdown

| axis | baseline | evolved | Δ |
|---|---|---|---|
| anti | 0.500 | 0.612 | +0.112 |
| prob | 0.487 | 0.703 | +0.216 |
| recall | 0.521 | 0.842 | +0.321 |
| total | 0.503 | 0.724 | +0.221 |

## Prompt body diff

```diff
- 你是一名严谨的中文校招求职顾问，背景是统计学。
+ 你是一名严谨的中文校招求职顾问，背景是统计学。**当用户简历明确缺少 JD 列出
+ 的硬性技能时，你必须把这些缺失列入 deal_breakers 而不是仅在 reasoning 里提到**
...
```

---

## Quick start

```bash
# 1. install
pip install -e ".[dev,ui,evolution,scheduling]"

# 2. set up
export DEEPSEEK_API_KEY=sk-...
export OFFERGUIDE_RESUME_PDF=/path/to/your_resume.pdf

# 3. quickstart — exercises every layer through W8 (offline by default)
python examples/quickstart.py /path/to/your_resume.pdf
# add --invoke-skills to actually call the LLM
# add --invoke-agent to also build the LangGraph and run action='everything'

# 4. start the conversational UI
python -m offerguide.ui.web  # http://localhost:8000

# 5. workers (cron candidates)
python -m offerguide.workers tracker run                   # 沉默扫描
python -m offerguide.workers scout nowcoder --limit 50     # 牛客 sitemap

# 6. evolve a SKILL (any of the 3 — adapters/ supports all)
python -m offerguide.evolution evolve score_match --auto light
python -m offerguide.evolution evolve analyze_gaps
python -m offerguide.evolution evolve prepare_interview

# 7. before/after report
python -m offerguide.evolution diff score_match
python -m offerguide.evolution diff score_match --markdown > evolution.md
```

### Boss browser extension

```bash
# Chrome → chrome://extensions → 开发者模式 ON → 加载已解压的扩展程序
#         指向本仓库的 browser_extension/ 目录
```

打开 Boss直聘 JD 页面 → 点击 OfferGuide 扩展图标 → 自动从页面提取（无 content_script，**点了才提取**）
→ 确认 → 发到本地 `http://localhost:8000/api/extension/ingest` → 入 jobs 表。

---

## Repo layout

```
src/offerguide/
├── agent/                # LangGraph 单 agent + state
├── application_events.py # 应用事件日志（W5'）
├── state_machine.py      # event kind → applications.status
├── config.py             # env-driven Settings
├── evolution/
│   ├── cli.py            # python -m offerguide.evolution {evolve,diff}
│   ├── runner.py         # SKILL-agnostic GEPA 编排 (W8' refactor)
│   ├── adapters/         # one module per evolvable SKILL:
│   │   ├── _base.py      #   - generic MetricBreakdown, aggregate
│   │   ├── score_match.py     # 10 examples + 3-axis metric
│   │   ├── analyze_gaps.py    # 7 examples + 4-axis metric
│   │   └── prepare_interview.py # 6 examples + 5-axis metric
│   ├── golden_trainset.py # back-compat shim → adapters/score_match
│   ├── metrics.py        # back-compat shim → adapters/_base + score_match
│   ├── dspy_module.py    # SkillSpec → dspy.Signature
│   └── diff.py           # 进化前后对比报告
├── inbox.py              # HITL 队列
├── interview_corpus.py   # 面经 RAG
├── llm/                  # DeepSeek V4 OpenAI-compat httpx client
├── memory/               # SQLite + sqlite-vec
├── platforms/            # nowcoder / manual / boss_extension
├── profile/              # PDF 简历解析
├── skills/
│   ├── score_match/      ★ evolvable
│   ├── analyze_gaps/     ★ evolvable
│   └── prepare_interview/ ★ evolvable
├── ui/
│   ├── web.py            # FastAPI + HTMX
│   └── notify/           # 飞书 / Telegram / console
└── workers/
    ├── __main__.py       # python -m offerguide.workers {tracker,scout}
    ├── scout.py          # 牛客 sitemap crawler + ingest
    └── tracker.py        # 沉默检测 + 状态机 + 提醒

browser_extension/        # Manifest V3 Chrome 扩展（Boss 页面提取）
docs/                     # strategy + architecture
tests/                    # 290+ tests, all green
```

---

## Status

- [x] **W1** — scaffold + SKILL.md loader + memory + profile + agent skeleton
- [x] **W2** — Scout v1 (牛客 sitemap, manual paste) + `score_match` SKILL v1
- [x] **W3** — `analyze_gaps` SKILL（带 AI 检测风险标注）
- [x] **W4** — Conversational agent + Inbox + 飞书/Telegram 双轨通知
- [x] **W5'** — application_events 日志 + 严格 SKILL 输入规范化 + extras_json 修复
- [x] **W6** — GEPA 进化基础设施（11-case golden trainset + 3-axis metric + DSPy 模块 + writeback CLI）
- [x] **W7** — Tracker worker（应用状态机 + 7/14/30d 沉默检测）+ Boss 浏览器扩展 v1
- [x] **W8** — `prepare_interview` SKILL + `evolution diff` CLI + README 重写
- [x] **W8'** — Generalize GEPA to all 3 SKILLs (adapter pattern); wire `prepare_interview`
       into the agent (`graph.py` prep_node + `interview_corpus` retrieval); workers CLI
- [ ] **dogfood** — 4 周持续投递收集真实 reply rate 数据；跑首次 GEPA 真活；填 `[TBD]` 数字

### What's still TBD（需 dogfood 数据）

- [TBD-1] 真实 reply rate baseline (W1-W2) vs 进化后 (W3+)
- [TBD-2] 面试题命中率（prepare_interview 预测 vs 实际）
- [TBD-3] match_score 校准曲线（calibrated probability vs 实际 reply rate）
- [TBD-4] **进化前后 prompt diff** —— `python -m offerguide.evolution diff score_match`
  跑出来贴在这里
- [TBD-5] 单次 GEPA 运行成本（预计 $2-10）

---

## License

MIT. See [ATTRIBUTION.md](ATTRIBUTION.md) — OfferGuide 借鉴 Hermes Agent 的 SKILL.md 设计、
使用 DSPy GEPA 进化算法，均 MIT。
