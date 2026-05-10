# Agent Search Seed Keywords — 真 Dogfood Run (2026-05-11)

W18 加了 seed_keywords 注入 agent_search 的 north_star, 当时只 unit test
没真跑. CLAUDE.md 第 6 节列为"未 dogfood 验证". 这次 sandbox 里**真跑了
一轮 LLM** (DeepSeek-V4-Flash + Tavily + 用户真简历), 把真值记下.

## 跑参数

- model: deepseek-v4-flash
- north_star: "AI Agent / LLM 应用 暑期实习 校招 岗位"
- seed_keywords: 用户真简历抽出的 5 个 niche keyword
  - 大模型微调 / Deep Research / RLHF / Diffusion 模型 / HuggingFace
- 调用: `_run_agent_search_blocking(store, settings, seed_keywords=[...])`

## 真结果

- 用时: 174.8s
- iterations: 19 (LLM 多轮 ReAct)
- cost: **$0.0078**
- inserted: 41 jobs
- skipped_dup: 0
- agent 跑的 search queries (前 8 条):
  1. AI Agent LLM 大模型微调 暑期实习 校招 2025 创业公司
  2. AI创业公司 大模型微调 RLHF 校招实习 2025 招聘
  3. AI Agent 创业公司 校招 暑期实习 Diffusion模型 HuggingFace 2025
  4. **DeepSeek 深度求索 校招 实习 LLM 大模型 2025 招聘**
  5. **智谱AI 百川智能 MiniMax 月之暗面 校招 实习 AI Agent 大模型 2025**
  6. (其余 3 条)

## 真 ingest 公司分布

| 公司 | 数 | 来源 |
|---|---|---|
| 腾讯 | 20 (10 校招 + 10 社招) | tencent_campus / tencent_social verified API |
| 百度 | 20 (校招 + INTERN 实习) | baidu_campus verified API |
| **上海麟鲤科技** | 1 | **agent_search via wondercv.com 聚合招聘页** |

**大厂 40 / 中小厂 (niche) 1**

## 真验证结论

✅ agent_search 真用 seed_keywords —— search queries 里出现了 "大模型微调"
   "RLHF" "Diffusion 模型" "HuggingFace"

✅ agent 真去找 niche 公司 —— 主动搜了 "DeepSeek/智谱/百川/MiniMax/月之暗面"
   这种用户简历直接命中的方向

⚠️ **实际 ingest 中 niche 比例只有 1/41** (麟鲤科技 via wondercv 聚合).
   原因 (verified, 不假设):
   - LLM 调 `search_verified_official_jobs(keyword='AI Agent')` 直接拉到
     腾讯+百度 verified API 数据 40 条 — 高优先级, agent 自然倾向用这个
     (因为不会失败, 一次拉 5-10 条)
   - 真 web search 路径 (web_search → fetch_url → extract_and_ingest_jd)
     效率低: niche AI 创业公司招聘页大多 SPA / 需登录, fetch_url 拿不到
     JD 详情. agent 只成功 1 次 (麟鲤科技, 因为 wondercv.com 是聚合 SSR
     页, 不是公司自家 SPA)

## 真启示

niche AI 创业公司的覆盖**不能靠 agent_search web search 路径**, 应该:

1. **靠 0voice 聚合 repo** — W19+ 已接, 一次拉 475 真岗 / 189 AI 相关,
   含商汤 55 / 智元机器人 10 等 niche 公司 (见 W19+ 0voice commit)
2. **加 wondercv 等聚合招聘网站 adapter** — Tier 3 todo, 因为这次发现
   wondercv 是 SSR 页 fetch_url 能拿到结构化数据, 跟 0voice 类似
3. **agent_search 真有用的场景**: 找特定公司近期发的新岗(比如"商汤 2026
   春招新岗"), 不适合做主流量 ingest

不假装 "agent 自动找 niche 工作得很好", 真值是:
- agent_search 是补充源 (1-3 条/cycle), 不是主力
- niche 主力是 0voice 聚合 repo + verified_official API

## 下一步 (W20 候选)

- [ ] wondercv.com SSR 页 adapter (类似 zerovoice.py)
- [ ] AgentSeek / 实习僧 / 牛客实习广场 SSR 抓取 (实习专用聚合)
- [ ] 让 ambient daemon 把 search agent 的 max_iterations 调小 (现在 19
      iter 是 LLM 自由跑, 1 cycle 174s; 强约束到 5 iter / 30s 更适合定时)
