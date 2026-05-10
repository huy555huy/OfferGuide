# OfferGuide — Claude 工作守则

## 0. 最高优先级铁律: 没有调查, 没有发言权, 也没有写代码权

**写代码前任何事实假设必须先用真 source 验证. 包括但不限于:**

- **数据库字段 / 表名 / SQL** → 必须 `grep "CREATE TABLE" src/` 看真 schema, 不能凭"应该叫这个名"写
- **API 参数 / 函数签名** → 必须 Read 源码看 `def foo(...)`, 不能凭函数名猜参数
- **SKILL inputs/outputs** → 必须 `cat src/offerguide/skills/<name>/SKILL.md` 看 `inputs:` YAML + `output_schema`. **不要假设字段叫 `score` / `key_gaps`**, 真实字段是 `probability` / `deal_breakers`
- **DOM selector** (BOSS / 牛客 / 任何网站) → 必须有真 outerHTML 样本, 不能从 search 拼 chain
- **URL 模式 / 路由** → 必须真 fetch 看 redirect, 或让用户帮抓真 URL
- **平台限额 / 反爬规则** → 必须官方 help 页 / 用户实测的数字, 不能从 random 博客拼
- **浏览器 API 兼容性** → 必须 MDN 查或在用户真环境 verify
- **远程 API 是否真能调** → 用 `python -c "import httpx; print(httpx.get('...'))"` 真打一次, 不能凭"应该能用"

**研究先行**:
- 不知道某事 → 输出"TBD - 需要 [具体动作] 验证", **不要写代码假装知道**
- 工程要做但事实未知 → 把"事实采集"作为第一步交付 (例如 DOM probe), 不要在采集前就写依赖该事实的逻辑
- "差不多应该是这样" = 90% 概率错, 等于浪费用户时间 + 污染代码

**最狠的两条 (W19+ 用户骂第 8 次后加)**:

A. **凡是心里说"应该"的, 不能说出口, 必须先 verify**.
   - "应该没 key" → `python -c "from offerguide.config import Settings; print(Settings.from_env().deepseek_api_key)"` 真看
   - "应该 grep 一下能看到" → grep 出来空 ≠ 没有, 可能名字不一样
   - "sandbox 应该不能" → 真试一次, 失败了再说
   说出"应该"的句子 = 还没 verify 的判断. 把它降级成"我 try 了 X, 失败了, 所以我推断 Y, 但我没 100% 确认".

B. **环境/配置/前提条件 也是事实假设**.
   - 之前 7 类清单 (DB 字段 / API 参数 / SKILL inputs / DOM / URL / 限额 /
     浏览器 API) 漏了"环境变量是否配了, 名字叫什么". 
   - 加第 8 类: **环境变量 / 配置项 / 系统资源 (LLM key, PDF 路径, 端口
     占用, 磁盘空间)** — 必须 `python -c` 或直接读才能说是事实
   - "**显然的事就是假设的温床**" — 越觉得显然越要 verify

**违反时 = 对用户的伤害**: 用户已经压力大要找工作. 我每写一行假设的代码, 他都得花时间验证 / 报错 / 等下个版本修. **比不写还差.**

**血淋淋的反面教材** (从 W14 起到 W15.22):
- 假设 score_match SKILL 输入是 `job_title/jd_text/candidate_resume` → 真实是 `(job_text, user_profile)` → SkillRuntime ValueError 静默吞了 → 自 W14 起所有 ambient flow 一行 LLM SKILL 都没调成功过 → 整个 W15 hero flow 是空壳
- 假设 `events` 表存在 → 真实只有 `harness_events` → endpoint 写库 silent 失败
- 假设 BOSS chat selector 是 `.chat-im-wrap` → 没真 inspect 过, 全是搜出来的猜
- 假设 paste JD 是主流程 → 用户原话: "找岗位要全自动化, 别让用户一点点的自己去找, 不然哪有这个时间和精力"

每一条都是 "没调查就动手" 的代价.

## 1. 项目背景

OfferGuide — 上海财经大学应用统计专硕 2027 届的求职 copilot. 目标 2026 暑期 AI Agent / LLM 应用岗实习 (5 月节奏 = 应届暑期实习, 现在投, 6-9 月入职, 10-11 月评估转正).

**真主流程** (用户多次澄清):

```
agent[全自动找岗位, 多源] → /recommended[排好序] → 用户挑 →
[投递包: 微调简历 + 自我介绍 + 网申 QA + 内推码] →
用户去官网/BOSS 投 → 点"我投了" →
[投后包: 公司画像 + 高频题 + 学习清单 + 模拟面试]
```

**paste JD 只是应急 fallback, 不是主流程.**

**半自动**, 不全自动: agent 写所有文案/材料, 用户审核 + 决策 + 按发送. 永不替按发送.

## 2. 求职阶段术语 (区分清楚, 不混)

| 类型 | 时间 | 转正 path | 用户当前优先级 |
|---|---|---|---|
| **暑期实习** (应届实习) | 5-9 月入职 | 是, 9-10 月评估 | ⭐ 最高 (5 月节奏) |
| **日常实习** | 任何时间 | 不一定 | 中 (填 5 月空闲) |
| **校招正式** | 9-10 月秋招开 | n/a | 低 (现在还早) |
| **社招** | 任何时间 | n/a | ❌ 不投 (要工作经验) |

平台 enum (verified 2026-05-10, 不假设):
- 腾讯校招 `join.qq.com` `projectName='应届实习'` 当前 596 条全是这一个
- 百度 `talent.baidu.com/jobs/list?recruitType=INTERN` 给 `'暑期实习项目' / '日常实习项目'`
- 百度 `?recruitType=GRADUATE` 给 `'校招' / 'AIDU项目' / '管培生项目'`
- nowcoder title 含"暑期实习"明标, 没标 "实习" 默认日常

`src/offerguide/recruit_type.py` 是 deterministic classifier; 别让 LLM 干这事.

## 3. 已 verified 真实可用的官方源 (W16+)

| 源 | URL / 路径 | 状态 | 适用 |
|---|---|---|---|
| 牛客 | sitemap chain → `/jobs/detail/<id>` | ✅ 公开, parse `__INITIAL_STATE__` | 校招/实习 (混) |
| 腾讯校招 | `join.qq.com/api/v1/position/searchPosition` | ✅ 公开 JSON API | 应届实习 |
| 腾讯社招 | `careers.tencent.com/.../api/post/Query` | ✅ 公开 JSON API | 社招 (应届生不该投) |
| 百度校招 | `talent.baidu.com/jobs/list?recruitType=GRADUATE` | ✅ SSR `__INITIAL_DATA__` | 校招/AIDU |
| 百度实习 | `talent.baidu.com/jobs/list?recruitType=INTERN` | ✅ SSR `__INITIAL_DATA__` | 暑期+日常实习 |
| 字节 / 阿里 / 美团 / 小红书 | — | ⚠️ `unverified_js_shell` / `login_required` | 不能假装支持 |
| BOSS 直聘 | — | ⚠️ `browser_session_required` | 走 extension, 不能 server scrape |

`SOURCE_LANDSCAPE` enum 在 `src/offerguide/platforms/official_jobs.py` 强制不能假装支持没验证过的源.

## 4. 模型 / Cost

- 锁定 **DeepSeek-V4** (用户成本约束, 不能用 Claude API)
- 日 cost cap $5 (`src/offerguide/llm/budget.py`)
- DeepSeek 自动 prompt cache (cache_hit_tokens 算 ~10% 价)

## 5. 工作流约束

- **commit message 不带** `Co-Authored-By Claude` / `🤖 Generated`. 这是用户简历项目, 写他自己的.
- 触发 plan mode 之前先想清楚: 这是真的需要 plan 还是又在拖? auto-mode 状态默认动手不规划.
- 写测试: 真测端到端 (用 stub LLM 跑), 不要只写 mock-mock-mock 的浮夸覆盖.
- 不喊"完美/全套/突破", 不报"已完成/已通关". 用户对这种话敏感.
- 不写大于必要的 README / docs. 用户没要求别加 markdown.

## 6. 当前已知未解决的事 (真清单, 不假装通关)

- **BOSS 沟通框 selector** — 没真 DOM 样本; W15.21 加了 probe 钩子等用户
  抓. 用户**没用过**, 收 0 样本. 实际上 W18 把主流程移到 OfferGuide
  /apply-pack, BOSS 浮窗变成可选的 ambient browsing 模式 — probe 是"如果
  用户某天想在 BOSS 直接做"的兜底, 不该作为主路径承诺
- **大厂官网半自动投递** — 反爬 + form 字段动态, 用户自己 5 分钟点完官网
  比让 agent 去精准. application_plan.py 区分 BOSS / 牛客 / 官网 / 未知
  4 种, 每种给 deterministic 步骤 + 字段填充指引, 但**不点发送**
- ~~**字节 / 阿里 招聘官方源**~~ — W19+ 已做:
  - **字节**: 实测 `jobs.bytedance.com/api/v1/search/job/posts` 是真公开 JSON
    API (1334+ 岗位), 加 `search_bytedance_jobs()` adapter. 但 API 全是
    `recruit_type='正式'` (社招), classifier 标 SOCIAL, 应届实习走 0voice
    间接. SOURCE_LANDSCAPE 状态从 unverified_js_shell → verified_public_social_only
  - **阿里**: 没找到直 API, 通过 0voice repo 间接拿到 124 个真
    campus-talent.alibaba.com ATS URL, SOURCE_LANDSCAPE 状态从
    unverified_js_shell → verified_via_aggregator
- ~~**agent_search seed_keywords 真去找 niche 公司了吗?**~~ — W19+ 已
  dogfood 验证 (2026-05-11, 见 docs/dogfood_2026-05-11/agent_search_seed_keywords.md):
  agent **真用了** seed_keywords, search queries 含 "DeepSeek/智谱/月之
  暗面" niche 关键词, 但实际 ingest 中**中小厂只有 1/41** (麟鲤科技 via
  wondercv). 真原因: niche AI 创业公司招聘页多 SPA / 反爬, web_search
  路径效率低. 主力 niche 覆盖应靠 0voice 聚合 repo (W19+ 已接). agent_search
  当补充源 (1-3 条/cycle), 不是主力流量.
- ~~**application_plan 还没识别 baidu_intern / agent_search 源**~~ — W19+
  已做: 加了 host-based 大厂细化 (腾讯校招/社招/百度实习/百度校招/字节/
  阿里/美团/agent_search) 共 8 条具体投递路径, 每条含登录方式 + 内推码
  提示 + 简历附件方式 + 业务线注意事项
- ~~**0voice GitHub 校招 repo 聚合**~~ — W19+ 已做: server-side fetch
  README → parse markdown 表格 → 475 真岗 (189 个 AI 相关) ingest, 含
  阿里 124 个 campus-talent.alibaba.com 真 ATS URL (codex W16 标
  unverified_js_shell 的, 通过 0voice 间接拿到了). 含北森 SaaS app.mokahr.com
  专属 plan

已交付 (W17/W18, 别再说没做):
- ✅ 暑期/日常/校招/社招 deterministic classifier (recruit_type.py)
- ✅ 百度 INTERN endpoint (实测拿到 暑期+日常实习项目)
- ✅ /recommended 按 recruit_type filter (默认 ?type=intern)
- ✅ 多 keyword 派发 (从 user 真简历抽 8 个 niche keyword, 不只大厂关键词)
- ✅ discovered_via 来源 attribution (verified_official + nowcoder_sitemap
  都已标, /recommended 卡片显示 "↳ 由 keyword X 找到")
- ✅ /recommended diversity bar (大厂 vs 中小厂分桶)
- ✅ W19 SKILL invoke DRY helper (skill_view.py 收掉 4 view 重复 boilerplate)

每次 commit 前对照这两条清单, **别假装解决了再加新功能**.
