# W20 — 实习僧 (shixiseng.com) Adapter 真 Dogfood

## 为什么加这个源

CLAUDE.md 第 6 节列出 W20 候选第 1 条是 wondercv adapter, 第 2 条是
"AgentSeek / 实习僧 / 牛客实习广场 SSR 抓取 (实习专用聚合)". 这次选了
实习僧 因为:

- **用户当前最高优先级是 2026 暑期实习** (CLAUDE.md 第 2 节, 5 月节奏)
- W19+ 现有源里实习覆盖偏弱:
  - nowcoder sitemap: 实习+校招+社招混
  - 0voice repo: 75% 校招, 25% 实习
  - baidu_intern: 只 10 个百度自己实习
  - 字节: 全社招
  - agent_search: 1-3 niche/cycle
- 实习僧是**国内大学生找实习首选聚合站**, /interns 端点全部是实习

## 真 probe 结论 (2026-05-11)

| 检查 | 真值 |
|---|---|
| List page (`/interns?keyword=AI`) | SSR Nuxt, 但字段全 font-encoded 反爬 |
| Detail page (`/intern/inn_xxx`) | SSR + **真字符** (不再 font-obf) |
| `/intern/<token>` token in list HTML | ✅ 20 个/page, 真清楚 |
| `<title>` pattern | `<岗位>实习招聘-<公司>实习生招聘-实习僧` |
| `new_job_name` div | 岗位名 |
| `com-name` link | 公司名 |
| `job_position` span title attr | 城市 |
| `job_money` div | 薪资 (`100-200/天`) |
| `job_detail` div | 完整 JD body |

策略: list page 只取 `inn_xxx` token (1 req), 每个 token 再 fetch
detail page (1 req each) — 21 reqs/keyword.

## 真 cycle 跑 (`scripts/audit_w20_shixiseng_live.py`)

参数: `_crawl_shixiseng_per_keyword(keywords=["AI Agent", "大模型"], limit_per_kw=8)`

| 指标 | 真值 |
|---|---|
| 真用时 | ~7s |
| inserted | **16** |
| duplicate | 0 |
| parsed | 16 |
| errors | [] |
| **company_diversity** | **16 (1 job/company avg)** |

### 真公司分布 (16 unique, 12 niche / 4 大厂)

```
大厂 (4):
  百度 / 淘宝闪购 / 网易 / 小红书

niche AI / 中小厂 / 研究院 / 出海 (12):
  行知启新 / 香巴拉科技 / 美图公司 / 莱坊 / 算力大陆 / AKULAKU /
  聚宽投资 / 澳鹏科技 / 量坤科技 / 同道猎聘集团 /
  清华四川能源互联网研究院 / 医者
```

### 真 recruit_type 分布

| type | count |
|---|---|
| daily_intern | 16 |

100% 实习 (符合 shixiseng /interns 端点的事实). recruit_type classifier
默认把 shixiseng 标 `DAILY_INTERN`, title 含 "暑期"/"summer" 的标
`SUMMER_INTERN`. 这次 16 个 title 全无 "暑期" 字样, 所以全 daily.

### 真 application_plan 路由

样本: `job#1 行知启新 · AI Agent 实习生`

```
platform: shixiseng
platform_label: 实习僧 · 实习专用聚合
steps: 6 步 (注册 → 简历 → 上传 → 投递 → 跟进 → 标记)
fields: 6 字段 (含可实习时长 / 周到岗天数)
material_checklist: 5 条 (含 PDF 5MB / 时间灵活提示)
post_apply_actions: 5 条 (含微信加好友 / 站内信 / 7-10 天再投)
```

template 真渲染所有 4 个 list (W19 audit 修过的 material_checklist +
post_apply_actions).

## 跟之前源对比 (公司多样性)

| 源 | inserted | 真公司数 | 大厂占比 | 实习占比 |
|---|---|---|---|---|
| 0voice (cap=30) | 30 | 30 | ~50% | ~25% |
| nowcoder (limit=15) | 15 | ~10 | 30-40% | 50% |
| baidu_intern | 10 | 1 (百度) | 100% | 100% |
| **shixiseng (kw×2, lim=8)** | **16** | **16** | **25%** | **100%** |

shixiseng 这次填补的真空白:
- 全部 100% 实习 (vs 0voice 25%)
- 75% niche / 25% 大厂 (vs 0voice 50/50, 用户最想要的"非大厂"覆盖)
- 含 4 类 niche: AI 创业 (香巴拉/算力大陆/量坤/医者) + 出海 (莱坊/AKULAKU)
  + 投资/数据 (聚宽/澳鹏/同道猎聘) + 研究院 (清华四川能源)

## 真 verify 完成的 wiring

- ✅ `crawl_shixiseng()` adapter: list + detail 两阶段 fetch, 真测过 stub +
  live network
- ✅ `recruit_type.classify_recruit_type` 加了 shixiseng 分支
- ✅ `application_plan.build_application_plan` 加了 shixiseng route +
  `_shixiseng_plan` 6 字段 plan
- ✅ `_load_unscored_discovered_ids` sources 里加了 'shixiseng' (真测了)
- ✅ `_run_one_cycle` 加了 shixiseng 阶段 (在 verified_official 之后,
  agent_search 之前)
- ✅ `_crawl_shixiseng_per_keyword` 帮 ambient 多 keyword 派发, 真测过

## 没 verify 的 (诚实清单)

- [ ] 真 ambient 一次完整 cycle (4 阶段: nowcoder → 0voice → verified ×
      keywords → shixiseng × keywords → agent_search) 端到端跑 — 因为这
      要真 LLM key 跑 score_match. sandbox 没真 key 验过 score_match
      end-to-end (W19 audit 验过 LLM key 真存在为 OFFERGUIDE_LLM_API_KEY,
      但本次未跑全 cycle)
- [ ] /recommended 默认 ?type=intern filter 下 shixiseng 16 条都该出现
      在卡片里 (应该 OK, 因为 daily_intern 是 default filter 之一,
      但没真 GET 过 /recommended 看真 HTML)

第 1 条用户起 server 后真跑一次就 verify; 第 2 条只是 TestClient + grep
HTML, 待用户要的话可以加 audit script.

## Bug 没碰到的事 (诚实)

这次没发现新 bug. parse 真稳; live network test PASS; 全测试 1015 通过.

## 测试覆盖

`tests/test_w20_shixiseng.py`: 22 case + 1 live skip
- parser 6 case (含 fallback / 缺字段 / HTML 标签 strip)
- list token extraction 1 case (dedup + order)
- to_raw_job 2 case (full + missing optional)
- crawl_shixiseng 端到端 4 case (full + cap + list-fail + 单 detail crash)
- recruit_type wiring 4 case (default daily / 暑期 → summer / english summer / 无实习字 → daily)
- application_plan wiring 3 case (source route / host route / post_apply)
- ambient wiring 2 case (sources list / per_keyword helper)
- live network smoke 1 case (skip 默认, 设 OFFERGUIDE_RUN_NETWORK_TESTS=1 跑真)

22 + 1 skip = 23 case. 全 PASS.
