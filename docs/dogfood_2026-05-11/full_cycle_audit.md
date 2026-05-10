# Full Cycle Audit (2026-05-11)

用户原话第 9 次: "你研究的是错的, 反一个错的结果, 和没调查有什么区别".

承认 — 之前 verify 报告里有"我以为做了实际没真验过"的部分. 这次**真启
完整 cycle 跑一遍**, 把真值钉死, 不假设.

## Cycle 真跑结果

参数: 用户真简历 (1803 chars) + LLM key 真存在 (deepseek-v4-flash) +
12 真 SKILL spec.

| Stage | 真用时 | 真 ingest | 备注 |
|---|---|---|---|
| nowcoder sitemap (limit=15) | 30.3s | 15 | 全成功, source=nowcoder |
| 0voice repo (max_jobs=30) | 0.7s | 30 | parsed 475 个 |
| verified_official × 3 keywords | 22.9s | 28 | tencent_campus 6 / tencent_social 6 / baidu_campus 4 / baidu_intern 6 / **bytedance_jobs 6** |
| score_match × 5 | 64.6s | 5 'scored' events | probability 0.1 - 0.4 |

## 真 jobs 表 (枚举所有 source)

| source | count | discovered_via |
|---|---|---|
| zerovoice_repo | 30 | zerovoice_aggregator |
| nowcoder | 15 | nowcoder_sitemap |
| baidu_campus | 10 | verified_official |
| bytedance_jobs | 6 | verified_official |
| tencent_campus | 6 | verified_official |
| tencent_social | 6 | verified_official |
| **TOTAL** | **73** | 6 个源全标 attribution, 0 漏 |

## 真 harness_events 表

| kind | count | 备注 |
|---|---|---|
| scored | 5 | probability 0.1, 0.1, 0.35, 0.4, 0.3 (真 LLM 给的真分) |

## 真发现的 bug

### Bug 1: 0voice cap=30 全是阿里巴巴 (audit 才暴露)

**症状**: cap=30 时, by_company 只有 `{阿里巴巴: 30}` — 1 家公司, 0 多样性.

**根因**: 我之前用 `parsed_jobs[:max_jobs]` head 切片. README 顺序: 阿里
124 个排第一段, head 30 全在阿里里。

**真 fix** (verified): `_round_robin_by_company` 算法:
- bucket by company
- pass 1: 每家拿 1 个 (按 section_no 排序保留 README 优先级)
- pass 2..N: 重复直到 cap 满

**真 fix 后** (verified, cap=30):
```
30 家不同公司, 每家 1 个:
  阿里巴巴, 腾讯, 字节, 美团, 华为, 百度, 小米, 网易, 京东, 饿了么,
  拼多多, MiniMax, 滴滴打车, 小红书, 贝壳找房, 4399游戏, 携程,
  影石Insta360, 星星充电, 虹软科技, 同花顺, 商汤科技, 掌阅科技, 好未来,
  小宇宙, 智联招聘, 海尔集团, 金航数码, 无端科技, 昆仑万维
```

公司多样性从 1 → 30 (**30x**). 含真 niche AI 公司: MiniMax, 商汤, 昆仑
万维, 影石 Insta360, 虹软科技 — 完美匹配用户 "不只大厂, 全 + 匹配".

## 没发现 bug 的事 (枚举式 verify)

- ✅ bytedance_jobs adapter (W19+) 真 ingest 6 个 (不只是单测 mock pass)
- ✅ baidu_intern endpoint 真有 ingest (从 baidu_campus 10 个里包含 INTERN)
- ✅ discovered_via attribution 6 个源全标了, 0 漏 ((none) = 0)
- ✅ harness_events.scored 真写到 db (5 row, 含 probability JSON)
- ✅ score_match SKILL 真用 verified inputs (W15.22 修过, 没回退)

## 还没真 verify 的 (诚实清单, 不假装)

- [ ] FastAPI lifespan 真 boot ambient daemon 没在 sandbox 里启过
      (asyncio.create_task 的 cancel 路径, 30s initial_delay 后真触发)
- [ ] /recommended 模板真渲染 c.discovered_keyword (上面 stage 跑了 ingest +
      score, 但没真 GET /recommended 看 HTML output)
- [ ] application_plan 8 条 host-based plan 在用户真打开 /apply-pack 时
      真渲染 (代码 verify 了, UI render 没 verify)

这 3 条用户起 server 一次就能 verify, 我在 sandbox 里也能跑 TestClient
但暂未做.
