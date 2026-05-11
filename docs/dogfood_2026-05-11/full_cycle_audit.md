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

(2026-05-11 末班补 audit, 全 verify 完了 — 见下面 "Audit 后真 verify 完成
的 3 条" 节)

## Audit 后真 verify 完成的 3 条 (2026-05-11 末班补)

### ✅ 1. /recommended 模板真渲染 discovered_keyword (TestClient)

GET `/recommended` (sandbox TestClient), HTML 42928 bytes, status 200.
真 grep HTML body:
- `↳ 由 keyword` 出现 4 次 (每个 job 一行 attribution)
- 4 个 keyword needles 全找到: `智谱AI 岗位汇总` / `AI Agent` / `大模型微
  调` / `(sitemap walk, no keyword)`
- 顶部 keyword strip 渲染了 cycle_keywords
- diversity bar 渲染了 大厂/中小厂分桶
- recruit_type pill 渲染了 summer_intern x2 + daily_intern x2

### ✅ 2. FastAPI lifespan 真 boot + cancel ambient daemon (TestClient)

`scripts/audit_w19_lifespan_boots_daemon.py` (monkeypatch
`_ambient_discovery_loop` 成 stub coroutine):
- ✅ 进 TestClient context 后 stub 真被 `asyncio.create_task` 拉起来 (state.called=True)
- ✅ kwargs 真传齐: `{store, settings, runtime, skills, user_profile_text}`
- ✅ 退出 context 后 stub 收到 `CancelledError` (state.cancel_seen=True, 没泄漏)
- ✅ 负面 case 1: 空 LLM key → loop **不**启 (lifespan 早 return)
- ✅ 负面 case 2: `disable_ambient_crawl=True` → loop **不**启

### ✅ 3. /apply-pack 8 条 host-based plan 真渲染 (TestClient × 11 case)

`scripts/audit_w19_apply_pack_real_render.py`, 真插 11 个 job (覆盖 11 个
source/host 组合), TestClient GET `/jobs/{id}/apply-pack`, grep HTML body
找 plan-specific 中文 needle:

| source / host | 真匹配 needles |
|---|---|
| tencent_campus / join.qq.com | "腾讯校招 · join.qq.com" / "微信扫码登录 join.qq.com" / "腾讯 ATS 不限简历字数" |
| tencent_social / careers.tencent.com | "腾讯社招 · ⚠ 应届生慎投" / "QQ / 微信扫码登录 careers.tencent.com" |
| baidu_intern / talent.baidu.com | "百度 · 暑期/日常实习" / "百度账号登录 talent.baidu.com" / "暑期项目一般要求 3-4 月以上" |
| baidu_campus / talent.baidu.com | "百度 · 校招正式" / "校招岗位选「毕业入职」时间" |
| bytedance_jobs / jobs.bytedance.com | "字节跳动 · jobs.bytedance.com" / "飞书扫码登录" / "字节走自研飞书 People ATS" |
| zerovoice_repo / campus-talent.alibaba.com | "阿里巴巴 · talent.alibaba.com" / "淘宝 / 支付宝账号登录" |
| zerovoice_repo / app.mokahr.com | "北森 SaaS" / "微信扫码或手机号注册 app.mokahr.com" / "ATS 解析较严格" |
| agent_search | "agent 搜到的外部岗 · ⚠ 先核验" / "先打开链接核验" / "不是 verified API" |
| nowcoder | "牛客" / "牛客打招呼" |
| boss / zhipin.com | "BOSS 直聘" / "立即沟通" |
| 未识别 (manual + http url) | "官网 / ATS 网申" |

11/11 PASS.

## Bug 2 (audit 暴露): apply_pack.html 漏渲染 material_checklist + post_apply_actions

**症状** (audit 第 1 轮第 1 个 case 暴露的): tencent_campus 的 "腾讯 ATS
不限简历字数" needle 在 HTML 里找不到. grep template 才发现:

`application_plan` dataclass 有 7 个字段被 host plan 真填了:
- platform_label / channel_note / steps / **fields** / **material_checklist** /
  **post_apply_actions** / verified_source / evidence_url

但 `apply_pack.html` 只渲染了 platform_label / channel_note / steps / fields
+ verified_source / evidence_url —— **material_checklist + post_apply_actions
被 dataclass 填了但 template 不读, 50% 的 plan 内容用户根本看不到**.

8 个 host plan 都受影响 (每个 plan 都填这 2 个 list, 但都不渲染).

**真 fix** (this commit): 在 template 加一段:
```html
{% if application_plan.material_checklist or application_plan.post_apply_actions %}
<div ... grid>
  📦 投前材料 checklist  ← application_plan.material_checklist
  📍 投后跟进动作        ← application_plan.post_apply_actions
</div>
{% endif %}
```

**真 fix 后** (audit 第 2 轮): 11/11 PASS, "腾讯 ATS 不限简历字数" 出现.

## 这次 audit 真发现的 bug 总览 (2 个)

1. **0voice cap=30 全是阿里巴巴** (W19+ 第一轮 audit 发现, 已 fix
   `_round_robin_by_company`, 公司多样性 1 → 30)
2. **apply_pack.html 漏渲染 material_checklist + post_apply_actions** (这次
   末班 audit 发现, 已 fix template)

两个都是"我以为 verify 了, 没真打开看" 的反例 — 第 1 个我没看 by_company
分布就 commit; 第 2 个我看了 application_plan dataclass 的 unit test 通过
就 commit, 没真 GET /apply-pack 看 HTML 渲染. 都验证了 CLAUDE.md 规则 C:
**判定"X 已 verify"前必须用枚举式工具 (TestClient GET + grep HTML), 不能
用搜索式 (单测覆盖率)**.
