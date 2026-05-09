# Ambient Discovery 真实路径调研 (2026-05-10)

## 起因

用户暴怒: "你怎么还在说粘 jd, 这个我只允许用户今天自己突然看到 jd, 然后粘,
不允许把这个作为主要的流程, 不然要你干嘛. 那些大厂的或者其他的公司的官网投
递也做好了？你简直听不懂人话"

**翻译**: 真主流程 = agent 主动从多源拉岗位 → 用户照着点投. 不是用户找 JD 让
agent 帮看. paste JD = 应急 fallback, 主流程要做好.

之前 W15.18 / W15.20 / W15.21 都假设"用户自己刷 BOSS / paste JD" 是主流程,
是错的. 这份文档把真实可行的路径列清楚.

---

## 一、岗位池源头盘点 (不假设, 只列调研到的)

### A. 公开聚合源 (server-side 不登录可拉)

| 源 | URL | 内容 | 可行性 |
|---|---|---|---|
| 牛客 校招日程 | `nowcoder.com/jobs/school/schedule` | 各大厂校招/实习官方时间线 + 投递链接 (官方信息聚合) | **HIGH** — 不需登录, HTML 公开 |
| 牛客 校招职位 | `nowcoder.com/jobs/school/jobs` | 实时校招岗位列表, 可按城市/届次/规模/薪资筛 | **HIGH** — 不需登录 |
| 牛客 实习广场 | `nowcoder.com/jobs/intern/center` | 实习岗位 (HR 直发) | **HIGH** — 不需登录 |
| 0voice GitHub repo | `github.com/0voice/2026-Computer-Spring-Recruitment-Job-Compilation` | 每日更新 markdown 表格, 含投递链接 + 面经 | **HIGH** — git pull 即可 |
| BOSS 校招专区 | `zhipin.com/school/` | 应届生校招汇总 | **MEDIUM** — 反爬, 需要登录看推荐 |

**关键发现**: 牛客的校招日程页本身就是高质量聚合源 — 给的是各家**官方校招时间线**
(例: 腾讯 2026 实习 4/27-5/27 这种). 这是用户最需要的信息, 而且不需要任何登录.

### B. 大厂官网招聘 (公开浏览, 投递需登录)

| 公司 | 招聘 URL | ATS 系统 | 浏览岗位 | 投递 |
|---|---|---|---|---|
| 字节跳动 | `jobs.bytedance.com/campus` | 自研 (字节飞书 People) | 公开 | 需登录 + 内推码 |
| 阿里巴巴 | `talent.alibaba.com` | 北森 (服务阿里) | 公开 | 需登录 |
| 腾讯 | `join.qq.com` | 自研 | 公开 | 需 QQ 登录 |
| 美团 | (campus.meituan.com 等) | 北森 (推断) | 公开 | 需登录 |
| 百度 | `talent.baidu.com` | 北森 | 公开 | 需登录 |

**关键发现**:
- **北森 ATS** 服务 6000+ 中大型企业, 含阿里 / 字节 / 百度等. 
  → 适配北森一家就解一批公司
- 大部分大厂**岗位列表本身公开**, 不需登录就能看
- **投递必须登录**, 而且每家有自己的 form 字段 + 反自动化检测

### C. 用户私域源 (BOSS / 等需登录平台)

- 用户在 BOSS 看到的"推荐池" — 强依赖 BOSS 的算法 + 用户行为, **唯一拿到途径
  是用户自己浏览器登录态**. server-side 拿不到. → **必须 extension**

---

## 二、可行架构 (按工程可行性排序)

### 路径 A: Server-side 聚合 (今天就能做, ROI 最高)

每 N 小时定时拉:
1. **牛客校招日程** — 解析 HTML, 拿"公司 + 岗位类型 + 时间线 + 投递链接"
2. **0voice GitHub repo** — 拉 markdown 表格 parse
3. (可选) 学校就业办 / 微信公众号

→ 入 jobs 表, source='aggregator_X', 标"投递链接 = 官网 URL"
→ agent 跑 score_match (现在 W15.22 修了真的能跑了)
→ /recommended 排好序
→ **用户点投递链接 → 跳大厂官网 → 在官网完成投递** (本身就是行业惯例,
   大厂官网投递的回报率最高 + 有内推码加成)

**这是 ambient discovery, 不需用户先有 JD**. 用户每天打开 OfferGuide 看
`/recommended`, 直接挨个点. 不再需要他手动 paste 或刷 BOSS.

### 路径 B: Extension 增强 (BOSS / 牛客 私域池)

延续 W15.18 思路:
- 用户在 BOSS 推荐页 → extension popup 一键 sync 整页 (已做 W15.18)
- 用户在牛客 实习广场 → extension 同样一键 sync (没做)
- 用户在大厂招聘官网浏览 → extension 抓岗位详情 (没做)

补 BOSS / 牛客 / 大厂官网的 selector 适配. 这是补强 path A 拿不到的私域池.

### 路径 C: 半自动投递 (高难, 低 ROI, 暂不做)

- BOSS 沟通: DOM 没研究透 → 已做 probe, 等真样本 (W15.21 留的钩)
- 大厂官网投递: 北森 / 自研都有反自动化 + 字段动态 + 多步 form, 而且**投递本身
  是一次性事件**. 帮 user 自动填表 ROI 不高
- **正确做法**: OfferGuide 给"投递准备包"(投递链接 + 简历建议 + 内推码 + 
  注意事项 + 面试前必读), 用户自己 5 分钟在官网点完

---

## 三、对应到代码现状的诚实评估

### 已做但被假设废掉的

| 功能 | 状态 |
|---|---|
| W15.14 evaluate_job (paste JD) | W15.22 修好了, 但 paste JD 不是主流程, 仅 fallback |
| W15.18 BOSS extension popup | 用户必须手动点, 非 ambient |
| W15.20 浮窗评分 | 用户必须已经在 BOSS, 非 ambient |
| W15.21 /recommended 待点列表 | 池子是用户手动 sync 来的, 不是 agent 拉的 |
| W15.21 BOSS 沟通 DOM probe | 等真样本 — 用户没装 + 没用过 → 没收到 |

### 关键的没做

1. **服务端定时拉牛客校招日程 + 0voice repo** — 真 ambient discovery, 0 成本
2. **大厂官网招聘页岗位抓取** (公开 part)
3. **投递链接展示 + 内推码搭配 + 投递准备包**

### 之前 BOSS 沟通自动化的真相

- 我假设 selector 是 `.chat-im-wrap` 等 — 没验证, 全是猜
- W15.21 加了 DOM probe 让用户帮抓样本 — **正确思路**, 但用户没用过, 没收到
- 在收到真样本前**不能写自动 fill 沟通框** — 写了 90% 概率污染用户聊天

---

## 四、新的优先级建议 (用户拍板)

### Tier 1 (做了立刻能用, 解你"agent 找岗位"核心痛点)

1. **服务端定时拉聚合源** — 牛客校招日程 + 0voice repo
   - 每 6 小时跑一次, 入 jobs 表
   - `/recommended` 显示岗位 + **投递链接** + 截止日期
   - 用户每天看一次, 点链接去大厂官网投
   - 工程量: 1-2 天 (HTML 解析 + GitHub raw fetch + dedup + cron)

2. **/recommended 改造 — 显示投递链接 + 是否要内推码 + 截止日期**
   - 现在只显示岗位元信息, 不告诉用户该去哪投
   - 工程量: 半天

### Tier 2 (扩源, 需要 extension 增强)

3. **牛客实习广场抓取** (extension content_script)
4. **大厂官网招聘页抓取** (extension content_script, 适配 jobs.bytedance / talent.alibaba)

### Tier 3 (低优先, 暂不做)

5. **BOSS 沟通自动 fill** — 等用户先用 W15.21 的 probe 抓 5+ 真样本
6. **大厂官网半自动投递** — ROI 低, 投递本身一次性

---

## 五、引用 (调研出处)

- 字节校招官网 https://jobs.bytedance.com/campus
- 阿里招聘官网 https://talent.alibaba.com
- 腾讯校招 https://join.qq.com
- 牛客 校招日程 https://www.nowcoder.com/jobs/school/schedule
- 牛客 校招职位 https://www.nowcoder.com/jobs/school/jobs
- 牛客 实习广场 https://www.nowcoder.com/jobs/intern/center
- 0voice 校招 repo https://github.com/0voice/2026-Computer-Spring-Recruitment-Job-Compilation
- 大厂 ATS 调研 (北森服务阿里/字节/百度等 6000+ 企业) https://news.qq.com/rain/a/20240819A03DNT00
- BOSS 沟通日上限 ~80 https://github.com/wvit/BOSS_batch_deliver
- 国内招聘平台对比 https://zhuanlan.zhihu.com/p/1976717795640223529
- 互联网大厂 2026 校招汇总 https://zhuanlan.zhihu.com/p/2008172991129854482
- 腾讯阿里 2026 校招 AI 岗 https://adg.csdn.net/69708b82437a6b40336aa432.html

## 六、明确不再假设的事

- ❌ 不假设 paste JD 是主流程
- ❌ 不假设用户会自己刷 BOSS
- ❌ 不假设 BOSS 沟通框 DOM 我知道
- ❌ 不假设大厂官网能 server scrape
- ❌ 不假设大厂官网投递可以全自动
- ✅ 真做的: agent server-side 定时聚合 → /recommended → 用户照官网链接投
