# 修法补丁 — 重写两条错误建议

**日期**: 2026-05-07
**起因**: 用户反馈我前份 `usability_verdict_2026-05-07.md` 第 2.3 节和 2.4 节末尾的修法**思路错了**
**关系**: 本文是对前份的修正，覆盖原 §2.3 和 §2.4 末尾的"修法"段落

---

## 我前份错在哪

| 议题 | 我前份的修法 | 思路问题 | 用户反馈的正确思路 |
|---|---|---|---|
| **反编造防线漏（tailor 加了 RAG/Attention）** | 加 keyword containment check，没在原文就 `raise TailorViolation` 拒绝下载 | **审查阻拦**思路 — 把"编造"当 bug 拦死 | **透明告知 + 学习引导** — 告诉用户加了什么、为什么加、怎么去补 |
| **一岗多投限额抄的不准** | 给 hardcoded 表加 `source: "estimated"` 字段、UI 标"低置信度" | **降级宣传**思路 — 把不准当免责声明 | **agent 自调研** — 这种事就该让 agent 现搜现整合，hardcoded 表本身就该消失 |

两条都是同一种病：**把产品问题当成"标记问题"——给个红色标签糊弄过去**。
正确思路是：**让产品长出主动能力**——
- tailor 不是"产物"是"学习清单的入口"
- compare_jobs 不是"查表"是"agent 当下去搜"

---

## 正确修法 1: tailor_resume 透明告知 + 学习引导

### 哲学

简历微调本来就要往里加 JD 关键词——这件事本身没错，错的是**塞进去就当事实让用户带去面试**。

正确的产品形态：**tailor 完不是结束，是开始**。每条新加的关键词都该让用户看见、明白为什么加、知道在面试前要怎么补。

这跟 `profile_resume_gap` 的"短期能补"桶是**同一种哲学**——把缺口转成可执行的学习计划。tailor_resume 应该跟它**复用同一种数据结构**。

### 数据结构改造

`TailorResult` 加一个字段：

```python
@dataclass
class InsertedClaim:
    keyword: str                      # "RAG"
    inserted_in_paragraph: int        # 简历第 31 段
    insertion_text_snippet: str       # "...熟悉 Transformer、Attention 机制与 RAG 原理..."
    
    source_kind: Literal[
        "from_jd",              # JD 里明确要求的
        "supported_by_project", # 简历项目里能侧面证明
        "completely_new",       # 原文 0 提及, 项目里也没痕迹 — 最危险
    ]
    
    why_inserted: str                 # "JD 要求 'RAG / 检索增强生成 经验', 是字节 AI Agent 岗高频考点"
    
    # 关键: 学习引导, 跟 profile_resume_gap 的 short_term_補 桶共用 schema
    prep_plan: PrepPlan
    
@dataclass
class PrepPlan:
    estimated_days: int               # 5
    must_read: list[str]              # ["Lewis et al. 2020 RAG paper", "LangChain RAG tutorial"]
    must_build: list[str]             # ["跑通 LangChain + ChromaDB 最小 demo, 至少 100 条文档", "测一下 hit@5 数字"]
    must_prepare_qa: list[str]        # ["RAG 怎么 debug retrieval miss", "BM25 vs dense embedding 各擅长什么场景", "怎么处理 chunk 边界丢上下文", ...]
    fallback_if_asked: str            # 面试真被问且没准备好时怎么诚实说: "RAG 我读过原理但还没真上手, 项目里 retrieval 用的是 BM25 + 简单余弦"
```

### UI 流程

`/tailor` 页面 review 阶段：

```
┌─ 字节跳动 AI Agent 实习 — 简历微调结果 ─────────────┐
│                                                        │
│ ✓ 9 段改写 (查看 diff)                                 │
│ ✓ 50 段保护跳过 (学校/日期/标题不动)                   │
│                                                        │
│ ⚠ 3 条新加的关键词需要你确认 + 准备                    │
│                                                        │
│   ┌─ "RAG" — 加进了「专业技能」段 ─────┐               │
│   │ source: completely_new ⚠           │               │
│   │ 原文从未提过 RAG; 加是因为 JD 第 4 │               │
│   │ 条明确要求 + 字节 Agent 岗 4/5 面经│               │
│   │ 都问到 retrieval miss debug        │               │
│   │                                     │               │
│   │ 5 天准备方案:                       │               │
│   │   📖 读 Lewis 2020 RAG paper        │               │
│   │   🔨 跑通 LangChain + ChromaDB 100 │               │
│   │      条文档 demo, 测 hit@5         │               │
│   │   🎤 准备 5 道高频题:               │               │
│   │      - retrieval miss 怎么 debug    │               │
│   │      - BM25 vs dense embedding      │               │
│   │      - chunk 边界丢上下文怎么解     │               │
│   │      - rerank 是否必须              │               │
│   │      - 知识更新策略                 │               │
│   │   🪤 万一还没学就被问, 诚实说:     │               │
│   │      "原理读过, 项目里只用了 BM25" │               │
│   │                                     │               │
│   │ [ ✓ 保留, 我去学 ] [ ✗ 删掉这条 ]  │               │
│   └────────────────────────────────────┘               │
│                                                        │
│   ┌─ "Attention" — 加进了「专业技能」段 ┐               │
│   │ source: supported_by_project ✓     │               │
│   │ 简历有 Transformer 项目 (RemeDi),  │               │
│   │ 提到 attention 是顺理成章          │               │
│   │ 1 天巩固方案: ...                  │               │
│   │ [ ✓ 保留 ] [ ✗ 删掉 ]              │               │
│   └────────────────────────────────────┘               │
│                                                        │
│   ┌─ "agent workflow" — Deep Research..┐                │
│   │ source: from_jd                    │                │
│   │ JD 关键词, 项目本身就是这件事      │                │
│   │ 0 天: 你已经在做了                 │                │
│   │ [ 自动保留 ]                       │                │
│   └────────────────────────────────────┘                │
│                                                        │
│ [⬇ 下载 docx (确认所有保留)] [📋 复制学习清单到 todo]   │
└────────────────────────────────────────────────────────┘
```

### 实现复用

`prep_plan` 字段跟 `profile_resume_gap.short_term_补.action` 是**同一个 schema**——直接调用现有的 SKILL：

1. tailor_resume 跑完识别出 3 条 `completely_new` 类
2. 内部调 `profile_resume_gap` 那条"5 天补 Transformer + 写 blog" 的逻辑——给每条新加的 keyword 出 PrepPlan
3. tailor 输出 = docx + InsertedClaim 列表（每条带 PrepPlan）
4. UI 上让用户**逐条选保留 / 删掉**，保留的进 todo 列表（可以复制到日历）

### 为什么这个比"raise violation"好

| 维度 | raise violation | 透明 + 引导 |
|---|---|---|
| 用户体感 | 程序拒绝、给我个错 | 程序帮我看到了、还告诉我怎么办 |
| 是否 inflation | 阻拦但不教 | 用户主动选择 + 准备完后真的会 |
| 是否能用 | 删掉的 keyword 永远进不了简历 | 学完后再加 = 不再是"编"的 |
| 跟 profile_resume_gap 关系 | 两个独立功能 | 同一种哲学，schema 复用 |
| **核心区别** | **把"编造"当 bug 拦死** | **把"加了什么"当学习起点** |

最关键的：用户面试被问到 RAG，**他真的会答**——因为 5 天前他读了 paper、跑了 demo、备了 5 道题。这才是 OfferGuide 的卖点：**简历微调不是给 ATS 看的虚假，是给用户的学习清单**。

---

## 正确修法 2: compare_jobs 让 agent 自己调研

### 哲学

"字节硬限 2 / 阿里 3"这种动态信息**永远不该 hardcoded 在 Python 表里**。这种数据：
- 每年校招政策都在变
- 不同部门 / 不同 BG 限额不一样
- 网传"硬限"和实际操作经常不一致
- 真实情况只能从**当下的面经 / 校招公告 / 用户实际投递历史**里读出来

`briefs.refresh_brief()` + `effective_app_limit()` 是已经搭好的双轨架构——但**默认 fallback 到 hardcoded**，而不是 fallback 到"我不知道，让我现搜"。

正确思路：**移除 hardcoded 表、让 agent 在用户问到时现场调研**。

### 流程改造

用户问"字节我能投几个"→ agent 走这条链：

```
┌─ Step 1: 看本地证据 ─────────────────────────────┐
│ - corpus 里关于"字节 一岗多投"的面经/帖子有几条？│
│   → 用 sqlite-vec 搜 query 匹配                   │
│ - user_facts 有没有相关历史投递记录？             │
│   → 用户之前投过字节几次, 第 N 次还能投吗         │
│ - briefs 表里 Bytedance 的最近合成 brief 多旧？   │
└──────────────────────────────────────────────────┘
                ↓ 证据足够 (≥ 3 条 2026 年源 + brief < 7 天)
                  → 直接给答案 + 来源
                ↓ 证据不够
┌─ Step 2: 现搜 (Tavily / Bing CN) ────────────────┐
│ query 1: "字节 2026 校招 一岗多投 限制"          │
│ query 2: "ByteDance campus recruitment one job   │
│           per person policy 2026"                 │
│ query 3: 搜小红书 / 牛客 / 一亩三分地 近 3 月     │
│         "字节 投了 X 个 还能投吗"                 │
│                                                   │
│ 每条结果 LLM 评估:                                │
│  - 来源时间 (2026/4 之后才采纳)                   │
│  - 来源类型 (官方公告 > 应聘者实测 > 道听途说)    │
│  - 是否提到部门 / BG 差异                         │
└──────────────────────────────────────────────────┘
                ↓
┌─ Step 3: 综合给答案 + 置信度 + 来源 ─────────────┐
│ answer: "据 2026/3 之后的 4 条来源, 字节 BG       │
│   普遍限 2-3 个, 同时投不同 BG 不冲突, 但         │
│   同 BG 投第 3 个 HR 会主动联系你撤其中之一"      │
│                                                   │
│ confidence: 0.7 (medium-high)                     │
│                                                   │
│ sources:                                          │
│   - 牛客 [2026/4/12] 一面后第 3 个被 HR 撤        │
│   - 小红书 [2026/3/28] 投 3 个 BG 不撤            │
│   - 一亩三分地 [2026/4/2] 同 BG 限 2              │
│   - 字节飞书校招 [2026/3/15] 提到"建议聚焦"       │
│                                                   │
│ 入 briefs 表 (TTL 7 天) 给后续用户复用            │
└──────────────────────────────────────────────────┘
                ↓
┌─ Step 4: compare_jobs 决策 ──────────────────────┐
│ "你已投字节朝夕光年 1 个 + AI Lab 1 个,           │
│  根据上面 brief 还能投 1-2 个 (跨 BG)。           │
│  推荐 Seed 而不是 Doubao 因为 ..."                │
└──────────────────────────────────────────────────┘
```

### 关键代码改动

```python
# src/offerguide/skills/compare_jobs/helpers.py

# 删除这个表
# COMPANY_APPLICATION_LIMITS = {
#     "字节跳动": 2, "阿里": 3, "淘天": 3, ...   ← 全删
# }

def effective_app_limit(store: Store, company: str) -> AppLimitAnswer:
    # Step 1: 本地证据
    local = _check_local_corpus(store, company)
    brief = _check_recent_brief(store, company, max_age_days=7)
    if local.evidence_count >= 3 and brief is not None:
        return _synthesize(local, brief, source="local")
    
    # Step 2: 触发 agent 调研 (异步, 用户能看到 streaming "正在搜...")
    return _trigger_agent_research(store, company)
    # 返回 pending answer, agent 跑完写回 briefs 表 + 通过 SSE 推给 UI
```

### UI 体感

用户在 compare_jobs 页面问"字节我能投几个"：

```
你: 字节我能投几个？

agent: 让我现查一下 (本地 brief 已 5 天前, 不够新)
       ⠋ 搜 4 个 2026/4 之后的源...
       ⠙ 评估每条来源时间/类型/部门差异...
       ⠹ 综合中...
       
答案: 跨 BG 普遍可投 2-3 个 (置信度 中-高)
来源:
  • 牛客 2026/4/12 — 一面后第 3 个 HR 撤
  • 小红书 2026/3/28 — 投 3 个 BG 不撤  
  • 一亩三分地 2026/4/2 — 同 BG 限 2
  • 飞书校招公告 2026/3/15 — "建议聚焦"

你的现状: 已投朝夕光年 + AI Lab (跨 BG, 各 1 个)
建议: 还可投 1-2 个跨 BG, 推荐 Seed 而不是 Doubao 因为 ...
```

——这才是 agent 的体感。不是查表给数字，是**让用户看到 agent 当场调研、给来源、给置信度**。这件事 ChatGPT 做不到（没有 Tavily 后端 + 没有本地 corpus）—— OfferGuide 做得到。

### 为什么这个比"加 source 字段"好

| 维度 | 加 source 标 estimated | agent 自调研 |
|---|---|---|
| 数据新鲜度 | 永远跟不上政策变化 | 每次问都是当下数据 |
| 用户信任 | 看到"low confidence"反而更不敢用 | 看到来源 + 时间 + 置信度可以自己判断 |
| 部门/BG 差异 | 表里平的，做不到 | 搜出来的源里就有差异信息 |
| 用 agent 能力 | 不用 | 用满了 (corpus + Tavily + LLM 综合 + briefs cache) |
| 可推广 | 字节 / 阿里两家硬抄 | 任何公司都能问 |

最关键的：这件事用户问 ChatGPT 它会编（"字节限 2"——它的训练截止数据已经过期），用户问 OfferGuide 它会**当场搜带证据**。**这就是把 agent 该做的事还给 agent**。

---

## 我前份哪两段该改

### 原 `usability_verdict_2026-05-07.md` §2.3 末尾「修法」段：

> **修法**：tailor_resume 改写后做一次 keyword diff——新出现的核心技能名（Transformer/Attention/RAG/PyTorch/Kubernetes 这种）必须**有原文证据**才能保留。已经有 `master_resume_text` 了，加一个 `containment check` 一行代码的事。

**改成**：

> **修法**：tailor_resume 输出多一个 `inserted_claims` 字段，每条新加的关键词带 `source_kind`（from_jd / supported_by_project / completely_new）+ `why_inserted` + `prep_plan`（5-7 天具体学习方案，跟 profile_resume_gap 共用 schema）。UI 上让用户逐条 review、保留的进 todo。`completely_new` 不阻拦下载——但用户必须看到。哲学：tailor 不是"产物"是"学习清单的入口"，跟 profile_resume_gap 的"短期能补"桶同一哲学。详见 `usability_verdict_2026-05-07_erratum.md` §1。

### 原 `usability_verdict_2026-05-07.md` §5「一岗多投限额降级（半天）」段：

> 全部 hardcoded limit 加 source 字段（"estimated, please verify"）；compare_jobs 输出加 "本数据源置信度: low" 标签

**改成**：

> 删掉 `COMPANY_APPLICATION_LIMITS` 硬编码表；`effective_app_limit()` 改成 agent 调研流程（本地 corpus + briefs 7 天缓存 → 不足时 Tavily 搜 4 条 2026/4+ 来源 → LLM 综合给置信度 + 来源），UI 上 streaming 展示"正在搜..."。详见 `usability_verdict_2026-05-07_erratum.md` §2。工时从半天调到 2-3 天，但产品形态完全不一样。

---

## 我犯的错的元层面反思

我前份这两条思路都是**"快速 patch 不出错"导向**，不是**"产品该长成什么样"导向**：

- raise violation = 出错就拦 = 工程师思维
- 加 source 字段 = 不准就免责 = 律师思维

你的反馈是**产品 / agent 思维**：
- 给用户透明 + 教育
- 让 agent 主动调研

这两个修法**总工时从原来 0.5+0.5=1 天变成 3-5 天**——但做出来的产品形态从"防御性"变成"主动性"。
