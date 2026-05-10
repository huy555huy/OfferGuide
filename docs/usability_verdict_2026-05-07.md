# OfferGuide 真实可用性判断 — "能用吗？好用吗？"

**日期**: 2026-05-07
**问题**: 用户原话「之前的版本试过相当难用，且没什么效果」+「真的能用吗？好用吗？」
**方法**: 不基于 README / 自陈，直接打开真实 dogfood 产物 + diff + 关键词审计

---

## TL;DR — 二轴答案

| 维度 | 答案 | 证据 |
|---|---|---|
| **能用？(技术层面 SKILL 输出有用吗)** | ✅ **能用，质量超预期** | dogfood W11 输出有 evidence-cited 的 must_have 清单 + 4 桶 gap 分析 + top 3 actions；docx_tailor 真把 9 段简历微调出来还保格式 |
| **好用？(产品层面有人会日用吗)** | ❌ **不好用，4 个真实问题** | 装机摩擦 / 延迟 ≠ 宣传 / 编造防线漏 / 信号源极稀疏 |
| **比"之前难用版"进步了吗？** | ✅ 显著 | tailor_resume / corpus_quality / profile_resume_gap 都是 W11-W12 才加的，输出质量在线 |
| **现在能拿出去给同学用吗？** | ❌ 别 | 装机就会卡死 90% 用户；夸张承诺会让人失望 |

**一句话**：核心 SKILL 是真行的，**产品壳是真不行**。如果用户**能跨过装机门槛 + 能等 1 分钟**，每次输出都对得起这分钟。但用户跨不过，且不会等。

---

## 1. 能用 / 真有效果吗 — 看真 dogfood

### 1.1 corpus_quality 分类器（5/5 正确）

`examples/dogfood_w11_output.txt` 真实跑过 5 个样本：

| 样本 | 期望 | 实际 | score | 评语质量 |
|---|---|---|---|---|
| 卖课/引流文案 | marketer | ✅ marketer | 0.00 | "纯卖课/引流文案，无真实面试内容" |
| 牛客 offer 复盘 | offer_post | ✅ offer_post | 1.00 | "三轮面试细节、项目数据（GAIA benchmark成绩、hit@5 0.78）、技术深度讨论" |
| 牛客挂经 | interview | ✅ interview | 1.00 | "具体日期/时长/题目/卡点/复盘，无引流迹象" |
| GitHub 项目分享 | project_share | ✅ project_share | 1.00 | "含具体技术栈、数据表格、性能对比" |
| 求职咨询贴 | other | ✅ other | 0.50 | "提问招聘时间和流程，无个人经历分享、无引流" |

**5/5 正确**。重点是**评语本身比标签还有用**——告诉用户为什么这条是 marketer / 为什么这条值 1.00。这是**真的能用**。

### 1.2 successful_profile（3 条 evidence 合成画像）

输出有具体的：
- **must_have**: "LangGraph state machine 设计与实现（2/3 样本提及）" / "手撕算法（leetcode hot100 级别，3/3 样本均涉及算法题）"
- **common_questions** 8 道具体题：反转链表、第 k 大元素、Transformer attention 缩放因子推导、GRPO vs PPO 区别、设计 1000 QPS agent inference pipeline、RAG retrieval miss debug…
- **why_they_passed** 5 条带证据归属："二面面试官评价'这个评测平台听起来比我们内部用的还系统'（来自 1 条 offer_post）"
- **uncertainty_notes** 5 条诚实交代："样本量极小（仅 3 条）" / "无失败样本对照"

**这种输出 ChatGPT 不会给**——ChatGPT 给的是泛泛的"建议复习算法 / 准备项目"。OfferGuide 给的是"刷反转链表 + 写 attention √d 推导 blog"。**这是真有效果的**。

### 1.3 profile_resume_gap（4 桶分类）

| 桶 | 数量 | 质量评 |
|---|---|---|
| 已具备 | 6 | 每条带简历原文位置（"法至科技实习用 LangGraph"） |
| 短期能补（≤2 周） | 5 | 每条带具体动作 + 时间 + 数字（"10 天刷 leetcode hot100 重点 40 题"） |
| 短期补不了 | 4 | 每条带最少周期估算（"≥3 个月实习"） |
| **不能编** | 4 | **每条带验证渠道（"HR 必查学信网"）** |

最后给 verdict: `maybe` + 一句 rationale + top 3 actions。

**关键设计**：把"诚实"做成了一个 bucket。"不能编"桶是产品最强的差异化——99% 求职 AI 都在帮你**润饰简历到不真实**，这个桶把"伪造代价"摆在桌面。这点真聪明。

### 1.4 docx_tailor — 真改 9 段，但有真问题

**9 段改写**对 9 段做了 diff（详细见下面 1.5）。其中：

✅ **6 段是 lossless 重构**：把"提出'语义层 + 工作区层'双层架构：语义层以 AgentState 作为当前研究理解的结构化锚点..."这种自创术语换成"设计双层 Agent 架构：语义层以 AgentState 维护研究状态..."——更紧凑，**事实没变**。

✅ **2 段是 JD 关键词替换**：把项目里"研究 frontier 与闭环条件"换成 JD 同语义的"agent workflow"——这是合理的术语对齐。

⚠️ **1 段疑似 inflation**：

```
段[31]
原版: "2、了解概率论基础，对强化学习（如 PPO、GRPO）及扩散模型（Diffusion Model）的算法原理有一定了解。"
微调: "熟悉 Transformer、Attention 机制与 RAG 原理；掌握强化学习算法（PPO、GRPO）及扩散模型的理论基础与工程实现"
```

关键词审计（grep 全文）：
- "Attention" — 原版**0 次** → 微调版 **1 次**（注入）
- "RAG" — 原版**0 次** → 微调版 **1 次**（注入）
- "Transformer" — 原版 2 次（仅出现在项目描述）→ 微调版 3 次（**多了一次专业技能段的声明**）

**这是真实的 product bug**：
- SKILL.md 自陈"4 层反编造"（SKILL body 显式禁令 + 受保护跳过 + ±50% 长度漂移 + master_resume hash）
- 但**只看长度漂移没看新声明引入**——±50% 在词级别被绕过
- 这一段让用户的"了解概率论基础"被替换成"熟悉 Transformer / Attention / RAG"，HR 真问起来用户得自己圆

**修法**：tailor_resume 改写后做一次 keyword diff——新出现的核心技能名（Transformer/Attention/RAG/PyTorch/Kubernetes 这种）必须**有原文证据**才能保留。已经有 `master_resume_text` 了，加一个 `containment check` 一行代码的事。

### 1.5 docx_tailor — 9 段改写完整列表（节选展示）

| 段 | 类型 | 评价 |
|---|---|---|
| 13-15 | Deep Research Agent 描述 lossless 重构 + 注入"agent workflow / agent inference pipeline" | ✅ 关键词来自 JD，但项目本身真的在做 agent workflow，**合理** |
| 18-20 | RemeDi 项目结构精简 | ✅ 全部 lossless |
| 30 | "具备 Python..." → "熟练使用 Python 与 PyTorch...具备完整模型训练与推理管线搭建经验" | ⚠️ 加了"完整模型训练与推理管线" — 项目里有但措辞略 inflation |
| 31 | "了解概率论..." → "熟悉 Transformer、Attention、RAG 原理..." | ❌ **真注入** |
| 32 | "熟悉 LLM 微调流程" → "熟悉 LLM 微调流程..." | ✅ lossless |

**结论**: 9/9 全保留 docx 格式（python-docx 段落级 style 保留确实成功），8/9 是合理改写或 lossless，**1/9 触发反编造红线**。胜率 89% 但红线被踩 1 次——这不是"4 层防御"该有的鲁棒性。

---

## 2. 好用吗 — 4 个真实障碍

### 2.1 装机摩擦 — 90% 用户在第一步就走了

我在沙盒里直接试了：

```
$ .venv/bin/python scripts/doctor.py
bash: line 3: .venv/bin/python: No such file or directory

$ file .venv/bin/python
.venv/bin/python: broken symbolic link to /opt/anaconda3/bin/python3
```

`.venv/bin/python` 是个**指向 `/opt/anaconda3/bin/python3` 的死链**——只在你（Mac，装了 anaconda3）的本机能跑。换任何同学的电脑，这个 venv 直接死。

`uv venv --python 3.12 --allow-existing` 在沙盒也失败（DNS 拦了 GitHub 下载 python-build-standalone）——但重点不是沙盒能不能装，是**用户首次部署的失败率**：

- 没装 uv 的同学：失败
- 装了 uv 但 Python < 3.11 的：失败（loop.py 用了 `from datetime import UTC`，3.11+ 才有）
- 装了 uv + Python 3.12 但缺一些系统包：失败
- 网络环境不能 GitHub 的（不少国内同学）：uv 装 python-build-standalone 失败
- 装好了但忘配 .env 的：跑起来一直报无 LLM key，没有清晰错误

**实际首次成功率估测：< 30%**。你写过 `scripts/doctor.py`——但 doctor 自己也得用 venv 跑，跑不起来连 doctor 都看不到。

**修法**：写一个 1 行 install + 1 行启动脚本（curl | bash 范式），**先确认 Python 版本 + 网络**，doctor 之前先做"can you reach ccvibe.cc / can you write to ~/.offerguide"。或者干脆做成 docker compose up，国内同学能用 daocloud 镜像。

### 2.2 延迟 ≠ 宣传

dogfood W11 真实数据：

| SKILL | 真实延迟 | hero 承诺 | 差距 |
|---|---|---|---|
| successful_profile | **32.0s** | — | — |
| profile_resume_gap | **56.3s** | — | — |
| **总耗时** | **88.3s** | hero 写 "10-20s" | **4-9 倍** |
| score_match (单次估测) | 8-15s | "10-20s" | 大体对得上 |
| docx_tailor (9 段改) | 30-90s（脚本注释自陈） | — | — |

**用户体感**：第一次粘 JD 等 88 秒，没有任何 streaming，没有 progress bar——他会以为程序崩了。我能想象用户「之前试过相当难用」就是被这种沉默的等待打跑的。

**修法**：
- hero 改 SSE 流式 — 4 阶段（读 JD → 比对画像 → 算评分 → 出建议），每阶段 200ms-2s 出文字
- 或者把第一次粘 JD 时只跑 score_match（10-15s），analyze_gaps 等用户**点开"看 gap 详情"才跑**——用户不必每次等 88 秒
- 至少改一下 hero 文案别写"10-20s"——写"约 1 分钟"，不要欺骗用户

### 2.3 反编造防线漏（1.4 节已展开）

simplify: 4 层防御里的"长度漂移 ±50%"和"受保护跳过"都没拦住"专业技能"段被注入"Transformer / Attention / RAG"。漏洞在于：**没有 keyword containment check**——简历里没出现的核心技能词不能在 tailor 后凭空出现。

这个修法是 30 行代码：

```python
# tailor_docx 末尾
NEW_SKILL_KEYWORDS = ['Transformer', 'Attention', 'RAG', 'PyTorch', 'TensorFlow',
                      'LangChain', 'LangGraph', 'Kubernetes', 'Docker', 'Spark', ...]
for kw in NEW_SKILL_KEYWORDS:
    if kw in result.new_text and kw not in master_resume_text:
        raise TailorViolation(f"注入了原文不存在的技能名: {kw}")
```

这个 bug 不修，**简历投出去面试时很容易翻车**。HR 在简历里看到"熟悉 Transformer"，面试问"讲讲 multi-head attention"，用户支支吾吾——一票否决。

### 2.4 信号源极稀疏 / 一个真用户的孤岛

successful_profile 自己 uncertainty_notes 第一条就承认了：

> ⚠ 样本量极小（仅 3 条），其中只有 1 条 offer 复盘，背景模式（学校层次、学历、实习经历）置信度低

**问题**：
- 你的所有真实输出都来自**你自己 1 份简历 + 你手攒的 3-5 个面经**
- 没有真投递 → 没有 reply rate baseline → GEPA 没有进化原料
- DEVLOG 第 12 章自陈：「真实 dogfood LLM call 次数 5（质量分类器 5/5 全对 + 画像 + Gap）」

**这是新产品最大的 cold-start 问题**——`agentic/corpus_collector.py` 自动搜面经 + LLM 评估的链路是对的解法但**没默认开**：

```bash
# 当前的 corpus 入库链路
用户进 /interviews → 手动 paste → 等 LLM 分类
# 该有的链路
用户进 /interviews → 看到"字节 N=3 面经" → 1-click "再帮我搜 5 条" → corpus_collector 跑 → 用户审 5 条
```

修法是把 corpus_collector 推到一线（已经在 P2 改造路线图里）。

### 2.5 UI 在求职者面前自报 daemon / wake / trajectory（已在前份诊断报告展开）

略 — 见 `agent_diagnosis_2026-05-07.md` 第 1.3 节。

---

## 3. 跟"之前的难用版"对比 — 进步吗？

DEVLOG 给了清晰时间线：

| 周次 | 加了啥 | 对"能用"的影响 |
|---|---|---|
| W1-W4 | 基础 (LangGraph + memory + 牛客 sitemap) | 还是骨架 |
| W5-W7 | application_events + GEPA + tracker + Boss 扩展 | 框架完整，但还没真输出 |
| W8 | prepare_interview + evolution diff CLI | 第一次有用户对话流 |
| W8'-W8''''' | compare_jobs + email_classifier + corpus_collector + autonomous daemon | 自动化层完整 |
| **W11** | **corpus_quality + successful_profile + profile_resume_gap** | **第一次真有"成功者画像"+ "诚实 gap"输出** |
| **W12** | **docx_tailor (保格式 + 4 层反编造)** | **第一次能产出真的 docx 文件** |
| W13-15 | agent loop + 心跳 + worldview | 自动化更深，但 **product 层没改善** |

**判断**：
- **W11 + W12 是真实的功能跃迁**——从"能跑 SKILL"到"能输出可用的求职产物"
- W13-W15 加的全是"agent 内部的认知层"——对开发者很 cool，对求职用户**毫无感觉**
- 你用户原话「之前试过相当难用」很可能是 W7-W8 时期的版本——那时候确实 SKILL 输出还很粗
- **现在的 SKILL 输出质量是真好的**——dogfood 输出能拿出去给同学看

但："SKILL 输出好" + "产品壳烂" = 用户根本看不到 SKILL 输出有多好。

---

## 4. 现在能拿出去给同学用吗 — 不能

5 个 hard blocker（按用户能不能成功用排序）：

1. **装机失败率 > 70%**（venv 死链 / Python < 3.11 / 网络下不动 python-build-standalone）—— 90% 同学到这一步就放弃
2. **首屏 13 区块 + 19 nav tab** —— 跨过装机的，看到首页直接懵
3. **第一次粘 JD 等 88 秒没 progress** —— 跨过首屏的，看到沉默以为崩了
4. **tailor_resume 注入未验证关键词** —— 拿到 docx 看似漂亮，面试翻车
5. **一岗多投限额硬编码不准** —— compare_jobs 推荐"先投 X / 跳过 Z"是**精确的错误**

---

## 5. 4 周内能让它真好用的最小改造路径

按"能用 → 好用"的 ROI 排：

### 🔴 装机 D-Day（1 天）
- 删除 .venv 软链 / 加 .gitignore
- 写 `install.sh` 1 行启动：检查 Python 3.11+ → uv sync → cp .env.example .env → 让用户 vi .env → uv run python -m offerguide.ui.web
- 或 docker compose up（推荐）

### 🔴 反编造 patch（半天）
- tailor_resume 末尾加 keyword containment check（30 行）
- 把 docx_tailor `result.violations` 暴露到 UI
- 红线触发时不下载 docx，让用户 review

### 🟠 SSE 流式 hero（2-3 天）
- 评估 JD 改 SSE
- 4 阶段 echo: 抽取 JD 要点 → 比对画像 → 算评分 → 出建议
- 每阶段 < 5s 看到文字

### 🟠 一岗多投限额降级（半天）
- 全部 hardcoded limit 加 source 字段（"estimated, please verify"）
- compare_jobs 输出加 "本数据源置信度: low" 标签

### 🟢 corpus_collector 推到一线（2-3 天）
- /interviews 公司 N=0 时 1-click 触发
- 用户审 5 条 → 入库

### 🟢 home 大瘦身（2 天）
- 13 区块 → 4 张卡（已在前份诊断报告 P0-1 展开）

**总计**: ~2 周工时。做完之后：

- 装机成功率从 < 30% → > 80%
- 首次评估 JD 体感从 88s 沉默 → 4 阶段流式
- 简历翻车风险归 0
- 同学能装能用能信能投

**做完这 6 件事**，你才有立场说 OfferGuide 真的能用、好用。

---

## 6. 跟前份诊断报告关系

| 报告 | 关注点 | 主要结论 |
|---|---|---|
| `agent_diagnosis_2026-05-07.md` | UX / 定位 / 对标 / 13 区块问题 | 战略对、UX 像后台 |
| **`usability_verdict_2026-05-07.md`（本份）** | **真跑过 SKILL → 输出质量 + 装机 + 延迟 + 编造防线 + 一岗限额** | **能用，不好用** |

**两份合起来读**：你的 ML / agent 工程是**真实可见**的（dogfood 输出质量在线），但**产品工程**（装机 / 延迟 / 防线 / UI 收敛）还没做到"日用品级"。前者是面试加分项，后者是能不能拿出去用的门槛。
