"""Hand-curated golden examples for the score_match SKILL.

Why hand-curated rather than auto-mined from skill_runs:

The 4-week dogfood window hasn't started yet, so we have **zero** real
reply-rate labels. Per the user's W6 brief ("即使样本少也跳"), we still want
to run GEPA — but the optimizer needs a defensible metric, and a metric needs
ground truth.

Solution: a small (8-12) set of carefully-chosen (JD, profile, expected
probability range, must-mention rationale, must-NOT-mention false-positives)
cases that span the decision space — clear-fit, clear-misfit, and ambiguous-
middle. These are calibrated by inspection: we know what a competent human
recruiter would say, and we encode that as the target.

When real dogfood data lands (Phase 3), this file gets supplemented with
mined examples whose targets are derived from actual reply rates. The format
of each example stays the same, so swap-in is mechanical.

Sources for the example JDs:
- 牛客 fixture (446211) AI Agent @ 阿里巴巴 — public sitemap data
- Distilled / paraphrased typical校招 JDs — no proprietary content
- Synthetic edge cases (year mismatch, location mismatch, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenExample:
    """One ground-truth case for evaluating score_match outputs."""

    name: str
    """Stable id (snake_case). Stable across reruns so deltas are comparable."""

    job_text: str
    user_profile: str
    expected_probability_range: tuple[float, float]
    """(low, high) — predictions inside this band score full marks on the
    probability axis. Width reflects how confident a competent recruiter would
    be about the call. Tight bands for clear cases (~0.10 wide), wider bands
    (~0.20-0.30) for ambiguous middle cases."""

    must_mention: tuple[str, ...] = ()
    """Substrings that should appear in the model's `reasoning`. Pick concrete
    keywords that prove the model identified the right factor (e.g. 'C++' if
    that's a missing requirement; 'PyTorch' if that's a strong fit)."""

    must_not_mention: tuple[str, ...] = ()
    """Substrings that betray failure modes — e.g. praising a missing skill, or
    asserting a deal-breaker that doesn't exist. Anti-pattern guard."""

    band: str = "middle"
    """'fit' | 'misfit' | 'middle' — used for split balancing in train/val."""

    notes: str = ""
    """Free-form rationale for why this case has its target — for human review."""


# The user's actual profile context (paraphrased from the resume that's
# already wired through the W1 PDF loader). Avoids leaking specific contact
# info into version control.
_USER_PROFILE_BASELINE = """胡阳，上海财经大学 应用统计专硕（2025-2027 在读），本科南京审计大学统计学。
技能：Python、PyTorch、SFT/LoRA、HuggingFace Transformers/TRL、强化学习（PPO/GRPO）、
扩散模型 (Diffusion Model)、AI Agent 架构。
项目经历：
1. Deep Research Agent（法至科技实习，12/2025-04/2026）—— 单 agent runtime / 双层架构
   （语义层 + 工作区层）/ evidence-centric 上下文机制 / closure semantics
2. RemeDi（11/2025）—— 基于 LLaDA-8B 的扩散语言模型，双流架构 (TPS + UPS) + 零初始化
   投影 + LoRA 微调；SFT 训练管线 + GRPO 强化学习；DeepSpeed ZeRO-2 + 梯度累积。
偏好：AI Agent / LLM 应用算法岗；地点上海/北京/杭州；2026 暑期实习。"""


GOLDEN_EXAMPLES: tuple[GoldenExample, ...] = (
    # ---------- 强匹配（band="fit"） ----------
    GoldenExample(
        name="ali_ai_agent_intern",
        band="fit",
        job_text=(
            "AI Agent 实习生（2027届）— 阿里巴巴集团 — 北京 — 15-35K × 16\n"
            "## 岗位职责\n"
            "1. 研发并优化 Agent 的核心能力模块——自主规划（Planning）、多步推理、"
            "工具调用 (Tool Use)、长短期记忆 (Memory)、RAG 增强；\n"
            "2. 应用 RL、SFT、偏好对齐 (DPO/PPO) 提升复杂任务执行成功率；\n"
            "3. 构建多 Agent 协同与自我迭代学习范式；\n"
            "4. 智能体评测体系。\n"
            "## 任职要求\n"
            "1. 计算机/AI/数学/电信相关专业本科及以上；\n"
            "2. 精通 Python，扎实的 C/C++/Java（至少一门）；\n"
            "3. 深入理解 Transformer 与 LLM；熟悉 PyTorch；掌握 RL/NLP/CV 基础理论；\n"
            "4. 求知欲、逻辑、沟通能力。"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.55, 0.78),
        must_mention=("Agent", "PyTorch", "C"),  # 简历有 Agent 项目和 PyTorch；C/C++ 是 gap
        must_not_mention=("不熟悉 Agent", "缺少 LLM 经验"),
        notes="用户两个项目分别命中 agent 架构和 LLM 训练管线；C/C++ 是真实 gap，但单一 gap 不足以一票否决。",
    ),
    GoldenExample(
        name="bytedance_llm_post_training",
        band="fit",
        job_text=(
            "大模型后训练算法实习生 — 字节跳动 Seed — 上海/北京 — 18-40K × 14\n"
            "职责：\n"
            "- 大模型 SFT / DPO / GRPO / RLHF 流程优化\n"
            "- 数据合成 + 偏好数据构建\n"
            "- 训练框架 (DeepSpeed/Megatron) 调优\n"
            "要求：\n"
            "- 计算机 / AI / 数学相关专业\n"
            "- 熟练 PyTorch，理解 Transformer 内部机制\n"
            "- 至少有一段 LLM 训练或 RL 训练相关项目经验"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.65, 0.85),
        must_mention=("RemeDi", "GRPO", "DeepSpeed"),
        must_not_mention=("没有训练经验", "需要补充 SFT"),
        notes="RemeDi 项目几乎逐项命中 JD 要求（SFT/GRPO/DeepSpeed/扩散）。",
    ),
    GoldenExample(
        name="agent_application_dev",
        band="fit",
        job_text=(
            "AI Agent 应用开发实习生 — 某 AI 创业公司 — 上海\n"
            "职责：\n"
            "- 用 LangGraph / DSPy 搭建垂直场景 Agent（金融、教育等）\n"
            "- 与业务方对齐需求，迭代 Agent 工作流和评测\n"
            "- 持续优化 prompt 与 tool 设计\n"
            "要求：\n"
            "- 熟悉 Python，至少做过 1 个 Agent / LLM 应用 demo\n"
            "- 了解 RAG / Memory / Tool Use 等概念\n"
            "- 良好沟通能力"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.62, 0.82),
        must_mention=("Agent", "LangGraph"),
        must_not_mention=("缺乏 Python 经验",),
        notes="Deep Research Agent 项目 + AI Agent 架构能力直接对齐。",
    ),

    # ---------- 弱匹配（band="misfit"） ----------
    GoldenExample(
        name="senior_backend_5yr",
        band="misfit",
        job_text=(
            "后端高级开发工程师 — 5 年以上经验\n"
            "职责：\n"
            "- 主导大型分布式系统设计\n"
            "- 带领 3-5 人团队\n"
            "要求：\n"
            "- 5 年以上 Java/Go 后端开发经验\n"
            "- 精通 MySQL / Redis / Kafka 等\n"
            "- 有大型互联网公司架构经验"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.02, 0.10),
        must_mention=("5 年", "应届",),  # 必须指出年限不符
        must_not_mention=(),
        notes="硬性年限 + 技术栈不符。Deal-breaker 必须命中。",
    ),
    GoldenExample(
        name="frontend_react_intern",
        band="misfit",
        job_text=(
            "前端实习生（React + TypeScript）— 某中厂 — 上海\n"
            "职责：\n"
            "- 中后台 H5 页面开发\n"
            "要求：\n"
            "- 熟练 React / TypeScript / Webpack / Vite\n"
            "- 关注 Web 性能与可访问性\n"
            "- 计算机相关专业本科及以上"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.08, 0.22),
        must_mention=("前端", "React"),  # 技术栈完全不沾必须指出
        must_not_mention=("强匹配", "高度契合"),
        notes="技术栈完全不沾。但是城市/学历对得上，所以不是 0.02。",
    ),
    GoldenExample(
        name="dba_role",
        band="misfit",
        job_text=(
            "数据库管理员 (DBA) 校招\n"
            "职责：\n"
            "- 维护生产数据库（MySQL / Oracle / PostgreSQL）\n"
            "- 性能调优、备份、恢复\n"
            "要求：\n"
            "- 计算机相关专业本科\n"
            "- 至少做过一个数据库相关项目\n"
            "- 熟悉 SQL 调优"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.05, 0.18),
        must_mention=("DBA", "数据库"),
        must_not_mention=(),
        notes="方向不符；应届背景对得上但岗位天职完全不同。",
    ),

    # ---------- 中等匹配（band="middle"） ----------
    GoldenExample(
        name="data_analyst_internet",
        band="middle",
        job_text=(
            "数据分析实习生 — 某互联网公司 — 上海\n"
            "职责：\n"
            "- 业务指标拉取与归因分析\n"
            "- A/B 实验设计与结果解读\n"
            "- 撰写分析报告\n"
            "要求：\n"
            "- 统计/数学/计算机相关专业\n"
            "- 熟悉 SQL，会用 Python (pandas)\n"
            "- 有过 A/B 实验或假设检验经验加分"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.32, 0.55),
        must_mention=("统计",),
        must_not_mention=("完全不符",),
        notes="专业对得上但用户简历偏 Agent / 训练，业务数据分析弱，回复率中等偏下。",
    ),
    GoldenExample(
        name="ml_research_phd_pref",
        band="middle",
        job_text=(
            "机器学习研究实习生（博士优先）— 某研究院 — 北京\n"
            "职责：\n"
            "- 大模型推理优化前沿研究\n"
            "- 论文撰写与开源项目贡献\n"
            "要求：\n"
            "- AI / CS 相关博士在读优先（硕士可接受）\n"
            "- 有 NeurIPS/ICLR/ICML 论文经验加分\n"
            "- PyTorch 熟练，能独立设计实验"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.25, 0.50),
        must_mention=("硕士", "博士"),
        must_not_mention=("不匹配技术",),
        notes="专硕 vs 博士优先；技术栈相关但科研产出未必够。回复率中等偏下。",
    ),
    GoldenExample(
        name="quant_research_intern",
        band="middle",
        job_text=(
            "量化研究实习生 — 某私募 — 上海\n"
            "职责：\n"
            "- 因子挖掘与回测\n"
            "- 时序模型与机器学习方法在金融中的应用\n"
            "要求：\n"
            "- 数学/统计/计算机/金融工程相关专业\n"
            "- 熟练 Python，了解时序分析\n"
            "- 有相关项目或竞赛经验加分"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.30, 0.55),
        must_mention=("统计",),
        must_not_mention=("没有金融经验" "因此完全不符",),
        notes="专业 + 时序背景对得上，但简历无量化项目，回复率中等。",
    ),
    GoldenExample(
        name="agent_eval_engineer",
        band="middle",
        job_text=(
            "Agent 评测工程师 — 某大模型公司 — 杭州\n"
            "职责：\n"
            "- 设计 Agent 多维度评测指标\n"
            "- 维护评测平台与 benchmark\n"
            "- 与算法团队协同迭代\n"
            "要求：\n"
            "- CS/AI/数学相关专业本科及以上\n"
            "- 至少做过一个 Agent / LLM 应用项目\n"
            "- 工程能力扎实，会写自动化测试"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.45, 0.68),
        must_mention=("Agent", "评测"),
        must_not_mention=(),
        notes="Agent 经验对得上，评测/自动化是相对弱项。回复率中等偏上。",
    ),
)


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[GoldenExample], list[GoldenExample]]:
    """Deterministic train/val split. Stratified by `band` so each split has
    a balance of fit/misfit/middle cases (otherwise GEPA could overfit one
    band's prompt-style and look great on val without generalizing).

    `seed` is honored only as a tie-breaker; the primary order is `name` so
    deltas across runs are comparable.
    """
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[GoldenExample]] = {}
    for ex in sorted(GOLDEN_EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[GoldenExample] = []
    val: list[GoldenExample] = []
    for _band, examples in sorted(by_band.items()):
        n_val = max(1, round(len(examples) * val_fraction))
        # Deterministic shuffle within band so split is stable but not adversarial
        shuffled = examples[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
