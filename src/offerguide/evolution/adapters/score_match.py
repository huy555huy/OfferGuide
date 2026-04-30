"""``score_match`` SKILL adapter.

This is the canonical adapter — was originally hand-written as
``golden_trainset.py`` + ``metrics.py``, then refactored here when
analyze_gaps and prepare_interview joined the evolvable set.

Score components for ``score_match`` (weights chosen so a model that hits
all three at full strength gets 1.0; one that nails probability but
misses reasoning anchors gets ~0.5):

- 0.5 × probability_in_band — does ``prediction.probability`` fall inside
  the golden's ``expected_probability_range``? Linear penalty outside.
- 0.3 × reasoning_recall    — fraction of ``must_mention`` substrings
  that appear in ``prediction.reasoning``. Catches "got the score right
  by accident but didn't identify the actual driver."
- 0.2 × anti_false_positive — penalty for any ``must_not_mention``
  substring appearing — the model claimed something we know is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._base import MetricBreakdown, parse_json_output

name: str = "score_match"
INPUT_NAMES: list[str] = ["job_text", "user_profile"]
METRIC_AXES: list[str] = ["total", "prob", "recall", "anti"]

# Weights chosen by inspection. Sum to 1.0.
_W_PROB = 0.5
_W_RECALL = 0.3
_W_ANTI = 0.2


@dataclass(frozen=True)
class ScoreMatchExample:
    """One ground-truth case for evaluating score_match outputs."""

    name: str
    """Stable id (snake_case). Stable across reruns so deltas are comparable."""

    job_text: str
    user_profile: str
    expected_probability_range: tuple[float, float]
    """(low, high) — predictions inside this band score full marks on the
    probability axis. Width reflects how confident a competent recruiter
    would be."""

    must_mention: tuple[str, ...] = ()
    """Substrings that should appear in ``reasoning``. Pick concrete
    keywords that prove the model identified the right factor."""

    must_not_mention: tuple[str, ...] = ()
    """Substrings that betray failure modes — e.g. praising a missing skill."""

    band: str = "middle"
    """'fit' | 'misfit' | 'middle' — used for stratified split."""

    notes: str = ""


# ── examples ───────────────────────────────────────────────────────

_USER_PROFILE_BASELINE = """胡阳，上海财经大学 应用统计专硕（2025-2027 在读），本科南京审计大学统计学。
技能：Python、PyTorch、SFT/LoRA、HuggingFace Transformers/TRL、强化学习（PPO/GRPO）、
扩散模型 (Diffusion Model)、AI Agent 架构。
项目经历：
1. Deep Research Agent（法至科技实习，12/2025-04/2026）—— 单 agent runtime / 双层架构
   （语义层 + 工作区层）/ evidence-centric 上下文机制 / closure semantics
2. RemeDi（11/2025）—— 基于 LLaDA-8B 的扩散语言模型，双流架构 (TPS + UPS) + 零初始化
   投影 + LoRA 微调；SFT 训练管线 + GRPO 强化学习；DeepSpeed ZeRO-2 + 梯度累积。
偏好：AI Agent / LLM 应用算法岗；地点上海/北京/杭州；2026 暑期实习。"""


EXAMPLES: tuple[ScoreMatchExample, ...] = (
    # ── 强匹配 (band="fit") ──
    ScoreMatchExample(
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
        must_mention=("Agent", "PyTorch", "C"),
        must_not_mention=("不熟悉 Agent", "缺少 LLM 经验"),
        notes="用户两个项目分别命中 agent 架构和 LLM 训练管线；C/C++ 是真实 gap。",
    ),
    ScoreMatchExample(
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
        notes="RemeDi 项目逐项命中 (SFT/GRPO/DeepSpeed/扩散)。",
    ),
    ScoreMatchExample(
        name="agent_application_dev",
        band="fit",
        job_text=(
            "AI Agent 应用开发实习生 — 某 AI 创业公司 — 上海\n"
            "职责：\n"
            "- 用 LangGraph / DSPy 搭建垂直场景 Agent（金融、教育等）\n"
            "- 与业务方对齐需求，迭代 Agent 工作流和评测\n"
            "要求：\n"
            "- 熟悉 Python，至少做过 1 个 Agent / LLM 应用 demo\n"
            "- 了解 RAG / Memory / Tool Use 等概念"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.62, 0.82),
        must_mention=("Agent", "LangGraph"),
        must_not_mention=("缺乏 Python 经验",),
        notes="Deep Research Agent 项目 + AI Agent 架构能力直接对齐。",
    ),
    # ── 弱匹配 (band="misfit") ──
    ScoreMatchExample(
        name="senior_backend_5yr",
        band="misfit",
        job_text=(
            "后端高级开发工程师 — 5 年以上经验\n"
            "职责：\n"
            "- 主导大型分布式系统设计\n"
            "- 带领 3-5 人团队\n"
            "要求：\n"
            "- 5 年以上 Java/Go 后端开发经验\n"
            "- 精通 MySQL / Redis / Kafka 等"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.02, 0.10),
        must_mention=("5 年", "应届"),
        must_not_mention=(),
        notes="硬性年限 + 技术栈不符。Deal-breaker 必须命中。",
    ),
    ScoreMatchExample(
        name="frontend_react_intern",
        band="misfit",
        job_text=(
            "前端实习生（React + TypeScript）— 某中厂 — 上海\n"
            "职责：\n"
            "- 中后台 H5 页面开发\n"
            "要求：\n"
            "- 熟练 React / TypeScript / Webpack / Vite\n"
            "- 关注 Web 性能与可访问性"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.08, 0.22),
        must_mention=("前端", "React"),
        must_not_mention=("强匹配", "高度契合"),
        notes="技术栈完全不沾，但城市/学历对得上。",
    ),
    ScoreMatchExample(
        name="dba_role",
        band="misfit",
        job_text=(
            "数据库管理员 (DBA) 校招\n"
            "职责：\n"
            "- 维护生产数据库（MySQL / Oracle / PostgreSQL）\n"
            "- 性能调优、备份、恢复\n"
            "要求：\n"
            "- 计算机相关专业本科\n"
            "- 至少做过一个数据库相关项目"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.05, 0.18),
        must_mention=("DBA", "数据库"),
        must_not_mention=(),
        notes="方向不符；岗位天职完全不同。",
    ),
    # ── 中等匹配 (band="middle") ──
    ScoreMatchExample(
        name="data_analyst_internet",
        band="middle",
        job_text=(
            "数据分析实习生 — 某互联网公司 — 上海\n"
            "职责：\n"
            "- 业务指标拉取与归因分析\n"
            "- A/B 实验设计与结果解读\n"
            "要求：\n"
            "- 统计/数学/计算机相关专业\n"
            "- 熟悉 SQL，会用 Python (pandas)\n"
            "- 有过 A/B 实验或假设检验经验加分"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.32, 0.55),
        must_mention=("统计",),
        must_not_mention=("完全不符",),
        notes="专业对得上但用户简历偏 Agent / 训练，业务数据分析弱。",
    ),
    ScoreMatchExample(
        name="ml_research_phd_pref",
        band="middle",
        job_text=(
            "机器学习研究实习生（博士优先）— 某研究院 — 北京\n"
            "职责：\n"
            "- 大模型推理优化前沿研究\n"
            "- 论文撰写与开源项目贡献\n"
            "要求：\n"
            "- AI / CS 相关博士在读优先（硕士可接受）\n"
            "- 有 NeurIPS/ICLR/ICML 论文经验加分"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.25, 0.50),
        must_mention=("硕士", "博士"),
        must_not_mention=("不匹配技术",),
        notes="专硕 vs 博士优先；技术栈相关但科研产出未必够。",
    ),
    ScoreMatchExample(
        name="quant_research_intern",
        band="middle",
        job_text=(
            "量化研究实习生 — 某私募 — 上海\n"
            "职责：\n"
            "- 因子挖掘与回测\n"
            "- 时序模型与机器学习方法在金融中的应用\n"
            "要求：\n"
            "- 数学/统计/计算机/金融工程相关专业\n"
            "- 熟练 Python，了解时序分析"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.30, 0.55),
        must_mention=("统计",),
        must_not_mention=("没有金融经验" "因此完全不符",),
        notes="专业 + 时序背景对得上，但简历无量化项目。",
    ),
    ScoreMatchExample(
        name="agent_eval_engineer",
        band="middle",
        job_text=(
            "Agent 评测工程师 — 某大模型公司 — 杭州\n"
            "职责：\n"
            "- 设计 Agent 多维度评测指标\n"
            "- 维护评测平台与 benchmark\n"
            "要求：\n"
            "- CS/AI/数学相关专业本科及以上\n"
            "- 至少做过一个 Agent / LLM 应用项目\n"
            "- 工程能力扎实，会写自动化测试"
        ),
        user_profile=_USER_PROFILE_BASELINE,
        expected_probability_range=(0.45, 0.68),
        must_mention=("Agent", "评测"),
        must_not_mention=(),
        notes="Agent 经验对得上，评测/自动化是相对弱项。",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


def metric(
    example: ScoreMatchExample,
    raw_output: str | dict[str, Any],
) -> MetricBreakdown:
    """Score a single (example, output) pair.

    ``raw_output`` accepts either the parsed dict or the raw JSON string.
    """
    parsed = parse_json_output(raw_output)
    if not parsed:
        return MetricBreakdown(
            total=0.0,
            breakdown={"prob": 0.0, "recall": 0.0, "anti": 0.0},
            feedback=(
                f"案例: {example.name} ({example.band})\n"
                "OUTPUT_PARSE_FAILURE: model did not emit valid JSON for score_match."
            ),
        )

    prob_score, prob_note = _score_probability(parsed, example)
    recall_score, recall_note = _score_recall(parsed, example)
    anti_score, anti_note = _score_anti_false_positive(parsed, example)

    total = _W_PROB * prob_score + _W_RECALL * recall_score + _W_ANTI * anti_score

    feedback_parts = [f"案例: {example.name} ({example.band})", prob_note]
    if example.must_mention:
        feedback_parts.append(recall_note)
    if example.must_not_mention:
        feedback_parts.append(anti_note)
    feedback_parts.append(
        f"score: prob={prob_score:.2f} recall={recall_score:.2f} anti={anti_score:.2f}"
        f" → total={total:.2f}"
    )

    return MetricBreakdown(
        total=total,
        breakdown={"prob": prob_score, "recall": recall_score, "anti": anti_score},
        feedback="\n".join(feedback_parts),
    )


def _score_probability(parsed: dict, example: ScoreMatchExample) -> tuple[float, str]:
    raw = parsed.get("probability")
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        return 0.0, f"probability 字段缺失或非数值（实际: {raw!r}）"
    if not 0.0 <= prob <= 1.0:
        return 0.0, f"probability 越界 [0,1]（实际 {prob:.3f}）"

    low, high = example.expected_probability_range
    if low <= prob <= high:
        return 1.0, f"probability {prob:.2f} 落在期望区间 [{low:.2f}, {high:.2f}] ✓"
    dist = min(abs(prob - low), abs(prob - high))
    score = max(0.0, 1.0 - 4.0 * dist)
    direction = "偏低" if prob < low else "偏高"
    return (
        score,
        f"probability {prob:.2f} {direction}于期望 [{low:.2f}, {high:.2f}]，"
        f"距离 {dist:.2f} → 分 {score:.2f}",
    )


def _score_recall(parsed: dict, example: ScoreMatchExample) -> tuple[float, str]:
    if not example.must_mention:
        return 1.0, ""
    reasoning = str(parsed.get("reasoning") or "")
    deal_breakers = " ".join(str(x) for x in (parsed.get("deal_breakers") or []))
    haystack = reasoning + "\n" + deal_breakers
    hits = [k for k in example.must_mention if k in haystack]
    misses = [k for k in example.must_mention if k not in haystack]
    score = len(hits) / len(example.must_mention)
    note = (
        f"reasoning 命中所有关键点 {list(example.must_mention)} ✓"
        if not misses
        else f"reasoning 未命中: {misses}（必须指出）"
    )
    return score, note


def _score_anti_false_positive(
    parsed: dict, example: ScoreMatchExample
) -> tuple[float, str]:
    if not example.must_not_mention:
        return 1.0, ""
    reasoning = str(parsed.get("reasoning") or "")
    false_pos = [k for k in example.must_not_mention if k in reasoning]
    if not false_pos:
        return 1.0, "reasoning 没有触发反向词 ✓"
    score = max(0.0, 1.0 - 0.5 * len(false_pos))
    return score, f"reasoning 触发反向词: {false_pos} → 扣分 {score:.2f}"


# ── train/val split ────────────────────────────────────────────────


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[ScoreMatchExample], list[ScoreMatchExample]]:
    """Stratified by ``band`` so each split has a balance of fit/misfit/middle."""
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[ScoreMatchExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[ScoreMatchExample] = []
    val: list[ScoreMatchExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
