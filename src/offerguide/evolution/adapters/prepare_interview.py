"""``prepare_interview`` SKILL adapter.

Without dogfood data on which predicted questions actually got asked in
real interviews, the metric optimizes for **shape correctness** that we
know correlates with usefulness:

- ``schema``           — output validates against ``PrepareInterviewResult``
                          (extra='forbid', likelihood ∈ [0,1]).
- ``coverage``         — fraction of categories covered (target ≥ 3 of 5).
                          School recruiting interviews mix categories;
                          all-technical or all-behavioral fails users.
- ``count_in_range``   — ``len(expected_questions)`` ∈ [5, 8]. Too few =
                          shallow prep; too many = won't memorize all.
- ``rationale_grounded`` — fraction of rationales that mention either a
                          ``profile_keyword`` or a ``jd_keyword``. Anti-
                          hallucination — if the model writes "this is a
                          common big-tech question" with no anchor in the
                          JD or profile, it's a content-free placeholder.
- ``calibration_signal`` — penalty when all likelihoods are clustered at
                          the same value (e.g. every question = 0.7).
                          That's the SKILL prompt failing to produce
                          calibrated estimates.

When real dogfood data lands (~30 interview reflections), replace
``rationale_grounded`` with ``actual_question_hit_rate`` from the user's
reflection notes.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from pydantic import ValidationError

from ...skills.prepare_interview.helpers import PrepareInterviewResult
from ._base import MetricBreakdown, parse_json_output

name: str = "prepare_interview"
INPUT_NAMES: list[str] = ["company", "job_text", "user_profile", "past_experiences"]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "coverage",
    "count",
    "grounded",
    "calibration",
]

_W_SCHEMA = 0.20
_W_COVERAGE = 0.25
_W_COUNT = 0.10
_W_GROUNDED = 0.30
_W_CALIBRATION = 0.15


@dataclass(frozen=True)
class PrepareInterviewExample:
    """One ground-truth case for prepare_interview."""

    name: str
    company: str
    job_text: str
    user_profile: str
    past_experiences: str = ""

    profile_keywords: tuple[str, ...] = ()
    """Keywords from the user's profile that *some* rationale should
    reference (e.g. 'RemeDi', 'Deep Research Agent', 'PyTorch')."""

    jd_keywords: tuple[str, ...] = ()
    """Keywords from the JD that *some* rationale should reference."""

    band: str = "with_面经"
    """'with_面经' | 'no_面经' | 'edge_case' for stratification."""

    notes: str = ""


_USER = """胡阳，上海财经大学 应用统计专硕。
项目: Deep Research Agent (法至科技实习, agent runtime + 双层架构)；
RemeDi (LLaDA-8B 扩散模型, SFT + GRPO)。
技能: Python, PyTorch, Transformers, RL (PPO/GRPO), AI Agent 架构。"""


EXAMPLES: tuple[PrepareInterviewExample, ...] = (
    PrepareInterviewExample(
        name="bytedance_seed_with_面经",
        band="with_面经",
        company="字节跳动",
        job_text=(
            "大模型后训练算法实习生 — 字节 Seed\n"
            "要求: PyTorch, SFT/DPO/GRPO 经验, DeepSpeed/Megatron, Transformer 内部机制"
        ),
        user_profile=_USER,
        past_experiences=(
            "面经 1 · 字节 Seed: 一面问 attention 缩放推导、KV cache、为什么 RoPE。\n"
            "面经 2 · 字节 Seed: 二面问 RLHF 训练不稳怎么调；项目深挖。\n"
            "面经 3 · 字节 Seed: 终面 HR 问职业规划。"
        ),
        profile_keywords=("RemeDi", "GRPO", "DeepSpeed"),
        jd_keywords=("PyTorch", "Transformer", "Megatron"),
        notes="有 3 篇面经，likelihood 应该有信号 (≥ 0.7 出现)。",
    ),
    PrepareInterviewExample(
        name="ali_agent_no_面经",
        band="no_面经",
        company="阿里巴巴",
        job_text=(
            "AI Agent 实习生 — 阿里\n"
            "要求: Python + C/C++/Java, Transformer + LLM, PyTorch, Agent 项目"
        ),
        user_profile=_USER,
        past_experiences="",
        profile_keywords=("Deep Research Agent", "Agent", "PyTorch"),
        jd_keywords=("C/C++", "Transformer", "Agent", "LLM"),
        notes="无面经，likelihood 应该整体偏低 (no peak above 0.75)。",
    ),
    PrepareInterviewExample(
        name="tencent_llm_app_no_面经",
        band="no_面经",
        company="腾讯",
        job_text=(
            "LLM 应用工程师 — 腾讯\n"
            "要求: 大模型应用落地, RAG / 多模态 / Agent 任一; Python; 有 2C 产品经验加分"
        ),
        user_profile=_USER,
        past_experiences="",
        profile_keywords=("Agent", "RemeDi"),
        jd_keywords=("RAG", "Agent", "Python"),
        notes="国内大厂业务岗，无面经；likelihood 应分散 (calibration)。",
    ),
    PrepareInterviewExample(
        name="quant_finance_edge",
        band="edge_case",
        company="某私募",
        job_text=(
            "量化研究实习 — 某私募\n"
            "要求: Python, 时序分析, Alpha 因子挖掘, 回测平台经验"
        ),
        user_profile=_USER,
        past_experiences="",
        profile_keywords=("Python",),
        jd_keywords=("Alpha", "回测", "时序"),
        notes="技术栈半沾，应突出用户弱点 (weak_spots)，category 应 ≥ 3。",
    ),
    PrepareInterviewExample(
        name="ali_with_面经",
        band="with_面经",
        company="阿里巴巴",
        job_text=(
            "Agent 评测工程师 — 阿里达摩院\n"
            "要求: Agent 项目经验，评测 / benchmark 设计；Python；LangGraph/AutoGen 加分"
        ),
        user_profile=_USER,
        past_experiences=(
            "面经 1 · 阿里达摩院 Agent: 一面写 OOD 评测脚本；问 BLEU vs LLM-as-judge。\n"
            "面经 2 · 阿里达摩院: 二面深挖项目，问 agent loop 平均迭代多少次。"
        ),
        profile_keywords=("Deep Research Agent", "agent"),
        jd_keywords=("Agent", "评测", "Python"),
        notes="有面经，project_deep_dive 类必须出现 (loop 迭代次数)。",
    ),
    PrepareInterviewExample(
        name="meituan_llm_app_no_面经",
        band="no_面经",
        company="美团",
        job_text=(
            "LLM 应用算法实习 — 美团\n"
            "要求: 一段以上 LLM 应用项目; Python; 业务理解能力; 有 search/推荐背景加分"
        ),
        user_profile=_USER,
        past_experiences="",
        profile_keywords=("RemeDi", "Agent"),
        jd_keywords=("LLM", "Python"),
        notes="O2O 大厂业务岗，company_specific 必须问外卖/团购业务。",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


def metric(
    example: PrepareInterviewExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    """Score one (example, output) pair on 5 axes."""
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: model did not emit valid JSON.")

    # Schema
    schema_score = 0.0
    schema_note = ""
    result: PrepareInterviewResult | None = None
    try:
        result = PrepareInterviewResult.model_validate(parsed)
        schema_score = 1.0
        schema_note = "schema 通过 ✓"
    except ValidationError as e:
        err = e.errors()[0]
        loc = ".".join(str(x) for x in err.get("loc", []))
        schema_note = f"schema 失败: {loc}: {err.get('msg', 'validation failed')}"

    if result is None:
        return MetricBreakdown(
            total=0.0,
            breakdown={
                "schema": 0.0,
                "coverage": 0.0,
                "count": 0.0,
                "grounded": 0.0,
                "calibration": 0.0,
            },
            feedback=f"案例: {example.name}\n{schema_note}",
        )

    # Coverage — at least 3/5 categories
    n_cats = len(result.categories_covered())
    coverage_score = min(1.0, n_cats / 3.0)
    coverage_note = (
        f"覆盖 {n_cats}/5 类题型（target ≥ 3）"
        + (" ✓" if n_cats >= 3 else " — 需要更广覆盖")
    )

    # Count in [5, 8]
    n_q = len(result.expected_questions)
    if 5 <= n_q <= 8:
        count_score = 1.0
        count_note = f"questions 数 {n_q} ✓"
    elif n_q < 5:
        count_score = max(0.0, n_q / 5.0)
        count_note = f"questions 太少 ({n_q}) — 备战覆盖不足"
    else:
        count_score = max(0.0, 1.0 - 0.2 * (n_q - 8))
        count_note = f"questions 太多 ({n_q}) — 用户记不住所有"

    # Grounded rationales
    keyword_pool = set(example.profile_keywords) | set(example.jd_keywords)
    if keyword_pool and result.expected_questions:
        grounded = sum(
            1
            for q in result.expected_questions
            if any(k in q.rationale for k in keyword_pool)
        )
        grounded_score = grounded / len(result.expected_questions)
        grounded_note = (
            f"{grounded}/{len(result.expected_questions)} rationale 引用了"
            f" profile/JD 关键词"
        )
    else:
        grounded_score = 1.0  # nothing to check
        grounded_note = "(no anchor keywords specified — skipping grounded check)"

    # Calibration: penalize likelihood uniformity
    likelihoods = [q.likelihood for q in result.expected_questions]
    if len(likelihoods) >= 3:
        std = statistics.pstdev(likelihoods)
        # Healthy spread: std >= 0.10. Crash to 0 at std == 0.
        calibration_score = min(1.0, std / 0.10)
        if std < 0.05:
            calibration_note = (
                f"likelihood 全部聚簇 (std={std:.3f}) — 模型在偷懒，全填默认值"
            )
        else:
            calibration_note = f"likelihood 分散度合理 (std={std:.3f}) ✓"
    else:
        calibration_score = 1.0
        calibration_note = "(question 太少，跳过 calibration 检查)"

    total = (
        _W_SCHEMA * schema_score
        + _W_COVERAGE * coverage_score
        + _W_COUNT * count_score
        + _W_GROUNDED * grounded_score
        + _W_CALIBRATION * calibration_score
    )

    feedback = "\n".join(
        [
            f"案例: {example.name} ({example.band})",
            schema_note,
            coverage_note,
            count_note,
            grounded_note,
            calibration_note,
            f"score: schema={schema_score:.2f} coverage={coverage_score:.2f}"
            f" count={count_score:.2f} grounded={grounded_score:.2f}"
            f" calib={calibration_score:.2f} → total={total:.2f}",
        ]
    )

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "coverage": coverage_score,
            "count": count_score,
            "grounded": grounded_score,
            "calibration": calibration_score,
        },
        feedback=feedback,
    )


def _zero(example: PrepareInterviewExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={
            "schema": 0.0,
            "coverage": 0.0,
            "count": 0.0,
            "grounded": 0.0,
            "calibration": 0.0,
        },
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


# Suppress unused-import warning for `field` (kept for future extension)
_ = field


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[PrepareInterviewExample], list[PrepareInterviewExample]]:
    """Stratified by ``band`` (with_面经 / no_面经 / edge_case)."""
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[PrepareInterviewExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[PrepareInterviewExample] = []
    val: list[PrepareInterviewExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
