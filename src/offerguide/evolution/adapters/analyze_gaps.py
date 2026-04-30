"""``analyze_gaps`` SKILL adapter.

Without 4 weeks of dogfood data on which suggestions actually got
accepted by recruiters, we can't evolve directly toward "reply rate
delta after applying suggestion N."  Instead this metric optimizes for
**signs of non-laziness** that we know correlate with usefulness:

- ``schema``        — output validates against ``AnalyzeGapsResult``
                       Pydantic schema (extra='forbid'). Forces the model
                       to fill every required field.
- ``keyword_recall`` — fraction of ``expected_keywords`` (the JD's
                       hard-requirement skills) that appear in
                       ``keyword_gaps[].jd_keyword``. Penalizes models
                       that miss obvious requirements.
- ``ai_risk_floor``  — at least one ``ai_risk='high'`` suggestion when
                       ``expected_high_risk_floor > 0``. Penalizes the
                       failure mode where every suggestion is rated
                       low-risk regardless of content (which would defeat
                       the 49% AI-detection guard).
- ``count_in_range`` — ``len(suggestions)`` is in [3, 8]. Too few = useless,
                       too many = user won't act on any.

Weights chosen by inspection. When real dogfood data lands, replace
``ai_risk_floor`` with the true acceptance rate and add a
``reply_after_accept`` axis.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from ...skills.analyze_gaps.helpers import AnalyzeGapsResult
from ._base import MetricBreakdown, parse_json_output

name: str = "analyze_gaps"
INPUT_NAMES: list[str] = ["job_text", "user_profile"]
METRIC_AXES: list[str] = ["total", "schema", "keyword_recall", "ai_risk", "count"]

_W_SCHEMA = 0.30
_W_RECALL = 0.40
_W_AI_RISK = 0.15
_W_COUNT = 0.15


@dataclass(frozen=True)
class AnalyzeGapsExample:
    """One ground-truth case for analyze_gaps."""

    name: str
    job_text: str
    user_profile: str
    expected_keywords: tuple[str, ...]
    """Keywords from the JD that the SKILL should detect as gaps (or at
    least flag in keyword_gaps). Substring match against ``jd_keyword``."""

    expected_high_risk_floor: int = 0
    """Minimum number of ``ai_risk='high'`` suggestions expected. Set
    to 1 when the JD lists a buzzword-heavy ATS bait that any AI-tilted
    suggestion would trip."""

    expected_total_min: int = 3
    expected_total_max: int = 8

    band: str = "real"
    """'real' | 'edge_case' for split balancing."""

    notes: str = ""


# Profile from score_match — same user.
_USER = """胡阳，上海财经大学 应用统计专硕（2025-2027 在读）。
技能：Python、PyTorch、SFT/LoRA、HuggingFace Transformers/TRL、强化学习（PPO/GRPO）、AI Agent 架构。
项目: Deep Research Agent（法至科技实习），RemeDi（LLaDA-8B 扩散语言模型）。"""


EXAMPLES: tuple[AnalyzeGapsExample, ...] = (
    AnalyzeGapsExample(
        name="ali_agent_intern_gaps",
        band="real",
        job_text=(
            "AI Agent 实习生 — 阿里 — 北京\n"
            "任职要求：\n"
            "1. 计算机/AI/数学相关专业本科及以上\n"
            "2. **精通 Python，扎实的 C/C++/Java（至少一门）**\n"
            "3. 深入理解 Transformer 与 LLM；熟悉 PyTorch；掌握 RL/NLP/CV 基础\n"
            "4. 有 Agent 项目经验者优先；熟悉 LangChain / LangGraph 加分"
        ),
        user_profile=_USER,
        expected_keywords=("C/C++", "C++", "Java", "LangChain", "LangGraph"),
        expected_high_risk_floor=0,
        expected_total_min=3,
        expected_total_max=8,
        notes="C/C++ 和 LangGraph 是真实 gap，必须命中。",
    ),
    AnalyzeGapsExample(
        name="bytedance_post_training_gaps",
        band="real",
        job_text=(
            "大模型后训练算法实习生 — 字节 Seed\n"
            "要求：\n"
            "- 熟练 PyTorch，理解 Transformer 内部机制\n"
            "- 有 SFT / DPO / GRPO / RLHF 任一方向项目经验\n"
            "- 熟悉 DeepSpeed / Megatron 等训练框架\n"
            "- 有 Megatron 大规模分布式训练经验加分"
        ),
        user_profile=_USER,
        expected_keywords=("Megatron", "DPO", "RLHF"),
        notes="用户有 SFT/GRPO/DeepSpeed，但 Megatron 和 DPO 没沾过。",
    ),
    AnalyzeGapsExample(
        name="data_analyst_gaps",
        band="real",
        job_text=(
            "数据分析实习生\n"
            "要求：\n"
            "- 统计/数学/计算机相关\n"
            "- **熟悉 SQL，会用 Python (pandas)**\n"
            "- 有过 A/B 实验设计经验\n"
            "- 熟悉 Tableau / FineBI 等 BI 工具加分"
        ),
        user_profile=_USER,
        expected_keywords=("SQL", "A/B", "Tableau"),
        notes="A/B 和 BI 工具是 gap；SQL 用户简历没明确提。",
    ),
    AnalyzeGapsExample(
        name="quant_research_gaps",
        band="real",
        job_text=(
            "量化研究实习生 — 某私募\n"
            "要求：\n"
            "- 熟练 Python，了解时序分析\n"
            "- 有 Alpha 因子挖掘 / 回测项目经验加分\n"
            "- 熟悉常用机器学习算法\n"
            "- C++ 或 Rust 优先"
        ),
        user_profile=_USER,
        expected_keywords=("Alpha", "回测", "C++", "Rust"),
        notes="所有金融项目都是 gap；C++/Rust 也没有。",
    ),
    AnalyzeGapsExample(
        name="agent_app_gaps",
        band="real",
        job_text=(
            "AI Agent 应用开发实习生\n"
            "要求：\n"
            "- 熟悉 Python，至少做过 1 个 Agent 应用 demo\n"
            "- 了解 RAG / Memory / Tool Use 概念\n"
            "- **熟悉 LangGraph / DSPy 加分**\n"
            "- 有过工业级 RAG 项目优先"
        ),
        user_profile=_USER,
        expected_keywords=("LangGraph", "DSPy", "RAG"),
        notes="用户有 Agent 经验但 LangGraph/DSPy 没用过；RAG 也没明显项目。",
    ),
    AnalyzeGapsExample(
        name="frontend_gaps_misfit",
        band="edge_case",
        job_text=(
            "前端实习生（React + TypeScript）\n"
            "要求：\n"
            "- 熟练 React / TypeScript / Webpack / Vite\n"
            "- 关注 Web 性能与可访问性\n"
            "- CSS / SCSS / Tailwind 任一精通"
        ),
        user_profile=_USER,
        expected_keywords=("React", "TypeScript", "CSS"),
        expected_high_risk_floor=0,
        expected_total_min=2,  # 可以更少 — 大多 keyword 都缺，给少量诚实建议比硬塞 8 条好
        expected_total_max=6,
        notes="技术栈完全不沾。健康行为是承认匹配度低 + 少量诚实建议（甚至建议不投）。",
    ),
    AnalyzeGapsExample(
        name="ats_buzzword_gaps",
        band="edge_case",
        job_text=(
            "AI 应用工程师\n"
            "要求：\n"
            "- 赋能业务，打造 AI Agent 闭环\n"
            "- 端到端落地大模型应用，提升产品智能化水平\n"
            "- 数据驱动，构建可持续增长的智能化解决方案\n"
            "- Python / Java / Go 任一精通"
        ),
        user_profile=_USER,
        expected_keywords=("Python", "Java", "Go", "Agent"),
        expected_high_risk_floor=1,
        notes="JD 全是营销空话，模型若无脑加营销词会被 49% AI 检测掉，必须有 high-risk 警告。",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


def metric(
    example: AnalyzeGapsExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    """Score one (example, output) pair on 4 axes."""
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: model did not emit valid JSON.")

    # Schema validation
    schema_score = 0.0
    schema_note = ""
    result: AnalyzeGapsResult | None = None
    try:
        result = AnalyzeGapsResult.model_validate(parsed)
        schema_score = 1.0
        schema_note = "schema 通过 ✓"
    except ValidationError as e:
        # Get first error short-form
        err = e.errors()[0]
        loc = ".".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "validation failed")
        schema_note = f"schema 失败: {loc}: {msg}"

    if result is None:
        return MetricBreakdown(
            total=0.0,
            breakdown={"schema": 0.0, "keyword_recall": 0.0, "ai_risk": 0.0, "count": 0.0},
            feedback=f"案例: {example.name}\n{schema_note}",
        )

    # Keyword recall
    flagged = [g.jd_keyword for g in result.keyword_gaps]
    flagged_blob = " ".join(flagged)
    hits = [k for k in example.expected_keywords if k in flagged_blob]
    misses = [k for k in example.expected_keywords if k not in flagged_blob]
    recall_score = len(hits) / max(1, len(example.expected_keywords))
    recall_note = (
        f"keyword_gaps 命中 {hits}/{list(example.expected_keywords)} ✓"
        if not misses
        else f"keyword_gaps 漏掉: {misses}（应在 jd_keyword 里出现）"
    )

    # AI risk floor
    n_high = result.high_risk_count()
    if n_high >= example.expected_high_risk_floor:
        ai_risk_score = 1.0
        ai_risk_note = (
            f"ai_risk=high 有 {n_high} 个，达到下限 ≥ {example.expected_high_risk_floor} ✓"
        )
    else:
        ai_risk_score = 0.0
        ai_risk_note = (
            f"ai_risk=high 只有 {n_high} 个，需要至少 {example.expected_high_risk_floor}"
            f"（JD 像 ATS 营销词陷阱时必须警告）"
        )

    # Count in range
    n_sug = len(result.suggestions)
    if example.expected_total_min <= n_sug <= example.expected_total_max:
        count_score = 1.0
        count_note = f"suggestions 数 {n_sug} 在期望区间 ✓"
    else:
        # Linear fall-off
        if n_sug < example.expected_total_min:
            dist = example.expected_total_min - n_sug
            count_score = max(0.0, 1.0 - 0.3 * dist)
            count_note = f"suggestions 太少 ({n_sug} < {example.expected_total_min})"
        else:
            dist = n_sug - example.expected_total_max
            count_score = max(0.0, 1.0 - 0.2 * dist)
            count_note = (
                f"suggestions 太多 ({n_sug} > {example.expected_total_max}) — 用户不会全做"
            )

    total = (
        _W_SCHEMA * schema_score
        + _W_RECALL * recall_score
        + _W_AI_RISK * ai_risk_score
        + _W_COUNT * count_score
    )

    feedback = "\n".join(
        [
            f"案例: {example.name} ({example.band})",
            schema_note,
            recall_note,
            ai_risk_note,
            count_note,
            f"score: schema={schema_score:.2f} recall={recall_score:.2f}"
            f" ai_risk={ai_risk_score:.2f} count={count_score:.2f} → total={total:.2f}",
        ]
    )

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "keyword_recall": recall_score,
            "ai_risk": ai_risk_score,
            "count": count_score,
        },
        feedback=feedback,
    )


def _zero(example: AnalyzeGapsExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={"schema": 0.0, "keyword_recall": 0.0, "ai_risk": 0.0, "count": 0.0},
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


# ── train/val split ────────────────────────────────────────────────


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[AnalyzeGapsExample], list[AnalyzeGapsExample]]:
    """Stratified by ``band`` (real / edge_case)."""
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[AnalyzeGapsExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[AnalyzeGapsExample] = []
    val: list[AnalyzeGapsExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
