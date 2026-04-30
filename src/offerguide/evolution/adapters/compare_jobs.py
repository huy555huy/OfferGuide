"""``compare_jobs`` SKILL adapter.

Without dogfood data on which curated picks actually got HR replies,
the metric optimizes for **decision-quality structure** that we know
correlates with usefulness:

- ``schema``               — output validates against ``CompareJobsResult``
                              (rank ∈ [1, N], action one of 4 values,
                              all probabilities ∈ [0,1])
- ``rank_validity``        — rank is a permutation of 1..N AND every input
                              job_id appears exactly once in rankings
- ``limit_consistency``    — recommended_apply_count ≤ application_limit AND
                              count(action=='apply_first') ≤ application_limit
- ``action_coherence``     — action distribution makes sense:
                              ranks 1..limit map to apply_first,
                              skipped jobs have low match OR hard
                              risk_factors
- ``distinguishing_quality`` — distinguishing_factors are NOT identical
                              across jobs (they should differentiate)
- ``strategic_specificity`` — strategic_summary mentions specific job_ids
                              or titles (not generic boilerplate)

When dogfood data lands, replace ``distinguishing_quality`` with
``actual_reply_rate_after_apply_first`` to optimize directly for
"recommended apply_first jobs got HR replies".
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from ...skills.compare_jobs.helpers import (
    COMPANY_APPLICATION_LIMITS,
    CompareJobsResult,
)
from ._base import MetricBreakdown, parse_json_output

name: str = "compare_jobs"
INPUT_NAMES: list[str] = ["company", "user_profile", "jobs_json"]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "rank_validity",
    "limit_consistency",
    "action_coherence",
    "distinguishing_quality",
    "strategic_specificity",
]

_W_SCHEMA = 0.20
_W_RANK = 0.20
_W_LIMIT = 0.15
_W_ACTION = 0.20
_W_DISTINGUISH = 0.15
_W_STRATEGY = 0.10


@dataclass(frozen=True)
class CompareJobsExample:
    """One ground-truth case for compare_jobs."""

    name: str
    company: str
    user_profile: str
    job_ids: tuple[int, ...]
    """Synthetic job_ids the SKILL must reproduce in rankings."""

    jobs_json: str
    """JSON string passed as the SKILL input."""

    expected_application_limit: int
    """Expected application_limit_estimate the SKILL should output
    (within ±1 tolerance based on COMPANY_APPLICATION_LIMITS)."""

    expected_apply_first_count: int
    """Expected count of action='apply_first' (== application_limit
    typically; can be lower if all options are weak)."""

    band: str = "real"
    notes: str = ""


_USER = """胡阳，上海财经大学应用统计专硕。
项目: Deep Research Agent (法至科技实习, agent runtime + 双层架构);
RemeDi (LLaDA-8B 扩散模型, SFT + GRPO + DeepSpeed ZeRO-2)。
技能: Python, PyTorch, Transformers, RL (PPO/GRPO), AI Agent 架构。"""


# Pre-built jobs_json strings for each example


def _make_jobs_json(jobs: list[dict]) -> str:
    return _json.dumps(jobs, ensure_ascii=False)


_BYTEDANCE_5_JOBS = _make_jobs_json([
    {"job_id": 101, "title": "AI Agent 实习生 - Doubao",
     "raw_text": "Python + Transformer + Agent. 业务: Doubao 应用. 北京/上海.",
     "source": "nowcoder"},
    {"job_id": 102, "title": "大模型后训练算法 - Seed",
     "raw_text": "PyTorch + SFT/GRPO/RLHF + DeepSpeed/Megatron. Seed 实验室.",
     "source": "nowcoder"},
    {"job_id": 103, "title": "前端工程师 - 抖音电商",
     "raw_text": "React + TypeScript. 抖音电商业务.",
     "source": "boss_extension"},
    {"job_id": 104, "title": "Agent 评测工程师 - Doubao",
     "raw_text": "Agent 评测平台. Python + 自动化测试.",
     "source": "nowcoder"},
    {"job_id": 105, "title": "推荐算法 - 抖音",
     "raw_text": "推荐算法 + 召回排序. Python + Spark.",
     "source": "manual"},
])


_ALI_3_JOBS = _make_jobs_json([
    {"job_id": 201, "title": "AI Agent 实习 - 达摩院",
     "raw_text": "Agent runtime + LangGraph 加分. C/C++/Java 任一. PyTorch.",
     "source": "nowcoder"},
    {"job_id": 202, "title": "搜索算法实习 - 淘宝",
     "raw_text": "搜索召回 + 大模型应用. Python.",
     "source": "boss_extension"},
    {"job_id": 203, "title": "数据分析实习 - 1688",
     "raw_text": "SQL + Pandas + A/B 实验设计.",
     "source": "manual"},
])


_QUANT_2_JOBS = _make_jobs_json([
    {"job_id": 301, "title": "量化研究实习 - A 私募",
     "raw_text": "因子挖掘 + 回测. Python + C++.",
     "source": "manual"},
    {"job_id": 302, "title": "量化开发 - B 私募",
     "raw_text": "高频交易系统. C++ + Linux 必备.",
     "source": "manual"},
])


EXAMPLES: tuple[CompareJobsExample, ...] = (
    CompareJobsExample(
        name="bytedance_5_options",
        band="real",
        company="字节跳动",
        user_profile=_USER,
        job_ids=(101, 102, 103, 104, 105),
        jobs_json=_BYTEDANCE_5_JOBS,
        expected_application_limit=2,  # 校招硬限
        expected_apply_first_count=2,  # 用满
        notes="字节校招经典场景：5 选 2。Seed 后训练 (102) + Agent (101 or 104) 是答案",
    ),
    CompareJobsExample(
        name="ali_3_jobs",
        band="real",
        company="阿里巴巴",
        user_profile=_USER,
        job_ids=(201, 202, 203),
        jobs_json=_ALI_3_JOBS,
        expected_application_limit=3,
        expected_apply_first_count=2,  # 数据分析弱不应推 first
        notes="阿里 3 个意向 ceiling。Agent 必投，搜索次选，数据分析弱不应 apply_first",
    ),
    CompareJobsExample(
        name="all_misfit_quant",
        band="edge_case",
        company="某私募",
        user_profile=_USER,
        job_ids=(301, 302),
        jobs_json=_QUANT_2_JOBS,
        expected_application_limit=3,  # 默认值
        expected_apply_first_count=0,  # 都不应推 first，全 skip
        notes="完全不沾的方向，健康行为是承认 + recommended_apply_count=0",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


def metric(
    example: CompareJobsExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    """Score one (example, output) pair on 6 axes."""
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: model did not emit valid JSON.")

    # 1. Schema
    schema_score = 0.0
    schema_note = ""
    result: CompareJobsResult | None = None
    try:
        result = CompareJobsResult.model_validate(parsed)
        schema_score = 1.0
        schema_note = "schema 通过 ✓"
    except ValidationError as e:
        err = e.errors()[0]
        loc = ".".join(str(x) for x in err.get("loc", []))
        schema_note = f"schema 失败: {loc}: {err.get('msg', '...')}"

    if result is None:
        return MetricBreakdown(
            total=0.0,
            breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
            feedback=f"案例: {example.name}\n{schema_note}",
        )

    # 2. rank_validity: every input job_id appears once, ranks 1..N permutation
    expected_ids = set(example.job_ids)
    actual_ids = result.all_ids()
    ranks = [r.rank for r in result.rankings]
    n = len(example.job_ids)
    rank_score = 0.0
    if expected_ids != actual_ids:
        rank_note = (
            f"rank_validity 失败: input ids {sorted(expected_ids)}, "
            f"output ids {sorted(actual_ids)}"
        )
    elif sorted(ranks) != list(range(1, n + 1)):
        rank_note = f"rank 不是 1..{n} 的排列: {sorted(ranks)}"
    else:
        rank_score = 1.0
        rank_note = f"rank 是 1..{n} 排列，所有 job_id 出现 ✓"

    # 3. limit_consistency
    apply_first_count = sum(
        1 for r in result.rankings if r.action == "apply_first"
    )
    limit = result.application_limit_estimate
    expected_limit = example.expected_application_limit
    limit_score = 1.0
    limit_notes: list[str] = []
    if abs(limit - expected_limit) > 1:
        limit_score *= 0.5
        limit_notes.append(
            f"application_limit_estimate {limit} 偏离期望 {expected_limit}（容忍 ±1）"
        )
    if result.recommended_apply_count > limit:
        limit_score *= 0.5
        limit_notes.append(
            f"recommended_apply_count {result.recommended_apply_count} > limit {limit}"
        )
    if apply_first_count > limit:
        limit_score *= 0.5
        limit_notes.append(
            f"apply_first 数 {apply_first_count} > limit {limit}"
        )
    limit_note = "; ".join(limit_notes) if limit_notes else "limit 一致 ✓"

    # 4. action_coherence: ranks 1..limit should map to apply_first;
    #    "skip" should have low match_probability OR hard risk_factors
    action_score = 1.0
    action_issues: list[str] = []
    sorted_rankings = sorted(result.rankings, key=lambda r: r.rank)
    for r in sorted_rankings[:limit]:
        if r.action != "apply_first" and r.match_probability >= 0.45:
            action_score *= 0.7
            action_issues.append(
                f"rank {r.rank} (match {r.match_probability:.2f}) 不是 apply_first"
            )
    for r in result.rankings:
        if r.action == "skip" and r.match_probability >= 0.5 and not r.risk_factors:
            action_score *= 0.6
            action_issues.append(
                f"rank {r.rank} skipped 但 match {r.match_probability:.2f} 高且无 risk"
            )
    action_note = "action 与 rank 一致 ✓" if action_score == 1.0 else "; ".join(action_issues)

    # Special check for the all-misfit case
    if example.expected_apply_first_count == 0 and apply_first_count > 0:
        action_score = 0.0
        action_note = (
            f"全 misfit 案例不应有 apply_first，但有 {apply_first_count} 个"
        )

    # 5. distinguishing_quality: factors should differ across jobs
    all_factors = [
        f for r in result.rankings for f in r.distinguishing_factors
    ]
    if not all_factors:
        distinguishing_score = 0.5
        distinguishing_note = "所有 distinguishing_factors 为空"
    else:
        unique_ratio = len(set(all_factors)) / len(all_factors)
        distinguishing_score = unique_ratio
        distinguishing_note = (
            f"distinguishing_factors 唯一率 {unique_ratio:.2f}"
            f" ({len(set(all_factors))}/{len(all_factors)})"
        )

    # 6. strategic_specificity: summary mentions specific titles or job_ids
    summary = result.strategic_summary
    has_id = any(str(jid) in summary for jid in example.job_ids)
    has_title = bool(re.search(r"[#A-Z][A-Za-z0-9]+|实习|算法|工程师", summary))
    if has_id or len(summary) > 60:
        strategy_score = 1.0
        strategy_note = "strategic_summary 具体 ✓"
    elif has_title:
        strategy_score = 0.6
        strategy_note = "strategic_summary 提到岗位类型但没引 id（部分得分）"
    else:
        strategy_score = 0.0
        strategy_note = "strategic_summary 太空泛或太短"

    total = (
        _W_SCHEMA * schema_score
        + _W_RANK * rank_score
        + _W_LIMIT * limit_score
        + _W_ACTION * action_score
        + _W_DISTINGUISH * distinguishing_score
        + _W_STRATEGY * strategy_score
    )

    feedback = "\n".join(
        [
            f"案例: {example.name} ({example.band})",
            schema_note,
            rank_note,
            limit_note,
            action_note,
            distinguishing_note,
            strategy_note,
            f"score: schema={schema_score:.2f} rank={rank_score:.2f}"
            f" limit={limit_score:.2f} action={action_score:.2f}"
            f" distinguish={distinguishing_score:.2f} strategy={strategy_score:.2f}"
            f" → total={total:.2f}",
        ]
    )

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema":                schema_score,
            "rank_validity":         rank_score,
            "limit_consistency":     limit_score,
            "action_coherence":      action_score,
            "distinguishing_quality": distinguishing_score,
            "strategic_specificity": strategy_score,
        },
        feedback=feedback,
    )


def _zero(example: CompareJobsExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


# Suppress unused-import warning for `field` (kept for future extension)
_ = field
_ = COMPANY_APPLICATION_LIMITS  # also used by the SKILL helpers


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[CompareJobsExample], list[CompareJobsExample]]:
    """Stratified by ``band`` (real / edge_case)."""
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[CompareJobsExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[CompareJobsExample] = []
    val: list[CompareJobsExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
