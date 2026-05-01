"""``profile_resume_gap`` SKILL adapter — 4-bucket gap analysis.

Metric optimizes for:

- ``schema``                       — validates against ProfileResumeGapResult
- ``has_all_4_buckets_or_explains`` — output considers all 4 buckets,
                                      empty buckets allowed but not unfilled
- ``concrete_short_term_actions``  — short_term_fix items have time budgets
                                      and resume signals (not vague advice)
- ``unfakeable_honesty``           — cannot_fake items name verification
                                      channels (not just "学历背景")
- ``verdict_consistency``          — verdict matches the bucket distribution
                                      (lots of cannot_fake → skip; mostly
                                      have → go)
- ``actionable_top3``              — top_3_pre_apply_actions are concrete
                                      verbs + deliverables, not motivational

Same as successful_profile, the metric is structural pre-dogfood.
Once we have user-completed-action data we can replace ``concrete_short_term_actions``
with ``post_action_resume_match`` (did the user actually update their resume
to match ``skill_signal_after`` after doing the action).
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from ...skills.profile_resume_gap.helpers import ProfileResumeGapResult
from ._base import MetricBreakdown, parse_json_output

name: str = "profile_resume_gap"
INPUT_NAMES: list[str] = ["successful_profile_json", "user_resume"]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "has_all_4_buckets_or_explains",
    "concrete_short_term_actions",
    "unfakeable_honesty",
    "verdict_consistency",
    "actionable_top3",
]

_W_SCHEMA = 0.20
_W_BUCKETS = 0.10
_W_ACTIONS = 0.20
_W_HONESTY = 0.20
_W_VERDICT = 0.15
_W_TOP3 = 0.15


@dataclass(frozen=True)
class ProfileGapExample:
    name: str
    successful_profile_json: str
    user_resume: str
    band: str = "real"
    notes: str = ""


_RICH_PROFILE = _json.dumps({
    "company": "字节跳动",
    "role_focus": "AI Agent 后端实习",
    "evidence_count": 3,
    "evidence_kinds": ["offer_post", "interview", "project_share"],
    "background_pattern": {
        "education_level": "硕士占多数",
        "school_tier": "985 头部",
        "majors": ["计算机", "数据科学"],
        "internships": ["美团 / 头部 AI 实验室一段实习"],
        "competitions": [],
        "publications": [],
    },
    "skill_pattern": {
        "must_have": ["Python", "PyTorch", "LangGraph", "RAG"],
        "highly_valued": ["DSPy", "GRPO/RLHF"],
        "differentiators": ["开源项目"],
    },
    "project_pattern": {
        "typical_project_themes": ["LLM agent", "RAG", "评测框架"],
        "common_tech_stacks": ["LangGraph", "BGE", "Pydantic"],
        "scale_signals": ["200+ 测试用例"],
        "outcome_signals": ["GAIA benchmark 0.52"],
    },
    "interview_pattern": {
        "common_questions": [
            {"question": "LangGraph state machine 设计",
             "category": "technical", "evidence_count": 2},
            {"question": "RAG retrieval miss debug",
             "category": "technical", "evidence_count": 2},
        ],
        "behavioral_themes": [],
        "decision_factors": [
            "项目深度", "系统设计的工程取舍",
        ],
    },
    "why_they_passed": [
        "项目深度被多次表扬 (来自 2 条 offer_post)",
        "对 RAG / agent 工程取舍熟悉 (来自 2 条 interview + 1 条 project_share)",
        "开源 agent 评测项目作为差异化 (来自 1 条 project_share)",
    ],
    "evidence_sources": [],
    "uncertainty_notes": ["背景模式只 3 条样本，置信度中等"],
}, ensure_ascii=False)


_USER_RESUME_GOOD_FIT = """
胡阳 上海财经大学 应用统计学 硕士 (2026 年毕业)

实习经历:
- 法至科技 (2025/3 至今, NLP 工程师): 用 LangChain + LangGraph 搭建多 agent
  评测系统, 引入 GAIA benchmark 评估; 在 RAG retrieval 失败的 OOD 用例上做了
  query rewriting 改进，hit@5 从 0.61 提升到 0.74

项目:
- RemeDi: 基于 BERT 的医疗文本双流推荐, AUC 提升 0.04
- Deep Research Agent: 基于 LangGraph + DSPy 的研究助手 (开源)

技能: Python, PyTorch, LangGraph, DSPy, BGE, Pydantic
"""


_USER_RESUME_WEAK_FIT = """
张三 双非二本 计算机科学 本科 (2026 年毕业)

实习: 无

项目:
- 学校课程作业: 用 Tensorflow 训了个简单的 CNN 做猫狗分类

技能: Python (基础), Linux 命令
"""


EXAMPLES: tuple[ProfileGapExample, ...] = (
    ProfileGapExample(
        name="strong_fit",
        band="real",
        successful_profile_json=_RICH_PROFILE,
        user_resume=_USER_RESUME_GOOD_FIT,
        notes="高匹配度: verdict 应为 go, have 桶应充实",
    ),
    ProfileGapExample(
        name="weak_fit_unfakeable_blockers",
        band="real",
        successful_profile_json=_RICH_PROFILE,
        user_resume=_USER_RESUME_WEAK_FIT,
        notes="低匹配 + 学校学历差距是 cannot_fake 重点; verdict 应为 hold/skip",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


_VAGUE_ACTION_PATTERNS = (
    "加强", "提升", "好好", "扎实", "多练", "认真",
)
_TIME_PATTERN = re.compile(r"(\d+\s*(小时|h|day|天|周|week|month|月))", re.IGNORECASE)
_VERIFICATION_KEYWORDS = (
    "学信网", "背调", "前 leader", "前leader", "证书", "官方榜单",
    "公开 page", "公开page", "github", "open source",
)
_MOTIVATIONAL_PATTERNS = (
    "加油", "相信自己", "调整心态", "保持自信", "积极",
)


def metric(
    example: ProfileGapExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: invalid JSON")

    result: ProfileResumeGapResult | None = None
    schema_score = 0.0
    schema_note = ""
    try:
        result = ProfileResumeGapResult.model_validate(parsed)
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

    # has_all_4_buckets_or_explains: empty buckets are OK but the model
    # should have populated them (have/short_term_fix in particular for
    # any non-trivial profile)
    populated = sum(1 for b in [
        result.have, result.short_term_fix,
        result.cannot_short_term, result.cannot_fake,
    ] if b)
    bucket_score = populated / 4
    bucket_note = f"4 桶填充 {populated}/4"

    # Concrete short_term_actions
    if result.short_term_fix:
        ok = 0
        for item in result.short_term_fix:
            text = item.concrete_action
            has_verb = not any(v in text for v in _VAGUE_ACTION_PATTERNS)
            has_time = bool(_TIME_PATTERN.search(text)) or item.estimated_hours > 0
            has_signal = bool(item.skill_signal_after.strip())
            if has_verb and has_time and has_signal:
                ok += 1
        action_score = ok / len(result.short_term_fix)
        action_note = (
            f"short_term_fix 具体可行 {ok}/{len(result.short_term_fix)}"
        )
    else:
        action_score = 1.0  # empty bucket is acceptable for a "go" case
        action_note = "short_term_fix 为空 (skip)"

    # Unfakeable honesty: cannot_fake items must name verification channels
    if result.cannot_fake:
        ok = sum(
            1 for c in result.cannot_fake
            if any(k in c.why_unfakeable for k in _VERIFICATION_KEYWORDS)
        )
        honesty_score = ok / len(result.cannot_fake)
        honesty_note = (
            f"cannot_fake 引用验证渠道 {ok}/{len(result.cannot_fake)}"
        )
    else:
        honesty_score = 1.0
        honesty_note = "cannot_fake 为空 (skip)"

    # Verdict consistency
    verdict = result.apply_advice.verdict
    have_n = len(result.have)
    cannot_n = len(result.cannot_fake)
    short_n = len(result.short_term_fix)
    cannot_short_n = len(result.cannot_short_term)
    expected_verdict: str
    if cannot_n >= 2 or cannot_short_n >= 3:
        expected_verdict = "skip" if cannot_n >= 2 else "hold"
    elif have_n >= 4 and short_n <= 2:
        expected_verdict = "go"
    elif have_n >= 2:
        expected_verdict = "maybe"
    else:
        expected_verdict = "hold"

    if verdict == expected_verdict:
        verdict_score = 1.0
    elif _ordered_distance(verdict, expected_verdict) == 1:
        verdict_score = 0.6  # one-step off
    else:
        verdict_score = 0.0
    verdict_note = (
        f"verdict={verdict} expected≈{expected_verdict} → {verdict_score:.2f}"
    )

    # Actionable top_3
    top3 = result.apply_advice.top_3_pre_apply_actions
    if top3:
        ok = sum(
            1 for a in top3
            if not any(m in a for m in _MOTIVATIONAL_PATTERNS)
            and len(a) >= 8
        )
        top3_score = ok / len(top3)
        top3_note = f"top_3 actionable {ok}/{len(top3)}"
    else:
        top3_score = 0.0
        top3_note = "top_3_pre_apply_actions 空"

    total = (
        _W_SCHEMA * schema_score
        + _W_BUCKETS * bucket_score
        + _W_ACTIONS * action_score
        + _W_HONESTY * honesty_score
        + _W_VERDICT * verdict_score
        + _W_TOP3 * top3_score
    )

    feedback = "\n".join([
        f"案例: {example.name} ({example.band})",
        schema_note, bucket_note, action_note,
        honesty_note, verdict_note, top3_note,
        f"score: schema={schema_score:.2f} buckets={bucket_score:.2f}"
        f" actions={action_score:.2f} honesty={honesty_score:.2f}"
        f" verdict={verdict_score:.2f} top3={top3_score:.2f}"
        f" → total={total:.2f}",
    ])

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "has_all_4_buckets_or_explains": bucket_score,
            "concrete_short_term_actions": action_score,
            "unfakeable_honesty": honesty_score,
            "verdict_consistency": verdict_score,
            "actionable_top3": top3_score,
        },
        feedback=feedback,
    )


def _ordered_distance(a: str, b: str) -> int:
    """Distance between two verdicts on the go→maybe→hold→skip ordinal."""
    order = ["go", "maybe", "hold", "skip"]
    try:
        return abs(order.index(a) - order.index(b))
    except ValueError:
        return 99


def _zero(example: ProfileGapExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[ProfileGapExample], list[ProfileGapExample]]:
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[ProfileGapExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[ProfileGapExample] = []
    val: list[ProfileGapExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
