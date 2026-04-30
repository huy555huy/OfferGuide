"""``post_interview_reflection`` SKILL adapter.

Borrowed pattern from `Pytai <https://github.com/getFrontend/app-ai-interviews>`_
(MIT) — post-interview transcript analysis as the dogfood loop closure.

Metric optimizes for:
- ``schema``                 — output validates against PostInterviewReflection
- ``hit_rate_in_band``       — hit_rate matches what the transcript signals
                                (we don't know ground truth pre-dogfood, but
                                we can sanity-check internally consistent
                                output)
- ``calibration_low_error``  — mean(|predicted_likelihood - actual_hit|) ≤ 0.3
- ``surprises_explained``    — every surprise has a non-trivial why_we_missed
                                (longer than 10 chars, not "意外问题")
- ``stories_grounded``       — suggested_stories' triggered_by references
                                actual_transcript content (substring check)
- ``brief_delta_specific``   — interview_style_addition (if non-null) is
                                more than 12 chars and contains at least one
                                specific noun-y phrase

When real dogfood lands, replace ``hit_rate_in_band`` with
``hit_rate_matches_user_judgement`` (compare with user's own scoring).
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from ...skills.post_interview_reflection.helpers import PostInterviewReflection
from ._base import MetricBreakdown, parse_json_output

name: str = "post_interview_reflection"
INPUT_NAMES: list[str] = ["company", "prep_questions_json", "actual_transcript"]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "hit_rate_in_band",
    "calibration_low_error",
    "surprises_explained",
    "stories_grounded",
    "brief_delta_specific",
]

_W_SCHEMA = 0.20
_W_HIT_BAND = 0.15
_W_CALIB = 0.20
_W_SURPRISES = 0.15
_W_STORIES = 0.15
_W_BRIEF = 0.15


@dataclass(frozen=True)
class PostInterviewExample:
    """One ground-truth case for post_interview_reflection."""

    name: str
    company: str
    prep_questions_json: str
    actual_transcript: str
    expected_hit_rate_min: float
    expected_hit_rate_max: float
    band: str = "real"
    notes: str = ""


# ── examples ───────────────────────────────────────────────────────


_BYTEDANCE_PREP = _json.dumps([
    {"question": "讲讲 Transformer 自注意力为什么除以 √d",
     "category": "technical", "likelihood": 0.85,
     "rationale": "JD 要求 Transformer 内部机制"},
    {"question": "GRPO vs PPO 区别在哪",
     "category": "technical", "likelihood": 0.7,
     "rationale": "JD 提到 GRPO，简历也用了"},
    {"question": "讲一个跨团队协作的例子",
     "category": "behavioral", "likelihood": 0.55,
     "rationale": "校招通用 STAR"},
    {"question": "你做的 RemeDi loss 曲线长什么样？",
     "category": "project_deep_dive", "likelihood": 0.75,
     "rationale": "简历项目"},
    {"question": "字节做 LLM 应用 vs 平台你怎么看",
     "category": "company_specific", "likelihood": 0.4,
     "rationale": "字节业务相关"},
], ensure_ascii=False)

_BYTEDANCE_TRANSCRIPT_HIGH_HIT = """字节 Seed 实习一面 复盘 (2026-04-30):

面试官先问 Transformer attention 缩放因子推导，我答了 √d 防止 dot-product 高维饱和，
softmax 进入饱和区梯度消失。面试官点头，追问 "如果维度更高怎么办"，我答用 layer norm
+ scaled dot-product。

第二题: GRPO 和 PPO 区别，我答 GRPO 不需要 critic，用 group baseline。这个我答得很自信。

第三题: 让我深挖 RemeDi 项目，问 loss 曲线，问双流架构选型。我答得不错，把 TPS+UPS 设计
讲清楚了。

最后 system design: 设计一个支持百万 QPS 的 agent inference pipeline。我没准备过这个，
答得磕磕巴巴，提到了 batch + speculative decoding 但没展开。

整体: 4 个题预测命中 3 个，system design 是 surprise，我没准备好。"""


_ALI_PREP = _json.dumps([
    {"question": "讲讲 evidence-centric 上下文",
     "category": "foundational", "likelihood": 0.8,
     "rationale": "简历明确提到"},
    {"question": "为什么不用 ReAct",
     "category": "challenge", "likelihood": 0.7,
     "rationale": "设计选择会被挑战"},
    {"question": "100 个并发 agent 怎么改",
     "category": "extension", "likelihood": 0.5,
     "rationale": "JD 大规模 hint"},
], ensure_ascii=False)

_ALI_TRANSCRIPT_LOW_HIT = """阿里达摩院 Agent 评测一面 复盘:

面试官几乎没问我项目，全程在问业务场景:
1. 给你一个新业务，你怎么设计 agent 评测平台
2. OOD 数据怎么生成
3. BLEU vs LLM-as-judge 各自适用场景

我准备的 evidence-centric 完全没问到，ReAct 对比也没问。最后才简单问了下 RemeDi。

整体: 我们预测严重失准，达摩院偏业务设计而不是项目深挖。"""


EXAMPLES: tuple[PostInterviewExample, ...] = (
    PostInterviewExample(
        name="bytedance_high_hit",
        band="real",
        company="字节跳动",
        prep_questions_json=_BYTEDANCE_PREP,
        actual_transcript=_BYTEDANCE_TRANSCRIPT_HIGH_HIT,
        expected_hit_rate_min=0.5,
        expected_hit_rate_max=0.85,
        notes="高命中率案例：4/5 预测在某种形式上出现",
    ),
    PostInterviewExample(
        name="ali_low_hit",
        band="real",
        company="阿里巴巴",
        prep_questions_json=_ALI_PREP,
        actual_transcript=_ALI_TRANSCRIPT_LOW_HIT,
        expected_hit_rate_min=0.0,
        expected_hit_rate_max=0.3,
        notes="低命中率：达摩院偏业务，我们预测了项目深挖。brief_delta 应大",
    ),
    PostInterviewExample(
        name="empty_transcript",
        band="edge_case",
        company="某公司",
        prep_questions_json=_BYTEDANCE_PREP,
        actual_transcript="还没面",
        expected_hit_rate_min=0.0,
        expected_hit_rate_max=0.1,
        notes="空 transcript：所有 match=miss，hit_rate=0",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


_GENERIC_MISS_PHRASES = ("意外问题", "出乎意料", "没想到", "不知道为什么")


def metric(
    example: PostInterviewExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: model did not emit valid JSON.")

    schema_score = 0.0
    schema_note = ""
    result: PostInterviewReflection | None = None
    try:
        result = PostInterviewReflection.model_validate(parsed)
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

    # Hit rate in band
    if example.expected_hit_rate_min <= result.hit_rate <= example.expected_hit_rate_max:
        hit_band_score = 1.0
        hit_note = f"hit_rate {result.hit_rate:.2f} 在期望区间 ✓"
    else:
        dist = min(
            abs(result.hit_rate - example.expected_hit_rate_min),
            abs(result.hit_rate - example.expected_hit_rate_max),
        )
        hit_band_score = max(0.0, 1.0 - 3.0 * dist)
        hit_note = (
            f"hit_rate {result.hit_rate:.2f} 偏离期望 "
            f"[{example.expected_hit_rate_min:.2f}, {example.expected_hit_rate_max:.2f}]"
        )

    # Calibration: lower error = higher score. Threshold 0.3 mean abs err.
    cal_err = result.calibration_score()
    calib_score = max(0.0, 1.0 - cal_err / 0.5)  # err=0 → 1, err=0.5 → 0
    calib_note = f"calibration_error {cal_err:.2f} (越低越好，0.5 上限)"

    # Surprises explained
    if result.surprises:
        ok = sum(
            1 for s in result.surprises
            if len(s.why_we_missed) > 10
            and not any(p in s.why_we_missed for p in _GENERIC_MISS_PHRASES)
        )
        surprises_score = ok / len(result.surprises)
        surprises_note = f"surprises 有效解释 {ok}/{len(result.surprises)}"
    else:
        surprises_score = 1.0
        surprises_note = "no surprises (skip)"

    # Stories grounded — triggered_by must reference transcript
    if result.suggested_stories:
        transcript = example.actual_transcript
        grounded = sum(
            1 for s in result.suggested_stories
            if any(token in transcript for token in re.findall(r"\w{2,}", s.triggered_by)[:3])
        )
        stories_score = grounded / len(result.suggested_stories)
        stories_note = f"suggested_stories 锚定 transcript {grounded}/{len(result.suggested_stories)}"
    else:
        stories_score = 1.0
        stories_note = "no story suggestions (skip)"

    # Brief delta specificity
    addition = result.brief_delta.interview_style_addition
    if addition is None:
        # Null is acceptable when miss/empty case
        brief_score = 1.0
        brief_note = "brief_delta.interview_style_addition = null (acceptable)"
    elif len(addition) > 12 and any(
        c.isalnum() or '一' <= c <= '鿿' for c in addition
    ):
        brief_score = 1.0
        brief_note = f"brief_delta interview_style_addition 具体 ({len(addition)} chars) ✓"
    else:
        brief_score = 0.4
        brief_note = f"brief_delta 太短或空泛: {addition[:40]!r}"

    total = (
        _W_SCHEMA * schema_score
        + _W_HIT_BAND * hit_band_score
        + _W_CALIB * calib_score
        + _W_SURPRISES * surprises_score
        + _W_STORIES * stories_score
        + _W_BRIEF * brief_score
    )

    feedback = "\n".join([
        f"案例: {example.name} ({example.band})",
        schema_note, hit_note, calib_note, surprises_note,
        stories_note, brief_note,
        f"score: schema={schema_score:.2f} hit_band={hit_band_score:.2f}"
        f" calib={calib_score:.2f} surprises={surprises_score:.2f}"
        f" stories={stories_score:.2f} brief={brief_score:.2f}"
        f" → total={total:.2f}",
    ])

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "hit_rate_in_band": hit_band_score,
            "calibration_low_error": calib_score,
            "surprises_explained": surprises_score,
            "stories_grounded": stories_score,
            "brief_delta_specific": brief_score,
        },
        feedback=feedback,
    )


def _zero(example: PostInterviewExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[PostInterviewExample], list[PostInterviewExample]]:
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[PostInterviewExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[PostInterviewExample] = []
    val: list[PostInterviewExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
