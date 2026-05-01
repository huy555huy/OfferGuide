"""``successful_profile`` SKILL adapter.

Metric optimizes for:

- ``schema``                — output validates against SuccessfulProfileResult
- ``evidence_attribution``  — every why_they_passed item references evidence
                              with "(来自 X 条 ...)" pattern (anti-fabrication)
- ``cross_kind_diversity``  — uses ≥ 2 distinct content_kinds
                              (single-kind synthesis is weaker than multi-kind)
- ``concrete_skill_pattern``— must_have items contain concrete keywords
                              (no "深度学习基础" placeholders)
- ``honest_uncertainty``    — uncertainty_notes is non-trivial when sample
                              is small (< 5 evidence)
- ``no_marketer_residue``   — no marketer phrases ("加微信", "训练营")
                              leaked into the synthesis output

The metric is mostly *structural* — we don't have ground-truth profiles
to compare against pre-dogfood. Once 5-10 users have actually applied
to companies and reported back, we can replace ``schema`` weight with
``user_validation_match`` (user marks each profile field as accurate/
inaccurate/missing and the metric measures recall + precision).
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from ...skills.successful_profile.helpers import SuccessfulProfileResult
from ._base import MetricBreakdown, parse_json_output

name: str = "successful_profile"
INPUT_NAMES: list[str] = ["company", "role_hint", "high_quality_samples_json"]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "evidence_attribution",
    "cross_kind_diversity",
    "concrete_skill_pattern",
    "honest_uncertainty",
    "no_marketer_residue",
]

_W_SCHEMA = 0.20
_W_EVIDENCE = 0.20
_W_DIVERSITY = 0.15
_W_CONCRETE = 0.20
_W_UNCERTAINTY = 0.10
_W_NO_MARKETER = 0.15


@dataclass(frozen=True)
class SuccessfulProfileExample:
    """One golden case for evaluating successful_profile outputs."""
    name: str
    company: str
    role_hint: str
    high_quality_samples_json: str
    band: str = "real"
    notes: str = ""


# ── examples ───────────────────────────────────────────────────────


_BYTEDANCE_AI_AGENT_SAMPLES = _json.dumps([
    {
        "id": 1, "content_kind": "offer_post", "source": "nowcoder",
        "source_url": "https://nowcoder.com/discuss/x", "quality_score": 0.85,
        "raw_text": (
            "字节 AI Agent 后端实习 offer 复盘\n"
            "本人: 上海交大计算机硕士, 之前美团 ML 实习一段 (3 个月)\n"
            "项目: 1) 自主搭建的 LLM 多 agent 评测平台 (基于 LangGraph),\n"
            "  收集了 200+ 测试用例, 覆盖 ReAct/CoT/Tool-call 等模式\n"
            "  2) RAG 系统 (BGE-large + reranker), 内部知识库召回 hit@5 0.78\n"
            "面试: 一面问了 LangGraph state 设计、agent loop 控制权、\n"
            "  RAG 怎么 debug retrieval miss; 二面深挖项目 + 问 GRPO/PPO 区别\n"
            "  三面 system design: 设计支持 1000 QPS agent 服务\n"
            "感觉过的原因: 项目深度 + 系统设计的工程取舍 + 问问题问得深\n"
        ),
    },
    {
        "id": 2, "content_kind": "interview", "source": "nowcoder",
        "source_url": "https://nowcoder.com/discuss/y", "quality_score": 0.72,
        "raw_text": (
            "字节 Seed 实习一面 (2026-04):\n"
            "1) 手撕反转链表 (leetcode 206)\n"
            "2) Transformer attention 缩放 √d 推导\n"
            "3) 你 LangGraph 项目 state machine 怎么设计\n"
            "4) RAG 你的 reranker 用什么\n"
            "5) GRPO vs PPO 对比\n"
            "面试官 30min 中后期一直追问项目细节, 我答得不错\n"
        ),
    },
    {
        "id": 3, "content_kind": "project_share", "source": "github",
        "source_url": "https://github.com/x/agent-eval", "quality_score": 0.80,
        "raw_text": (
            "我做的开源 agent 评测框架介绍 (拿到字节 offer 后写的)\n"
            "技术栈: LangGraph + DSPy + Pydantic\n"
            "核心: 把 200+ 测试用例参数化, 支持 ReAct/CoT/Tool-call 三模式对比\n"
            "评估指标: 任务完成率、tool call 成功率、token 消耗\n"
            "数据: 在 GAIA benchmark 上 ReAct 0.34, CoT 0.41, Tool-call 0.52\n"
        ),
    },
], ensure_ascii=False)


_TENCENT_BACKEND_SPARSE = _json.dumps([
    {
        "id": 4, "content_kind": "interview", "source": "manual_paste",
        "source_url": "", "quality_score": 0.65,
        "raw_text": "腾讯 PCG 后端一面: 1) MySQL 索引 2) Redis 数据结构 3) 项目 4) HR\n",
    },
], ensure_ascii=False)


EXAMPLES: tuple[SuccessfulProfileExample, ...] = (
    SuccessfulProfileExample(
        name="bytedance_ai_agent_rich",
        band="real",
        company="字节跳动",
        role_hint="AI Agent 后端实习",
        high_quality_samples_json=_BYTEDANCE_AI_AGENT_SAMPLES,
        notes="3 条高质量样本 + 跨 kind (offer_post + interview + project_share)",
    ),
    SuccessfulProfileExample(
        name="tencent_backend_sparse",
        band="edge_case",
        company="腾讯",
        role_hint="后端实习",
        high_quality_samples_json=_TENCENT_BACKEND_SPARSE,
        notes="只有 1 条质量一般的样本; uncertainty_notes 必须诚实",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


_VAGUE_SKILLS = (
    "深度学习基础", "扎实基础", "良好的代码能力",
    "学习能力强", "团队精神", "沟通能力",
)
_MARKETER_RESIDUE = (
    "加微信", "加 V", "加wx", "训练营", "包过", "资料包",
    "1v1 辅导", "网课代购", "公众号关注", "私信我", "DM 我",
)
_EVIDENCE_PATTERN = re.compile(r"来自\s*\d+\s*条|根据\s*\d+\s*条|\d+\s*条样本")


def metric(
    example: SuccessfulProfileExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: model did not emit valid JSON.")

    schema_score = 0.0
    schema_note = ""
    result: SuccessfulProfileResult | None = None
    try:
        result = SuccessfulProfileResult.model_validate(parsed)
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

    # Evidence attribution: each why_they_passed item should reference
    # evidence count, e.g. "(来自 2 条 offer_post)"
    if result.why_they_passed:
        attributed = sum(
            1 for r in result.why_they_passed if _EVIDENCE_PATTERN.search(r)
        )
        evidence_score = attributed / len(result.why_they_passed)
        evidence_note = (
            f"why_they_passed 带证据归属 {attributed}/{len(result.why_they_passed)}"
        )
    else:
        evidence_score = 0.0
        evidence_note = "why_they_passed 为空"

    # Cross-kind diversity
    unique_kinds = len(set(result.evidence_kinds))
    if unique_kinds >= 2:
        diversity_score = 1.0
    elif unique_kinds == 1:
        diversity_score = 0.5
    else:
        diversity_score = 0.0
    diversity_note = f"覆盖 {unique_kinds} 种 content_kind"

    # Concrete skill pattern
    must_have = result.skill_pattern.must_have
    if must_have:
        vague = sum(1 for s in must_have if any(v in s for v in _VAGUE_SKILLS))
        concrete_score = max(0.0, 1.0 - 1.5 * vague / len(must_have))
        concrete_note = (
            f"must_have 具体度: {len(must_have) - vague}/{len(must_have)} 条非空泛"
        )
    else:
        concrete_score = 0.5  # no skills listed is neither great nor terrible
        concrete_note = "must_have 为空 (中性)"

    # Honest uncertainty: when evidence is small (< 5), expect non-trivial
    # uncertainty_notes
    if result.evidence_count < 5:
        if result.uncertainty_notes and any(
            len(n) > 10 for n in result.uncertainty_notes
        ):
            uncertainty_score = 1.0
            uncertainty_note = "样本少时承认不确定性 ✓"
        else:
            uncertainty_score = 0.0
            uncertainty_note = (
                f"样本只有 {result.evidence_count} 条，但没有像样的 uncertainty_notes"
            )
    else:
        # With ample evidence, uncertainty_notes is allowed to be empty
        uncertainty_score = 1.0
        uncertainty_note = "样本充足，uncertainty_notes 不强制"

    # No marketer residue
    full_text = _json.dumps(parsed, ensure_ascii=False)
    leaks = sum(1 for m in _MARKETER_RESIDUE if m in full_text)
    if leaks == 0:
        no_marketer_score = 1.0
        no_marketer_note = "无卖课残留 ✓"
    else:
        no_marketer_score = max(0.0, 1.0 - 0.5 * leaks)
        no_marketer_note = f"⚠ 出现 {leaks} 处卖课残留词"

    total = (
        _W_SCHEMA * schema_score
        + _W_EVIDENCE * evidence_score
        + _W_DIVERSITY * diversity_score
        + _W_CONCRETE * concrete_score
        + _W_UNCERTAINTY * uncertainty_score
        + _W_NO_MARKETER * no_marketer_score
    )

    feedback = "\n".join([
        f"案例: {example.name} ({example.band})",
        schema_note, evidence_note, diversity_note,
        concrete_note, uncertainty_note, no_marketer_note,
        f"score: schema={schema_score:.2f} ev={evidence_score:.2f}"
        f" div={diversity_score:.2f} concrete={concrete_score:.2f}"
        f" unc={uncertainty_score:.2f} no_mkt={no_marketer_score:.2f}"
        f" → total={total:.2f}",
    ])

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "evidence_attribution": evidence_score,
            "cross_kind_diversity": diversity_score,
            "concrete_skill_pattern": concrete_score,
            "honest_uncertainty": uncertainty_score,
            "no_marketer_residue": no_marketer_score,
        },
        feedback=feedback,
    )


def _zero(example: SuccessfulProfileExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[SuccessfulProfileExample], list[SuccessfulProfileExample]]:
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[SuccessfulProfileExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[SuccessfulProfileExample] = []
    val: list[SuccessfulProfileExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
