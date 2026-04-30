"""``write_cover_letter`` SKILL adapter.

Without dogfood data on which cover letters actually got HR responses,
this metric optimizes for **shape correctness** that we know correlates
with usefulness:

- ``schema``                  — output validates against
                                 ``CoverLetterResult`` (extra='forbid',
                                 length bounds, score in [0,1])
- ``ats_density``             — 0.3 ≤ keywords / paragraphs ≤ 0.85
                                 (too few = didn't read JD, too many =
                                 stuffing)
- ``customization_signal_count`` — ≥ 2 customization_signals
                                    (Career-Ops requires hard evidence)
- ``ai_risk_clean``           — penalize AI-giveaway phrases the model
                                 emitted in its own output (regex check)
- ``length_in_band``          — internship: 150-300 words; full-time:
                                 300-500 words. Penalize either extreme.
- ``personalization_realism`` — declared personalization_score ≤ actual
                                 customization_signal_count + ats_density
                                 (anti-overclaim)

When dogfood data lands, replace ``ai_risk_clean`` with the real
"reply-after-cover-letter" rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from ...skills.write_cover_letter.helpers import CoverLetterResult
from ._base import MetricBreakdown, parse_json_output

name: str = "write_cover_letter"
INPUT_NAMES: list[str] = ["company", "job_text", "user_profile"]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "ats_density",
    "customization",
    "ai_risk_clean",
    "length_in_band",
    "personalization_realism",
]

_W_SCHEMA = 0.20
_W_ATS = 0.20
_W_CUSTOM = 0.20
_W_AI_CLEAN = 0.15
_W_LENGTH = 0.10
_W_PERSONAL = 0.15


# AI-giveaway phrases (CN + EN) that 49% of ATS auto-flag.
_AI_GIVEAWAY_PATTERNS = [
    r"我谨此致函",
    r"由衷的兴趣",
    r"贵公司在.*?领域的卓越成就",
    r"本人热情饱满",
    r"赋能",
    r"打造闭环",
    r"全方位",
    r"I am writing to express my (?:keen )?interest",
    r"leverage my skills",
    r"\bsynergiz",
    r"highly motivated individual",
    r"exciting opportunity",
    r"cutting-edge team",
]


@dataclass(frozen=True)
class WriteCoverLetterExample:
    """One ground-truth case for write_cover_letter."""

    name: str
    company: str
    job_text: str
    user_profile: str
    expected_min_keywords: int = 4
    """JD keywords the cover letter must touch on (bare minimum)."""

    expected_word_min: int = 150
    expected_word_max: int = 500
    """Banded range for `overall_word_count`."""

    band: str = "real"
    notes: str = ""


_USER = """胡阳，上海财经大学应用统计专硕。
项目: Deep Research Agent (法至科技实习)；RemeDi (LLaDA-8B 扩散模型 + GRPO + DeepSpeed ZeRO-2)。
技能: Python, PyTorch, Transformers, RL (PPO/GRPO), AI Agent 架构。
偏好: 2026 暑期实习；地点上海/北京/杭州。"""


EXAMPLES: tuple[WriteCoverLetterExample, ...] = (
    WriteCoverLetterExample(
        name="bytedance_seed_intern",
        band="real",
        company="字节跳动",
        job_text=(
            "大模型后训练算法实习生 — 字节 Seed\n"
            "要求: PyTorch + SFT/DPO/GRPO/RLHF; DeepSpeed/Megatron 经验加分; "
            "Transformer 内部机制理解扎实。"
        ),
        user_profile=_USER,
        expected_min_keywords=4,
        expected_word_min=150,
        expected_word_max=350,
        notes="技术栈高度重合，应至少触 GRPO + DeepSpeed + PyTorch + Transformer",
    ),
    WriteCoverLetterExample(
        name="ali_agent_intern",
        band="real",
        company="阿里巴巴",
        job_text=(
            "AI Agent 实习生 — 阿里达摩院\n"
            "要求: Python; Transformer + LLM; Agent 项目经验; LangGraph/DSPy 加分。"
        ),
        user_profile=_USER,
        expected_min_keywords=3,
        expected_word_min=150,
        expected_word_max=350,
        notes="Agent 项目对得上，但 LangGraph 是 gap，cover letter 不应假装会",
    ),
    WriteCoverLetterExample(
        name="quant_misalign",
        band="edge_case",
        company="某私募",
        job_text=(
            "量化研究实习 — 某私募\n"
            "要求: Python; 时序分析; Alpha 因子挖掘; 回测平台经验; C++ 加分。"
        ),
        user_profile=_USER,
        expected_min_keywords=1,
        expected_word_min=120,
        expected_word_max=280,
        notes="技术栈半沾，cover letter 应坦诚 gap，不应 padding",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


def metric(
    example: WriteCoverLetterExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: model did not emit valid JSON.")

    # Schema
    schema_score = 0.0
    schema_note = ""
    result: CoverLetterResult | None = None
    try:
        result = CoverLetterResult.model_validate(parsed)
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

    # ATS density
    n_keywords = len(result.ats_keywords_used)
    n_paragraphs = max(1, len(result.narrative_body))
    density = n_keywords / n_paragraphs
    if 1.5 <= density <= 5.0:
        ats_density_score = 1.0
        ats_note = f"ats_density {density:.2f} 在合理区间 ✓ ({n_keywords}/{n_paragraphs})"
    elif density < 1.5:
        ats_density_score = max(0.0, density / 1.5)
        ats_note = f"ats_density 太低 {density:.2f} — JD 关键词没用上"
    else:
        ats_density_score = max(0.0, 1.0 - 0.2 * (density - 5.0))
        ats_note = f"ats_density 过高 {density:.2f} — 关键词堆砌"
    if n_keywords < example.expected_min_keywords:
        ats_density_score *= 0.6
        ats_note += f" (低于 expected_min={example.expected_min_keywords})"

    # Customization signals
    n_signals = len(result.customization_signals)
    if n_signals >= 2:
        custom_score = 1.0
        custom_note = f"customization_signals 数 {n_signals} ≥ 2 ✓"
    else:
        custom_score = n_signals / 2.0
        custom_note = (
            f"customization_signals 只有 {n_signals} 条，应该 ≥ 2 (Career-Ops 标准)"
        )

    # AI giveaway scan over actual rendered text
    rendered = result.render_plain()
    ai_hits: list[str] = []
    for pattern in _AI_GIVEAWAY_PATTERNS:
        if re.search(pattern, rendered, flags=re.IGNORECASE):
            ai_hits.append(pattern)
    if ai_hits:
        ai_clean_score = max(0.0, 1.0 - 0.25 * len(ai_hits))
        ai_note = f"AI 套话命中 {len(ai_hits)} 条: {ai_hits[:3]}"
    else:
        ai_clean_score = 1.0
        ai_note = "无 AI 套话 ✓"

    # Length
    wc = result.overall_word_count
    if example.expected_word_min <= wc <= example.expected_word_max:
        length_score = 1.0
        length_note = f"word_count {wc} 在 {example.expected_word_min}-{example.expected_word_max} ✓"
    elif wc < example.expected_word_min:
        length_score = max(0.0, wc / example.expected_word_min)
        length_note = f"太短 {wc} < {example.expected_word_min}"
    else:
        length_score = max(0.0, 1.0 - 0.005 * (wc - example.expected_word_max))
        length_note = f"太长 {wc} > {example.expected_word_max}"

    # Personalization realism: declared score should be supported by
    # signal count and ats density. Anti-overclaim.
    declared = result.personalization_score
    realistic_max = min(1.0, 0.3 + 0.2 * n_signals + 0.1 * min(n_keywords, 8) / 8)
    if declared <= realistic_max + 0.15:
        personal_score = 1.0
        personal_note = f"personalization {declared:.2f} ≤ realistic {realistic_max:.2f} ✓"
    else:
        gap = declared - realistic_max
        personal_score = max(0.0, 1.0 - 2 * gap)
        personal_note = (
            f"personalization 过自评: declared {declared:.2f} > realistic {realistic_max:.2f}"
        )

    total = (
        _W_SCHEMA * schema_score
        + _W_ATS * ats_density_score
        + _W_CUSTOM * custom_score
        + _W_AI_CLEAN * ai_clean_score
        + _W_LENGTH * length_score
        + _W_PERSONAL * personal_score
    )

    feedback = "\n".join([
        f"案例: {example.name} ({example.band})",
        schema_note, ats_note, custom_note, ai_note, length_note, personal_note,
        f"score: schema={schema_score:.2f} ats={ats_density_score:.2f}"
        f" custom={custom_score:.2f} ai_clean={ai_clean_score:.2f}"
        f" length={length_score:.2f} personal={personal_score:.2f}"
        f" → total={total:.2f}",
    ])

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "ats_density": ats_density_score,
            "customization": custom_score,
            "ai_risk_clean": ai_clean_score,
            "length_in_band": length_score,
            "personalization_realism": personal_score,
        },
        feedback=feedback,
    )


def _zero(example: WriteCoverLetterExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[WriteCoverLetterExample], list[WriteCoverLetterExample]]:
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[WriteCoverLetterExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[WriteCoverLetterExample] = []
    val: list[WriteCoverLetterExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
