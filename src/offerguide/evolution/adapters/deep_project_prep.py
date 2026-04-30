"""``deep_project_prep`` SKILL adapter.

Without dogfood data on which generated questions actually got asked,
the metric optimizes for **shape correctness** that we know correlates
with usefulness for project-deep-dive prep:

- ``schema``                — output validates against
                               ``DeepProjectPrepResult`` (extra='forbid',
                               likelihoods ∈ [0,1], min/max list lengths)
- ``project_count``         — analyzed 1-4 projects (sweet spot, neither
                               cherry-pick one nor sprawl)
- ``type_coverage``         — each project's probing_questions span ≥ 3
                               of the 5 question types (avoids "all
                               foundational" or "all deep_dive" failure)
- ``rationale_grounded``    — fraction of probing rationales that
                               reference profile keywords or JD keywords
                               (anti-hallucination)
- ``outline_concreteness``  — answer_outline bullets must look concrete:
                               contain numbers, technical terms, or
                               named tradeoffs (not just "be confident")
- ``behavioral_specificity`` — fraction of behavioral questions that
                               name a specific experience from the user
                               profile (not generic STAR)

When ~30 dogfood interview reflections exist, replace
``rationale_grounded`` with ``actual_question_hit_rate``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from ...skills.deep_project_prep.helpers import DeepProjectPrepResult
from ._base import MetricBreakdown, parse_json_output

name: str = "deep_project_prep"
INPUT_NAMES: list[str] = ["company", "job_text", "user_profile"]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "project_count",
    "type_coverage",
    "rationale_grounded",
    "outline_concreteness",
    "behavioral_specificity",
]

_W_SCHEMA = 0.20
_W_PROJECT_COUNT = 0.10
_W_TYPE_COVERAGE = 0.20
_W_RATIONALE_GROUNDED = 0.20
_W_OUTLINE_CONCRETE = 0.15
_W_BEHAVIORAL_SPEC = 0.15


@dataclass(frozen=True)
class DeepProjectPrepExample:
    """One ground-truth case for deep_project_prep."""

    name: str
    company: str
    job_text: str
    user_profile: str

    profile_keywords: tuple[str, ...] = field(default_factory=tuple)
    """Keywords from the profile that *some* probing rationale or
    behavioral question should reference."""

    jd_keywords: tuple[str, ...] = field(default_factory=tuple)
    """JD keywords that *some* probing rationale should reference."""

    expected_min_projects: int = 2
    """The simulated panel ought to deep-dive at least this many."""

    band: str = "real"
    notes: str = ""


# The same canonical user profile used by the other adapters.
_USER = """胡阳，上海财经大学应用统计专硕（2025-2027 在读）。
项目:
1) Deep Research Agent（法至科技实习）— agent runtime，双层架构（语义层 + 工作区层），
   evidence-centric 上下文管理，closure semantics。
2) RemeDi（个人）— 基于 LLaDA-8B 的扩散语言模型，双流架构 (TPS + UPS) + 零初始化投影 +
   LoRA 微调；SFT + GRPO；DeepSpeed ZeRO-2 + 梯度累积。
技能：Python, PyTorch, Transformers, RL (PPO/GRPO), DeepSpeed, AI Agent 架构。"""


EXAMPLES: tuple[DeepProjectPrepExample, ...] = (
    DeepProjectPrepExample(
        name="bytedance_seed_post_training",
        band="real",
        company="字节跳动 Seed",
        job_text=(
            "大模型后训练算法实习生 — 字节 Seed\n"
            "要求：精通 PyTorch + Transformer 内部机制；至少一段 SFT/DPO/GRPO/RLHF 项目；"
            "熟悉 DeepSpeed/Megatron 等分布式训练；能 debug 训练不稳。"
        ),
        user_profile=_USER,
        profile_keywords=("RemeDi", "GRPO", "DeepSpeed", "扩散", "LoRA"),
        jd_keywords=("PyTorch", "Transformer", "GRPO", "DeepSpeed", "Megatron"),
        expected_min_projects=2,
        notes="字节后训练岗，必深挖 RemeDi (训练管线) + Deep Research Agent (RL 应用)。",
    ),
    DeepProjectPrepExample(
        name="ali_agent_intern",
        band="real",
        company="阿里巴巴",
        job_text=(
            "AI Agent 实习生 — 阿里巴巴\n"
            "要求：Python + C/C++/Java；Transformer 与 LLM 深入理解；至少一个 Agent 项目；"
            "熟悉 RAG / Memory / Tool Use；LangGraph / DSPy 加分。"
        ),
        user_profile=_USER,
        profile_keywords=("Deep Research Agent", "agent runtime", "evidence-centric"),
        jd_keywords=("Agent", "LangGraph", "RAG", "Tool Use", "Memory"),
        expected_min_projects=2,
        notes="阿里 agent 岗，深挖 Deep Research Agent 是必然；C/C++ 是简历真实 gap。",
    ),
    DeepProjectPrepExample(
        name="tencent_llm_application",
        band="real",
        company="腾讯",
        job_text=(
            "LLM 应用工程师 — 腾讯\n"
            "要求：大模型应用落地经验；RAG/多模态/Agent 任一方向；Python；2C 产品 sense 加分。"
        ),
        user_profile=_USER,
        profile_keywords=("Agent", "RemeDi", "agent runtime"),
        jd_keywords=("LLM", "应用", "RAG", "Agent"),
        expected_min_projects=1,
        notes="腾讯偏业务岗，会问'你的项目能不能落地业务'。",
    ),
    DeepProjectPrepExample(
        name="quant_research_misalign",
        band="edge_case",
        company="某私募",
        job_text=(
            "量化研究实习 — 某私募\n"
            "要求：Python；时序分析；Alpha 因子挖掘；回测平台经验。"
        ),
        user_profile=_USER,
        profile_keywords=("Python",),
        jd_keywords=("时序", "Alpha", "回测"),
        expected_min_projects=1,
        notes="技术栈半沾，weak_points 应该突出'金融经验为零'。",
    ),
    DeepProjectPrepExample(
        name="meituan_business_llm",
        band="real",
        company="美团",
        job_text=(
            "LLM 应用算法实习 — 美团\n"
            "要求：一段 LLM 应用项目；Python；业务理解；search/推荐背景加分。"
        ),
        user_profile=_USER,
        profile_keywords=("Agent", "Deep Research Agent"),
        jd_keywords=("LLM", "应用"),
        expected_min_projects=1,
        notes="美团 O2O 业务岗，company_specific 题应聊外卖/到店业务。",
    ),
)


# ── concreteness heuristic ────────────────────────────────────────


_NUMBER_RE = re.compile(r"\d")
# Technical signal words: keep deliberately broad — false positive < false negative
_TECH_TERMS = {
    "softmax", "attention", "transformer", "layer", "loss", "gradient",
    "kv cache", "rope", "lora", "ppo", "grpo", "sft", "dpo", "rlhf",
    "deepspeed", "zero", "fsdp", "fp16", "bf16",
    "embedding", "tokenizer", "context", "agent", "rag", "memory",
    "langgraph", "dspy", "tool use",
    "diffusion", "denoising", "llama", "qwen",
    "vs", "对比", "权衡", "区别", "差异", "选型", "因为",
}


def _outline_concreteness_score(outline: list[str]) -> float:
    """Fraction of bullets containing a number OR a known technical term.

    Pure heuristic but discriminates "公式包含 √d 和 layer norm" from
    "解释清楚 attention" reliably enough for a starter metric.
    """
    if not outline:
        return 0.0
    hits = 0
    for bullet in outline:
        b = bullet.lower()
        if _NUMBER_RE.search(b):
            hits += 1
            continue
        if any(t in b for t in _TECH_TERMS):
            hits += 1
    return hits / len(outline)


def _rationale_grounded_score(
    questions: list, profile_keys: set[str], jd_keys: set[str]
) -> float:
    """Fraction of question rationales that reference at least one keyword
    from the profile or JD (vs. generic 'big tech often asks this')."""
    if not questions:
        return 1.0
    if not profile_keys and not jd_keys:
        return 1.0  # nothing to anchor against
    pool = profile_keys | jd_keys
    grounded = sum(1 for q in questions if any(k in q.rationale for k in pool))
    return grounded / len(questions)


def _behavioral_specificity_score(
    behaviorals: list, profile_keys: set[str]
) -> float:
    """Fraction of behavioral questions that mention a profile keyword
    (proxy for 'this isn't generic STAR template')."""
    if not behaviorals:
        return 1.0  # nothing to score
    if not profile_keys:
        return 1.0
    specific = sum(
        1 for q in behaviorals if any(k in q.question for k in profile_keys)
    )
    return specific / len(behaviorals)


# ── metric ─────────────────────────────────────────────────────────


def metric(
    example: DeepProjectPrepExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    """Score one (example, output) pair on 6 axes."""
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: model did not emit valid JSON.")

    # Schema
    schema_score = 0.0
    schema_note = ""
    result: DeepProjectPrepResult | None = None
    try:
        result = DeepProjectPrepResult.model_validate(parsed)
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
                "schema": 0.0, "project_count": 0.0, "type_coverage": 0.0,
                "rationale_grounded": 0.0, "outline_concreteness": 0.0,
                "behavioral_specificity": 0.0,
            },
            feedback=f"案例: {example.name}\n{schema_note}",
        )

    # 1. Project count score (sweet spot 2-4)
    n_proj = len(result.projects_analyzed)
    if example.expected_min_projects <= n_proj <= 4:
        project_count_score = 1.0
        proj_note = f"project_count={n_proj} ✓ (期望 ≥ {example.expected_min_projects})"
    else:
        if n_proj < example.expected_min_projects:
            project_count_score = max(0.0, n_proj / example.expected_min_projects)
            proj_note = f"project_count 太少: {n_proj} < {example.expected_min_projects}"
        else:
            project_count_score = max(0.0, 1.0 - 0.25 * (n_proj - 4))
            proj_note = f"project_count 太多: {n_proj} > 4"

    # 2. Type coverage per project — average over projects
    if result.projects_analyzed:
        coverage_per_proj = []
        for p in result.projects_analyzed:
            n_types = len(p.question_types_covered())
            coverage_per_proj.append(min(1.0, n_types / 3.0))
        type_coverage_score = sum(coverage_per_proj) / len(coverage_per_proj)
        worst = min(coverage_per_proj)
        type_note = (
            f"type_coverage 平均 {type_coverage_score:.2f}（worst project = {worst:.2f}）"
        )
    else:
        type_coverage_score = 0.0
        type_note = "no projects to score"

    # 3. Rationale grounded — over all questions (project + cross + behavioral)
    profile_keys = set(example.profile_keywords)
    jd_keys = set(example.jd_keywords)
    all_q = result.all_questions()
    rationale_grounded_score = _rationale_grounded_score(all_q, profile_keys, jd_keys)
    grounded_note = (
        f"rationale_grounded {rationale_grounded_score:.2f}"
        f" ({sum(1 for q in all_q if any(k in q.rationale for k in (profile_keys | jd_keys)))}"
        f"/{len(all_q)} 引用了 profile/JD 关键词)"
    )

    # 4. Outline concreteness — over all probing questions, mean per-bullet score
    if all_q:
        outline_scores = [_outline_concreteness_score(q.answer_outline) for q in all_q]
        outline_concreteness_score = sum(outline_scores) / len(outline_scores)
    else:
        outline_concreteness_score = 0.0
    outline_note = f"outline_concreteness {outline_concreteness_score:.2f}（理想 ≥ 0.6）"

    # 5. Behavioral specificity
    behavioral_specificity_score = _behavioral_specificity_score(
        result.behavioral_questions_tailored, profile_keys
    )
    behav_note = (
        f"behavioral_specificity {behavioral_specificity_score:.2f}"
        f" ({len(result.behavioral_questions_tailored)} 题)"
    )

    total = (
        _W_SCHEMA * schema_score
        + _W_PROJECT_COUNT * project_count_score
        + _W_TYPE_COVERAGE * type_coverage_score
        + _W_RATIONALE_GROUNDED * rationale_grounded_score
        + _W_OUTLINE_CONCRETE * outline_concreteness_score
        + _W_BEHAVIORAL_SPEC * behavioral_specificity_score
    )

    feedback = "\n".join(
        [
            f"案例: {example.name} ({example.band})",
            schema_note,
            proj_note,
            type_note,
            grounded_note,
            outline_note,
            behav_note,
            f"score: schema={schema_score:.2f} proj={project_count_score:.2f}"
            f" type={type_coverage_score:.2f} grounded={rationale_grounded_score:.2f}"
            f" outline={outline_concreteness_score:.2f} behav={behavioral_specificity_score:.2f}"
            f" → total={total:.2f}",
        ]
    )

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "project_count": project_count_score,
            "type_coverage": type_coverage_score,
            "rationale_grounded": rationale_grounded_score,
            "outline_concreteness": outline_concreteness_score,
            "behavioral_specificity": behavioral_specificity_score,
        },
        feedback=feedback,
    )


def _zero(example: DeepProjectPrepExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={
            "schema": 0.0,
            "project_count": 0.0,
            "type_coverage": 0.0,
            "rationale_grounded": 0.0,
            "outline_concreteness": 0.0,
            "behavioral_specificity": 0.0,
        },
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


def split_train_val(
    *,
    val_fraction: float = 0.4,
    seed: int = 0,
) -> tuple[list[DeepProjectPrepExample], list[DeepProjectPrepExample]]:
    """Stratified by ``band`` (real / edge_case)."""
    import random

    rng = random.Random(seed)
    by_band: dict[str, list[DeepProjectPrepExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)

    train: list[DeepProjectPrepExample] = []
    val: list[DeepProjectPrepExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val
