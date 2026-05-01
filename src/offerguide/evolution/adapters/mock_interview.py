"""``mock_interview`` SKILL adapter.

Metric axes:

- ``schema``                  — output validates as MockInterviewResult
- ``score_calibration``       — score not centered on 0.7 (校招 distribution)
- ``improvements_specific``   — each improvement is "X instead of Y" not vague
- ``coverage_diversity``      — over an 8-round session, ≥ 3 categories
- ``adaptive_difficulty``     — difficulty adapts to last score (>= 0.75 → hard,
                                < 0.4 → easy)
- ``no_filler_praise``        — strengths don't contain "答得不错/很好/加油"

Pre-dogfood structural metric — once we have user feedback on score
accuracy (user re-grades own answer vs agent's grade), replace
``score_calibration`` with ``user_score_match``.
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from ...skills.mock_interview.helpers import MockInterviewResult
from ._base import MetricBreakdown, parse_json_output

name: str = "mock_interview"
INPUT_NAMES: list[str] = [
    "company", "role_focus", "user_resume",
    "prep_questions_json", "turn_history_json", "last_user_answer",
]
METRIC_AXES: list[str] = [
    "total",
    "schema",
    "score_calibration",
    "improvements_specific",
    "coverage_diversity",
    "adaptive_difficulty",
    "no_filler_praise",
]

_W_SCHEMA = 0.20
_W_CAL = 0.15
_W_IMPR = 0.20
_W_COV = 0.15
_W_ADAPT = 0.15
_W_NO_FILLER = 0.15


@dataclass(frozen=True)
class MockInterviewExample:
    name: str
    company: str
    role_focus: str
    user_resume: str
    prep_questions_json: str
    turn_history_json: str
    last_user_answer: str
    band: str = "real"
    notes: str = ""


# ── examples ───────────────────────────────────────────────────────


_USER_RESUME = (
    "胡阳 上财应统硕士 2027 届\n"
    "实习: 法至科技 NLP 工程师 (LangGraph 多 agent 评测)\n"
    "项目: RemeDi (BERT 双塔 AUC+0.04), Deep Research Agent (LangGraph+DSPy)\n"
)

_PREP_QS = _json.dumps([
    {"question": "讲讲 LangGraph state 怎么设计", "category": "technical",
     "likelihood": 0.85, "rationale": "JD 要求"},
    {"question": "RemeDi 双塔为什么选 BERT", "category": "project_deep_dive",
     "likelihood": 0.7, "rationale": "简历项目"},
], ensure_ascii=False)


EXAMPLES: tuple[MockInterviewExample, ...] = (
    MockInterviewExample(
        name="turn_1_no_history",
        band="real",
        company="字节跳动",
        role_focus="AI Agent 后端实习",
        user_resume=_USER_RESUME,
        prep_questions_json=_PREP_QS,
        turn_history_json="[]",
        last_user_answer="",
        notes="第 1 轮 — 应该 evaluation=null, next_question 出 medium 难度的 technical",
    ),
    MockInterviewExample(
        name="turn_3_strong_answer",
        band="real",
        company="字节跳动",
        role_focus="AI Agent 后端实习",
        user_resume=_USER_RESUME,
        prep_questions_json=_PREP_QS,
        turn_history_json=_json.dumps([
            {"question": "讲讲 LangGraph state 设计",
             "user_answer": "...",
             "evaluation": {"score": 0.78, "scoring_dimensions":
                {"factual_accuracy": 0.8, "depth": 0.75,
                 "structure": 0.85, "evidence": 0.7}}},
            {"question": "RAG 怎么 debug retrieval miss",
             "user_answer": "...",
             "evaluation": {"score": 0.82, "scoring_dimensions":
                {"factual_accuracy": 0.85, "depth": 0.85,
                 "structure": 0.75, "evidence": 0.8}}},
        ], ensure_ascii=False),
        last_user_answer=(
            "GRPO 比 PPO 简化了 advantage 估计, 不需要 critic, "
            "用 group baseline. 但 group size 选小了方差大, 选大了"
            "效率低, 我会用 8-16 之间; 在 RemeDi 里我没用 RL 但读了 trl 库"
        ),
        notes="连续两轮 0.75+ 后 — 这轮 score 应在 0.7+, next_question 应该 hard",
    ),
)


# ── metric ─────────────────────────────────────────────────────────


_FILLER_PHRASES = (
    "答得不错", "很好", "加油", "保持", "继续努力", "回答得很棒",
    "总体不错", "不错的回答",
)
_VAGUE_IMPROVEMENT_PATTERNS = (
    "加强", "提升", "扎实", "好好", "多练", "深入理解",
)


def metric(
    example: MockInterviewExample,
    raw_output: str | dict,
) -> MetricBreakdown:
    parsed = parse_json_output(raw_output)
    if not parsed:
        return _zero(example, "OUTPUT_PARSE_FAILURE: invalid JSON")

    result: MockInterviewResult | None = None
    schema_score = 0.0
    schema_note = ""
    try:
        result = MockInterviewResult.model_validate(parsed)
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

    # score_calibration: when an evaluation is present, score should NOT
    # cluster around 0.7 (校招 reality is mean ~0.55)
    eval_present = result.evaluation_of_last_answer is not None
    if eval_present:
        s = result.evaluation_of_last_answer.score  # type: ignore[union-attr]
        # Penalize if score is in the 0.65-0.78 "lazy default" band when
        # the user's actual answer might warrant something more decisive
        if 0.65 <= s <= 0.78:
            cal_score = 0.5
            cal_note = f"score={s:.2f} 落在'平均偏 0.7'懒区"
        else:
            cal_score = 1.0
            cal_note = f"score={s:.2f} 决定性 ✓"
    else:
        cal_score = 1.0
        cal_note = "first turn (no eval, skip)"

    # improvements_specific
    if eval_present and result.evaluation_of_last_answer.improvements:  # type: ignore[union-attr]
        imps = result.evaluation_of_last_answer.improvements  # type: ignore[union-attr]
        ok = sum(
            1 for imp in imps
            if not any(p in imp for p in _VAGUE_IMPROVEMENT_PATTERNS)
            and len(imp) >= 12
        )
        impr_score = ok / len(imps)
        impr_note = f"improvements 具体度 {ok}/{len(imps)}"
    else:
        impr_score = 1.0
        impr_note = "no improvements (skip)"

    # coverage_diversity: at this turn, count categories in turn_history + this
    cats: set[str] = set()
    try:
        prior = _json.loads(example.turn_history_json)
        for t in prior:
            q = t.get("question") or ""
            cats.add(_guess_category(q))
    except _json.JSONDecodeError:
        prior = []
    if result.next_question:
        cats.add(result.next_question.category)
    coverage_score = min(1.0, len(cats) / 3)
    coverage_note = f"覆盖 {len(cats)} 类 (target ≥3)"

    # adaptive_difficulty: if last eval has score, next q difficulty should
    # adapt
    if eval_present and result.next_question:
        last_score = result.evaluation_of_last_answer.score  # type: ignore[union-attr]
        diff = result.next_question.difficulty
        expected = (
            "hard" if last_score >= 0.75
            else "easy" if last_score < 0.4
            else "medium"
        )
        if diff == expected:
            adapt_score = 1.0
            adapt_note = f"难度自适应正确 ({last_score:.2f} → {diff}) ✓"
        else:
            adapt_score = 0.5
            adapt_note = f"难度 {diff} ≠ 期望 {expected} ({last_score:.2f})"
    else:
        adapt_score = 1.0
        adapt_note = "first turn 或终局 (skip)"

    # no_filler_praise
    if eval_present:
        strengths = result.evaluation_of_last_answer.strengths  # type: ignore[union-attr]
        bad = sum(
            1 for s in strengths
            if any(p in s for p in _FILLER_PHRASES)
        )
        if not strengths:
            filler_score = 1.0
            filler_note = "strengths 为空 (skip)"
        elif bad == 0:
            filler_score = 1.0
            filler_note = "无水货 ✓"
        else:
            filler_score = max(0.0, 1.0 - 0.5 * bad)
            filler_note = f"⚠ {bad} 处水货措辞"
    else:
        filler_score = 1.0
        filler_note = "first turn (skip)"

    total = (
        _W_SCHEMA * schema_score
        + _W_CAL * cal_score
        + _W_IMPR * impr_score
        + _W_COV * coverage_score
        + _W_ADAPT * adapt_score
        + _W_NO_FILLER * filler_score
    )

    feedback = "\n".join([
        f"案例: {example.name} ({example.band})",
        schema_note, cal_note, impr_note, coverage_note,
        adapt_note, filler_note,
        f"score: schema={schema_score:.2f} cal={cal_score:.2f}"
        f" impr={impr_score:.2f} cov={coverage_score:.2f}"
        f" adapt={adapt_score:.2f} no_fill={filler_score:.2f}"
        f" → total={total:.2f}",
    ])

    return MetricBreakdown(
        total=total,
        breakdown={
            "schema": schema_score,
            "score_calibration": cal_score,
            "improvements_specific": impr_score,
            "coverage_diversity": coverage_score,
            "adaptive_difficulty": adapt_score,
            "no_filler_praise": filler_score,
        },
        feedback=feedback,
    )


def _guess_category(question: str) -> str:
    """Cheap guesser used by coverage_diversity to bucket prior questions."""
    if any(k in question for k in ("项目", "你做的", "你的简历")):
        return "project_deep_dive"
    if any(k in question for k in ("如果你是", "你怎么处理", "讲一个")):
        return "behavioral"
    if any(k in question for k in ("设计一个", "如何设计", "QPS", "百万")):
        return "system_design"
    if any(k in question for k in ("字节", "阿里", "腾讯", "你为什么选")):
        return "company_specific"
    return "technical"


def _zero(example: MockInterviewExample, reason: str) -> MetricBreakdown:
    return MetricBreakdown(
        total=0.0,
        breakdown={ax: 0.0 for ax in METRIC_AXES if ax != "total"},
        feedback=f"案例: {example.name} ({example.band})\n{reason}",
    )


def split_train_val(
    *,
    val_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[list[MockInterviewExample], list[MockInterviewExample]]:
    import random
    rng = random.Random(seed)
    by_band: dict[str, list[MockInterviewExample]] = {}
    for ex in sorted(EXAMPLES, key=lambda e: e.name):
        by_band.setdefault(ex.band, []).append(ex)
    train: list[MockInterviewExample] = []
    val: list[MockInterviewExample] = []
    for _band, exs in sorted(by_band.items()):
        n_val = max(1, round(len(exs) * val_fraction))
        shuffled = exs[:]
        rng.shuffle(shuffled)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


_ = re  # quiet unused-import lint (re is imported for future patterns)
