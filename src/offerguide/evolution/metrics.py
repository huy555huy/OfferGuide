"""Metric functions for GEPA optimization.

A metric is `(example, prediction, trace=None) -> float in [0, 1]`. GEPA uses
the float to rank candidate prompt mutations and the optional `feedback` field
to drive reflective evolution. We wrap our metric to return a `dspy.Prediction`
when called inside the GEPA loop so we can return both — but the same metric
function works in plain test mode where you just want the score.

Score components for `score_match` (weights chosen so a model that hits all
three at full strength gets 1.0; a model that nails probability but misses
reasoning anchors gets ~0.5):

- 0.5 × probability_in_band — does `prediction.probability` fall inside the
  golden's `expected_probability_range`? Linear penalty outside.
- 0.3 × reasoning_recall      — fraction of `must_mention` substrings that
  appear in `prediction.reasoning`. Catches "got the score right by accident
  but didn't identify the actual driver."
- 0.2 × anti_false_positive   — penalty for any `must_not_mention` substring
  appearing — the model claimed something we know is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .golden_trainset import GoldenExample

# Weights chosen by inspection. Sum to 1.0.
_W_PROB = 0.5
_W_RECALL = 0.3
_W_ANTI = 0.2


@dataclass(frozen=True)
class MetricBreakdown:
    """Per-component scores — surfaced to GEPA's reflection LM as feedback."""

    total: float
    prob_score: float
    recall_score: float
    anti_score: float
    feedback: str
    """Human-readable explanation. GEPA's reflection LM consumes this verbatim
    when proposing prompt mutations, so write it like you would write it to a
    teammate: 'too high; reasoning didn't mention C/C++ gap; ...'."""


def parse_score_match_output(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Coerce a SKILL output into a dict, accepting either pre-parsed JSON or a string.

    Returns ``{}`` for unparseable input — the caller's metric handles this
    by giving a low score, which is the right outcome (a parse failure on the
    output IS a failure of the prompt).
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def score_match_metric(
    example: GoldenExample,
    prediction: dict[str, Any] | str,
) -> MetricBreakdown:
    """Score a single (golden example, model prediction) pair.

    Returns a breakdown so callers can surface per-axis feedback to GEPA's
    reflection LM. ``prediction`` accepts either the parsed dict or the raw
    JSON string the model emitted.
    """
    parsed = parse_score_match_output(prediction)

    if not parsed:
        return MetricBreakdown(
            total=0.0,
            prob_score=0.0,
            recall_score=0.0,
            anti_score=0.0,
            feedback="OUTPUT_PARSE_FAILURE: model did not emit valid JSON for score_match.",
        )

    prob_score, prob_note = _score_probability(parsed, example)
    recall_score, recall_note = _score_recall(parsed, example)
    anti_score, anti_note = _score_anti_false_positive(parsed, example)

    total = _W_PROB * prob_score + _W_RECALL * recall_score + _W_ANTI * anti_score

    feedback_parts = [f"案例: {example.name} ({example.band})"]
    feedback_parts.append(prob_note)
    if example.must_mention:
        feedback_parts.append(recall_note)
    if example.must_not_mention:
        feedback_parts.append(anti_note)
    feedback_parts.append(
        f"score: prob={prob_score:.2f} recall={recall_score:.2f} anti={anti_score:.2f} → total={total:.2f}"
    )

    return MetricBreakdown(
        total=total,
        prob_score=prob_score,
        recall_score=recall_score,
        anti_score=anti_score,
        feedback="\n".join(feedback_parts),
    )


def _score_probability(
    parsed: dict[str, Any], example: GoldenExample
) -> tuple[float, str]:
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
    # Linear penalty: 4× the distance, clamped to 0
    dist = min(abs(prob - low), abs(prob - high))
    score = max(0.0, 1.0 - 4.0 * dist)
    direction = "偏低" if prob < low else "偏高"
    return (
        score,
        f"probability {prob:.2f} {direction}于期望 [{low:.2f}, {high:.2f}]，距离 {dist:.2f} → 分 {score:.2f}",
    )


def _score_recall(parsed: dict[str, Any], example: GoldenExample) -> tuple[float, str]:
    if not example.must_mention:
        return 1.0, ""
    reasoning = str(parsed.get("reasoning") or "")
    # Also consider reasoning that's distributed across deal_breakers field
    deal_breakers = " ".join(str(x) for x in (parsed.get("deal_breakers") or []))
    haystack = reasoning + "\n" + deal_breakers

    hits = [k for k in example.must_mention if k in haystack]
    misses = [k for k in example.must_mention if k not in haystack]
    score = len(hits) / len(example.must_mention)
    if not misses:
        note = f"reasoning 命中所有关键点 {list(example.must_mention)} ✓"
    else:
        note = f"reasoning 未命中: {misses}（必须指出）"
    return score, note


def _score_anti_false_positive(
    parsed: dict[str, Any], example: GoldenExample
) -> tuple[float, str]:
    if not example.must_not_mention:
        return 1.0, ""
    reasoning = str(parsed.get("reasoning") or "")
    false_pos = [k for k in example.must_not_mention if k in reasoning]
    if not false_pos:
        return 1.0, "reasoning 没有触发反向词 ✓"
    # Each false-positive subtracts 0.5
    score = max(0.0, 1.0 - 0.5 * len(false_pos))
    return score, f"reasoning 触发反向词: {false_pos} → 扣分 {score:.2f}"


def aggregate(metrics: list[MetricBreakdown]) -> dict[str, float]:
    """Mean per-axis scores across a list of evaluations. Useful for before/after diff reporting."""
    if not metrics:
        return {"total": 0.0, "prob": 0.0, "recall": 0.0, "anti": 0.0, "n": 0}
    n = len(metrics)
    return {
        "total": sum(m.total for m in metrics) / n,
        "prob": sum(m.prob_score for m in metrics) / n,
        "recall": sum(m.recall_score for m in metrics) / n,
        "anti": sum(m.anti_score for m in metrics) / n,
        "n": n,
    }
