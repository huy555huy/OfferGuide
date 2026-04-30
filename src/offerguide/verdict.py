"""Verdict synthesizer — turn a multi-SKILL run into a one-line recommendation.

When the user runs the "everything" action, they get back six chunks
of analysis (score / gaps / prepare_interview / deep_project_prep /
cover_letter / brief). Reading all of that to decide *should I invest
time in this JD?* is exactly the friction this module removes.

Verdict logic is deterministic — no LLM call, just a small rule engine
over the structured outputs. The user can always override by reading the
full report; the verdict is a *summary* that frames everything else.

Categories:

- ``go``      投——综合分高、无 deal-breaker、brief 自信
- ``maybe``   可以一试——边缘分或证据不足
- ``hold``    暂缓——deal-breaker 或综合分明显偏低
- ``skip``    不投——硬条件不匹配或综合分极低

The verdict carries a short one-line explanation + up to 4 prioritized
next-step action items. Action items always include concrete pointers
(file/skill/page) so the user can act in one click.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VerdictKind = Literal["go", "maybe", "hold", "skip"]


@dataclass(frozen=True)
class VerdictAction:
    """One concrete next step to take after reading the report.

    ``action_url`` is optional — None means "no specific page", just
    follow the textual advice. ``priority``: 'high'|'medium'|'low'.
    """
    title: str
    detail: str
    priority: str
    action_url: str | None = None


@dataclass
class Verdict:
    """The headline summary card shown at the top of an everything-mode report."""
    kind: VerdictKind
    """One of go|maybe|hold|skip — drives the badge color."""

    label: str
    """Short Chinese label like '建议投' / '可以一试' / '暂缓' / '不投'."""

    summary: str
    """One-line rationale: ≤80 chars. Combines score + key gap signals."""

    overall_score: float
    """0-1, the synthesized priority score combining match prob,
    deal-breaker presence, brief confidence, etc. Used to sort multiple
    JDs in batch mode (future)."""

    actions: list[VerdictAction] = field(default_factory=list)
    """Top 4 prioritized next steps (high → low)."""

    pillars: dict[str, str] = field(default_factory=dict)
    """Quick-glance status for each ran SKILL: {'score': '72%',
    'gaps': '5 高风险', 'prep': '12 题', ...}. Surfaced as a row of
    pills under the verdict line."""


def synthesize(
    *,
    score: dict[str, Any] | None,
    gaps: dict[str, Any] | None,
    prep: dict[str, Any] | None,
    deep_prep: dict[str, Any] | None,
    cover_letter: dict[str, Any] | None,
    brief_confidence: float | None = None,
    brief_app_limit: int | None = None,
) -> Verdict:
    """Combine SKILL outputs into one Verdict.

    Any input may be ``None`` (skill wasn't run); we degrade gracefully.
    Returns a Verdict even with no inputs (kind='maybe', summary tells
    user to run something).
    """
    prob = _safe_float((score or {}).get("probability"), 0.5)
    deal_breakers = _ensure_list((score or {}).get("deal_breakers"))
    high_risk_gaps = _count_by(gaps, "suggestions", "ai_risk", "high")

    expected_qs = _ensure_list((prep or {}).get("expected_questions"))
    weak_spots = _ensure_list((deep_prep or {}).get("weak_spots_to_practice"))
    cl_ai_warnings = _ensure_list((cover_letter or {}).get("ai_risk_warnings"))

    pillars: dict[str, str] = {}
    if score is not None:
        pillars["score"] = f"{int(prob * 100)}%"
    if gaps is not None:
        n = len(_ensure_list((gaps or {}).get("suggestions")))
        pillars["gaps"] = f"{n} 项{f' · {high_risk_gaps} 高' if high_risk_gaps else ''}"
    if prep is not None:
        pillars["prep"] = f"{len(expected_qs)} 题"
    if deep_prep is not None:
        n_proj = len(_ensure_list((deep_prep or {}).get("projects_analyzed")))
        pillars["deep"] = f"{n_proj} 项目 · {len(weak_spots)} 弱点"
    if cover_letter is not None:
        wc = _safe_int((cover_letter or {}).get("overall_word_count"))
        risk_label = " ⚠" if cl_ai_warnings else ""
        pillars["cover"] = f"{wc} 字{risk_label}"
    if brief_confidence is not None:
        pillars["brief"] = f"信心 {int(brief_confidence * 100)}%"

    # ── decide kind ───────────────────────────────────────────
    has_dealbreaker = bool(deal_breakers)

    if has_dealbreaker and prob < 0.5:
        kind: VerdictKind = "skip"
    elif has_dealbreaker:
        kind = "hold"
    elif prob >= 0.7:
        kind = "go"
    elif prob >= 0.5:
        kind = "maybe"
    elif prob >= 0.3:
        kind = "hold"
    else:
        kind = "skip"

    # If brief confidence is super low, downgrade go→maybe
    # ("don't commit hard until we know more about this company")
    if kind == "go" and brief_confidence is not None and brief_confidence < 0.35:
        kind = "maybe"

    # If app_limit is 1-2 (super tight), upgrade scrutiny
    if brief_app_limit is not None and brief_app_limit <= 2 and kind == "maybe":
        kind = "hold"

    label_map = {
        "go": "建议投",
        "maybe": "可以一试",
        "hold": "暂缓",
        "skip": "不投",
    }
    label = label_map[kind]

    # ── build one-line summary ─────────────────────────────────
    parts: list[str] = []
    parts.append(f"匹配 {int(prob * 100)}%")
    if has_dealbreaker:
        parts.append(f"{len(deal_breakers)} 个硬伤")
    if high_risk_gaps:
        parts.append(f"{high_risk_gaps} 项高 AI 风险")
    if cl_ai_warnings:
        parts.append("cover letter 有 AI 痕迹")
    if brief_app_limit is not None and brief_app_limit <= 2:
        parts.append(f"该司限投 {brief_app_limit} 个")
    if brief_confidence is not None and brief_confidence < 0.35:
        parts.append("brief 证据不足")
    summary = " · ".join(parts)[:120]

    # ── build action list (priority high → low) ───────────────
    actions: list[VerdictAction] = []

    if has_dealbreaker:
        actions.append(
            VerdictAction(
                title="先评估 deal-breaker 是否真的硬",
                detail=f"score 报了 {len(deal_breakers)} 项；只要 1 项是硬条件就别投",
                priority="high",
            )
        )
    if high_risk_gaps:
        actions.append(
            VerdictAction(
                title=f"先改简历的 {high_risk_gaps} 处 AI 风险点",
                detail="49% 公司会因 AI 痕迹自动 dismiss——优先级第一",
                priority="high",
            )
        )
    if cl_ai_warnings:
        actions.append(
            VerdictAction(
                title="改 cover letter 的 AI 痕迹",
                detail="ai_risk_warnings 不为空——按提示改完再用",
                priority="high",
            )
        )
    if weak_spots:
        actions.append(
            VerdictAction(
                title=f"针对 {len(weak_spots)} 个项目弱点准备",
                detail="deep_project_prep 标的弱点是面试的高频追问点",
                priority="medium",
            )
        )
    if expected_qs and len(expected_qs) >= 5 and kind in ("go", "maybe"):
        # 5+ predicted questions → prep is meaningful
        actions.append(
            VerdictAction(
                title=f"刷 prepare_interview 的 {len(expected_qs)} 道题",
                detail="把高 likelihood 的题先想清楚答题骨架",
                priority="medium",
            )
        )
    if brief_confidence is not None and brief_confidence < 0.35:
        actions.append(
            VerdictAction(
                title="先 sweep 这家公司",
                detail=f"brief 信心 {int(brief_confidence*100)}% 偏低——/api/agent/sweep 跑一次",
                priority="medium",
            )
        )
    if kind == "skip":
        actions.append(
            VerdictAction(
                title="把 JD 在 inbox 标 dismiss",
                detail="不投也要记一笔——避免重复评估",
                priority="low",
                action_url="/inbox",
            )
        )
    if kind == "go":
        actions.insert(
            0,
            VerdictAction(
                title="排进今天/明天投递",
                detail="综合分高，时间投入有 ROI",
                priority="high",
            )
        )

    # If we got zero actions out of the rules above, the user has a
    # signal-free report. Add a guidance line so the verdict card always
    # carries at least one next-step.
    if not actions:
        if score is None:
            actions.append(
                VerdictAction(
                    title="先跑 score_match 拿到匹配概率",
                    detail="没有 score 没法判断这个 JD 值不值得花时间",
                    priority="medium",
                )
            )
        elif kind == "maybe":
            actions.append(
                VerdictAction(
                    title="再跑 analyze_gaps 看简历需不需要改",
                    detail="边缘分案——一份针对性强的简历能把 maybe 推到 go",
                    priority="medium",
                )
            )

    # Cap at 4 most important
    weight = {"high": 0, "medium": 1, "low": 2}
    actions.sort(key=lambda a: weight.get(a.priority, 3))
    actions = actions[:4]

    # ── overall_score: weighted blend for sort/comparison ─────
    overall = prob
    if has_dealbreaker:
        overall *= 0.5
    if high_risk_gaps:
        overall *= 0.85
    if brief_confidence is not None:
        overall *= max(0.7, brief_confidence)
    if brief_app_limit is not None and brief_app_limit <= 2:
        overall *= 0.85
    overall = max(0.0, min(1.0, overall))

    return Verdict(
        kind=kind,
        label=label,
        summary=summary,
        overall_score=overall,
        actions=actions,
        pillars=pillars,
    )


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _ensure_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def _count_by(d: dict | None, list_key: str, attr: str, value: str) -> int:
    """Count entries in d[list_key] whose entry[attr] == value."""
    if d is None:
        return 0
    items = _ensure_list(d.get(list_key))
    return sum(1 for it in items if isinstance(it, dict) and it.get(attr) == value)
