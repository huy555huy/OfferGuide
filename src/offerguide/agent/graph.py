"""LangGraph single-agent dispatch graph.

Topology:

    START ──(route)──┬──► score_node ──(after_score)──┬──► gaps_node ──(after_gaps)──┬──► prep_node ──► summarize ──► END
                     │                                 └──► (skip)                    └──► (skip)
                     ├──► gaps_node ─────────────────────────────────────────────────────► summarize ──► END
                     ├──► prep_node ─────────────────────────────────────────────────────► summarize ──► END
                     └──► summarize ──► END

``requested_action`` from the inputs decides the path:
- ``score`` → score_node only
- ``gaps`` → gaps_node only
- ``score_and_gaps`` → score → gaps → summarize
- ``prepare_interview`` → prep_node only
- ``everything`` → score → gaps → prep → summarize

The prep node retrieves 面经 from ``interview_corpus`` when a Store is
available and the caller didn't pre-fill ``past_experiences``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langgraph.graph import END, START, StateGraph

from .. import interview_corpus
from ..memory import Store
from ..skills import SkillRuntime, SkillSpec
from .state import AgentState

DEFAULT_ACTION = "score_and_gaps"

# How many 面经 snippets to pull from the corpus when prep_node retrieves
_PREP_FETCH_LIMIT = 5
_PREP_RENDER_BUDGET_CHARS = 4000


class _MissingSkillError(RuntimeError):
    pass


def build_graph(
    *,
    skills: Iterable[SkillSpec],
    runtime: SkillRuntime | None = None,
    store: Store | None = None,
):
    """Build the agent graph.

    ``runtime`` is required for any node that actually calls an LLM;
    passing None is acceptable when the caller only wants to inspect the
    topology (e.g. tests that mock runtime entirely).

    ``store`` is optional — if present, the prep node retrieves past
    interview experiences from the corpus when ``past_experiences`` is
    empty in state. If absent, the prep node sends an empty string and
    relies on the SKILL's "no past experience" handling.
    """
    skill_index = {s.name: s for s in skills}

    def _need(name: str) -> SkillSpec:
        spec = skill_index.get(name)
        if spec is None:
            raise _MissingSkillError(f"SKILL '{name}' not loaded; cannot dispatch")
        return spec

    # ---- nodes ----------------------------------------------------------

    def score_node(state: AgentState) -> AgentState:
        if runtime is None:
            return {"error": "no SkillRuntime configured"}
        try:
            spec = _need("score_match")
        except _MissingSkillError as e:
            return {"error": str(e)}
        result = runtime.invoke(
            spec,
            {
                "job_text": state.get("job_text") or "",
                "user_profile": state.get("user_profile_text") or "",
            },
        )
        return {
            "score_result": result.parsed,
            "score_run_id": result.skill_run_id,
        }

    def gaps_node(state: AgentState) -> AgentState:
        if runtime is None:
            return {"error": "no SkillRuntime configured"}
        try:
            spec = _need("analyze_gaps")
        except _MissingSkillError as e:
            return {"error": str(e)}
        result = runtime.invoke(
            spec,
            {
                "job_text": state.get("job_text") or "",
                "user_profile": state.get("user_profile_text") or "",
            },
        )
        return {
            "gaps_result": result.parsed,
            "gaps_run_id": result.skill_run_id,
        }

    def cover_letter_node(state: AgentState) -> AgentState:
        if runtime is None:
            return {"error": "no SkillRuntime configured"}
        try:
            spec = _need("write_cover_letter")
        except _MissingSkillError as e:
            return {"error": str(e)}
        company = (state.get("company") or "").strip()
        if not company:
            return {
                "error": (
                    "求职信生成需要 `company` 字段（从 web UI 表单或调用方传入），"
                    " 但当前 state 里为空。"
                )
            }
        result = runtime.invoke(
            spec,
            {
                "company": company,
                "job_text": state.get("job_text") or "",
                "user_profile": state.get("user_profile_text") or "",
            },
        )
        return {
            "cover_letter_result": result.parsed,
            "cover_letter_run_id": result.skill_run_id,
        }

    def deep_prep_node(state: AgentState) -> AgentState:
        if runtime is None:
            return {"error": "no SkillRuntime configured"}
        try:
            spec = _need("deep_project_prep")
        except _MissingSkillError as e:
            return {"error": str(e)}

        company = (state.get("company") or "").strip()
        if not company:
            return {
                "error": (
                    "深度项目备战需要 `company` 字段（从 web UI 表单或调用方传入），"
                    " 但当前 state 里为空。"
                )
            }

        result = runtime.invoke(
            spec,
            {
                "company": company,
                "job_text": state.get("job_text") or "",
                "user_profile": state.get("user_profile_text") or "",
            },
        )
        return {
            "deep_prep_result": result.parsed,
            "deep_prep_run_id": result.skill_run_id,
        }

    def prep_node(state: AgentState) -> AgentState:
        if runtime is None:
            return {"error": "no SkillRuntime configured"}
        try:
            spec = _need("prepare_interview")
        except _MissingSkillError as e:
            return {"error": str(e)}

        company = (state.get("company") or "").strip()
        if not company:
            return {
                "error": (
                    "面试备战需要 `company` 字段（从 web UI 表单或调用方传入），"
                    " 但当前 state 里为空。"
                )
            }

        # Resolve past_experiences: caller-provided > corpus retrieval > empty
        provided = (state.get("past_experiences") or "").strip()
        used_count = 0
        if provided:
            past_experiences = provided
        elif store is not None:
            experiences = interview_corpus.fetch_for_company(
                store, company, limit=_PREP_FETCH_LIMIT
            )
            past_experiences = interview_corpus.render_snippets(
                experiences, max_chars=_PREP_RENDER_BUDGET_CHARS
            )
            used_count = len(experiences)
        else:
            past_experiences = ""

        result = runtime.invoke(
            spec,
            {
                "company": company,
                "job_text": state.get("job_text") or "",
                "user_profile": state.get("user_profile_text") or "",
                "past_experiences": past_experiences,
            },
        )
        return {
            "prep_result": result.parsed,
            "prep_run_id": result.skill_run_id,
            "prep_used_experiences": used_count,
        }

    def summarize(state: AgentState) -> AgentState:
        msg = _format_summary(state)
        return {
            "final_response": msg,
            "messages": [{"role": "assistant", "content": msg}],
        }

    # ---- routing -------------------------------------------------------

    def start_router(state: AgentState) -> str:
        action = state.get("requested_action") or DEFAULT_ACTION
        if action in ("score", "score_and_gaps", "everything"):
            return "score_node"
        if action == "gaps":
            return "gaps_node"
        if action == "prepare_interview":
            return "prep_node"
        if action == "deep_prep":
            return "deep_prep_node"
        if action == "cover_letter":
            return "cover_letter_node"
        return "summarize"

    def after_score_router(state: AgentState) -> str:
        if state.get("error"):
            return "summarize"
        action = state.get("requested_action") or DEFAULT_ACTION
        if action in ("score_and_gaps", "everything"):
            return "gaps_node"
        return "summarize"

    def after_gaps_router(state: AgentState) -> str:
        if state.get("error"):
            return "summarize"
        action = state.get("requested_action") or DEFAULT_ACTION
        return "prep_node" if action == "everything" else "summarize"

    def after_prep_router(state: AgentState) -> str:
        if state.get("error"):
            return "summarize"
        action = state.get("requested_action") or DEFAULT_ACTION
        return "deep_prep_node" if action == "everything" else "summarize"

    def after_deep_prep_router(state: AgentState) -> str:
        if state.get("error"):
            return "summarize"
        action = state.get("requested_action") or DEFAULT_ACTION
        return "cover_letter_node" if action == "everything" else "summarize"

    g: StateGraph = StateGraph(AgentState)
    g.add_node("score_node", score_node)
    g.add_node("gaps_node", gaps_node)
    g.add_node("prep_node", prep_node)
    g.add_node("deep_prep_node", deep_prep_node)
    g.add_node("cover_letter_node", cover_letter_node)
    g.add_node("summarize", summarize)
    g.add_conditional_edges(
        START,
        start_router,
        {
            "score_node": "score_node",
            "gaps_node": "gaps_node",
            "prep_node": "prep_node",
            "deep_prep_node": "deep_prep_node",
            "cover_letter_node": "cover_letter_node",
            "summarize": "summarize",
        },
    )
    g.add_conditional_edges(
        "score_node",
        after_score_router,
        {"gaps_node": "gaps_node", "summarize": "summarize"},
    )
    g.add_conditional_edges(
        "gaps_node",
        after_gaps_router,
        {"prep_node": "prep_node", "summarize": "summarize"},
    )
    g.add_conditional_edges(
        "prep_node",
        after_prep_router,
        {"deep_prep_node": "deep_prep_node", "summarize": "summarize"},
    )
    g.add_conditional_edges(
        "deep_prep_node",
        after_deep_prep_router,
        {"cover_letter_node": "cover_letter_node", "summarize": "summarize"},
    )
    g.add_edge("cover_letter_node", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


# -------------------------- response composition --------------------------


def _format_summary(state: AgentState) -> str:
    """Render a Chinese summary from whichever skill results landed."""
    parts: list[str] = []
    if state.get("error"):
        parts.append(f"⚠️ 出错：{state['error']}")
        return "\n".join(parts)

    score = state.get("score_result")
    if score:
        prob = _safe_float(score.get("probability"))
        dims = score.get("dimensions") or {}
        deal = score.get("deal_breakers") or []
        parts.append("## 匹配评分")
        parts.append(
            f"- 校准概率: **{prob:.2f}**"
            if prob is not None
            else "- 校准概率: (LLM 未返回)"
        )
        if dims:
            parts.append(
                "- 维度: "
                + " / ".join(
                    f"{k}={_safe_float(v):.2f}" if _safe_float(v) is not None else f"{k}=?"
                    for k, v in dims.items()
                )
            )
        if deal:
            parts.append("- ⚠️ Deal-breakers: " + "; ".join(map(str, deal)))
        if score.get("reasoning"):
            parts.append(f"- 理由: {score['reasoning']}")

    gaps = state.get("gaps_result")
    if gaps:
        parts.append("\n## 差距与建议")
        if gaps.get("summary"):
            parts.append(gaps["summary"])
        suggestions = gaps.get("suggestions") or []
        for i, s in enumerate(suggestions[:6], start=1):
            risk = s.get("ai_risk", "?")
            sec = s.get("section", "?")
            action = s.get("action", "?")
            parts.append(f"\n**[{i}] {sec} · {action} · ai_risk={risk}**")
            if s.get("current_text"):
                parts.append(f"  - 现有: {s['current_text']}")
            if s.get("proposed_addition"):
                parts.append(f"  - 建议: {s['proposed_addition']}")
            if s.get("reason"):
                parts.append(f"  - 原因: {s['reason']}")
        if gaps.get("do_not_add"):
            parts.append("\n**禁止添加** (避免编造):")
            for d in gaps["do_not_add"]:
                parts.append(f"  - {d}")
        if gaps.get("ai_detection_warnings"):
            parts.append("\n**AI 检测风险提示:**")
            for w in gaps["ai_detection_warnings"]:
                parts.append(f"  - {w}")

    prep = state.get("prep_result")
    if prep:
        parts.append("\n## 面试备战")
        used = state.get("prep_used_experiences", 0) or 0
        parts.append(f"_(基于 {used} 篇面经；公司: {state.get('company') or '?'})_")
        if prep.get("company_snapshot"):
            parts.append(f"\n**公司画像**: {prep['company_snapshot']}")

        questions = prep.get("expected_questions") or []
        if questions:
            sorted_q = sorted(
                questions,
                key=lambda q: float(q.get("likelihood", 0.0)),
                reverse=True,
            )
            parts.append("\n**预测题目** (按概率降序，最多 6):")
            for i, q in enumerate(sorted_q[:6], start=1):
                cat = q.get("category", "?")
                lik = _safe_float(q.get("likelihood"))
                lik_s = f"{lik:.2f}" if lik is not None else "?"
                parts.append(
                    f"\n[{i}] **[{cat}] {q.get('question', '?')}**"
                    f" — likelihood={lik_s}"
                )
                if q.get("rationale"):
                    parts.append(f"  - 依据: {q['rationale']}")

        focus = prep.get("prep_focus_areas") or []
        if focus:
            parts.append("\n**重点备战**:")
            for f in focus:
                parts.append(f"  - {f}")

        weak = prep.get("weak_spots") or []
        if weak:
            parts.append("\n**用户弱点**:")
            for w in weak:
                parts.append(f"  - {w}")

    deep = state.get("deep_prep_result")
    if deep:
        parts.append("\n## 项目深度备战 · deep_project_prep")
        if deep.get("company_style_summary"):
            parts.append(f"\n**公司风格**: {deep['company_style_summary']}")
        for proj in (deep.get("projects_analyzed") or [])[:3]:
            parts.append(f"\n### 项目 · {proj.get('project_name', '?')}")
            if proj.get("project_summary"):
                parts.append(proj["project_summary"])
            for q in (proj.get("probing_questions") or [])[:5]:
                lik = _safe_float(q.get("likelihood"))
                lik_s = f"{lik:.2f}" if lik is not None else "?"
                parts.append(
                    f"\n[{q.get('type', '?')}] **{q.get('question', '?')}**"
                    f" — likelihood={lik_s}"
                )
                if q.get("answer_outline"):
                    for a in q["answer_outline"][:4]:
                        parts.append(f"  · {a}")
            for w in (proj.get("weak_points") or [])[:2]:
                parts.append(
                    f"\n  ⚠ 弱点: {w.get('weakness', '?')}"
                    f" → {w.get('mitigation', '?')}"
                )
        if deep.get("behavioral_questions_tailored"):
            parts.append("\n### 行为题（结合用户经历）")
            for q in deep["behavioral_questions_tailored"][:3]:
                parts.append(f"  - {q.get('question', '?')}")

    cover = state.get("cover_letter_result")
    if cover:
        parts.append("\n## 求职信 · write_cover_letter")
        if cover.get("opening_hook"):
            parts.append(f"\n{cover['opening_hook']}")
        for para in (cover.get("narrative_body") or [])[:3]:
            parts.append(f"\n{para}")
        if cover.get("closing_call_to_action"):
            parts.append(f"\n{cover['closing_call_to_action']}")
        parts.append(
            f"\n_(personalization {cover.get('personalization_score', '?')} · "
            f"{cover.get('overall_word_count', '?')} words · "
            f"{cover.get('suggested_tone', '?')})_"
        )

    if not parts:
        return "(没有可显示的结果——请检查 `requested_action` 与输入。)"
    return "\n".join(parts)


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
