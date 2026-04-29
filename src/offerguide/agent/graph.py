"""LangGraph single-agent dispatch graph.

Topology (sequential, no parallelism — keeps state ownership obvious):

    START ──(route)──┬──► score_node ──(after_score)──┬──► gaps_node ──► summarize ──► END
                     │                                 └──► summarize ──► END
                     ├──► gaps_node ──────────────────────► summarize ──► END
                     └──► summarize ──► END

`requested_action` from the inputs decides the path. Each skill node calls
SkillRuntime → records to the `skill_runs` table → stashes parsed output and
the run_id into state. `summarize` composes a Chinese assistant message from
whichever results are populated.

Why no parallel fan-out for score+gaps even though both could run together:
LangGraph parallel branches need explicit reducers for any state field two
branches write to. For W4 we want the simplest correct shape; W6 can revisit
when GEPA's evaluator might want concurrent scoring of N candidate prompts.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..skills import SkillRuntime, SkillSpec
from .state import AgentState

DEFAULT_ACTION = "score_and_gaps"


class _MissingSkillError(RuntimeError):
    pass


def build_graph(
    *,
    skills: Iterable[SkillSpec],
    runtime: SkillRuntime | None = None,
):
    """Build the agent graph.

    `runtime` is required for any node that actually calls an LLM; passing
    None is acceptable when the caller only wants to inspect the topology
    (e.g. tests that mock runtime entirely).
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

    def summarize(state: AgentState) -> AgentState:
        msg = _format_summary(state)
        return {
            "final_response": msg,
            "messages": [{"role": "assistant", "content": msg}],
        }

    # ---- routing -------------------------------------------------------

    def start_router(state: AgentState) -> str:
        action = state.get("requested_action") or DEFAULT_ACTION
        if action == "score" or action == "score_and_gaps":
            return "score_node"
        if action == "gaps":
            return "gaps_node"
        return "summarize"

    def after_score_router(state: AgentState) -> str:
        action = state.get("requested_action") or DEFAULT_ACTION
        if state.get("error"):
            return "summarize"
        return "gaps_node" if action == "score_and_gaps" else "summarize"

    g: StateGraph = StateGraph(AgentState)
    g.add_node("score_node", score_node)
    g.add_node("gaps_node", gaps_node)
    g.add_node("summarize", summarize)
    g.add_conditional_edges(
        START,
        start_router,
        {"score_node": "score_node", "gaps_node": "gaps_node", "summarize": "summarize"},
    )
    g.add_conditional_edges(
        "score_node",
        after_score_router,
        {"gaps_node": "gaps_node", "summarize": "summarize"},
    )
    g.add_edge("gaps_node", "summarize")
    g.add_edge("summarize", END)
    return g.compile()


# -------------------------- response composition --------------------------


def _format_summary(state: AgentState) -> str:
    """Render a human-readable Chinese summary from whichever skill results landed.

    Kept as a pure function so test cases can exercise the rendering against
    canned dicts without spinning up the whole graph.
    """
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

    if not parts:
        return "(没有可显示的结果——请检查 `requested_action` 与输入。)"
    return "\n".join(parts)


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
