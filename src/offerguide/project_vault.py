"""Truthful project fact vault.

The project vault is private grounding material for resume tailoring and
interview preparation. It intentionally favors mainstream direction, real
ownership, contribution boundaries, artifacts, and "do not claim" guardrails
over AI-looking fake metrics. Public market/context references can calibrate
wording, but they are never treated as evidence for the user's own results.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from .memory import Store

ContributionType = Literal[
    "method_innovation",
    "engineering_improvement",
    "application_transfer",
    "process_improvement",
    "integration",
    "reproduction",
    "main_contribution",
]

CONTRIBUTION_LABELS: dict[str, str] = {
    "method_innovation": "方法创新",
    "engineering_improvement": "工程改进",
    "application_transfer": "应用迁移",
    "process_improvement": "流程改进",
    "integration": "组合实现",
    "reproduction": "复现理解",
    "main_contribution": "主要贡献",
}

RECOMMENDED_DIRECTIONS: tuple[str, ...] = (
    "LLM 应用",
    "AI Agent",
    "NLP",
    "推荐系统",
    "计算机视觉",
    "数据分析",
    "机器学习建模",
    "后端系统",
    "知识图谱",
    "科研复现",
    "课程项目",
)


class _SearchLike(Protocol):
    def search(self, query: str, *, max_results: int = 10) -> list[Any]:
        ...


class _LLMLike(Protocol):
    def chat(
        self,
        messages: Any,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class ProjectRecord:
    """One defensible project record."""

    id: int
    title: str
    mainstream_direction: str
    typical_problem: str | None
    project_task: str
    my_work: str
    method_route: str | None
    market_context: str | None
    reference_sources: str | None
    contribution_type: ContributionType
    contribution_detail: str | None
    key_difficulties: str | None
    resolution_process: str | None
    project_outputs: str | None
    evidence: str | None
    askable_points: str | None
    expression_boundary: str | None
    do_not_claim: str | None
    tags: list[str]
    confidence: float
    created_at: float
    updated_at: float

    @property
    def contribution_label(self) -> str:
        return CONTRIBUTION_LABELS.get(self.contribution_type, self.contribution_type)


@dataclass(frozen=True)
class MarketContextDraft:
    """Public expression research for a project direction."""

    market_context: str
    reference_sources: str
    warnings: list[str]


AgentNextAction = Literal[
    "ask_user",
    "research_context",
    "draft_record",
    "ready_to_save",
]


@dataclass(frozen=True)
class ProjectAgentAssessment:
    """Structured assessment returned by the main agent's project tool."""

    is_agent_project: bool
    agent_reason: str
    next_action: AgentNextAction
    agent_signals: list[str]
    non_agent_signals: list[str]
    missing_facts: list[str]
    risk_flags: list[str]
    next_questions: list[str]
    suggested_fields: dict[str, str]
    market_context: str = ""
    reference_sources: str = ""
    action_trace: list[str] | None = None


def insert(
    store: Store,
    *,
    title: str,
    mainstream_direction: str,
    project_task: str,
    my_work: str,
    typical_problem: str | None = None,
    method_route: str | None = None,
    market_context: str | None = None,
    reference_sources: str | None = None,
    contribution_type: str = "main_contribution",
    contribution_detail: str | None = None,
    key_difficulties: str | None = None,
    resolution_process: str | None = None,
    project_outputs: str | None = None,
    evidence: str | None = None,
    askable_points: str | None = None,
    expression_boundary: str | None = None,
    do_not_claim: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 0.5,
) -> ProjectRecord:
    """Add a project record with required truth-grounding fields."""
    for name, val in (
        ("title", title),
        ("mainstream_direction", mainstream_direction),
        ("project_task", project_task),
        ("my_work", my_work),
    ):
        if not val or not val.strip():
            raise ValueError(f"{name} is required")

    ctype = _clean_contribution_type(contribution_type)
    tags_json = json.dumps(_clean_tags(tags or []), ensure_ascii=False)

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO project_records("
            "title, mainstream_direction, typical_problem, project_task, "
            "my_work, method_route, market_context, reference_sources, "
            "contribution_type, contribution_detail, key_difficulties, "
            "resolution_process, project_outputs, evidence, askable_points, "
            "expression_boundary, do_not_claim, tags_json, confidence"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                title.strip(),
                mainstream_direction.strip(),
                _none_if_blank(typical_problem),
                project_task.strip(),
                my_work.strip(),
                _none_if_blank(method_route),
                _none_if_blank(market_context),
                _none_if_blank(reference_sources),
                ctype,
                _none_if_blank(contribution_detail),
                _none_if_blank(key_difficulties),
                _none_if_blank(resolution_process),
                _none_if_blank(project_outputs),
                _none_if_blank(evidence),
                _none_if_blank(askable_points),
                _none_if_blank(expression_boundary),
                _none_if_blank(do_not_claim),
                tags_json,
                _clamp_confidence(confidence),
            ),
        )
        record_id = int(cur.lastrowid or 0)
        row = conn.execute(_SELECT_SQL + " WHERE id = ?", (record_id,)).fetchone()
    return _row_to_project(row)


def draft_market_context(
    *,
    title: str = "",
    mainstream_direction: str,
    project_task: str,
    my_work: str = "",
    search: _SearchLike,
    llm: _LLMLike | None = None,
    max_sources: int = 6,
) -> MarketContextDraft:
    """Research how similar projects are publicly explained.

    The draft is wording context only. It may calibrate problem framing and
    vocabulary, but it is not evidence for the user's metrics, scale, ranking,
    deployment, or novelty.
    """
    direction = mainstream_direction.strip()
    task = project_task.strip()
    if not direction and not task:
        raise ValueError("mainstream_direction or project_task is required")

    hits: list[Any] = []
    seen_urls: set[str] = set()
    for query in _market_context_queries(
        title=title, mainstream_direction=direction, project_task=task,
    ):
        try:
            found = search.search(query, max_results=4)
        except Exception:
            found = []
        for hit in found:
            url = str(getattr(hit, "url", "") or "").strip()
            if not url or url in seen_urls:
                continue
            hits.append(hit)
            seen_urls.add(url)
            if len(hits) >= max_sources:
                break
        if len(hits) >= max_sources:
            break

    if not hits:
        return MarketContextDraft(
            market_context="",
            reference_sources="",
            warnings=["没有搜到足够的公开参考；建议手动补充 2-3 个同类项目/产品/论文来源。"],
        )

    source_brief = _format_hits_for_prompt(hits)
    if llm is not None:
        try:
            resp = llm.chat(
                messages=[
                    {"role": "system", "content": _MARKET_CONTEXT_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"项目名称: {title or '(未填写)'}\n"
                            f"项目方向: {direction or '(未填写)'}\n"
                            f"项目任务: {task or '(未填写)'}\n"
                            f"我的真实工作: {my_work.strip() or '(未填写)'}\n\n"
                            f"公开搜索结果:\n{source_brief}"
                        ),
                    },
                ],
                temperature=0.0,
                json_mode=True,
            )
            data = json.loads(resp.content)
            context = _stringify_lines(data.get("market_context")).strip()
            sources = _stringify_lines(data.get("reference_sources")).strip()
            warnings = _stringify_list(data.get("warnings"))
            if context:
                return MarketContextDraft(
                    market_context=context,
                    reference_sources=sources or _format_sources(hits),
                    warnings=warnings,
                )
        except Exception:
            pass

    return MarketContextDraft(
        market_context=(
            "可参考这些公开资料的表达方式来描述项目的问题、用户价值和系统形态；"
            "不要借用其中的性能指标、用户规模、排名或上线结果，除非你的项目证据材料单独支持。"
        ),
        reference_sources=_format_sources(hits),
        warnings=["LLM 总结不可用，已保留搜索来源供手动整理。"],
    )


def run_intake_agent(
    *,
    raw_project_note: str,
    search: _SearchLike | None = None,
    llm: _LLMLike | None = None,
) -> ProjectAgentAssessment:
    """Assess a free-form project note for the main Agent Chat loop.

    This helper is not the product agent. It is a bounded tool the main agent
    calls to judge whether a project is truly agent-like, identify missing
    facts and risk flags, optionally gather public expression context, then
    return auditable Project Vault fields.
    """
    note = raw_project_note.strip()
    if not note:
        raise ValueError("raw_project_note is required")

    trace = ["observe: received raw project note"]
    assessment = _heuristic_agent_assessment(note)
    trace.append("think: assessed agent signals, missing facts, and risk flags")

    if llm is not None:
        llm_assessment = _llm_intake_assessment(note, llm=llm)
        if llm_assessment is not None and _has_meaningful_llm_assessment(llm_assessment):
            assessment = llm_assessment
            trace.append("think: incorporated LLM structured assessment")

    if assessment.next_action == "research_context" and search is not None:
        trace.append("act: searched public references for comparable wording")
        direction = assessment.suggested_fields.get("mainstream_direction", "")
        task = assessment.suggested_fields.get("project_task", note[:180])
        draft = draft_market_context(
            title=assessment.suggested_fields.get("title", ""),
            mainstream_direction=direction,
            project_task=task,
            my_work=assessment.suggested_fields.get("my_work", ""),
            search=search,
            llm=llm,
        )
        assessment = replace(
            assessment,
            market_context=draft.market_context,
            reference_sources=draft.reference_sources,
            risk_flags=_dedupe(assessment.risk_flags + draft.warnings),
            suggested_fields={
                **assessment.suggested_fields,
                "market_context": draft.market_context,
                "reference_sources": draft.reference_sources,
            },
        )
        trace.append("observe: attached market context as expression reference")

    return replace(assessment, action_trace=trace)


def get(store: Store, record_id: int) -> ProjectRecord | None:
    with store.connect() as conn:
        row = conn.execute(_SELECT_SQL + " WHERE id = ?", (record_id,)).fetchone()
    return _row_to_project(row) if row else None


def list_all(
    store: Store,
    *,
    direction: str | None = None,
    tag: str | None = None,
    limit: int = 100,
) -> list[ProjectRecord]:
    where: list[str] = []
    params: list[object] = []
    if direction and direction.strip():
        where.append("mainstream_direction LIKE ?")
        params.append(f"%{direction.strip()}%")
    if tag and tag.strip():
        where.append("tags_json LIKE ?")
        params.append(f'%"{tag.strip()}"%')
    sql = _SELECT_SQL
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY confidence DESC, updated_at DESC LIMIT ?"
    params.append(limit)
    with store.connect() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_project(r) for r in rows]


def delete(store: Store, record_id: int) -> bool:
    with store.connect() as conn:
        cur = conn.execute("DELETE FROM project_records WHERE id = ?", (record_id,))
    return (cur.rowcount or 0) > 0


def render_for_skill(
    projects: list[ProjectRecord],
    *,
    max_chars: int = 5000,
) -> str:
    """Render project records as a compact, anti-fabrication prompt block."""
    if not projects:
        return ""
    out: list[str] = [
        "## 项目事实档案（Project Vault）",
        "使用原则：这些是用户保存的真实项目素材。优先按项目方向、真实工作、"
        "贡献边界和可追问点来使用；不要自动生成百分比提升、用户规模、排名、"
        "测试数量等档案里没有明确证据的数字。若项目含“外部表达参考”，"
        "只能学习同类项目如何解释问题和价值，不能把参考来源里的指标、规模、"
        "部署结果或创新结论写成用户自己的成果。",
    ]
    used = sum(len(x) for x in out)
    for p in projects:
        block = _render_one_for_skill(p)
        if used + len(block) > max_chars:
            out.append(f"\n…（还有 {len(projects) - (len(out) - 2)} 个项目因长度省略）")
            break
        out.append(block)
        used += len(block)
    return "\n".join(out).strip()


def render_store_for_skill(store: Store, *, max_chars: int = 5000) -> str:
    """Fetch and render the user's strongest project records."""
    return render_for_skill(list_all(store, limit=12), max_chars=max_chars)


def append_to_profile_text(
    store: Store,
    base_text: str,
    *,
    max_project_chars: int = 5000,
) -> str:
    """Append project vault context to a legacy generic prompt input.

    The appended block is explicitly labelled as project facts and guardrails,
    not polished resume text. This keeps existing SKILL input schemas stable
    while giving downstream prompts better grounding.
    """
    block = render_store_for_skill(store, max_chars=max_project_chars)
    if not block:
        return base_text
    return f"{base_text.rstrip()}\n\n---\n\n{block}\n"


def _render_one_for_skill(p: ProjectRecord) -> str:
    parts = [
        f"\n### {p.title}",
        f"- 项目方向: {p.mainstream_direction}",
        f"- 项目任务: {p.project_task}",
        f"- 我的真实工作: {p.my_work}",
        f"- 贡献类型: {p.contribution_label}",
    ]
    optional = (
        ("典型问题", p.typical_problem),
        ("方法路线", p.method_route),
        ("外部表达参考", p.market_context),
        ("参考来源", p.reference_sources),
        ("创新/改进/主要贡献", p.contribution_detail),
        ("关键难点", p.key_difficulties),
        ("解决过程", p.resolution_process),
        ("项目产出", p.project_outputs),
        ("证据材料", p.evidence),
        ("可追问点", p.askable_points),
        ("表达边界", p.expression_boundary),
        ("不要写/不要说", p.do_not_claim),
    )
    for label, value in optional:
        if value:
            parts.append(f"- {label}: {value}")
    if p.tags:
        parts.append(f"- 标签: {', '.join(p.tags)}")
    parts.append(f"- 可防守置信度: {p.confidence:.2f}")
    return "\n".join(parts)


def _clean_contribution_type(value: str) -> ContributionType:
    if value in CONTRIBUTION_LABELS:
        return value  # type: ignore[return-value]
    return "main_contribution"


def _clean_tags(tags: list[str]) -> list[str]:
    return [t.strip() for t in tags if t and t.strip()]


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _clamp_confidence(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _heuristic_agent_assessment(note: str) -> ProjectAgentAssessment:
    lower = note.lower()
    signals = _present_keywords(
        lower,
        {
            "目标驱动": ("goal", "目标", "任务", "plan", "规划"),
            "多步循环": ("loop", "迭代", "多轮", "反复", "step", "workflow"),
            "工具调用": ("tool", "工具", "search", "搜索", "crawl", "抓取", "python", "api"),
            "状态/记忆": ("state", "状态", "memory", "记忆", "session", "持久化", "恢复"),
            "观察反馈": ("observe", "观察", "反馈", "反思", "evidence", "证据"),
            "自主决策": ("decide", "选择", "判断", "路由", "下一步", "调度"),
        },
    )
    signals = _remove_negated_signals(note, signals)
    non_agent_signals = _present_keywords(
        lower,
        {
            "单次 LLM 调用": ("一次调用", "单轮", "prompt", "提示词", "包装 api"),
            "纯 CRUD/表单": ("增删改查", "crud", "表单", "管理系统"),
            "固定流水线": ("固定流程", "固定 pipeline", "脚本", "批处理"),
        },
    )
    risk_flags = _present_keywords(
        lower,
        {
            "出现未证实数字": ("提升", "%", "准确率", "用户", "排名", "qps", "并发"),
            "可能夸大创新": ("首创", "sota", "最优", "突破", "创新算法"),
            "归属边界不清": ("我们", "团队", "参与", "协助"),
        },
    )
    missing_facts = _missing_facts(note, signals)
    is_agent = len(signals) >= 3 and "工具调用" in signals and (
        "自主决策" in signals or "多步循环" in signals
    )
    next_action = _choose_next_action(
        missing_facts=missing_facts,
        risk_flags=risk_flags,
        has_market_context=False,
    )
    direction = "AI Agent" if is_agent else _infer_direction(note)
    fields = {
        "title": _infer_title(note),
        "mainstream_direction": direction,
        "project_task": _first_sentence(note, limit=180),
        "my_work": "",
        "method_route": _join_signal_sentence(signals),
        "do_not_claim": _join_do_not_claim(risk_flags),
    }
    return ProjectAgentAssessment(
        is_agent_project=is_agent,
        agent_reason=_agent_reason(is_agent, signals, non_agent_signals),
        next_action=next_action,
        agent_signals=signals,
        non_agent_signals=non_agent_signals,
        missing_facts=missing_facts,
        risk_flags=risk_flags,
        next_questions=_next_questions(missing_facts, risk_flags, is_agent),
        suggested_fields={k: v for k, v in fields.items() if v},
    )


def _llm_intake_assessment(
    note: str, *, llm: _LLMLike,
) -> ProjectAgentAssessment | None:
    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": _INTAKE_AGENT_PROMPT},
                {"role": "user", "content": note},
            ],
            temperature=0.0,
            json_mode=True,
        )
        data = json.loads(resp.content)
    except Exception:
        return None

    suggested = data.get("suggested_fields")
    if not isinstance(suggested, dict):
        suggested = {}
    next_action = data.get("next_action")
    if next_action not in ("ask_user", "research_context", "draft_record", "ready_to_save"):
        next_action = "ask_user"
    return ProjectAgentAssessment(
        is_agent_project=bool(data.get("is_agent_project")),
        agent_reason=str(data.get("agent_reason") or "").strip(),
        next_action=next_action,
        agent_signals=_stringify_list(data.get("agent_signals")),
        non_agent_signals=_stringify_list(data.get("non_agent_signals")),
        missing_facts=_stringify_list(data.get("missing_facts")),
        risk_flags=_stringify_list(data.get("risk_flags")),
        next_questions=_stringify_list(data.get("next_questions")),
        suggested_fields={
            str(k): str(v).strip()
            for k, v in suggested.items()
            if str(k).strip() and str(v).strip()
        },
    )


def _has_meaningful_llm_assessment(assessment: ProjectAgentAssessment) -> bool:
    return bool(
        assessment.agent_reason
        or assessment.agent_signals
        or assessment.non_agent_signals
        or assessment.missing_facts
        or assessment.risk_flags
        or assessment.next_questions
        or assessment.suggested_fields
    )


def _present_keywords(lower: str, groups: dict[str, tuple[str, ...]]) -> list[str]:
    found: list[str] = []
    for label, needles in groups.items():
        if any(n in lower for n in needles):
            found.append(label)
    return found


def _remove_negated_signals(note: str, signals: list[str]) -> list[str]:
    negated_patterns = {
        "工具调用": ("没有工具", "无工具", "不调用工具", "没有工具选择"),
        "状态/记忆": ("没有状态", "无状态", "没有记忆", "没有状态恢复"),
        "多步循环": ("没有多轮", "无多轮", "单次调用", "单轮"),
        "自主决策": ("没有决策", "无决策", "固定流程", "固定 pipeline"),
        "观察反馈": ("没有反馈", "无反馈", "没有观察"),
    }
    lower = note.lower()
    out: list[str] = []
    for signal in signals:
        if any(p in lower for p in negated_patterns.get(signal, ())):
            continue
        out.append(signal)
    return out


def _missing_facts(note: str, signals: list[str]) -> list[str]:
    lower = note.lower()
    missing: list[str] = []
    checks = (
        ("我的真实工作", ("我负责", "我实现", "负责", "实现", "搭建", "设计")),
        ("证据材料", ("github", "repo", "测试", "截图", "报告", "日志", "demo")),
        ("决策逻辑", ("判断", "选择", "路由", "决定", "策略", "下一步")),
        ("状态/记忆设计", ("state", "状态", "memory", "session", "持久化", "恢复")),
        ("失败案例或边界", ("失败", "边界", "限制", "不能", "风险", "问题")),
    )
    for label, needles in checks:
        if not any(n in lower for n in needles):
            missing.append(label)
    if "工具调用" in signals and not any(n in lower for n in ("为什么", "选择", "取舍")):
        missing.append("工具选择取舍")
    return _dedupe(missing)


def _choose_next_action(
    *, missing_facts: list[str], risk_flags: list[str], has_market_context: bool,
) -> AgentNextAction:
    if missing_facts or risk_flags:
        return "ask_user"
    if not has_market_context:
        return "research_context"
    return "draft_record"


def _next_questions(
    missing_facts: list[str], risk_flags: list[str], is_agent: bool,
) -> list[str]:
    questions: list[str] = []
    for fact in missing_facts[:5]:
        if fact == "我的真实工作":
            questions.append("这个项目里哪些模块是你亲自设计或实现的？哪些只是团队/框架已有能力？")
        elif fact == "证据材料":
            questions.append("有没有 repo、测试输出、截图、报告或运行日志可以支撑这些描述？")
        elif fact == "决策逻辑":
            questions.append("系统每一轮是怎么决定下一步动作的：固定规则、LLM 判断，还是两者结合？")
        elif fact == "状态/记忆设计":
            questions.append("agent 的状态保存了什么？中断后能否恢复，后续决策会用哪些历史信息？")
        elif fact == "失败案例或边界":
            questions.append("这个系统在哪些情况下会失败或效果不好？你做了哪些限制来避免乱跑？")
        elif fact == "工具选择取舍":
            questions.append("为什么选择这些工具/API？有没有尝试过替代方案，最后如何取舍？")
    if risk_flags:
        questions.append("这些性能、准确率、用户数或排名有没有证据？如果没有，要不要放入“不要写/不要说”？")
    if not is_agent:
        questions.append("这个项目是否存在自主规划、多步循环、工具调用、状态更新？如果没有，就不要硬包装成 Agent。")
    return questions[:6]


def _agent_reason(
    is_agent: bool, signals: list[str], non_agent_signals: list[str],
) -> str:
    if is_agent:
        return "它具备目标驱动、多步执行、工具调用或状态反馈等 agent 信号，不只是单次 LLM 调用。"
    if non_agent_signals:
        return "目前更像普通应用/固定流程：" + "、".join(non_agent_signals) + "；还缺少自主决策和状态反馈。"
    return "目前材料不足，不能仅凭使用 LLM/API 就判断为 agent；需要补充循环、工具、状态和决策机制。"


def _infer_direction(note: str) -> str:
    lower = note.lower()
    if any(x in lower for x in ("llm", "大模型", "agent", "智能体")):
        return "LLM 应用"
    if any(x in lower for x in ("后端", "接口", "数据库")):
        return "后端系统"
    if any(x in lower for x in ("论文", "复现", "实验")):
        return "科研复现"
    return "课程项目"


def _infer_title(note: str) -> str:
    first = _first_sentence(note, limit=40)
    for sep in ("：", ":", "｜", "|", "-"):
        if sep in first:
            left = first.split(sep, 1)[0].strip()
            if 2 <= len(left) <= 40:
                return left
    return ""


def _first_sentence(text: str, *, limit: int) -> str:
    cleaned = " ".join(text.strip().split())
    for sep in ("。", "\n", "；", ";"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0]
            break
    return cleaned[:limit]


def _join_signal_sentence(signals: list[str]) -> str:
    if not signals:
        return ""
    return "可围绕这些 agent/工程信号继续核实：" + "、".join(signals)


def _join_do_not_claim(risk_flags: list[str]) -> str:
    if not risk_flags:
        return ""
    return "未提供证据前，不要写性能提升、准确率、用户规模、排名、SOTA 或不是自己完成的贡献。"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _market_context_queries(
    *, title: str, mainstream_direction: str, project_task: str,
) -> list[str]:
    direction = mainstream_direction.strip()
    task = " ".join(project_task.strip().split())[:80]
    title = title.strip()
    seed = " ".join(x for x in (direction, task) if x).strip() or title
    queries = [
        f"{seed} 项目 发布稿 解读",
        f"{seed} 技术方案 README",
        f"{seed} product launch case study",
    ]
    if direction:
        queries.append(f"{direction} similar project architecture README")
    return queries


def _format_hits_for_prompt(hits: list[Any]) -> str:
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        title = str(getattr(hit, "title", "") or "").strip()
        url = str(getattr(hit, "url", "") or "").strip()
        snippet = str(getattr(hit, "snippet", "") or "").strip()
        lines.append(f"{i}. {title}\nURL: {url}\n摘要: {snippet}")
    return "\n\n".join(lines)


def _format_sources(hits: list[Any]) -> str:
    lines: list[str] = []
    for hit in hits:
        title = str(getattr(hit, "title", "") or "").strip()
        url = str(getattr(hit, "url", "") or "").strip()
        if title and url:
            lines.append(f"- {title}: {url}")
    return "\n".join(lines)


def _stringify_lines(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(f"- {str(v).strip()}" for v in value if str(v).strip())
    return str(value)


def _stringify_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


_MARKET_CONTEXT_PROMPT = """你是 OfferGuide 的项目表达调研员。你的任务不是替用户编成果，
而是根据公开搜索结果，总结同类项目/产品/论文通常怎么解释自己。

返回严格 JSON:
{
  "market_context": [
    "3-6 条。只写同类项目常见的问题表述、用户/使用场景、系统能力、工作流、价值表达和边界。",
    "写法要像大厂发布稿/技术解读：先说项目解决什么问题，再说怎么做，不要堆术语。"
  ],
  "reference_sources": [
    "来源标题: URL"
  ],
  "warnings": [
    "哪些指标、规模、排名、上线结果或创新结论不能直接借用到用户项目里"
  ]
}

硬规则:
- 不要声称用户项目达到了公开来源里的性能、准确率、用户数、排名、部署规模。
- 不要把同类项目的创新写成用户自己的创新。
- 如果搜索结果只是泛泛介绍，也要诚实写成“可参考表达”，不要拔高。
- 只返回 JSON，不要 markdown 代码块。
"""


_INTAKE_AGENT_PROMPT = """你是 OfferGuide 主 Agent 的项目事实评估工具。
你的目标不是润色简历，而是把用户的自由项目描述校准成真实、可防守的项目档案草稿。

你必须独立判断这个项目是不是 AI Agent。不要因为出现 LLM/API/Prompt 就判定为 agent。
只有同时出现目标驱动、多步循环、工具调用、状态/记忆、观察反馈、自主决策中的多个信号，
才可以判断为 agent。否则要诚实说它更像 LLM 应用、后端系统、固定流水线或普通项目。

返回严格 JSON:
{
  "is_agent_project": true,
  "agent_reason": "为什么是/不是 agent，必须具体",
  "next_action": "ask_user | research_context | draft_record | ready_to_save",
  "agent_signals": ["目标驱动", "工具调用"],
  "non_agent_signals": ["单次 LLM 调用"],
  "missing_facts": ["我的真实工作", "证据材料"],
  "risk_flags": ["出现未证实数字"],
  "next_questions": ["下一轮最该问用户的问题"],
  "suggested_fields": {
    "title": "项目名，如材料不足可省略",
    "mainstream_direction": "AI Agent / LLM 应用 / 后端系统 / 科研复现等",
    "project_task": "项目实际任务，克制表述",
    "my_work": "用户真实工作，材料不足可省略",
    "method_route": "主流方法/系统路线，材料不足可省略",
    "do_not_claim": "不应写入简历的内容"
  }
}

硬规则:
- 不编造指标、准确率、性能提升、用户规模、排名、部署结果。
- 不把团队/开源框架能力写成用户个人贡献。
- 缺事实时 next_action 优先 ask_user，而不是强行 draft_record。
- 只返回 JSON，不要 markdown 代码块。
"""


_SELECT_SQL = (
    "SELECT id, title, mainstream_direction, typical_problem, project_task, "
    "my_work, method_route, market_context, reference_sources, "
    "contribution_type, contribution_detail, key_difficulties, resolution_process, project_outputs, evidence, "
    "askable_points, expression_boundary, do_not_claim, tags_json, confidence, "
    "created_at, updated_at FROM project_records"
)


def _row_to_project(row: tuple) -> ProjectRecord:
    try:
        tags = json.loads(row[18]) if row[18] else []
    except (json.JSONDecodeError, TypeError):
        tags = []
    contribution_type = _clean_contribution_type(row[9] or "main_contribution")
    return ProjectRecord(
        id=int(row[0]),
        title=row[1],
        mainstream_direction=row[2],
        typical_problem=row[3],
        project_task=row[4],
        my_work=row[5],
        method_route=row[6],
        market_context=row[7],
        reference_sources=row[8],
        contribution_type=contribution_type,
        contribution_detail=row[10],
        key_difficulties=row[11],
        resolution_process=row[12],
        project_outputs=row[13],
        evidence=row[14],
        askable_points=row[15],
        expression_boundary=row[16],
        do_not_claim=row[17],
        tags=list(tags),
        confidence=float(row[19]),
        created_at=float(row[20]),
        updated_at=float(row[21]),
    )
