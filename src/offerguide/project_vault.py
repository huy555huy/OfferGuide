"""Truthful project fact vault.

The project vault is private grounding material for resume tailoring and
interview preparation. It intentionally favors mainstream direction, real
ownership, contribution boundaries, artifacts, and "do not claim" guardrails
over AI-looking fake metrics. Public market/context references can calibrate
wording, but they are never treated as evidence for the user's own results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    def chat(self, messages: list[dict[str, str]], **kw: Any) -> Any:
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
    """Append project vault context to a resume/profile prompt input.

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
