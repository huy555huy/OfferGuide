"""Company briefs — agent-maintained per-company knowledge.

The autonomous agent reads recent observations (interview_experiences
that arrived in the last N days, application_events showing how this
user fared at this company, skill_runs scoring jobs at this company)
and produces a compact JSON brief that **overrides** the hardcoded
COMPANY_APPLICATION_LIMITS heuristic when newer signal contradicts it.

Schema (table ``company_briefs``):
    company           PK
    brief_json        TEXT — see ``CompanyBrief`` Pydantic schema
    last_updated_at   REAL — julianday
    update_count      INTEGER

Use ``refresh_brief(store, llm, company)`` to regenerate. The function
is LLM-driven (DeepSeek synthesizes the brief from observations); falls
back to a no-op when LLM is None or no observations exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .llm import LLMClient, LLMError
from .memory import Store


class CompanyBrief(BaseModel):
    """Schema validated against the LLM output. ``extra='forbid'`` so
    rogue keys can't sneak past the parser."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    """2-3 sentences: business focus + recent state of THIS company,
    grounded in observations the agent has."""

    current_app_limit: int | None = Field(default=None, ge=0, le=20)
    """Estimated application slots. None if no signal — caller falls
    back to COMPANY_APPLICATION_LIMITS hardcoded table."""

    interview_style: str
    """1-2 sentences from the recent 面经 corpus: how this company
    actually probes (technical depth / behavioral / project deep-dive
    mix). 'no recent 面经' if data missing."""

    recent_signals: list[str] = Field(default_factory=list, max_length=8)
    """Concrete observations: 'last 3 face_jings show heavy GRPO focus',
    'two applications stuck at viewed for 14+ days', etc."""

    hiring_trend: str
    """'expanding' / 'stable' / 'slowing' / 'unknown'. Inferred from
    job count + recency in the local DB, not external news."""

    confidence: float = Field(ge=0.0, le=1.0)
    """How sure the LLM is. < 0.4 = low data; 0.7+ = strong signal."""


@dataclass(frozen=True)
class BriefRow:
    """Read-side wrapper for one company_briefs row."""

    company: str
    brief: CompanyBrief
    last_updated_at: float
    update_count: int


# ── observations gathering ──────────────────────────────────────────


def gather_observations(store: Store, company: str) -> dict[str, Any]:
    """Collect the raw signals the LLM needs to produce a brief.

    Returns a dict with:
    - recent_面经 (most recent 5 from interview_experiences)
    - active_apps_count, terminal_apps_count
    - latest_app_events (last 8 events across all apps at this company)
    - jobs_count + by_source
    - score_match runs at this company (if any)
    """
    with store.connect() as conn:
        # Recent 面经
        face_jing = conn.execute(
            "SELECT raw_text, role_hint, source, created_at "
            "FROM interview_experiences WHERE company LIKE ? "
            "ORDER BY created_at DESC LIMIT 5",
            (f"%{company}%",),
        ).fetchall()

        # Application stats
        app_stats = conn.execute(
            "SELECT a.status, COUNT(*) FROM applications a "
            "JOIN jobs j ON j.id = a.job_id "
            "WHERE j.company LIKE ? GROUP BY a.status",
            (f"%{company}%",),
        ).fetchall()

        # Latest events
        events = conn.execute(
            "SELECT ae.kind, ae.occurred_at, ae.source, ae.payload_json "
            "FROM application_events ae "
            "JOIN applications a ON a.id = ae.application_id "
            "JOIN jobs j ON j.id = a.job_id "
            "WHERE j.company LIKE ? "
            "ORDER BY ae.occurred_at DESC LIMIT 8",
            (f"%{company}%",),
        ).fetchall()

        # Jobs count + by source
        jobs_by_source = conn.execute(
            "SELECT source, COUNT(*) FROM jobs WHERE company LIKE ? GROUP BY source",
            (f"%{company}%",),
        ).fetchall()

    return {
        "company": company,
        "recent_面经": [
            {
                "raw_text": r[0][:1500],  # cap to keep prompt sized
                "role_hint": r[1],
                "source": r[2],
            }
            for r in face_jing
        ],
        "applications_by_status": dict(app_stats),
        "recent_events": [
            {"kind": e[0], "source": e[2]}
            for e in events
        ],
        "jobs_count": sum(c for _, c in jobs_by_source),
        "jobs_by_source": dict(jobs_by_source),
    }


# ── LLM-driven refresh ─────────────────────────────────────────────


_REFRESH_PROMPT = """你是公司情报分析师。给定一家公司的近期观察数据（面经 / 应用状态 /
事件 / 已知 JD 数量），生成一份紧凑的 JSON brief，以**事实为锚**：

{
  "summary": <str, 2-3 句, 公司业务 + 近期状态>,
  "current_app_limit": <int 0-20 | null, 投递限额 (没信号填 null)>,
  "interview_style": <str, 1-2 句, 从近期面经看出的考察风格>,
  "recent_signals": <list[str], 0-8 条, 具体观察>,
  "hiring_trend": "expanding" | "stable" | "slowing" | "unknown",
  "confidence": <float 0-1>
}

规则：
1. **不要编造**——观察数据没说的事不要写
2. **summary** 必须基于观察，不能复述 wiki
3. **current_app_limit** 默认填 null；只有近期面经/事件**明确提到**才填具体数
4. **interview_style** 必须引用观察到的面经特点；没面经填 "暂无面经数据"
5. **recent_signals** 每条都是事实片段（"最近 3 篇面经都问了 GRPO" / "2 个应用 viewed 后沉默 14 天"）
6. **confidence**: 观察 < 5 条 → ≤ 0.4；观察 5-15 条 → 0.5-0.7；> 15 条 → 0.7-0.9
7. 严格 JSON，不要 markdown 代码块"""


def refresh_brief(
    store: Store,
    llm: LLMClient,
    company: str,
    *,
    model: str | None = None,
) -> BriefRow | None:
    """Regenerate the brief for ``company`` using LLM + recent observations.

    Returns the new BriefRow on success, None when:
    - No observations exist (empty company)
    - LLM call fails
    - LLM returns invalid JSON
    """
    obs = gather_observations(store, company)

    # Bail when there's nothing to synthesize from
    if (
        not obs["recent_面经"]
        and not obs["recent_events"]
        and obs["jobs_count"] == 0
    ):
        return None

    user_msg = (
        f"【公司】{company}\n"
        f"【已知 JD 数】{obs['jobs_count']}（按来源 {obs['jobs_by_source']}）\n"
        f"【应用状态分布】{obs['applications_by_status']}\n"
        f"【最近事件】{obs['recent_events']}\n"
        f"【最近面经 (最多 5 篇)】\n"
        + "\n---\n".join(
            f"{f['source']}/{f['role_hint'] or '?'}: {f['raw_text']}"
            for f in obs["recent_面经"]
        )
    )

    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": _REFRESH_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=0.2,
            json_mode=True,
        )
    except LLMError:
        return None

    try:
        parsed = json.loads(resp.content)
        brief = CompanyBrief.model_validate(parsed)
    except (json.JSONDecodeError, ValueError):
        return None

    return _upsert(store, company, brief)


def _upsert(store: Store, company: str, brief: CompanyBrief) -> BriefRow:
    """Insert or update the company_briefs row, return the persisted result."""
    blob = brief.model_dump_json()
    with store.connect() as conn:
        existing = conn.execute(
            "SELECT update_count FROM company_briefs WHERE company = ?",
            (company,),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE company_briefs "
                "SET brief_json = ?, last_updated_at = julianday('now'), "
                "    update_count = update_count + 1 "
                "WHERE company = ?",
                (blob, company),
            )
        else:
            conn.execute(
                "INSERT INTO company_briefs(company, brief_json) VALUES (?,?)",
                (company, blob),
            )
        row = conn.execute(
            "SELECT brief_json, last_updated_at, update_count "
            "FROM company_briefs WHERE company = ?",
            (company,),
        ).fetchone()
    return BriefRow(
        company=company,
        brief=CompanyBrief.model_validate(json.loads(row[0])),
        last_updated_at=float(row[1]),
        update_count=int(row[2]),
    )


# ── reads ──────────────────────────────────────────────────────────


def get_brief(store: Store, company: str) -> BriefRow | None:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT brief_json, last_updated_at, update_count "
            "FROM company_briefs WHERE company = ?",
            (company,),
        ).fetchone()
    if not row:
        return None
    try:
        brief = CompanyBrief.model_validate(json.loads(row[0]))
    except (json.JSONDecodeError, ValueError):
        return None
    return BriefRow(
        company=company,
        brief=brief,
        last_updated_at=float(row[1]),
        update_count=int(row[2]),
    )


def list_briefs(store: Store, *, limit: int = 50) -> list[BriefRow]:
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT company, brief_json, last_updated_at, update_count "
            "FROM company_briefs ORDER BY last_updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out: list[BriefRow] = []
    for company, blob, ts, n in rows:
        try:
            brief = CompanyBrief.model_validate(json.loads(blob))
        except (json.JSONDecodeError, ValueError):
            continue
        out.append(BriefRow(
            company=company, brief=brief,
            last_updated_at=float(ts),
            update_count=int(n),
        ))
    return out


@dataclass(frozen=True)
class AppLimitAnswer:
    """W15.17 (correct fix #2) — "what's company X's app limit" with
    explicit source attribution so callers can render honest UI.

    Replaces the lossy ``(int, str)`` tuple from effective_app_limit().
    The point: when source is 'community_estimate' or 'unknown', the
    caller should **invite the agent to research** rather than print a
    confident number — that's the W15.17 哲学修正 #2.
    """

    limit: int
    """Best-guess application slot count. Always usable as a default;
    accuracy depends on ``source``."""

    source: str
    """One of:
    - 'brief_high_confidence' — agent's recent observations, ≥0.7 conf
    - 'brief_low_confidence'  — brief exists but agent isn't sure
    - 'community_estimate'    — fallback hardcoded table (网传, 非官方)
    - 'default_fallback'      — unknown company, used 3 as guess
    """

    confidence: float
    """0.0-1.0. Render as opacity / "我有 X% 把握" badge in UI."""

    is_research_recommended: bool
    """True → caller should offer a "让 agent 现搜 4 条 2026 来源刷新"
    button. Set on community_estimate / default_fallback / low-confidence
    brief — i.e. anything not based on agent's actual observations."""

    notes: str
    """1-sentence explanation for the user (NOT for logs). Goes into the
    UI tooltip / badge text."""


def effective_app_limit(
    store: Store, company: str, *, hardcoded_default: int = 3
) -> tuple[int, str]:
    """**Deprecated** since W15.17 — use :func:`app_limit_with_attribution`.

    Kept for backward compat: returns the same answer as before (brief value
    only when confidence ≥ 0.6, else falls back to hardcoded). New code
    should call ``app_limit_with_attribution`` directly to get rich source
    attribution and ``is_research_recommended`` flags.
    """
    from .skills.compare_jobs.helpers import lookup_application_limit

    row = get_brief(store, company)
    if (
        row
        and row.brief.current_app_limit is not None
        and row.brief.confidence >= 0.6
    ):
        return row.brief.current_app_limit, "brief"
    return lookup_application_limit(company, default=hardcoded_default), "hardcoded"


def app_limit_with_attribution(
    store: Store, company: str, *, default: int = 3,
) -> AppLimitAnswer:
    """Resolve company's app limit with full source attribution (W15.17).

    Diagnosis report §正确修法 2: 一岗多投这种动态信息**永远不该 hardcoded**.
    Hardcoded table stays as last-resort fallback for graceful degradation
    when LLM is unavailable, but **callers should surface the uncertainty**
    via ``is_research_recommended`` and let the user trigger a refresh that
    runs Tavily + LLM synthesis (see ``refresh_brief``).
    """
    from .skills.compare_jobs.helpers import (
        COMPANY_APPLICATION_LIMITS,
        lookup_application_limit,
    )

    row = get_brief(store, company)
    if row and row.brief.current_app_limit is not None:
        if row.brief.confidence >= 0.7:
            return AppLimitAnswer(
                limit=row.brief.current_app_limit,
                source="brief_high_confidence",
                confidence=row.brief.confidence,
                is_research_recommended=False,
                notes=(
                    f"Agent 基于 {row.update_count} 次观察, 置信度 "
                    f"{row.brief.confidence:.0%}"
                ),
            )
        # Brief exists but low confidence — surface need for refresh
        return AppLimitAnswer(
            limit=row.brief.current_app_limit,
            source="brief_low_confidence",
            confidence=row.brief.confidence,
            is_research_recommended=True,
            notes=(
                f"Agent 数据不足 (置信度 {row.brief.confidence:.0%}). "
                "建议让 agent 重新搜一下"
            ),
        )

    # No brief — fall back to community-estimate hardcoded table
    in_table = (
        company in COMPANY_APPLICATION_LIMITS
        or any(k in company for k in COMPANY_APPLICATION_LIMITS)
    )
    if in_table:
        return AppLimitAnswer(
            limit=lookup_application_limit(company, default=default),
            source="community_estimate",
            confidence=0.3,
            is_research_recommended=True,
            notes=(
                "社区流传估计 (网传, 非官方文档). 政策每年变, 各 BG 不一. "
                "建议让 agent 现搜 4 条 2026 来源核实"
            ),
        )

    # Fully unknown
    return AppLimitAnswer(
        limit=default,
        source="default_fallback",
        confidence=0.1,
        is_research_recommended=True,
        notes=(
            f"未知公司, 默认假定 {default}. "
            "强烈建议先让 agent 调研这家的实际限额"
        ),
    )
