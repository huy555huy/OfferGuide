"""Pydantic schema for ``compare_jobs`` SKILL output + company application
limit defaults.

The SKILL takes a *set* of jobs at one company plus the user's profile and
emits a ranked priority list of which to apply to first, given that **most
companies cap the number of simultaneous applications**.

Real company application limits (sourced from each company's published
recruitment policy + 2026 校招 documentation):

- **字节跳动 校招**: hard limit of **2 positions** per person, earliest
  submission gets priority, no modification once submitted
  (source: bytedance jobs/campus 2026 policy)
- **字节跳动 日常实习**: unlimited (no curation needed at this layer,
  but compare_jobs still ranks by fit)
- **阿里巴巴 控股集团**: ≤ 16 业务 × 2 意向 each (sequential routing
  per business unit, 不互斥 across BUs)
- **淘天集团** (under 阿里): **3 意向** per round, unlimited rounds
- **腾讯 / 百度 / 美团 / 京东 / 快手 / 小红书**: less restrictive but
  recommend curating top 3-5 by fit to avoid HR fatigue

Schema uses ``extra='forbid'`` so any GEPA-evolved prompt that emits
stray fields fails validation strongly.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ── company → default application limit ────────────────────────────
# Used by the /compare page when user doesn't override the limit. The
# adapter / metric also uses this to grade whether the SKILL recommended
# a sane number of "apply" actions.
COMPANY_APPLICATION_LIMITS: dict[str, int] = {
    "字节跳动":   2,   # 校招 hard limit
    "字节":       2,
    "ByteDance":  2,
    "阿里巴巴":   3,
    "阿里":       3,
    "淘天":       3,
    "Alibaba":    3,
    "腾讯":       5,   # multi-BU but recommend curation
    "Tencent":    5,
    "百度":       5,
    "Baidu":      5,
    "美团":       5,
    "Meituan":    5,
    "京东":       5,
    "JD":         5,
    "快手":       5,
    "Kuaishou":   5,
    "小红书":     3,
    "shopee":     3,
    "拼多多":     5,
}


def lookup_application_limit(company: str, default: int = 3) -> int:
    """Resolve a company name to its known application-limit cap.

    Tries exact match, then prefix match (so '阿里云' matches '阿里').
    Returns ``default`` if no match.
    """
    if not company:
        return default
    if company in COMPANY_APPLICATION_LIMITS:
        return COMPANY_APPLICATION_LIMITS[company]
    for known_name, lim in COMPANY_APPLICATION_LIMITS.items():
        if known_name in company:
            return lim
    return default


# ── Pydantic output schema ─────────────────────────────────────────


JobAction = Literal[
    "apply_first",          # apply immediately, top priority
    "apply_backup",          # apply if capacity allows (after apply_first)
    "apply_if_capacity",    # apply only if you still have slots
    "skip",                  # don't apply — match too low or too competitive
]


class ProfileAlignment(BaseModel):
    """Per-axis alignment of user profile to this specific JD."""

    model_config = ConfigDict(extra="forbid")

    tech: float = Field(ge=0.0, le=1.0)
    """Technical stack overlap (PyTorch, LangGraph, ...) — 0=不沾, 1=完美."""

    exp: float = Field(ge=0.0, le=1.0)
    """Project / experience relevance — 0=完全无关, 1=简历项目几乎是 JD 描述."""

    culture: float = Field(ge=0.0, le=1.0)
    """Working style + business focus alignment — soft signal."""


class JobComparison(BaseModel):
    """One job's ranked entry within a company comparison."""

    model_config = ConfigDict(extra="forbid")

    job_id: int
    """Foreign key to ``jobs.id`` — must round-trip exactly so the UI
    can link back."""

    title: str
    """JD title (echoed for UI rendering — saves a DB query)."""

    rank: int = Field(ge=1)
    """1 = top priority, 2 = backup, etc. Distinct integers across the
    ``rankings`` list — no ties (ties → arbitrary tiebreak by job_id)."""

    action: JobAction
    """What the user should *do* with this job. Must match the rank
    cohort: ranks 1..apply_limit map to apply_first, the next ~2 to
    apply_backup, deeper ones to apply_if_capacity or skip."""

    match_probability: float = Field(ge=0.0, le=1.0)
    """Calibrated reply-rate probability (same definition as
    score_match's `probability`)."""

    competitiveness_estimate: float = Field(ge=0.0, le=1.0)
    """How competitive *this specific position* is (against other
    applicants), independent of user fit. 0 = niche, 1 = very crowded.
    Heuristic signals: salary range, generic JD phrasing, SoTA tech name
    (more applicants chase 'AI agent' than 'Java backend')."""

    profile_alignment: ProfileAlignment

    distinguishing_factors: list[str] = Field(default_factory=list, max_length=4)
    """What makes THIS job stand out vs sibling jobs at the same company.
    E.g. ['唯一明确写 LangGraph', '团队是 Seed 不是 Doubao', '城市在上海']."""

    risk_factors: list[str] = Field(default_factory=list, max_length=4)
    """Things that could go wrong with this specific application.
    E.g. ['Boss 显示 5000+ 投递', 'JD 要求 5 年经验且写 hard']."""

    reasoning: str
    """2-3 sentence Chinese explanation of why this rank/action.
    Anchored in distinguishing_factors + alignment scores, no platitudes."""


class CompareJobsResult(BaseModel):
    """Top-level output of compare_jobs."""

    model_config = ConfigDict(extra="forbid")

    company: str
    """Echoed from input — must match canonically."""

    application_limit_estimate: int = Field(ge=1, le=20)
    """How many slots the company allows (or our recommendation cap).
    Defaults from COMPANY_APPLICATION_LIMITS unless user overrode."""

    application_limit_source: Literal["known", "inferred", "user_override"]
    """Where the limit came from. ``known`` = hardcoded policy,
    ``inferred`` = LLM guessed, ``user_override`` = user told us directly."""

    rankings: list[JobComparison] = Field(min_length=1, max_length=12)
    """All jobs in the input set, ranked. Must include every input job
    (skipped ones get action='skip' but are still listed)."""

    recommended_apply_count: int = Field(ge=0)
    """How many of the rankings to actually pursue. Usually ==
    application_limit_estimate but may be lower (if all options are weak)
    or constrained by 'skip' rankings exceeding the limit."""

    strategic_summary: str
    """2-3 sentence Chinese strategic note: 'A 是首投, B 等 A 一周
    无回复再投; C/D/E 这次跳过 — 总体评估 ...'"""

    def by_action(self, action: JobAction) -> list[JobComparison]:
        """All rankings with the given action — useful in UI rendering."""
        return [r for r in self.rankings if r.action == action]

    def top_picks(self) -> list[JobComparison]:
        """Convenience: the apply_first cohort, sorted by rank."""
        return sorted(self.by_action("apply_first"), key=lambda r: r.rank)

    def all_ids(self) -> set[int]:
        return {r.job_id for r in self.rankings}
