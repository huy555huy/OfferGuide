"""Daily company-brief refresh.

For each company in the user's active set, the LLM reads recent
observations (interview_experiences from the past N days, application
events, JD count) and regenerates the per-company brief stored in
``company_briefs``.

The brief is what overrides ``COMPANY_APPLICATION_LIMITS`` when a
recently-observed signal contradicts the hardcoded heuristic — e.g.
agent reads "字节 校招 limit 改成 3 了" from a recent 面经 → brief
updates → next /compare run uses 3 not 2.
"""

from __future__ import annotations

import logging
from typing import Any

from ..scheduler import JobContext, JobSpec

log = logging.getLogger(__name__)

MAX_COMPANIES_PER_RUN = 12


def run(ctx: JobContext) -> dict[str, Any]:
    if ctx.llm is None:
        log.info("brief_update: LLM not configured, skipping")
        return {"skipped": "no_llm"}

    from ... import briefs

    companies = _select_companies(ctx)
    if not companies:
        return {"refreshed": 0, "skipped": "no_active_companies"}

    refreshed = 0
    no_op = 0
    per_company: dict[str, str] = {}
    for company in companies:
        result = briefs.refresh_brief(ctx.store, ctx.llm, company)
        if result is None:
            no_op += 1
            per_company[company] = "no_op"
        else:
            refreshed += 1
            per_company[company] = (
                f"updated (confidence={result.brief.confidence:.2f})"
            )

    return {
        "refreshed": refreshed,
        "no_op": no_op,
        "per_company": per_company,
    }


def _select_companies(ctx: JobContext) -> list[str]:
    """Companies with any local data — applications, jobs, or 面经."""
    with ctx.store.connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT company FROM (
              SELECT company FROM jobs WHERE company IS NOT NULL AND company != ''
              UNION
              SELECT company FROM interview_experiences
            )
            ORDER BY company
            LIMIT ?
            """,
            (MAX_COMPANIES_PER_RUN,),
        ).fetchall()
    return [r[0] for r in rows]


# Daily 23:00 Asia/Shanghai
BRIEF_UPDATE_JOB = JobSpec(
    name="brief_update",
    func=run,
    trigger="cron",
    trigger_kwargs={"hour": 23, "minute": 0},
)
