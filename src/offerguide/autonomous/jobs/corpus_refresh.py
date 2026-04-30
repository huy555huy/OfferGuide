"""Weekly 面经 corpus refresh.

For each company that has ≥ 1 active application or ≥ 1 ingested JD,
the agentic ``CorpusCollector`` searches the web for recent 面经,
filters via LLM, and ingests fresh ones into ``interview_experiences``.

Skipped gracefully when LLM or search backend isn't configured.
Per-company budget: capped to keep the weekly run bounded — at most
``MAX_COMPANIES_PER_RUN`` companies refreshed.
"""

from __future__ import annotations

import logging
from typing import Any

from ..scheduler import JobContext, JobSpec

log = logging.getLogger(__name__)

MAX_COMPANIES_PER_RUN = 8
"""Cap so a single weekly run doesn't blow the LLM budget. Companies
are picked in order of `last_application_activity` descending so the
ones the user is actively pursuing get refreshed first."""


def run(ctx: JobContext) -> dict[str, Any]:
    if ctx.llm is None:
        log.info("corpus_refresh: LLM not configured, skipping")
        return {"skipped": "no_llm"}
    if ctx.search is None:
        log.info("corpus_refresh: search backend not configured, skipping")
        return {"skipped": "no_search"}

    from ...agentic.corpus_collector import CorpusCollector

    companies = _select_companies(ctx)
    if not companies:
        return {"companies": 0, "skipped": "no_active_companies"}

    collector = CorpusCollector(store=ctx.store, llm=ctx.llm, search=ctx.search)
    total_inserted = 0
    total_evaluated = 0
    per_company: dict[str, int] = {}
    try:
        for company in companies:
            result = collector.collect(company)
            per_company[company] = result.inserted
            total_inserted += result.inserted
            total_evaluated += result.hits_evaluated
    finally:
        collector.close()

    return {
        "companies_refreshed": len(companies),
        "total_inserted": total_inserted,
        "total_evaluated": total_evaluated,
        "per_company": per_company,
    }


def _select_companies(ctx: JobContext) -> list[str]:
    """Pick companies the user is actively pursuing — newest activity first."""
    with ctx.store.connect() as conn:
        # Companies with active applications, ordered by latest event time
        rows = conn.execute(
            """
            SELECT j.company, MAX(ae.occurred_at) AS latest
            FROM applications a
            JOIN jobs j ON j.id = a.job_id
            LEFT JOIN application_events ae ON ae.application_id = a.id
            WHERE j.company IS NOT NULL AND j.company != ''
              AND a.status NOT IN ('rejected', 'offer', 'withdrawn')
            GROUP BY j.company
            ORDER BY latest DESC NULLS LAST
            LIMIT ?
            """,
            (MAX_COMPANIES_PER_RUN,),
        ).fetchall()
    return [r[0] for r in rows]


# Weekly Monday 08:00
CORPUS_REFRESH_JOB = JobSpec(
    name="corpus_refresh",
    func=run,
    trigger="cron",
    trigger_kwargs={"day_of_week": "mon", "hour": 8, "minute": 0},
)
