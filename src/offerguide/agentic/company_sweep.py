"""Meta-agent — orchestrates the agentic components in one company sweep.

A "company sweep" is the user's actual mental model: "I'm looking at
公司 X — go figure out the latest about them, fetch any new 面经,
review my pending applications, give me a heads-up."

This module composes:

- ``CorpusCollector`` (face_jing search + LLM filter + ingest)
- application status review (read existing application_events)
- A future hook for company-brief refresh (LLM synthesizes
  recent observations into a structured "what changed about
  company X this week" blob — TBD until we have user-real outcomes)

It deliberately does NOT invoke score_match / analyze_gaps /
prepare_interview / deep_project_prep — those are user-initiated
tools. The sweep is the *background* worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMClient
from ..memory import Store
from .corpus_collector import CollectionResult, CorpusCollector
from .search import SearchBackend

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompanySweepResult:
    """Top-level summary returned by ``sweep_company``."""

    company: str
    interview_corpus: CollectionResult | None
    """If ``do_corpus=True``, the result of ``CorpusCollector.collect``;
    None when skipped (no LLM, or user opted out)."""

    application_summary: dict[str, Any]
    """Counts of active/terminal applications at this company + their
    status distribution."""

    notes: list[str] = field(default_factory=list)


def sweep_company(
    company: str,
    *,
    store: Store,
    llm: LLMClient | None = None,
    search: SearchBackend | None = None,
    do_corpus: bool = True,
    role_hint: str | None = None,
) -> CompanySweepResult:
    """Run a sweep on one company.

    Skips the agentic pieces when LLM / search aren't configured —
    in that mode the sweep just summarizes what's already in the DB.
    """
    notes: list[str] = []

    # 1. Application summary (always available — pure DB read)
    app_summary = _application_summary(store, company)
    notes.append(
        f"applications: total={app_summary['total']}, active={app_summary['active']}, "
        f"by_status={app_summary['by_status']}"
    )

    # 2. Corpus collection (agentic — only if LLM + search configured)
    corpus_result: CollectionResult | None = None
    if do_corpus:
        if llm is None:
            notes.append(
                "corpus collection skipped: no LLMClient (set DEEPSEEK_API_KEY)"
            )
        elif search is None:
            notes.append(
                "corpus collection skipped: no SearchBackend (set OFFERGUIDE_SEARCH_BACKEND)"
            )
        else:
            collector = CorpusCollector(store=store, llm=llm, search=search)
            try:
                corpus_result = collector.collect(company, role_hint=role_hint)
                notes.append(
                    f"corpus: {corpus_result.inserted} new, "
                    f"{corpus_result.skipped_dup} dup, "
                    f"{corpus_result.skipped_low_quality} rejected"
                )
            finally:
                collector.close()

    return CompanySweepResult(
        company=company,
        interview_corpus=corpus_result,
        application_summary=app_summary,
        notes=notes,
    )


def _application_summary(store: Store, company: str) -> dict[str, Any]:
    """Count applications + by status at this company."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT a.status, COUNT(*) FROM applications a "
            "JOIN jobs j ON j.id = a.job_id "
            "WHERE j.company LIKE ? "
            "GROUP BY a.status",
            (f"%{company}%",),
        ).fetchall()
    by_status = dict(rows)
    total = sum(by_status.values())
    terminal = (
        by_status.get("rejected", 0)
        + by_status.get("offer", 0)
        + by_status.get("withdrawn", 0)
    )
    return {
        "total": total,
        "active": total - terminal,
        "terminal": terminal,
        "by_status": by_status,
    }
