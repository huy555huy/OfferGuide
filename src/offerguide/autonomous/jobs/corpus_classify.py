"""Daily corpus quality-classification job.

Runs ``corpus_quality.classify_pending`` on every newly-ingested
corpus item that hasn't been classified yet. Classifier verdicts feed
``successful_profile`` synthesis — without this job, every ingested 面经
sits at the default ``quality_score=0.5`` and never makes it into
profile generation.

Skip semantics (graceful degradation):
- No LLM configured → still runs, but only the deterministic pre-filter
  catches blatant marketer posts; gray-zone items stay at the default
  score until the next run with LLM
- LLM call fails on an item → that item keeps default score and gets
  retried tomorrow

Per-run cap (``MAX_PER_RUN``) prevents runaway token spend if the
corpus suddenly grows.
"""

from __future__ import annotations

import logging
from typing import Any

from ...corpus_quality import classify_pending
from ..scheduler import JobContext, JobSpec

log = logging.getLogger(__name__)

MAX_PER_RUN = 50
"""Per-run cap on items the classifier processes. ~50 LLM calls is
under $1 per run with claude-haiku-4-5; if the daemon catches up over
several days, growth is bounded. Real-time classification on insert
(via corpus_collector) handles steady-state."""


def run(ctx: JobContext) -> dict[str, Any]:
    """Classify all pending corpus items, return counters."""
    counters = classify_pending(
        ctx.store, llm=ctx.llm, limit=MAX_PER_RUN,
    )

    if ctx.notifier and counters.get("processed", 0) > 0:
        try:
            ctx.notifier.notify(
                title=f"OfferGuide: 语料分类完成 {counters['processed']}",
                body=(
                    f"高质量 {counters.get('high_quality', 0)} · "
                    f"低质量 {counters.get('low_quality', 0)} · "
                    f"卖课 {counters.get('marketer', 0)}"
                ),
                level="info",
            )
        except Exception:
            log.warning("corpus_classify: notify failed", exc_info=True)

    return counters


# Daily 07:00 — runs after discover_jobs (06:30) so newly-discovered
# corpus items get classified before silence_check (09:00) reads them.
CORPUS_CLASSIFY_JOB = JobSpec(
    name="corpus_classify",
    func=run,
    trigger="cron",
    trigger_kwargs={"hour": 7, "minute": 0},
    misfire_grace_time_s=1800,
)
