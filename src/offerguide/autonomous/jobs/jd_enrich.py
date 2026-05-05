"""Daily JD enrichment job — closes the spider→SKILL gap.

The audit pinpoint: ``awesome_jobs`` spider yields rows with raw_text
~138 bytes (just company + portal URL). Below the
``MIN_TEXT_FOR_AUTO_EVAL = 200`` threshold, all downstream SKILLs
(score_match / tailor_resume / 4-bucket Gap) skip auto-eval.

This job runs ``jd_enricher.enrich_pending`` on jobs whose raw_text is
still thin and that haven't been tried yet. Per-run cap keeps token
spend bounded; idempotent (already-enriched rows skip).

Schedule: daily 06:45 — runs 15 minutes after ``discover_jobs`` (06:30)
so newly-spider'd entries get enriched before ``corpus_classify`` (07:00)
sees them.
"""

from __future__ import annotations

import logging
from typing import Any

from ...jd_enricher import enrich_pending
from ..scheduler import JobContext, JobSpec

log = logging.getLogger(__name__)

MAX_PER_RUN = 15
"""Per-run cap. With Claude Sonnet at ~30s per JD enrich (HTML + LLM)
this keeps the cron tick under ~8 minutes wall-clock."""


def run(ctx: JobContext, *, limit: int | None = None) -> dict[str, Any]:
    """Enrich up to ``limit`` thin JDs in one tick. ``limit=None`` uses the
    module default (``MAX_PER_RUN``); the agent maintenance tool passes an
    explicit per-call limit so the agent can budget its tick.

    W14.9: previously ``limit`` was passed via the global env var
    ``OFFERGUIDE_JD_ENRICH_MAX``. Two issues with that:
      1) The env var was never read here (silent dead code — the agent
         thought it was throttling itself, but the daemon ran the default).
      2) os.environ is process-global; concurrent agents would race on it.
    Explicit kwarg fixes both.
    """
    if ctx.llm is None:
        log.info("jd_enrich: LLM not configured, skipping")
        return {"skipped": "no_llm"}

    actual_limit = limit if limit is not None else MAX_PER_RUN
    counters = enrich_pending(
        ctx.store, llm=ctx.llm, limit=actual_limit,
    )

    if ctx.notifier and counters.get("ok", 0) > 0:
        try:
            ctx.notifier.notify(
                title=f"OfferGuide: 补全 {counters['ok']} 条 JD 详情",
                body=(
                    f"扫了 {counters['scanned']} 条 thin JD: "
                    f"成功 {counters['ok']}, "
                    f"JS-rendered {counters['js_rendered']}, "
                    f"抓失败 {counters['fetch_failed']}, "
                    f"抽空 {counters['extracted_thin']}"
                ),
                level="info",
            )
        except Exception:
            log.warning("jd_enrich: notify failed", exc_info=True)

    return counters


# Daily 06:45 — between discover_jobs (06:30) and corpus_classify (07:00)
JD_ENRICH_JOB = JobSpec(
    name="jd_enrich",
    func=run,
    trigger="cron",
    trigger_kwargs={"hour": 6, "minute": 45},
    misfire_grace_time_s=3600,
)
