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


def run(ctx: JobContext) -> dict[str, Any]:
    if ctx.llm is None:
        log.info("jd_enrich: LLM not configured, skipping")
        return {"skipped": "no_llm"}

    counters = enrich_pending(
        ctx.store, llm=ctx.llm, limit=MAX_PER_RUN,
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
