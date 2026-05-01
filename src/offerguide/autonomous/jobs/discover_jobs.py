"""Daily JD-discovery job — runs spiders, ingests, optionally auto-scores.

Closes the "用户得自己一个个粘 JD" loop. The agent now wakes up every day,
runs the spider lineup, ingests new candidates into ``jobs``, and—when LLM
+ user profile are configured—runs ``score_match`` on JDs that have enough
text to score. High-score JDs auto-push to the inbox so they appear on the
next morning's daily-standup home.

Skip semantics (graceful degradation):
- No spiders configured → ``skipped='no_spiders'``
- No LLM / profile → still ingests, just doesn't auto-score (degraded mode)
- One spider crashes → others still run; failure recorded in result

Per-spider cap: ``MAX_ITEMS_PER_SPIDER`` keeps a runaway spider from
overwhelming the daily run; the daemon's ``max_instances=1`` covers
re-entrance.
"""

from __future__ import annotations

import logging
from typing import Any

from ...auto_pipeline import build_default_spider_set, run_all_spiders
from ..scheduler import JobContext, JobSpec

log = logging.getLogger(__name__)

MAX_ITEMS_PER_SPIDER = 30
"""Per-run cap. With current sources (1 spider, namewyf/Campus2026 ~30
互联网/AI rows), this is 'whole list' — it caps growth as more spiders
or sources are added."""


def run(ctx: JobContext) -> dict[str, Any]:
    spiders = build_default_spider_set()
    if not spiders:
        return {"skipped": "no_spiders"}

    # Resolve auto-eval prerequisites.  If either is missing, we still
    # ingest — we just don't score.  This is intentional: a spider sweep
    # is valuable even without the LLM (the user can browse the
    # 投递看板 and pick what to evaluate manually).
    skills = getattr(ctx, "skills", None)
    user_profile_text = getattr(ctx, "user_profile_text", None)
    runtime = getattr(ctx, "runtime", None)
    can_eval = bool(runtime and skills and user_profile_text)

    results = run_all_spiders(
        spiders,
        store=ctx.store,
        runtime=runtime,
        skills=skills,
        user_profile_text=user_profile_text,
        max_items_per_spider=MAX_ITEMS_PER_SPIDER,
        auto_eval=can_eval,
    )

    summary: dict[str, Any] = {
        "spiders_run": len(results),
        "auto_eval_enabled": can_eval,
        "by_spider": [r.summary() for r in results],
        "total_new_jobs": sum(len(r.new_jobs) for r in results),
        "total_inboxed": sum(len(r.pushed_to_inbox) for r in results),
        "total_errors": sum(len(r.errors) + len(r.spider_errors) for r in results),
    }

    # Notify on inbox pushes — these matter to the user
    if ctx.notifier and summary["total_inboxed"] > 0:
        try:
            ctx.notifier.notify(
                title=f"OfferGuide: 自动发现 {summary['total_inboxed']} 个高匹配 JD",
                body=f"今日新增 {summary['total_new_jobs']} 条候选；"
                     f"{summary['total_inboxed']} 条已推到收件箱待你决策。",
                level="info",
            )
        except Exception:
            log.warning("discover_jobs: notify failed", exc_info=True)

    return summary


# Daily 06:30 — earlier than silence_check (09:00) so the morning standup
# already has the latest discoveries.
DISCOVER_JOBS_JOB = JobSpec(
    name="discover_jobs",
    func=run,
    trigger="cron",
    trigger_kwargs={"hour": 6, "minute": 30},
    misfire_grace_time_s=3600,  # 1h tolerance — laptop suspended overnight is normal
)
