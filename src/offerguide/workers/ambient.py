"""W15.23 — Ambient job discovery loop.

The "agent 全自动找岗位" the user asked for: a background asyncio task
that runs forever, periodically:

1. Crawl nowcoder via the existing sitemap-based scout (scout.crawl_nowcoder).
   nowcoder publishes a public, robots-allowed sitemap chain — no login, no
   reverse engineering, just polite HTTP.
2. For each newly-ingested job, run the score_match SKILL so it appears
   immediately on /recommended with a real probability + ranked position.

This module is the wiring; the crawler (workers/scout.crawl_nowcoder),
the SKILL (skills/score_match), and the UI (/recommended) all already exist
— pre-W15.23 they just weren't connected end-to-end. The user has been
waiting for this since W14.

Design choices:

- **Single asyncio task per process**, not multi-source parallel — avoids
  rate-limiting cliff. Add new sources sequentially inside the same loop
  iteration once we ship them.
- **Delay 30s after startup** before first run so the server can serve
  HTTP responses while the crawl is in-flight.
- **Cap each cycle at 30 JD fetches** so we don't burn 1000 LLM calls on
  the very first crawl. Over 6h windows that's 120 jobs/day evaluated,
  which matches user dogfood capacity.
- **All errors logged + swallowed**, never crashes the loop. Network
  blips, parse changes, LLM rate limits — none should kill the daemon.
- **Honors daily budget** (llm.budget) — if cap exceeded, skip scoring,
  keep ingest.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .. import Store
    from ..config import Settings
    from ..skills import SkillRuntime, SkillSpec

log = logging.getLogger(__name__)

# Tuneable; user can override later. Values picked from the doc:
# 30 jd/cycle * 4 cycles/day = 120 jobs/day evaluated, well below LLM
# daily cap with 6h spacing.
DEFAULT_CRAWL_LIMIT_PER_CYCLE = 30
DEFAULT_CYCLE_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours
INITIAL_DELAY_SECONDS = 30  # let the app finish booting first


async def _ambient_discovery_loop(
    *,
    store: Store,
    settings: Settings,
    runtime: SkillRuntime | None,
    skills: list[SkillSpec],
    user_profile_text: str | None,
    crawl_limit_per_cycle: int = DEFAULT_CRAWL_LIMIT_PER_CYCLE,
    cycle_interval_s: int = DEFAULT_CYCLE_INTERVAL_SECONDS,
    initial_delay_s: int = INITIAL_DELAY_SECONDS,
) -> None:
    """Forever loop: crawl new jobs → score them → sleep cycle_interval."""
    log.info(
        "ambient discovery: starting (initial_delay=%ds, cycle=%ds, limit=%d)",
        initial_delay_s, cycle_interval_s, crawl_limit_per_cycle,
    )
    try:
        await asyncio.sleep(initial_delay_s)
    except asyncio.CancelledError:
        log.info("ambient discovery: cancelled before first run")
        raise

    while True:
        try:
            await _run_one_cycle(
                store=store, settings=settings, runtime=runtime,
                skills=skills, user_profile_text=user_profile_text,
                crawl_limit=crawl_limit_per_cycle,
            )
        except asyncio.CancelledError:
            log.info("ambient discovery: cancelled mid-cycle")
            raise
        except Exception as e:
            # Never let the loop die. Log and move on.
            log.exception("ambient discovery: cycle failed: %s", e)

        try:
            await asyncio.sleep(cycle_interval_s)
        except asyncio.CancelledError:
            log.info("ambient discovery: cancelled during sleep")
            raise


async def _run_one_cycle(
    *,
    store: Store,
    settings: Settings,
    runtime: SkillRuntime | None,
    skills: list[SkillSpec],
    user_profile_text: str | None,
    crawl_limit: int,
) -> None:
    """One cycle: crawl known sources + search the web, then score each."""
    from . import scout
    counters = await asyncio.to_thread(
        scout.crawl_nowcoder, store, limit=crawl_limit,
    )
    log.info("ambient discovery: nowcoder crawl done: %s", counters)

    # Search-agent discovery covers company career pages, official ATS pages,
    # and pages that are not in platform sitemaps. This is the missing
    # "don't make the user scroll boards manually" path.
    if settings.deepseek_api_key:
        try:
            search_result = await asyncio.to_thread(
                _run_agent_search_blocking,
                store=store,
                settings=settings,
            )
            log.info("ambient discovery: search agent done: %s", search_result)
        except Exception as e:
            log.exception("ambient discovery: search agent failed: %s", e)

    # Score newly-ingested jobs. We can't easily know all ids that were new
    # across adapters, so query jobs from automatic sources without a scored
    # event yet.
    if runtime is None or not user_profile_text or not skills:
        log.info("ambient discovery: skipping score (no runtime / no profile / no skills)")
        return

    unscored = await asyncio.to_thread(_load_unscored_discovered_ids, store)
    if not unscored:
        return

    log.info("ambient discovery: scoring %d new jobs", len(unscored))
    await asyncio.to_thread(
        _score_jobs_blocking,
        store=store, settings=settings, runtime=runtime, skills=skills,
        user_profile_text=user_profile_text, job_ids=unscored,
    )


def _load_unscored_discovered_ids(store: Store, limit: int = 30) -> list[int]:
    """Pick automatically discovered jobs with no 'scored' event yet.

    Limit is a per-cycle cap so we don't burn 1000 LLM calls if there's a
    big initial backlog.
    """
    try:
        from ..harness import _schema as _hs
        _hs.init_harness_schema(store)
    except Exception:
        pass
    sources = (
        "nowcoder",
        "agent_search",
        "user_paste_url",
        "boss_extension",
        "tencent_campus",
        "tencent_social",
        "baidu_campus",
        "baidu_intern",  # W17 — recruitType=INTERN 拉的暑期+日常实习
    )
    placeholders = ",".join("?" * len(sources))
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT j.id FROM jobs j "
            f"WHERE j.source IN ({placeholders}) "
            "  AND length(j.raw_text) >= 200 "
            "  AND NOT EXISTS ("
            "      SELECT 1 FROM harness_events he "
            "      WHERE he.kind = 'scored' AND he.job_id = j.id"
            "  ) "
            "ORDER BY j.id DESC LIMIT ?",
            (*sources, limit),
        ).fetchall()
    return [int(r[0]) for r in rows]


def _load_unscored_nowcoder_ids(store: Store, limit: int = 30) -> list[int]:
    """Backward-compatible helper used by tests and older callers."""
    try:
        from ..harness import _schema as _hs
        _hs.init_harness_schema(store)
    except Exception:
        pass
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT j.id FROM jobs j "
            "WHERE j.source = 'nowcoder' "
            "  AND NOT EXISTS ("
            "      SELECT 1 FROM harness_events he "
            "      WHERE he.kind = 'scored' AND he.job_id = j.id"
            "  ) "
            "ORDER BY j.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [int(r[0]) for r in rows]


def _run_agent_search_blocking(*, store: Store, settings: Settings) -> dict[str, Any]:
    """Run the LLM-driven web search agent once and record a lightweight trail."""
    from ..agentic.job_finder_agent import JobFinderAgent
    from ..agentic.search import build_default_search
    from ..harness import _schema as _hs
    from ..llm import BudgetExceeded, LLMClient, enforce_daily_budget

    try:
        enforce_daily_budget(store)
    except BudgetExceeded as e:
        return {"skipped": "budget_exceeded", "error": str(e)}

    north_star = _load_active_north_star(store)
    llm = LLMClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        default_model=settings.default_model,
    )
    search = build_default_search()
    agent = JobFinderAgent(store=store, llm=llm, search=search)
    try:
        result = agent.run(north_star=north_star)
    finally:
        agent.close()
        close_search = getattr(search, "close", None)
        if callable(close_search):
            close_search()
        llm.close()

    summary = {
        "north_star": north_star,
        "iterations": result.iterations,
        "inserted": result.inserted,
        "skipped_dup": result.skipped_dup,
        "job_ids": result.new_job_ids[:10],
        "queries": result.search_queries[:8],
        "finish_reason": result.finish_reason[:200],
        "cost_usd": round(result.total_cost_usd, 5),
    }
    try:
        _hs.init_harness_schema(store)
        import json
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_events(kind, note, source) VALUES (?, ?, ?)",
                (
                    "agent_search_discovered",
                    json.dumps(summary, ensure_ascii=False, default=str)[:2000],
                    "ambient",
                ),
            )
    except Exception:
        log.debug("ambient discovery: failed to record search summary", exc_info=True)
    return summary


def _load_active_north_star(store: Store) -> str:
    """Use the user's active goal as search criteria; fall back to a sane default."""
    try:
        from .. import goals as _goals
        active = _goals.list_active_goals(store)
    except Exception:
        active = []
    if not active:
        return "AI Agent / LLM 应用 暑期实习 校招 岗位"
    goal = active[0]
    parts = [goal.title]
    if goal.description:
        parts.append(goal.description)
    if goal.target_metric:
        parts.append(f"目标指标: {goal.target_metric}")
    return "；".join(p for p in parts if p)[:500]


def _score_jobs_blocking(
    *,
    store: Store,
    settings: Settings,
    runtime: SkillRuntime,
    skills: list[SkillSpec],
    user_profile_text: str,
    job_ids: list[int],
) -> None:
    """Run score_match for each job_id via the harness tool dispatch."""
    from ..harness import HarnessDeps, MemoryStore, default_worldview_dir
    from ..harness import _schema as _hs
    from ..harness.tools import _exec_score_match
    from ..llm import BudgetExceeded, enforce_daily_budget

    _hs.init_harness_schema(store)
    deps = HarnessDeps(
        settings=settings, store=store,
        memory_store=MemoryStore(root=default_worldview_dir(settings)),
        runtime=runtime, skills=skills,
        user_profile_text=user_profile_text,
    )

    for jid in job_ids:
        try:
            enforce_daily_budget(store)
        except BudgetExceeded as e:
            log.warning("ambient discovery: budget exceeded, stopping: %s", e)
            return
        try:
            result = _exec_score_match({"job_id": jid}, deps)
            if result.startswith("OK"):
                log.info("ambient discovery: scored job#%d", jid)
            else:
                log.warning("ambient discovery: score for job#%d returned: %s",
                            jid, result[:200])
        except Exception as e:
            log.exception("ambient discovery: score job#%d failed: %s", jid, e)


def status_summary(store: Store) -> dict[str, Any]:
    """For /metrics or debug pages — show the ambient daemon's footprint."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE source = 'nowcoder'"
        ).fetchone()
        nowcoder_count = int(row[0]) if row else 0
        try:
            row2 = conn.execute(
                "SELECT COUNT(*) FROM harness_events "
                "WHERE kind = 'scored' "
                "  AND created_at >= julianday('now', '-7 days')"
            ).fetchone()
            scored_week = int(row2[0]) if row2 else 0
        except Exception:
            scored_week = 0
    return {
        "nowcoder_jobs_total": nowcoder_count,
        "scored_last_7d": scored_week,
    }
