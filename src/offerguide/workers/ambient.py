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
    """One cycle: crawl known sources + search the web, then score each.

    W18: 多 keyword 派发. 用户原话: "不一定是这些大厂, 因为大厂基本上大家
    都知道投, 我们需要的是全以及与用户匹配". 所以这一轮:
    1. nowcoder sitemap 拉一遍 (~30 个全行业岗位, 含中小厂)
    2. 用 user resume 抽出来的 keywords 同时打 verified_official (腾讯+百度)
       — 用真 keyword "Diffusion 模型" / "RLHF" 等找到大厂里 niche 团队
    3. agent_search 用 keywords 找 AI 创业公司 / 中小厂 (大厂之外的世界)
    W19+: 4. 0voice GitHub repo 聚合 — 100+ 公司 1000+ 岗位汇总, 每日更新.
       一次拉到位, 含真 ATS URL (阿里 campus-talent / 字节 / 北森 SaaS)
    W20: 5. 实习僧 (shixiseng.com) 实习专用聚合 — niche AI 创业公司实习.

    W20.2 — 4 个 fetch 阶段 (nowcoder / 0voice / verified×kw / shixiseng×kw)
    并行 via asyncio.gather. 它们打不同域 (nowcoder.com / raw.githubusercontent.com
    / qq.com+baidu.com / shixiseng.com), 互不抢 rate limit. 顺序执行白浪费.
    agent_search 单独跑因为它是 LLM, 跟 fetch 抢 LLM rate limit/cost.
    """
    import time as _time
    cycle_t0 = _time.monotonic()

    # ── Stage 1: extract keywords (cheap, in-process) ─────────────────
    keywords = _extract_cycle_keywords(store=store, user_profile_text=user_profile_text)
    log.info(
        "ambient discovery: cycle keywords: %s",
        [k.keyword for k in keywords],
    )

    # ── Stage 2-5: 4 fetch sources in parallel ────────────────────────
    from . import scout
    from ..platforms.zerovoice import crawl_zerovoice

    async def _stage_nowcoder() -> tuple[str, float, Any, Exception | None]:
        t = _time.monotonic()
        try:
            r = await asyncio.to_thread(scout.crawl_nowcoder, store, limit=crawl_limit)
            return ("nowcoder", _time.monotonic() - t, r, None)
        except Exception as e:
            return ("nowcoder", _time.monotonic() - t, None, e)

    async def _stage_zerovoice() -> tuple[str, float, Any, Exception | None]:
        t = _time.monotonic()
        try:
            r = await asyncio.to_thread(crawl_zerovoice, store, max_jobs=80)
            return ("0voice", _time.monotonic() - t, r, None)
        except Exception as e:
            return ("0voice", _time.monotonic() - t, None, e)

    async def _stage_verified() -> tuple[str, float, Any, Exception | None]:
        t = _time.monotonic()
        if not keywords:
            return ("verified_official", 0.0, {"skipped": "no_keywords"}, None)
        try:
            r = await asyncio.to_thread(
                _crawl_verified_official_per_keyword,
                store=store,
                keywords=[k.keyword for k in keywords[:5]],
                limit_per_kw=3,
            )
            return ("verified_official", _time.monotonic() - t, r, None)
        except Exception as e:
            return ("verified_official", _time.monotonic() - t, None, e)

    async def _stage_shixiseng() -> tuple[str, float, Any, Exception | None]:
        t = _time.monotonic()
        if not keywords:
            return ("shixiseng", 0.0, {"skipped": "no_keywords"}, None)
        try:
            r = await asyncio.to_thread(
                _crawl_shixiseng_per_keyword,
                store=store,
                keywords=[k.keyword for k in keywords[:2]],
                limit_per_kw=8,
            )
            return ("shixiseng", _time.monotonic() - t, r, None)
        except Exception as e:
            return ("shixiseng", _time.monotonic() - t, None, e)

    fetch_results = await asyncio.gather(
        _stage_nowcoder(), _stage_zerovoice(),
        _stage_verified(), _stage_shixiseng(),
    )
    for name, dur, payload, err in fetch_results:
        if err is not None:
            log.exception("ambient discovery: %s failed (%.1fs): %s", name, dur, err)
        else:
            log.info("ambient discovery: %s done (%.1fs): %s", name, dur, payload)

    # ── Stage 6: agent_search (separate — uses LLM, can't share rate limit) ──
    if settings.deepseek_api_key:
        t = _time.monotonic()
        try:
            search_result = await asyncio.to_thread(
                _run_agent_search_blocking,
                store=store,
                settings=settings,
                seed_keywords=[k.keyword for k in keywords[:3]],
            )
            log.info(
                "ambient discovery: agent_search done (%.1fs): %s",
                _time.monotonic() - t, search_result,
            )
        except Exception as e:
            log.exception(
                "ambient discovery: agent_search failed (%.1fs): %s",
                _time.monotonic() - t, e,
            )

    # ── Stage 7: score newly-ingested jobs (parallel via ThreadPool) ──
    if runtime is None or not user_profile_text or not skills:
        log.info("ambient discovery: skipping score (no runtime / no profile / no skills)")
        log.info(
            "ambient discovery: cycle done in %.1fs (no scoring)",
            _time.monotonic() - cycle_t0,
        )
        return

    unscored = await asyncio.to_thread(_load_unscored_discovered_ids, store)
    if not unscored:
        log.info(
            "ambient discovery: cycle done in %.1fs (0 unscored)",
            _time.monotonic() - cycle_t0,
        )
        return

    score_t = _time.monotonic()
    await asyncio.to_thread(
        _score_jobs_blocking,
        store=store, settings=settings, runtime=runtime, skills=skills,
        user_profile_text=user_profile_text, job_ids=unscored,
    )
    log.info(
        "ambient discovery: cycle done in %.1fs (scoring took %.1fs of %d jobs)",
        _time.monotonic() - cycle_t0, _time.monotonic() - score_t, len(unscored),
    )


# W20.2 — cache by (resume_text_hash, active_goal_hash). 简历 + 目标都不变
# (用户 6h 内一般不改) → 命中率高. 每 hit 省 1 DB read + python regex match.
# 失效: 简历或目标改了, 自动重算 (hash 变).
_KW_CACHE: dict[tuple[str, str], list] = {}
_KW_CACHE_MAX_ENTRIES = 8  # 上限防泄漏 — 1 user 一般 1 entry, 多 user 部署也够


def _extract_cycle_keywords(*, store: Store, user_profile_text: str | None) -> list:
    """W18 — pull deterministic keywords from resume + active goal.

    W20.2 — cache by (resume_hash, active_goal). DB read + regex 不重算.
    """
    import hashlib
    from ..match_keywords import DEFAULT_KEYWORDS_PER_CYCLE, extract_keywords
    active = _load_active_north_star(store)
    resume_h = hashlib.sha1(
        (user_profile_text or "").encode("utf-8"),
    ).hexdigest()[:16]
    cache_key = (resume_h, active or "")
    if cache_key in _KW_CACHE:
        return _KW_CACHE[cache_key]
    out = extract_keywords(
        user_profile_text, active_goal=active,
        max_keywords=DEFAULT_KEYWORDS_PER_CYCLE,
    )
    # Bound cache size — drop oldest entry if at cap (FIFO is fine, no LRU needed)
    if len(_KW_CACHE) >= _KW_CACHE_MAX_ENTRIES:
        _KW_CACHE.pop(next(iter(_KW_CACHE)))
    _KW_CACHE[cache_key] = out
    return out


def _crawl_verified_official_per_keyword(
    *, store: Store, keywords: list[str], limit_per_kw: int = 3,
) -> dict[str, Any]:
    """For each keyword, hit verified_official (腾讯+百度) and ingest. Returns
    counter dict for log. De-dups via scout.ingest content_hash.
    """
    from ..platforms.official_jobs import search_verified_official_jobs
    from . import scout

    inserted_total = 0
    dup_total = 0
    per_kw: dict[str, int] = {}
    per_source: dict[str, int] = {}
    errors: list[str] = []

    for kw in keywords:
        try:
            results = search_verified_official_jobs(
                company=None, keyword=kw, limit=limit_per_kw,
            )
        except Exception as e:
            errors.append(f"{kw}: {type(e).__name__}: {e}")
            continue

        kw_inserted = 0
        for sr in results:
            for rj in sr.jobs:
                # Annotate which keyword found this job (audit trail)
                rj.extras.setdefault("discovered_via", "verified_official")
                rj.extras.setdefault("discovered_keyword", kw)
                try:
                    was_new, _ = scout.ingest(store, rj)
                    if was_new:
                        inserted_total += 1
                        kw_inserted += 1
                        per_source[sr.source] = per_source.get(sr.source, 0) + 1
                    else:
                        dup_total += 1
                except Exception as e:
                    errors.append(f"{kw}/{sr.source}: ingest {type(e).__name__}: {e}")
        per_kw[kw] = kw_inserted

    return {
        "inserted_total": inserted_total,
        "duplicate_total": dup_total,
        "per_keyword": per_kw,
        "per_source": per_source,
        "errors": errors[:5],
    }


def _crawl_shixiseng_per_keyword(
    *, store: Store, keywords: list[str], limit_per_kw: int = 8,
) -> dict[str, Any]:
    """W20 — for each keyword, hit shixiseng's list page + ingest top N
    detail pages. Returns counter dict for log."""
    from ..platforms.shixiseng import crawl_shixiseng

    inserted_total = 0
    dup_total = 0
    parsed_total = 0
    per_kw: dict[str, int] = {}
    by_company_total: dict[str, int] = {}
    errors: list[str] = []

    for kw in keywords:
        try:
            r = crawl_shixiseng(store, keyword=kw, max_jobs=limit_per_kw)
        except Exception as e:
            errors.append(f"{kw}: {type(e).__name__}: {e}")
            continue
        inserted_total += r.inserted
        dup_total += r.duplicate
        parsed_total += r.parsed
        per_kw[kw] = r.inserted
        for comp, n in r.by_company.items():
            by_company_total[comp] = by_company_total.get(comp, 0) + n
        if r.errors:
            errors.extend([f"{kw}: {e}" for e in r.errors[:3]])

    return {
        "inserted_total": inserted_total,
        "duplicate_total": dup_total,
        "parsed_total": parsed_total,
        "per_keyword": per_kw,
        "company_diversity": len(by_company_total),
        "errors": errors[:5],
    }


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
        "zerovoice_repo",  # W19+ — 0voice GitHub aggregator
        "bytedance_jobs",  # W19+ — 字节社招 JSON API (实测 1334 岗位)
        "shixiseng",  # W20 — 实习僧 实习专用聚合 (含 niche AI 创业公司实习)
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


DEFAULT_AGENT_SEARCH_MAX_ITERATIONS = 8
"""W20.2 — cap ambient JobFinderAgent iterations.

Dogfood 2026-05-11 (docs/dogfood_2026-05-11/agent_search_seed_keywords.md):
- Default 25 iter, agent ran 19 / 174s / $0.0078 / 41 jobs (40 大厂 + 1 niche)
- Diminishing returns past iter ~5 — first 5 calls hit verified_official
  and fill 30+ jobs cheap, later iters chase web search niche which mostly
  fails (SPA pages can't be fetched server-side)
- 0voice + shixiseng (W19+/W20) now cover niche better than agent web search
- Ambient is recurring (every 6h) — cap should reflect "incremental" use,
  not "first-time exhaustive"

8 iter ≈ 60s ≈ $0.003 per cycle. 4 cycles/day = $0.012/day on agent_search,
well under daily $5 cap and headroom for score_match's ~30 calls.
"""


def _run_agent_search_blocking(
    *, store: Store, settings: Settings,
    seed_keywords: list[str] | None = None,
    max_iterations: int = DEFAULT_AGENT_SEARCH_MAX_ITERATIONS,
) -> dict[str, Any]:
    """Run the LLM-driven web search agent once and record a lightweight trail.

    W18: ``seed_keywords`` (from user resume) are appended to the north_star
    so the agent searches for niche-fitting middle-tier companies (smart AI
    startups), not just generic 大厂 keywords.

    W20.2: ``max_iterations`` defaults to 8 (was unbounded → 25 default in
    JobFinderAgent). Saves ~110s + $0.005 per cycle vs. the 19-iter dogfood
    baseline.
    """
    import os as _os

    from ..agentic.job_finder_agent import JobFinderAgent
    from ..agentic.search import build_default_search
    from ..harness import _schema as _hs
    from ..llm import BudgetExceeded, LLMClient, enforce_daily_budget

    try:
        enforce_daily_budget(store)
    except BudgetExceeded as e:
        return {"skipped": "budget_exceeded", "error": str(e)}

    # Env override for users who want to tune
    env_iter = _os.environ.get("OFFERGUIDE_AGENT_SEARCH_MAX_ITER")
    if env_iter:
        try:
            max_iterations = max(1, min(50, int(env_iter)))
        except ValueError:
            pass

    north_star = _load_active_north_star(store)
    if seed_keywords:
        # W18 — surface niche keywords to the agent so it doesn't only search
        # 大厂. The agent's system prompt already says 'find AI 创业公司';
        # giving it the user's resume keywords focuses what to search on.
        north_star = (
            f"{north_star}\n\n"
            f"用户简历命中的 niche 关键词: {' / '.join(seed_keywords)}\n"
            f"优先用这些 keyword 找匹配的 AI 创业公司 / 中小厂, 不只大厂."
        )
    llm = LLMClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        default_model=settings.default_model,
    )
    search = build_default_search()
    agent = JobFinderAgent(
        store=store, llm=llm, search=search,
        max_iterations=max_iterations,
    )
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


DEFAULT_SCORE_PARALLELISM = 4
"""Concurrent score_match invocations per cycle.

LLM call ~5-10s each; running in parallel reduces 30 jobs from 150-300s to
~40-75s. Bound by:
- LLM provider's per-account rate limit (DeepSeek allows 10+ concurrent)
- SQLite WAL mode (multiple readers + serialized writers, fine for 4)
- Each thread gets its own Store connection (Store.connect is per-call)

4 picked as a safe default — bumps perf 4x without straining provider QPS.
Configurable via OFFERGUIDE_SCORE_PARALLELISM env if user hits rate limits.
"""


def _score_jobs_blocking(
    *,
    store: Store,
    settings: Settings,
    runtime: SkillRuntime,
    skills: list[SkillSpec],
    user_profile_text: str,
    job_ids: list[int],
    parallelism: int = DEFAULT_SCORE_PARALLELISM,
) -> None:
    """Run score_match for each job_id via the harness tool dispatch.

    W20.2 — parallelized. Pre-W20.2: sequential `for jid in job_ids: ...`,
    each ~5-10s LLM call → 30 jobs blocks 150-300s. Parallel via
    ThreadPoolExecutor cuts that ~4x.

    Each worker thread:
    - Calls _exec_score_match (loads job, calls LLM, writes harness_event)
    - Independent Store.connect() (sqlite WAL allows concurrent readers + serialized writers)
    - Independent runtime.invoke (httpx.Client is thread-safe; LLMClient also thread-safe)

    Budget check is per-job on the calling side. We re-check inside the worker
    for safety so concurrent jobs don't all stampede past a fresh budget cap.
    """
    import os as _os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ..harness import HarnessDeps, MemoryStore, default_worldview_dir
    from ..harness import _schema as _hs
    from ..harness.tools import _exec_score_match
    from ..llm import BudgetExceeded, enforce_daily_budget

    _hs.init_harness_schema(store)
    # Allow env override for users hitting rate limits
    env_par = _os.environ.get("OFFERGUIDE_SCORE_PARALLELISM")
    if env_par:
        try:
            parallelism = max(1, min(16, int(env_par)))
        except ValueError:
            pass

    if not job_ids:
        return

    # Pre-flight budget check (one DB read) — cheap fast-fail
    try:
        enforce_daily_budget(store)
    except BudgetExceeded as e:
        log.warning("ambient discovery: budget exceeded before scoring: %s", e)
        return

    # Single shared HarnessDeps — store is thread-safe (per-call connect),
    # runtime is thread-safe (LLMClient.chat is stateless beyond config),
    # MemoryStore writes are append-only.
    deps = HarnessDeps(
        settings=settings, store=store,
        memory_store=MemoryStore(root=default_worldview_dir(settings)),
        runtime=runtime, skills=skills,
        user_profile_text=user_profile_text,
    )

    def _score_one(jid: int) -> tuple[int, str | None, Exception | None]:
        try:
            enforce_daily_budget(store)
        except BudgetExceeded as e:
            return jid, f"BUDGET_EXCEEDED: {e}", None
        try:
            return jid, _exec_score_match({"job_id": jid}, deps), None
        except Exception as e:
            return jid, None, e

    n_workers = max(1, min(parallelism, len(job_ids)))
    log.info(
        "ambient discovery: scoring %d jobs with parallelism=%d",
        len(job_ids), n_workers,
    )
    ok = warn = fail = budget_stops = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_score_one, jid): jid for jid in job_ids}
        for fut in as_completed(futures):
            jid, result, exc = fut.result()
            if exc is not None:
                fail += 1
                log.exception("ambient discovery: score job#%d failed: %s", jid, exc)
                continue
            if result and result.startswith("OK"):
                ok += 1
                log.info("ambient discovery: scored job#%d", jid)
            elif result and result.startswith("BUDGET_EXCEEDED"):
                budget_stops += 1
                # Don't break here (would orphan in-flight futures); just log.
                log.warning("ambient discovery: budget cap hit on job#%d", jid)
            else:
                warn += 1
                log.warning(
                    "ambient discovery: score for job#%d returned: %s",
                    jid, (result or "")[:200],
                )
    log.info(
        "ambient discovery: scoring done — ok=%d warn=%d fail=%d budget_stops=%d",
        ok, warn, fail, budget_stops,
    )


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
