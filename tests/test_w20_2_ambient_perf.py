"""W20.2 — performance refactor tests for ambient.py.

Covers:
- Parallel score_match (ThreadPoolExecutor with N workers)
- Budget exceeded handling (no crash, graceful stop)
- env-var overrides (OFFERGUIDE_SCORE_PARALLELISM / OFFERGUIDE_AGENT_SEARCH_MAX_ITER)
- _run_agent_search_blocking passes max_iterations to JobFinderAgent
- _run_one_cycle's 4 fetch stages run in parallel (timing-asserted)
"""
from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock, patch

import pytest

import offerguide
from offerguide.config import Settings
from offerguide.harness import _schema as harness_schema
from offerguide.workers import ambient


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "ambient_perf.db")
    s.init_schema()
    harness_schema.init_harness_schema(s)
    return s


@pytest.fixture
def jobs(store):
    """Insert 4 score-eligible jobs and return their ids."""
    ids = []
    with store.connect() as conn:
        for i in range(4):
            cur = conn.execute(
                "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                ("nowcoder", f"job{i}", f"co{i}", "x" * 300, f"hash{i}"),
            )
            ids.append(cur.lastrowid or 0)
        conn.commit()
    return ids


# ── Parallel score_match ────────────────────────────────────────────────


class TestScoreJobsParallel:
    """Ensure _score_jobs_blocking actually runs in parallel."""

    def test_parallel_execution_via_timing(self, store, jobs, monkeypatch):
        """4 jobs × 0.5s LLM = 2s sequential, but parallelism=4 → ≤1s.

        Asserts the ThreadPoolExecutor fan-out actually overlaps work."""
        call_count = {"n": 0}

        def _slow_score_match(args, deps):
            call_count["n"] += 1
            time.sleep(0.5)
            return f"OK score_match for job#{args['job_id']}"

        monkeypatch.setattr(
            "offerguide.harness.tools._exec_score_match", _slow_score_match,
        )
        # bypass enforce_daily_budget
        monkeypatch.setattr(
            "offerguide.llm.enforce_daily_budget", lambda s: None,
        )

        settings = Settings(deepseek_api_key="sk-fake")
        runtime = MagicMock()
        skills = [MagicMock(name="score_match")]

        t0 = time.monotonic()
        ambient._score_jobs_blocking(
            store=store, settings=settings, runtime=runtime, skills=skills,
            user_profile_text="resume", job_ids=jobs, parallelism=4,
        )
        elapsed = time.monotonic() - t0
        assert call_count["n"] == 4, "all 4 jobs should be scored"
        assert elapsed < 1.5, (
            f"expected parallel (<1.5s), got {elapsed:.2f}s — "
            f"likely running sequentially"
        )

    def test_sequential_baseline_for_comparison(self, store, jobs, monkeypatch):
        """parallelism=1 → must take ≥1.8s (4 × 0.5s with thread overhead)."""
        def _slow(args, deps):
            time.sleep(0.5)
            return f"OK job#{args['job_id']}"

        monkeypatch.setattr(
            "offerguide.harness.tools._exec_score_match", _slow,
        )
        monkeypatch.setattr(
            "offerguide.llm.enforce_daily_budget", lambda s: None,
        )

        t0 = time.monotonic()
        ambient._score_jobs_blocking(
            store=store, settings=Settings(deepseek_api_key="sk-fake"),
            runtime=MagicMock(), skills=[MagicMock()],
            user_profile_text="r", job_ids=jobs, parallelism=1,
        )
        assert time.monotonic() - t0 >= 1.8

    def test_empty_job_ids_is_noop(self, store):
        """No jobs → return immediately, no errors."""
        ambient._score_jobs_blocking(
            store=store, settings=Settings(deepseek_api_key="sk-fake"),
            runtime=MagicMock(), skills=[MagicMock()],
            user_profile_text="r", job_ids=[], parallelism=4,
        )

    def test_budget_exceeded_skips_all(self, store, jobs, monkeypatch):
        """Pre-flight budget check fails → no LLM calls happen."""
        from offerguide.llm import BudgetExceeded
        called = {"n": 0}

        def _raise(s):
            raise BudgetExceeded(today_spent_usd=10.0, cap_usd=5.0)

        def _should_not_be_called(args, deps):
            called["n"] += 1
            return "OK"

        monkeypatch.setattr("offerguide.llm.enforce_daily_budget", _raise)
        monkeypatch.setattr(
            "offerguide.harness.tools._exec_score_match", _should_not_be_called,
        )

        ambient._score_jobs_blocking(
            store=store, settings=Settings(deepseek_api_key="sk-fake"),
            runtime=MagicMock(), skills=[MagicMock()],
            user_profile_text="r", job_ids=jobs, parallelism=4,
        )
        assert called["n"] == 0

    def test_env_var_overrides_parallelism(self, store, jobs, monkeypatch):
        """OFFERGUIDE_SCORE_PARALLELISM env var clamps to [1, 16]."""
        captured = {"workers": []}

        original_pool = ambient.ThreadPoolExecutor if hasattr(ambient, "ThreadPoolExecutor") else None

        # Patch ThreadPoolExecutor to capture max_workers used
        from concurrent.futures import ThreadPoolExecutor as _RealPool

        def _spy_pool(max_workers=None, *args, **kwargs):
            captured["workers"].append(max_workers)
            return _RealPool(max_workers=max_workers, *args, **kwargs)

        monkeypatch.setattr(
            "offerguide.workers.ambient.ThreadPoolExecutor",
            _spy_pool, raising=False,
        )
        # Need to patch at the import site — _score_jobs_blocking imports
        # ThreadPoolExecutor inside the function, so patch at the module
        # where it's resolved.
        import concurrent.futures
        monkeypatch.setattr(
            concurrent.futures, "ThreadPoolExecutor", _spy_pool,
        )

        monkeypatch.setattr(
            "offerguide.harness.tools._exec_score_match",
            lambda args, deps: "OK",
        )
        monkeypatch.setattr(
            "offerguide.llm.enforce_daily_budget", lambda s: None,
        )

        monkeypatch.setenv("OFFERGUIDE_SCORE_PARALLELISM", "2")
        ambient._score_jobs_blocking(
            store=store, settings=Settings(deepseek_api_key="sk-fake"),
            runtime=MagicMock(), skills=[MagicMock()],
            user_profile_text="r", job_ids=jobs,
            parallelism=8,  # default override should win
        )
        # ThreadPoolExecutor called with max_workers=2 (env var won)
        assert 2 in captured["workers"]


# ── agent_search max_iterations cap ─────────────────────────────────────


class TestAgentSearchMaxIter:
    """W20.2 — _run_agent_search_blocking caps iterations for ambient use."""

    def test_default_max_iterations_passed_to_agent(self, store, monkeypatch):
        """Default cap is 8, must reach JobFinderAgent constructor."""
        captured = {}

        class _StubAgent:
            def __init__(self, *args, max_iterations=None, **kwargs):
                captured["max_iter"] = max_iterations
            def run(self, *, north_star):
                from offerguide.agentic.job_finder_agent import JobFinderResult
                return JobFinderResult(
                    iterations=1, inserted=0, skipped_dup=0,
                    new_job_ids=[], visited_urls=[], search_queries=[],
                    notes=[], finish_reason="ok", total_cost_usd=0.0,
                )
            def close(self): pass

        monkeypatch.setattr(
            "offerguide.agentic.job_finder_agent.JobFinderAgent", _StubAgent,
        )
        # Stub the build_default_search + LLMClient to avoid real init
        monkeypatch.setattr(
            "offerguide.agentic.search.build_default_search",
            lambda: MagicMock(close=lambda: None),
        )
        monkeypatch.setattr(
            "offerguide.llm.LLMClient",
            lambda **kwargs: MagicMock(close=lambda: None),
        )
        monkeypatch.setattr(
            "offerguide.llm.enforce_daily_budget", lambda s: None,
        )

        ambient._run_agent_search_blocking(
            store=store,
            settings=Settings(deepseek_api_key="sk-fake"),
        )
        assert captured["max_iter"] == ambient.DEFAULT_AGENT_SEARCH_MAX_ITERATIONS
        assert captured["max_iter"] == 8

    def test_explicit_max_iter_arg_overrides_default(self, store, monkeypatch):
        captured = {}

        class _StubAgent:
            def __init__(self, *args, max_iterations=None, **kwargs):
                captured["max_iter"] = max_iterations
            def run(self, *, north_star):
                from offerguide.agentic.job_finder_agent import JobFinderResult
                return JobFinderResult(
                    iterations=1, inserted=0, skipped_dup=0,
                    new_job_ids=[], visited_urls=[], search_queries=[],
                    notes=[], finish_reason="ok", total_cost_usd=0.0,
                )
            def close(self): pass

        monkeypatch.setattr(
            "offerguide.agentic.job_finder_agent.JobFinderAgent", _StubAgent,
        )
        monkeypatch.setattr(
            "offerguide.agentic.search.build_default_search",
            lambda: MagicMock(close=lambda: None),
        )
        monkeypatch.setattr(
            "offerguide.llm.LLMClient",
            lambda **kwargs: MagicMock(close=lambda: None),
        )
        monkeypatch.setattr(
            "offerguide.llm.enforce_daily_budget", lambda s: None,
        )

        ambient._run_agent_search_blocking(
            store=store,
            settings=Settings(deepseek_api_key="sk-fake"),
            max_iterations=3,
        )
        assert captured["max_iter"] == 3

    def test_env_var_overrides_max_iter(self, store, monkeypatch):
        captured = {}

        class _StubAgent:
            def __init__(self, *args, max_iterations=None, **kwargs):
                captured["max_iter"] = max_iterations
            def run(self, *, north_star):
                from offerguide.agentic.job_finder_agent import JobFinderResult
                return JobFinderResult(
                    iterations=1, inserted=0, skipped_dup=0,
                    new_job_ids=[], visited_urls=[], search_queries=[],
                    notes=[], finish_reason="ok", total_cost_usd=0.0,
                )
            def close(self): pass

        monkeypatch.setattr(
            "offerguide.agentic.job_finder_agent.JobFinderAgent", _StubAgent,
        )
        monkeypatch.setattr(
            "offerguide.agentic.search.build_default_search",
            lambda: MagicMock(close=lambda: None),
        )
        monkeypatch.setattr(
            "offerguide.llm.LLMClient",
            lambda **kwargs: MagicMock(close=lambda: None),
        )
        monkeypatch.setattr(
            "offerguide.llm.enforce_daily_budget", lambda s: None,
        )

        monkeypatch.setenv("OFFERGUIDE_AGENT_SEARCH_MAX_ITER", "5")
        ambient._run_agent_search_blocking(
            store=store,
            settings=Settings(deepseek_api_key="sk-fake"),
        )
        assert captured["max_iter"] == 5


# ── Cycle parallelism (4 fetch stages run concurrently) ─────────────────


class TestCycleParallelism:
    """W20.2 — _run_one_cycle's 4 fetch stages run in parallel via gather."""

    @pytest.mark.asyncio
    async def test_4_fetch_stages_run_concurrently(self, store, monkeypatch):
        """Each stage sleeps 0.5s. Sequential = 2s+, parallel = ≤1s."""
        from offerguide.platforms.zerovoice import FetchResult
        from offerguide.workers import scout

        # All 4 stages: each sleeps 0.5s synchronously (sync function called
        # via asyncio.to_thread), so they should overlap if .gather is used.
        def _slow_nowcoder(s, limit):
            time.sleep(0.5)
            return MagicMock()

        def _slow_zerovoice(s, max_jobs):
            time.sleep(0.5)
            return FetchResult(parsed_total=0, inserted=0, duplicate=0)

        def _slow_verified(*, store, keywords, limit_per_kw):
            time.sleep(0.5)
            return {"inserted_total": 0}

        def _slow_shixiseng(*, store, keywords, limit_per_kw):
            time.sleep(0.5)
            return {"inserted_total": 0}

        monkeypatch.setattr(scout, "crawl_nowcoder", _slow_nowcoder)
        monkeypatch.setattr(
            "offerguide.platforms.zerovoice.crawl_zerovoice", _slow_zerovoice,
        )
        monkeypatch.setattr(
            ambient, "_crawl_verified_official_per_keyword", _slow_verified,
        )
        monkeypatch.setattr(
            ambient, "_crawl_shixiseng_per_keyword", _slow_shixiseng,
        )
        # Stub keyword extraction (returns 2 keywords so verified+shixiseng
        # don't skip)
        from offerguide.match_keywords import KeywordHit

        def _stub_extract(*, store, user_profile_text):
            return [
                KeywordHit(keyword="AI Agent", matched_aliases=(), weight=1),
                KeywordHit(keyword="LLM", matched_aliases=(), weight=1),
            ]

        monkeypatch.setattr(ambient, "_extract_cycle_keywords", _stub_extract)

        # No LLM key → skip agent_search + scoring (we're only timing fetch stages)
        settings = Settings(deepseek_api_key=None)

        t0 = time.monotonic()
        await ambient._run_one_cycle(
            store=store, settings=settings, runtime=None,
            skills=[], user_profile_text="resume", crawl_limit=10,
        )
        elapsed = time.monotonic() - t0
        # 4 stages × 0.5s sequential = 2.0s. Parallel = ~0.5s + thread overhead.
        # Allow up to 1.2s (thread spinup + asyncio.to_thread overhead).
        assert elapsed < 1.2, (
            f"expected parallel fetch (<1.2s), got {elapsed:.2f}s — "
            f"likely sequential"
        )

    @pytest.mark.asyncio
    async def test_keyword_cache_hits_on_second_call(self, store, monkeypatch):
        """W20.2 — _extract_cycle_keywords cache by (resume_hash, goal).
        Same inputs → same list, no recompute."""
        # Reset cache
        ambient._KW_CACHE.clear()

        call_count = {"n": 0}
        from offerguide.match_keywords import KeywordHit
        original = ambient.extract_keywords if hasattr(ambient, "extract_keywords") else None

        # Patch the internal extract_keywords to count calls
        import offerguide.match_keywords as mk
        original_extract = mk.extract_keywords

        def _spy(*args, **kwargs):
            call_count["n"] += 1
            return [KeywordHit(keyword="x", matched_aliases=(), weight=1)]

        monkeypatch.setattr(mk, "extract_keywords", _spy)

        # First call → cache miss → extract_keywords called
        out1 = ambient._extract_cycle_keywords(
            store=store, user_profile_text="resume A",
        )
        # Second call → cache hit → no extra call
        out2 = ambient._extract_cycle_keywords(
            store=store, user_profile_text="resume A",
        )
        # Third call with different resume → cache miss
        out3 = ambient._extract_cycle_keywords(
            store=store, user_profile_text="resume B (different)",
        )
        assert call_count["n"] == 2  # only 2 real extractions (1st + 3rd)
        assert out1 is out2  # same list returned
        assert out1 is not out3  # different cache entry

    def test_keyword_cache_eviction(self, store):
        """Cache bounded to MAX_ENTRIES — oldest dropped when full."""
        ambient._KW_CACHE.clear()
        for i in range(ambient._KW_CACHE_MAX_ENTRIES + 3):
            ambient._extract_cycle_keywords(
                store=store, user_profile_text=f"resume {i}",
            )
        # Cache should have grown to at most _KW_CACHE_MAX_ENTRIES
        assert len(ambient._KW_CACHE) <= ambient._KW_CACHE_MAX_ENTRIES

    @pytest.mark.asyncio
    async def test_one_stage_failure_doesnt_break_cycle(self, store, monkeypatch):
        """If 1 of 4 stages raises, others still complete."""
        from offerguide.platforms.zerovoice import FetchResult
        from offerguide.workers import scout

        def _crash(*args, **kwargs):
            raise RuntimeError("network down")

        ok_calls = {"n": 0}

        def _ok_nowcoder(s, limit):
            ok_calls["n"] += 1
            return MagicMock()

        def _ok_zerovoice(s, max_jobs):
            ok_calls["n"] += 1
            return FetchResult(parsed_total=0, inserted=0, duplicate=0)

        def _ok_shixiseng(*, store, keywords, limit_per_kw):
            ok_calls["n"] += 1
            return {"inserted_total": 0}

        monkeypatch.setattr(scout, "crawl_nowcoder", _ok_nowcoder)
        monkeypatch.setattr(
            "offerguide.platforms.zerovoice.crawl_zerovoice", _ok_zerovoice,
        )
        # verified_official is the one that crashes
        monkeypatch.setattr(
            ambient, "_crawl_verified_official_per_keyword", _crash,
        )
        monkeypatch.setattr(
            ambient, "_crawl_shixiseng_per_keyword", _ok_shixiseng,
        )

        from offerguide.match_keywords import KeywordHit
        monkeypatch.setattr(
            ambient, "_extract_cycle_keywords",
            lambda **k: [KeywordHit(keyword="x", matched_aliases=(), weight=1)],
        )

        await ambient._run_one_cycle(
            store=store, settings=Settings(deepseek_api_key=None),
            runtime=None, skills=[],
            user_profile_text="r", crawl_limit=10,
        )
        # 3 successful stages all ran (verified crashed but didn't break others)
        assert ok_calls["n"] == 3
