"""W20.2 audit — real benchmark of ambient.py refactor.

Measures real speedup of:
1. _score_jobs_blocking — parallel vs sequential for N jobs
2. _run_one_cycle — 4 fetch stages parallel vs sequential

Uses sleep() to simulate LLM latency, so the audit doesn't need a real
LLM key (and stays reproducible).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

os.environ["OFFERGUIDE_NO_AMBIENT"] = "1"

import offerguide
from offerguide.config import Settings
from offerguide.harness import _schema as harness_schema
from offerguide.workers import ambient


def setup_store():
    tmp = Path(tempfile.mkdtemp(prefix="ogfd_w20_2_perf_"))
    db = tmp / "store.db"
    s = offerguide.Store(db)
    s.init_schema()
    harness_schema.init_harness_schema(s)
    return s, db


def insert_jobs(store, n: int) -> list[int]:
    ids = []
    with store.connect() as conn:
        for i in range(n):
            cur = conn.execute(
                "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                ("nowcoder", f"job{i}", f"co{i}", "x" * 300, f"h{i}"),
            )
            ids.append(cur.lastrowid or 0)
        conn.commit()
    return ids


# ── 1. score_match parallel benchmark ──────────────────────────────────

print("=" * 70)
print("Benchmark 1: _score_jobs_blocking parallel vs sequential")
print("=" * 70)

LATENCY_PER_JOB = 0.4  # seconds (simulates ~5-10s real LLM, scaled down)
N_JOBS = 12  # realistic multi-source cycle

import offerguide.harness.tools as _tools_mod
import offerguide.llm as _llm_mod

original_score_match = _tools_mod._exec_score_match
original_budget = _llm_mod.enforce_daily_budget


def _stub_score(args, deps):
    time.sleep(LATENCY_PER_JOB)
    return f"OK score_match for job#{args.get('job_id')}"


_tools_mod._exec_score_match = _stub_score
_llm_mod.enforce_daily_budget = lambda s: None

settings = Settings(deepseek_api_key="sk-fake")
runtime = MagicMock()
skills = [MagicMock(name="score_match")]

results: list[tuple[int, float, float]] = []
for parallelism in [1, 2, 4, 8]:
    store, _db = setup_store()
    job_ids = insert_jobs(store, N_JOBS)
    # Re-init to avoid state leakage
    t0 = time.monotonic()
    ambient._score_jobs_blocking(
        store=store, settings=settings, runtime=runtime, skills=skills,
        user_profile_text="resume", job_ids=job_ids, parallelism=parallelism,
    )
    elapsed = time.monotonic() - t0
    theoretical_seq = N_JOBS * LATENCY_PER_JOB
    speedup = theoretical_seq / elapsed
    results.append((parallelism, elapsed, speedup))
    print(f"  parallelism={parallelism:>2}: {elapsed:5.2f}s "
          f"(theoretical seq={theoretical_seq:.1f}s, speedup={speedup:.1f}x)")

# Restore
_tools_mod._exec_score_match = original_score_match
_llm_mod.enforce_daily_budget = original_budget

best_par = max(results, key=lambda r: r[2])
seq_baseline = next(r for r in results if r[0] == 1)
print(f"\nReal speedup (parallelism=4 vs 1): {seq_baseline[1] / results[2][1]:.1f}x")


# ── 2. _run_one_cycle 4-fetch parallel benchmark ────────────────────────

print("\n" + "=" * 70)
print("Benchmark 2: _run_one_cycle 4 fetch stages parallel via gather")
print("=" * 70)

FETCH_LATENCY = 0.6  # simulates ~5-10s real network


def _stub_nowcoder(s, limit):
    time.sleep(FETCH_LATENCY)
    return MagicMock()


def _stub_zerovoice(s, max_jobs):
    time.sleep(FETCH_LATENCY)
    from offerguide.platforms.zerovoice import FetchResult
    return FetchResult(parsed_total=0, inserted=0, duplicate=0)


def _stub_verified(*, store, keywords, limit_per_kw):
    time.sleep(FETCH_LATENCY)
    return {"inserted_total": 0}


def _stub_shixiseng(*, store, keywords, limit_per_kw):
    time.sleep(FETCH_LATENCY)
    return {"inserted_total": 0}


from offerguide.match_keywords import KeywordHit
from offerguide.workers import scout
import offerguide.platforms.zerovoice as _zv

original_nowcoder = scout.crawl_nowcoder
original_zerovoice = _zv.crawl_zerovoice
original_verified = ambient._crawl_verified_official_per_keyword
original_shixiseng = ambient._crawl_shixiseng_per_keyword
original_extract = ambient._extract_cycle_keywords

scout.crawl_nowcoder = _stub_nowcoder
_zv.crawl_zerovoice = _stub_zerovoice
ambient._crawl_verified_official_per_keyword = _stub_verified
ambient._crawl_shixiseng_per_keyword = _stub_shixiseng
ambient._extract_cycle_keywords = lambda **kw: [
    KeywordHit(keyword="AI Agent", matched_aliases=(), weight=1),
    KeywordHit(keyword="LLM", matched_aliases=(), weight=1),
]

store, _db = setup_store()

t0 = time.monotonic()
asyncio.run(ambient._run_one_cycle(
    store=store, settings=Settings(deepseek_api_key=None),
    runtime=None, skills=[], user_profile_text="resume", crawl_limit=10,
))
parallel_elapsed = time.monotonic() - t0

# Restore
scout.crawl_nowcoder = original_nowcoder
_zv.crawl_zerovoice = original_zerovoice
ambient._crawl_verified_official_per_keyword = original_verified
ambient._crawl_shixiseng_per_keyword = original_shixiseng
ambient._extract_cycle_keywords = original_extract

theoretical_seq = 4 * FETCH_LATENCY
speedup_cycle = theoretical_seq / parallel_elapsed
print(f"  4 stages × {FETCH_LATENCY}s = {theoretical_seq:.1f}s sequential")
print(f"  parallel via asyncio.gather: {parallel_elapsed:.2f}s")
print(f"  Real speedup: {speedup_cycle:.1f}x")


# ── 3. Combined improvement estimate ───────────────────────────────────

print("\n" + "=" * 70)
print("Combined production estimate (typical 6h cycle):")
print("=" * 70)

# Real numbers from W20 dogfood:
# - nowcoder: ~30s (15 jobs)
# - 0voice: ~1s (cached, 80 jobs)
# - verified × 5 keywords: ~25s (28 jobs)
# - shixiseng × 2 keywords: ~7s (16 jobs)
# - agent_search: ~174s (40 jobs) → with W20.2 cap=8 iter ≈ 60s
# - score_match: 30 jobs × 6s LLM avg = 180s sequential

# Old (sequential fetch + sequential score):
old_fetch = 30 + 1 + 25 + 7  # 63s
old_search = 174  # original 19 iter
old_score = 30 * 6  # 180s
old_total = old_fetch + old_search + old_score
print(f"  PRE-W20.2 (sequential fetch + sequential score + 19-iter agent_search):")
print(f"    fetch={old_fetch}s + search={old_search}s + score={old_score}s = {old_total}s")

# New (parallel fetch + parallel score with parallelism=4 + capped agent):
new_fetch = max(30, 1, 25, 7)  # 30s (longest of the 4)
new_search = 60  # capped to ~8 iter
new_score = (30 / 4) * 6 + 1  # ~46s with 4-way parallelism
new_total = new_fetch + new_search + new_score
print(f"  POST-W20.2 (parallel fetch + parallel score parallelism=4 + cap=8):")
print(f"    fetch={new_fetch}s + search={new_search}s + score={new_score:.0f}s = {new_total:.0f}s")

print(f"\n  Cycle time: {old_total}s → {new_total:.0f}s "
      f"({old_total / new_total:.1f}x faster)")
print(f"  Per-day cost (4 cycles): {old_total * 4 / 60:.0f}min → {new_total * 4 / 60:.0f}min CPU/day")

print("\n" + "=" * 70)
print("ALL BENCHMARKS COMPLETE")
print("=" * 70)
sys.exit(0)
