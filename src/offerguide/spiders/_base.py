"""Spider base + shared HTTP helpers.

Design decisions:

- **Synchronous httpx**, not async. Spiders run in cron jobs; concurrent
  per-host pressure on a single Chinese job board (which often has tight
  rate limits) is a bad-citizen pattern. Sequential + polite delay >>
  parallel.
- **Rate-limited GET helper**: every spider goes through ``rate_limited_get``,
  which enforces a per-host minimum-gap between requests (default 2.5s).
  No spider can accidentally hammer.
- **User-Agent rotation off by default**: Most Chinese boards see UA
  rotation as a hostile signal. A static, honest UA ("OfferGuide/1.0")
  + low rate >>> rotating fake browser UAs at high rate.
- **Cache-aware**: spiders check `seen_url` against existing
  ``jobs.url`` to avoid re-scraping JD detail pages. The listing page
  itself is always re-fetched (that's the whole point of the cron).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

from ..platforms import RawJob

USER_AGENT = "OfferGuide/1.0 (+local-first job-hunt copilot; contact via repo)"
"""Identifies our crawler honestly. Most boards prefer this over fake-Chrome
UA strings; a stable UA also helps when boards eventually rate-limit per UA."""

DEFAULT_PER_HOST_GAP_S = 2.5
"""Minimum seconds between consecutive requests to the same host. 2.5s is
slow enough that no Chinese job board has ever rate-limited us; faster
crawls go via paid APIs, not screen-scraping."""


class SpiderError(RuntimeError):
    """Raised by spiders for transport / parse failures.

    Spider runners catch this so one broken spider doesn't kill the daemon
    sweep. The daemon logs the error and moves to the next spider.
    """


@dataclass
class SpiderResult:
    """Summary returned by ``Spider.run`` so the daemon can log + decide alerts.

    ``raw_jobs`` are the new candidates the spider found. Scout-side dedup
    (by ``content_hash``) means re-yielding an existing job is harmless;
    the count returned here is "candidates", not "newly inserted".
    """
    spider_name: str
    raw_jobs: list[RawJob] = field(default_factory=list)
    pages_fetched: int = 0
    errors: list[str] = field(default_factory=list)
    """Soft errors — pages skipped due to parse failures, not fatal."""

    @property
    def ok(self) -> bool:
        return not self.errors or len(self.raw_jobs) > 0


@runtime_checkable
class Spider(Protocol):
    """Discovery spider — produces ``RawJob`` candidates for scout to ingest.

    Implementations should:

    1. Be idempotent — re-running the spider with the same DB state should
       produce the same set of candidates (modulo upstream changes).
    2. Be polite — go through ``rate_limited_get`` for HTTP.
    3. Tolerate partial failure — wrap each item parse in try/except and
       record errors in :class:`SpiderResult` rather than raising.
    """

    name: str
    """Stable string id used as ``RawJob.source`` and in logs."""

    def run(self, *, max_items: int = 30) -> SpiderResult:
        """Crawl up to ``max_items`` listings and return the result."""
        ...


# ─────────── shared HTTP helpers ──────────────────────────────────


_HOST_LAST_REQUEST_AT: dict[str, float] = {}
_LOCK = Lock()


def rate_limited_get(
    url: str,
    *,
    timeout_s: float = 10.0,
    per_host_gap_s: float = DEFAULT_PER_HOST_GAP_S,
    extra_headers: dict[str, str] | None = None,
) -> httpx.Response:
    """GET with per-host rate limiting and an honest UA.

    Sleeps before the request if the previous request to the same host
    was less than ``per_host_gap_s`` ago. Thread-safe; the daemon never
    runs spiders concurrently anyway, but multiple spiders for the same
    host (rare) are still polite.

    Raises :class:`SpiderError` on transport failure or non-2xx status.
    """
    host = urlparse(url).netloc.lower()

    with _LOCK:
        last = _HOST_LAST_REQUEST_AT.get(host, 0.0)
        now = time.monotonic()
        wait = max(0.0, per_host_gap_s - (now - last))
        if wait > 0:
            time.sleep(wait)
        _HOST_LAST_REQUEST_AT[host] = time.monotonic()

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"}
    if extra_headers:
        headers.update(extra_headers)
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout_s, follow_redirects=True)
    except httpx.HTTPError as e:
        raise SpiderError(f"transport error fetching {url}: {e}") from e
    if resp.status_code >= 400:
        raise SpiderError(
            f"HTTP {resp.status_code} fetching {url}: {resp.text[:200]}"
        )
    return resp


def reset_host_throttle_for_tests() -> None:
    """Wipe the per-host throttle table — used by tests, not by app code."""
    with _LOCK:
        _HOST_LAST_REQUEST_AT.clear()


_ = Iterator  # quiet unused import warning
