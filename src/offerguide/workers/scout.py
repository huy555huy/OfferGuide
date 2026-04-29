"""Scout — pulls JDs from platform adapters into the local jobs table.

Two pull modes today:

- `crawl_nowcoder(store, limit=N)` — walk the public sitemap chain, fetch each
  JD page, parse, dedup, insert. Sequential with a polite 1s gap between fetches.
- `ingest(store, raw_job)` — write any platform's `RawJob` (e.g. from manual paste).

Dedup is by `content_hash(rj)` against the `jobs.content_hash` UNIQUE constraint —
re-running the crawl is safe and cheap.

W2 keeps Scout invokable as a script. APScheduler-driven background scheduling
lands in W7 once the inbox UX exists.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from ..memory import Store
from ..platforms import RawJob, content_hash, nowcoder

log = logging.getLogger(__name__)


def ingest(store: Store, rj: RawJob) -> tuple[bool, int]:
    """Insert one job. Returns (was_new, job_id).

    Idempotent: re-ingesting an identical RawJob is a no-op (returns the
    existing row's id and `was_new=False`).

    `rj.extras` is persisted into the structured `jobs.extras_json` column —
    NOT concatenated into `raw_text`. This is the W5' fix: previously the
    extras blob was glued onto raw_text, which (a) polluted the LLM context
    when SKILLs read jobs.raw_text and (b) made platform-native fields
    (e.g. nowcoder's `avgProcessRate` — actual platform-measured reply rate)
    unqueryable as structured data. The dedup hash is still computed over the
    canonical text only, so two scrapes with identical JD content but
    different extras still dedup correctly.
    """
    h = content_hash(rj)
    extras_payload = json.dumps(rj.extras or {}, ensure_ascii=False)

    with store.connect() as conn:
        existing = conn.execute(
            "SELECT id FROM jobs WHERE source = ? AND content_hash = ?",
            (rj.source, h),
        ).fetchone()
        if existing:
            return False, existing[0]

        cur = conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, company, location, "
            "raw_text, extras_json, content_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                rj.source,
                rj.source_id,
                rj.url,
                rj.title,
                rj.company,
                rj.location,
                rj.raw_text,
                extras_payload,
                h,
            ),
        )
        return True, int(cur.lastrowid or 0)


def crawl_nowcoder(
    store: Store,
    *,
    limit: int | None = None,
    client: nowcoder.NowcoderClient | None = None,
) -> dict[str, int]:
    """Walk nowcoder's sitemap chain, fetch each JD, dedup, insert.

    Returns counts: {discovered, fetched, ingested_new, dup}.
    `limit` caps how many JD URLs to actually fetch this run (None = no cap).
    """
    own_client = client or nowcoder.NowcoderClient()
    counters = {"discovered": 0, "fetched": 0, "ingested_new": 0, "dup": 0, "errors": 0}
    try:
        for i, url in enumerate(_capped(nowcoder.iter_jd_urls(own_client), limit)):
            counters["discovered"] += 1
            try:
                rj = nowcoder.fetch_and_parse(own_client, url)
            except Exception as e:
                # Per-URL failure (network blip, parse change) shouldn't kill the crawl.
                log.warning("nowcoder fetch failed for %s: %s", url, e)
                counters["errors"] += 1
                continue
            counters["fetched"] += 1
            was_new, _ = ingest(store, rj)
            if was_new:
                counters["ingested_new"] += 1
            else:
                counters["dup"] += 1
            if (i + 1) % 10 == 0:
                log.info("scout nowcoder progress: %s", counters)
    finally:
        if client is None:
            own_client.close()
    return counters


def _capped(iterable: Iterator[str], limit: int | None) -> Iterator[str]:
    if limit is None:
        yield from iterable
        return
    for i, item in enumerate(iterable):
        if i >= limit:
            return
        yield item
