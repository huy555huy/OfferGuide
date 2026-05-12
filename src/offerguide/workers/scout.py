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


# ── Off-topic filter ──────────────────────────────────────────────────
#
# nowcoder's sitemap is mixed-industry — alongside the AI / 数据 / 软件 jobs
# our user actually wants, it carries 制造业 / 餐饮 / 服务业 / 销售 etc.
# (real example from dogfood: a 兰州 玻尿酸装盒 普工岗 surfaced as a
# top-of-feed "未评分" card and made the recommended list look broken).
#
# This is a coarse blacklist, NOT a positive-match filter — we don't want
# to reject "数据分析师 实习" just because it doesn't say "AI". Anything that
# survives lands in jobs/ and score_match decides if it's a good fit per
# the user's CV. We only intercept the obvious factory / service-industry
# noise that has zero chance of fitting any tech-graduate profile.
#
# Each token below is matched as a substring in title + raw_text. If ANY
# match, the row is rejected at ingest time.
_OBVIOUS_OFFTOPIC_TOKENS: tuple[str, ...] = (
    # Factory / 蓝领
    "坐岗", "长白班", "普工", "厂工", "操作工", "装配工",
    "流水线", "包吃住", "倒班", "夜班", "焊工", "钳工",
    "电工", "车工", "贴标", "贴面膜", "包装工", "拣货",
    "分拣员", "理货员",
    # Service / 服务业
    "保洁", "保安", "门卫", "收银员", "服务员", "迎宾",
    "传菜", "厨师", "学徒", "送餐员", "外卖员", "骑手",
    "配送员", "快递员", "司机", "代驾", "美容", "美甲",
    "理发", "按摩", "足疗", "客房", "导购", "促销员",
    # Sales / 中介 / 直播
    "电销", "电话销售", "房产中介", "保险代理", "贷款专员",
    "信用卡专员", "微商", "直播带货", "网红主播",
    # Misc obvious-mismatch
    "幼师", "小学老师", "瑜伽教练", "健身教练",
)


def _is_obviously_offtopic(rj: RawJob) -> tuple[bool, str]:
    """Coarse pre-ingest filter for clearly mismatched jobs.

    Returns ``(reject, matched_token)``. matched_token is empty when not
    rejecting.

    **Title-only match** is intentional: factory / service / 销售 keywords
    naturally appear in the *title* of those jobs ("坐岗长白班", "电话销售").
    A previous version matched raw_text too and false-positive'd real
    targets ("云游戏-Agent应用工程师" was rejected because its JD body
    mentioned 流水线 in passing). The job's title is the cleaner signal.
    """
    title = rj.title or ""
    for tok in _OBVIOUS_OFFTOPIC_TOKENS:
        if tok in title:
            return True, tok
    return False, ""


def ingest(store: Store, rj: RawJob) -> tuple[bool, int]:
    """Insert one job. Returns (was_new, job_id).

    Idempotent: re-ingesting an identical RawJob is a no-op (returns the
    existing row's id and `was_new=False`).

    Pre-ingest filter (W21 follow-up): rows obviously not for a tech-graduate
    user (factory / service / 销售 / etc.) are rejected before they pollute
    the recommendations feed. ``(False, 0)`` is returned in that case so
    callers can count rejections without raising.

    `rj.extras` is persisted into the structured `jobs.extras_json` column —
    NOT concatenated into `raw_text`. This is the W5' fix: previously the
    extras blob was glued onto raw_text, which (a) polluted the LLM context
    when SKILLs read jobs.raw_text and (b) made platform-native fields
    (e.g. nowcoder's `avgProcessRate` — actual platform-measured reply rate)
    unqueryable as structured data. The dedup hash is still computed over the
    canonical text only, so two scrapes with identical JD content but
    different extras still dedup correctly.
    """
    reject, matched = _is_obviously_offtopic(rj)
    if reject:
        log.info(
            "scout.ingest: rejected offtopic job (matched=%r) "
            "source=%s url=%s title=%r",
            matched, rj.source, rj.url, (rj.title or "")[:60],
        )
        return False, 0

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
            # W18 — discovered_via attribution so /recommended cards can show
            # "↳ 由 nowcoder sitemap 找到". Set BEFORE ingest so it lands in
            # extras_json. Don't overwrite if the parser already set it.
            rj.extras.setdefault("discovered_via", "nowcoder_sitemap")
            rj.extras.setdefault("discovered_keyword", "(sitemap walk, no keyword)")
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
