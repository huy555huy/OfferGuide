"""Discovery tools — the Discovery sub-agent's hands.

8 fetcher tools wrap existing `platforms/` adapters as registry tools:
- fetch_nowcoder / fetch_tencent_campus / fetch_tencent_social
- fetch_baidu_grad / fetch_baidu_intern / fetch_bytedance
- fetch_zerovoice / fetch_shixiseng

Plus 1 helper tool:
- read_last_fetch_times — sub-agent decides whether to refetch

Each handler:
- Returns JSON: {inserted_count, duplicate_count, source, errors[], by_company{}, ...}
- Handlers catch all exceptions → return tool_error(...)
- Uses runtime_kwargs (store, settings) injected by registry.dispatch
- Per-handler short-lived httpx.Client (per-fetcher API needs one)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import httpx

from .registry import registry, tool_error, tool_result

log = logging.getLogger(__name__)


_HTTP_TIMEOUT_S = 20.0
_HTTP_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


@contextmanager
def _http_client():
    """Short-lived httpx client for one tool call."""
    with httpx.Client(
        timeout=_HTTP_TIMEOUT_S,
        headers={"User-Agent": _HTTP_UA},
        follow_redirects=True,
    ) as c:
        yield c


# ── nowcoder ──────────────────────────────────────────────────────


def _fetch_nowcoder_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    limit = int(args.get("limit") or 15)
    try:
        from ..workers import scout
        result = scout.crawl_nowcoder(store, limit=limit)
        return tool_result(
            source="nowcoder",
            inserted=result.inserted,
            duplicates=result.duplicate,
            errors=result.errors[:5],
        )
    except Exception as e:
        return tool_error(f"nowcoder fetch failed: {type(e).__name__}: {e}")


registry.register(
    name="fetch_nowcoder",
    group="discovery",
    schema={
        "name": "fetch_nowcoder",
        "description": (
            "拉 nowcoder 牛客的最新 JD (经 sitemap chain). "
            "无 keyword 概念, 拿到的是混合行业的最新 JD. "
            "适合: 每天 1 次刷新; 不需要按 keyword 找时用. "
            "返回: inserted (新入库数) / duplicates / errors."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "拉多少个 (default 15, max 30).",
                },
            },
        },
    },
    handler=_fetch_nowcoder_handler,
)


# ── tencent campus + social ───────────────────────────────────────


def _fetch_tencent_campus_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    keyword = args.get("keyword") or "AI"
    limit = int(args.get("limit") or 6)
    try:
        from ..platforms.official_jobs import search_tencent_campus_jobs
        from ..workers import scout
        with _http_client() as client:
            sr = search_tencent_campus_jobs(keyword=keyword, limit=limit, client=client)
        if sr.status != "ok":
            return tool_error(f"tencent_campus: {sr.note}")
        inserted, duplicates = 0, 0
        for rj in sr.jobs:
            rj.extras.setdefault("discovered_via", "verified_official")
            rj.extras.setdefault("discovered_keyword", keyword)
            was_new, _ = scout.ingest(store, rj)
            if was_new:
                inserted += 1
            else:
                duplicates += 1
        return tool_result(
            source="tencent_campus", keyword=keyword,
            inserted=inserted, duplicates=duplicates,
            jobs_returned=len(sr.jobs),
        )
    except Exception as e:
        return tool_error(f"tencent_campus failed: {type(e).__name__}: {e}")


registry.register(
    name="fetch_tencent_campus",
    group="discovery",
    schema={
        "name": "fetch_tencent_campus",
        "description": (
            "拉腾讯校招岗位 (join.qq.com 真公开 JSON API, "
            "projectName='应届实习' 是暑期实习). 按 keyword 搜. "
            "返回 inserted/duplicates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索词 (e.g. 'AI Agent')."},
                "limit": {"type": "integer", "description": "拉多少个 (default 6)."},
            },
            "required": ["keyword"],
        },
    },
    handler=_fetch_tencent_campus_handler,
)


def _fetch_tencent_social_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    keyword = args.get("keyword") or "AI"
    limit = int(args.get("limit") or 6)
    try:
        from ..platforms.official_jobs import search_tencent_social_jobs
        from ..workers import scout
        with _http_client() as client:
            sr = search_tencent_social_jobs(keyword=keyword, limit=limit, client=client)
        if sr.status != "ok":
            return tool_error(f"tencent_social: {sr.note}")
        inserted, duplicates = 0, 0
        for rj in sr.jobs:
            rj.extras.setdefault("discovered_via", "verified_official")
            rj.extras.setdefault("discovered_keyword", keyword)
            was_new, _ = scout.ingest(store, rj)
            if was_new:
                inserted += 1
            else:
                duplicates += 1
        return tool_result(
            source="tencent_social", keyword=keyword,
            inserted=inserted, duplicates=duplicates,
            jobs_returned=len(sr.jobs),
            note="腾讯社招 — 应届生大概率被卡, 慎投",
        )
    except Exception as e:
        return tool_error(f"tencent_social failed: {type(e).__name__}: {e}")


registry.register(
    name="fetch_tencent_social",
    group="discovery",
    schema={
        "name": "fetch_tencent_social",
        "description": (
            "拉腾讯社招岗位 (careers.tencent.com 真 JSON API). "
            "**应届生不该投社招** — 标 SOCIAL 是给完整收集用. "
            "/recommended 默认 ?type=intern filter 会隐藏."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "limit": {"type": "integer", "description": "default 6"},
            },
            "required": ["keyword"],
        },
    },
    handler=_fetch_tencent_social_handler,
)


# ── baidu (校招 + 实习) ──────────────────────────────────────────


def _fetch_baidu_grad_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    keyword = args.get("keyword") or "AI"
    limit = int(args.get("limit") or 6)
    try:
        from ..platforms.official_jobs import search_baidu_campus_jobs
        from ..workers import scout
        with _http_client() as client:
            sr = search_baidu_campus_jobs(
                keyword=keyword, limit=limit, client=client,
                recruit_type="GRADUATE",
            )
        if sr.status != "ok":
            return tool_error(f"baidu_grad: {sr.note}")
        inserted, duplicates = 0, 0
        for rj in sr.jobs:
            rj.extras.setdefault("discovered_via", "verified_official")
            rj.extras.setdefault("discovered_keyword", keyword)
            was_new, _ = scout.ingest(store, rj)
            if was_new:
                inserted += 1
            else:
                duplicates += 1
        return tool_result(
            source="baidu_campus", keyword=keyword,
            inserted=inserted, duplicates=duplicates,
            jobs_returned=len(sr.jobs),
        )
    except Exception as e:
        return tool_error(f"baidu_grad failed: {type(e).__name__}: {e}")


registry.register(
    name="fetch_baidu_grad",
    group="discovery",
    schema={
        "name": "fetch_baidu_grad",
        "description": (
            "拉百度校招正式岗位 (talent.baidu.com recruitType=GRADUATE SSR). "
            "返回的是 '校招' / 'AIDU项目' / '管培生项目'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["keyword"],
        },
    },
    handler=_fetch_baidu_grad_handler,
)


def _fetch_baidu_intern_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    keyword = args.get("keyword") or "AI"
    limit = int(args.get("limit") or 6)
    try:
        from ..platforms.official_jobs import search_baidu_intern_jobs
        from ..workers import scout
        with _http_client() as client:
            sr = search_baidu_intern_jobs(keyword=keyword, limit=limit, client=client)
        if sr.status != "ok":
            return tool_error(f"baidu_intern: {sr.note}")
        inserted, duplicates = 0, 0
        for rj in sr.jobs:
            rj.extras.setdefault("discovered_via", "verified_official")
            rj.extras.setdefault("discovered_keyword", keyword)
            was_new, _ = scout.ingest(store, rj)
            if was_new:
                inserted += 1
            else:
                duplicates += 1
        return tool_result(
            source="baidu_intern", keyword=keyword,
            inserted=inserted, duplicates=duplicates,
            jobs_returned=len(sr.jobs),
        )
    except Exception as e:
        return tool_error(f"baidu_intern failed: {type(e).__name__}: {e}")


registry.register(
    name="fetch_baidu_intern",
    group="discovery",
    schema={
        "name": "fetch_baidu_intern",
        "description": (
            "拉百度实习岗位 (talent.baidu.com recruitType=INTERN SSR). "
            "返回 '暑期实习项目' / '日常实习项目'. 用户暑期实习优先用这个."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["keyword"],
        },
    },
    handler=_fetch_baidu_intern_handler,
)


# ── bytedance (社招 only - empirically verified) ─────────────────


def _fetch_bytedance_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    keyword = args.get("keyword") or "AI Agent"
    limit = int(args.get("limit") or 6)
    try:
        from ..platforms.official_jobs import search_bytedance_jobs
        from ..workers import scout
        with _http_client() as client:
            sr = search_bytedance_jobs(keyword=keyword, limit=limit, client=client)
        if sr.status != "ok":
            return tool_error(f"bytedance: {sr.note}")
        inserted, duplicates = 0, 0
        for rj in sr.jobs:
            rj.extras.setdefault("discovered_via", "verified_official")
            rj.extras.setdefault("discovered_keyword", keyword)
            was_new, _ = scout.ingest(store, rj)
            if was_new:
                inserted += 1
            else:
                duplicates += 1
        return tool_result(
            source="bytedance_jobs", keyword=keyword,
            inserted=inserted, duplicates=duplicates,
            jobs_returned=len(sr.jobs),
            note="字节 API 实测全是 recruit_type='正式' = 社招, 应届实习走 0voice 间接",
        )
    except Exception as e:
        return tool_error(f"bytedance failed: {type(e).__name__}: {e}")


registry.register(
    name="fetch_bytedance",
    group="discovery",
    schema={
        "name": "fetch_bytedance",
        "description": (
            "拉字节 jobs.bytedance.com 公开 JSON API (verified 1334+ 岗). "
            "实测全是社招 (recruit_type='正式'), 应届实习要走 0voice 间接."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["keyword"],
        },
    },
    handler=_fetch_bytedance_handler,
)


# ── zerovoice GitHub aggregator ──────────────────────────────────


def _fetch_zerovoice_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    max_jobs = int(args.get("max_jobs") or 80)
    try:
        from ..platforms.zerovoice import crawl_zerovoice
        result = crawl_zerovoice(store, max_jobs=max_jobs, verify_urls=True)
        return tool_result(
            source="zerovoice_repo",
            parsed_total=result.parsed_total,
            inserted=result.inserted,
            duplicates=result.duplicate,
            skipped_non_ats=result.skipped_non_ats,
            skipped_dead=result.skipped_dead,
            errors=result.errors[:3],
            by_company_top10=dict(sorted(
                result.by_company.items(), key=lambda x: -x[1])[:10]),
        )
    except Exception as e:
        return tool_error(f"zerovoice failed: {type(e).__name__}: {e}")


registry.register(
    name="fetch_zerovoice",
    group="discovery",
    schema={
        "name": "fetch_zerovoice",
        "description": (
            "拉 0voice GitHub repo 聚合 (https://github.com/0voice/"
            "2026-Computer-Spring-Recruitment-Job-Compilation). "
            "100+ 公司, 1000+ 岗位, 每日刷, 含真 ATS URL (含阿里 / 北森 SaaS). "
            "round_robin_by_company 平均分配, 不让一家公司占满 cap."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "max_jobs": {
                    "type": "integer",
                    "description": "拉多少个 (default 80). 公司多样性靠 round-robin 保证.",
                },
            },
        },
    },
    handler=_fetch_zerovoice_handler,
)


# ── shixiseng (实习专用) ─────────────────────────────────────────


def _fetch_shixiseng_handler(args: dict[str, Any], **rt: Any) -> str:
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    keyword = args.get("keyword") or "AI Agent"
    max_jobs = int(args.get("max_jobs") or 8)
    try:
        from ..platforms.shixiseng import crawl_shixiseng
        result = crawl_shixiseng(store, keyword=keyword, max_jobs=max_jobs)
        return tool_result(
            source="shixiseng", keyword=keyword,
            listed_total=result.listed_total,
            fetched=result.fetched,
            inserted=result.inserted,
            duplicates=result.duplicate,
            errors=result.errors[:3],
            by_company=result.by_company,
            note="100% 实习. niche AI 创业公司多 (香巴拉/算力大陆/AKULAKU 等).",
        )
    except Exception as e:
        return tool_error(f"shixiseng failed: {type(e).__name__}: {e}")


registry.register(
    name="fetch_shixiseng",
    group="discovery",
    schema={
        "name": "fetch_shixiseng",
        "description": (
            "拉实习僧 (shixiseng.com) — 国内大学生实习首选聚合. "
            "100% 实习, niche AI 创业公司多. 按 keyword. "
            "每 keyword 拿 N 个 detail page (cap 12)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "max_jobs": {"type": "integer", "description": "default 8"},
            },
            "required": ["keyword"],
        },
    },
    handler=_fetch_shixiseng_handler,
)


# ── helper: read_last_fetch_times ────────────────────────────────


def _read_last_fetch_times_handler(args: dict[str, Any], **rt: Any) -> str:
    """Show when each source was last fetched. Sub-agent uses this to
    decide whether to refetch (e.g., skip nowcoder if pulled <1h ago)."""
    store = rt.get("store")
    if store is None:
        return tool_error("internal: store not provided")
    try:
        with store.connect() as conn:
            # created_at is REAL (julianday) — MAX directly, no julianday() cast
            rows = conn.execute(
                "SELECT source, "
                "       MAX(created_at) as last_julian, "
                "       COUNT(*) as total, "
                "       (julianday('now') - MAX(created_at)) * 24 as hours_ago "
                "FROM jobs WHERE source IS NOT NULL "
                "GROUP BY source"
            ).fetchall()
        out: dict[str, dict[str, Any]] = {}
        for src, _last_julian, total, hours_ago in rows:
            out[src] = {
                "total_jobs": int(total),
                "hours_since_last": (
                    round(float(hours_ago), 1) if hours_ago is not None else None
                ),
            }
        return tool_result(by_source=out, note=(
            "用这个决定 refetch 还是跳过. "
            "建议: hours_since_last > 6 才 refetch."
        ))
    except Exception as e:
        return tool_error(f"read_last_fetch failed: {type(e).__name__}: {e}")


registry.register(
    name="read_last_fetch_times",
    group="discovery",
    schema={
        "name": "read_last_fetch_times",
        "description": (
            "看每个 source 上次抓岗时间 + 已入库 job 数. "
            "Sub-agent 决定 refetch 还是跳过. "
            "返回 {by_source: {source: {total_jobs, hours_since_last}}}."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    handler=_read_last_fetch_times_handler,
)
