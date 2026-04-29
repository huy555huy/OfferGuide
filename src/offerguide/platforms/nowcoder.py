"""nowcoder.com platform adapter.

Discovery: nowcoder publishes a public, robots-allowed sitemap chain
(`/sitemap.xml` → `nowpick/sitemap1.xml` → individual `/jobs/detail/<id>` URLs).
We follow that chain to enumerate JD URLs without scraping search pages.

Parsing: each JD page is a SPA but ships its full server-state as
`window.__INITIAL_STATE__ = {...};` inside a `<script>` tag. We extract that
JSON instead of touching the rendered DOM — far more stable than CSS-selector
scraping. Schema observed on 2026-04-28 from `/jobs/detail/446211`:

    store.jobDetail.detail = {
        id, jobName, jobCity, salaryMin, salaryMax, salaryMonth,
        careerJobName, industryName, ext: "<json string>", ...
    }

    ext = {requirements, infos}  # requirements + responsibilities

This module is NETWORK-ONLY at `fetch_*` boundaries; everything else is pure
functions over strings, so most behavior is testable from a fixture HTML file.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from ._spec import RawJob

NAME = "nowcoder"
ROOT_SITEMAP = "https://www.nowcoder.com/sitemap.xml"
NOWPICK_SITEMAP = "https://www.nowcoder.com/sitemap/nowpick/sitemap.xml"

USER_AGENT = "OfferGuide/0.0 (+https://github.com/hu-yang/offerguide)"

# `/jobs/detail/<id>` — the canonical JD URL pattern.
JD_URL_RE = re.compile(r"^https://www\.nowcoder\.com/jobs/detail/(\d+)")

_INITIAL_STATE_RE = re.compile(r"window\.__INITIAL_STATE__\s*=\s*", re.MULTILINE)
_LOC_RE = re.compile(r"<loc>([^<]+)</loc>")


@dataclass
class NowcoderClient:
    """Thin polite-fetch wrapper. One instance per Scout run.

    `min_interval_s` is enforced between successive `fetch()` calls to be a good
    citizen — nowcoder doesn't rate-limit small bursts but we still don't want
    to be the reason they tighten things.
    """

    timeout_s: float = 15.0
    min_interval_s: float = 1.0
    user_agent: str = USER_AGENT
    _last_fetch_at: float = 0.0

    def __post_init__(self) -> None:
        self._http = httpx.Client(
            timeout=self.timeout_s,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        )

    def fetch(self, url: str) -> str:
        gap = time.monotonic() - self._last_fetch_at
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)
        try:
            resp = self._http.get(url)
            resp.raise_for_status()
            return resp.text
        finally:
            self._last_fetch_at = time.monotonic()

    def close(self) -> None:
        self._http.close()


# ---------- pure parsing (no network) -------------------------------------


def parse_sitemap_locs(xml: str) -> list[str]:
    """Return all `<loc>` entries from a sitemap XML — works for both index and urlset."""
    return _LOC_RE.findall(xml)


def filter_jd_urls(urls: list[str]) -> list[str]:
    """Keep only `/jobs/detail/<id>` URLs (drop enterprise / discuss / etc.)."""
    return [u for u in urls if JD_URL_RE.match(u)]


def jd_url_to_id(url: str) -> str | None:
    m = JD_URL_RE.match(url)
    return m.group(1) if m else None


def parse_jd_html(html: str, url: str | None = None) -> RawJob:
    """Extract a `RawJob` from one JD page's HTML.

    Reads `window.__INITIAL_STATE__` → `store.jobDetail.detail` (and its
    embedded `ext` JSON). Raises `ValueError` if the SPA shape changed.
    """
    m = _INITIAL_STATE_RE.search(html)
    if not m:
        raise ValueError("nowcoder JD page: window.__INITIAL_STATE__ not found")

    decoder = json.JSONDecoder()
    try:
        state, _consumed = decoder.raw_decode(html[m.end() :])
    except json.JSONDecodeError as e:
        raise ValueError(f"nowcoder JD page: __INITIAL_STATE__ is not valid JSON: {e}") from e

    detail = state.get("store", {}).get("jobDetail", {}).get("detail")
    if not detail:
        raise ValueError("nowcoder JD page: store.jobDetail.detail missing")

    ext_raw = detail.get("ext") or "{}"
    try:
        ext = json.loads(ext_raw) if isinstance(ext_raw, str) else dict(ext_raw)
    except json.JSONDecodeError:
        ext = {}

    title = str(detail.get("jobName") or "").strip()
    city = detail.get("jobCity")
    requirements = ext.get("requirements", "").strip()
    responsibilities = ext.get("infos", "").strip()

    salary_min = detail.get("salaryMin")
    salary_max = detail.get("salaryMax")
    salary_months = detail.get("salaryMonth")
    salary_line = ""
    if salary_min is not None and salary_max is not None:
        m_part = f" × {salary_months}" if salary_months else ""
        salary_line = f"薪资: {salary_min}-{salary_max}K{m_part}"

    raw_text_parts = [
        f"职位名: {title}" if title else "",
        f"城市: {city}" if city else "",
        salary_line,
        f"职业方向: {detail.get('careerJobName')}" if detail.get("careerJobName") else "",
        f"行业: {detail.get('industryName')}" if detail.get("industryName") else "",
        f"届别: {detail.get('graduationYear')}" if detail.get("graduationYear") else "",
        "",
        "## 岗位职责",
        responsibilities,
        "",
        "## 任职要求",
        requirements,
    ]
    raw_text = "\n".join(p for p in raw_text_parts if p != "").strip()

    company = (
        state.get("store", {}).get("jobDetail", {}).get("jobCompany", {}).get("companyName")
    )

    extras: dict = {}
    for k in (
        "id",
        "salaryMin",
        "salaryMax",
        "salaryMonth",
        "careerJobName",
        "industryName",
        "companyFinancing",
        "graduationYear",
        "avgProcessRate",  # 平台实测回复率 — 后续 reply rate prior 用
        "avgProcessDay",
        "jobKeys",
    ):
        if detail.get(k) is not None:
            extras[k] = detail[k]

    return RawJob(
        source=NAME,
        source_id=str(detail.get("id")) if detail.get("id") is not None else None,
        url=url,
        title=title or "(untitled)",
        company=company,
        location=city,
        raw_text=raw_text,
        extras=extras,
    )


# ---------- network-touching helpers --------------------------------------


def iter_jd_urls(client: NowcoderClient) -> Iterator[str]:
    """Yield every JD URL discoverable through nowcoder's sitemap chain.

    Walks: root sitemap → sub-sitemaps → urlset → filter to `/jobs/detail/<id>`.
    """
    seen: set[str] = set()

    sub_sitemaps: list[str] = parse_sitemap_locs(client.fetch(NOWPICK_SITEMAP))
    for sub in sub_sitemaps:
        if sub in seen:
            continue
        seen.add(sub)
        urls = parse_sitemap_locs(client.fetch(sub))
        yield from filter_jd_urls(urls)


def fetch_and_parse(client: NowcoderClient, url: str) -> RawJob:
    return parse_jd_html(client.fetch(url), url=url)
