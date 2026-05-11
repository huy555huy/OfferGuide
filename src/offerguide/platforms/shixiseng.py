"""W20 — 实习僧 (shixiseng.com) 实习岗位 adapter.

Why this source:
- 实习僧是国内最大实习专用聚合, 主流大学生找暑期/日常实习的入口.
- 用户当前最高优先级是 2026 暑期实习 (CLAUDE.md 第 2 节).
- W19+ 0voice repo 主打校招混实习 (75% 校招), nowcoder sitemap 实习占比也低,
  baidu_intern 只 10 个百度自己的. 实习专用源缺口明显.

Empirically observed format (probed 2026-05-11):

  List page: ``https://www.shixiseng.com/interns?keyword=AI&page=N``
  - SSR Nuxt page, 但是 list view 的字段全在 ``window.__NUXT__`` 的 minified
    IIFE 里, 而且 title/salary/city 用了 font-encoded 私有 unicode 反爬
    (``cutom_font`` class). 普通 regex 拿不到真字符.
  - **唯一 reliable 的 list 数据**: ``/intern/<inn_xxxxxxxx>`` href 链接,
    20 个/page, 真清楚地暴露在 SSR HTML 里.

  Detail page: ``https://www.shixiseng.com/intern/<inn_xxx>``
  - 真 SSR + 真字符, 不再 font-obfuscate. 可解析:
    - ``<title>`` tag: ``<岗位名>-<公司>实习生招聘-实习僧``
    - ``new_job_name`` div: 岗位名
    - ``com-name`` link: 公司名
    - ``job_position`` span ``title=""``: 城市
    - ``job_money`` div: 薪资 (格式 "100-200/天" 等)
    - ``job_date`` span: 刷新日期
    - ``job_detail`` div: 完整 JD body (岗位职责 + 任职要求)

Adapter strategy:
1. Fetch list page → parse 20 个 inn_xxx token (cheap, 1 req)
2. For each token, fetch detail page → parse真字段 (1 req each)
3. 21 reqs/keyword/cycle. Conservative cap (default max_jobs=20) so一个 cycle
   一个 keyword 一次 fetch 之后 enough.

Verified status: SOURCE_LANDSCAPE 标 ``verified_intern_aggregator``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from ._spec import RawJob

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

LIST_URL = "https://www.shixiseng.com/interns"
DETAIL_URL_TEMPLATE = "https://www.shixiseng.com/intern/{token}"
EVIDENCE_URL_LIST = "https://www.shixiseng.com/interns?keyword=<kw>"

# inn_xxxxxxxx token pattern
_TOKEN_RE = re.compile(r"/intern/(inn_[a-z0-9]{8,})")

# Detail page parse regexes — keyed by class name from probed HTML.
_DETAIL_TITLE_TAG_RE = re.compile(r"<title>(.+?)</title>", re.DOTALL)
# Real <title> pattern (probed 2026-05-11):
#   "<岗位名>实习招聘-<公司>实习生招聘-实习僧"
# 例: "ai动画师实习招聘-耀漫文化实习生招聘-实习僧"
# Falls back gracefully if 实习招聘/实习生招聘/实习僧 suffix isn't there.
_TITLE_SPLIT_RE = re.compile(
    r"^(.+?)实习招聘-(.+?)实习生招聘-实习僧$",
)

_NEW_JOB_NAME_RE = re.compile(
    r'<div class="new_job_name"[^>]*>\s*<span[^>]*>(.+?)</span>',
    re.DOTALL,
)
_COM_NAME_RE = re.compile(
    r'<a[^>]*class="com-name"[^>]*>\s*(.+?)\s*</a>',
    re.DOTALL,
)
_JOB_POSITION_RE = re.compile(
    r'<span[^>]*title="([^"]+?)"[^>]*class="job_position"',
)
_JOB_MONEY_RE = re.compile(
    r'<span[^>]*class="job_money[^"]*"[^>]*>(.+?)</span>',
    re.DOTALL,
)
_JOB_DATE_RE = re.compile(
    r'<span[^>]*class="cutom_font"[^>]*>([\d\-:\s]+)</span>',
)
_JOB_DETAIL_RE = re.compile(
    r'<div[^>]*class="job_detail"[^>]*>\s*(.+?)\s*</div>',
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedDetail:
    token: str
    url: str
    title: str
    company: str | None
    location: str | None
    salary: str | None
    refresh_date: str | None
    body: str
    """Full JD body (岗位职责 + 任职要求 + 学历 + 实习要求 stripped of HTML)."""


@dataclass
class FetchResult:
    listed_total: int = 0
    fetched: int = 0
    parsed: int = 0
    inserted: int = 0
    duplicate: int = 0
    errors: list[str] = None  # type: ignore[assignment]
    by_company: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.by_company is None:
            self.by_company = {}


def fetch_list_tokens(
    keyword: str, page: int = 1, *, timeout_s: float = 20.0,
    client: httpx.Client | None = None,
) -> list[str]:
    """List page → unique intern tokens. ~20 per page."""
    url = LIST_URL
    params = {"keyword": keyword, "page": page}
    if client is None:
        with httpx.Client(
            timeout=timeout_s, headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as c:
            r = c.get(url, params=params)
            r.raise_for_status()
            html = r.text
    else:
        r = client.get(url, params=params)
        r.raise_for_status()
        html = r.text

    # dedup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for m in _TOKEN_RE.finditer(html):
        t = m.group(1)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def parse_detail(html: str, *, token: str) -> ParsedDetail | None:
    """Parse one detail-page HTML → ParsedDetail. None on parse failure."""
    # Title from <title>: <岗位名>-<公司>实习生招聘-实习僧
    title = company = None
    m = _DETAIL_TITLE_TAG_RE.search(html)
    if m:
        title_blob = _strip_html_entities(m.group(1).strip())
        sm = _TITLE_SPLIT_RE.match(title_blob)
        if sm:
            title = sm.group(1).strip()
            company = sm.group(2).strip()

    # Fallback: new_job_name div if title-tag parse failed
    if not title:
        m = _NEW_JOB_NAME_RE.search(html)
        if m:
            title = _strip_html_entities(_collapse_ws(m.group(1)))

    # Fallback: com-name link if title-tag parse missed company
    if not company:
        m = _COM_NAME_RE.search(html)
        if m:
            company = _strip_html_entities(_collapse_ws(m.group(1)))

    if not title:
        return None

    # Location from job_position title attr
    location = None
    m = _JOB_POSITION_RE.search(html)
    if m:
        location = _strip_html_entities(m.group(1).strip())

    # Salary from job_money div (format "100-200/天" / "1.5k-3k/月" etc.)
    salary = None
    m = _JOB_MONEY_RE.search(html)
    if m:
        salary = _strip_html_entities(_collapse_ws(m.group(1)))

    # Refresh date
    refresh_date = None
    m = _JOB_DATE_RE.search(html)
    if m:
        refresh_date = m.group(1).strip()

    # JD body
    body = ""
    m = _JOB_DETAIL_RE.search(html)
    if m:
        body = _strip_html_entities(_collapse_ws(_strip_tags(m.group(1))))

    return ParsedDetail(
        token=token,
        url=DETAIL_URL_TEMPLATE.format(token=token),
        title=title,
        company=company,
        location=location,
        salary=salary,
        refresh_date=refresh_date,
        body=body,
    )


def fetch_detail(
    token: str, *, timeout_s: float = 20.0,
    client: httpx.Client | None = None,
) -> ParsedDetail | None:
    """Fetch + parse one detail page."""
    url = DETAIL_URL_TEMPLATE.format(token=token)
    if client is None:
        with httpx.Client(
            timeout=timeout_s, headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as c:
            r = c.get(url)
            r.raise_for_status()
            html = r.text
    else:
        r = client.get(url)
        r.raise_for_status()
        html = r.text
    return parse_detail(html, token=token)


def to_raw_job(parsed: ParsedDetail, *, discovered_keyword: str) -> RawJob:
    """ParsedDetail → RawJob suitable for ingest."""
    raw_parts = [
        f"# {parsed.title}",
    ]
    if parsed.company:
        raw_parts.append(f"公司: {parsed.company}")
    if parsed.location:
        raw_parts.append(f"地点: {parsed.location}")
    if parsed.salary:
        raw_parts.append(f"薪资: {parsed.salary}")
    if parsed.refresh_date:
        raw_parts.append(f"刷新于: {parsed.refresh_date}")
    raw_parts.append("")
    raw_parts.append(parsed.body or "(JD body 未抓到 — 用户去原链接看)")
    raw_parts.append("")
    raw_parts.append(f"投递入口: {parsed.url}")

    return RawJob(
        source="shixiseng",
        source_id=parsed.token,
        url=parsed.url,
        title=parsed.title,
        company=parsed.company,
        location=parsed.location,
        raw_text="\n".join(raw_parts),
        extras={
            "discovered_via": "shixiseng_list_search",
            "discovered_keyword": discovered_keyword,
            "salary": parsed.salary,
            "refresh_date": parsed.refresh_date,
            # 实习僧是聚合站, 链接可能跳第三方 ATS 或站内沟通
            "source_verified": True,  # 站内 detail 页是 verified 真实 JD
            "evidence_url": EVIDENCE_URL_LIST.replace("<kw>", discovered_keyword),
        },
    )


def crawl_shixiseng(
    store: Any, *, keyword: str, max_jobs: int = 20,
) -> FetchResult:
    """One-shot crawl: list page + N detail pages → ingest.

    Defensive about partial failure — ``errors`` list collects what failed
    so a single broken JD doesn't kill the whole batch.
    """
    from ..workers import scout
    result = FetchResult()
    # Single shared client for connection reuse
    try:
        with httpx.Client(
            timeout=20.0, headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as client:
            try:
                tokens = fetch_list_tokens(keyword, page=1, client=client)
            except Exception as e:
                result.errors.append(
                    f"list fetch failed: {type(e).__name__}: {e}"
                )
                return result

            result.listed_total = len(tokens)
            tokens = tokens[: max(0, max_jobs)]

            for tok in tokens:
                try:
                    parsed = fetch_detail(tok, client=client)
                    result.fetched += 1
                except Exception as e:
                    result.errors.append(
                        f"detail {tok}: {type(e).__name__}: {e}"
                    )
                    continue
                if parsed is None:
                    result.errors.append(f"detail {tok}: parse returned None")
                    continue
                result.parsed += 1
                try:
                    rj = to_raw_job(parsed, discovered_keyword=keyword)
                    was_new, _ = scout.ingest(store, rj)
                    if was_new:
                        result.inserted += 1
                        if parsed.company:
                            result.by_company[parsed.company] = (
                                result.by_company.get(parsed.company, 0) + 1
                            )
                    else:
                        result.duplicate += 1
                except Exception as e:
                    result.errors.append(
                        f"ingest {tok}: {type(e).__name__}: {e}"
                    )
    except Exception as e:
        result.errors.append(f"client setup failed: {type(e).__name__}: {e}")
    return result


# ── helpers ─────────────────────────────────────────────────────────────


def _strip_tags(html: str) -> str:
    """Strip HTML tags from a snippet, keeping inner text."""
    # Replace <br> + block tags with newlines first to keep readability
    html = re.sub(r"<br\s*/?>", "\n", html)
    html = re.sub(r"</?(?:p|div|li|h[1-6])[^>]*>", "\n", html)
    # Strip remaining tags
    return re.sub(r"<[^>]+>", "", html)


def _collapse_ws(text: str) -> str:
    """Multiple spaces/newlines → single space, but preserve hard newlines
    that came from paragraph breaks."""
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [re.sub(r"[ \t ]+", " ", ln) for ln in lines if ln]
    return "\n".join(lines)


def _strip_html_entities(text: str) -> str:
    """Decode common HTML entities. (Detail pages don't have font-obf private
    unicode, so this is just &amp; &lt; etc.)"""
    return (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
