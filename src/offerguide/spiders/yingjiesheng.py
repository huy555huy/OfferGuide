"""应届生网 (yingjiesheng.com) BBS 校招公告 spider.

应届生网是 2026 暑期实习投递的主战场之一——大量公司在 BBS 板块发校招公告 +
HR 答疑帖。BBS 反爬温和（无登录墙、无 JS 渲染），分类页 / RSS 都是静态 HTML，
非常适合 cron 定期拉。

本 spider 的策略：

- 抓「校园招聘」「企业专场」「实习信息」三个分类的列表页
- 每个帖子 = 一个候选 JD (RawJob)
- 帖子标题 + 简介就足以让 score_match 给出初步评分；详情页可以 lazy fetch
- ``content_hash`` 由 ``platforms._spec.content_hash`` 算，重复贴自动去重

不抓的内容：

- 楼内回复 (HR 答疑) — 无结构化价值
- 已经过期的 2025 校招贴 — 用 ``min_date_iso`` 过滤
- 内推贴中没有 JD 内容的 (内推码 only) — 用关键词过滤

Listing page format (verified 2026-04):

  https://www.yingjiesheng.com/gongsi/c-14.html  (校园招聘)
  https://www.yingjiesheng.com/gongsi/c-3.html   (企业专场)
  https://www.yingjiesheng.com/gongsi/c-9.html   (实习信息)

Each row is roughly:

  <tr>
    <td><a href="/gongsi/_xxx_yyyyyy.html">某公司 · 2026 暑期实习招聘</a></td>
    <td>北京 · 上海</td>
    <td>2026-04-15</td>
  </tr>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Final
from urllib.parse import urljoin

from ..platforms import RawJob
from ._base import (
    SpiderError,
    SpiderResult,
    rate_limited_get,
)

YINGJIESHENG_BASE: Final[str] = "https://www.yingjiesheng.com"

DEFAULT_CATEGORIES: Final[dict[str, str]] = {
    "campus":   "/gongsi/c-14.html",  # 校园招聘
    "intern":   "/gongsi/c-9.html",   # 实习信息
    "specials": "/gongsi/c-3.html",   # 企业专场
}
"""Category-key → relative path. Spider runs all by default."""

# Filter keywords that signal "this is just a 内推码 / pure 闲聊"
# rather than a real JD. Title-only, conservative.
_NOISE_TITLE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"^内推码"),
    re.compile(r"^求介绍"),
    re.compile(r"^求内推"),
    re.compile(r"^问一下"),
]

# Match listing rows from the BBS gongsi pages.  The site uses inline
# tables; one row's anchor href + title text is what we need.  We
# capture: relative URL, title, optional location, optional date.
_ROW_RE: Final[re.Pattern[str]] = re.compile(
    r'<a[^>]+href="(/gongsi/_[^"]+\.html)"[^>]*>([^<]+)</a>'
    r'(?:[^<]*<td[^>]*>([^<]*)</td>)?'
    r'(?:[^<]*<td[^>]*>(\d{4}-\d{2}-\d{2})</td>)?',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Posting:
    """One row from the listing page."""
    url: str
    title: str
    location: str
    posted_date: str  # ISO 'YYYY-MM-DD' or '' if unknown


class YingjieshengSpider:
    """应届生网 BBS spider — 公开 HTML，反爬温和。"""

    name: str = "yingjiesheng"

    def __init__(
        self,
        *,
        categories: dict[str, str] | None = None,
        min_date_iso: str | None = None,
    ) -> None:
        """Build the spider.

        ``categories`` overrides which BBS category pages to crawl
        (default: campus + intern + specials). Pass a subset for tests.

        ``min_date_iso`` filters out postings older than this date
        (e.g. "2026-01-01" to skip 2025 校招). Default: no filter.
        Posts without a parseable date are kept.
        """
        self.categories = dict(categories or DEFAULT_CATEGORIES)
        self.min_date = (
            datetime.fromisoformat(min_date_iso).date()
            if min_date_iso
            else None
        )

    def run(self, *, max_items: int = 30) -> SpiderResult:
        out = SpiderResult(spider_name=self.name)

        per_category_cap = max(1, max_items // max(1, len(self.categories)))
        for cat_key, rel_path in self.categories.items():
            url = urljoin(YINGJIESHENG_BASE, rel_path)
            try:
                resp = rate_limited_get(url)
            except SpiderError as e:
                out.errors.append(f"{cat_key}: {e}")
                continue
            out.pages_fetched += 1

            for posting in self._parse_listing(resp.text)[:per_category_cap]:
                if self._is_noise(posting):
                    continue
                if self._is_too_old(posting):
                    continue
                rj = self._posting_to_raw_job(posting, category_key=cat_key)
                out.raw_jobs.append(rj)
                if len(out.raw_jobs) >= max_items:
                    return out
        return out

    # ───── internals ─────

    def _parse_listing(self, html: str) -> list[_Posting]:
        rows: list[_Posting] = []
        for m in _ROW_RE.finditer(html):
            rel_url, title, location, date_str = m.groups()
            title_clean = _strip_html(title).strip()
            if not title_clean:
                continue
            rows.append(
                _Posting(
                    url=urljoin(YINGJIESHENG_BASE, rel_url),
                    title=title_clean,
                    location=_strip_html(location or "").strip(),
                    posted_date=(date_str or "").strip(),
                )
            )
        return rows

    def _is_noise(self, posting: _Posting) -> bool:
        for pat in _NOISE_TITLE_PATTERNS:
            if pat.search(posting.title):
                return True
        # Empty / very short titles (e.g. only punctuation) are noise too
        return len(posting.title.strip()) < 5

    def _is_too_old(self, posting: _Posting) -> bool:
        if self.min_date is None:
            return False
        if not posting.posted_date:
            return False  # no date → keep
        try:
            posted = datetime.fromisoformat(posting.posted_date).date()
        except ValueError:
            return False
        return posted < self.min_date

    def _posting_to_raw_job(self, posting: _Posting, *, category_key: str) -> RawJob:
        company = _extract_company_from_title(posting.title)
        # raw_text is intentionally compact — title + location + date is enough
        # for score_match to give a first-pass probability. For deep analysis
        # the user can click the URL to read the full BBS post.
        raw_text = "\n".join(filter(None, [
            posting.title,
            f"地点: {posting.location}" if posting.location else "",
            f"发布: {posting.posted_date}" if posting.posted_date else "",
            f"分类: {category_key}",
            f"来源: 应届生网 BBS — {posting.url}",
        ]))
        return RawJob(
            source="yingjiesheng",
            source_id=_extract_source_id(posting.url),
            url=posting.url,
            title=posting.title,
            company=company,
            location=posting.location or None,
            raw_text=raw_text,
            extras={
                "category": category_key,
                "posted_date": posting.posted_date,
            },
        )


# ─────────── helpers ───────────────


_TITLE_COMPANY_RE: Final[re.Pattern[str]] = re.compile(
    r"^[【\[]?\s*([^】\]·\-—|/]{2,18})\s*[】\]·\-—|]"
)


def _extract_company_from_title(title: str) -> str | None:
    r"""Pull a company name from a BBS post title heuristically.

    Title formats observed (2026-04):
        "【字节跳动】2026 暑期实习招聘"
        "[蔚来] 算法实习生招聘"
        "美团 · 2026 校招正式批"
        "腾讯-CSIG 后端实习生"
        "阿里巴巴 | 大模型应用算法实习"

    We capture the first segment before a delimiter ([】\]·\-—|]).
    Returns None if no clean match.
    """
    m = _TITLE_COMPANY_RE.match(title)
    if not m:
        return None
    candidate = m.group(1).strip()
    # Reject super-noisy candidates ("最新", "招聘" etc.)
    if candidate in {"最新", "招聘", "通知", "公告", "实习", "校招", "原创"}:
        return None
    return candidate


_SOURCE_ID_RE: Final[re.Pattern[str]] = re.compile(r"_(\w+)\.html")


def _extract_source_id(url: str) -> str | None:
    m = _SOURCE_ID_RE.search(url)
    return m.group(1) if m else None


_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    return _TAG_RE.sub("", s)
