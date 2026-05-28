"""W19+ — 0voice/2026-Computer-Spring-Recruitment-Job-Compilation aggregator.

Public GitHub repo, MIT-licensed, 243 stars, daily updated, 1000+ JDs across
腾讯/阿里/字节/美团/华为/百度/小米/快手/拼多多/网易/京东/360 + 独角兽/中厂/外企
+ 算法岗专题 (AI 大模型 / 机器学习 / NLP / 数据分析).

Why aggregate this:
- W17/W18 verified_official only covers 腾讯/百度 (实测可拉的 JSON API).
  字节/阿里/美团/小红书 are SOURCE_LANDSCAPE marked unverified_js_shell.
- 0voice repo aggregates 100+ companies' real ATS / 内推链接 manually,
  including those unverified_js_shell ones.
- Maintained by community, daily refresh — much wider net than what we
  can server-scrape ourselves.

Empirically observed format (probed 2026-05-11):
  README.md (~180 KB) — top-level company sections in markdown:

    ## <h3 id="N">公司名</h3>
    |NO.|工作岗位|详细内容|
    |--|--|--|
    |1|岗位名|[点击查看](https://...)|

  Link types:
  - https://campus-talent.alibaba.com/...    → 真阿里 ATS detail page
  - https://app.mokahr.com/campus_apply/...  → 北森 ATS apply page
  - https://job.bytedance.com/...            → 真字节 detail page
  - https://mp.weixin.qq.com/s?...           → 公众号文章 (内推码所在,
                                              不是 JD 直链)
  - https://join.qq.com/...                  → 腾讯 ATS

Not yet verified (out of scope this version):
  - Per-company subdirs (e.g. 腾讯/readme.md) are 面试题集, not JD lists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from ._spec import RawJob

log = logging.getLogger(__name__)

REPO_OWNER = "0voice"
REPO_NAME = "2026-Computer-Spring-Recruitment-Job-Compilation"
README_URL = (
    f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/README.md"
)
USER_AGENT = "OfferGuide/0.0 (+aggregator)"

# Section header: ## <h3 id="...">公司名</h3>
_SECTION_HEADER_RE = re.compile(
    r'##\s*<h3\s+id="[^"]*">\s*([^<]+?)\s*</h3>',
)
# Table row: |NO|岗位名|[link text](url)|
# Allow flexible whitespace and the NO column being a number or "-"
_TABLE_ROW_RE = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*\[([^\]]*)\]\(([^)]+)\)\s*\|\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ParsedJob:
    company: str
    title: str
    url: str
    section_no: int
    """Position within the company's section (1-indexed). Useful for dedup
    against future 0voice updates that may renumber."""


def fetch_readme(*, timeout_s: float = 20.0) -> str:
    """Fetch raw README.md. Raises httpx.HTTPError on network failure."""
    with httpx.Client(
        timeout=timeout_s, headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    ) as c:
        r = c.get(README_URL)
        r.raise_for_status()
        return r.text


def parse_readme(markdown: str) -> list[ParsedJob]:
    """Walk markdown, yield (company, title, url) per row in each company section.

    Empty sections / non-table content between headers are silently skipped.
    Multiple sections with same company name (rare) get merged downstream by
    content_hash anyway.
    """
    out: list[ParsedJob] = []
    sections = list(_SECTION_HEADER_RE.finditer(markdown))
    for i, m in enumerate(sections):
        company = m.group(1).strip()
        if not company:
            continue
        start = m.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(markdown)
        chunk = markdown[start:end]
        # Find all table rows in this section
        for row in _TABLE_ROW_RE.finditer(chunk):
            no_str, title, _link_text, url = row.groups()
            title = title.strip()
            url = url.strip()
            if not title or not url:
                continue
            # Skip the markdown table separator row (|--|--|--|) which has
            # title chars all '-' — defensive, regex already filters numbers
            if all(c == "-" for c in title):
                continue
            try:
                no = int(no_str)
            except ValueError:
                continue
            out.append(ParsedJob(
                company=company, title=title, url=url, section_no=no,
            ))
    return out


# Hosts where the URL is the real ATS detail page (worth treating as
# verified investment plan target). Empirically seen in the 0voice README.
_REAL_ATS_HOSTS: tuple[str, ...] = (
    "campus-talent.alibaba.com",
    "talent.alibaba.com",
    "talent.baidu.com",
    "join.qq.com",
    "careers.tencent.com",
    "jobs.bytedance.com",
    "job.bytedance.com",
    "zhaopin.meituan.com",
    "wd.tencent.com",
    "app.mokahr.com",  # 北森 SaaS, used by many companies
    "career.huawei.com",
    "campus.xiaomi.com",
    "tencent.wd1.myworkdayjobs.com",
    "careers.pddglobalhr.com",
)


def link_is_official_ats(url: str) -> bool:
    """Cheap classifier: real ATS page vs WeChat aggregator article."""
    u = url.lower()
    return any(host in u for host in _REAL_ATS_HOSTS)


def to_raw_job(parsed: ParsedJob) -> RawJob:
    """Convert one ParsedJob → RawJob for ingest. raw_text is short since the
    0voice repo doesn't include JD body — just the title/company link.
    Downstream evaluate / score_match still works against title + company.
    """
    is_ats = link_is_official_ats(parsed.url)
    raw_parts = [
        f"# {parsed.title}",
        f"公司: {parsed.company}",
        f"投递入口: {parsed.url}",
        "",
        ("→ 这是真官方 ATS 投递页, 打开后按页面提示投递。" if is_ats
         else "→ 这是公众号汇总文章, 含内推码或岗位详情, 不是 JD 直链。"),
        "",
        "(0voice GitHub repo 聚合, JD body 没收录 — 用户去链接看完整描述)",
    ]
    return RawJob(
        source="zerovoice_repo",
        source_id=f"{parsed.company}-{parsed.section_no}",
        url=parsed.url,
        title=parsed.title,
        company=parsed.company,
        location=None,  # 0voice doesn't include location
        raw_text="\n".join(raw_parts),
        extras={
            "discovered_via": "zerovoice_aggregator",
            "discovered_keyword": f"{parsed.company} 岗位汇总",
            "link_type": "official_ats" if is_ats else "wechat_article",
            "section_no": parsed.section_no,
            # Mark verified-source if URL points to a known ATS (so
            # application_plan can show "已核验来源" pill)
            "source_verified": is_ats,
            "evidence_url": (
                f"https://github.com/{REPO_OWNER}/{REPO_NAME}/blob/main/README.md"
            ),
        },
    )


@dataclass
class FetchResult:
    """Counter for one ambient cycle's 0voice fetch."""
    parsed_total: int = 0
    inserted: int = 0
    duplicate: int = 0
    skipped_non_ats: int = 0
    skipped_dead: int = 0
    errors: list[str] = None  # type: ignore[assignment]
    by_company: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.by_company is None:
            self.by_company = {}


def crawl_zerovoice(
    store: Any, *, max_jobs: int | None = None, verify_urls: bool = False,
) -> FetchResult:
    """Walk the 0voice README, parse, ingest. Returns FetchResult counter.

    `max_jobs` caps how many jobs to ingest this run — useful for not
    flooding the daemon's first cycle with 1000 entries that would all
    need score_match. None = no cap.

    W19+ bug fix (verified by real audit): pre-bugfix used `parsed_jobs[:max_jobs]`
    which took stable head order. Aliibaba has 124 entries at the top of
    the README — head 30 = all 阿里巴巴, no diversity. Fixed by **round-robin
    across companies** so cap distributes coverage.
    """
    from ..workers import scout
    result = FetchResult()
    try:
        md = fetch_readme()
    except Exception as e:
        result.errors.append(f"fetch README failed: {type(e).__name__}: {e}")
        return result

    parsed_jobs = parse_readme(md)
    result.parsed_total = len(parsed_jobs)
    parsed_jobs = [pj for pj in parsed_jobs if link_is_official_ats(pj.url)]
    result.skipped_non_ats = result.parsed_total - len(parsed_jobs)
    if max_jobs is not None and len(parsed_jobs) > max_jobs:
        parsed_jobs = _round_robin_by_company(parsed_jobs, max_jobs)

    live_cache: dict[str, bool] = {}
    for pj in parsed_jobs:
        try:
            if verify_urls and not _url_looks_live(pj.url, cache=live_cache):
                result.skipped_dead += 1
                continue
            rj = to_raw_job(pj)
            was_new, _ = scout.ingest(store, rj)
            if was_new:
                result.inserted += 1
                result.by_company[pj.company] = (
                    result.by_company.get(pj.company, 0) + 1
                )
            else:
                result.duplicate += 1
        except Exception as e:
            result.errors.append(f"{pj.company}/{pj.title[:30]}: {type(e).__name__}: {e}")

    return result


def _url_looks_live(url: str, *, cache: dict[str, bool] | None = None) -> bool:
    """Best-effort reachability gate for aggregator links.

    It intentionally only has a hard veto for clear dead signals: HTTP 404/410
    or a final redirected URL ending in /404.html. Timeouts/403/anti-bot are
    treated as live enough because many ATS hosts block headless clients while
    still working in a browser.
    """
    cache_key = _url_cache_key(url)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    ok = True
    try:
        with httpx.Client(
            timeout=8.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.head(url)
            if resp.status_code in (403, 405):
                resp = client.get(url)
            final_path = str(resp.url).lower()
            ok = resp.status_code not in (404, 410) and not final_path.endswith("/404.html")
    except httpx.HTTPError:
        ok = True

    if cache is not None:
        cache[cache_key] = ok
    return ok


def _url_cache_key(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{parsed.query}"


def _round_robin_by_company(
    parsed_jobs: list[ParsedJob], cap: int,
) -> list[ParsedJob]:
    """Pick `cap` jobs but spread across companies (not head-first).

    Algorithm: bucket by company, then in each pass take 1 job from each
    bucket until cap reached. Within a company keeps the README's section
    order (1, 2, 3, ...) so user sees prioritized jobs first.
    """
    if cap <= 0:
        return []
    buckets: dict[str, list[ParsedJob]] = {}
    for pj in parsed_jobs:
        buckets.setdefault(pj.company, []).append(pj)
    # Sort each bucket by section_no to keep README priority within a company
    for comp_jobs in buckets.values():
        comp_jobs.sort(key=lambda p: p.section_no)
    # Round-robin: pop one from each bucket per pass
    out: list[ParsedJob] = []
    company_order = list(buckets.keys())  # preserve README company order
    while len(out) < cap:
        picked_this_pass = False
        for comp in company_order:
            if not buckets[comp]:
                continue
            out.append(buckets[comp].pop(0))
            picked_this_pass = True
            if len(out) >= cap:
                break
        if not picked_this_pass:
            break  # all buckets empty
    return out
