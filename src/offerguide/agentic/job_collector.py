"""Agentic 找岗位 collector — Tavily 搜索 + LLM 抽 JD body, 写 jobs 表.

Replaces (and obsoletes the user-visible behavior of) the old
``spiders.awesome_jobs`` path. The W14.12 audit found that the only
default-enabled spider returned **company directories** (~150-char rows
like "百度 · 校招 / 投递入口: ..."), not real JDs — so score_match
returned garbage and apply_assistant had nothing to chew on.

This module is a sibling of ``agentic.corpus_collector`` (which does the
same shape for 面经). The two share design choices because the tradeoffs
are identical: search → domain filter → fetch → LLM extract → ingest.

Why this is real "自动找岗位":

- Queries are **derived from the user's north-star goal** (e.g.
  "AI Agent 暑期实习 2026"), not a fixed source list. Update the goal,
  the agent searches differently next tick.
- LLM does the noise filtering: actively-hiring vs expired, JD body vs
  company directory, on-target vs off-target. The score_match SKILL
  doesn't have to compensate.
- Output is a real RawJob with multi-hundred-char raw_text — downstream
  (score_match, apply_assistant, tailor_resume) finally has substance
  to work on.

Cost: ~6-12 LLM calls per sweep at deepseek-v4-flash (~$0.005-0.012 per
sweep) + Tavily search (free up to 1000/month).
"""

from __future__ import annotations

import json
import logging
import re as _re
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..llm import LLMClient, LLMError
from ..memory import Store
from ..platforms import RawJob
from ..workers import scout
from .search import SearchBackend, SearchHit

log = logging.getLogger(__name__)


# Domains we'll fetch + LLM-evaluate. Boss直聘 / 智联 / 拉勾 etc. are
# anti-bot heavy — fetching usually returns a CAPTCHA wall, not the JD.
# We include them anyway because Tavily's snippet alone may carry enough
# info, and on the off-chance the page is reachable we'd rather try.
_PREFERRED_DOMAINS = (
    "nowcoder.com",          # 牛客校招板, 通常拿得到 JD body
    "shixiseng.com",         # 实习僧, 实习信息聚合
    "yingjiesheng.com",      # 应届生网 BBS
    "zhipin.com",            # Boss直聘 (HTML 反爬狠, snippet 兜底)
    "lagou.com",             # 拉勾
    "liepin.com",            # 猎聘
    "zhihu.com",             # 知乎招聘文章
    "xiaohongshu.com",       # 小红书 careers (低概率)
    "weixin.qq.com",         # 公众号 careers 文章
    "mp.weixin.qq.com",
    # Company-direct careers — usually reachable
    "bytedance.com",
    "feishu.cn",
    "alibaba.com",
    "tencent.com",
    "qq.com",
    "baidu.com",
    "kuaishou.cn",
    "xiaohongshu.com",
    "meituan.com",
    "jd.com",
    "zhipuai.cn",            # 智谱
    "moonshot.cn",           # 月之暗面
    "deepseek.com",
    "anthropic.com",
    "openai.com",
)


# Default companies to seed queries with when the north-star is generic
# (e.g. "拿到 1 个 AI Agent 暑期实习 offer" doesn't say which company).
# Skewed toward AI/LLM-application companies because that's the project's
# stated target. Override via JobCollector.collect(companies=[...]).
_DEFAULT_AI_COMPANIES = (
    "字节跳动", "腾讯", "阿里巴巴", "百度", "快手",
    "智谱", "月之暗面", "DeepSeek",
    "美团", "京东", "小红书", "美团",
)


@dataclass(frozen=True)
class JobCollectionResult:
    """Summary of one sweep — returned to the agent so it can reason about
    what happened (and write a reasonable inbox suggestion if needed)."""
    queries_run: list[str]
    hits_seen: int
    hits_evaluated: int
    """Hits that survived domain filtering and got LLM-evaluated."""
    inserted: int
    """New jobs that landed in the jobs table after dedup."""
    skipped_dup: int
    skipped_low_quality: int
    """Skipped because LLM judged not-a-real-JD or not-actively-hiring or
    too short."""
    new_job_ids: list[int] = field(default_factory=list)
    """jobs.id of every freshly inserted row — handed off to downstream
    auto-score / auto-suggest pipelines."""
    notes: list[str] = field(default_factory=list)


_FILTER_PROMPT = """你是 JD 抽取器。给定一段网页文本, 判断 + 抽取。

用户当前 north star: 「{north_star}」

判断标准 (W14.14 放宽):

1. **has_jd_content** (宽松): 文本里**包含**有用的招聘信息就 true。可以是:
   - 一个具体岗位 (强 — 直接 ingestible)
   - 或 一个公司的招聘合集页 / 校招介绍 (有"招什么角色 / 工作内容 / 投递入口"的)
   - 或 多个岗位列表 (你抽出最匹配 north star 的那个)
   只有当文本完全无招聘信息 (404 页 / 客服页 / 不相关博客) 才 false。

2. **is_recent** (宽松): 不能是明确 ≥ 2024 已结束的招聘。其他都视为 true。
   提到 "2024/2025 已结束" → false。提到 "2026" 或没具体年份 → true。

3. **relevance_score** (0-3): 跟 north star 多相关:
   - 3 = 角色名/方向**直接对得上** ("AI Agent 实习" 对 "AI Agent 实习")
   - 2 = 同领域 ("LLM 应用工程师" 对 "AI Agent 实习" 算 2)
   - 1 = 相关公司但角色偏 ("字节后端实习" 对 "AI Agent 实习" 算 1)
   - 0 = 完全不相关 (运营 / 设计 / 测试)
   ≥ 2 就值得入库 (宁可入了再 score_match 严判, 不要前期一刀切)

4. 抽出 jd_body_clean: 工作内容 + 任职要求 + 公司 + 岗位 + 地点。≥ 150 字就够。

输出严格 JSON:
{{
  "has_jd_content": <bool>,
  "is_recent": <bool>,
  "relevance_score": <int 0-3>,
  "company": <str, 公司中文名, 没明确就猜>,
  "title": <str, 岗位 title — 多岗位列表里挑跟 north star 最近的一个>,
  "location": <str | null>,
  "jd_body_clean": <str, ≥ 150 字, 含具体工作内容>,
  "rationale": <str, 1 句话: 为啥判这个 score, 抽了哪个岗位>
}}

如果 has_jd_content=false, 其他可以填默认值。
**不要 markdown 代码块**。"""


class JobCollector:
    """Search-driven JD collector. One instance per app lifetime.

    Mirror of ``CorpusCollector`` for the JD-instead-of-面经 case. Output
    flows through ``scout.ingest`` so dedup + extras_json behave identically
    to spider/extension paths.
    """

    def __init__(
        self,
        *,
        store: Store,
        llm: LLMClient,
        search: SearchBackend,
        max_pages: int = 8,
        page_fetch_timeout_s: float = 15.0,
    ) -> None:
        self.store = store
        self.llm = llm
        self.search = search
        self.max_pages = max_pages
        self._http = httpx.Client(
            timeout=page_fetch_timeout_s,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 Safari/605.1.15"
                ),
            },
        )

    def collect(
        self,
        *,
        north_star: str,
        role_keywords: list[str] | None = None,
        companies: list[str] | None = None,
    ) -> JobCollectionResult:
        """Run one search-driven sweep.

        ``north_star`` — the user's current goal (e.g. "拿 1 个 AI Agent
        暑期实习 offer"). Drives the LLM filter's match decision and is
        injected into the query text.

        ``role_keywords`` — short keyword list to slot into search queries.
        Auto-extracted from north_star if not provided.

        ``companies`` — companies to seed company-specific queries with.
        Defaults to a curated AI/LLM company list.
        """
        queries = self._make_queries(
            north_star=north_star,
            role_keywords=role_keywords or self._extract_keywords(north_star),
            companies=companies or list(_DEFAULT_AI_COMPANIES),
        )

        all_hits: list[SearchHit] = []
        seen_urls: set[str] = set()
        notes: list[str] = []

        for q in queries:
            hits = self.search.search(q, max_results=8)
            for h in hits:
                if h.url in seen_urls:
                    continue
                seen_urls.add(h.url)
                all_hits.append(h)
            notes.append(f"query {q!r}: {len(hits)} hits")

        # Filter to fetchable / known-friendly domains
        candidates = [h for h in all_hits if _is_preferred_domain(h.url)]
        candidates = candidates[: self.max_pages]
        notes.append(
            f"after domain filter: {len(candidates)} / {len(all_hits)} candidates",
        )

        inserted = 0
        skipped_dup = 0
        skipped_low_quality = 0
        evaluated = 0
        new_job_ids: list[int] = []

        for hit in candidates:
            evaluated += 1
            page_text = self._fetch_text(hit.url)
            if not page_text:
                # Fall back to the search snippet — sometimes Tavily's
                # excerpt has the JD core even when the full page 403s.
                if len(hit.snippet) > 200:
                    page_text = hit.snippet
                else:
                    notes.append(f"skip {hit.url[:60]}: fetch failed + snippet too thin")
                    skipped_low_quality += 1
                    continue

            verdict = self._llm_evaluate(north_star, page_text)
            if verdict is None:
                notes.append(f"skip {hit.url[:60]}: LLM 返回非 JSON")
                skipped_low_quality += 1
                continue

            # W14.14 — relaxed: has_jd_content + is_recent + relevance >= 2
            has_content = bool(verdict.get("has_jd_content"))
            is_recent = bool(verdict.get("is_recent", True))  # default True if missing
            relevance = int(verdict.get("relevance_score", 0) or 0)

            if not has_content:
                notes.append(f"skip {hit.url[:60]}: no JD content ({verdict.get('rationale', '')[:80]})")
                skipped_low_quality += 1
                continue
            if not is_recent:
                notes.append(f"skip {hit.url[:60]}: outdated ({verdict.get('rationale', '')[:80]})")
                skipped_low_quality += 1
                continue
            if relevance < 2:
                notes.append(f"skip {hit.url[:60]}: relevance={relevance} ({verdict.get('rationale', '')[:80]})")
                skipped_low_quality += 1
                continue

            jd_body = (verdict.get("jd_body_clean") or "").strip()
            # W14.14: lower body threshold from 200 → 150 chars (matches the
            # prompt's new "≥ 150 字就够" target). Below 150 the auto_score
            # daemon won't pick it up anyway (MIN_TEXT_FOR_AUTO_EVAL=200), so
            # we still skip. This drops borderline rows pre-ingest.
            if len(jd_body) < 150:
                notes.append(f"skip {hit.url[:60]}: body too short ({len(jd_body)} 字)")
                skipped_low_quality += 1
                continue

            # Build a RawJob and ingest via the same path the extension /
            # spiders use. dedup by content_hash means re-running the
            # sweep is safe (same JD won't double-count).
            # RawJob.title is required (str, not Optional). Fall back to a
            # synthetic title built from the URL if the LLM didn't extract one.
            extracted_title = (verdict.get("title") or "").strip()[:200]
            extracted_company = (verdict.get("company") or "").strip()[:100]
            extracted_location = (verdict.get("location") or "").strip()[:100]
            rj = RawJob(
                source="agent_search",
                url=hit.url,
                title=extracted_title or f"(JD from {hit.url[:50]})",
                company=extracted_company or None,
                location=extracted_location or None,
                raw_text=jd_body,
                extras={
                    "rationale": verdict.get("rationale", ""),
                    "search_query_url": hit.url,
                },
            )
            try:
                was_new, job_id = scout.ingest(self.store, rj)
                if was_new:
                    inserted += 1
                    new_job_ids.append(job_id)
                else:
                    skipped_dup += 1
            except Exception as e:
                notes.append(f"ingest failed for {hit.url[:60]}: {e}")
                skipped_low_quality += 1

        return JobCollectionResult(
            queries_run=queries,
            hits_seen=len(all_hits),
            hits_evaluated=evaluated,
            inserted=inserted,
            skipped_dup=skipped_dup,
            skipped_low_quality=skipped_low_quality,
            new_job_ids=new_job_ids,
            notes=notes,
        )

    # ── internals ──────────────────────────────────────────────────

    def _make_queries(
        self,
        *,
        north_star: str,
        role_keywords: list[str],
        companies: list[str],
    ) -> list[str]:
        """Build a 6-10 query lineup balancing breadth (generic role
        searches) and depth (company-specific). Year tokens skew toward
        the user's current cycle (2026 暑期 / 2027 校招)."""
        role_str = " ".join(role_keywords[:3]) if role_keywords else "实习"
        out = [
            # Generic role searches (catch broad listings)
            f"{role_str} 暑期实习 2026 校招",
            f"{role_str} 招聘 牛客 校招",
            f"{role_str} 实习生 招聘 北京 上海",
        ]
        # Company-specific searches (most useful when north star names
        # specific targets, but the AI default list is a good seed too)
        for c in companies[:4]:
            out.append(f"{c} {role_str} 实习 招聘 2026")
        return out

    def _extract_keywords(self, north_star: str) -> list[str]:
        """Minimal keyword extraction — split on punctuation, drop stop
        tokens, keep order. We're not trying to be smart, just to forward
        whatever role-relevant words the user wrote."""
        # Split on common Chinese / English separators and punctuation
        toks = _re.split(r"[ \t,，、。;；:：!?？/\-_]+", north_star)
        stop = {
            "我", "想", "拿", "到", "1", "个", "找", "的", "了",
            "工作", "岗位", "职位", "意向", "暑期", "校招", "实习",
            "offer", "Offer", "OFFER",
        }
        kept = [t for t in toks if t and t not in stop and len(t) >= 2]
        return kept[:5]

    def _fetch_text(self, url: str) -> str:
        try:
            r = self._http.get(url, follow_redirects=True)
            if r.status_code != 200:
                return ""
            ct = r.headers.get("content-type", "")
            if "text/html" not in ct and "text/plain" not in ct:
                return ""
            return _strip_html_to_text(r.text)
        except httpx.HTTPError as e:
            log.debug("fetch failed %s: %s", url, e)
            return ""

    def _llm_evaluate(
        self,
        north_star: str,
        page_text: str,
    ) -> dict[str, Any] | None:
        snippet = page_text[:6000]
        try:
            resp = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": _FILTER_PROMPT.format(north_star=north_star),
                    },
                    {"role": "user", "content": snippet},
                ],
                temperature=0.0,
                json_mode=True,
            )
        except LLMError as e:
            log.warning("LLM JD-filter failed: %s", e)
            return None
        try:
            return json.loads(resp.content)
        except json.JSONDecodeError:
            return None

    def close(self) -> None:
        self._http.close()


# ── helpers (mirror corpus_collector for HTML→text consistency) ──


def _is_preferred_domain(url: str) -> bool:
    return any(d in url for d in _PREFERRED_DOMAINS)


_TAG = _re.compile(r"<[^>]+>")
_SPACE = _re.compile(r"[ \t\r]+")
_BLANK = _re.compile(r"\n{3,}")


def _strip_html_to_text(html: str) -> str:
    """HTML → plaintext for LLM consumption. Not perfect but good enough."""
    html = _re.sub(r"<script[^>]*>.*?</script>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
    html = _re.sub(r"<style[^>]*>.*?</style>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
    text = _TAG.sub(" ", html)
    text = _SPACE.sub(" ", text)
    text = _BLANK.sub("\n\n", text)
    return text.strip()
