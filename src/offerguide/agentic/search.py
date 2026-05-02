"""Search backend abstraction.

Three real backends + 1 stub + 1 chain:

- ``BingCNSearch`` — ``cn.bing.com``, **零配置中国大陆默认**。HTML SERP
  无需 API key, 国内访问稳定。质量中等。
- ``TavilySearch`` — Tavily API (https://tavily.com), 1000 次/月免费,
  专给 AI agent 设计的高质量 SERP。需要 ``TAVILY_API_KEY``。
- ``DuckDuckGoSearch`` — DDG HTML SERP, 国外用着方便, **国内常被墙**。
  默认不再使用。
- ``StubSearch`` — 测试用, 永不联网。
- ``ChainedSearch`` — 按顺序 try 多个 backend, 第一个返回 ≥1 hit 的赢。

W12-fix 里默认改成 **Bing → Tavily (if key) → DDG fallback chain**, 这样:
- 用户不配任何东西也能搜（Bing CN 国内通）
- 配 TAVILY_API_KEY 自动升级到 Tavily（更准）
- DDG 作为最后兜底（国外 IP 可用）

Override:

    OFFERGUIDE_SEARCH_BACKEND=tavily   # 强制 Tavily, 没 key 报错
    OFFERGUIDE_SEARCH_BACKEND=bing     # 强制 Bing CN
    OFFERGUIDE_SEARCH_BACKEND=duckduckgo
    OFFERGUIDE_SEARCH_BACKEND=stub     # CI / 离线
    (默认未设 = chain)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote_plus, urlparse

import httpx


@dataclass(frozen=True)
class SearchHit:
    """One result from any search backend."""

    title: str
    url: str
    snippet: str
    """Excerpt the search engine showed alongside the result."""


class SearchBackend(Protocol):
    name: str

    def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        """Run a query, return hits. Should never raise; on error return []."""


# ── DuckDuckGo HTML backend ────────────────────────────────────────


class DuckDuckGoSearch:
    """Free, no-auth backend using DuckDuckGo's HTML SERP.

    ToS: rate-limited per IP. Don't run this aggressively (≤ 10 q/min
    is the rough sweet spot before they cut you off). Brittle to HTML
    changes — a strong agent should retry once and bail gracefully.
    """

    name = "duckduckgo"

    def __init__(self, *, timeout_s: float = 15.0) -> None:
        self._http = httpx.Client(
            timeout=timeout_s,
            headers={
                # DDG's HTML endpoint requires a real-looking UA
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                    "Version/17.0 Safari/605.1.15"
                ),
            },
        )

    def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        try:
            resp = self._http.get(
                f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            )
            if resp.status_code != 200:
                return []
            return _parse_ddg_html(resp.text, max_results=max_results)
        except httpx.HTTPError:
            return []

    def close(self) -> None:
        self._http.close()


# DDG HTML structure: <a class="result__a" href="..."><span>title</span></a>
# followed by <a class="result__snippet">snippet</a>
_RESULT_PATTERN = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    flags=re.DOTALL,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")


def _parse_ddg_html(html: str, *, max_results: int) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for match in _RESULT_PATTERN.finditer(html):
        if len(hits) >= max_results:
            break
        url, title_html, snippet_html = match.groups()
        # DDG wraps real URLs in /l/?uddg=<encoded>; strip the wrapper
        url = _unwrap_ddg_redirect(url)
        title = _strip_html(title_html).strip()
        snippet = _strip_html(snippet_html).strip()
        if title and url and url.startswith("http"):
            hits.append(SearchHit(title=title, url=url, snippet=snippet))
    return hits


def _unwrap_ddg_redirect(url: str) -> str:
    """DDG wraps URLs in `/l/?uddg=<encoded>&rut=...`. Decode."""
    if "uddg=" not in url:
        return url
    from urllib.parse import parse_qs, unquote

    parsed = urlparse(url if url.startswith("http") else "https:" + url)
    qs = parse_qs(parsed.query)
    uddg_values = qs.get("uddg")
    if uddg_values:
        return unquote(uddg_values[0])
    return url


def _strip_html(s: str) -> str:
    return _TAG_PATTERN.sub("", s).replace("&amp;", "&").replace("&quot;", '"')


# ── Bing CN backend (国内默认) ─────────────────────────────────────


class BingCNSearch:
    """``cn.bing.com`` HTML SERP — 零配置, 国内默认 backend.

    优势:
    - 不需要 API key
    - cn.bing.com 在中国大陆访问稳定 (微软在国内有 ICP)
    - HTML 结构相对稳定

    限制:
    - 跟所有 HTML 抓取一样, ToS-fragile
    - 中文 query 比英文 query 命中率高 (微软国内域名以中文为主)
    - rate-limited per IP, 跑太快会被 captcha
    """

    name = "bing_cn"

    def __init__(self, *, timeout_s: float = 12.0) -> None:
        self._http = httpx.Client(
            timeout=timeout_s,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            },
        )

    def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        # Use www.bing.com/search directly with mkt=zh-CN. Avoids the
        # cn.bing.com → www.bing.com/?q=... redirect that drops the /search
        # path and lands on the homepage (Bing 2026-05).
        try:
            resp = self._http.get(
                f"https://www.bing.com/search?q={quote_plus(query)}&mkt=zh-CN&setlang=zh",
                follow_redirects=True,
            )
            if resp.status_code != 200:
                return []
            return _parse_bing_html(resp.text, max_results=max_results)
        except httpx.HTTPError:
            return []

    def close(self) -> None:
        self._http.close()


# Bing HTML structure (2026-05): each result is one <li class="b_algo">
# block. Inside, multiple <a> tags exist (sitemap pretty-link + actual h2
# anchor); we want only the <h2><a href=...>title</a></h2> one. Snippet
# is in <p class="b_lineclamp{1,2,3}">snippet</p>. Two-stage parse:
# split into b_algo blocks first, then per-block grab h2 anchor + p.
_BING_BLOCK_RE = re.compile(
    r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>',
    flags=re.DOTALL,
)
_BING_TITLE_RE = re.compile(
    r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>',
    flags=re.DOTALL,
)
_BING_SNIPPET_RE = re.compile(
    r'<p[^>]*class="[^"]*b_(?:lineclamp\d|paractl|caption)[^"]*"[^>]*>(.*?)</p>',
    flags=re.DOTALL,
)


def _parse_bing_html(html: str, *, max_results: int) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for block_match in _BING_BLOCK_RE.finditer(html):
        if len(hits) >= max_results:
            break
        block = block_match.group(1)
        title_match = _BING_TITLE_RE.search(block)
        if not title_match:
            continue
        url, title_html = title_match.groups()
        if not url.startswith("http"):
            continue
        snippet_match = _BING_SNIPPET_RE.search(block)
        snippet_html = snippet_match.group(1) if snippet_match else ""
        title = _strip_html(title_html).strip()
        snippet = _strip_html(snippet_html).strip()[:300]
        if title:
            hits.append(SearchHit(title=title, url=url, snippet=snippet))
    return hits


# ── Tavily API backend (高质量, 1000 免费/月) ──────────────────────


class TavilySearch:
    """`Tavily <https://tavily.com>`_ — search API designed for AI agents.

    Why this is the "good" backend:
    - 专门为 LLM agent 设计的 SERP, 比通用 SE 给的结果更适合 RAG / 评估
    - 免费 quota 1000 次/月 (我们一周大概 50-100 次, 远不到上限)
    - 一行 ``TAVILY_API_KEY=tvly-xxx`` 就升级
    - 国内通

    错误处理: 任何 transport / API 错都 silent return [], 不影响 daemon
    继续跑别的 job。
    """

    name = "tavily"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_s: float = 15.0,
        search_depth: str = "basic",
    ) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "TavilySearch needs TAVILY_API_KEY (env var or constructor arg). "
                "Get a free key at https://tavily.com (1000 calls/月免费)"
            )
        self.search_depth = search_depth  # "basic" | "advanced"
        self._http = httpx.Client(timeout=timeout_s)

    def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        try:
            resp = self._http.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": self.search_depth,
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
        except httpx.HTTPError:
            return []
        if resp.status_code != 200:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        out: list[SearchHit] = []
        for r in (data.get("results") or [])[:max_results]:
            url = r.get("url", "")
            title = (r.get("title") or "").strip()
            snippet = (r.get("content") or "")[:300].strip()
            if url and title:
                out.append(SearchHit(title=title, url=url, snippet=snippet))
        return out

    def close(self) -> None:
        self._http.close()


# ── Chained backend (zero-config fallback) ─────────────────────────


class ChainedSearch:
    """Run backends in order; first one to return ≥1 hit wins.

    Used as the default factory output: Bing CN (国内默认) → Tavily
    (if key) → DDG (fallback). If everything returns empty for a query,
    callers see [] and corpus_collector skips the query gracefully.
    """

    name = "chain"

    def __init__(self, backends: list[SearchBackend]) -> None:
        if not backends:
            raise ValueError("ChainedSearch needs ≥1 backend")
        self.backends = backends

    def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        for be in self.backends:
            try:
                hits = be.search(query, max_results=max_results)
            except Exception:
                hits = []
            if hits:
                return hits
        return []

    def close(self) -> None:
        import contextlib
        for be in self.backends:
            close = getattr(be, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()


# ── Stub backend for tests ─────────────────────────────────────────


class StubSearch:
    """Returns canned hits — used by tests to avoid real network calls."""

    name = "stub"

    def __init__(self, hits_per_query: dict[str, list[SearchHit]] | None = None) -> None:
        self._hits = hits_per_query or {}
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        self.calls.append((query, max_results))
        return list(self._hits.get(query, []))[:max_results]


# ── Factory ────────────────────────────────────────────────────────


def build_default_search() -> SearchBackend:
    """Resolve ``OFFERGUIDE_SEARCH_BACKEND`` env var → backend instance.

    Default (env unset) is **chain** = ``Bing CN → (Tavily if key) → DDG``.
    Bing CN is reliable in mainland China; Tavily improves quality if the
    user pays for an API key; DDG is the international fallback.

    Explicit overrides:
      - ``bing``       — Bing CN only
      - ``tavily``     — Tavily only (errors if no key)
      - ``duckduckgo`` — DDG only (国外 IP 用)
      - ``stub``       — never networks (CI / dev offline)
      - ``chain``      — explicit chain (same as default)
    """
    name = (os.environ.get("OFFERGUIDE_SEARCH_BACKEND") or "chain").lower()

    if name == "stub":
        return StubSearch()
    if name in ("bing", "bing_cn"):
        return BingCNSearch()
    if name == "tavily":
        return TavilySearch()
    if name in ("ddg", "duckduckgo"):
        return DuckDuckGoSearch()

    # Default: chain. Tavily added only if key is configured.
    chain: list[SearchBackend] = [BingCNSearch()]
    if os.environ.get("TAVILY_API_KEY"):
        # Insert Tavily *after* Bing CN — Tavily often better quality
        # but Bing CN is faster + free. Order: Bing → Tavily → DDG.
        chain.append(TavilySearch())
    chain.append(DuckDuckGoSearch())
    return ChainedSearch(chain)


# ── Health check (used by /api/search/test route) ─────────────────


def health_check(backend: SearchBackend) -> dict:
    """Run a canary query and report whether the backend actually works.

    Returns ``{name, ok, hit_count, sample_titles, error_str}``. UI surfaces
    this so user knows whether their search backend is reachable BEFORE
    relying on it for a corpus_refresh sweep.
    """
    canary = "字节跳动 暑期实习 面经"
    name = getattr(backend, "name", "?")
    try:
        hits = backend.search(canary, max_results=3)
    except Exception as e:
        return {
            "name": name, "ok": False, "hit_count": 0,
            "sample_titles": [], "error_str": str(e)[:200],
        }
    return {
        "name": name, "ok": len(hits) > 0,
        "hit_count": len(hits),
        "sample_titles": [h.title[:60] for h in hits[:3]],
        "error_str": None,
    }
