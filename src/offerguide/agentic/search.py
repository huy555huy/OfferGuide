"""Search backend abstraction.

Why abstract:
- Local prototyping uses DuckDuckGo's HTML endpoint (no API key, low
  rate limit, brittle but free)
- Production should swap in Google Custom Search / Bing Search / Serper
  / Tavily — any of which need an API key
- Tests inject a stub backend that returns canned hits

Default backend (``DuckDuckGoSearch``) hits the no-auth `html.duckduckgo.com`
endpoint and parses the HTML. Limits per IP, ToS-fragile, but enough to
prototype agentic 面经 collection without making the user pay for a
search API.

For real long-running ambient use, the user should set:

    OFFERGUIDE_SEARCH_BACKEND=tavily
    TAVILY_API_KEY=tvly-...

(Tavily implementation is a TODO — DuckDuckGo is just for getting
the architecture right.)
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

    - ``duckduckgo`` (default): no API key, free, rate-limited
    - ``stub``: returns nothing — useful when you want the agent to
      skip the search step (e.g. on CI, or before user sets up search)
    """
    name = os.environ.get("OFFERGUIDE_SEARCH_BACKEND", "duckduckgo").lower()
    if name == "stub":
        return StubSearch()
    return DuckDuckGoSearch()
