"""Agentic 面经 collector — searches the web + LLM-filters + ingests.

Replaces the "user must paste 面经 manually" workflow. The agent:

1. Searches the web (via ``SearchBackend``) for queries like
   "{company} 面经 校招 实习" and English "{company} interview
   experience" (multi-query for breadth).
2. Fetches each candidate URL (via httpx with a 15s timeout).
3. **LLM evaluates** each fetched page: is this real 面经? from a
   recent 校招? matched the asked-for company? Returns a structured
   verdict + extracted clean text.
4. Inserts kept ones into ``interview_experiences`` table via the
   existing ``interview_corpus.insert`` helper (deduped by
   content_hash).

Cost: ~5 LLM calls per company (one per filtered candidate at
DeepSeek-V4-flash pricing ≈ $0.001 per company sweep).

Why this is *real* agency:
- Each step is a decision, not a hardcoded rule
- The LLM reads context to evaluate quality
- Failure modes are graceful (empty search → log, skip; LLM rejects
  → drop, don't store)
"""

from __future__ import annotations

import json
import logging
import re as _re
from dataclasses import dataclass
from typing import Any

import httpx

from .. import interview_corpus
from ..llm import LLMClient, LLMError
from ..memory import Store
from .search import SearchBackend, SearchHit

log = logging.getLogger(__name__)


# Domains likely to host 面经-style content + are not aggressively
# anti-bot. (Manual paste remains the answer for 小红书.)
_PREFERRED_DOMAINS = (
    "nowcoder.com",
    "zhihu.com",
    "csdn.net",
    "jianshu.com",
    "1point3acres.com",
    "github.com",
    "1point3acres",
    "weibo.com",
)


@dataclass(frozen=True)
class CollectionResult:
    """Summary of one company sweep."""

    company: str
    queries_run: list[str]
    hits_seen: int
    hits_evaluated: int
    """Hits we actually fetched + sent to the LLM (after preferred-domain
    filtering)."""

    inserted: int
    """How many new 面经 went into interview_experiences (after dedup)."""

    skipped_dup: int
    skipped_low_quality: int
    notes: list[str]


_FILTER_PROMPT = """你是面经质量评估器。给定一段网页文本，判断：

1. 这是不是关于 **{company}** 的真实**面试经验** (面经)？
2. 是哪个岗位的（实习 / 校招 / 社招 / 不清楚）？
3. 大致是哪一年的？（2024 / 2025 / 2026 / 不清楚）
4. 提取 raw_text 干净版本（去掉网页 nav / footer / 广告）

输出严格 JSON：
{{
  "is_genuine_interview_exp": <bool>,
  "company_match": <bool, 是否与 {company} 一致>,
  "role_hint": <str | null>,
  "year_guess": <str | null>,
  "clean_raw_text": <str, 去除网页噪声后的面经原文，<= 1500 字>,
  "rationale": <str, 1 句话说明你的判断依据>
}}

如果 is_genuine_interview_exp=false 或 company_match=false，clean_raw_text 可以填空字符串。
**不要 markdown 代码块**。"""


class CorpusCollector:
    """The agentic 面经 collector. One instance per app lifetime."""

    def __init__(
        self,
        *,
        store: Store,
        llm: LLMClient,
        search: SearchBackend,
        max_pages_per_company: int = 6,
        page_fetch_timeout_s: float = 15.0,
    ) -> None:
        self.store = store
        self.llm = llm
        self.search = search
        self.max_pages = max_pages_per_company
        self._http = httpx.Client(
            timeout=page_fetch_timeout_s,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/605.1.15 Safari/605.1.15"
                ),
            },
        )

    def collect(self, company: str, *, role_hint: str | None = None) -> CollectionResult:
        """Sweep the web for 面经 about ``company`` and ingest the good ones."""
        queries = self._make_queries(company, role_hint)
        all_hits: list[SearchHit] = []
        seen_urls: set[str] = set()

        notes: list[str] = []
        for q in queries:
            hits = self.search.search(q, max_results=10)
            for h in hits:
                if h.url in seen_urls:
                    continue
                seen_urls.add(h.url)
                all_hits.append(h)
            notes.append(f"query {q!r}: {len(hits)} hits")

        # Filter by preferred domains (avoid 小红书 anti-bot etc.)
        candidates = [h for h in all_hits if _is_preferred_domain(h.url)]
        candidates = candidates[: self.max_pages]
        notes.append(
            f"after domain filter: {len(candidates)} / {len(all_hits)} candidates"
        )

        inserted = 0
        skipped_dup = 0
        skipped_low_quality = 0
        evaluated = 0

        for hit in candidates:
            evaluated += 1
            page_text = self._fetch_text(hit.url)
            if not page_text:
                notes.append(f"skip {hit.url[:60]}: fetch failed")
                continue

            verdict = self._llm_evaluate(company, page_text, role_hint)
            if verdict is None:
                notes.append(f"skip {hit.url[:60]}: LLM rejected JSON")
                skipped_low_quality += 1
                continue

            if not verdict.get("is_genuine_interview_exp") or not verdict.get(
                "company_match"
            ):
                notes.append(
                    f"skip {hit.url[:60]}: {verdict.get('rationale', 'low quality')}"
                )
                skipped_low_quality += 1
                continue

            clean_text = (verdict.get("clean_raw_text") or "").strip()
            if len(clean_text) < 80:
                notes.append(f"skip {hit.url[:60]}: clean_text too short")
                skipped_low_quality += 1
                continue

            try:
                was_new, _ = interview_corpus.insert(
                    self.store,
                    company=company,
                    raw_text=clean_text,
                    source="agent_search",
                    role_hint=verdict.get("role_hint") or role_hint,
                    source_url=hit.url,
                )
                if was_new:
                    inserted += 1
                else:
                    skipped_dup += 1
            except ValueError as e:
                notes.append(f"insert failed for {hit.url[:60]}: {e}")
                skipped_low_quality += 1

        return CollectionResult(
            company=company,
            queries_run=queries,
            hits_seen=len(all_hits),
            hits_evaluated=evaluated,
            inserted=inserted,
            skipped_dup=skipped_dup,
            skipped_low_quality=skipped_low_quality,
            notes=notes,
        )

    # ── internals ──────────────────────────────────────────────────

    def _make_queries(self, company: str, role_hint: str | None) -> list[str]:
        out = [
            f"{company} 面经 校招 2026",
            f"{company} 实习面经 牛客",
        ]
        if role_hint:
            out.append(f"{company} {role_hint} 面经")
        out.append(f'"{company}" interview experience site:nowcoder.com')
        return out

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
            log.warning("fetch failed %s: %s", url, e)
            return ""

    def _llm_evaluate(
        self, company: str, page_text: str, role_hint: str | None
    ) -> dict[str, Any] | None:
        # Cap page text — LLM doesn't need 100K of HTML soup
        snippet = page_text[:6000]
        if role_hint:
            snippet = f"【关注岗位】{role_hint}\n\n{snippet}"

        try:
            resp = self.llm.chat(
                messages=[
                    {"role": "system", "content": _FILTER_PROMPT.format(company=company)},
                    {"role": "user", "content": snippet},
                ],
                temperature=0.0,
                json_mode=True,
            )
        except LLMError as e:
            log.warning("LLM filter failed: %s", e)
            return None

        try:
            return json.loads(resp.content)
        except json.JSONDecodeError:
            return None

    def close(self) -> None:
        self._http.close()


# ── helpers ────────────────────────────────────────────────────────


def _is_preferred_domain(url: str) -> bool:
    return any(d in url for d in _PREFERRED_DOMAINS)


_TAG = _re.compile(r"<[^>]+>")
_SPACE = _re.compile(r"[ \t\r]+")
_BLANK = _re.compile(r"\n{3,}")


def _strip_html_to_text(html: str) -> str:
    """Quick HTML → plaintext. Not perfect but good enough for the LLM
    filter (which is forgiving)."""
    # Drop <script>, <style> blocks entirely
    html = _re.sub(r"<script[^>]*>.*?</script>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
    html = _re.sub(r"<style[^>]*>.*?</style>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
    # All other tags → whitespace
    text = _TAG.sub(" ", html)
    # Collapse whitespace
    text = _SPACE.sub(" ", text)
    text = _BLANK.sub("\n\n", text)
    return text.strip()
