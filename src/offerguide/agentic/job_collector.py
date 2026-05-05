"""Agentic 找岗位 collector — LLM-driven multi-hop retrieval.

W14.15: rewritten as a real agent loop instead of single-pass filter.

Why the rewrite — user feedback (verbatim):
> "不应该是放宽 filter, 而应该是 LLM 根据这个自己去进一步找 JD 在哪,
>  这就是接 LLM 的意义啊, 不然我要他干嘛呢"

Old behavior (W14.12-W14.14): Tavily search → domain filter → fetch →
LLM extracts whatever it can → ingest. Failure mode: Tavily returns a
大厂 careers index page → LLM either rejects ("not a concrete JD") or
fabricates a low-quality JD body. Either way: useless ingest.

New behavior (W14.15): LLM examines each fetched page and decides what
the page IS, then chooses the next action:

  - ``concrete_jd``      → extract body, ingest, stop
  - ``careers_index``    → return 1-3 sub-page URLs the agent should fetch next
  - ``listing``          → return 1-3 specific JD URLs from the listing
  - ``irrelevant``/``outdated`` → stop, no ingest

The collector runs a BFS over (url, depth) pairs, feeding sub-page URLs
back into the queue, capped by:
  - ``max_llm_calls`` (default 12) — total budget per sweep
  - ``max_depth``     (default 2) — at most 2 hops past the initial Tavily hit

Cost vs old: ~2-3× LLM calls per sweep (~$0.02-0.04 with deepseek-v4-flash)
but the ingested rows are ACTUAL JDs instead of fabricated content — the
downstream score_match + apply_assistant get real input to work with.
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
from .search import SearchBackend

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


_DECIDE_PROMPT = """你是 JD 检索 agent。给定**一个网页 + 它上面的链接**, 判断这页是什么, 决定下一步动作。

用户当前 north star: 「{north_star}」

5 种 page_kind:

1. **concrete_jd** — 这页是一个具体岗位的 JD (有工作内容 / 任职要求 / 公司 / 地点 / 投递入口)。
   抽 body 入库, next_action.kind = "ingest"

2. **careers_index** — 公司 careers 入口页 (例 talent.bytedance.com 主页), 不是 JD。
   但页面 / links 里**指向了**该公司具体职位列表。
   挑 1-3 个跟 north star 最相关的子页 URL → next_action.kind = "follow_links"

3. **listing** — 多岗位列表页 (例 牛客 "字节 2026 校招" 帖, 列了 20 个岗位)。
   挑 1-3 个具体 JD URL → next_action.kind = "follow_links"

4. **irrelevant** — 客服页 / 博客 / 跟招聘无关 / 不匹配 north star 方向。
   next_action.kind = "stop"

5. **outdated** — 明确是 2024 或更早的过期招聘。
   next_action.kind = "stop"

链接里**只挑符合 north star 方向**的 (例: 找 AI Agent 实习, 别返算法 OD / 后端 / 测试岗)。

输出严格 JSON:
{{
  "page_kind": "concrete_jd" | "careers_index" | "listing" | "irrelevant" | "outdated",
  "company": <str | null, 没明确就 null>,
  "title": <str | null, page_kind=concrete_jd 时填岗位 title>,
  "location": <str | null>,
  "jd_body_clean": <str, ≥ 200 字, 仅 page_kind=concrete_jd 时填; 其他填 "">,
  "next_action": {{
    "kind": "ingest" | "follow_links" | "stop",
    "urls": [<str>],   // follow_links 时 1-3 个绝对 URL, 其他空数组
    "rationale": <str, 1 句话: 为啥这个 page_kind, 选了哪些链接>
  }}
}}

**不要 markdown 代码块**。"""


@dataclass
class _PendingHit:
    """One URL queued for fetch + LLM evaluation."""
    url: str
    depth: int  # 0 = initial Tavily hit, 1+ = follow_link
    snippet: str = ""  # Tavily snippet, used as fallback when fetch fails
    parent_url: str | None = None  # for audit trail in notes


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
        max_llm_calls: int = 12,
        max_depth: int = 2,
        page_fetch_timeout_s: float = 12.0,
    ) -> None:
        """``max_llm_calls`` caps total LLM calls per sweep (cost cap).
        ``max_depth`` caps how many follow_links hops past the initial
        Tavily hit (depth 0 = Tavily hit, depth 1 = first follow, ...).
        """
        self.store = store
        self.llm = llm
        self.search = search
        self.max_llm_calls = max_llm_calls
        self.max_depth = max_depth
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
        """Run one LLM-driven multi-hop JD sweep.

        Flow:
          1. Tavily search → seed URLs at depth=0
          2. BFS: pop URL, fetch, ask LLM what it is + what to do next
          3. Either ingest (concrete JD) or queue follow_links (depth+1)
          4. Stop when LLM budget hits or queue empties

        ``north_star`` — drives both query generation and LLM relevance judgment.
        ``role_keywords`` / ``companies`` — auto-derived if omitted.
        """
        queries = self._make_queries(
            north_star=north_star,
            role_keywords=role_keywords or self._extract_keywords(north_star),
            companies=companies or list(_DEFAULT_AI_COMPANIES),
        )

        # Step 1 — Tavily search to seed the BFS queue
        seen_urls: set[str] = set()
        notes: list[str] = []
        queue: list[_PendingHit] = []

        for q in queries:
            hits = self.search.search(q, max_results=6)
            for h in hits:
                if h.url in seen_urls:
                    continue
                seen_urls.add(h.url)
                if not _is_preferred_domain(h.url):
                    continue
                queue.append(_PendingHit(url=h.url, depth=0, snippet=h.snippet))
            notes.append(f"query {q!r}: {len(hits)} hits")
        notes.append(f"queue seeded: {len(queue)} URLs from preferred domains")

        # Step 2 — BFS with LLM as the decision-maker at each node
        inserted = 0
        skipped_dup = 0
        skipped_low_quality = 0
        evaluated = 0
        llm_calls = 0
        new_job_ids: list[int] = []

        while queue and llm_calls < self.max_llm_calls:
            hit = queue.pop(0)
            evaluated += 1

            page_text = self._fetch_text(hit.url)
            if not page_text:
                if len(hit.snippet) > 200:
                    page_text = hit.snippet
                else:
                    notes.append(
                        f"d{hit.depth} skip {hit.url[:55]}: fetch fail + snippet薄",
                    )
                    skipped_low_quality += 1
                    continue

            verdict = self._llm_decide(north_star, hit.url, page_text)
            llm_calls += 1
            if verdict is None:
                notes.append(f"d{hit.depth} skip {hit.url[:55]}: LLM 返回非 JSON")
                skipped_low_quality += 1
                continue

            kind = verdict.get("page_kind", "irrelevant")
            next_act = verdict.get("next_action", {}) or {}
            next_kind = next_act.get("kind", "stop")
            rationale = (next_act.get("rationale") or verdict.get("rationale") or "")[:90]

            # Ingest if LLM said this is a concrete JD
            if kind == "concrete_jd" and next_kind == "ingest":
                jd_body = (verdict.get("jd_body_clean") or "").strip()
                if len(jd_body) < 150:
                    notes.append(
                        f"d{hit.depth} skip {hit.url[:55]}: 标 concrete_jd 但 body 太短 ({len(jd_body)})",
                    )
                    skipped_low_quality += 1
                    continue
                title = (verdict.get("title") or "").strip()[:200]
                company = (verdict.get("company") or "").strip()[:100]
                location = (verdict.get("location") or "").strip()[:100]
                rj = RawJob(
                    source="agent_search",
                    url=hit.url,
                    title=title or f"(JD from {hit.url[:50]})",
                    company=company or None,
                    location=location or None,
                    raw_text=jd_body,
                    extras={
                        "rationale": rationale,
                        "via_depth": hit.depth,
                        "parent_url": hit.parent_url,
                    },
                )
                try:
                    was_new, job_id = scout.ingest(self.store, rj)
                    if was_new:
                        inserted += 1
                        new_job_ids.append(job_id)
                        notes.append(
                            f"d{hit.depth} ✓ INGEST {hit.url[:50]} → job#{job_id} ({company} · {title[:30]})",
                        )
                    else:
                        skipped_dup += 1
                        notes.append(f"d{hit.depth} dup {hit.url[:55]}")
                except Exception as e:
                    notes.append(f"d{hit.depth} ingest 失败 {hit.url[:50]}: {e}")
                    skipped_low_quality += 1
                continue

            # Follow links if LLM said this is a careers_index / listing
            if kind in ("careers_index", "listing") and next_kind == "follow_links":
                if hit.depth >= self.max_depth:
                    notes.append(
                        f"d{hit.depth} skip {hit.url[:55]}: {kind} 但已到 max_depth",
                    )
                    skipped_low_quality += 1
                    continue
                sub_urls = next_act.get("urls", []) or []
                added = 0
                for sub_url in sub_urls[:3]:
                    sub_url = (sub_url or "").strip()
                    if not sub_url or sub_url in seen_urls:
                        continue
                    if not sub_url.startswith(("http://", "https://")):
                        # Resolve relative URLs against the parent
                        from urllib.parse import urljoin
                        sub_url = urljoin(hit.url, sub_url)
                    if sub_url in seen_urls:
                        continue
                    seen_urls.add(sub_url)
                    queue.append(_PendingHit(
                        url=sub_url, depth=hit.depth + 1,
                        parent_url=hit.url,
                    ))
                    added += 1
                notes.append(
                    f"d{hit.depth} {kind} {hit.url[:50]} → 加 {added} 个子 URL ({rationale})",
                )
                continue

            # Anything else: irrelevant / outdated / stop
            notes.append(f"d{hit.depth} skip {hit.url[:55]}: {kind} ({rationale})")
            skipped_low_quality += 1

        notes.append(
            f"sweep done: {llm_calls} LLM calls, {evaluated} URLs evaluated, "
            f"{inserted} ingested, {skipped_dup} dups, {skipped_low_quality} skipped",
        )

        return JobCollectionResult(
            queries_run=queries,
            hits_seen=len(seen_urls),
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

    def _llm_decide(
        self,
        north_star: str,
        url: str,
        page_text: str,
    ) -> dict[str, Any] | None:
        """Ask LLM: what is this page + what should I do next?

        Returns the parsed verdict dict, or None on LLM/JSON failure.
        Includes URL in user msg so LLM can resolve relative links and
        recognize careers-index URL patterns.
        """
        snippet = page_text[:6000]
        user_msg = f"【当前 URL】 {url}\n\n【页面内容】\n{snippet}"
        try:
            resp = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": _DECIDE_PROMPT.format(north_star=north_star),
                    },
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                json_mode=True,
            )
        except LLMError as e:
            log.warning("LLM JD-decide failed: %s", e)
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
