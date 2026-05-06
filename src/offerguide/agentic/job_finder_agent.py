"""LLM-driven JD discovery — true agent loop, not classifier-in-disguise.

W14.16: rewrites W14.15's JobCollector. Same goal (find JDs matching the
user's north star), but the LLM **drives every step** instead of being
shoehorned into "pick one of 5 page kinds".

User feedback (verbatim, two iterations in):
> "得 LLM 自己去看网页是什么, 这样才能具体去做事情啊。你这样做和不要
>  模型有什么区别?你到现在还没能理解 LLM 的作用是什么。"

The point: in W14.15 I gave LLM 5 page-kind labels + 3 next-action options
and called it agentic. It wasn't — I had hardcoded the BFS, the ingest
pipeline, the search query templates. LLM was a tagger.

W14.16 — actual ReAct loop:

    LLM gets goal + 4 tools:
      - web_search(query)          : Tavily, returns hits
      - fetch_url(url)             : returns page text
      - extract_and_ingest_jd(...) : LLM decides field values, ingests
      - done(reason)               : LLM decides when to stop

    LLM decides each turn what to call. We just provide tools and feed
    results back. Iterate until done() or budget runs out.

This is the difference between "model-in-the-loop" (W14.15, classifier)
and "model-in-the-driver-seat" (W14.16, agent). Same naming we use for
the central AgentLoop in agent/loop.py — for the same reason.

Cost: 25 LLM turns × ~2-3K tokens each at deepseek-v4-flash ≈ $0.10-0.20
per sweep. 3×/day cron ≈ ~¥6-15/month single user. Worth it because the
ingested rows are real JDs the LLM personally validated, not "page that
matched the keyword filter".
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


# ── Tool schemas (OpenAI function-calling format) ──────────────────


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索引擎 (Tavily). 返一组 hit: 每个含 url + title + 摘要. "
                "重复同 query 会被拒."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {
                        "type": "integer",
                        "description": "默认 6, 最多 10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "HTTP GET 一个 URL, HTML 转纯文本, 返前 6000 字. "
                "失败 (反爬 / 404 / 超时) 返 'ERROR: ...'. "
                "重复 fetch 同 URL 会被拒."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_and_ingest_jd",
            "description": (
                "把一份 JD 入 jobs 表. 你提供抽好的字段 (url + jd_body + "
                "company + title + location). 返 job_id 或 'duplicate'. "
                "硬约束: jd_body 必须 ≥ 200 字; url 必须是你 fetch 过的 "
                "(防编造). 不满足返 ERROR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "jd_body": {"type": "string"},
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["url", "jd_body", "company", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": (
                "结束本轮. 提供 reason (1 句话, 给后台日志看)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]


# W14.17 — 砍光"如果 X 就 Y"的硬编码流程提示。
# 之前 (W14.16) 我在 prompt 里替 LLM 想了 5 种 page kind + 每种该怎么办,
# 等于 LLM 不能在我没列出的 case 里灵活反应。用户原话:
# "你设定好了, 那你就能确定, 你设定的网页状况适合所有的? 你去查找的时候
#  也不是这么做的吧"
# 现在只给目标 + 工具 + budget, LLM 自己思考 (跟人查岗位的方式一样).
_SYSTEM_PROMPT = """你是 OfferGuide 的 JD 检索助手. 给用户找跟 north star 匹配的真 JD, 入库.

# 用户当前 north star
「{north_star}」

# Budget
最多 25 轮工具调用, 找 3-5 个真 JD 就够; 找不到也别硬撑, done 走人即可.

# 怎么干
**像你帮人找工作那样自己想.** 看到搜索结果 / 网页 / 错误, 就想 "这是啥情况, 我接下来怎么办".
没有固定流程, 没有"5 类页面分别怎么处理"那种规矩. 你能用的就 4 个工具, 怎么组合
完全你说了算. 唯一硬规矩:
- 想 ingest 一个 URL → 你必须先 fetch 过它 (否则就是编造)
- 浪费 budget 在重复 search/fetch 上, 我会拒
"""


@dataclass
class JobFinderResult:
    """Summary of one agentic discovery sweep."""
    iterations: int
    inserted: int
    skipped_dup: int
    new_job_ids: list[int] = field(default_factory=list)
    visited_urls: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    finish_reason: str = ""
    notes: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    """W15.12 review fix (Bug 5): total LLM cost burned by this sweep.
    The harness tools layer reads this and accumulates into the parent
    harness_runs.cost_usd so /debug shows true total cost (sub-agent
    + main agent), not just the main agent's tool-decision tokens."""


class JobFinderAgent:
    """LLM-driven JD finder. Each instance handles one sweep."""

    def __init__(
        self,
        *,
        store: Store,
        llm: LLMClient,
        search: SearchBackend,
        max_iterations: int = 25,
        page_fetch_timeout_s: float = 12.0,
    ) -> None:
        self.store = store
        self.llm = llm
        self.search = search
        self.max_iter = max_iterations
        self._http = httpx.Client(
            timeout=page_fetch_timeout_s,
            headers={"User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 Safari/605.1.15"
            )},
        )
        # Per-sweep state — reset each .run()
        self._visited: set[str] = set()
        self._search_qs: list[str] = []
        self._inserted_ids: list[int] = []
        self._skipped_dup = 0
        self._notes: list[str] = []

    def run(self, *, north_star: str) -> JobFinderResult:
        """Run one ReAct loop. Returns result with audit trail."""
        # Reset state
        self._visited = set()
        self._search_qs = []
        self._inserted_ids = []
        self._skipped_dup = 0
        self._notes = []
        total_cost = 0.0  # W15.12 Bug 5: track cost so harness can include it

        messages: list[dict[str, Any]] = [
            {"role": "system",
             "content": _SYSTEM_PROMPT.format(north_star=north_star)},
            # W14.17: kicked out the "first step: search X" hand-holding.
            # LLM should decide its own first move from goal + budget alone.
            {"role": "user", "content": "开始."},
        ]

        finish_reason = "budget_exhausted"
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            try:
                resp = self.llm.chat_with_tools(
                    messages=messages, tools=_TOOL_SCHEMAS,
                    temperature=0.3,
                )
            except LLMError as e:
                self._notes.append(f"iter {iteration} LLM error: {e}")
                finish_reason = f"llm_error: {e}"
                break

            total_cost += resp.cost_usd or 0.0

            # Append the assistant's tool-call announcement to the convo
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": resp.content or "",
            }
            if resp.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(
                                tc.arguments, ensure_ascii=False,
                            ),
                        },
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(assistant_msg)

            if not resp.tool_calls:
                # LLM didn't call any tool — treat as graceful stop
                self._notes.append(
                    f"iter {iteration}: agent stopped without tool_call",
                )
                finish_reason = "no_tool_call"
                break

            # Dispatch each tool call (typically one per turn, but support N)
            done_called = False
            for tc in resp.tool_calls:
                tool_result = self._dispatch(tc.name, tc.arguments)
                self._notes.append(
                    f"iter {iteration}: {tc.name}({_brief_args(tc.arguments)}) "
                    f"→ {tool_result[:80]}",
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })
                if tc.name == "done":
                    done_called = True
                    finish_reason = (
                        f"done: {(tc.arguments.get('reason') or '')[:120]}"
                    )

            if done_called:
                break

        return JobFinderResult(
            iterations=iteration,
            inserted=len(self._inserted_ids),
            skipped_dup=self._skipped_dup,
            new_job_ids=list(self._inserted_ids),
            visited_urls=sorted(self._visited),
            search_queries=list(self._search_qs),
            finish_reason=finish_reason,
            notes=self._notes,
            total_cost_usd=total_cost,
        )

    # ── Tool dispatchers ──────────────────────────────────────────

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "web_search":
                return self._tool_web_search(args)
            if name == "fetch_url":
                return self._tool_fetch_url(args)
            if name == "extract_and_ingest_jd":
                return self._tool_ingest(args)
            if name == "done":
                return f"OK done. ({args.get('reason', 'no reason')})"
            return f"ERROR: unknown tool '{name}'"
        except Exception as e:
            log.exception("tool dispatch crashed: %s(%s)", name, args)
            return f"ERROR: {type(e).__name__}: {e}"

    def _tool_web_search(self, args: dict[str, Any]) -> str:
        q = (args.get("query") or "").strip()
        if not q:
            return "ERROR: query is empty"
        if q in self._search_qs:
            return "ERROR: 这个 query 你 已经搜过了, 换个 query"
        max_n = int(args.get("max_results") or 6)
        max_n = max(1, min(max_n, 10))
        try:
            hits = self.search.search(q, max_results=max_n)
        except Exception as e:
            return f"ERROR: search failed: {e}"
        self._search_qs.append(q)
        if not hits:
            return f"OK: 0 hits for {q!r}. 试别的 query."
        lines = [f"OK: {len(hits)} hits for {q!r}:"]
        for i, h in enumerate(hits, 1):
            lines.append(
                f"  {i}. {h.title[:70]}\n"
                f"     URL: {h.url}\n"
                f"     摘要: {h.snippet[:200]}",
            )
        return "\n".join(lines)[:3500]

    def _tool_fetch_url(self, args: dict[str, Any]) -> str:
        url = (args.get("url") or "").strip()
        if not url:
            return "ERROR: url is empty"
        if url in self._visited:
            return f"ERROR: 你已经 fetch 过 {url}, 别浪费 budget"
        self._visited.add(url)
        try:
            r = self._http.get(url, follow_redirects=True)
        except httpx.HTTPError as e:
            return f"ERROR: fetch failed ({type(e).__name__}: {e})"
        if r.status_code != 200:
            return f"ERROR: HTTP {r.status_code} from {url}"
        ct = r.headers.get("content-type", "")
        if "text/html" not in ct and "text/plain" not in ct:
            return f"ERROR: 非文本类型 ({ct}), 跳过"
        text = _strip_html_to_text(r.text)
        if len(text) < 80:
            return f"ERROR: 抓到的内容太短 ({len(text)} 字), 大概率被反爬"
        return f"OK ({len(text)} chars):\n{text[:6000]}"

    def _tool_ingest(self, args: dict[str, Any]) -> str:
        url = (args.get("url") or "").strip()
        jd_body = (args.get("jd_body") or "").strip()
        company = (args.get("company") or "").strip()[:100]
        title = (args.get("title") or "").strip()[:200]
        location = (args.get("location") or "").strip()[:100] or None
        if not url:
            return "ERROR: url required"
        if not company:
            return "ERROR: company required"
        if not title:
            return "ERROR: title required"
        if len(jd_body) < 200:
            return f"ERROR: jd_body 短于 200 字 ({len(jd_body)}), 不入库"
        if url not in self._visited:
            return (
                f"ERROR: 你没 fetch 过 {url}, 不能 ingest "
                "(可能是你编了 URL / 编了 body, 别凑合)"
            )
        rj = RawJob(
            source="agent_search",
            url=url,
            title=title,
            company=company,
            location=location,
            raw_text=jd_body,
            extras={"via_agent": "job_finder_agent"},
        )
        try:
            was_new, job_id = scout.ingest(self.store, rj)
        except Exception as e:
            return f"ERROR: ingest crashed: {e}"
        if was_new:
            self._inserted_ids.append(job_id)
            return (
                f"OK: ingested as job#{job_id} ({company} · {title}). "
                f"已入 {len(self._inserted_ids)} 个, 目标 3-5。"
            )
        self._skipped_dup += 1
        return f"NOTE: dup, this URL is already job#{job_id}. 看下个 hit。"

    def close(self) -> None:
        self._http.close()


# ── helpers ─────────────────────────────────────────────────────


def _brief_args(args: dict[str, Any]) -> str:
    """Compact one-line args summary for note logs."""
    if not args:
        return ""
    items = []
    for k, v in args.items():
        s = str(v)
        items.append(f"{k}={s[:40]!r}" if len(s) > 40 else f"{k}={v!r}")
    return ", ".join(items)[:120]


_TAG = _re.compile(r"<[^>]+>")
_SPACE = _re.compile(r"[ \t\r]+")
_BLANK = _re.compile(r"\n{3,}")


def _strip_html_to_text(html: str) -> str:
    """HTML → plaintext for LLM consumption."""
    html = _re.sub(r"<script[^>]*>.*?</script>", "",
                   html, flags=_re.DOTALL | _re.IGNORECASE)
    html = _re.sub(r"<style[^>]*>.*?</style>", "",
                   html, flags=_re.DOTALL | _re.IGNORECASE)
    text = _TAG.sub(" ", html)
    text = _SPACE.sub(" ", text)
    text = _BLANK.sub("\n\n", text)
    return text.strip()
