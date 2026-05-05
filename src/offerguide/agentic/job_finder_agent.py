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
                "通过 Tavily 搜索引擎搜一个 query, 返回相关网页的 URL + 标题 + 摘要。"
                "用来找符合 north star 的岗位 / 公司 careers 入口 / 招聘列表。"
                "示例 query: 'AI Agent 暑期实习 字节跳动 2026' / "
                "'LLM 应用工程师 实习 北京 site:nowcoder.com' / "
                "'智谱 AI 校招 招聘'。query 越具体, 结果越精准。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索 query, 用中英文都行, 一句话",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回多少条结果 (默认 6, 最多 10)",
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
                "抓一个 URL 的网页内容 (HTML 转纯文本)。返回页面前 6000 字。"
                "用来看 web_search 返回的某个 URL 的实际内容, 判断这是不是真 JD、"
                "公司 careers 主页、还是无关页面。"
                "如果返回 'fetch failed', 说明这页拿不到 (反爬 / 404), 别再试。"
                "**别 fetch 同一个 URL 两次 — 浪费时间, 我会直接返错。**"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "完整 URL (含 https://)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_and_ingest_jd",
            "description": (
                "**只在你 fetch 了一个页面、确认是真 JD 后**调这个工具。"
                "由你来从页面抽取结构化字段 + 入库到 jobs 表。"
                "返回 job_id (or 'duplicate' if already exists)。"
                "判断标准: 有具体工作内容 + 任职要求 + 公司, 不是 careers 主页 / 多岗位列表。"
                "如果 jd_body 短于 200 字, 别调 — 数据没价值。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "JD 的 URL"},
                    "jd_body": {
                        "type": "string",
                        "description": (
                            "JD 正文 (≥ 200 字, 含工作内容 + 任职要求 + 地点等)。"
                            "你从 fetch 的页面里抽干净, 去 nav/footer 等噪声。"
                        ),
                    },
                    "company": {"type": "string", "description": "公司中文名"},
                    "title": {"type": "string", "description": "岗位 title"},
                    "location": {
                        "type": "string",
                        "description": "工作地点, 没明确就填 '未明确'",
                    },
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
                "**结束本轮检索**。在以下情况调:\n"
                "1. 你已经入库 ≥ 3 个高质量 JD (够这一轮了)\n"
                "2. 你试了 5+ 次但都没找到匹配的, 没必要继续浪费 budget\n"
                "3. 你判断该方向网上信息匮乏 (例: small startup 找不到具体 JD)\n"
                "**不要在没 ingest 任何 JD 时也不 search 就 done。**"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "1 句话: 为什么结束 (找够了 / 找不到 / ...)",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]


_SYSTEM_PROMPT = """你是 OfferGuide 的 JD 检索 agent。

# 任务

给用户找 **3-5 个真实在招的, 跟 north star 匹配的具体岗位 JD**, 用
extract_and_ingest_jd 入库。

# 用户当前 north star

「{north_star}」

# 你能调的工具

- web_search(query, max_results) — Tavily 搜
- fetch_url(url) — 抓网页内容
- extract_and_ingest_jd(url, jd_body, company, title, location) — 入库
- done(reason) — 结束

# 怎么干

1. 想想 north star 里的关键词 (角色 / 公司 / 城市), 构造 1-2 个 search query
2. 调 web_search, 看返回的 URL + snippet
3. 挑最像 JD / careers 入口的 URL → fetch_url 看实际内容
4. 看 fetched 内容判断:
   - **是具体岗位 JD** (有工作内容 + 要求): 抽字段, 调 extract_and_ingest_jd
   - **是 careers 主页 / 多岗位列表**: 别 ingest! 再 search 这家公司更具体的 query
     (例 "字节跳动 AI Agent 实习 招聘 2026"), 拿到具体 JD URL 再 fetch
   - **不相关 / 过期 / 反爬拿不到**: 跳过, 看下一个 hit
5. 重复直到入库 ≥ 3 个, 调 done。

# 重要原则

- **不要 fetch 同一个 URL 两次** (我会直接返错)
- **不要无脑调 done** (没 search 没 fetch 就 done = 偷懒)
- **不要 ingest 你没 fetch 过的 URL** (你不知道是不是 JD)
- **不要 ingest 公司目录页 / 招聘介绍** (raw_text 必须是具体岗位)
- 如果 5 次 fetch 都没拿到真 JD, 调 done(reason="该方向信息匮乏")
- budget: 最多 25 轮工具调用, 优先质量

# 输出

每次决策后, 调一个工具。不要只输出文字 — 必须 tool_call。
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

        messages: list[dict[str, Any]] = [
            {"role": "system",
             "content": _SYSTEM_PROMPT.format(north_star=north_star)},
            {"role": "user",
             "content": "开始检索。第一步: 想清楚要 search 什么 query, 调 web_search。"},
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
