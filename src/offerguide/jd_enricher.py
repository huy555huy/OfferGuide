"""JD enricher — turns spider-discovered "company portal entry" rows into
SKILL-ready JDs.

The audit pinpoint: ``awesome_jobs`` spider yields rows with raw_text ~138
bytes (just company name + portal URL + section). Below the
``MIN_TEXT_FOR_AUTO_EVAL = 200`` threshold, so ``score_match`` /
``tailor_resume`` / 4-bucket Gap all skip auto-eval. The entire main
chain breaks at the very first SKILL.

This module is the bridge: for any job row with thin raw_text + a real
URL, fetch the URL → strip HTML → ask LLM to extract a structured JD
(title / responsibilities / requirements / nice-to-have / location /
team) → write back to ``jobs.raw_text`` as a much richer document.

Failure modes handled:

- **SSR-rendered portal** (Vue/React without SSR) returns ~empty HTML →
  enricher records ``enrich_status = 'js_rendered'`` and surfaces in UI
  so user can manually paste JD body
- **404 / rate-limited** → ``enrich_status = 'fetch_failed'``,
  retry-friendly tomorrow
- **LLM extracts nothing useful** → ``enrich_status = 'extracted_thin'``

Each enriched job records ``extras_json.enrich_status`` so the daemon
knows which to retry vs leave alone.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .llm import LLMClient, LLMError
from .memory import Store

log = logging.getLogger(__name__)

EnrichStatus = Literal[
    "ok",                # successfully enriched, raw_text >= 200
    "js_rendered",       # HTML had no real content (Vue/React SSR-less)
    "fetch_failed",      # HTTP error / timeout
    "extracted_thin",    # LLM extracted but result still too short
    "no_url",            # job row has no URL to fetch
    "no_llm",            # LLM not configured, can't extract
    "skipped_already",   # already enriched, status=ok
]

# Length thresholds — match auto_pipeline.MIN_TEXT_FOR_AUTO_EVAL
MIN_RAW_TEXT_AFTER_ENRICH = 200
"""Below this even after enrichment, status='extracted_thin'."""

MIN_USEFUL_HTML_TEXT = 300
"""HTML text body below this = probably JS-rendered shell page."""


@dataclass
class EnrichResult:
    """Per-job enrich outcome — used for daemon counters + UI display."""
    job_id: int
    status: EnrichStatus
    new_raw_text_len: int = 0
    note: str = ""


# ─────────── HTTP fetch + HTML strip ─────────────────────────────


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_BLANK_RE = re.compile(r"\n{3,}")
_SPACE_RE = re.compile(r"[ \t]+")


def _fetch_html(url: str, *, timeout_s: float = 12.0) -> str | None:
    """GET the URL with a polite UA. Returns None on transport failure."""
    try:
        with httpx.Client(timeout=timeout_s) as c:
            resp = c.get(
                url,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/605.1.15 Safari/605.1.15"
                    ),
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
                },
            )
    except httpx.HTTPError as e:
        log.warning("jd_enricher fetch error %s: %s", url, e)
        return None
    if resp.status_code >= 400:
        log.warning("jd_enricher HTTP %d for %s", resp.status_code, url)
        return None
    ct = resp.headers.get("content-type", "")
    if "text/html" not in ct and "text/plain" not in ct:
        return None
    return resp.text


def _html_to_text(html: str) -> str:
    """HTML → plaintext. Drops <script>/<style>; collapses whitespace."""
    if not html:
        return ""
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", html)
    text = _SPACE_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


# ─────────── LLM-based JD extraction ─────────────────────────────


_EXTRACT_PROMPT = """你是 OfferGuide 的 JD 抽取员。给一段公司招聘页的纯文本，
抽取这个**最相关岗位**的结构化 JD。返回严格 JSON：

{{
  "page_kind": "single_jd" | "job_list" | "career_landing" | "other",
  "best_role_title":  <str | null, 页面里和'{role_hint}'最相关的岗位标题>,
  "responsibilities": <list[str], 工作职责, 5-10 条>,
  "requirements":     <list[str], 硬要求, 4-8 条>,
  "nice_to_have":     <list[str], 加分项, 0-5 条>,
  "team_or_business": <str | null, 所属团队/业务方向>,
  "location":         <str | null>,
  "salary_or_level":  <str | null, 如果有的话>,
  "deadline":         <str | null, 投递截止日期>,
  "rationale":        <str, 一句话, 为什么这个岗位是最相关的>
}}

如果页面是 job_list，挑和 role_hint 最匹配的一条；如果是 career_landing
（只列入口）或 other (无岗位信息), 把所有 list 字段填空数组。

只返回 JSON，不要 markdown 代码块。
"""


def _extract_with_llm(
    *,
    html_text: str,
    company: str,
    role_hint: str,
    llm: LLMClient,
) -> dict[str, Any] | None:
    """Run extraction LLM call. Returns parsed dict or None on failure."""
    snippet = html_text[:8000]  # cap to keep cost predictable
    user_msg = (
        f"【目标公司】{company}\n"
        f"【关注岗位线索】{role_hint or '不指定，挑最相关的'}\n\n"
        f"【页面文本】\n{snippet}"
    )
    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": _EXTRACT_PROMPT.format(
                    role_hint=role_hint or "AI / 算法 / 后端"
                )},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            json_mode=True,
        )
    except LLMError as e:
        log.warning("jd_enricher LLM failed: %s", e)
        return None
    try:
        return json.loads(resp.content)
    except json.JSONDecodeError:
        log.warning("jd_enricher LLM returned non-JSON")
        return None


def _format_extracted_as_raw_text(
    *, extracted: dict[str, Any], original_raw: str, company: str,
) -> str:
    """Compose a structured raw_text from the extracted dict.

    Format is intentionally markdown-ish so SKILL prompts (which read
    raw_text as part of the user message) get a readable body.
    """
    lines: list[str] = [
        f"# {company} · {extracted.get('best_role_title') or '岗位详情'}",
        "",
    ]
    if extracted.get("team_or_business"):
        lines.append(f"**团队/业务**: {extracted['team_or_business']}")
    if extracted.get("location"):
        lines.append(f"**地点**: {extracted['location']}")
    if extracted.get("salary_or_level"):
        lines.append(f"**薪资/层级**: {extracted['salary_or_level']}")
    if extracted.get("deadline"):
        lines.append(f"**投递截止**: {extracted['deadline']}")
    lines.append("")
    if extracted.get("responsibilities"):
        lines.append("## 工作职责")
        lines.extend(f"- {r}" for r in extracted["responsibilities"])
        lines.append("")
    if extracted.get("requirements"):
        lines.append("## 任职要求")
        lines.extend(f"- {r}" for r in extracted["requirements"])
        lines.append("")
    if extracted.get("nice_to_have"):
        lines.append("## 加分项")
        lines.extend(f"- {r}" for r in extracted["nice_to_have"])
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 入口元数据 (来自 spider)")
    lines.append(original_raw)
    return "\n".join(lines)


# ─────────── DB-aware enrich ─────────────────────────────


def enrich_one(
    store: Store,
    *,
    job_id: int,
    llm: LLMClient | None,
    role_hint: str = "",
    fetch_delay_s: float = 1.5,
) -> EnrichResult:
    """Enrich a single job row. Idempotent — already-enriched rows skip.

    ``fetch_delay_s`` is a polite gap before HTTP fetch (per-host rate
    limiting; we don't have a global throttle for this spider's URLs
    since they hit many domains).
    """
    with store.connect() as conn:
        row = conn.execute(
            "SELECT company, url, raw_text, extras_json FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return EnrichResult(job_id=job_id, status="fetch_failed",
                            note="job row not found")

    company, url, raw_text, extras_json = row
    extras: dict[str, Any] = json.loads(extras_json) if extras_json else {}

    if extras.get("enrich_status") == "ok":
        return EnrichResult(
            job_id=job_id, status="skipped_already",
            new_raw_text_len=len(raw_text or ""),
        )
    if not url:
        _persist_status(store, job_id=job_id, status="no_url")
        return EnrichResult(job_id=job_id, status="no_url")
    if llm is None:
        _persist_status(store, job_id=job_id, status="no_llm")
        return EnrichResult(job_id=job_id, status="no_llm",
                            note="LLM not configured")

    # ── fetch
    if fetch_delay_s > 0:
        time.sleep(fetch_delay_s)
    html = _fetch_html(url)
    if html is None:
        _persist_status(store, job_id=job_id, status="fetch_failed")
        return EnrichResult(job_id=job_id, status="fetch_failed",
                            note="HTTP error or non-html response")

    text = _html_to_text(html)
    if len(text) < MIN_USEFUL_HTML_TEXT:
        _persist_status(store, job_id=job_id, status="js_rendered")
        return EnrichResult(
            job_id=job_id, status="js_rendered",
            note=f"HTML body only {len(text)} chars — JS-rendered?",
        )

    # ── LLM extract
    extracted = _extract_with_llm(
        html_text=text, company=company or "", role_hint=role_hint, llm=llm,
    )
    if extracted is None:
        _persist_status(store, job_id=job_id, status="extracted_thin",
                        extra_notes={"llm_failed": True})
        return EnrichResult(job_id=job_id, status="extracted_thin",
                            note="LLM extraction failed")

    # Page-kind reality check
    if extracted.get("page_kind") in ("career_landing", "other"):
        _persist_status(store, job_id=job_id, status="extracted_thin",
                        extra_notes={"page_kind": extracted.get("page_kind")})
        return EnrichResult(
            job_id=job_id, status="extracted_thin",
            note=f"page_kind={extracted.get('page_kind')} — no specific JD",
        )

    new_raw = _format_extracted_as_raw_text(
        extracted=extracted, original_raw=raw_text or "",
        company=company or "",
    )
    if len(new_raw) < MIN_RAW_TEXT_AFTER_ENRICH:
        _persist_status(store, job_id=job_id, status="extracted_thin",
                        extra_notes={"new_len": len(new_raw)})
        return EnrichResult(job_id=job_id, status="extracted_thin",
                            note=f"after enrich still only {len(new_raw)} chars")

    # Persist enriched raw_text + bump title if extracted has better
    new_extras = dict(extras)
    new_extras["enrich_status"] = "ok"
    new_extras["enriched_at"] = time.time()
    new_extras["original_raw_text"] = raw_text  # preserved for audit

    new_title = extracted.get("best_role_title")
    with store.connect() as conn:
        if new_title:
            conn.execute(
                "UPDATE jobs SET raw_text = ?, title = ?, extras_json = ? "
                "WHERE id = ?",
                (new_raw, new_title, json.dumps(new_extras, ensure_ascii=False), job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET raw_text = ?, extras_json = ? WHERE id = ?",
                (new_raw, json.dumps(new_extras, ensure_ascii=False), job_id),
            )

    return EnrichResult(
        job_id=job_id, status="ok", new_raw_text_len=len(new_raw),
    )


def _persist_status(
    store: Store,
    *,
    job_id: int,
    status: EnrichStatus,
    extra_notes: dict[str, Any] | None = None,
) -> None:
    """Update only extras_json with the enrich status (no raw_text change)."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT extras_json FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return
        extras: dict[str, Any] = json.loads(row[0]) if row[0] else {}
        extras["enrich_status"] = status
        extras["enriched_at"] = time.time()
        if extra_notes:
            extras.setdefault("enrich_notes", {}).update(extra_notes)
        conn.execute(
            "UPDATE jobs SET extras_json = ? WHERE id = ?",
            (json.dumps(extras, ensure_ascii=False), job_id),
        )


def enrich_pending(
    store: Store,
    *,
    llm: LLMClient | None,
    limit: int = 10,
    role_hint: str = "",
) -> dict[str, int]:
    """Sweep jobs that haven't been enriched yet. Returns counters.

    "Not enriched" = ``raw_text`` length < MIN_RAW_TEXT_AFTER_ENRICH AND
    ``extras_json.enrich_status`` not set / not 'ok'.

    Per-call cap to bound LLM token spend.
    """
    counters = {
        "scanned": 0,
        "ok": 0,
        "js_rendered": 0,
        "fetch_failed": 0,
        "extracted_thin": 0,
        "no_url": 0,
        "no_llm": 0,
        "skipped_already": 0,
    }
    with store.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id FROM jobs
            WHERE LENGTH(raw_text) < {MIN_RAW_TEXT_AFTER_ENRICH}
              AND (
                  extras_json IS NULL
                  OR json_extract(extras_json, '$.enrich_status') IS NULL
                  OR json_extract(extras_json, '$.enrich_status') NOT IN
                     ('ok', 'js_rendered', 'fetch_failed', 'extracted_thin', 'no_url')
              )
            ORDER BY fetched_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    for r in rows:
        result = enrich_one(
            store, job_id=int(r[0]), llm=llm, role_hint=role_hint,
        )
        counters["scanned"] += 1
        counters[result.status] += 1
    return counters
