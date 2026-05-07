"""One-shot job evaluation — reactive direct flow, not agent-driven.

Why a separate module instead of going through the agent loop?

The agent loop is for **agent-driven** decisions (find new jobs, when to
notify, what to update in worldview). For **user-driven** "evaluate this
specific JD I'm looking at right now", we KNOW the user wants:

1. Fetch the JD
2. Score the match
3. Get tailor advice
4. Record the evaluation

Going through 8 iter of the agent loop ($0.10, 30-90s) is overkill when
3 direct SKILL invocations + ingest do it in $0.02 / 10-20s. Plus the
UI gets a structured response (score, gaps, advice as data, not markdown
the agent dumps in `final_text`).

This is the "主动 vs 被动" boundary applying:
- Agent main path: discovery, proactive notifications (user passive)
- Direct flow: per-JD evaluation (user active, wants speed)

W15.14 (review answer to "core use case end-to-end"): without this
endpoint, the user pastes a JD and waits 30-90s for an unstructured
markdown blob. Now: 10-20s for a structured card with action buttons.
"""

from __future__ import annotations

import logging
import re as _re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from ..llm import BudgetExceeded, enforce_daily_budget
from .tools import HarnessDeps, _record_event_row, _strip_html_to_text

if TYPE_CHECKING:
    from ..skills import SkillResult, SkillSpec
else:
    SkillResult = Any  # type: ignore[assignment,misc]
    SkillSpec = Any  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structured response for the /api/evaluate-job endpoint.

    Each step's status is independent — fetch can succeed but score
    fail (e.g. no resume). The frontend renders whatever's available.
    """

    # Step 1: fetch + ingest
    job_id: int | None = None
    company: str = ""
    title: str = ""
    location: str = ""
    fetch_status: str = "pending"  # 'ok' | 'dup' | 'error'
    fetch_error: str | None = None

    # Step 2: score_match
    score: float | None = None
    score_reasoning: str = ""
    key_gaps: list[str] = field(default_factory=list)
    score_skill_run_id: int | None = None
    score_status: str = "pending"

    # Step 3: tailor_advice
    tailor_advice: dict[str, Any] | None = None
    tailor_skill_run_id: int | None = None
    tailor_status: str = "pending"

    # Aggregate
    cost_usd: float = 0.0
    duration_ms: int = 0
    user_facing_error: str | None = None
    """Set when something user-actionable fails (e.g. 'paste JD text instead')."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_job(
    *,
    url_or_text: str,
    deps: HarnessDeps,
    company_hint: str | None = None,
    title_hint: str | None = None,
) -> EvaluationResult:
    """Run fetch → score → tailor in one direct flow. Returns structured result.

    Steps are independent: if score fails (no resume), tailor still
    attempts. The frontend renders whatever succeeded.

    Cost: ~$0.02-0.05, ~10-20s real-world. vs full agent loop $0.10-0.30, 30-90s.
    """
    import time
    t0 = time.monotonic()
    result = EvaluationResult()

    raw = (url_or_text or "").strip()
    if not raw:
        result.user_facing_error = "请粘贴 JD 链接或文本"
        result.fetch_status = "error"
        return result

    # W15.15 — daily budget guard. evaluate flow is cheaper than full agent
    # loop ($0.02 vs $0.10) but still enforce — cumulative paste-spam can
    # add up.
    try:
        enforce_daily_budget(deps.store)
    except BudgetExceeded as e:
        result.user_facing_error = str(e)
        result.fetch_status = "error"
        log.warning("evaluate: refusing, %s", e)
        return result

    # ── Step 1: fetch + ingest ─────────────────────────────────────
    job_id, fetch_err = _fetch_and_ingest(
        raw, deps,
        company_hint=company_hint or "",
        title_hint=title_hint or "",
    )
    if fetch_err:
        result.fetch_status = "error"
        result.fetch_error = fetch_err
        result.user_facing_error = fetch_err
        return result
    result.job_id = job_id
    result.fetch_status = "ok"
    job_row = _load_job_meta(deps, job_id)
    if job_row is None:
        result.fetch_status = "error"
        result.user_facing_error = f"job#{job_id} not retrievable after ingest"
        return result
    result.company = job_row["company"]
    result.title = job_row["title"]
    result.location = job_row.get("location") or ""

    # Record an audit event so /debug shows the user pasted this
    try:
        _record_event_row(
            deps, kind="evaluated", job_id=job_id,
            note=f"user paste, fetch_status={result.fetch_status}",
        )
    except Exception as e:
        log.warning("evaluate: record_event failed: %s", e)

    # ── Step 2: score_match ─────────────────────────────────────────
    if deps.runtime is None or not deps.user_profile_text:
        result.score_status = "skipped"
    else:
        score_spec = deps.find_skill("score_match")
        if score_spec is None:
            result.score_status = "skipped"
        else:
            sr = _invoke_skill(
                deps, score_spec,
                inputs={
                    "job_title": job_row["title"],
                    "job_company": job_row["company"],
                    "jd_text": (job_row.get("raw_text") or "")[:4000],
                    "candidate_resume": deps.user_profile_text[:4000],
                },
            )
            if sr is None:
                result.score_status = "error"
            elif sr.parsed is None:
                result.score_status = "error"
                result.score_reasoning = f"(SKILL JSON parse failed; raw: {sr.raw_text[:200]})"
                result.score_skill_run_id = sr.skill_run_id
            else:
                p = sr.parsed
                result.score = _safe_float(p.get("score"))
                result.score_reasoning = (p.get("reasoning") or "")[:1500]
                gaps = p.get("key_gaps") or []
                if isinstance(gaps, list):
                    result.key_gaps = [str(g)[:200] for g in gaps[:8]]
                result.score_skill_run_id = sr.skill_run_id
                result.score_status = "ok"
                result.cost_usd += sr.cost_usd or 0.0

    # ── Step 3: tailor_advice ────────────────────────────────────────
    if deps.runtime is None or not deps.user_profile_text:
        result.tailor_status = "skipped"
    else:
        tailor_spec = deps.find_skill("tailor_resume")
        if tailor_spec is None:
            result.tailor_status = "skipped"
        else:
            sr = _invoke_skill(
                deps, tailor_spec,
                inputs={
                    "job_title": job_row["title"],
                    "company": job_row["company"],
                    "jd_text": (job_row.get("raw_text") or "")[:4000],
                    "current_resume": deps.user_profile_text[:6000],
                },
            )
            if sr is None:
                result.tailor_status = "error"
            elif sr.parsed is None:
                result.tailor_status = "error"
                result.tailor_advice = {"_raw": sr.raw_text[:500]}
                result.tailor_skill_run_id = sr.skill_run_id
            else:
                result.tailor_advice = sr.parsed
                result.tailor_skill_run_id = sr.skill_run_id
                result.tailor_status = "ok"
                result.cost_usd += sr.cost_usd or 0.0

    result.duration_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        "evaluate_job: job#%s (%s · %s) score=%s, %dms, $%.4f",
        job_id, result.company, result.title,
        result.score, result.duration_ms, result.cost_usd,
    )
    return result


# ── helpers ─────────────────────────────────────────────────────────


def _fetch_and_ingest(
    raw: str, deps: HarnessDeps, *, company_hint: str, title_hint: str,
) -> tuple[int, str | None]:
    """Mirror of tools._exec_fetch_jd's logic, but returns (job_id, error_msg).

    Returns (job_id, None) on success, (0, err_msg) on failure.
    """
    from ..platforms import RawJob
    from ..workers import scout

    if raw.startswith(("http://", "https://")):
        try:
            r = httpx.get(
                raw, follow_redirects=True, timeout=12.0,
                headers={"User-Agent": "Mozilla/5.0"},
            )
        except httpx.HTTPError as e:
            return (
                0,
                f"无法抓取这个链接 ({type(e).__name__}). "
                "国内招聘平台 (BOSS/牛客/拉勾) 一般都反爬, 请直接粘 JD 文本.",
            )
        if r.status_code != 200:
            return (
                0,
                f"链接返回 HTTP {r.status_code}, 大概率反爬. "
                "请直接粘 JD 文本.",
            )
        body_text = _strip_html_to_text(r.text)
        if len(body_text) < 200:
            return (
                0,
                f"页面正文太短 ({len(body_text)} 字符), 大概率反爬. "
                "请直接粘 JD 文本.",
            )
        rj = RawJob(
            source="user_paste_url",
            url=raw,
            title=title_hint or _guess_title_from_text(body_text) or "未识别岗位",
            company=company_hint or _guess_company_from_url(raw) or "未识别公司",
            location=None,
            raw_text=body_text[:10000],
            extras={"via": "evaluate_endpoint"},
        )
    else:
        if len(raw) < 200:
            return (
                0,
                f"JD 文本太短 ({len(raw)} 字符) — 至少 200 字, 请粘完整 JD",
            )
        import hashlib
        url_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        rj = RawJob(
            source="user_paste_text",
            url=f"paste://{url_digest}",
            title=title_hint or "粘贴的 JD",
            company=company_hint or "未指定公司",
            location=None,
            raw_text=raw[:10000],
            extras={"via": "evaluate_endpoint"},
        )

    _was_new, job_id = scout.ingest(deps.store, rj)
    return job_id, None


def _invoke_skill(
    deps: HarnessDeps, spec: Any, *, inputs: dict[str, Any],
) -> SkillResult | None:
    if deps.runtime is None:
        return None
    try:
        return deps.runtime.invoke(spec, inputs)
    except Exception as e:
        log.exception("evaluate: SKILL %s invoke failed: %s", spec.name, e)
        return None


def _load_job_meta(deps: HarnessDeps, job_id: int) -> dict[str, Any] | None:
    with deps.store.connect() as conn:
        row = conn.execute(
            "SELECT id, title, company, location, raw_text "
            "FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0], "title": row[1], "company": row[2],
        "location": row[3], "raw_text": row[4],
    }


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _guess_title_from_text(body: str) -> str | None:
    for line in body.splitlines():
        line = line.strip()
        if 5 < len(line) < 80:
            return line
    return None


def _guess_company_from_url(url: str) -> str | None:
    m = _re.search(r"https?://(?:www\.)?([^/]+)", url)
    if m:
        host = m.group(1).split(":")[0]
        parts = host.split(".")
        if len(parts) >= 2:
            return parts[-2].title()
    return None
