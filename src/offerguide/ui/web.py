"""FastAPI web UI — chat + inbox.

Routes:

    GET  /                     home page: chat form + recent inbox preview
    POST /chat                 run agent on (job_text, action) → render report fragment
    GET  /inbox                full inbox view (pending + recent decided)
    POST /inbox/{id}/decide    mark item approved|rejected|dismissed
    POST /inbox/from-report    enqueue a "consider_jd" item from the latest chat report

The app is intentionally HTMX-driven (no SPA, no JS framework). Server-side
templates render fragments; the client just swaps DOM nodes. Easier to test,
easier to ship as a small local tool.

Application-factory pattern (`create_app(...)`) lets tests inject stub stores,
runtimes, profiles, and notifiers without touching env vars.
"""

from __future__ import annotations

import re
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .. import inbox as inbox_mod
from ..agent import build_graph
from ..agent.state import RequestedAction
from ..config import Settings
from ..llm import LLMClient, LLMError
from ..memory import Store
from ..platforms._spec import RawJob
from ..profile import UserProfile, load_resume_pdf
from ..skills import SkillRuntime, SkillSpec, discover_skills
from ..workers import scout
from .notify import Notifier, make_notifier

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(
    *,
    settings: Settings,
    store: Store,
    profile: UserProfile | None,
    skills: list[SkillSpec],
    runtime: SkillRuntime | None,
    notifier: Notifier | None = None,
) -> FastAPI:
    """Build the FastAPI application with explicit dependencies (testable)."""
    app = FastAPI(title="OfferGuide", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
        base = {
            "request": request,
            "profile_loaded": profile is not None,
            "profile_chars": len(profile.raw_resume_text) if profile else 0,
        }
        base.update(extra)
        return base

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> Any:
        items = inbox_mod.list_items(store, status="pending", limit=10)
        return templates.TemplateResponse(
            request,
            "home.html",
            _ctx(
                request,
                items=items,
                stats=_quick_stats(store),
                active_tab="home",
            ),
        )

    @app.get("/inbox", response_class=HTMLResponse)
    def inbox_view(request: Request) -> Any:
        pending = inbox_mod.list_items(store, status="pending", limit=100)
        decided = (
            inbox_mod.list_items(store, status="approved", limit=20)
            + inbox_mod.list_items(store, status="rejected", limit=20)
            + inbox_mod.list_items(store, status="dismissed", limit=10)
        )
        decided.sort(key=lambda i: i.decided_at or i.created_at, reverse=True)
        return templates.TemplateResponse(
            request,
            "inbox.html",
            _ctx(
                request,
                items=pending,
                decided=decided[:50],
                include_decided=bool(decided),
                active_tab="inbox",
            ),
        )

    @app.get("/compare", response_class=HTMLResponse)
    def compare_view(request: Request, company: str = "") -> Any:
        """List companies with ≥2 jobs; show comparison form for one company."""
        from ..skills.compare_jobs.helpers import lookup_application_limit

        company_groups = _list_company_groups(store)

        target_jobs: list[dict] | None = None
        target_limit: int | None = None
        if company:
            target_jobs = _list_jobs_for_company(store, company)
            target_limit = lookup_application_limit(company)

        return templates.TemplateResponse(
            request,
            "compare.html",
            _ctx(
                request,
                company_groups=company_groups,
                selected_company=company,
                target_jobs=target_jobs,
                target_limit=target_limit,
                active_tab="compare",
            ),
        )

    @app.post("/compare/run", response_class=HTMLResponse)
    def compare_run(
        request: Request,
        company: str = Form(...),
        application_limit: str = Form(""),
        job_ids: list[str] = Form(...),  # noqa: B008
    ) -> Any:
        """Run compare_jobs SKILL on selected jobs."""
        if profile is None:
            return templates.TemplateResponse(
                request, "_compare_result.html",
                _ctx(request, error="未加载简历——设 OFFERGUIDE_RESUME_PDF 后重启。"),
            )
        if runtime is None:
            return templates.TemplateResponse(
                request, "_compare_result.html",
                _ctx(request, error="未配置 LLM——设 DEEPSEEK_API_KEY 后重启。"),
            )

        # Resolve job_ids → full job records
        try:
            ids = [int(s) for s in job_ids if s]
        except ValueError:
            raise HTTPException(400, "job_ids must be integers") from None
        if len(ids) < 2:
            return templates.TemplateResponse(
                request, "_compare_result.html",
                _ctx(request, error="至少要选 2 个职位才有比较的意义。"),
            )
        if len(ids) > 10:
            return templates.TemplateResponse(
                request, "_compare_result.html",
                _ctx(request, error="一次最多比较 10 个职位（避免 LLM 上下文过载）。"),
            )

        rows = _list_jobs_by_ids(store, ids)
        import json as _json
        jobs_json = _json.dumps(
            [
                {"job_id": r["id"], "title": r["title"] or "(无标题)",
                 "raw_text": r["raw_text"][:1500],
                 "source": r["source"]}
                for r in rows
            ],
            ensure_ascii=False,
        )

        # Find SKILL spec
        spec = next((s for s in skills if s.name == "compare_jobs"), None)
        if spec is None:
            return templates.TemplateResponse(
                request, "_compare_result.html",
                _ctx(request, error="compare_jobs SKILL 未加载，检查 skills/ 目录。"),
            )

        try:
            result = runtime.invoke(
                spec,
                {
                    "company": company,
                    "user_profile": profile.raw_resume_text,
                    "jobs_json": jobs_json,
                },
            )
        except LLMError as e:
            return templates.TemplateResponse(
                request, "_compare_result.html",
                _ctx(request, error=f"LLM 调用失败: {e}"),
            )

        # Build a job_id → full job record lookup so the template can
        # link rankings back to the source jobs
        job_by_id = {r["id"]: r for r in rows}

        return templates.TemplateResponse(
            request,
            "_compare_result.html",
            _ctx(
                request,
                comparison=result.parsed,
                run_id=result.skill_run_id,
                job_by_id=job_by_id,
                company=company,
                application_limit_user=application_limit,
            ),
        )

    @app.get("/applications", response_class=HTMLResponse)
    def applications_view(request: Request) -> Any:
        rows = _list_applications_with_events(store)
        active = sum(1 for r in rows if r["status"] not in ("rejected", "offer", "withdrawn"))
        return templates.TemplateResponse(
            request,
            "applications.html",
            _ctx(
                request,
                applications=rows,
                active_count=active,
                terminal_count=len(rows) - active,
                active_tab="applications",
            ),
        )

    @app.post("/api/applications/{app_id}/event", response_class=HTMLResponse)
    def applications_log_event(
        request: Request,
        app_id: int,
        kind: str = Form(...),
    ) -> Any:
        from .. import application_events as ae
        from ..state_machine import sync_status

        valid_kinds = {
            "submitted", "viewed", "replied", "assessment",
            "interview", "rejected", "offer", "withdrawn",
        }
        if kind not in valid_kinds:
            raise HTTPException(400, f"unknown event kind: {kind}")
        try:
            ae.record(store, application_id=app_id, kind=kind, source="manual")  # type: ignore[arg-type]
        except Exception as e:
            raise HTTPException(400, f"failed to record: {e}") from None
        sync_status(store, app_id, kind)

        # Re-render only this row so HTMX can swap it in place
        rows = _list_applications_with_events(store, where_id=app_id)
        if not rows:
            raise HTTPException(404, f"application {app_id} not found after event")
        return templates.TemplateResponse(
            request,
            "_application_card.html",
            _ctx(request, app=rows[0]),
        )

    @app.get("/interviews", response_class=HTMLResponse)
    def interviews_view(request: Request, company: str = "") -> Any:
        """List 面经 corpus + paste-in form for adding more."""
        companies = _list_interview_companies(store)
        experiences = (
            _list_interview_experiences(store, company=company)
            if company
            else _list_interview_experiences(store, limit=20)
        )
        return templates.TemplateResponse(
            request, "interviews.html",
            _ctx(
                request,
                companies=companies,
                experiences=experiences,
                selected_company=company,
                active_tab="interviews",
            ),
        )

    @app.post("/api/interviews/paste", response_class=HTMLResponse)
    def interviews_paste(
        request: Request,
        company: str = Form(...),
        raw_text: str = Form(...),
        source: str = Form("manual_paste"),
        role_hint: str = Form(""),
        source_url: str = Form(""),
    ) -> Any:
        from .. import interview_corpus
        if not company.strip() or not raw_text.strip():
            raise HTTPException(400, "company 和 raw_text 都必填")
        try:
            was_new, exp_id = interview_corpus.insert(
                store,
                company=company.strip(),
                raw_text=raw_text.strip(),
                source=source.strip() or "manual_paste",
                role_hint=role_hint.strip() or None,
                source_url=source_url.strip() or None,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

        # Return the updated list fragment for HTMX swap
        experiences = _list_interview_experiences(store, company=company.strip())
        return templates.TemplateResponse(
            request, "_interview_list.html",
            _ctx(request, experiences=experiences, just_added=exp_id, was_new=was_new),
        )

    @app.post("/api/email/classify", response_class=JSONResponse)
    def email_classify_endpoint(payload: EmailClassifyPayload) -> dict:
        """Classify pasted-in email text(s) → event kinds.

        Two modes:
        - ``mode='regex'``: pure-Python regex (free, deterministic, dumb)
        - ``mode='llm'``: LLM-driven classification (real understanding,
          extracts structured info like interview_time / contact_name /
          referenced_role; requires DEEPSEEK_API_KEY)
        - ``mode='auto'``: llm if configured, else regex fallback (default)
        """
        # Build the company → app_ids index from real DB state
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT j.company, a.id FROM applications a "
                "JOIN jobs j ON j.id = a.job_id "
                "WHERE j.company IS NOT NULL AND j.company != '' "
                "AND a.status NOT IN ('rejected', 'offer', 'withdrawn')"
            ).fetchall()
        known_apps_by_company: dict[str, list[int]] = {}
        known_companies: list[str] = []
        for company, app_id in rows:
            known_apps_by_company.setdefault(company, []).append(app_id)
            if company not in known_companies:
                known_companies.append(company)

        # Resolve mode
        chosen_mode = payload.mode
        if chosen_mode == "auto":
            chosen_mode = "llm" if settings.deepseek_api_key else "regex"

        if payload.batch:
            from .. import email_classifier as ec
            chunks = ec.split_email_dump(payload.text)
        else:
            chunks = [payload.text] if payload.text.strip() else []

        if chosen_mode == "llm":
            from ..agentic.email_classifier_llm import classify_email_batch_llm
            from ..llm import LLMClient
            llm = LLMClient(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                default_model=settings.default_model,
            )
            llm_results = classify_email_batch_llm(
                chunks, llm=llm,
                known_companies=known_companies,
                known_apps_by_company=known_apps_by_company,
            )
            return {
                "count": len(llm_results),
                "mode": "llm",
                "results": [
                    {
                        "kind": r.kind,
                        "confidence": r.confidence,
                        "matched_company": r.matched_company,
                        "matched_application_id": r.matched_application_id,
                        "extracted": r.extracted,
                        "evidence": r.evidence,
                    }
                    for r in llm_results
                ],
            }

        # regex fallback
        from .. import email_classifier as ec
        regex_results = ec.classify_batch(
            chunks,
            known_companies=known_companies,
            known_apps_by_company=known_apps_by_company,
        )
        return {
            "count": len(regex_results),
            "mode": "regex",
            "results": [
                {
                    "kind": r.kind,
                    "confidence": r.confidence,
                    "matched_company": r.matched_company,
                    "matched_application_id": r.matched_application_id,
                    "extracted": {},  # regex has no structured extraction
                    "evidence": r.evidence,
                }
                for r in regex_results
            ],
        }

    @app.post("/api/agent/sweep", response_class=JSONResponse)
    def agent_sweep_endpoint(payload: SweepPayload) -> dict:
        """Run a meta-agent sweep on one company.

        Combines application summary (always) + agentic 面经 collection
        (when LLM + search are configured). Use this instead of asking
        the user to manually paste 面经.
        """
        from ..agentic import build_default_search, sweep_company
        from ..llm import LLMClient

        llm: LLMClient | None = None
        if settings.deepseek_api_key:
            llm = LLMClient(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                default_model=settings.default_model,
            )

        search = build_default_search() if payload.do_corpus else None

        result = sweep_company(
            payload.company,
            store=store,
            llm=llm,
            search=search,
            do_corpus=payload.do_corpus,
            role_hint=payload.role_hint or None,
        )

        return {
            "company": result.company,
            "application_summary": result.application_summary,
            "interview_corpus": (
                {
                    "queries_run": result.interview_corpus.queries_run,
                    "hits_seen": result.interview_corpus.hits_seen,
                    "hits_evaluated": result.interview_corpus.hits_evaluated,
                    "inserted": result.interview_corpus.inserted,
                    "skipped_dup": result.interview_corpus.skipped_dup,
                    "skipped_low_quality": result.interview_corpus.skipped_low_quality,
                    "notes": result.interview_corpus.notes,
                }
                if result.interview_corpus
                else None
            ),
            "notes": result.notes,
        }

    @app.post("/api/applications/{app_id}/events/ics", response_class=JSONResponse)
    def applications_log_ics(
        app_id: int,
        ics_text: str = Form(...),
    ) -> dict:
        """Upload an ICS calendar file → record interview event(s)."""
        from .. import application_events as ae
        from .. import ics_parser
        from ..state_machine import sync_status

        events = ics_parser.parse_ics(ics_text)
        chosen = ics_parser.select_first_interview(events)
        if chosen is None:
            raise HTTPException(
                400,
                "ICS file did not contain a recognizable interview event "
                "(no 面试/interview keyword in summary/description).",
            )

        occurred_at = (
            ics_parser.datetime_to_julianday(chosen.dtstart_utc)
            if chosen.dtstart_utc
            else None
        )
        try:
            ae.record(
                store,
                application_id=app_id,
                kind="interview",
                source="calendar",
                occurred_at=occurred_at,
                payload={
                    "summary": chosen.summary[:200],
                    "scheduled_at": (
                        chosen.dtstart_utc.isoformat()
                        if chosen.dtstart_utc
                        else None
                    ),
                    "description": chosen.description[:500],
                    "ics_event_count": len(events),
                },
            )
        except Exception as e:
            raise HTTPException(400, f"failed to record: {e}") from None
        sync_status(store, app_id, "interview")

        return {
            "ok": True,
            "application_id": app_id,
            "scheduled_at": (
                chosen.dtstart_utc.isoformat() if chosen.dtstart_utc else None
            ),
            "summary": chosen.summary,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_view(request: Request) -> Any:
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            _ctx(
                request,
                stats=_full_stats(store),
                funnel=_application_funnel(store),
                evolutions=_recent_evolutions(store, limit=10),
                recent_runs=_recent_skill_runs(store, limit=10),
                active_tab="dashboard",
            ),
        )

    @app.post("/chat", response_class=HTMLResponse)
    def chat(
        request: Request,
        job_text: str = Form(...),
        action: str = Form("score_and_gaps"),
        company: str = Form(""),
    ) -> Any:
        if profile is None:
            return templates.TemplateResponse(
                request,
                "_report.html",
                _ctx(request, error="未加载简历——设 OFFERGUIDE_RESUME_PDF 后重启。"),
            )
        if runtime is None:
            return templates.TemplateResponse(
                request,
                "_report.html",
                _ctx(request, error="未配置 LLM——设 DEEPSEEK_API_KEY 后重启。"),
            )

        valid_actions = (
            "score", "gaps", "score_and_gaps", "prepare_interview", "deep_prep",
            "everything",
        )
        action_norm: RequestedAction = (
            action if action in valid_actions else "score_and_gaps"
        )  # type: ignore[assignment]

        # If user picked an action that needs `company` and didn't provide one,
        # surface the requirement clearly rather than silently falling back.
        company_required = ("prepare_interview", "deep_prep", "everything")
        if action_norm in company_required and not company.strip():
            return templates.TemplateResponse(
                request,
                "_report.html",
                _ctx(
                    request,
                    error=(
                        "面试备战 / 三件套需要填「公司名」字段。请在表单里补充后重试。"
                    ),
                ),
            )

        graph = build_graph(skills=skills, runtime=runtime, store=store)
        try:
            result = graph.invoke(
                {
                    "messages": [{"role": "user", "content": job_text}],
                    "requested_action": action_norm,
                    "job_text": job_text,
                    "user_profile_text": profile.raw_resume_text,
                    "company": company.strip() or None,
                }
            )
        except LLMError as e:
            return templates.TemplateResponse(
                request,
                "_report.html",
                _ctx(request, error=f"LLM 调用失败: {e}"),
            )

        title_first_line = (job_text.splitlines() or [""])[0][:60]
        return templates.TemplateResponse(
            request,
            "_report.html",
            _ctx(
                request,
                response=result.get("final_response") or "(空响应)",
                error=result.get("error"),
                # Structured agent results — templates render visualizations
                # off these dicts; the markdown ``response`` is the fallback.
                score=result.get("score_result"),
                gaps=result.get("gaps_result"),
                prep=result.get("prep_result"),
                prep_used_experiences=result.get("prep_used_experiences", 0),
                deep_prep=result.get("deep_prep_result"),
                company=company.strip() or None,
                inbox_title=f"考虑投递: {title_first_line}",
                inbox_body=(result.get("final_response") or "")[:800],
                job_text=job_text[:2000],
                score_run_id=result.get("score_run_id"),
                gaps_run_id=result.get("gaps_run_id"),
                prep_run_id=result.get("prep_run_id"),
                deep_prep_run_id=result.get("deep_prep_run_id"),
            ),
        )

    @app.post("/inbox/from-report", response_class=HTMLResponse)
    def inbox_from_report(
        request: Request,
        title: str = Form(...),
        body: str = Form(""),
        job_text: str = Form(""),
        score_run_id: str = Form(""),
        gaps_run_id: str = Form(""),
    ) -> Any:
        payload: dict[str, Any] = {"job_text_preview": job_text[:200]}
        if score_run_id:
            payload["score_run_id"] = int(score_run_id)
        if gaps_run_id:
            payload["gaps_run_id"] = int(gaps_run_id)
        item = inbox_mod.enqueue(
            store, kind="consider_jd", title=title, body=body, payload=payload
        )
        if notifier is not None:
            notifier.notify(
                title=f"OfferGuide: 新候选 #{item.id}",
                body=item.title,
                level="info",
            )
        items = inbox_mod.list_items(store, status="pending", limit=10)
        return templates.TemplateResponse(
            request, "_inbox_list.html", _ctx(request, items=items)
        )

    @app.post("/inbox/{item_id}/decide", response_class=HTMLResponse)
    def decide(
        request: Request,
        item_id: int,
        decision: str = Form(...),
    ) -> Any:
        if decision not in ("approved", "rejected", "dismissed"):
            raise HTTPException(400, f"unknown decision: {decision}")
        try:
            item = inbox_mod.decide(store, item_id, decision=decision)  # type: ignore[arg-type]
        except KeyError:
            raise HTTPException(404, f"inbox item {item_id} not found") from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return templates.TemplateResponse(
            request, "_inbox_list.html", _ctx(request, items=[item])
        )

    # ── Browser extension ingest endpoint ──────────────────────────────

    @app.post("/api/extension/ingest", response_class=JSONResponse)
    def extension_ingest(payload: ExtensionJDPayload) -> dict:
        """Accept JD data from the Boss browser extension and ingest as a job."""
        raw_text_parts = [payload.description]
        if payload.tags:
            raw_text_parts.append("标签: " + ", ".join(payload.tags))
        raw_text = "\n".join(raw_text_parts).strip()
        if not raw_text:
            raise HTTPException(400, "empty JD text")

        extras: dict = {}
        if payload.salary:
            extras["salary"] = payload.salary
        if payload.tags:
            extras["tags"] = payload.tags

        rj = RawJob(
            source="boss_extension",
            source_id=_extract_boss_id(payload.url) if payload.url else None,
            url=payload.url,
            title=payload.title,
            company=payload.company,
            location=payload.location,
            raw_text=raw_text,
            extras=extras,
        )
        is_new, job_id = scout.ingest(store, rj)
        return {"is_new": is_new, "job_id": job_id}

    return app


class ExtensionJDPayload(BaseModel):
    """Request body from the Boss browser extension."""

    url: str | None = None
    title: str = "(untitled)"
    company: str | None = None
    location: str | None = None
    salary: str | None = None
    description: str
    tags: list[str] = []


class EmailClassifyPayload(BaseModel):
    """Request body for /api/email/classify.

    ``text`` can be a single email or a multi-email dump (with
    'From: ' separators or 2+ blank-line separators) — set
    ``batch=True`` to split before classifying.

    ``mode`` controls regex vs LLM:
    - ``regex``: deterministic pattern match (no API key needed)
    - ``llm``: real LLM classification with structured extraction
    - ``auto``: llm when DEEPSEEK_API_KEY is set, else regex
    """

    text: str
    batch: bool = True
    mode: Literal["regex", "llm", "auto"] = "auto"


class SweepPayload(BaseModel):
    """Request body for /api/agent/sweep — the meta-agent endpoint."""

    company: str
    role_hint: str | None = None
    do_corpus: bool = True
    """When True, the agent searches the web for new 面经 about this
    company and ingests them. Requires LLM + search backend."""


_BOSS_ID_RE = re.compile(r"/job_detail/([^/.]+)")


def _extract_boss_id(url: str | None) -> str | None:
    """Pull the job id from a Boss URL like /job_detail/abc123.html."""
    if not url:
        return None
    m = _BOSS_ID_RE.search(url)
    return m.group(1) if m else None


# ─────────────────────── stats / dashboard helpers ──────────────────


def _quick_stats(store: Store) -> dict[str, Any]:
    """Lightweight counts for the home page strip — single SQL trip."""
    with store.connect() as conn:
        return {
            "jobs":          conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0],
            "skill_runs":    conn.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0],
            "inbox_pending": conn.execute(
                "SELECT COUNT(*) FROM inbox_items WHERE status='pending'"
            ).fetchone()[0],
            "evolutions":    conn.execute("SELECT COUNT(*) FROM evolution_log").fetchone()[0],
        }


def _full_stats(store: Store) -> dict[str, Any]:
    """Richer stats for the dashboard."""
    with store.connect() as conn:
        jobs_by_source = dict(
            conn.execute("SELECT source, COUNT(*) FROM jobs GROUP BY source").fetchall()
        )
        runs_by_skill = dict(
            conn.execute(
                "SELECT skill_name, COUNT(*) FROM skill_runs GROUP BY skill_name"
            ).fetchall()
        )
        evos_by_skill = dict(
            conn.execute(
                "SELECT skill_name, COUNT(*) FROM evolution_log GROUP BY skill_name"
            ).fetchall()
        )
        inbox_decided = conn.execute(
            "SELECT COUNT(*) FROM inbox_items WHERE status != 'pending'"
        ).fetchone()[0]
        inbox_pending = conn.execute(
            "SELECT COUNT(*) FROM inbox_items WHERE status = 'pending'"
        ).fetchone()[0]

    def _fmt(d: dict, sep: str = " · ") -> str:
        if not d:
            return "—"
        return sep.join(f"{k}={v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))

    return {
        "jobs": sum(jobs_by_source.values()),
        "jobs_by_source_str": _fmt(jobs_by_source),
        "skill_runs": sum(runs_by_skill.values()),
        "skill_runs_by_skill_str": _fmt(runs_by_skill),
        "inbox_pending": inbox_pending,
        "inbox_decided": inbox_decided,
        "evolutions": sum(evos_by_skill.values()),
        "evolutions_by_skill_str": _fmt(evos_by_skill),
    }


_FUNNEL_STAGES: list[tuple[str, str]] = [
    ("submitted",  "投递"),
    ("viewed",     "HR 已查看"),
    ("replied",    "HR 已回复"),
    ("assessment", "笔试 / OA"),
    ("interview",  "面试"),
    ("offer",      "Offer"),
]


def _application_funnel(store: Store) -> dict[str, Any]:
    """Count applications that reached each stage, derived from event log.

    A row counts at a stage if its application_events has *any* event of
    that kind, regardless of subsequent rejections — this is a "reached"
    funnel, which is what dashboards usually want.
    """
    counts: list[tuple[str, int]] = []
    with store.connect() as conn:
        total = conn.execute(
            "SELECT COUNT(DISTINCT application_id) FROM application_events"
        ).fetchone()[0]
        for kind, label in _FUNNEL_STAGES:
            n = conn.execute(
                "SELECT COUNT(DISTINCT application_id) FROM application_events "
                "WHERE kind = ? AND source != 'inferred'",
                (kind,),
            ).fetchone()[0]
            counts.append((label, n))
    return {"total": total, "stages": counts}


def _recent_evolutions(store: Store, *, limit: int = 10) -> list[Any]:
    """Latest evolution_log rows as EvolutionRecord objects."""
    from ..evolution.diff import _row_to_record

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, skill_name, parent_version, new_version, metric_name, "
            "metric_before, metric_after, notes, created_at "
            "FROM evolution_log ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def _recent_skill_runs(store: Store, *, limit: int = 10) -> list[dict[str, Any]]:
    """Recent skill_runs with a human-readable 'when_ago' field."""
    import time as _time

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, skill_name, skill_version, latency_ms, cost_usd, "
            "(julianday('now') - created_at) * 86400 AS age_seconds "
            "FROM skill_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        age = float(r[5] or 0)
        out.append(
            {
                "id": r[0],
                "skill_name": r[1],
                "skill_version": r[2],
                "latency_ms": r[3] or 0,
                "cost_usd": r[4] or 0.0,
                "when_ago": _humanize_age(age),
            }
        )
    _ = _time  # silence unused
    return out


def _humanize_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h ago"
    return f"{int(seconds / 86400)}d ago"


# ─────────────── applications timeline helpers ──────────────────────


_STATUS_CLASS = {
    "applied":         "primary",
    "considered":      "",
    "viewed":          "",
    "hr_replied":      "low",
    "screening":       "primary",
    "written_test":    "medium",
    "1st_interview":   "medium",
    "2nd_interview":   "medium",
    "final_interview": "medium",
    "offer":           "low",
    "rejected":        "high",
    "withdrawn":       "dismissed",
}


def _list_applications_with_events(
    store: Store, *, where_id: int | None = None
) -> list[dict[str, Any]]:
    """Return all applications + their event timelines, newest first.

    When ``where_id`` is set, returns only that application (used by the
    HTMX swap-in-place after logging an event).
    """
    where_clause = "WHERE a.id = ?" if where_id is not None else ""
    params: tuple = (where_id,) if where_id is not None else ()

    with store.connect() as conn:
        app_rows = conn.execute(
            f"SELECT a.id, a.job_id, a.status, a.applied_at, "
            f"j.title, j.company, j.location, j.source "
            f"FROM applications a JOIN jobs j ON j.id = a.job_id "
            f"{where_clause} "
            f"ORDER BY a.last_status_change DESC, a.id DESC",
            params,
        ).fetchall()
        if not app_rows:
            return []
        ids = tuple(r[0] for r in app_rows)
        placeholders = ",".join("?" * len(ids))
        ev_rows = conn.execute(
            f"SELECT application_id, kind, source, occurred_at "
            f"FROM application_events "
            f"WHERE application_id IN ({placeholders}) "
            f"ORDER BY occurred_at ASC, id ASC",
            ids,
        ).fetchall()

    events_by_app: dict[int, list[dict]] = {i: [] for i in ids}
    for app_id, kind, source, occurred_at in ev_rows:
        events_by_app[app_id].append(
            {
                "kind": kind, "source": source,
                "occurred_at": occurred_at,
                "when_str": _julian_to_human(occurred_at),
            }
        )

    out: list[dict[str, Any]] = []
    for r in app_rows:
        app_id, job_id, status, applied_at, title, company, location, source = r
        events = events_by_app.get(app_id, [])
        # Silence age (days since latest non-inferred event)
        real = [e for e in events if e["source"] != "inferred"]
        if real:
            from datetime import datetime
            now_jd = _to_julian(datetime.now(tz=UTC))
            silence_days = max(0.0, now_jd - real[-1]["occurred_at"])
        else:
            silence_days = None
        out.append(
            {
                "id": app_id,
                "job_id": job_id,
                "status": status,
                "status_class": _STATUS_CLASS.get(status, ""),
                "title": title, "company": company,
                "location": location, "source": source,
                "applied_at": applied_at,
                "events": events,
                "silence_days": silence_days,
            }
        )
    return out


def _julian_to_human(jd: float) -> str:
    """Render a julianday timestamp as a human-readable 'Nm ago' string."""
    from datetime import datetime
    now_jd = _to_julian(datetime.now(tz=UTC))
    age_days = max(0.0, now_jd - jd)
    seconds = age_days * 86400
    return _humanize_age(seconds)


def _list_company_groups(store: Store) -> list[dict[str, Any]]:
    """Companies with ≥ 2 jobs in the DB, with job counts.

    Used by /compare to suggest which groups are worth comparing.
    Sorted by job count desc.
    """
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT company, COUNT(*) AS n FROM jobs "
            "WHERE company IS NOT NULL AND company != '' "
            "GROUP BY company HAVING COUNT(*) >= 2 ORDER BY n DESC, company ASC"
        ).fetchall()
    return [{"company": r[0], "n": r[1]} for r in rows]


def _list_jobs_for_company(store: Store, company: str) -> list[dict[str, Any]]:
    """All jobs in the DB for a company, newest first."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, title, location, source, raw_text "
            "FROM jobs WHERE company = ? ORDER BY fetched_at DESC, id DESC",
            (company,),
        ).fetchall()
    return [
        {
            "id": r[0], "title": r[1], "location": r[2],
            "source": r[3], "raw_text": r[4] or "",
        }
        for r in rows
    ]


def _list_interview_companies(store: Store) -> list[dict[str, Any]]:
    """Companies with stored 面经, with counts."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT company, COUNT(*) FROM interview_experiences "
            "GROUP BY company ORDER BY COUNT(*) DESC, company ASC"
        ).fetchall()
    return [{"company": r[0], "n": r[1]} for r in rows]


def _list_interview_experiences(
    store: Store, *, company: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Recent 面经, optionally filtered by company."""
    with store.connect() as conn:
        if company:
            rows = conn.execute(
                "SELECT id, company, role_hint, raw_text, source, source_url, created_at "
                "FROM interview_experiences WHERE company LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (f"%{company}%", limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, company, role_hint, raw_text, source, source_url, created_at "
                "FROM interview_experiences ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [
        {
            "id": r[0], "company": r[1], "role_hint": r[2], "raw_text": r[3],
            "source": r[4], "source_url": r[5], "created_at": r[6],
        }
        for r in rows
    ]


def _list_jobs_by_ids(store: Store, ids: list[int]) -> list[dict[str, Any]]:
    """Fetch a specific set of jobs by id, in input order."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with store.connect() as conn:
        rows = conn.execute(
            f"SELECT id, title, company, location, source, raw_text "
            f"FROM jobs WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
    by_id = {
        r[0]: {
            "id": r[0], "title": r[1], "company": r[2], "location": r[3],
            "source": r[4], "raw_text": r[5] or "",
        }
        for r in rows
    }
    return [by_id[i] for i in ids if i in by_id]


def _to_julian(dt) -> float:
    """Calendar UTC datetime → SQLite julianday float."""
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    frac = (dt.hour - 12) / 24 + dt.minute / 1440 + dt.second / 86400
    return jdn + frac


# -------------------- entry point used by `python -m offerguide.ui.web` --------------------


def main() -> None:
    """Build everything from env vars and serve via uvicorn."""
    import uvicorn

    settings = Settings.from_env()
    store = Store(settings.db_path)
    store.init_schema()

    profile: UserProfile | None = None
    if settings.resume_pdf and settings.resume_pdf.exists():
        profile = load_resume_pdf(settings.resume_pdf)

    skills_root = Path(__file__).parent.parent / "skills"
    skills = discover_skills(skills_root)

    runtime: SkillRuntime | None = None
    if settings.deepseek_api_key:
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        runtime = SkillRuntime(llm, store)

    notifier = make_notifier(settings)

    app = create_app(
        settings=settings,
        store=store,
        profile=profile,
        skills=skills,
        runtime=runtime,
        notifier=notifier,
    )

    print(f"\n✦ OfferGuide UI on http://{settings.web_host}:{settings.web_port}")
    print(f"  resume = {settings.resume_pdf or '(none — set OFFERGUIDE_RESUME_PDF)'}")
    print(
        f"  llm    = {'configured' if settings.deepseek_api_key else 'NOT configured (set DEEPSEEK_API_KEY)'}"
    )
    print(f"  notify = {settings.notify_channel} ({'ready' if settings.notify_ready() else 'fallback console'})")
    uvicorn.run(app, host=settings.web_host, port=settings.web_port, log_level="info")


# `_` to silence unused-symbol lint in linters that don't read entry points
_ = RedirectResponse


if __name__ == "__main__":
    main()
