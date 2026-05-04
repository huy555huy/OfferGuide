"""FastAPI web UI — agent + supporting pages.

Routes (after W13.1 cleanup):

    GET  /                     daily standup home
    GET  /agent                W13 central agent loop entry point
    GET  /api/agent/stream     SSE streaming agent execution events
    GET  /agent/runs/{id}      view a persisted agent_runs trajectory
    GET  /inbox                pending agent suggestions
    POST /inbox/{id}/decide    mark item approved|rejected|dismissed

The app is intentionally HTMX-driven (no SPA, no JS framework). Server-side
templates render fragments; the client just swaps DOM nodes. Easier to test,
easier to ship as a small local tool.

Application-factory pattern (`create_app(...)`) lets tests inject stub stores,
runtimes, profiles, and notifiers without touching env vars.
"""

from __future__ import annotations

import json
import re
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

import asyncio

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .. import inbox as inbox_mod
from ..agent import AgentLoop
from ..config import Settings
from ..llm import LLMClient, LLMError
from ..memory import Store
from ..platforms._spec import RawJob
from ..profile import UserProfile, load_resume_pdf
from ..skills import SkillRuntime, SkillSpec, discover_skills
from ..workers import scout
from .notify import Notifier, make_notifier

# Convenience aliases used in handlers (post here so lint isort is happy).
json_loads = json.loads
json_dumps = json.dumps

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
        """Home (W13.x rewrite) — agent-driven, not dashboard-driven.

        The pre-W13 home was a daemon-style stat panel ("4 stat cards + 5
        action items"). That's "show me what cron observed", not "what does
        my copilot think about my situation right now".

        New shape:
          - Centerpiece: the most recent agent run's final answer (with
            critic_score + run timestamp + "rerun now" button)
          - Pending agent_suggestion inbox items (the things agent thinks
            you should decide)
          - Quick-stats strip (just numbers, not "推荐操作" — leave that to agent)
          - Quick links to /agent + /apply + /evolution
        """
        # Most recent agent_run for the hero
        with store.connect() as conn:
            row = conn.execute(
                "SELECT id, goal, final_answer, critic_score, critic_notes, "
                "       latency_ms, started_at, status, iterations, trigger_kind "
                "FROM agent_runs WHERE final_answer IS NOT NULL "
                "  AND status = 'ok' "
                "ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        latest_run = None
        if row:
            latest_run = {
                "id": row[0], "goal": row[1], "final_answer": row[2],
                "critic_score": row[3], "critic_notes": row[4],
                "latency_ms": row[5], "started_at": row[6],
                "status": row[7], "iterations": row[8],
                "trigger_kind": row[9],
            }

        # Pending agent_suggestion items
        suggestions = [
            i for i in inbox_mod.list_items(store, status="pending", limit=20)
            if i.kind == "agent_suggestion"
        ][:6]

        # Just-stats (not action lists — agent gives those)
        with store.connect() as conn:
            n_jobs = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE length(raw_text) >= 200"
            ).fetchone()[0]
            n_apps_active = conn.execute(
                "SELECT COUNT(*) FROM applications "
                "WHERE status NOT IN ('offer','rejected','withdrawn')"
            ).fetchone()[0]
            n_apps_offer = conn.execute(
                "SELECT COUNT(*) FROM applications WHERE status='offer'"
            ).fetchone()[0]
            n_pending_inbox = conn.execute(
                "SELECT COUNT(*) FROM inbox_items WHERE status='pending'"
            ).fetchone()[0]

        return templates.TemplateResponse(
            request,
            "home.html",
            _ctx(
                request,
                latest_run=latest_run,
                suggestions=suggestions,
                stats={
                    "jobs": n_jobs,
                    "active_apps": n_apps_active,
                    "offers": n_apps_offer,
                    "pending_inbox": n_pending_inbox,
                },
                runtime_ready=runtime is not None and bool(settings.deepseek_api_key),
                active_tab="home",
            ),
        )

    @app.post("/api/home/wake-agent", response_class=JSONResponse)
    async def home_wake_agent(request: Request) -> Any:
        """Trigger an agent run from the home page (manual user kickoff).

        Reuses the /api/agent/stream behavior but with a "巡检" goal preset.
        Returns the new run's id so the page can poll/redirect.
        """
        if runtime is None or not settings.deepseek_api_key:
            raise HTTPException(400, "agent 不可用 — 缺 OFFERGUIDE_LLM_API_KEY")

        from ..agent.loop import AgentLoop
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        try:
            agent = AgentLoop(
                llm=llm, runtime=runtime, store=store, skills=skills,
                master_resume_text=profile.raw_resume_text if profile else "",
                max_iterations=6, critic_enabled=True,
            )
            result = agent.run(
                goal=(
                    "用户刚打开 home 页, 想看你对当前求职状况的评估。"
                    "看 snapshot, 给一段诚实的当下情况评估 (做了啥 / 待办优先级 / 有没有该提醒的事)。"
                    "不要为了显得忙就强行调工具——简短判断更有价值。"
                ),
                trigger_kind="user_button",
            )
        finally:
            try:
                llm.close()
            except Exception:
                pass

        return {
            "run_id": result.run_id,
            "iterations": result.iterations,
            "critic_score": result.critic_score,
            "latency_ms": result.latency_ms,
            "final_answer": result.final_answer,
        }

    # /quick-eval + /chat removed in W13.1 — superseded by /agent (model in main
    # position decides what to do, no need for a separate "paste JD then pick
    # action" form). The old chat() handler also depended on the W4 LangGraph
    # build_graph, which is also being retired.

    @app.get("/pipeline", response_class=HTMLResponse)
    def pipeline_view(request: Request) -> Any:
        """5-stage kanban view of every JD/application in the system.

        Each card has inline transition buttons that record an event +
        re-render the card via HTMX. The page is the "投递战况" overview.
        """
        from .. import pipeline_view as pv_mod

        view = pv_mod.build(store)
        return templates.TemplateResponse(
            request,
            "pipeline.html",
            _ctx(
                request,
                pipeline=view,
                pipeline_stages=pv_mod.KANBAN_STAGES,
                stage_color=pv_mod.stage_color,
                render_age=pv_mod.render_age,
                transition_options=pv_mod.transition_options,
                active_tab="pipeline",
            ),
        )

    @app.post(
        "/api/pipeline/applications/{app_id}/event",
        response_class=HTMLResponse,
    )
    def pipeline_log_event(
        request: Request,
        app_id: int,
        kind: str = Form(...),
    ) -> Any:
        """Record an event from the kanban card menu, then re-render
        just the affected card so HTMX swaps it in place."""
        from .. import application_events as ae
        from .. import pipeline_view as pv_mod
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

        # Rebuild + find the updated card so we can re-render it
        view = pv_mod.build(store)
        card = None
        new_stage = None
        for stage_key, cards in view.columns.items():
            for c in cards:
                if c.application_id == app_id:
                    card = c
                    new_stage = stage_key
                    break
            if card is not None:
                break
        if card is None:
            # Card moved to terminal & got hidden, or app vanished — return empty
            return HTMLResponse("")
        return templates.TemplateResponse(
            request,
            "_kanban_card.html",
            _ctx(
                request,
                card=card,
                stage_key=new_stage,
                stage_color=pv_mod.stage_color,
                render_age=pv_mod.render_age,
                transition_options=pv_mod.transition_options,
            ),
        )

    @app.post("/api/pipeline/jobs/{job_id}/submit", response_class=HTMLResponse)
    def pipeline_submit_job(
        request: Request,
        job_id: int,
    ) -> Any:
        """Promote a 'scanned' job to 'applied' by creating an
        application row + recording a submitted event.

        The kanban surfaces this as the only transition available on
        scanned cards. After this, the row leaves 'scanned' and shows
        up in 'applied'.
        """
        from .. import application_events as ae
        from .. import pipeline_view as pv_mod

        with store.connect() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(404, f"job {job_id} not found")
            existing = conn.execute(
                "SELECT id FROM applications WHERE job_id = ?", (job_id,)
            ).fetchone()
            if existing:
                app_id = int(existing[0])
            else:
                cur = conn.execute(
                    "INSERT INTO applications(job_id, status) "
                    "VALUES (?, 'applied') RETURNING id",
                    (job_id,),
                )
                app_id = int(cur.fetchone()[0])

        ae.record(store, application_id=app_id, kind="submitted", source="manual")

        view = pv_mod.build(store)
        # Card is now in 'applied' — find and render it
        card = next(
            (c for c in view.columns["applied"] if c.application_id == app_id),
            None,
        )
        if card is None:
            return HTMLResponse("")
        return templates.TemplateResponse(
            request,
            "_kanban_card.html",
            _ctx(
                request,
                card=card,
                stage_key="applied",
                stage_color=pv_mod.stage_color,
                render_age=pv_mod.render_age,
                transition_options=pv_mod.transition_options,
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

    @app.get("/stories", response_class=HTMLResponse)
    def stories_view(request: Request) -> Any:
        from .. import story_bank
        return templates.TemplateResponse(
            request, "stories.html",
            _ctx(
                request,
                stories=story_bank.list_all(store, limit=50),
                recommended_tags=story_bank.RECOMMENDED_TAGS,
                active_tab="stories",
            ),
        )

    @app.post("/api/stories/insert", response_class=HTMLResponse)
    def stories_insert(
        request: Request,
        title: str = Form(...),
        situation: str = Form(...),
        task: str = Form(...),
        action: str = Form(...),
        result: str = Form(...),
        reflection: str = Form(""),
        tags: str = Form(""),
        confidence: str = Form("0.5"),
    ) -> Any:
        from .. import story_bank
        try:
            conf = max(0.0, min(1.0, float(confidence or "0.5")))
        except ValueError:
            conf = 0.5
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        try:
            new_story = story_bank.insert(
                store, title=title, situation=situation, task=task,
                action=action, result=result,
                reflection=reflection or None,
                tags=tag_list, confidence=conf,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

        return templates.TemplateResponse(
            request, "_story_list.html",
            _ctx(
                request,
                stories=story_bank.list_all(store, limit=50),
                just_added=new_story.id,
            ),
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

    # ─────────────────────────── W13 central agent loop ───────────────────────
    # Model in the driver's seat: model decides which SKILL to call (via OpenAI
    # tool-calling), when to stop. The /agent page is the user-facing surface;
    # the SSE endpoint streams every event (state_snapshot / thinking /
    # tool_call / tool_result / critique / final) so user sees the agent
    # think + act in real time.

    @app.get("/agent", response_class=HTMLResponse)
    def agent_page(request: Request) -> Any:
        """W13 agent loop UI — pick a goal, watch the agent think + act live."""
        # Recent runs to show below the form (audit trail)
        try:
            with store.connect() as conn:
                rows = conn.execute(
                    "SELECT id, trigger_kind, goal, status, iterations, "
                    "       critic_score, latency_ms, started_at "
                    "FROM agent_runs ORDER BY started_at DESC LIMIT 8"
                ).fetchall()
        except Exception:
            rows = []
        recent_runs = [
            {
                "id": r[0], "trigger_kind": r[1], "goal": r[2],
                "status": r[3], "iterations": r[4],
                "critic_score": r[5], "latency_ms": r[6],
            }
            for r in rows
        ]
        return templates.TemplateResponse(
            request,
            "agent.html",
            _ctx(
                request,
                recent_runs=recent_runs,
                skill_count=len(skills),
                tools_ready=runtime is not None and bool(settings.deepseek_api_key),
                active_tab="agent",
            ),
        )

    @app.get("/api/agent/stream")
    async def agent_stream(
        request: Request,
        goal: str,
        trigger_kind: str = "user_button",
        max_iterations: int = 6,
    ) -> StreamingResponse:
        """SSE endpoint that runs the agent loop in a thread + streams events.

        Each event becomes one ``data: {...}\\n\\n`` SSE frame. The browser-
        side EventSource (in agent.html) appends each frame to the live
        panel as it arrives. The connection closes after the loop returns.

        Implementation note: AgentLoop is sync (each LLM call blocks), so we
        run it in a thread via asyncio.to_thread + a thread-safe queue back
        to the async generator. We never await inside the agent loop itself —
        that would defeat the per-iteration streaming effect.
        """
        if runtime is None or not settings.deepseek_api_key:
            async def _err_stream():
                yield (
                    "data: " + json_dumps({
                        "kind": "error",
                        "message": "OFFERGUIDE_LLM_API_KEY 没配 — agent 无法启动",
                    }) + "\n\n"
                )
            return StreamingResponse(_err_stream(), media_type="text/event-stream")

        # Build a fresh LLMClient per request (cheap; httpx.Client lifecycle)
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        agent = AgentLoop(
            llm=llm,
            runtime=runtime,
            store=store,
            skills=skills,
            master_resume_text=profile.raw_resume_text if profile else "",
            max_iterations=max(1, min(int(max_iterations), 12)),
        )

        main_loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _on_event_from_thread(ev: Any) -> None:
            # AgentLoop calls this from its worker thread — bridge to async queue
            try:
                main_loop.call_soon_threadsafe(queue.put_nowait, dict(ev))
            except RuntimeError:
                # Event loop closed (client disconnected) — drop event
                pass

        def _run_blocking() -> None:
            try:
                result = agent.run(
                    goal=goal,
                    trigger_kind=trigger_kind,
                    on_event=_on_event_from_thread,
                )
                _on_event_from_thread({
                    "kind": "_done",
                    "run_id": result.run_id,
                    "iterations": result.iterations,
                    "critic_score": result.critic_score,
                    "latency_ms": result.latency_ms,
                })
            except Exception as e:
                _on_event_from_thread({
                    "kind": "_done", "error": str(e),
                })
            finally:
                # Sentinel so the SSE generator knows to close
                main_loop.call_soon_threadsafe(queue.put_nowait, None)

        # Kick off the agent in a background thread
        asyncio.create_task(asyncio.to_thread(_run_blocking))

        async def _sse_gen():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=60.0)
                    except asyncio.TimeoutError:
                        # Heartbeat to keep the connection open through silent stretches
                        yield ": keepalive\n\n"
                        continue
                    if ev is None:
                        break
                    yield "data: " + json_dumps(ev, ensure_ascii=False, default=str) + "\n\n"
            finally:
                try:
                    llm.close()
                except Exception:
                    pass

        return StreamingResponse(
            _sse_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",  # disable buffering on nginx-style proxies
                "Connection": "keep-alive",
            },
        )

    # ─────────── W13.6 long-horizon goals (north star) ────────────────

    # ─────────────── W13.8 funnel + portfolio ────────────────────────

    @app.get("/funnel", response_class=HTMLResponse)
    def funnel_view(request: Request) -> Any:
        """Conversion funnel: applied → submitted → viewed → replied →
        interview → offer. Computes per-stage counts + percentage carryovers."""
        funnel_stages = [
            ("applied",    "投递", None),
            ("submitted",  "已提交", "submitted"),
            ("viewed",     "HR 看了", "viewed"),
            ("replied",    "HR 回复", "replied"),
            ("interview",  "进面试", "interview"),
            ("offer",      "拿 offer", "offer"),
        ]
        stage_counts: list[dict] = []
        with store.connect() as conn:
            # 总投递数 = applications 里所有
            total_apps = conn.execute(
                "SELECT COUNT(*) FROM applications"
            ).fetchone()[0]
            stage_counts.append({
                "key": "applied", "label": "投递", "count": total_apps,
                "from_prev_pct": None, "from_top_pct": 100.0 if total_apps else 0.0,
            })
            prev = total_apps
            for key, label, event_kind in funnel_stages[1:]:
                if event_kind:
                    n = conn.execute(
                        "SELECT COUNT(DISTINCT application_id) "
                        "FROM application_events WHERE kind = ?",
                        (event_kind,),
                    ).fetchone()[0]
                else:
                    n = 0
                from_prev = (n / prev * 100) if prev else None
                from_top = (n / total_apps * 100) if total_apps else 0.0
                stage_counts.append({
                    "key": key, "label": label, "count": n,
                    "from_prev_pct": from_prev, "from_top_pct": from_top,
                })
                prev = n if n else prev  # don't divide by 0 in next iter

            # Per-company breakdown
            company_rows = conn.execute(
                "SELECT j.company, COUNT(DISTINCT a.id) AS n "
                "FROM applications a LEFT JOIN jobs j ON j.id = a.job_id "
                "WHERE j.company IS NOT NULL "
                "GROUP BY j.company ORDER BY n DESC LIMIT 10"
            ).fetchall()
            companies = [
                {"name": r[0], "count": r[1]} for r in company_rows
            ]

        return templates.TemplateResponse(
            request, "funnel.html",
            _ctx(request, stages=stage_counts, companies=companies, active_tab=None),
        )

    @app.get("/portfolio", response_class=HTMLResponse)
    def portfolio_view(request: Request) -> Any:
        """Public-friendly metrics page about the OfferGuide build itself.

        Privacy: NO user data (resume, names, application details, company
        names). Just aggregate numbers + agent-level metrics that are
        meaningful as evidence-of-build for a portfolio.
        """
        with store.connect() as conn:
            # Agent runs metrics
            n_agent_runs = conn.execute(
                "SELECT COUNT(*) FROM agent_runs"
            ).fetchone()[0]
            n_skill_runs = conn.execute(
                "SELECT COUNT(*) FROM skill_runs"
            ).fetchone()[0]
            avg_critic = conn.execute(
                "SELECT AVG(critic_score) FROM agent_runs WHERE critic_score IS NOT NULL"
            ).fetchone()[0]
            total_cost = conn.execute(
                "SELECT SUM(cost_usd) FROM agent_runs"
            ).fetchone()[0]
            # Evolution metrics
            n_variants = conn.execute(
                "SELECT COUNT(*) FROM skill_variants"
            ).fetchone()[0]
            n_live_variants = conn.execute(
                "SELECT COUNT(*) FROM skill_variants WHERE status='live'"
            ).fetchone()[0]
            n_signals = conn.execute(
                "SELECT COUNT(*) FROM evolution_signals"
            ).fetchone()[0]
            # Per-skill recent fitness (no SKILL details, just name + count + score)
            skill_rows = conn.execute(
                "SELECT skill_name, COUNT(*) as n, AVG(signal_value) as avg_v "
                "FROM evolution_signals WHERE signal_kind='critic' "
                "GROUP BY skill_name HAVING n >= 3 "
                "ORDER BY n DESC LIMIT 12"
            ).fetchall()
            skill_metrics = [
                {"name": r[0], "n_signals": r[1], "avg_critic": r[2]}
                for r in skill_rows
            ]
            # Last 14 days agent run count for sparkline
            daily_rows = conn.execute(
                "SELECT CAST((julianday('now') - started_at) AS INT) AS days_ago, "
                "       COUNT(*) AS n "
                "FROM agent_runs WHERE started_at >= julianday('now') - 14 "
                "GROUP BY days_ago"
            ).fetchall()
            daily_counts = [0] * 14
            for d, n in daily_rows:
                if 0 <= d < 14:
                    daily_counts[13 - d] = n  # right-most = today
        return templates.TemplateResponse(
            request, "portfolio.html",
            _ctx(
                request,
                n_agent_runs=n_agent_runs,
                n_skill_runs=n_skill_runs,
                avg_critic=avg_critic,
                total_cost=total_cost or 0.0,
                n_variants=n_variants,
                n_live_variants=n_live_variants,
                n_signals=n_signals,
                skill_metrics=skill_metrics,
                daily_counts=daily_counts,
                active_tab=None,
            ),
        )

    @app.get("/goals", response_class=HTMLResponse)
    def goals_view(request: Request) -> Any:
        from .. import goals as _goals
        active = _goals.list_active_goals(store)
        progress_list = [(g, _goals.compute_progress(store, g)) for g in active]
        with store.connect() as conn:
            history_rows = conn.execute(
                "SELECT id, title, status, target_date, achieved_at "
                "FROM user_goals WHERE status != 'active' "
                "ORDER BY updated_at DESC LIMIT 10"
            ).fetchall()
        history = [
            {"id": r[0], "title": r[1], "status": r[2],
             "target_date": r[3], "achieved_at": r[4]}
            for r in history_rows
        ]
        # Self-observations the agent has accumulated
        self_obs = _goals.list_active_self_observations(store, limit=15)
        return templates.TemplateResponse(
            request, "goals.html",
            _ctx(
                request,
                active=active, progress_list=progress_list,
                history=history, self_obs=self_obs,
                active_tab="goals",
            ),
        )

    @app.post("/api/goals/add", response_class=JSONResponse)
    def goals_add(
        title: str = Form(...),
        description: str = Form(""),
        target_date: str = Form(""),
        target_metric: str = Form(""),
    ) -> dict:
        from .. import goals as _goals
        if not title.strip():
            raise HTTPException(400, "title required")
        td = target_date.strip() or None
        if td:
            try:
                # Validate ISO
                from datetime import date as _date
                _date.fromisoformat(td)
            except ValueError:
                raise HTTPException(400, f"target_date must be YYYY-MM-DD, got {td!r}") from None
        g = _goals.add_goal(
            store, title=title.strip(),
            description=description.strip() or None,
            target_date=td,
            target_metric=target_metric.strip() or None,
        )
        return {"id": g.id, "title": g.title, "status": g.status}

    @app.post("/api/goals/{goal_id}/status", response_class=JSONResponse)
    def goals_set_status(goal_id: int, status: str = Form(...)) -> dict:
        from .. import goals as _goals
        if status not in ("active", "paused", "achieved", "abandoned"):
            raise HTTPException(400, f"unknown status {status!r}")
        ok = _goals.update_goal_status(store, goal_id, status=status)  # type: ignore[arg-type]
        if not ok:
            raise HTTPException(404, f"goal#{goal_id} not found")
        return {"id": goal_id, "status": status}

    @app.get("/evolution", response_class=HTMLResponse)
    def evolution_page(request: Request) -> Any:
        """W13.1 evolution observatory.

        Shows: per-SKILL fitness + variants tree + manual promote/fail/evolve
        buttons. Read-only view; mutations go through dedicated POST routes
        (audit-friendly).
        """
        from .. import evolution as _evo

        # Get every SKILL the system knows about, with current fitness
        skill_specs = sorted(skills, key=lambda s: s.name) if skills else []
        skill_views: list[dict[str, Any]] = []
        for spec in skill_specs:
            live = _evo.get_live_variant(store, spec.name)
            canaries = _evo.get_canary_variants(store, spec.name)
            shadows = _evo.get_shadow_variants(store, spec.name)
            current_version = live.version if live else spec.version
            fitness = _evo.compute_fitness(
                store, skill_name=spec.name, skill_version=current_version,
            )
            all_variants = _evo.list_all_variants(
                store, skill_name=spec.name, limit=20,
            )
            skill_views.append({
                "name": spec.name,
                "description": (spec.description or "")[:160],
                "seed_version": spec.version,
                "current_version": current_version,
                "fitness": fitness.fitness,
                "sample_count": fitness.sample_count,
                "by_kind": fitness.by_kind,
                "live": live,
                "canaries": canaries,
                "shadows": shadows,
                "variants": all_variants,
            })

        # Detect candidates ready for evolution (model-bypass check)
        try:
            triggers = _evo.detect_evolution_candidates(store)
        except Exception:
            triggers = []

        return templates.TemplateResponse(
            request,
            "evolution.html",
            _ctx(
                request,
                skill_views=skill_views,
                triggers=triggers,
                evolution_threshold=_evo.EVOLUTION_THRESHOLD,
                min_signals=_evo.MIN_SIGNALS_FOR_TRIGGER,
                cooldown_days=_evo.EVOLUTION_COOLDOWN_DAYS,
                active_tab="evolution",
            ),
        )

    @app.post("/api/evolution/evolve/{skill_name}", response_class=JSONResponse)
    def evolution_trigger_evolve(skill_name: str, num_variants: int = 3) -> dict:
        """Manually trigger evolve_skill on one SKILL (admin override).

        Normally the agent calls evolve_skill itself when it detects the
        candidate; this endpoint lets the user kick off evolution by hand
        from the /evolution UI.
        """
        from ..evolution.evolve import evolve_skill as _evolve

        if not settings.deepseek_api_key:
            raise HTTPException(400, "OFFERGUIDE_LLM_API_KEY 没配, 无法 evolve")
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        try:
            result = _evolve(
                store=store, llm=llm,
                skill_name=skill_name,
                num_variants=max(1, min(int(num_variants), 5)),
            )
        finally:
            llm.close()
        return {
            "skill_name": result.skill_name,
            "parent_version": result.parent_version,
            "candidates_generated": result.candidates_generated,
            "candidates_persisted": result.candidates_persisted,
            "variant_versions": result.variant_versions,
            "notes": result.notes,
        }

    @app.post("/api/evolution/release_cycle", response_class=JSONResponse)
    def evolution_run_release_cycle(dry_run: bool = False) -> dict:
        """Manually run the gray-release cycle once.

        Same effect as agent calling run_gray_release. Useful for the user
        to nudge the rollout forward without waiting for the next agent run.
        """
        from ..evolution.release import run_release_cycle

        cycle = run_release_cycle(store, dry_run=dry_run)
        return {
            "actions": [
                {"skill_name": a.skill_name, "action": a.action,
                 "version": a.version, "reason": a.reason}
                for a in cycle.actions
            ],
            "skipped": cycle.skipped,
            "summary": cycle.render_summary(),
        }

    @app.post("/api/evolution/promote/{skill_name}/{version}", response_class=JSONResponse)
    def evolution_manual_promote(skill_name: str, version: str) -> dict:
        """Manually promote a variant to live (admin override of A/B)."""
        from ..evolution.registry import (
            get_variant_by_version,
            promote_to_canary,
            promote_to_live,
        )

        v = get_variant_by_version(store, skill_name, version)
        if v is None:
            raise HTTPException(404, f"variant {skill_name}/{version} not found")
        if v.status == "shadow":
            ok = promote_to_canary(
                store, skill_name=skill_name, version=version, traffic_pct=0.2,
            )
            return {"action": "promoted_to_canary", "ok": ok,
                    "traffic_pct": 0.2}
        if v.status == "canary":
            ok = promote_to_live(store, skill_name=skill_name, version=version)
            return {"action": "promoted_to_live", "ok": ok}
        raise HTTPException(400, f"variant in status {v.status} cannot be promoted")

    @app.post("/api/evolution/fail/{skill_name}/{version}", response_class=JSONResponse)
    def evolution_manual_fail(skill_name: str, version: str) -> dict:
        """Manually fail a shadow/canary variant (admin override)."""
        from ..evolution.registry import fail_variant

        ok = fail_variant(
            store, skill_name=skill_name, version=version,
            reason="manual fail from /evolution UI",
        )
        return {"action": "failed", "ok": ok}

    # ─────────────────────── W13.4 apply assistant ────────────────────────
    # The actual job-search value: turn a tailored resume + scored job into a
    # paste-ready application package (Boss直聘 self-intro + form Q/A +
    # submission strategy). User opens /apply/<job_id> → one-click copy each
    # piece → goes to Boss/牛客 with everything ready.

    @app.get("/apply/{job_id}", response_class=HTMLResponse)
    def apply_view(request: Request, job_id: int) -> Any:
        """Render the apply package for one job. Generates on first visit
        if no cached SKILL output exists; subsequent visits show the cached run.
        """
        with store.connect() as conn:
            job_row = conn.execute(
                "SELECT id, company, title, location, source, url, raw_text "
                "FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
        if job_row is None:
            raise HTTPException(404, f"job#{job_id} not found")

        job = {
            "id": job_row[0], "company": job_row[1], "title": job_row[2],
            "location": job_row[3], "source": job_row[4], "url": job_row[5],
            "raw_text": job_row[6],
        }

        # Find the most recent apply_assistant run for this job (cached package)
        package_dict = None
        package_run_id = None
        try:
            with store.connect() as conn:
                row = conn.execute(
                    "SELECT id, output_json FROM skill_runs "
                    "WHERE skill_name='apply_assistant' "
                    "  AND input_json LIKE ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (f'%"company": "{job["company"]}"%',),
                ).fetchone()
            if row:
                package_run_id = row[0]
                package_dict = json_loads(row[1])
        except Exception:
            pass

        # Existing applications for this job (lifecycle status)
        with store.connect() as conn:
            apps = conn.execute(
                "SELECT id, status, applied_at, last_status_change "
                "FROM applications WHERE job_id = ? ORDER BY id DESC",
                (job_id,),
            ).fetchall()
        applications = [
            {"id": r[0], "status": r[1], "applied_at": r[2], "last_change": r[3]}
            for r in apps
        ]

        return templates.TemplateResponse(
            request, "apply.html",
            _ctx(
                request, job=job,
                package=package_dict, package_run_id=package_run_id,
                applications=applications,
                profile_loaded=profile is not None,
                runtime_ready=runtime is not None and bool(settings.deepseek_api_key),
                active_tab=None,
            ),
        )

    @app.post("/api/apply/{job_id}/generate", response_class=JSONResponse)
    def apply_generate(job_id: int) -> dict:
        """Trigger apply_assistant SKILL for this job.

        Synchronous (10-30s LLM). Returns the generated package as JSON for
        the UI to render in place. Cached afterwards (subsequent /apply/N
        page loads find this skill_run via input_json LIKE).
        """
        if runtime is None or profile is None:
            raise HTTPException(
                400,
                "Need both runtime + profile (OFFERGUIDE_LLM_API_KEY + OFFERGUIDE_RESUME_PDF)",
            )
        spec = next((s for s in skills if s.name == "apply_assistant"), None)
        if spec is None:
            raise HTTPException(500, "apply_assistant SKILL not loaded")

        with store.connect() as conn:
            row = conn.execute(
                "SELECT company, title, raw_text FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"job#{job_id} not found")
        company, title, raw_text = row

        if not raw_text or len(raw_text) < 200:
            raise HTTPException(
                400,
                f"job#{job_id} raw_text too thin ({len(raw_text or '')}字), "
                "先 enrich 或人工补全 JD",
            )

        try:
            result = runtime.invoke(
                spec,
                {
                    "company": company or "?",
                    "role_focus": title or "?",
                    "job_text": raw_text,
                    "user_profile": profile.raw_resume_text,
                },
            )
        except LLMError as e:
            raise HTTPException(502, f"LLM 调用失败: {e}") from None

        return {
            "skill_run_id": result.skill_run_id,
            "package": result.parsed,
            "raw_text": result.raw_text if result.parsed is None else None,
        }

    @app.post("/api/apply/{job_id}/mark", response_class=JSONResponse)
    def apply_mark(job_id: int, status: str = Form(...)) -> dict:
        """User marks an application status (submitted / hr_viewed / replied / rejected).

        Updates applications table + writes an application_event for audit.
        Drives the W13.x feedback loop: app_outcome signals fan to evolution.
        """
        VALID_STATES = {
            "considered", "submitted", "hr_viewed", "replied",
            "interview", "offer", "rejected", "withdrawn",
        }
        if status not in VALID_STATES:
            raise HTTPException(400, f"unknown status '{status}'; valid: {sorted(VALID_STATES)}")

        with store.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM applications WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
            if existing is None:
                cur = conn.execute(
                    "INSERT INTO applications(job_id, status, applied_at) "
                    "VALUES (?, ?, julianday('now'))",
                    (job_id, status),
                )
                app_id = int(cur.lastrowid or 0)
            else:
                app_id = int(existing[0])
                conn.execute(
                    "UPDATE applications SET status = ?, "
                    "  last_status_change = julianday('now') "
                    "WHERE id = ?",
                    (status, app_id),
                )
            # Event log entry
            event_kind = {
                "submitted": "submitted", "hr_viewed": "viewed",
                "replied": "replied", "interview": "interview",
                "offer": "offer", "rejected": "rejected",
                "withdrawn": "withdrawn",
            }.get(status, "submitted")
            conn.execute(
                "INSERT INTO application_events(application_id, kind, source) "
                "VALUES (?, ?, 'manual')",
                (app_id, event_kind),
            )

        # W13.x feedback loop: positive/negative outcomes → evolution_signals
        # attributed to apply_assistant + most recent SKILLs that ran for this job
        try:
            from .. import evolution as _evo
            outcome_map = {
                "offer": "offer", "interview": "interview",
                "replied": "interview",  # reply ~= positive
                "rejected": "rejected",
                # 'submitted' / 'hr_viewed' too early to score
            }
            outcome = outcome_map.get(status)
            if outcome:
                # Attribute to apply_assistant (most direct link)
                _evo.record_app_outcome(
                    store,
                    skill_name="apply_assistant",
                    skill_version="0.1.0",  # TODO: lookup current live version
                    skill_run_id=None,
                    outcome=outcome,  # type: ignore[arg-type]
                    weight=1.0,
                )
        except Exception as e:
            log_mod = __import__("logging").getLogger(__name__)
            log_mod.debug("apply outcome signal write failed: %s", e)

        return {"app_id": app_id, "status": status}

    @app.get("/agent/runs/{run_id}", response_class=HTMLResponse)
    def agent_run_detail(request: Request, run_id: int) -> Any:
        """Read a persisted agent_runs row + render its trajectory + critic."""
        with store.connect() as conn:
            row = conn.execute(
                "SELECT trigger_kind, goal, status, iterations, final_answer, "
                "       trajectory_json, critic_score, critic_notes, latency_ms, "
                "       started_at, ended_at, error_text "
                "FROM agent_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"agent_runs#{run_id} not found")
        try:
            trajectory = json_loads(row[5] or "[]")
        except json.JSONDecodeError:
            trajectory = []
        return templates.TemplateResponse(
            request,
            "agent_run_detail.html",
            _ctx(
                request,
                run_id=run_id,
                trigger_kind=row[0], goal=row[1], status=row[2],
                iterations=row[3], final_answer=row[4],
                trajectory=trajectory,
                critic_score=row[6], critic_notes=row[7],
                latency_ms=row[8],
                started_at=row[9], ended_at=row[10],
                error_text=row[11],
                active_tab="agent",
            ),
        )

    @app.get("/cover-letter/{run_id}.html", response_class=HTMLResponse)
    def cover_letter_print(request: Request, run_id: int) -> Any:
        """Print-ready standalone HTML page for one cover letter run.

        User opens ⌘P / Save as PDF in their browser to get a real document.
        Borrowed from Career-Ops' Playwright HTML→PDF approach (we skip
        the Playwright dep and let the browser do the print).
        """
        with store.connect() as conn:
            row = conn.execute(
                "SELECT skill_name, output_json FROM skill_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None or row[0] != "write_cover_letter":
            raise HTTPException(404, f"no write_cover_letter run with id {run_id}")
        try:
            cover = json_loads(row[1])
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(500, "stored cover letter output is corrupt") from None

        return templates.TemplateResponse(
            request, "cover_letter_print.html",
            _ctx(request, cover=cover, run_id=run_id),
        )

    @app.get("/tailor", response_class=HTMLResponse)
    def tailor_view(request: Request, job_id: str = "") -> Any:
        """简历微调入口页 — 选 JD + 显示 master resume + 一键 tailor。"""
        from .. import briefs as briefs_mod  # noqa: F401 (potential import cycle guard)

        # List recent jobs that have raw_text >= 200 chars (real JD body, not just metadata)
        with store.connect() as conn:
            jobs_rows = conn.execute(
                "SELECT id, title, company, location, source, length(raw_text) AS L "
                "FROM jobs WHERE length(raw_text) >= 200 "
                "ORDER BY fetched_at DESC LIMIT 30"
            ).fetchall()
        jobs = [
            {"id": r[0], "title": r[1] or "(无标题)", "company": r[2] or "?",
             "location": r[3] or "", "source": r[4], "raw_text_len": r[5]}
            for r in jobs_rows
        ]
        master_resume_text = profile.raw_resume_text if profile else ""
        return templates.TemplateResponse(
            request, "tailor.html",
            _ctx(
                request,
                jobs=jobs,
                selected_job_id=job_id,
                master_resume=master_resume_text,
                active_tab="tailor",
            ),
        )

    @app.post("/api/tailor/docx", response_class=HTMLResponse)
    def tailor_docx_run(
        request: Request,
        job_id: str = Form(...),
    ) -> Any:
        """Real .docx tailoring — preserves Word format end-to-end.

        Reads the user's master resume (must be .docx via OFFERGUIDE_RESUME_PDF),
        the selected job's JD text, runs ``docx_tailor.tailor_docx()`` with
        real LLM, and saves the output to ``data/tailored/<job_id>_<ts>.docx``.
        Renders a result fragment with summary + change_log + download link.
        """
        import time as _time
        from pathlib import Path as _Path

        from ..skills.tailor_resume.docx_tailor import tailor_docx

        if profile is None or not getattr(profile, "source_pdf", None):
            return templates.TemplateResponse(
                request, "_tailor_docx_result.html",
                _ctx(request, error="没加载简历——设 OFFERGUIDE_RESUME_PDF 后重启"),
            )
        master_path = _Path(profile.source_pdf)
        if master_path.suffix.lower() != ".docx":
            return templates.TemplateResponse(
                request, "_tailor_docx_result.html",
                _ctx(request, error=(
                    "DOCX tailoring 要求 master 简历是 .docx 格式。"
                    f"现在是 {master_path.suffix}。把 OFFERGUIDE_RESUME_PDF "
                    "指向 .docx 文件后重启。"
                )),
            )
        if runtime is None:
            return templates.TemplateResponse(
                request, "_tailor_docx_result.html",
                _ctx(request, error="未配置 LLM——设 OFFERGUIDE_LLM_API_KEY 后重启"),
            )

        try:
            jid = int(job_id)
        except ValueError:
            return templates.TemplateResponse(
                request, "_tailor_docx_result.html",
                _ctx(request, error="job_id 必须是整数"),
            )

        with store.connect() as conn:
            row = conn.execute(
                "SELECT title, company, raw_text FROM jobs WHERE id = ?", (jid,),
            ).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "_tailor_docx_result.html",
                _ctx(request, error=f"找不到 job #{jid}"),
            )
        title, company, job_text = row
        company = (company or "").strip() or "(未知公司)"

        # Build output filename + path
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company = company.replace("/", "_").replace(" ", "_")[:20]
        output_dir = _Path("data/tailored")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"tailored_{safe_company}_{ts}.docx"
        output_path = output_dir / output_filename

        try:
            # Use the runtime's LLM directly (it's the same instance)
            llm = runtime._llm
            result = tailor_docx(
                input_path=master_path,
                output_path=output_path,
                jd_text=job_text,
                company=company,
                role_focus=title or "",
                master_resume_text=profile.raw_resume_text,
                llm=llm,
            )
        except Exception as e:
            return templates.TemplateResponse(
                request, "_tailor_docx_result.html",
                _ctx(request, error=f"tailor_docx 失败: {e}"),
            )

        return templates.TemplateResponse(
            request, "_tailor_docx_result.html",
            _ctx(
                request,
                docx_result=result,
                summary=result.summary(),
                changes=result.changes,
                skipped=result.skipped[:10],  # cap display
                download_filename=output_filename,
                job_title=title, job_company=company,
                _time=_time,  # for cache-bust query
            ),
        )

    @app.get("/api/tailor/preview/{filename}", response_class=HTMLResponse)
    def tailor_preview(filename: str) -> Any:
        """Render a tailored .docx as in-browser HTML (W13.9).

        Embedded as iframe in /tailor result page so user can see the
        tailored resume before downloading. User can ⌘P → "Save as PDF"
        to get a portable PDF (browser embeds fonts).
        """
        from pathlib import Path as _Path

        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(400, "invalid filename")
        if not filename.endswith(".docx"):
            raise HTTPException(400, "not a docx")

        path = _Path("data/tailored") / filename
        if not path.exists():
            raise HTTPException(404, "file not found")

        try:
            from ..skills.tailor_resume.preview import docx_to_html
            html = docx_to_html(path)
        except ImportError:
            raise HTTPException(
                500, "mammoth not installed — pip install mammoth"
            ) from None
        except Exception as e:
            raise HTTPException(500, f"preview generation failed: {e}") from None
        return HTMLResponse(content=html, media_type="text/html; charset=utf-8")

    @app.get("/api/tailor/pdf/{filename}")
    def tailor_pdf(filename: str) -> Any:
        """Convert .docx to PDF via libreoffice (if installed) and serve it.

        Returns 503 if libreoffice not available — UI then suggests user
        use the HTML preview's ⌘P / Ctrl-P print-to-PDF instead.
        """
        from pathlib import Path as _Path

        from fastapi.responses import FileResponse

        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(400, "invalid filename")
        if not filename.endswith(".docx"):
            raise HTTPException(400, "not a docx")

        path = _Path("data/tailored") / filename
        if not path.exists():
            raise HTTPException(404, "file not found")

        try:
            from ..skills.tailor_resume.preview import soffice_to_pdf
            pdf_path = soffice_to_pdf(path, _Path("data/tailored/pdf"))
        except Exception as e:
            raise HTTPException(500, f"pdf conversion failed: {e}") from None

        if pdf_path is None:
            raise HTTPException(
                503,
                "libreoffice not installed; use the HTML preview's "
                "⌘P / Ctrl-P → 'Save as PDF' instead",
            )

        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            filename=pdf_path.name,
        )

    @app.get("/api/tailor/download/{filename}")
    def tailor_download(filename: str) -> Any:
        """Serve a previously-tailored .docx for download.

        Filename validation: must match the format we wrote
        (``tailored_<company>_<ts>.docx``) — refuse path traversal.
        """
        from pathlib import Path as _Path

        from fastapi.responses import FileResponse

        # Defense in depth: filename must not contain path separators
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(400, "invalid filename")
        if not filename.endswith(".docx"):
            raise HTTPException(400, "not a docx")

        path = _Path("data/tailored") / filename
        if not path.exists():
            raise HTTPException(404, "file not found (regenerate?)")
        return FileResponse(
            str(path),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename,
        )

    @app.post("/api/tailor/run", response_class=HTMLResponse)
    def tailor_run(
        request: Request,
        job_id: str = Form(...),
    ) -> Any:
        """Invoke tailor_resume SKILL on selected job + master resume + (optional) profile."""
        if profile is None:
            return templates.TemplateResponse(
                request, "_tailor_result.html",
                _ctx(request, error="未加载简历——设 OFFERGUIDE_RESUME_PDF 后重启。"),
            )
        if runtime is None:
            return templates.TemplateResponse(
                request, "_tailor_result.html",
                _ctx(request, error="未配置 LLM——设 BASE_URL/TOKEN 后重启。"),
            )
        try:
            jid = int(job_id)
        except ValueError:
            return templates.TemplateResponse(
                request, "_tailor_result.html",
                _ctx(request, error="job_id 必须是整数"),
            )

        with store.connect() as conn:
            row = conn.execute(
                "SELECT title, company, raw_text FROM jobs WHERE id = ?", (jid,)
            ).fetchone()
        if row is None:
            return templates.TemplateResponse(
                request, "_tailor_result.html",
                _ctx(request, error=f"找不到 job #{jid}"),
            )
        title, company, job_text = row
        company = (company or "").strip() or "(未知公司)"

        # Optional successful_profile lookup — best-effort
        from .. import corpus_quality
        successful_profile_json = "{}"
        if company:
            samples = corpus_quality.fetch_high_quality(
                store, company=company, limit=5,
            )
            if samples:
                # Try to find an existing successful_profile run we can reuse
                with store.connect() as conn:
                    sp_row = conn.execute(
                        "SELECT output_json FROM skill_runs "
                        "WHERE skill_name = 'successful_profile' "
                        "  AND input_json LIKE ? "
                        "ORDER BY created_at DESC LIMIT 1",
                        (f'%"{company}"%',),
                    ).fetchone()
                if sp_row:
                    successful_profile_json = sp_row[0]

        spec = next((s for s in skills if s.name == "tailor_resume"), None)
        if spec is None:
            return templates.TemplateResponse(
                request, "_tailor_result.html",
                _ctx(request, error="tailor_resume SKILL 未加载"),
            )

        try:
            result = runtime.invoke(
                spec,
                {
                    "master_resume": profile.raw_resume_text,
                    "job_text": job_text,
                    "company": company,
                    "successful_profile_json": successful_profile_json,
                },
            )
        except LLMError as e:
            return templates.TemplateResponse(
                request, "_tailor_result.html",
                _ctx(request, error=f"tailor_resume LLM 失败: {e}"),
            )

        return templates.TemplateResponse(
            request, "_tailor_result.html",
            _ctx(
                request,
                tailored=result.parsed or {},
                run_id=result.skill_run_id,
                job_title=title, job_company=company,
            ),
        )

    @app.get("/mock", response_class=HTMLResponse)
    def mock_view(request: Request) -> Any:
        """Mock interview 入口 — 选公司 + role + 启动 turn-based 会话。

        会话状态保存在 URL query / form 里 (无 session 中间件), turn_history
        作为 hidden input 在每次 POST 来回传递。简单可靠。
        """
        return templates.TemplateResponse(
            request, "mock.html",
            _ctx(request, active_tab="mock"),
        )

    @app.post("/api/mock/turn", response_class=HTMLResponse)
    def mock_turn(
        request: Request,
        company: str = Form(...),
        role_focus: str = Form(""),
        turn_history_json: str = Form("[]"),
        last_user_answer: str = Form(""),
    ) -> Any:
        """Run one mock_interview turn. Returns the next-question + eval card."""
        if profile is None:
            return templates.TemplateResponse(
                request, "_mock_turn.html",
                _ctx(request, error="未加载简历——先设 OFFERGUIDE_RESUME_PDF"),
            )
        if runtime is None:
            return templates.TemplateResponse(
                request, "_mock_turn.html",
                _ctx(request, error="未配置 LLM——先设 BASE_URL/TOKEN"),
            )
        if not company.strip():
            return templates.TemplateResponse(
                request, "_mock_turn.html",
                _ctx(request, error="company 必填"),
            )

        # Pull latest prepare_interview run for this company (优先用预测题)
        prep_questions_json = "[]"
        with store.connect() as conn:
            prep_row = conn.execute(
                "SELECT output_json FROM skill_runs "
                "WHERE skill_name = 'prepare_interview' "
                "  AND input_json LIKE ? "
                "ORDER BY created_at DESC LIMIT 1",
                (f'%"{company.strip()}"%',),
            ).fetchone()
        if prep_row:
            try:
                pj = json_loads(prep_row[0])
                prep_questions_json = json_dumps(pj.get("expected_questions", []))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        spec = next((s for s in skills if s.name == "mock_interview"), None)
        if spec is None:
            return templates.TemplateResponse(
                request, "_mock_turn.html",
                _ctx(request, error="mock_interview SKILL 未加载"),
            )

        try:
            result = runtime.invoke(
                spec,
                {
                    "company": company.strip(),
                    "role_focus": role_focus.strip(),
                    "user_resume": profile.raw_resume_text,
                    "prep_questions_json": prep_questions_json,
                    "turn_history_json": turn_history_json,
                    "last_user_answer": last_user_answer,
                },
            )
        except LLMError as e:
            return templates.TemplateResponse(
                request, "_mock_turn.html",
                _ctx(request, error=f"mock_interview LLM 失败: {e}"),
            )

        parsed = result.parsed or {}

        # Append to turn_history if there was a previous question to evaluate
        new_history: list[dict] = []
        try:
            new_history = json_loads(turn_history_json)
        except (json.JSONDecodeError, TypeError):
            new_history = []
        if last_user_answer.strip() and parsed.get("evaluation_of_last_answer"):
            new_history.append({
                "question": parsed["evaluation_of_last_answer"]["question"],
                "user_answer": last_user_answer.strip(),
                "evaluation": parsed["evaluation_of_last_answer"],
            })

        # When complete, auto-feed transcript to post_interview_reflection
        reflection_run_id = None
        if parsed.get("session_status") == "complete" and new_history:
            transcript = _mock_history_to_transcript(
                new_history, company.strip(),
            )
            refl_spec = next(
                (s for s in skills if s.name == "post_interview_reflection"), None,
            )
            if refl_spec is not None:
                try:
                    refl = runtime.invoke(
                        refl_spec,
                        {
                            "company": company.strip(),
                            "prep_questions_json": prep_questions_json,
                            "actual_transcript": transcript,
                        },
                    )
                    reflection_run_id = refl.skill_run_id
                except LLMError:
                    pass  # non-fatal

        return templates.TemplateResponse(
            request, "_mock_turn.html",
            _ctx(
                request,
                turn=parsed,
                run_id=result.skill_run_id,
                company=company.strip(),
                role_focus=role_focus.strip(),
                turn_history_json=json_dumps(new_history, ensure_ascii=False),
                reflection_run_id=reflection_run_id,
            ),
        )

    @app.get("/profile/{company}", response_class=HTMLResponse)
    def profile_view(request: Request, company: str, role: str = "") -> Any:
        """成功者画像 + 简历 gap 投递前 briefing 页面。

        渲染流程：
        1. ``corpus_quality.fetch_high_quality`` 拉公司 + 角色匹配的高质量样本
        2. 调用 ``successful_profile`` SKILL 合成画像
        3. 调用 ``profile_resume_gap`` SKILL 对比简历，输出 4 桶 gap

        如果样本数 < 1，提示用户去 sweep 或先粘几条面经；
        如果 LLM/profile 未配置，给降级提示但仍展示样本数据。
        """
        from .. import corpus_quality

        samples = corpus_quality.fetch_high_quality(
            store, company=company, role_hint=role or None, limit=8,
        )

        # 数据不足，直接渲染空状态
        if not samples:
            return templates.TemplateResponse(
                request, "profile_briefing.html",
                _ctx(
                    request,
                    company=company,
                    role=role,
                    samples=[],
                    profile_result=None,
                    gap_result=None,
                    error="样本不足——还没有该公司的高质量面经/offer 帖。"
                          "去 /interviews 粘几条，或运行 corpus_refresh 任务。",
                    active_tab="profile",
                ),
            )

        if profile is None or runtime is None:
            return templates.TemplateResponse(
                request, "profile_briefing.html",
                _ctx(
                    request,
                    company=company,
                    role=role,
                    samples=samples,
                    profile_result=None,
                    gap_result=None,
                    error="未配置简历或 LLM——只能展示样本，无法合成画像。",
                    active_tab="profile",
                ),
            )

        # Find SKILLs
        profile_spec = next(
            (s for s in skills if s.name == "successful_profile"), None,
        )
        gap_spec = next(
            (s for s in skills if s.name == "profile_resume_gap"), None,
        )
        if profile_spec is None or gap_spec is None:
            return templates.TemplateResponse(
                request, "profile_briefing.html",
                _ctx(
                    request, company=company, role=role, samples=samples,
                    profile_result=None, gap_result=None,
                    error="successful_profile / profile_resume_gap SKILL 未加载",
                    active_tab="profile",
                ),
            )

        # Run successful_profile
        samples_for_skill = [
            {
                "id": s["id"],
                "content_kind": s["content_kind"],
                "raw_text": (s["raw_text"] or "")[:3000],  # cap to keep prompt small
                "source": s["source"],
                "source_url": s["source_url"] or "",
                "quality_score": s["quality_score"],
            }
            for s in samples
        ]
        profile_result = None
        profile_run_id = None
        try:
            sp = runtime.invoke(
                profile_spec,
                {
                    "company": company,
                    "role_hint": role or "",
                    "high_quality_samples_json": json_dumps(samples_for_skill),
                },
            )
            profile_result = sp.parsed or {}
            profile_run_id = sp.skill_run_id
        except LLMError as e:
            return templates.TemplateResponse(
                request, "profile_briefing.html",
                _ctx(
                    request, company=company, role=role, samples=samples,
                    profile_result=None, gap_result=None,
                    error=f"successful_profile 调用失败: {e}",
                    active_tab="profile",
                ),
            )

        # Run profile_resume_gap
        gap_result = None
        gap_run_id = None
        try:
            gr = runtime.invoke(
                gap_spec,
                {
                    "successful_profile_json": json_dumps(profile_result),
                    "user_resume": profile.raw_resume_text,
                },
            )
            gap_result = gr.parsed or {}
            gap_run_id = gr.skill_run_id
        except LLMError as e:
            # Profile rendered, gap missing — degrade gracefully
            return templates.TemplateResponse(
                request, "profile_briefing.html",
                _ctx(
                    request, company=company, role=role, samples=samples,
                    profile_result=profile_result, profile_run_id=profile_run_id,
                    gap_result=None,
                    error=f"profile_resume_gap 调用失败: {e}",
                    active_tab="profile",
                ),
            )

        return templates.TemplateResponse(
            request, "profile_briefing.html",
            _ctx(
                request,
                company=company,
                role=role,
                samples=samples,
                profile_result=profile_result,
                profile_run_id=profile_run_id,
                gap_result=gap_result,
                gap_run_id=gap_run_id,
                active_tab="profile",
            ),
        )

    @app.get("/reflect", response_class=HTMLResponse)
    def reflect_view(request: Request) -> Any:
        """Page where the user submits an interview transcript for analysis."""
        with store.connect() as conn:
            recent_runs = conn.execute(
                "SELECT id, skill_name, skill_version, "
                "(julianday('now') - created_at) * 86400 AS age_seconds "
                "FROM skill_runs WHERE skill_name IN "
                "('prepare_interview', 'deep_project_prep') "
                "ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
        prep_runs = [
            {
                "id": r[0], "skill": r[1], "version": r[2],
                "when_ago": _humanize_age(float(r[3] or 0)),
            }
            for r in recent_runs
        ]
        return templates.TemplateResponse(
            request, "reflect.html",
            _ctx(request, prep_runs=prep_runs, active_tab="reflect"),
        )

    @app.post("/api/reflect/run", response_class=HTMLResponse)
    def reflect_run(
        request: Request,
        company: str = Form(...),
        prep_run_id: str = Form(""),
        actual_transcript: str = Form(...),
        auto_apply_stories: str = Form(""),  # checkbox value
        auto_apply_brief: str = Form(""),
    ) -> Any:
        """Run post_interview_reflection SKILL on the transcript + previous prep.

        When auto_apply_* checkboxes are set, automatically:
        - Insert suggested_stories into behavioral_stories table
        - Append brief_delta.interview_style_addition to company_briefs
        """
        if profile is None:
            return templates.TemplateResponse(
                request, "_reflect_result.html",
                _ctx(request, error="未加载简历——设 OFFERGUIDE_RESUME_PDF 后重启。"),
            )
        if runtime is None:
            return templates.TemplateResponse(
                request, "_reflect_result.html",
                _ctx(request, error="未配置 LLM——设 DEEPSEEK_API_KEY 后重启。"),
            )
        if not company.strip() or not actual_transcript.strip():
            return templates.TemplateResponse(
                request, "_reflect_result.html",
                _ctx(request, error="company 和 actual_transcript 都必填。"),
            )

        # Pull the prep run output to seed prep_questions_json
        prep_questions: list[dict] = []
        if prep_run_id.strip():
            try:
                rid = int(prep_run_id)
                with store.connect() as conn:
                    row = conn.execute(
                        "SELECT skill_name, output_json FROM skill_runs WHERE id = ?",
                        (rid,),
                    ).fetchone()
                if row:
                    skill_name, out_json = row
                    try:
                        out = json_loads(out_json)
                    except Exception:
                        out = {}
                    if skill_name == "prepare_interview":
                        prep_questions = out.get("expected_questions", [])
                    elif skill_name == "deep_project_prep":
                        # Flatten probing questions across projects + cross + behavioral
                        prep_questions = []
                        for proj in (out.get("projects_analyzed") or []):
                            prep_questions.extend(proj.get("probing_questions", []))
                        prep_questions.extend(out.get("cross_project_questions", []))
                        prep_questions.extend(out.get("behavioral_questions_tailored", []))
            except (ValueError, KeyError):
                pass

        # Find SKILL spec
        spec = next((s for s in skills if s.name == "post_interview_reflection"), None)
        if spec is None:
            return templates.TemplateResponse(
                request, "_reflect_result.html",
                _ctx(request, error="post_interview_reflection SKILL 未加载。"),
            )

        try:
            result = runtime.invoke(
                spec,
                {
                    "company": company.strip(),
                    "prep_questions_json": json_dumps(prep_questions),
                    "actual_transcript": actual_transcript.strip(),
                },
            )
        except LLMError as e:
            return templates.TemplateResponse(
                request, "_reflect_result.html",
                _ctx(request, error=f"LLM 调用失败: {e}"),
            )

        parsed = result.parsed or {}

        # Auto-apply: insert suggested stories
        applied_stories: list[int] = []
        if auto_apply_stories and parsed.get("suggested_stories"):
            from .. import story_bank
            for s in parsed["suggested_stories"]:
                try:
                    new_story = story_bank.insert(
                        store,
                        title=s.get("title", "(no title)"),
                        situation=s.get("suggested_situation", ""),
                        task=s.get("suggested_task", ""),
                        action=s.get("suggested_action", ""),
                        result=s.get("suggested_result", ""),
                        reflection=s.get("suggested_reflection") or None,
                        tags=s.get("suggested_tags", []),
                        confidence=0.5,
                    )
                    applied_stories.append(new_story.id)
                except (ValueError, KeyError):
                    pass

        # Auto-apply: brief delta
        brief_updated = False
        if auto_apply_brief and parsed.get("brief_delta"):
            from .. import briefs as briefs_mod
            existing = briefs_mod.get_brief(store, company.strip())
            delta = parsed["brief_delta"]
            addition = (delta.get("interview_style_addition") or "").strip()
            new_signals = list(delta.get("new_recent_signals") or [])
            conf_adj = float(delta.get("confidence_adjustment") or 0.0)
            if existing:
                # Merge: append addition + extend signals + adjust confidence
                merged_style = existing.brief.interview_style
                if addition and addition not in merged_style:
                    sep = " · " if merged_style.strip() else ""
                    merged_style = f"{merged_style.strip()}{sep}{addition}"
                merged_signals = list(existing.brief.recent_signals) + new_signals
                merged_signals = list(dict.fromkeys(merged_signals))[:8]
                from ..briefs import CompanyBrief, _upsert
                new_brief = CompanyBrief(
                    summary=existing.brief.summary,
                    current_app_limit=existing.brief.current_app_limit,
                    interview_style=merged_style,
                    recent_signals=merged_signals,
                    hiring_trend=existing.brief.hiring_trend,
                    confidence=max(0.0, min(1.0, existing.brief.confidence + conf_adj)),
                )
                _upsert(store, company.strip(), new_brief)
                brief_updated = True

        return templates.TemplateResponse(
            request, "_reflect_result.html",
            _ctx(
                request,
                reflection=parsed,
                run_id=result.skill_run_id,
                company=company.strip(),
                applied_stories=applied_stories,
                brief_updated=brief_updated,
            ),
        )

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
        from .. import briefs as briefs_mod
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            _ctx(
                request,
                stats=_full_stats(store),
                funnel=_application_funnel(store),
                evolutions=_recent_evolutions(store, limit=10),
                recent_runs=_recent_skill_runs(store, limit=10),
                briefs=briefs_mod.list_briefs(store, limit=10),
                daemon_health=_daemon_health(store),
                active_tab="dashboard",
            ),
        )

    # /chat removed in W13.1 — replaced by /agent (W13 central agent loop).
    # The old handler dispatched a hardcoded LangGraph (W4 graph.py) over the
    # SKILLs based on a `requested_action` enum from the form; users now go
    # to /agent and write a natural-language goal instead. The agent decides
    # which SKILLs to call, in what order, with what arguments. /inbox/from-report
    # also went away because it only existed to enqueue from /chat's report.

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

    @app.get("/api/search/test", response_class=JSONResponse)
    def search_test() -> dict:
        """Run a canary query against each search backend, return health.

        UI uses this to tell the user "your search backend is reachable"
        BEFORE relying on it for daily corpus_refresh sweeps. National
        firewalls can block DDG; Bing might serve CAPTCHA on certain
        IPs; Tavily depends on API key. This endpoint surfaces all 3.
        """
        import os as _os

        from ..agentic.search import (
            BingCNSearch,
            DuckDuckGoSearch,
            build_default_search,
            health_check,
        )
        try:
            tavily_check: dict | None = None
            if _os.environ.get("TAVILY_API_KEY"):
                from ..agentic.search import TavilySearch
                try:
                    tavily_check = health_check(TavilySearch())
                except Exception as e:
                    tavily_check = {
                        "name": "tavily", "ok": False, "hit_count": 0,
                        "sample_titles": [], "error_str": str(e)[:200],
                    }
            return {
                "default_chain": health_check(build_default_search()),
                "bing_cn":       health_check(BingCNSearch()),
                "duckduckgo":    health_check(DuckDuckGoSearch()),
                "tavily":        tavily_check or {
                    "name": "tavily", "ok": None, "hit_count": 0,
                    "sample_titles": [],
                    "error_str": "TAVILY_API_KEY 未配置 — 设了自动启用",
                },
                "guidance": _search_guidance_message(),
            }
        except Exception as e:
            return {"error": str(e)[:300]}

    # ─────────────────── W13.8 extension support ─────────────────────────
    # Boss直聘/牛客 浏览器扩展用 GET /api/extension/{ping,package} 查/拉
    # 投递包。CORS 必须开 (扩展从 https://www.zhipin.com 来 fetch http://localhost)。

    @app.get("/api/extension/ping", response_class=JSONResponse)
    def extension_ping() -> JSONResponse:
        """Health check endpoint for the browser extension popup."""
        resp = JSONResponse({"ok": True, "version": "0.1.0"})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.get("/api/extension/package", response_class=JSONResponse)
    def extension_get_package(company: str) -> JSONResponse:
        """Return the most recent apply_assistant package for a company.

        Used by the browser extension content script — when user is on a
        Boss直聘 / 牛客 page, the extension sniffs company name from
        title and asks here for a paste-ready package.

        Looks up via skill_runs.input_json LIKE filter (a bit hacky but
        the pre-W13.x schema doesn't index by company; would need a
        join through jobs to do better, leave for later).
        """
        company = (company or "").strip()
        if not company:
            return _ext_response(404, {"error": "company required"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT s.id, s.output_json, j.id "
                "FROM skill_runs s "
                "LEFT JOIN jobs j ON j.company = ? "
                "WHERE s.skill_name = 'apply_assistant' "
                "  AND s.input_json LIKE ? "
                "ORDER BY s.created_at DESC LIMIT 1",
                (company, f'%"company": "{company}"%'),
            ).fetchone()
        if row is None:
            return _ext_response(404, {
                "error": "no apply package",
                "hint": f"先去 OfferGuide /apply/<job_id> 跑 apply_assistant 给 {company} 准备一份",
            })
        run_id, output_json, job_id = row
        try:
            package = json_loads(output_json)
        except (json.JSONDecodeError, TypeError):
            return _ext_response(500, {"error": "stored package is corrupt"})
        return _ext_response(200, {
            "skill_run_id": run_id,
            "package": package,
            "job_id": job_id,
            "company": company,
        })

    def _ext_response(status: int, body: dict) -> JSONResponse:
        """Wrap with CORS headers (extension origin is the platform site, not localhost)."""
        resp = JSONResponse(body, status_code=status)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

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


def _recent_evolutions(store: Store, *, limit: int = 10) -> list[dict[str, Any]]:
    """Latest evolution_log rows as plain dicts (for dashboard rendering).

    Inlined the row→dict conversion in W13.1 so we can drop the old
    evolution/diff.py module. The dashboard route gets a full rewrite
    in W13.5; this is just to keep templates rendering until then.
    """
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, skill_name, parent_version, new_version, metric_name, "
            "metric_before, metric_after, notes, created_at "
            "FROM evolution_log ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        before = float(r[5]) if r[5] is not None else 0.0
        after = float(r[6]) if r[6] is not None else 0.0
        out.append({
            "id": r[0], "skill_name": r[1],
            "parent_version": r[2], "new_version": r[3],
            "metric_name": r[4],
            "metric_before": before, "metric_after": after,
            "metric_before_total": before, "metric_after_total": after,
            "delta_total": after - before,
            "notes": r[7], "created_at": r[8],
        })
    return out


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


def _search_guidance_message() -> str:
    """One-line guidance shown next to /api/search/test results."""
    import os as _os
    if _os.environ.get("TAVILY_API_KEY"):
        return "已配 Tavily — 最稳定。Bing/DDG 作为兜底。"
    return (
        "未配 TAVILY_API_KEY。强烈推荐：去 https://tavily.com 注册"
        "（1000 次/月免费），TAVILY_API_KEY=tvly-... 加到 .env, "
        "重启即自动启用。Bing CN 国内偶发 CAPTCHA, DDG 国内常被墙。"
    )


def _daemon_health(store: Store) -> list[dict[str, Any]]:
    """Per-job health summary for /dashboard 'daemon 健康' card.

    For each known job name, find the latest daemon_runs row and report:
    last_run_when (humanized), last_status, last_summary_str, run_count,
    error_count_24h. If a job has no rows ever, status = 'never'.

    The 7 known job names are hardcoded — order matches the daily timeline.
    """
    import json as _json

    job_names = (
        "extract_facts",   # 02:00
        "discover_jobs",   # 06:30
        "jd_enrich",       # 06:45
        "corpus_classify", # 07:00
        "silence_check",   # 09:00
        "corpus_refresh",  # Mon 08:00
        "brief_update",    # 23:00
    )
    out: list[dict[str, Any]] = []
    with store.connect() as conn:
        for name in job_names:
            latest = conn.execute(
                "SELECT started_at, ended_at, status, summary_json, error_text "
                "FROM daemon_runs WHERE job_name = ? "
                "ORDER BY started_at DESC LIMIT 1",
                (name,),
            ).fetchone()
            count24h = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) "
                "FROM daemon_runs WHERE job_name = ? "
                "  AND started_at >= julianday('now', '-1 day')",
                (name,),
            ).fetchone()
            run_count = int(count24h[0] or 0) if count24h else 0
            err_count = int(count24h[1] or 0) if count24h else 0
            if latest is None:
                out.append({
                    "name": name, "status": "never", "last_run_when": "—",
                    "last_summary_str": "从未运行 — daemon 可能没启动",
                    "run_count_24h": 0, "error_count_24h": 0,
                })
                continue
            started_at, ended_at, status, summary_json, error_text = latest
            try:
                age_seconds = (
                    (_to_julian(__import__('datetime').datetime.now(tz=UTC))
                     - float(started_at)) * 86400
                )
            except Exception:
                age_seconds = 0
            try:
                summary = _json.loads(summary_json) if summary_json else {}
            except Exception:
                summary = {}
            summary_str = (
                ", ".join(f"{k}={v}" for k, v in list(summary.items())[:5])
                if summary else (error_text or "(no summary)")
            )
            out.append({
                "name": name, "status": status,
                "last_run_when": _humanize_age(age_seconds),
                "last_summary_str": summary_str,
                "run_count_24h": run_count,
                "error_count_24h": err_count,
            })
    return out


def _mock_history_to_transcript(history: list[dict], company: str) -> str:
    """Render mock interview turns as a transcript that
    post_interview_reflection's expects."""
    lines = [f"{company} mock interview transcript ({len(history)} 轮):", ""]
    for i, turn in enumerate(history, 1):
        q = turn.get("question") or "(无题)"
        a = turn.get("user_answer") or "(无答)"
        ev = turn.get("evaluation") or {}
        score = ev.get("score") or 0
        lines.append(f"## 第 {i} 题 — {q}")
        lines.append(f"答: {a}")
        lines.append(f"  agent 评分: {score:.2f}")
        lines.append("")
    return "\n".join(lines)


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
