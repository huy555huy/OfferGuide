"""FastAPI web UI — agent + supporting pages.

Routes (after W13.1 cleanup):

    GET  /                     Agent Chat workbench
    GET  /today                daily standup / status home
    GET  /agent                alias for the central agent loop entry point
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

import asyncio
import contextlib
import hashlib
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError

from .. import inbox as inbox_mod
from ..config import Settings
from ..interview_research import (
    InterviewResearchConflictError,
    InterviewResearchRepository,
    InterviewResearchSubject,
    init_interview_research_schema,
)
from ..llm import LLMClient
from ..manual_job import ManualJobIntakeError, intake_manual_job
from ..memory import Store
from ..research_agents.browser_bridge import AuthenticatedBrowserBridgeStore
from ..research_agents.coordinator import init_agent_invocation_schema
from ..research_agents.job_discovery import (
    JobDiscoveryRepository,
    JobDiscoveryRevisionConflict,
    init_job_discovery_schema,
)
from ..research_agents.service import ResearchAgentService
from ..research_agents.sources import EvidenceNotFoundError, SourceEvidenceStore
from ..resume import (
    ApplicationPackage,
    MasterResumeDocument,
    MasterResumeSource,
    ResumeEditor,
    ResumeWorkflow,
    ResumeWorkspaceError,
    ResumeWorkspaceRepository,
    WorkspaceNotFoundError,
    generate_application_package,
    job_snapshot_from_evidence,
    load_resume_pdf,
)
from ..skills import SkillInvoker, SkillRuntime, SkillSpec, discover_skills
from .browser_bridge import build_browser_bridge_router
from .notify import Notifier, make_notifier

log = logging.getLogger(__name__)

# Convenience aliases used in handlers (post here so lint isort is happy).
json_loads = json.loads
json_dumps = json.dumps

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    *,
    settings: Settings,
    store: Store,
    master_source: MasterResumeSource | None,
    skills: list[SkillSpec],
    runtime: SkillInvoker | None,
    resume_editor: ResumeEditor | None = None,
    visual_resume_editor: ResumeEditor | None = None,
    notifier: Notifier | None = None,
    research_agents: ResearchAgentService | None = None,
) -> FastAPI:
    """Build the FastAPI application with explicit dependencies (testable)."""
    master_repository = ResumeWorkspaceRepository(store)
    interview_repository = (
        research_agents.interview_repository
        if research_agents is not None
        else InterviewResearchRepository(store)
    )
    interview_source_store = (
        research_agents.source_store
        if research_agents is not None
        else SourceEvidenceStore(store)
    )
    interview_source_store.init_schema()
    browser_bridge_store = (
        getattr(research_agents, "browser_bridge_store", None)
        if research_agents is not None
        else None
    ) or AuthenticatedBrowserBridgeStore(store)
    browser_bridge_store.init_schema()
    init_interview_research_schema(store)
    init_job_discovery_schema(store)
    init_agent_invocation_schema(store)
    def _effective_master_text() -> str | None:
        return master_repository.effective_master_text(master_source)

    # The background trigger owns no discovery logic. It only wakes the same
    # JobDiscoveryAgent used by the page and main-agent delegation.
    @contextlib.asynccontextmanager
    async def _lifespan(app: FastAPI):
        background_task: asyncio.Task[None] | None = None

        async def _scheduled_job_discovery() -> None:
            while True:
                try:
                    if (
                        research_agents is not None
                        and research_agents.job_repository.get_search_context() is not None
                    ):
                        research_agents.enqueue_job_discovery(
                            trigger_reason="scheduled market refresh"
                        )
                except Exception:
                    log.exception("scheduled JobDiscoveryAgent trigger failed")
                await asyncio.sleep(6 * 60 * 60)

        if research_agents is not None and not settings.disable_background_agents:
            background_task = asyncio.create_task(
                _scheduled_job_discovery(),
                name="offerguide-job-discovery-agent",
            )
        try:
            yield
        finally:
            if background_task is not None:
                background_task.cancel()
                with contextlib.suppress(BaseException):
                    await background_task
            if research_agents is not None:
                research_agents.close()

    app = FastAPI(
        title="OfferGuide", docs_url=None, redoc_url=None, lifespan=_lifespan,
    )
    app.include_router(build_browser_bridge_router(browser_bridge_store))
    app.state.browser_bridge_store = browser_bridge_store
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Static asset mount for the live UI stylesheet and browser extension assets.
    if STATIC_DIR.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _ctx(request: Request, **extra: Any) -> dict[str, Any]:
        # W14.5 — compute nav badges so every page shows live counts:
        #   inbox: pending items
        #   goals: off-track goals
        #   agent: last critic score (color-coded in template)
        # Single SQL pass each — fast, doesn't dominate page render.
        # Note: `last_critic` is always None now — W13 LLM-self-critique was
        # retired; real signal lives in evolution_signals (user thumbs / app
        # outcome / follow-through), not a per-run score. Kept as a field
        # so templates with `is not none` guards keep working.
        nav = {"inbox_pending": 0, "goals_off_track": 0, "last_critic": None}
        try:
            with store.connect() as conn:
                nav["inbox_pending"] = conn.execute(
                    "SELECT COUNT(*) FROM inbox_items WHERE status='pending'"
                ).fetchone()[0]
            # Off-track goal count needs goals module (avoid circular)
            try:
                from .. import goals as _gmod
                for g in _gmod.list_active_goals(store):
                    p = _gmod.compute_progress(store, g)
                    if not p.is_on_track:
                        nav["goals_off_track"] += 1
            except Exception:
                pass
        except Exception:
            pass

        base = {
            "request": request,
            "master_loaded": master_source is not None,
            "master_chars": len(_effective_master_text() or ""),
            "nav": nav,
        }
        base.update(extra)
        return base


    @app.get("/", response_class=HTMLResponse)
    def mission_control(request: Request) -> Any:
        """Mission Control — the W21 redesign home (2026-05-15).

        Replaces /today /agent /inbox /goals /dashboard. Layout from
        static/redesign/screens/a-home.html — bone bg, navy signal, left
        56px rail + center main + right 360px agent live rail.

        Data binding kept minimal in commit 3 — interrupts / recent_jobs
        / agent log are wired to real db. Further binding (SSE-streamed
        agent_now etc.) lands in later commits.
        """
        import datetime as _dt
        import json as _json
        # Greeting by local hour
        hour = _dt.datetime.now().hour
        greeting = "早上好" if hour < 11 else ("下午好" if hour < 18 else "晚上好")

        # Interrupts: pending inbox items (kind=question + agent_suggestion)
        interrupts: list[dict] = []
        try:
            pending = inbox_mod.list_items(store, status="pending", limit=10)
            for it in pending:
                meta = []
                meta.append({"text": it.kind, "cls": "tag"})
                interrupts.append({
                    "id": it.id,
                    "kind": it.kind,
                    "score": None,
                    "title": it.title or "(无标题)",
                    "meta": meta,
                    "ask": (it.body or "")[:160] if it.body else None,
                    "question_options": it.question_options or [],
                    "actions": [{"label": "处理", "kind": "signal"}],
                })
        except Exception:
            pass
        # Cap to 5 on hero strip
        interrupts = interrupts[:5]

        # Home shows the same current selection as /recommended. It does not
        # independently score, filter, or manufacture another recommendation set.
        recent_jobs: list[dict] = []
        try:
            job_repository = (
                getattr(research_agents, "job_repository", None)
                or JobDiscoveryRepository(store)
            )
            selection = job_repository.get_current_selection()
            with store.connect() as conn:
                workspace_job_ids = {
                    int(row[0])
                    for row in conn.execute(
                        "SELECT DISTINCT a.job_id FROM resume_workspaces AS rw "
                        "JOIN applications AS a ON a.id = rw.application_id"
                    ).fetchall()
                }
            _src_map = {
                "nowcoder": ("牛", "nc"), "tencent_campus": ("腾", "tc"),
                "tencent_social": ("腾", "tc"), "baidu_campus": ("百", "bd"),
                "baidu_intern": ("百", "bd"), "bytedance_jobs": ("字", "ali"),
                "shixiseng": ("僧", ""), "zerovoice_repo": ("0v", ""),
                "manual": ("M", ""),
            }
            for selected in selection.items if selection is not None else ():
                assert selection is not None
                evidence = selected.job_evidence
                if evidence is None or evidence.job_id is None:
                    continue
                live_evidence = job_repository.get_job_evidence(
                    selected.job_evidence_id
                )
                live_status = (
                    live_evidence.source_status
                    if live_evidence is not None
                    else "unknown"
                )
                has_workspace = evidence.job_id in workspace_job_ids
                source = evidence.source_name
                src_label, src_cls = _src_map.get(source, (source[:1].upper() if source else "?", ""))
                recent_jobs.append({
                    "id": evidence.job_id,
                    "title": evidence.title,
                    "subtitle": " · ".join(
                        value for value in (
                            evidence.company,
                            evidence.location or "",
                            source,
                        ) if value
                    ),
                    "src_label": src_label, "src_cls": src_cls,
                    "tag_cls": "ok" if live_status != "closed" else "",
                    "tag_label": (
                        "已有投递包"
                        if has_workspace
                        else ("已关闭" if live_status == "closed" else "已核验")
                    ),
                    "apply_pack_url": (
                        f"/jobs/{evidence.job_id}/apply-pack"
                        if has_workspace
                        else (
                            None
                            if live_status == "closed"
                            else (
                                f"/jobs/{evidence.job_id}/apply-pack"
                                f"?selection_revision={selection.result_revision}"
                                f"&job_evidence_id={selected.job_evidence_id}"
                            )
                        )
                    ),
                })
                if len(recent_jobs) >= 5:
                    break
            jobs_today_count = len(selection.items) if selection is not None else 0
        except Exception:
            jobs_today_count = 0

        # Durable application resources keep apply packs and interview results
        # reachable after the user leaves their original page.
        try:
            application_resources = _application_resources()
        except Exception:
            application_resources = {}
        resources_by_job = {
            int(resource["job_id"]): resource
            for resource in application_resources.values()
        }

        # Recent agent events (last 6h), including the domain Agents that keep
        # running after the initiating chat request has already returned.
        recent_events: list[dict] = []
        recent_artifacts: list[dict] = []
        try:
            with store.connect() as conn:
                erows = conn.execute(
                    "SELECT kind, job_id, note, created_at FROM harness_events "
                    "WHERE created_at >= julianday('now') - 0.25 "
                    "ORDER BY id DESC LIMIT 8"
                ).fetchall()
            for ek, ej, en, ec in erows:
                ts = _dt.datetime.fromtimestamp(
                    (float(ec) - 2440587.5) * 86400, tz=_dt.UTC,
                ).astimezone().strftime("%H:%M")
                verb_map = {
                    "scored": "评估", "applied": "投递", "user_marked_applied": "标记已投",
                    "project_record_saved": "存项目档案", "dead_url_reported": "标失效",
                }
                note_obj = {}
                with contextlib.suppress(Exception):
                    note_obj = _json.loads(en or "{}")
                title = ""
                view = ""
                if isinstance(note_obj, dict):
                    title = str(note_obj.get("title") or "")
                    view = str(note_obj.get("view") or "")
                if ek == "project_record_saved" and title:
                    recent_artifacts.append({
                        "time": ts,
                        "title": title,
                        "kind": str(note_obj.get("direction") or "project"),
                        "href": view or "/project-vault",
                        "_sort_at": float(ec),
                    })
                recent_events.append({
                    "time": ts, "kind": "log",
                    "verb": verb_map.get(ek, ek),
                    "obj": title or (f"job#{ej}" if ej else ""),
                    "tail": "", "href": view or None, "_sort_at": float(ec),
                })

                if ej and int(ej) in resources_by_job:
                    resource = resources_by_job[int(ej)]
                    recent_events[-1]["obj"] = " · ".join(
                        value
                        for value in (resource["company"], resource["title"])
                        if value
                    )
                    if ek == "user_marked_applied" and resource["interview_url"]:
                        recent_events[-1]["href"] = resource["interview_url"]

            with store.connect() as conn:
                invocation_rows = conn.execute(
                    "SELECT id, subject_kind, subject_id, status, updated_at "
                    "FROM research_agent_invocations "
                    "WHERE updated_at >= julianday('now') - 0.25 "
                    "ORDER BY updated_at DESC LIMIT 10"
                ).fetchall()
            for invocation_id, subject_kind, subject_id, status, updated_at in invocation_rows:
                timestamp = float(updated_at)
                ts = _dt.datetime.fromtimestamp(
                    (timestamp - 2440587.5) * 86400, tz=_dt.UTC,
                ).astimezone().strftime("%H:%M")
                if subject_kind == "job_search":
                    verb = "正在找岗位" if status in {"queued", "running"} else (
                        "岗位已更新" if status == "published" else "岗位已检查"
                    )
                    obj = f"当前 {jobs_today_count} 个候选"
                    href = "/recommended"
                elif subject_kind == "interview_research":
                    parts = str(subject_id).split(":")
                    application_id = (
                        int(parts[1])
                        if len(parts) == 4 and parts[0] == "application" and parts[1].isdigit()
                        else None
                    )
                    resource = application_resources.get(application_id) if application_id else None
                    verb = "正在搜索面经" if status in {"queued", "running"} else (
                        "面经已完成" if status == "published" else "面经已检查"
                    )
                    obj = (
                        " · ".join(
                            value
                            for value in (resource["company"], resource["title"])
                            if value
                        )
                        if resource
                        else "真实面经"
                    )
                    href = resource["interview_url"] if resource else None
                else:
                    continue
                recent_events.append({
                    "time": ts,
                    "kind": "log",
                    "verb": verb,
                    "obj": obj,
                    "tail": "",
                    "href": href,
                    "invocation_id": str(invocation_id),
                    "_sort_at": timestamp,
                })
        except Exception:
            pass

        for resource in application_resources.values():
            if not resource["interview_url"] or resource["updated_at"] is None:
                continue
            timestamp = float(resource["updated_at"])
            if timestamp < _to_julian(_dt.datetime.now(tz=_dt.UTC)) - 0.25:
                continue
            ts = _dt.datetime.fromtimestamp(
                (timestamp - 2440587.5) * 86400, tz=_dt.UTC,
            ).astimezone().strftime("%H:%M")
            recent_artifacts.append({
                "time": ts,
                "title": " · ".join(
                    value
                    for value in (resource["company"], resource["title"])
                    if value
                ),
                "kind": resource["interview_label"],
                "href": resource["interview_url"],
                "_sort_at": timestamp,
            })

        recent_events.sort(key=lambda item: float(item.get("_sort_at") or 0), reverse=True)
        recent_events = recent_events[:10]
        recent_artifacts.sort(
            key=lambda item: float(item.get("_sort_at") or 0), reverse=True
        )
        recent_artifacts = recent_artifacts[:6]

        agent_now_task = "当前没有运行中的任务"
        agent_now_href: str | None = None
        agent_poll_invocation_id: str | None = None
        agent_state_dot = "ok"
        agent_state_label = "待命"
        agent_overall_label = "待命"
        try:
            with store.connect() as conn:
                latest_invocation = conn.execute(
                    "SELECT id, subject_kind, subject_id, status, message "
                    "FROM research_agent_invocations "
                    "ORDER BY CASE WHEN status IN ('queued', 'running') THEN 0 ELSE 1 END, "
                    "updated_at DESC LIMIT 1"
                ).fetchone()
            if latest_invocation is not None:
                invocation_id, subject_kind, subject_id, status, message = latest_invocation
                active = status in {"queued", "running"}
                if subject_kind == "job_search":
                    agent_now_task = (
                        "正在核验来源并整理当前岗位选择"
                        if active
                        else f"找岗已完成，当前有 {jobs_today_count} 个候选"
                    )
                    agent_now_href = "/recommended"
                else:
                    parts = str(subject_id).split(":")
                    application_id = (
                        int(parts[1])
                        if len(parts) == 4 and parts[0] == "application" and parts[1].isdigit()
                        else None
                    )
                    resource = application_resources.get(application_id) if application_id else None
                    target = (
                        " · ".join(
                            value
                            for value in (resource["company"], resource["title"])
                            if value
                        )
                        if resource
                        else "当前投递"
                    )
                    agent_now_task = (
                        f"正在为 {target} 搜索真实面经"
                        if active
                        else (
                            f"{target}：{resource['interview_label']}"
                            if resource
                            else str(message or "面经搜索已结束")
                        )
                    )
                    agent_now_href = resource["interview_url"] if resource else None
                if active:
                    agent_poll_invocation_id = str(invocation_id)
                    agent_state_dot = "live"
                    agent_state_label = "运行中"
                    agent_overall_label = "运行中"
                elif status in {"failed", "blocked"}:
                    agent_state_dot = "warn"
                    agent_state_label = "需要处理"
                    agent_overall_label = "需要处理"
        except Exception:
            pass

        # Hero subtitle — what agent did in last 6h
        hero_subtitle = (
            f"Agent 在过去 6 小时里处理了 {len(recent_events)} 件事。"
            f"当前岗位选择集有 {jobs_today_count} 个岗位。"
            if recent_events or jobs_today_count else
            "Agent 待命中。可以修改当前找岗意图或手动刷新。"
        )

        ctx = _ctx(
            request,
            greeting=greeting,
            interrupts=interrupts,
            interrupts_count=len(interrupts),
            recent_jobs=recent_jobs,
            jobs_today_count=jobs_today_count,
            application_count=len(application_resources),
            agent_now_task=agent_now_task,
            agent_now_href=agent_now_href,
            agent_poll_invocation_id=agent_poll_invocation_id,
            agent_now_progress=None,
            agent_now_progress_label="",
            agent_now_eta="",
            agent_state_dot=agent_state_dot,
            agent_state_label=agent_state_label,
            agent_overall_label=agent_overall_label,
            recent_window_label="最近 6 小时",
            next_wake_label="Agent 运行记录",
            recent_events=recent_events,
            recent_artifacts=recent_artifacts,
            hero_subtitle=hero_subtitle,
            runtime_ready=runtime is not None and bool(settings.deepseek_api_key),
        )
        return templates.TemplateResponse(request, "mission_control.html", ctx)

    @app.get("/today", response_class=HTMLResponse)
    def home_legacy(_request: Request) -> Any:
        return RedirectResponse(url="/", status_code=301)

    @app.post("/api/home/wake-agent", response_class=JSONResponse)
    async def home_wake_agent(request: Request) -> Any:
        """Trigger an agent runtime run from the home page.

        User clicks "wake agent" on home page → agent runtime runs one loop with
        a user_button trigger, returns the run summary.
        """
        if not settings.deepseek_api_key:
            raise HTTPException(400, "agent 不可用 — 缺 OFFERGUIDE_LLM_API_KEY")

        from ..agent_runtime import (
            AgentRuntimeDeps,
            MemoryStore,
            TriggerEvent,
            default_worldview_dir,
        )
        from ..agent_runtime import _schema as _hs
        from ..agent_runtime import run as agent_runtime_run

        _hs.init_agent_runtime_schema(store)
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        try:
            from ..agentic.search import build_default_search
            try:
                _search = build_default_search()
            except Exception:
                _search = None
            deps = AgentRuntimeDeps(
                settings=settings, store=store,
                memory_store=MemoryStore(root=default_worldview_dir(settings)),
                llm=llm, runtime=runtime, skills=skills,
                search=_search, notifier=notifier,
                user_profile_text=_effective_master_text(),
                research_agents=research_agents,
            )
            import asyncio
            result = await asyncio.to_thread(
                agent_runtime_run,
                trigger=TriggerEvent(
                    kind="user_button",
                    detail={"reason": "home page wake-agent button"},
                ),
                deps=deps,
                max_iterations=6,
            )
        finally:
            with contextlib.suppress(Exception):
                llm.close()

        return {
            "run_id": result.run_id,
            "iterations": result.iterations,
            "latency_ms": result.latency_ms,
            "final_answer": result.final_text,
            "finish_reason": result.finish_reason,
        }

    @app.post("/api/home/chat", response_class=JSONResponse)
    async def home_chat(request: Request) -> Any:
        """W15.9 — chat input on home → user_input trigger to agent runtime.

        User types something ("帮我找几个字节实习" or "我要面字节明天准备一下").
        We package as `user_input` trigger and let the agent decide what tools
        to call. Returns a summary the home page shows ("做了 X, 看 Y").
        """
        if not settings.deepseek_api_key:
            raise HTTPException(400, "agent 不可用 — 缺 LLM API key")
        body = await request.json()
        message = (body.get("message") or "").strip()
        if not message:
            raise HTTPException(400, "message 不能为空")
        if len(message) > 2000:
            raise HTTPException(400, "message 太长 (max 2000)")

        from ..agent_runtime import (
            AgentRuntimeDeps,
            MemoryStore,
            default_worldview_dir,
            make_user_input_trigger,
        )
        from ..agent_runtime import _schema as _hs
        from ..agent_runtime import run as agent_runtime_run

        _hs.init_agent_runtime_schema(store)
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        try:
            from ..agentic.search import build_default_search
            try:
                _search = build_default_search()
            except Exception:
                _search = None
            deps = AgentRuntimeDeps(
                settings=settings, store=store,
                memory_store=MemoryStore(root=default_worldview_dir(settings)),
                llm=llm, runtime=runtime, skills=skills,
                search=_search, notifier=notifier,
                user_profile_text=_effective_master_text(),
                research_agents=research_agents,
            )
            import asyncio as _asyncio
            res = await _asyncio.to_thread(
                agent_runtime_run,
                trigger=make_user_input_trigger(message),
                deps=deps,
                max_iterations=15,
            )
        finally:
            with contextlib.suppress(Exception):
                llm.close()
        research_followup = None
        if res.run_id is not None:
            with store.connect() as conn:
                invocation_row = conn.execute(
                    "SELECT id, subject_kind, subject_id, status, message "
                    "FROM research_agent_invocations "
                    "WHERE status IN ('queued', 'running') "
                    "OR created_at >= COALESCE(("
                    "SELECT started_at FROM harness_runs WHERE id = ?"
                    "), julianday('now')) "
                    "ORDER BY created_at DESC LIMIT 1",
                    (res.run_id,),
                ).fetchone()
            if invocation_row is not None:
                followup_status = str(invocation_row[3])
                research_followup = {
                    "id": str(invocation_row[0]),
                    "status": followup_status,
                    "message": str(invocation_row[4] or ""),
                    "result_url": _research_invocation_result_url(
                        str(invocation_row[1]), str(invocation_row[2])
                    ),
                    "terminal": followup_status not in {"queued", "running"},
                }
        return {
            "run_id": res.run_id,
            "iterations": res.iterations,
            "finish_reason": res.finish_reason,
            "final_text": res.final_text[:2000],
            "tool_calls": res.tool_call_log[-10:],
            "cost_usd": round(res.cost_usd, 4),
            "research_invocation": research_followup,
        }

    @app.post("/api/jobs/manual", response_class=JSONResponse)
    async def create_manual_job(request: Request) -> Any:
        """Manual JD fallback: preserve the complete input and open one workspace."""
        body = await request.json()
        url_or_text = (body.get("url_or_text") or "").strip()
        try:
            result = await asyncio.to_thread(
                intake_manual_job,
                store=store,
                source_store=interview_source_store,
                source_reader=getattr(research_agents, "source_reader", None),
                url_or_text=url_or_text,
                company_hint=str(body.get("company_hint") or ""),
                title_hint=str(body.get("title_hint") or ""),
                location_hint=str(body.get("location_hint") or ""),
                source_url=(str(body.get("source_url") or "").strip() or None),
            )
        except ManualJobIntakeError as exc:
            status = 422 if url_or_text.startswith(("http://", "https://")) else 400
            raise HTTPException(status, str(exc)) from None
        return {
            "job_id": result.job_id,
            "is_new": result.is_new,
            "company": result.company,
            "title": result.title,
            "source_evidence_id": result.source_evidence_id,
            "apply_pack_url": f"/jobs/{result.job_id}/apply-pack",
        }

    def _application_for_job(self_job_id: int, *, create: bool) -> int | None:
        """Return the one active application row for a selected job."""
        try:
            application_id, _ = ResumeWorkspaceRepository(store).application_for_job(
                self_job_id,
                create=create,
            )
        except WorkspaceNotFoundError:
            raise HTTPException(404, f"job#{self_job_id} not found") from None
        except ResumeWorkspaceError as exc:
            raise HTTPException(409, str(exc)) from None
        return application_id
    def _write_apply_pack(
        job_snapshot: dict[str, Any],
        context: Any,
        editor_result: Any,
    ) -> dict[str, Any]:
        return generate_application_package(
            runtime=runtime,
            skills=skills,
            job_snapshot=job_snapshot,
            context=context,
            editor_result=editor_result,
        )

    def _resume_workflow() -> ResumeWorkflow:
        if master_source is None:
            raise HTTPException(409, "未加载 master PDF，请先配置 OFFERGUIDE_RESUME_PDF")
        if resume_editor is None:
            raise HTTPException(409, "简历编辑模型未配置")
        return ResumeWorkflow(
            store=store,
            master_source=master_source,
            editor=resume_editor,
            visual_editor=visual_resume_editor,
            apply_pack_writer=_write_apply_pack,
            artifact_root=(
                settings.db_path.expanduser().resolve().parent / "resume_workspaces"
            ),
        )

    def _submitted_interview_target(application_id: int) -> tuple[int, Any]:
        with store.connect() as conn:
            row = conn.execute(
                "SELECT job_id FROM applications WHERE id = ?",
                (application_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"application#{application_id} not found")
        workspace = ResumeWorkspaceRepository(store).get(application_id)
        if workspace is None or not workspace.is_submitted:
            raise HTTPException(409, "面试研究只接受已经实际提交并冻结的投递版本")
        return int(row[0]), workspace

    def _maybe_enqueue_interview_research(
        *,
        application_id: int,
        workspace_id: int,
        trigger_reason: str,
        force: bool,
    ) -> Any:
        if research_agents is None:
            return None
        subject = interview_repository.ensure_subject(application_id, workspace_id)
        latest = research_agents.latest_interview_invocation(
            application_id=application_id,
            workspace_id=workspace_id,
        )
        if (
            not force
            and latest is not None
            and latest.subject_revision == subject.agent_subject_revision
        ):
            return latest
        invocation, _created = research_agents.enqueue_interview_research(
            application_id=application_id,
            workspace_id=workspace_id,
            trigger_reason=trigger_reason,
        )
        return invocation

    @app.get("/api/resume-workspaces/{workspace_id}/pdf")
    def resume_workspace_pdf(workspace_id: int) -> Any:
        from fastapi.responses import FileResponse

        with store.connect() as conn:
            row = conn.execute(
                "SELECT pdf_path, pdf_sha256 FROM resume_workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
        if row is None or not row[0] or not row[1]:
            raise HTTPException(404, "resume PDF not found")
        path = Path(str(row[0])).expanduser().resolve()
        allowed_root = settings.db_path.expanduser().resolve().parent / "resume_workspaces"
        if allowed_root != path and allowed_root not in path.parents:
            raise HTTPException(400, "resume PDF is outside the artifact directory")
        if not path.is_file():
            raise HTTPException(404, "resume PDF file missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(row[1]):
            raise HTTPException(409, "resume PDF failed its integrity check")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=path.name,
            content_disposition_type="inline",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.post("/api/jobs/{job_id}/track", response_class=JSONResponse)
    def track_job(job_id: int, request: Request) -> Any:
        """W15.14: user clicks "加入跟踪" on an evaluation result card.

        Adds an applications row with status='considered' so /jobs page
        shows it and agent sees it in worldview/tracked-jobs.md (next
        wake will reflect via record_event).

        Idempotent: if already tracked, returns existing application_id.
        """
        try:
            application_id, created = ResumeWorkspaceRepository(store).application_for_job(
                job_id,
                create=True,
            )
        except WorkspaceNotFoundError:
            raise HTTPException(404, f"job#{job_id} not found") from None
        except ResumeWorkspaceError as exc:
            raise HTTPException(409, str(exc)) from None
        assert application_id is not None
        if not created:
            return {"application_id": application_id, "created": False}

        # Also fire a harness event so agent knows next wake
        from ..agent_runtime import _schema as _hs
        _hs.init_agent_runtime_schema(store)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_events(kind, job_id, note, source) "
                "VALUES (?, ?, ?, ?)",
                ("user_tracked", job_id, "user clicked '加入跟踪'", "user"),
            )
        return {"application_id": application_id, "created": True}

    @app.post("/api/jobs/{job_id}/applied", response_class=JSONResponse)
    def mark_applied(job_id: int, request: Request) -> Any:
        """Freeze the reviewed package only after the user confirms submission."""
        from ..resume import ResumeWorkspaceError, ResumeWorkspaceRepository

        application_id = _application_for_job(job_id, create=False)
        if application_id is None:
            raise HTTPException(409, "请先生成并审核投递包，再确认“我投了”")

        with store.connect() as conn:
            already_submitted = (
                conn.execute(
                    "SELECT 1 FROM application_events "
                    "WHERE application_id = ? AND kind = 'submitted' LIMIT 1",
                    (application_id,),
                ).fetchone()
                is not None
            )
        try:
            submitted = ResumeWorkspaceRepository(store).submit(application_id)
        except ResumeWorkspaceError as exc:
            raise HTTPException(409, f"无法冻结投递材料: {exc}") from None
        if not already_submitted:
            from ..agent_runtime import _schema as _hs

            _hs.init_agent_runtime_schema(store)
            with store.connect() as conn:
                conn.execute(
                    "INSERT INTO harness_events(kind, job_id, note, source) "
                    "VALUES ('user_marked_applied', ?, ?, 'user')",
                    (job_id, json.dumps({"workspace_id": submitted.id})),
                )
        research_invocation = None
        research_error = None
        try:
            interview_repository.ensure_subject(application_id, submitted.id)
            research_invocation = _maybe_enqueue_interview_research(
                application_id=application_id,
                workspace_id=submitted.id,
                trigger_reason="application submitted",
                force=False,
            )
        except Exception as exc:
            research_error = f"{type(exc).__name__}: {exc}"
            log.exception(
                "could not initialize interview research for application %s",
                application_id,
            )
        return {
            "application_id": application_id,
            "workspace_id": submitted.id,
            "workspace_status": submitted.status,
            "interview_research_status": (
                research_invocation.status if research_invocation is not None else None
            ),
            "interview_research_error": research_error,
        }

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_view(request: Request) -> Any:
        """W15.14: tracked jobs visualization. Replaces "look in worldview
        markdown" with a real UI."""
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT j.id, j.title, j.company, j.location, j.url, "
                "       a.id, a.status, a.applied_at, a.last_status_change "
                "FROM jobs j "
                "LEFT JOIN applications a ON a.job_id = j.id "
                "ORDER BY COALESCE(a.last_status_change, j.id) DESC "
                "LIMIT 100"
            ).fetchall()
        jobs_view_data: list[dict[str, Any]] = []
        for r in rows:
            jobs_view_data.append({
                "job_id": int(r[0]), "title": r[1], "company": r[2],
                "location": r[3], "url": r[4],
                "application_id": int(r[5]) if r[5] is not None else None,
                "status": r[6] or "evaluated",
                "applied_at": r[7], "last_status_change": r[8],
            })
        return templates.TemplateResponse(
            request,
            "jobs.html",
            _ctx(
                request,
                jobs=jobs_view_data,
                runtime_ready=runtime is not None and bool(settings.deepseek_api_key),
                active_tab="jobs",
            ),
        )


    def _published_job_snapshot(
        job_id: int,
        *,
        selection_revision: int | None = None,
        job_evidence_id: int | None = None,
    ) -> dict[str, Any] | None:
        repository = (
            getattr(research_agents, "job_repository", None)
            or JobDiscoveryRepository(store)
        )
        if (selection_revision is None) != (job_evidence_id is None):
            raise HTTPException(
                400,
                "selection_revision 和 job_evidence_id 必须同时提供",
            )
        if selection_revision is None:
            return None
        selection = repository.get_selection_revision(selection_revision)
        if selection is None:
            raise HTTPException(409, "这条岗位选择记录已不可读取，请返回推荐页重试")
        for item in selection.items:
            evidence = item.job_evidence
            if (
                evidence is not None
                and evidence.job_id == job_id
                and item.job_evidence_id == job_evidence_id
            ):
                live_evidence = repository.get_job_evidence(item.job_evidence_id)
                if (
                    live_evidence is not None
                    and live_evidence.job_id != evidence.job_id
                ):
                    raise HTTPException(
                        409,
                        "岗位证据的当前身份与用户看到的发布版本不一致",
                    )
                snapshot = job_snapshot_from_evidence(evidence.model_dump(mode="json"))
                snapshot["selection_result_revision"] = selection.result_revision
                snapshot["job_evidence_id"] = item.job_evidence_id
                snapshot["live_source_status"] = (
                    live_evidence.source_status
                    if live_evidence is not None
                    else "unknown"
                )
                snapshot["live_checked_at"] = (
                    live_evidence.checked_at if live_evidence is not None else None
                )
                return snapshot
        raise HTTPException(409, "岗位与用户看到的选择记录不一致，请返回推荐页重试")

    def _apply_pack_job(
        job_id: int,
        *,
        frozen_snapshot: Mapping[str, Any] | None = None,
        selection_revision: int | None = None,
        job_evidence_id: int | None = None,
    ) -> dict[str, Any]:
        if frozen_snapshot is not None:
            snapshot = dict(frozen_snapshot)
            snapshot_id = snapshot.get("job_id", snapshot.get("id"))
            if str(snapshot_id) != str(job_id):
                raise HTTPException(409, "投递工作区的岗位快照与当前岗位不一致")
            snapshot["id"] = job_id
            snapshot["job_id"] = job_id
            snapshot["raw_text"] = str(
                snapshot.get("raw_text") or snapshot.get("jd_text") or ""
            )
            snapshot["url"] = str(
                snapshot.get("url") or snapshot.get("canonical_url") or ""
            )
            return snapshot

        published = _published_job_snapshot(
            job_id,
            selection_revision=selection_revision,
            job_evidence_id=job_evidence_id,
        )
        if published is not None:
            return {"id": job_id, **published}

        with store.connect() as conn:
            row = conn.execute(
                "SELECT id, title, company, location, url, raw_text, source, extras_json "
                "FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"job#{job_id} 不存在")
        if str(row[6] or "") != "manual":
            raise HTTPException(
                409,
                "Agent 找到的岗位必须从带版本的岗位选择入口进入；"
                "只有用户手动补充的 JD 可以仅使用 job_id",
            )
        try:
            extras = json.loads(str(row[7] or "{}"))
        except (json.JSONDecodeError, TypeError):
            extras = {"unparsed_extras": str(row[7] or "")}
        return {
            "id": int(row[0]),
            "title": str(row[1] or ""),
            "company": str(row[2] or ""),
            "location": str(row[3] or ""),
            "url": str(row[4] or ""),
            "raw_text": str(row[5] or ""),
            "source": str(row[6] or ""),
            "extras": extras if isinstance(extras, dict) else {"value": extras},
        }

    def _ensure_new_workspace_allowed(job: Mapping[str, Any]) -> None:
        if job.get("live_source_status") == "closed":
            raise HTTPException(
                409,
                "该岗位最近一次检查显示已关闭，不能新建投递工作区",
            )

    def _application_entry_urls(job_ids: set[int]) -> dict[int, str | None]:
        """Resolve safe apply-pack links for non-recommendation UI surfaces."""
        if not job_ids:
            return {}
        placeholders = ",".join("?" for _ in job_ids)
        values = tuple(sorted(job_ids))
        with store.connect() as conn:
            rows = conn.execute(
                f"SELECT id, source FROM jobs WHERE id IN ({placeholders})",
                values,
            ).fetchall()
            workspace_job_ids = {
                int(row[0])
                for row in conn.execute(
                    f"SELECT DISTINCT a.job_id FROM applications AS a "
                    f"JOIN resume_workspaces AS rw ON rw.application_id = a.id "
                    f"WHERE a.job_id IN ({placeholders})",
                    values,
                ).fetchall()
            }
        sources = {int(row[0]): str(row[1] or "") for row in rows}
        repository = (
            getattr(research_agents, "job_repository", None)
            or JobDiscoveryRepository(store)
        )
        selection = repository.get_current_selection()
        selected_by_job = {
            item.job_evidence.job_id: item
            for item in (selection.items if selection is not None else ())
            if item.job_evidence is not None and item.job_evidence.job_id is not None
        }
        urls: dict[int, str | None] = {}
        for job_id in job_ids:
            if job_id in workspace_job_ids or sources.get(job_id) == "manual":
                urls[job_id] = f"/jobs/{job_id}/apply-pack"
                continue
            selected = selected_by_job.get(job_id)
            if selected is None or selection is None:
                urls[job_id] = None
                continue
            live = repository.get_job_evidence(selected.job_evidence_id)
            if live is not None and live.source_status == "closed":
                urls[job_id] = None
                continue
            urls[job_id] = (
                f"/jobs/{job_id}/apply-pack"
                f"?selection_revision={selection.result_revision}"
                f"&job_evidence_id={selected.job_evidence_id}"
            )
        return urls

    def _application_resources(
        application_ids: set[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Return the one durable apply-pack/interview entry for each application."""
        params: tuple[int, ...] = ()
        where_clause = ""
        if application_ids is not None:
            if not application_ids:
                return {}
            params = tuple(sorted(application_ids))
            where_clause = (
                "WHERE a.id IN (" + ",".join("?" for _ in params) + ")"
            )
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT a.id, a.job_id, j.company, j.title, rw.id, rw.status, "
                "rw.updated_at, rw.submitted_at, irc.current_result_revision, "
                "material.answer_set_json, material.created_at "
                "FROM applications AS a "
                "JOIN jobs AS j ON j.id = a.job_id "
                "LEFT JOIN resume_workspaces AS rw ON rw.application_id = a.id "
                "LEFT JOIN interview_research_contexts AS irc "
                "ON irc.workspace_id = rw.id AND irc.application_id = a.id "
                "LEFT JOIN interview_answer_set_revisions AS material "
                "ON material.workspace_id = rw.id "
                "AND material.result_revision = irc.current_result_revision "
                f"{where_clause} ORDER BY a.id DESC",
                params,
            ).fetchall()
            invocation_rows = conn.execute(
                "SELECT id, subject_id, status, message, error_text, updated_at "
                "FROM research_agent_invocations "
                "WHERE agent_name = 'interview_research_agent' "
                "ORDER BY created_at DESC"
            ).fetchall()

        latest_invocation_by_subject: dict[str, tuple[Any, ...]] = {}
        for invocation_row in invocation_rows:
            latest_invocation_by_subject.setdefault(
                str(invocation_row[1]), tuple(invocation_row)
            )

        entry_urls = _application_entry_urls({int(row[1]) for row in rows})
        resources: dict[int, dict[str, Any]] = {}
        for row in rows:
            application_id = int(row[0])
            job_id = int(row[1])
            workspace_id = int(row[4]) if row[4] is not None else None
            workspace_status = str(row[5]) if row[5] is not None else None
            material_status: str | None = None
            answer_count = 0
            if row[9]:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    material = json.loads(str(row[9]))
                    if isinstance(material, dict):
                        material_status = str(material.get("status") or "") or None
                        answers = material.get("answers")
                        if isinstance(answers, list):
                            answer_count = len(answers)

            invocation = None
            if workspace_id is not None:
                invocation = latest_invocation_by_subject.get(
                    f"application:{application_id}:workspace:{workspace_id}"
                )
            invocation_status = str(invocation[2]) if invocation is not None else None
            interview_url = (
                f"/jobs/{job_id}/post-apply-pack"
                if workspace_status == "submitted"
                else None
            )
            if invocation_status in {"queued", "running"}:
                interview_label = "面经搜索中"
                interview_status = "running"
                interview_status_label = "搜索中"
            elif material_status == "answered":
                interview_label = f"真实面经 · {answer_count} 题"
                interview_status = "ready"
                interview_status_label = "已完成"
            elif material_status == "not_found":
                interview_label = "未找到真实面经"
                interview_status = "not_found"
                interview_status_label = "未找到"
            elif invocation_status == "failed":
                interview_label = "面经搜索失败"
                interview_status = "error"
                interview_status_label = "失败"
            elif invocation_status == "blocked":
                interview_label = "面经证据不足"
                interview_status = "blocked"
                interview_status_label = "证据不足"
            else:
                interview_label = "打开面经搜索"
                interview_status = "idle"
                interview_status_label = "待开始"

            timestamps = [
                float(value)
                for value in (
                    row[6],
                    row[7],
                    row[10],
                    invocation[5] if invocation is not None else None,
                )
                if value is not None
            ]
            resources[application_id] = {
                "application_id": application_id,
                "job_id": job_id,
                "company": str(row[2] or ""),
                "title": str(row[3] or ""),
                "workspace_id": workspace_id,
                "workspace_status": workspace_status,
                "apply_pack_url": entry_urls.get(job_id),
                "interview_url": interview_url,
                "interview_label": interview_label,
                "interview_status": interview_status,
                "interview_status_label": interview_status_label,
                "answer_count": answer_count,
                "material_status": material_status,
                "invocation_id": str(invocation[0]) if invocation is not None else None,
                "invocation_status": invocation_status,
                "invocation_message": str(invocation[3] or "") if invocation else "",
                "invocation_error": str(invocation[4]) if invocation and invocation[4] else None,
                "updated_at": max(timestamps) if timestamps else None,
            }
        return resources

    def _pipeline_pending_job_ids() -> list[int]:
        """Current published candidates plus user-supplied fallback JDs, in order."""
        repository = (
            getattr(research_agents, "job_repository", None)
            or JobDiscoveryRepository(store)
        )
        selection = repository.get_current_selection()
        pending = [
            int(item.job_evidence.job_id)
            for item in (selection.items if selection is not None else ())
            if item.job_evidence is not None and item.job_evidence.job_id is not None
        ]
        with store.connect() as conn:
            manual_rows = conn.execute(
                "SELECT j.id FROM jobs AS j "
                "LEFT JOIN applications AS a ON a.job_id = j.id "
                "WHERE j.source = 'manual' AND a.id IS NULL "
                "ORDER BY j.fetched_at DESC, j.id DESC"
            ).fetchall()
        pending.extend(int(row[0]) for row in manual_rows)
        return list(dict.fromkeys(pending))

    def _research_invocation_result_url(
        subject_kind: str,
        subject_id: str,
    ) -> str | None:
        if subject_kind == "job_search":
            return "/recommended"
        if subject_kind != "interview_research":
            return None
        subject_parts = subject_id.split(":")
        if (
            len(subject_parts) != 4
            or subject_parts[0] != "application"
            or not subject_parts[1].isdigit()
        ):
            return None
        with store.connect() as conn:
            target = conn.execute(
                "SELECT job_id FROM applications WHERE id = ?",
                (int(subject_parts[1]),),
            ).fetchone()
        return (
            f"/jobs/{int(target[0])}/post-apply-pack"
            if target is not None
            else None
        )

    @app.get("/jobs/{job_id}/apply-pack", response_class=HTMLResponse)
    def apply_pack_view(
        job_id: int,
        request: Request,
        selection_revision: int | None = None,
        job_evidence_id: int | None = None,
    ) -> Any:
        """Read the one current workspace. Generating or editing requires POST."""
        application_id = _application_for_job(job_id, create=False)
        repo = ResumeWorkspaceRepository(store)
        workspace = repo.get(application_id) if application_id is not None else None
        if workspace is not None and (
            selection_revision is not None or job_evidence_id is not None
        ):
            _published_job_snapshot(
                job_id,
                selection_revision=selection_revision,
                job_evidence_id=job_evidence_id,
            )
        job = _apply_pack_job(
            job_id,
            frozen_snapshot=workspace.job_snapshot if workspace is not None else None,
            selection_revision=(selection_revision if workspace is None else None),
            job_evidence_id=(job_evidence_id if workspace is None else None),
        )
        stored_master = repo.get_master()
        master_ready = False
        master_text = master_source.extracted_text if master_source is not None else ""
        if (
            master_source is not None
            and stored_master is not None
            and stored_master.source_sha256 == master_source.sha256
        ):
            try:
                semantic = MasterResumeDocument.model_validate(stored_master.semantic_document)
            except ValidationError:
                semantic = None
            if semantic is not None:
                master_text = semantic.semantic_text
                master_ready = (
                    stored_master.semantic_status == "confirmed"
                    and semantic.confirmed_by_user
                )
        saved = workspace.apply_pack if workspace is not None else {}
        assistant = saved.get("assistant")
        pack = assistant if isinstance(assistant, dict) and "_error" not in assistant else {}
        pack_error = (
            str(assistant.get("_error"))
            if isinstance(assistant, dict) and assistant.get("_error")
            else None
        )
        visual_review = saved.get("visual_review")
        if not isinstance(visual_review, dict):
            visual_review = {}
        preparation_notes = saved.get("preparation_notes")
        if not isinstance(preparation_notes, list):
            preparation_notes = []
        pdf_url = None
        if workspace is not None and workspace.pdf_path and workspace.pdf_sha256:
            pdf_url = (
                f"/api/resume-workspaces/{workspace.id}/pdf"
                f"?v={workspace.pdf_sha256[:12]}"
            )
        return templates.TemplateResponse(
            request,
            "apply_pack.html",
            _ctx(
                request,
                job=job,
                selection_binding={
                    "selection_revision": job.get("selection_result_revision"),
                    "job_evidence_id": job.get("job_evidence_id"),
                },
                workspace=workspace,
                pdf_url=pdf_url,
                editor_note=str(saved.get("editor_note") or ""),
                preparation_notes=preparation_notes,
                visual_review=visual_review,
                master_available=master_source is not None,
                master_ready=master_ready,
                master_text=master_text,
                pack=pack,
                error=pack_error,
                workspace_creation_blocked=(
                    workspace is None and job.get("live_source_status") == "closed"
                ),
                active_tab="recommended",
            ),
        )

    @app.post("/api/jobs/{job_id}/resume-workspace/start")
    def start_resume_workspace(
        job_id: int,
        selection_revision: int | None = Form(None),
        job_evidence_id: int | None = Form(None),
    ) -> RedirectResponse:
        try:
            job_snapshot = _apply_pack_job(
                job_id,
                selection_revision=selection_revision,
                job_evidence_id=job_evidence_id,
            )
            _ensure_new_workspace_allowed(job_snapshot)
            _resume_workflow().start(
                job_id,
                job_snapshot=job_snapshot,
            )
        except HTTPException:
            raise
        except ResumeWorkspaceError as exc:
            raise HTTPException(409, str(exc)) from None
        except Exception as exc:
            log.exception("resume workspace start failed: %s", exc)
            raise HTTPException(500, f"简历生成失败: {exc}") from None
        return RedirectResponse(f"/jobs/{job_id}/apply-pack", status_code=303)

    @app.post("/api/jobs/{job_id}/resume-workspace/confirm-master")
    def confirm_master_and_start(
        job_id: int,
        semantic_text: str = Form(...),
        selection_revision: int | None = Form(None),
        job_evidence_id: int | None = Form(None),
    ) -> RedirectResponse:
        job_snapshot = _apply_pack_job(
            job_id,
            selection_revision=selection_revision,
            job_evidence_id=job_evidence_id,
        )
        _ensure_new_workspace_allowed(job_snapshot)
        if master_source is None:
            raise HTTPException(409, "未加载 master PDF，请先配置 OFFERGUIDE_RESUME_PDF")
        text = semantic_text.strip()
        if not text:
            raise HTTPException(400, "master 简历文本不能为空")
        semantic = MasterResumeDocument(
            source_sha256=master_source.sha256,
            semantic_text=text,
            confirmed_by_user=True,
        )
        ResumeWorkspaceRepository(store).save_master(
            source_path=master_source.source_path,
            source_sha256=master_source.sha256,
            extracted_text=master_source.extracted_text,
            semantic_document=semantic.model_dump(mode="json"),
            confirmed=True,
        )
        try:
            _resume_workflow().start(
                job_id,
                job_snapshot=job_snapshot,
            )
        except ResumeWorkspaceError as exc:
            raise HTTPException(409, str(exc)) from None
        except Exception as exc:
            log.exception("resume workspace start after master confirmation failed: %s", exc)
            raise HTTPException(500, f"简历生成失败: {exc}") from None
        return RedirectResponse(f"/jobs/{job_id}/apply-pack", status_code=303)

    @app.post("/api/jobs/{job_id}/resume-workspace/revise")
    def revise_resume_workspace(
        job_id: int,
        feedback: str = Form(...),
    ) -> RedirectResponse:
        try:
            _resume_workflow().revise(job_id, feedback=feedback)
        except HTTPException:
            raise
        except ResumeWorkspaceError as exc:
            raise HTTPException(409, str(exc)) from None
        except Exception as exc:
            log.exception("resume workspace revision failed: %s", exc)
            raise HTTPException(500, f"简历修改失败: {exc}") from None
        return RedirectResponse(f"/jobs/{job_id}/apply-pack", status_code=303)

    @app.post("/api/jobs/{job_id}/resume-workspace/rerender")
    def rerender_resume_workspace(job_id: int) -> RedirectResponse:
        try:
            _resume_workflow().rerender(job_id)
        except HTTPException:
            raise
        except ResumeWorkspaceError as exc:
            raise HTTPException(409, str(exc)) from None
        except Exception as exc:
            log.exception("resume workspace rerender failed: %s", exc)
            raise HTTPException(500, f"简历重新排版失败: {exc}") from None
        return RedirectResponse(f"/jobs/{job_id}/apply-pack", status_code=303)
    _grounding_labels = {
        "job_description": "冻结 JD",
        "submitted_resume": "实际提交简历",
        "project_vault": "项目事实",
        "preparation_note": "需准备能力",
    }

    def _interview_material_view(
        subject: InterviewResearchSubject,
    ) -> dict[str, Any]:
        # Keep the last complete result visible while a refreshed result is built.
        # Publication is atomic, so staleness is a status banner rather than a
        # reason to replace useful Q&A with an empty page.
        published = subject.current_material
        if published is None:
            return {
                "status": None,
                "answers": [],
                "sources": [],
                "result_revision": 0,
                "updated_at": None,
            }

        assessments = {
            str(item.get("evidence_id")): item.get("assessment")
            for item in subject.source_assessments
            if item.get("is_current")
            and isinstance(item.get("assessment"), dict)
        }
        evidence_ids: list[str] = []
        for answer in published.answer_set.answers:
            for citation in answer.source_citations:
                if citation.evidence_id not in evidence_ids:
                    evidence_ids.append(citation.evidence_id)

        sources: dict[str, dict[str, Any]] = {}
        for evidence_id in evidence_ids:
            assessment = assessments.get(evidence_id) or {}
            try:
                evidence = interview_source_store.get(int(evidence_id))
            except (ValueError, EvidenceNotFoundError):
                sources[evidence_id] = {
                    "evidence_id": evidence_id,
                    "title": "来源记录目前不可读取",
                    "saved_url": None,
                    "external_url": None,
                    "rationale": str(assessment.get("rationale") or ""),
                }
                continue
            sources[evidence_id] = {
                "evidence_id": evidence_id,
                "title": evidence.title or evidence.final_url,
                "saved_url": (
                    f"/interview-research/{subject.application_id}/workspaces/"
                    f"{subject.submitted_workspace_id}/sources/{evidence_id}"
                ),
                "external_url": (
                    evidence.final_url
                    if evidence.final_url.startswith(("http://", "https://"))
                    else None
                ),
                "rationale": str(assessment.get("rationale") or ""),
            }

        answers: list[dict[str, Any]] = []
        for index, answer in enumerate(published.answer_set.answers, start=1):
            citations = []
            for citation in answer.source_citations:
                source = sources.get(citation.evidence_id)
                citations.append({
                    "label": source["title"] if source else "已引用面经",
                    "saved_url": source.get("saved_url") if source else None,
                    "external_url": source.get("external_url") if source else None,
                    "quote": citation.quote,
                })
            grounding = [
                {
                    "label": _grounding_labels.get(item.kind, item.kind),
                    "quote": item.quote,
                }
                for item in answer.grounding
            ]
            answers.append({
                "number": index,
                "question": answer.question,
                "answer": answer.answer,
                "citations": citations,
                "grounding": grounding,
            })

        return {
            "status": published.answer_set.status,
            "answers": answers,
            "sources": [sources[evidence_id] for evidence_id in evidence_ids],
            "result_revision": published.result_revision,
            "updated_at": _julian_to_human(published.created_at),
        }

    @app.get(
        "/interview-research/{application_id}/workspaces/"
        "{submitted_workspace_id}/sources/{evidence_id}",
        response_class=HTMLResponse,
    )
    def interview_research_source_view(
        application_id: int,
        submitted_workspace_id: int,
        evidence_id: int,
        request: Request,
    ) -> Any:
        source_subject_id = (
            f"application:{application_id}:workspace:{submitted_workspace_id}"
        )
        with store.connect() as conn:
            linked = conn.execute(
                "SELECT 1 FROM source_subject_evidence "
                "WHERE subject_kind = 'interview_research' AND subject_id = ? "
                "AND evidence_id = ? LIMIT 1",
                (source_subject_id, evidence_id),
            ).fetchone()
            if linked is None:
                linked = conn.execute(
                    "SELECT 1 FROM interview_research_source_links "
                    "WHERE workspace_id = ? AND evidence_id = ? LIMIT 1",
                    (submitted_workspace_id, str(evidence_id)),
                ).fetchone()
            target = conn.execute(
                "SELECT 1 FROM interview_research_contexts "
                "WHERE workspace_id = ? AND application_id = ?",
                (submitted_workspace_id, application_id),
            ).fetchone()
        if linked is None or target is None:
            raise HTTPException(404, "interview source not found")
        try:
            evidence = interview_source_store.get(evidence_id)
        except EvidenceNotFoundError:
            raise HTTPException(404, "interview source not found") from None
        external_url = (
            evidence.final_url
            if evidence.final_url.startswith(("http://", "https://"))
            else None
        )
        return templates.TemplateResponse(
            request,
            "interview_source.html",
            _ctx(
                request,
                evidence=evidence,
                external_url=external_url,
                provenance_label={
                    "web": "公开网页正文",
                    "user_provided": "用户提供",
                }.get(evidence.provenance, "保存的面经原文"),
                fetched_at=_format_julian_time(evidence.fetched_at),
                active_tab="recommended",
            ),
        )

    @app.get("/jobs/{job_id}/post-apply-pack", response_class=HTMLResponse)
    def post_apply_pack_view(job_id: int, request: Request) -> Any:
        """Show the sole current research result for the frozen submission."""
        application_id = _application_for_job(job_id, create=False)
        if application_id is None:
            raise HTTPException(409, "这个岗位还没有真实投递记录")
        actual_job_id, workspace = _submitted_interview_target(application_id)
        if actual_job_id != job_id:
            raise HTTPException(409, "投递记录与岗位不匹配")
        subject = interview_repository.ensure_subject(application_id, workspace.id)

        invocation = None
        if research_agents is not None:
            invocation = _maybe_enqueue_interview_research(
                application_id=application_id,
                workspace_id=workspace.id,
                trigger_reason="post-application page opened",
                force=False,
            )
            invocation = research_agents.latest_interview_invocation(
                application_id=application_id,
                workspace_id=workspace.id,
            ) or invocation

        submitted = workspace.job_snapshot
        job = {
            "id": job_id,
            "title": str(submitted.get("title") or ""),
            "company": str(submitted.get("company") or ""),
            "location": str(submitted.get("location") or ""),
            "url": str(submitted.get("url") or ""),
            "raw_text": str(submitted.get("raw_text") or ""),
            "source": str(submitted.get("source") or ""),
        }
        return templates.TemplateResponse(
            request,
            "post_apply_pack.html",
            _ctx(
                request,
                job=job,
                application_id=application_id,
                submitted_workspace=workspace,
                submitted_at_label=_format_julian_time(workspace.submitted_at),
                subject=subject,
                material=_interview_material_view(subject),
                material_is_stale=subject.current_material_is_stale,
                agent_run=_invocation_view(invocation),
                agent_ready=research_agents is not None,
                active_tab="recommended",
            ),
        )

    @app.post("/interview-research/{application_id}/refresh")
    def refresh_interview_research(application_id: int) -> RedirectResponse:
        if research_agents is None:
            raise HTTPException(409, "面经研究 Agent 尚未配置")
        job_id, workspace = _submitted_interview_target(application_id)
        _maybe_enqueue_interview_research(
            application_id=application_id,
            workspace_id=workspace.id,
            trigger_reason="user requested interview research refresh",
            force=True,
        )
        return RedirectResponse(
            f"/jobs/{job_id}/post-apply-pack", status_code=303
        )

    @app.post("/interview-research/{application_id}/sources")
    def add_interview_research_source(
        application_id: int,
        text: str = Form(...),
        title: str = Form(""),
        source_url: str = Form(""),
    ) -> RedirectResponse:
        if research_agents is None:
            raise HTTPException(409, "面经研究 Agent 尚未配置")
        body = text.strip()
        if not body:
            raise HTTPException(400, "粘贴的面经正文不能为空")
        job_id, workspace = _submitted_interview_target(application_id)
        try:
            research_agents.add_interview_source(
                application_id=application_id,
                workspace_id=workspace.id,
                text=body,
                title=title.strip() or "用户提供的面经",
                source_url=source_url.strip() or None,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except InterviewResearchConflictError as exc:
            raise HTTPException(409, str(exc)) from None
        _maybe_enqueue_interview_research(
            application_id=application_id,
            workspace_id=workspace.id,
            trigger_reason="user provided interview source",
            force=True,
        )
        return RedirectResponse(
            f"/jobs/{job_id}/post-apply-pack", status_code=303
        )

    @app.get(
        "/api/research-agent-invocations/{invocation_id}",
        response_class=JSONResponse,
    )
    def research_agent_invocation_status(invocation_id: str) -> Any:
        """Return one durable Agent status for stable, non-refreshing UI polling."""
        with store.connect() as conn:
            row = conn.execute(
                "SELECT id, agent_name, subject_kind, subject_id, status, message, "
                "error_text, unresolved_json, updated_at "
                "FROM research_agent_invocations WHERE id = ?",
                (invocation_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, "Agent 运行记录不存在")
        status = str(row[4])
        labels = {
            "queued": "等待 Agent 开始",
            "running": "Agent 正在研究",
            "published": "已更新当前结果",
            "unchanged": "当前结果无需更新",
            "blocked": "暂时缺少足够证据",
            "stale": "上下文已变化，本次结果未发布",
            "failed": "Agent 运行失败",
        }
        unresolved: list[str] = []
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            parsed = json.loads(str(row[7] or "[]"))
            if isinstance(parsed, list):
                unresolved = [str(value) for value in parsed if str(value).strip()]

        result_url = _research_invocation_result_url(str(row[2]), str(row[3]))

        payload = {
            "id": str(row[0]),
            "agent_name": str(row[1]),
            "subject_kind": str(row[2]),
            "status": status,
            "status_label": labels.get(status, status),
            "message": str(row[5] or ""),
            "error": str(row[6]) if row[6] else None,
            "unresolved": unresolved,
            "updated_at": _format_julian_time(float(row[8])),
            "terminal": status not in {"queued", "running"},
            "result_changed": status == "published",
            "result_url": result_url,
        }
        return JSONResponse(
            payload,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/recommended", response_class=HTMLResponse)
    def recommended_view(request: Request) -> Any:
        """Render only the current selection published by JobDiscoveryAgent."""
        repository = (
            getattr(research_agents, "job_repository", None)
            or JobDiscoveryRepository(store)
        )
        search_context = repository.get_search_context()
        current = repository.get_current_selection()
        jobs: list[dict[str, Any]] = []
        with store.connect() as conn:
            workspace_job_ids = {
                int(row[0])
                for row in conn.execute(
                    "SELECT DISTINCT a.job_id FROM resume_workspaces AS rw "
                    "JOIN applications AS a ON a.id = rw.application_id"
                ).fetchall()
            }
        if current is not None:
            for item in current.items:
                evidence = item.job_evidence
                if evidence is None or evidence.job_id is None:
                    continue
                live_evidence = repository.get_job_evidence(item.job_evidence_id)
                live_status = (
                    live_evidence.source_status if live_evidence is not None else "unknown"
                )
                live_checked_at = (
                    live_evidence.checked_at if live_evidence is not None else evidence.checked_at
                )
                live_last_seen_at = (
                    live_evidence.last_seen_at
                    if live_evidence is not None
                    else evidence.last_seen_at
                )
                has_workspace = evidence.job_id in workspace_job_ids
                jobs.append({
                    "job_id": evidence.job_id,
                    "job_evidence_id": item.job_evidence_id,
                    "title": evidence.title,
                    "company": evidence.company,
                    "location": evidence.location or "",
                    "url": evidence.canonical_url,
                    "source": evidence.source_name,
                    "recruitment_type": evidence.recruitment_type,
                    "page_time_information": evidence.page_time_information,
                    "checked_at": _format_julian_time(live_checked_at),
                    "last_seen_at": _format_julian_time(live_last_seen_at),
                    "source_status": live_status,
                    "is_closed": live_status == "closed",
                    "has_workspace": has_workspace,
                    "open_status": {
                        "open": "上次检查时可见",
                        "closed": "上次检查显示已关闭",
                        "unknown": "上次检查未能确认开放状态",
                    }.get(live_status, "上次检查未能确认开放状态"),
                    "raw_text": evidence.jd_text,
                    "why_worth_attention": item.why_worth_attention,
                    "concerns": item.concerns,
                    "unknowns": item.unknowns,
                    "grounding_quotes": [
                        quote.model_dump(mode="json")
                        for quote in getattr(item, "grounding_quotes", [])
                    ],
                    "apply_pack_url": (
                        f"/jobs/{evidence.job_id}/apply-pack"
                        if has_workspace
                        else (
                            None
                            if live_status == "closed"
                            else (
                                f"/jobs/{evidence.job_id}/apply-pack"
                                f"?selection_revision={current.result_revision}"
                                f"&job_evidence_id={item.job_evidence_id}"
                            )
                        )
                    ),
                })
        invocation = (
            research_agents.latest_job_invocation()
            if research_agents is not None
            else None
        )
        return templates.TemplateResponse(
            request,
            "recommended.html",
            _ctx(
                request,
                search_context=(
                    {
                        "revision": search_context.revision,
                        "intent_text": search_context.intent,
                        "hard_constraints": search_context.hard_constraints,
                        "feedback": search_context.feedback,
                    }
                    if search_context is not None
                    else None
                ),
                selection=(
                    {
                        "context_revision": current.context_revision,
                        "result_revision": current.result_revision,
                        "summary": current.coverage_summary,
                        "evidence_gaps": current.evidence_gaps,
                        "published_at": _format_julian_time(current.published_at),
                        "jobs": jobs,
                    }
                    if current is not None
                    else None
                ),
                selection_is_stale=(
                    current is not None
                    and search_context is not None
                    and current.context_revision != search_context.revision
                ),
                agent_run=_invocation_view(invocation),
                agent_ready=research_agents is not None,
                active_tab="recommended",
            ),
        )

    @app.post("/job-search/context")
    def update_job_search_context(
        intent_text: str = Form(...),
        context_revision: int = Form(0),
        hard_constraints_text: str = Form(""),
    ) -> Any:
        if research_agents is None:
            raise HTTPException(503, "找岗 Agent 尚未配置")
        intent = intent_text.strip()
        if not intent:
            raise HTTPException(400, "当前找岗意图不能为空")
        current_context = research_agents.job_repository.get_search_context()
        expected_revision = context_revision or None
        if current_context is not None and expected_revision is None:
            raise HTTPException(409, "找岗意图已经存在，请刷新页面后再保存")
        hard_constraints = [
            line.strip()
            for line in hard_constraints_text.splitlines()
            if line.strip()
        ]
        try:
            research_agents.replace_job_search_context(
                intent,
                hard_constraints=hard_constraints,
                expected_revision=expected_revision,
            )
        except JobDiscoveryRevisionConflict as exc:
            raise HTTPException(409, f"找岗意图已被更新，请刷新页面：{exc}") from None
        research_agents.enqueue_job_discovery(
            trigger_reason="user saved the current job-search intent"
        )
        return RedirectResponse("/recommended", status_code=303)

    @app.post("/job-search/run")
    def run_job_search_agent() -> Any:
        if research_agents is None:
            raise HTTPException(503, "找岗 Agent 尚未配置")
        research_agents.enqueue_job_discovery(trigger_reason="user requested refresh")
        return RedirectResponse("/recommended", status_code=303)

    @app.post("/job-search/jobs/{job_id}/dismiss")
    def dismiss_recommended_job(
        job_id: int,
        selection_revision: int = Form(...),
        job_evidence_id: int = Form(...),
        context_revision: int = Form(...),
    ) -> Any:
        if research_agents is None:
            raise HTTPException(503, "找岗 Agent 尚未配置")
        current = research_agents.job_repository.get_selection_revision(
            selection_revision
        )
        evidence = next(
            (
                item.job_evidence
                for item in (current.items if current is not None else ())
                if (
                    item.job_evidence is not None
                    and item.job_evidence.job_id == job_id
                    and item.job_evidence_id == job_evidence_id
                )
            ),
            None,
        )
        if evidence is None:
            raise HTTPException(404, f"job#{job_id} 不在当前岗位证据中")
        try:
            research_agents.append_job_feedback(
                f"暂不考虑具体岗位：{evidence.company} · {evidence.title}（job#{job_id}）",
                expected_revision=context_revision,
            )
        except JobDiscoveryRevisionConflict as exc:
            raise HTTPException(409, f"找岗意图已被更新，请刷新页面：{exc}") from None
        research_agents.enqueue_job_discovery(
            trigger_reason="user dismissed one specific job"
        )
        return RedirectResponse("/recommended", status_code=303)

    @app.post("/job-search/jobs/{job_id}/report-closed")
    def report_recommended_job_closed(
        job_id: int,
        selection_revision: int = Form(...),
        job_evidence_id: int = Form(...),
    ) -> Any:
        repository = (
            getattr(research_agents, "job_repository", None)
            or JobDiscoveryRepository(store)
        )
        _published_job_snapshot(
            job_id,
            selection_revision=selection_revision,
            job_evidence_id=job_evidence_id,
        )
        updated = repository.record_user_closed_report(job_evidence_id)
        if updated is None or updated.job_id != job_id:
            raise HTTPException(409, "岗位证据已变化，请刷新页面后重试")
        return RedirectResponse("/recommended", status_code=303)

    @app.post("/job-search/feedback/{feedback_index}")
    def update_job_search_feedback(
        feedback_index: int,
        feedback: str = Form(...),
        context_revision: int = Form(...),
    ) -> Any:
        if research_agents is None:
            raise HTTPException(503, "找岗 Agent 尚未配置")
        try:
            research_agents.update_job_feedback(
                feedback_index,
                feedback,
                expected_revision=context_revision,
            )
        except JobDiscoveryRevisionConflict as exc:
            raise HTTPException(409, f"找岗意图已被更新，请刷新页面：{exc}") from None
        except (IndexError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from None
        research_agents.enqueue_job_discovery(
            trigger_reason="user edited visible job-search feedback"
        )
        return RedirectResponse("/recommended", status_code=303)

    @app.post("/job-search/feedback/{feedback_index}/remove")
    def remove_job_search_feedback(
        feedback_index: int,
        context_revision: int = Form(...),
    ) -> Any:
        if research_agents is None:
            raise HTTPException(503, "找岗 Agent 尚未配置")
        try:
            research_agents.remove_job_feedback(
                feedback_index,
                expected_revision=context_revision,
            )
        except JobDiscoveryRevisionConflict as exc:
            raise HTTPException(409, f"找岗意图已被更新，请刷新页面：{exc}") from None
        except IndexError as exc:
            raise HTTPException(400, str(exc)) from None
        research_agents.enqueue_job_discovery(
            trigger_reason="user removed visible job-search feedback"
        )
        return RedirectResponse("/recommended", status_code=303)
    @app.get("/metrics", response_class=HTMLResponse)
    def metrics_view(request: Request) -> Any:
        """W15.19 — Dogfood metrics dashboard.

        本周/累计 真实使用数据可视化 — 给"录视频 + 写专栏"用. 用户简历项目
        最值钱那 1 行 ("我用它评估了 N 个 JD, 拿了 K 个面试") 就靠这页.

        信号源:
        - jobs.created_at → 评估的岗位数
        - applications.status → 投递 / 面试 / offer 数
        - skill_runs.cost_usd → 累计 cost
        - harness_runs → agent wake 次数 + 成本
        - inbox_items → agent 推荐数 / 接受率
        """
        with store.connect() as conn:
            # Job evaluation (this week + all)
            jobs_week = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE created_at >= julianday('now') - 7"
            ).fetchone()[0]
            jobs_all = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

            # Applications by status
            apps_by_status = dict(conn.execute(
                "SELECT status, COUNT(*) FROM applications GROUP BY status"
            ).fetchall())

            apps_total = sum(apps_by_status.values())
            apps_applied = apps_by_status.get("applied", 0)
            apps_interview = sum(apps_by_status.get(s, 0) for s in (
                "1st_interview", "2nd_interview", "final_interview", "screening",
            ))
            apps_offer = apps_by_status.get("offer", 0)
            apps_rejected = apps_by_status.get("rejected", 0)

            # SKILL costs
            skill_cost_week = float(conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM skill_runs "
                "WHERE created_at >= julianday('now') - 7"
            ).fetchone()[0] or 0)
            skill_cost_all = float(conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM skill_runs"
            ).fetchone()[0] or 0)
            skill_runs_week = conn.execute(
                "SELECT COUNT(*) FROM skill_runs WHERE created_at >= julianday('now') - 7"
            ).fetchone()[0]

            # Top SKILLs by call count
            top_skills = conn.execute(
                "SELECT skill_name, COUNT(*), COALESCE(SUM(cost_usd), 0) "
                "FROM skill_runs WHERE created_at >= julianday('now') - 30 "
                "GROUP BY skill_name ORDER BY COUNT(*) DESC LIMIT 8"
            ).fetchall()

            # Harness runs (agent wake) — this week
            try:
                harness_week = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) "
                    "FROM harness_runs WHERE started_at >= julianday('now') - 7"
                ).fetchone()
                harness_runs_week = int(harness_week[0] or 0)
                harness_cost_week = float(harness_week[1] or 0)
            except Exception:
                harness_runs_week = 0
                harness_cost_week = 0.0

            # Inbox suggestions accept/reject rate
            inbox_total = conn.execute(
                "SELECT COUNT(*) FROM inbox_items WHERE kind = 'agent_suggestion'"
            ).fetchone()[0]
            inbox_approved = conn.execute(
                "SELECT COUNT(*) FROM inbox_items "
                "WHERE kind = 'agent_suggestion' AND status = 'approved'"
            ).fetchone()[0]
            inbox_rejected = conn.execute(
                "SELECT COUNT(*) FROM inbox_items "
                "WHERE kind = 'agent_suggestion' AND status = 'rejected'"
            ).fetchone()[0]

        # Compute reply rate (applied → any-event)
        reply_rate = None
        if apps_applied > 0:
            replies = (
                apps_by_status.get("hr_replied", 0)
                + apps_by_status.get("screening", 0)
                + apps_interview + apps_offer
            )
            reply_rate = replies / apps_applied

        accept_rate = None
        if inbox_approved + inbox_rejected > 0:
            accept_rate = inbox_approved / (inbox_approved + inbox_rejected)

        return templates.TemplateResponse(
            request,
            "metrics.html",
            _ctx(
                request,
                jobs_week=jobs_week,
                jobs_all=jobs_all,
                apps_by_status=apps_by_status,
                apps_total=apps_total,
                apps_applied=apps_applied,
                apps_interview=apps_interview,
                apps_offer=apps_offer,
                apps_rejected=apps_rejected,
                reply_rate=reply_rate,
                skill_cost_week=skill_cost_week,
                skill_cost_all=skill_cost_all,
                skill_runs_week=skill_runs_week,
                top_skills=[
                    {"name": r[0], "count": int(r[1]), "cost_usd": float(r[2])}
                    for r in top_skills
                ],
                harness_runs_week=harness_runs_week,
                harness_cost_week=harness_cost_week,
                inbox_total=inbox_total,
                inbox_approved=inbox_approved,
                inbox_rejected=inbox_rejected,
                accept_rate=accept_rate,
                runtime_ready=runtime is not None and bool(settings.deepseek_api_key),
                active_tab="metrics",
            ),
        )

    @app.get("/debug", response_class=HTMLResponse)
    def debug_view(request: Request) -> Any:
        """W15.9 — Mission Control + harness telemetry debug view.

        Demoted from home (W14.13) to its own page. For developers /
        the user when they want to see "is the agent actually running".
        """
        # Ensure harness tables exist (no-op if already created) so a
        # fresh store doesn't throw OperationalError when /debug is hit
        # before any harness wake.
        from ..agent_runtime import _schema as _hs
        _hs.init_agent_runtime_schema(store)

        with store.connect() as conn:
            recent_harness = conn.execute(
                "SELECT id, trigger_kind, trigger_detail, started_at, "
                "       ended_at, iterations, status, final_text, "
                "       tool_calls_json, cost_usd, error_text "
                "FROM harness_runs ORDER BY started_at DESC LIMIT 30"
            ).fetchall()
            events = conn.execute(
                "SELECT id, kind, job_id, note, source, created_at "
                "FROM harness_events ORDER BY id DESC LIMIT 30"
            ).fetchall()
        harness_view = []
        for r in recent_harness:
            tcs: list[Any] = []
            with contextlib.suppress(Exception):
                raw_tool_calls = json.loads(r[8] or "[]")
                if isinstance(raw_tool_calls, list):
                    tcs = raw_tool_calls
                elif isinstance(raw_tool_calls, dict):
                    calls = raw_tool_calls.get("calls")
                    if isinstance(calls, list):
                        tcs = calls
            harness_view.append({
                "id": int(r[0]), "trigger_kind": r[1],
                "trigger_detail": (r[2] or "")[:200],
                "started_at": r[3], "ended_at": r[4],
                "iterations": int(r[5] or 0), "status": r[6],
                "final_text": (r[7] or "")[:300],
                "tool_calls": tcs[:8],
                "cost_usd": float(r[9] or 0.0),
                "error_text": r[10],
            })
        events_view = [
            {"id": int(r[0]), "kind": r[1], "job_id": r[2],
             "note": (r[3] or "")[:120], "source": r[4], "created_at": r[5]}
            for r in events
        ]
        return templates.TemplateResponse(
            request,
            "debug.html",
            _ctx(
                request,
                harness_runs=harness_view,
                harness_events=events_view,
                runtime_ready=runtime is not None and bool(settings.deepseek_api_key),
                active_tab="debug",
            ),
        )

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

        view = pv_mod.build(store, pending_job_ids=_pipeline_pending_job_ids())
        pipeline_apply_urls = _application_entry_urls(
            {
                card.job_id
                for cards in view.columns.values()
                for card in cards
                if card.status in {"scanned", "considered"}
            }
        )
        application_ids = {
            card.application_id
            for cards in view.columns.values()
            for card in cards
            if card.application_id is not None
        }
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
                pipeline_apply_urls=pipeline_apply_urls,
                pipeline_application_resources=_application_resources(application_ids),
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
            "viewed", "replied", "assessment",
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
        view = pv_mod.build(store, pending_job_ids=_pipeline_pending_job_ids())
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
                pipeline_apply_urls=_application_entry_urls({card.job_id}),
                pipeline_application_resources=_application_resources({app_id}),
            ),
        )

    @app.get("/inbox", response_class=HTMLResponse)
    def inbox_view(request: Request) -> Any:
        from fastapi.responses import RedirectResponse as _R
        return _R(url="/", status_code=301)  # W21 redesign: merged into Mission Control
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
                application_resources=_application_resources(
                    {int(row["id"]) for row in rows}
                ),
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
            "viewed", "replied", "assessment",
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
            _ctx(
                request,
                app=rows[0],
                application_resources=_application_resources({app_id}),
            ),
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

    @app.get("/project-vault", response_class=HTMLResponse)
    def project_vault_view(request: Request) -> Any:
        from .. import project_vault
        return templates.TemplateResponse(
            request,
            "project_vault.html",
            _ctx(
                request,
                projects=project_vault.list_all(store, limit=80),
                recommended_directions=project_vault.RECOMMENDED_DIRECTIONS,
                contribution_labels=project_vault.CONTRIBUTION_LABELS,
                active_tab="project_vault",
            ),
        )

    @app.post("/api/project-vault/insert", response_class=HTMLResponse)
    def project_vault_insert(
        request: Request,
        title: str = Form(...),
        mainstream_direction: str = Form(...),
        project_task: str = Form(...),
        my_work: str = Form(...),
        typical_problem: str = Form(""),
        method_route: str = Form(""),
        market_context: str = Form(""),
        reference_sources: str = Form(""),
        contribution_type: str = Form("main_contribution"),
        contribution_detail: str = Form(""),
        key_difficulties: str = Form(""),
        resolution_process: str = Form(""),
        project_outputs: str = Form(""),
        evidence: str = Form(""),
        askable_points: str = Form(""),
        expression_boundary: str = Form(""),
        do_not_claim: str = Form(""),
        tags: str = Form(""),
        confidence: str = Form("0.5"),
    ) -> Any:
        from .. import project_vault
        try:
            conf = max(0.0, min(1.0, float(confidence or "0.5")))
        except ValueError:
            conf = 0.5
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        try:
            record = project_vault.insert(
                store,
                title=title,
                mainstream_direction=mainstream_direction,
                typical_problem=typical_problem or None,
                project_task=project_task,
                my_work=my_work,
                method_route=method_route or None,
                market_context=market_context or None,
                reference_sources=reference_sources or None,
                contribution_type=contribution_type,
                contribution_detail=contribution_detail or None,
                key_difficulties=key_difficulties or None,
                resolution_process=resolution_process or None,
                project_outputs=project_outputs or None,
                evidence=evidence or None,
                askable_points=askable_points or None,
                expression_boundary=expression_boundary or None,
                do_not_claim=do_not_claim or None,
                tags=tag_list,
                confidence=conf,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

        return templates.TemplateResponse(
            request,
            "_project_list.html",
            _ctx(
                request,
                projects=project_vault.list_all(store, limit=80),
                just_added=record.id,
            ),
        )

    @app.post("/api/project-vault/draft-market-context", response_class=HTMLResponse)
    def project_vault_draft_market_context(
        request: Request,
        title: str = Form(""),
        mainstream_direction: str = Form(""),
        project_task: str = Form(""),
        my_work: str = Form(""),
    ) -> Any:
        from .. import project_vault
        from ..agentic.search import build_default_search

        llm = getattr(runtime, "_llm", None) if runtime is not None else None
        search = None
        try:
            search = build_default_search()
            draft = project_vault.draft_market_context(
                title=title,
                mainstream_direction=mainstream_direction,
                project_task=project_task,
                my_work=my_work,
                search=search,
                llm=llm,
            )
        except ValueError as e:
            draft = project_vault.MarketContextDraft(
                market_context="",
                reference_sources="",
                warnings=[str(e)],
            )
        except Exception as e:
            draft = project_vault.MarketContextDraft(
                market_context="",
                reference_sources="",
                warnings=[f"生成失败: {e}"],
            )
        finally:
            close = getattr(search, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

        return templates.TemplateResponse(
            request,
            "_project_market_context.html",
            _ctx(request, market_draft=draft),
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

    @app.get("/agent", response_class=HTMLResponse)
    def agent_page(request: Request) -> Any:
        from fastapi.responses import RedirectResponse as _R
        return _R(url="/", status_code=301)  # W21 redesign: merged into Mission Control
        """W13 agent loop UI — pick a goal, watch the agent think + act live."""
        # Recent runs to show below the form (audit trail)
        try:
            with store.connect() as conn:
                rows = conn.execute(
                    "SELECT id, trigger_kind, trigger_detail, status, iterations, "
                    "       started_at, ended_at "
                    "FROM harness_runs ORDER BY started_at DESC LIMIT 8"
                ).fetchall()
        except Exception:
            rows = []
        recent_runs = []
        for r in rows:
            latency_ms: int | None = None
            if r[6] is not None and r[5] is not None:
                latency_ms = int((float(r[6]) - float(r[5])) * 86400 * 1000)
            recent_runs.append({
                "id": r[0], "trigger_kind": r[1],
                "goal": _extract_trigger_goal(r[2]),
                "status": r[3], "iterations": r[4],
                "critic_score": None, "latency_ms": latency_ms,
            })
        first_use = False
        try:
            with store.connect() as conn:
                n_jobs = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE length(raw_text) >= 200"
                ).fetchone()[0]
                n_goals = conn.execute(
                    "SELECT COUNT(*) FROM user_goals WHERE status='active'"
                ).fetchone()[0]
            first_use = not recent_runs and n_jobs == 0 and n_goals == 0
        except Exception:
            first_use = False
        return templates.TemplateResponse(
            request,
            "agent.html",
            _ctx(
                request,
                recent_runs=recent_runs,
                recent_artifacts=_recent_agent_artifacts(store, limit=6),
                first_use=first_use,
                skill_count=len(skills),
                tools_ready=runtime is not None and bool(settings.deepseek_api_key),
                active_tab="agent",
            ),
        )

    @app.get("/api/agent/stream")
    async def agent_stream(
        request: Request,
        goal: str,
        trigger_kind: str = "user_input",
        max_iterations: int = 6,
    ) -> StreamingResponse:
        """SSE endpoint that runs the agent runtime in a thread + streams events.

        Each event becomes one ``data: {...}\\n\\n`` SSE frame. The browser-
        side EventSource (in agent.html) appends each frame to the live
        panel as it arrives. The connection closes after the loop returns.

        Implementation note: agent_runtime.run is sync (each LLM call blocks), so
        we run it in a thread via asyncio.to_thread + a thread-safe queue
        back to the async generator.
        """
        if not settings.deepseek_api_key:
            async def _err_stream():
                yield (
                    "data: " + json_dumps({
                        "kind": "error",
                        "message": "OFFERGUIDE_LLM_API_KEY 没配 — agent 无法启动",
                    }) + "\n\n"
                )
            return StreamingResponse(_err_stream(), media_type="text/event-stream")

        from ..agent_runtime import (
            AgentRuntimeDeps,
            MemoryStore,
            TriggerEvent,
            default_worldview_dir,
        )
        from ..agent_runtime import _schema as _hs
        from ..agent_runtime import run as agent_runtime_run
        _hs.init_agent_runtime_schema(store)

        # Build a fresh LLMClient per request (cheap; httpx.Client lifecycle)
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )

        from ..agentic.search import build_default_search
        try:
            _search = build_default_search()
        except Exception:
            _search = None
        deps = AgentRuntimeDeps(
            settings=settings, store=store,
            memory_store=MemoryStore(root=default_worldview_dir(settings)),
            llm=llm, runtime=runtime, skills=skills,
            search=_search, notifier=notifier,
            user_profile_text=_effective_master_text(),
            research_agents=research_agents,
        )
        trigger = TriggerEvent(
            kind=trigger_kind,
            detail={"message": goal} if trigger_kind == "user_input" else {"reason": goal},
        )
        capped_max_iter = max(1, min(int(max_iterations), 12))

        main_loop = asyncio.get_running_loop()
        # W14.9: bounded queue + dropped-event counter. An 8-iteration agent
        # produces ~50-100 events; we cap at 500 so a slow client + full queue
        # can never grow without bound. When full, drop the oldest event and
        # surface the count in a synthetic _dropped event so the UI knows.
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        dropped_count = 0
        # W14.9: real cooperative cancellation. asyncio.Task.cancel() cannot
        # interrupt code running in a thread (asyncio.to_thread submits to a
        # ThreadPoolExecutor and the future.cancel() only works pre-start).
        # The agent loop checks this Event at every iteration boundary and
        # exits gracefully when set. SSE finally block sets it on disconnect.
        import threading as _threading
        cancel_event = _threading.Event()

        def _on_event_from_thread(ev: Any) -> None:
            # agent_runtime.run calls this from its worker thread — bridge to async queue.
            # call_soon_threadsafe runs the put on the main event loop so the
            # asyncio.Queue mutation stays on its owning loop (thread-safe).
            def _do_put(payload: dict) -> None:
                nonlocal dropped_count
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    # Drop oldest to make room for newest (so the user always
                    # sees the latest activity, not stale events). Skip when
                    # the discarded event was a `_done` sentinel — those carry
                    # the run summary and should never silently disappear.
                    try:
                        old = queue.get_nowait()
                        if isinstance(old, dict) and old.get("kind") == "_done":
                            # Put it back; drop the new one instead.
                            queue.put_nowait(old)
                            dropped_count += 1
                            return
                    except asyncio.QueueEmpty:
                        pass
                    dropped_count += 1
                    with contextlib.suppress(asyncio.QueueFull):
                        queue.put_nowait(payload)

            with contextlib.suppress(RuntimeError):
                # Event loop closed (client disconnected) — drop event
                main_loop.call_soon_threadsafe(_do_put, dict(ev))

        def _run_blocking() -> None:
            try:
                result = agent_runtime_run(
                    trigger=trigger,
                    deps=deps,
                    max_iterations=capped_max_iter,
                    on_event=_on_event_from_thread,
                    cancel_event=cancel_event,
                )
                _on_event_from_thread({
                    "kind": "_done",
                    "run_id": result.run_id,
                    "iterations": result.iterations,
                    "latency_ms": result.latency_ms,
                    "cost_usd": result.cost_usd,
                    "finish_reason": result.finish_reason,
                })
            except Exception as e:
                _on_event_from_thread({
                    "kind": "_done", "error": str(e),
                })
            finally:
                # Sentinel so the SSE generator knows to close
                main_loop.call_soon_threadsafe(queue.put_nowait, None)

        # Kick off the agent in a background thread.
        # W14.8: keep a reference to the task — without it Python may GC the
        # task and silently cancel mid-run (RUF006). The reference lives in
        # the closure of _sse_gen, which keeps it alive for the SSE lifetime.
        bg_task = asyncio.create_task(asyncio.to_thread(_run_blocking))

        async def _sse_gen():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=60.0)
                    except TimeoutError:
                        # Heartbeat to keep the connection open through silent stretches
                        yield ": keepalive\n\n"
                        continue
                    if ev is None:
                        break
                    yield "data: " + json_dumps(ev, ensure_ascii=False, default=str) + "\n\n"
                # W14.9: surface any events the bounded queue had to drop so
                # the user knows the trajectory shown is incomplete.
                if dropped_count > 0:
                    yield (
                        "data: " + json_dumps({
                            "kind": "_dropped",
                            "count": dropped_count,
                            "note": "queue full — some intermediate events skipped",
                        }) + "\n\n"
                    )
            finally:
                # W14.9: real cooperative cancellation. bg_task.cancel() was
                # a no-op against asyncio.to_thread (ThreadPoolExecutor
                # futures can't be cancelled mid-thread; previously this
                # silently let the agent keep burning LLM credits after
                # client disconnect). Now we set a Event the agent loop
                # checks at every iteration boundary.
                cancel_event.set()
                # Wait briefly for the bg task to notice + persist its
                # cancelled state. Cap so we never hang the response.
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(bg_task), timeout=5.0)
                with contextlib.suppress(Exception):
                    llm.close()

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
            _ctx(request, stages=stage_counts, companies=companies, active_tab="funnel"),
        )

    @app.get("/portfolio", response_class=HTMLResponse)
    def portfolio_view(request: Request) -> Any:
        """Public-friendly metrics page about the OfferGuide build itself.

        Privacy: NO user data (resume, names, application details, company
        names). Just aggregate numbers + agent-level metrics that are
        meaningful as evidence-of-build for a portfolio.
        """
        with store.connect() as conn:
            # Harness run metrics (was agent_runs pre-W21)
            try:
                n_agent_runs = conn.execute(
                    "SELECT COUNT(*) FROM harness_runs"
                ).fetchone()[0]
                total_cost = conn.execute(
                    "SELECT SUM(cost_usd) FROM harness_runs"
                ).fetchone()[0]
            except Exception:
                n_agent_runs = 0
                total_cost = 0.0
            n_skill_runs = conn.execute(
                "SELECT COUNT(*) FROM skill_runs"
            ).fetchone()[0]
            avg_critic = None  # W13 self-critique retired; no per-run score
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
            # Per-skill recent fitness (user thumbs is the real-signal channel now)
            skill_rows = conn.execute(
                "SELECT skill_name, COUNT(*) as n, AVG(signal_value) as avg_v "
                "FROM evolution_signals WHERE signal_kind='user_thumbs' "
                "GROUP BY skill_name HAVING n >= 3 "
                "ORDER BY n DESC LIMIT 12"
            ).fetchall()
            skill_metrics = [
                {"name": r[0], "n_signals": r[1], "avg_critic": r[2]}
                for r in skill_rows
            ]
            # Last 14 days harness run count for sparkline
            try:
                daily_rows = conn.execute(
                    "SELECT CAST((julianday('now') - started_at) AS INT) AS days_ago, "
                    "       COUNT(*) AS n "
                    "FROM harness_runs WHERE started_at >= julianday('now') - 14 "
                    "GROUP BY days_ago"
                ).fetchall()
            except Exception:
                daily_rows = []
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
                active_tab="portfolio",
            ),
        )

    @app.get("/goals", response_class=HTMLResponse)
    def goals_view(request: Request) -> Any:
        from fastapi.responses import RedirectResponse as _R
        return _R(url="/", status_code=301)  # W21 redesign: merged into Mission Control
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

    @app.get("/agent/runs/{run_id}", response_class=HTMLResponse)
    def agent_run_detail(request: Request, run_id: int) -> Any:
        """Read a persisted harness_runs row + render its trajectory.

        Also surfaces inbox suggestions this run created (reverse link)
        and any user_thumbs signals that fed back into evolution_signals.
        """
        # Make sure the harness schema exists — first visit to /agent/runs/{id}
        # on a fresh install can land here before any harness.run has fired.
        from ..agent_runtime import _schema as _hs
        _hs.init_agent_runtime_schema(store)
        with store.connect() as conn:
            row = conn.execute(
                "SELECT trigger_kind, trigger_detail, status, iterations, final_text, "
                "       tool_calls_json, started_at, ended_at, error_text, cost_usd "
                "FROM harness_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, f"harness_runs#{run_id} not found")
            # Which inbox suggestions came from this run?
            sug_rows = conn.execute(
                "SELECT id, title, status, decided_at, decision_note "
                "FROM inbox_items "
                "WHERE source_agent_run_id = ? "
                "ORDER BY created_at DESC",
                (run_id,),
            ).fetchall()
            # Which evolution_signals reference this run?
            sig_rows = conn.execute(
                "SELECT skill_name, skill_version, signal_kind, signal_value, "
                "       signal_weight, notes "
                "FROM evolution_signals "
                "WHERE notes LIKE ? "
                "ORDER BY created_at DESC LIMIT 20",
                (f"%harness_run#{run_id}%",),
            ).fetchall()
            artifact_rows = conn.execute(
                "SELECT id, kind, job_id, note, created_at "
                "FROM harness_events "
                "WHERE note LIKE ? "
                "AND kind IN ('project_record_saved', 'project_assessed') "
                "ORDER BY created_at DESC LIMIT 20",
                (f"%\"agent_run_id\": {run_id}%",),
            ).fetchall()

        # tool_calls_json: {"calls": ["iter1.foo(...)", ...], "sub_agent_cost_usd": ...}
        try:
            tool_calls_payload = json_loads(row[5] or "{}")
        except json.JSONDecodeError:
            tool_calls_payload = {}
        # Wrap each tool-call string as an event with payload so the template
        # (which expects ev.payload.*) keeps working. harness_runs only logs
        # one-line summaries (vs. AgentLoop's typed trajectory events) — we
        # surface them under a `summary` kind that the template renders as plain text.
        trajectory = [
            {"kind": "summary", "payload": {"text": str(c)}}
            for c in (tool_calls_payload.get("calls") or [])
        ]
        latency_ms: int | None = None
        if row[7] is not None and row[6] is not None:
            latency_ms = int((float(row[7]) - float(row[6])) * 86400 * 1000)

        suggestions = [
            {"id": r[0], "title": r[1], "status": r[2],
             "decided_at": r[3], "decision_note": r[4]}
            for r in sug_rows
        ]
        signals = [
            {"skill_name": r[0], "skill_version": r[1], "kind": r[2],
             "value": r[3], "weight": r[4], "notes": r[5]}
            for r in sig_rows
        ]
        artifacts = [
            artifact for artifact in (
                _artifact_from_event_row(r) for r in artifact_rows
            )
            if artifact is not None
        ]

        return templates.TemplateResponse(
            request,
            "agent_run_detail.html",
            _ctx(
                request,
                run_id=run_id,
                trigger_kind=row[0],
                goal=_extract_trigger_goal(row[1]),
                status=row[2],
                iterations=row[3], final_answer=row[4],
                trajectory=trajectory,
                critic_score=None, critic_notes=None,
                latency_ms=latency_ms,
                started_at=row[6], ended_at=row[7],
                error_text=row[8],
                cost_usd=row[9],
                suggestions=suggestions,
                signals=signals,
                artifacts=artifacts,
                active_tab="agent",
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

        try:
            events = ics_parser.parse_ics(ics_text)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        chosen = ics_parser.select_first_interview(events)
        if chosen is None:
            chosen = next(
                (
                    event
                    for event in events
                    if event.calendar_action == "cancelled"
                    and ae.has_calendar_uid(
                        store,
                        application_id=app_id,
                        uid=event.uid,
                    )
                ),
                None,
            )
        if chosen is None:
            raise HTTPException(
                400,
                "ICS file did not contain a recognizable interview event "
                "or a cancellation matching an existing interview UID.",
            )
        if not chosen.uid:
            raise HTTPException(400, "ICS interview event is missing its required UID.")
        calendar_action = chosen.calendar_action
        if calendar_action == "unsupported":
            raise HTTPException(
                400,
                f"ICS METHOD {chosen.method!r} does not publish, schedule, or cancel an interview.",
            )

        payload = {
            "summary": chosen.summary,
            "description": chosen.description,
            "round": chosen.round,
            "scheduled_at": (
                chosen.dtstart_utc.isoformat() if chosen.dtstart_utc else None
            ),
            "scheduled_at_local": (
                chosen.dtstart_local.isoformat() if chosen.dtstart_local else None
            ),
            "scheduled_date": (
                chosen.dtstart_date.isoformat() if chosen.dtstart_date else None
            ),
            "scheduled_tzid": chosen.dtstart_tzid,
            "ics_status": chosen.status,
            "ics_method": chosen.method,
            "ics_text": ics_text,
            "ics_event_count": len(events),
        }

        try:
            outcome = ae.record_calendar_event(
                store,
                application_id=app_id,
                uid=chosen.uid,
                sequence=chosen.sequence,
                action=calendar_action,
                payload=payload,
            )
        except Exception as e:
            raise HTTPException(400, f"failed to record: {e}") from None
        return {
            "ok": True,
            "application_id": app_id,
            "event_kind": outcome.event.kind,
            "calendar_action": calendar_action,
            "calendar_record_state": outcome.state,
            "recorded": outcome.created,
            "uid": chosen.uid,
            "sequence": chosen.sequence,
            "round": chosen.round,
            "scheduled_at": (
                chosen.dtstart_utc.isoformat() if chosen.dtstart_utc else None
            ),
            "scheduled_at_local": (
                chosen.dtstart_local.isoformat() if chosen.dtstart_local else None
            ),
            "scheduled_date": (
                chosen.dtstart_date.isoformat() if chosen.dtstart_date else None
            ),
            "scheduled_tzid": chosen.dtstart_tzid,
            "summary": chosen.summary,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_view(request: Request) -> Any:
        from fastapi.responses import RedirectResponse as _R
        return _R(url="/", status_code=301)  # W21 redesign: merged into Mission Control

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

        # W15.11 — feed user reaction into evolution_signals so GEPA learns
        # what kinds of suggestions the user accepts/rejects. This is the
        # learning loop that lets agent get more selective without if-else
        # rules ("don't push X" — agent learns by seeing X get rejected).
        try:
            from ..agent_runtime import feedback as _hfb
            skill_run_id = None
            if isinstance(item.payload, dict):
                _srid = item.payload.get("source_skill_run_id")
                if isinstance(_srid, int):
                    skill_run_id = _srid
            if decision == "approved":
                _hfb.on_inbox_accepted(
                    store, inbox_id=item_id, skill_run_id=skill_run_id,
                )
            elif decision == "rejected":
                _hfb.on_inbox_rejected(
                    store, inbox_id=item_id, skill_run_id=skill_run_id,
                )
            # 'dismissed' = soft no — record as ignored (weaker signal)
            elif decision == "dismissed":
                _hfb.on_inbox_ignored(
                    store, inbox_id=item_id, days_ignored=0,
                )
        except Exception as _e:
            # Feedback recording must never fail the user-facing response.
            log.warning("inbox feedback recording failed (non-fatal): %s", _e)

        if request.headers.get("hx-request"):
            return templates.TemplateResponse(
                request, "_inbox_list.html", _ctx(request, items=[item])
            )
        return RedirectResponse("/", status_code=303)

    @app.post("/inbox/{item_id}/answer", response_class=RedirectResponse)
    def answer_question(
        request: Request,
        item_id: int,
        option_id: str = Form(...),
        free_text: str | None = Form(None),
    ) -> Any:
        """W14.20 — user picked an option for a kind='question' item.
        Writes user_facts so agent's next wake sees the answer.

        W15.11 — also records into evolution_signals so GEPA learns which
        questions yielded useful answers (vs which were ignored / dismissed).
        """
        try:
            inbox_mod.answer_question(
                store, item_id, option_id=option_id, free_text=free_text,
            )
        except KeyError:
            raise HTTPException(404, f"inbox item {item_id} not found") from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

        try:
            from ..agent_runtime import feedback as _hfb
            _hfb.on_question_answered(
                store, inbox_id=item_id, option_id=option_id,
                free_text=free_text,
            )
        except Exception as _e:
            log.warning("question feedback recording failed (non-fatal): %s", _e)

        return RedirectResponse("/", status_code=303)

    @app.get("/api/search/test", response_class=JSONResponse)
    def search_test() -> dict:
        """Run a canary query against each search backend, return health.

        This diagnoses the same search backends available to the two research
        Agents. Firewalls can block DDG, Bing may serve a CAPTCHA, and Tavily
        depends on an API key; this endpoint exposes those real failures.
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
    def extension_get_package(company: str, job_id: int | None = None) -> JSONResponse:
        """Return the package attached to an exact current resume workspace."""
        company = (company or "").strip()
        if not company:
            return _ext_response(404, {"error": "company required"})
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT rw.id, rw.apply_pack_json, j.id, rw.status, j.title "
                "FROM resume_workspaces rw "
                "JOIN applications a ON a.id = rw.application_id "
                "JOIN jobs j ON j.id = a.job_id "
                "WHERE j.company = ? AND (? IS NULL OR j.id = ?) "
                "ORDER BY rw.updated_at DESC",
                (company, job_id, job_id),
            ).fetchall()
        if not rows:
            return _ext_response(
                404,
                {
                    "error": "no reviewed application workspace",
                    "hint": f"先在 OfferGuide 为 {company} 的具体岗位完成投递包",
                },
            )
        if len(rows) > 1 and job_id is None:
            return _ext_response(
                409,
                {
                    "error": "multiple workspaces for this company; job_id is required",
                    "matches": [
                        {
                            "workspace_id": int(row[0]),
                            "job_id": int(row[2]),
                            "title": str(row[4] or f"job#{row[2]}"),
                        }
                        for row in rows
                    ],
                },
            )
        workspace_id, raw_pack, selected_job_id, status, _title = rows[0]
        try:
            stored = json_loads(raw_pack or "{}")
        except (json.JSONDecodeError, TypeError):
            return _ext_response(500, {"error": "stored workspace package is corrupt"})
        package = stored.get("assistant") if isinstance(stored, dict) else None
        if not isinstance(package, dict) or package.get("_error"):
            return _ext_response(409, {"error": "workspace has no usable application copy"})
        try:
            package = ApplicationPackage.model_validate(package).model_dump(mode="json")
        except ValidationError:
            return _ext_response(409, {"error": "workspace application copy is invalid"})
        return _ext_response(
            200,
            {
                "workspace_id": int(workspace_id),
                "workspace_status": str(status),
                "package": package,
                "job_id": int(selected_job_id),
                "company": company,
            },
        )
    def _ext_response(status: int, body: dict) -> JSONResponse:
        """Wrap with CORS headers (extension origin is the platform site, not localhost)."""
        resp = JSONResponse(body, status_code=status)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return resp

    return app


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


# ─────────────────────── stats / dashboard helpers ──────────────────


def _extract_trigger_goal(trigger_detail_json: str | None) -> str:
    """Pull a human-readable 'goal' string out of harness_runs.trigger_detail.

    The harness records the *trigger* (cron / event / user_input / scheduled);
    callers who previously read `agent_runs.goal` now read the trigger
    detail. For user_input we use `detail.message`; for cron/scheduled we
    use `detail.reason`; otherwise we synthesize from the event kind.
    """
    if not trigger_detail_json:
        return ""
    try:
        import json as _j
        detail = _j.loads(trigger_detail_json)
    except Exception:
        return ""
    if not isinstance(detail, dict):
        return ""
    msg = detail.get("message")
    if isinstance(msg, str) and msg.strip():
        return msg[:300]
    reason = detail.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason[:300]
    event = detail.get("event")
    if isinstance(event, str) and event.strip():
        job_id = detail.get("job_id")
        if job_id:
            return f"event {event} (job#{job_id})"
        return f"event {event}"
    return ""


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


def _recent_agent_artifacts(store: Store, *, limit: int = 6) -> list[dict[str, Any]]:
    """Latest artifacts produced through Agent Chat tools."""
    from ..agent_runtime import _schema as _hs

    try:
        _hs.init_agent_runtime_schema(store)
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT id, kind, job_id, note, "
                "(julianday('now') - created_at) * 86400 AS age_seconds "
                "FROM harness_events "
                "WHERE kind IN ('project_record_saved', 'project_assessed') "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    except Exception:
        return []

    artifacts: list[dict[str, Any]] = []
    for row in rows:
        artifact = _artifact_from_event_row(row)
        if artifact is None:
            continue
        age = float(row[4] or 0)
        artifact["when_ago"] = _humanize_age(age)
        artifacts.append(artifact)
    return artifacts


def _artifact_from_event_row(row: Any) -> dict[str, Any] | None:
    event_id, kind, _job_id, note = row[0], row[1], row[2], row[3]
    try:
        payload = json_loads(note or "{}")
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    if kind == "project_record_saved":
        project_id = payload.get("project_id")
        title = payload.get("title") or f"project#{project_id}"
        return {
            "event_id": event_id,
            "kind": kind,
            "label": "项目档案",
            "title": f"{title} · project#{project_id}",
            "view": payload.get("view") or "/project-vault",
            "detail": f"方向: {payload.get('direction') or '未记录'}",
        }
    if kind == "project_assessed":
        return {
            "event_id": event_id,
            "kind": kind,
            "label": "项目评估",
            "title": (
                "可按 AI Agent 写" if payload.get("is_agent_project")
                else "先别硬包装成 Agent"
            ),
            "view": "/project-vault",
            "detail": (
                f"next_action={payload.get('next_action') or 'unknown'} · "
                f"缺事实 {len(payload.get('missing_facts') or [])} 项"
            ),
        }
    return None


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


def _to_julian(dt) -> float:
    """Calendar UTC datetime → SQLite julianday float."""
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    frac = (dt.hour - 12) / 24 + dt.minute / 1440 + dt.second / 86400
    return jdn + frac


def _format_julian_time(value: float | None) -> str:
    """Render a SQLite Julian timestamp without exposing storage details."""
    if value is None:
        return ""
    timestamp = (float(value) - 2_440_587.5) * 86_400
    return datetime.fromtimestamp(timestamp, tz=UTC).astimezone().strftime(
        "%Y-%m-%d %H:%M"
    )


def _invocation_view(invocation: Any | None) -> dict[str, Any] | None:
    if invocation is None:
        return None
    labels = {
        "queued": "等待 Agent 开始",
        "running": "Agent 正在研究",
        "published": "已更新当前结果",
        "unchanged": "当前结果无需更新",
        "blocked": "暂时缺少足够证据",
        "stale": "上下文已变化，本次结果未发布",
        "failed": "Agent 运行失败",
    }
    return {
        "id": invocation.id,
        "status": invocation.status,
        "status_label": labels.get(invocation.status, invocation.status),
        "message": invocation.message,
        "updated_at": _format_julian_time(invocation.updated_at),
        "error": invocation.error_text,
        "unresolved": list(invocation.unresolved),
    }


# -------------------- entry point used by `python -m offerguide.ui.web` --------------------


def main() -> None:
    """Build everything from env vars and serve via uvicorn."""
    import uvicorn

    settings = Settings.from_env()
    store = Store(settings.db_path)
    store.init_schema()

    master_source: MasterResumeSource | None = None
    if settings.resume_pdf and settings.resume_pdf.exists():
        master_source = load_resume_pdf(settings.resume_pdf)

    skills_root = Path(__file__).parent.parent / "skills"
    skills = discover_skills(skills_root)

    llm: LLMClient | None = None
    runtime: SkillRuntime | None = None
    text_resume_editor: ResumeEditor | None = None
    if settings.deepseek_api_key:
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        runtime = SkillRuntime(llm, store)
        text_resume_editor = ResumeEditor(llm)

    visual_resume_editor: ResumeEditor | None = None
    if settings.vision_api_key and settings.vision_base_url and settings.vision_model:
        vision_llm = LLMClient(
            api_key=settings.vision_api_key,
            base_url=settings.vision_base_url,
            default_model=settings.vision_model,
        )
        visual_resume_editor = ResumeEditor(vision_llm)

    notifier = make_notifier(settings)
    research_agents = (
        ResearchAgentService(settings=settings, store=store, llm=llm)
        if llm is not None
        else None
    )

    app = create_app(
        settings=settings,
        store=store,
        master_source=master_source,
        skills=skills,
        runtime=runtime,
        resume_editor=text_resume_editor,
        visual_resume_editor=visual_resume_editor,
        notifier=notifier,
        research_agents=research_agents,
    )

    print(f"\n✦ OfferGuide UI on http://{settings.web_host}:{settings.web_port}")
    print(f"  resume    = {settings.resume_pdf or '(none — set OFFERGUIDE_RESUME_PDF)'}")
    print(
        f"  llm       = {'configured' if settings.deepseek_api_key else 'NOT configured (set DEEPSEEK_API_KEY)'}"
    )
    print(f"  notify    = {settings.notify_channel} ({'ready' if settings.notify_ready() else 'fallback console'})")
    print(
        "  research  = "
        + (
            "configured; background refresh disabled"
            if research_agents is not None and settings.disable_background_agents
            else "configured; JobDiscoveryAgent background refresh enabled"
            if research_agents is not None
            else "NOT configured (LLM key required)"
        )
    )
    uvicorn.run(app, host=settings.web_host, port=settings.web_port, log_level="info")


# `_` to silence unused-symbol lint in linters that don't read entry points
_ = RedirectResponse


if __name__ == "__main__":
    main()
