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

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import inbox as inbox_mod
from ..agent import build_graph
from ..agent.state import RequestedAction
from ..config import Settings
from ..llm import LLMClient, LLMError
from ..memory import Store
from ..profile import UserProfile, load_resume_pdf
from ..skills import SkillRuntime, SkillSpec, discover_skills
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
            request, "home.html", _ctx(request, items=items)
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
            ),
        )

    @app.post("/chat", response_class=HTMLResponse)
    def chat(
        request: Request,
        job_text: str = Form(...),
        action: str = Form("score_and_gaps"),
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

        action_norm: RequestedAction = (
            action if action in ("score", "gaps", "score_and_gaps") else "score_and_gaps"
        )  # type: ignore[assignment]

        graph = build_graph(skills=skills, runtime=runtime)
        try:
            result = graph.invoke(
                {
                    "messages": [{"role": "user", "content": job_text}],
                    "requested_action": action_norm,
                    "job_text": job_text,
                    "user_profile_text": profile.raw_resume_text,
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
                inbox_title=f"考虑投递: {title_first_line}",
                inbox_body=(result.get("final_response") or "")[:800],
                job_text=job_text[:2000],
                score_run_id=result.get("score_run_id"),
                gaps_run_id=result.get("gaps_run_id"),
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

    return app


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
