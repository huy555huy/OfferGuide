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
import json
import logging
import os
import re
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .. import inbox as inbox_mod
from ..application_plan import build_application_plan
from ..config import Settings
from ..llm import LLMClient, LLMError
from ..memory import Store
from ..platforms._spec import RawJob
from ..profile import UserProfile, load_resume_pdf
from ..skills import SkillRuntime, SkillSpec, discover_skills
from ..workers import scout
from .notify import Notifier, make_notifier

log = logging.getLogger(__name__)

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
    # FastAPI lifespan: start the ambient discovery task unless explicitly
    # disabled. The task is useful in normal app mode, but tests and one-off
    # UI probes must be able to opt out without surprise network writes.
    import asyncio as _async_mod
    import contextlib as _ctxlib

    from ..workers.ambient import _ambient_discovery_loop

    @_ctxlib.asynccontextmanager
    async def _lifespan(app: FastAPI):
        bg_task: _async_mod.Task[None] | None = None
        # Disabled when api_key is empty (no point crawling without downstream
        # score_match) or when env opt-out (tests / probes don't want network).
        if (settings.deepseek_api_key
                and not getattr(settings, "disable_ambient_crawl", False)):
            bg_task = _async_mod.create_task(
                _ambient_discovery_loop(
                    store=store, settings=settings,
                    runtime=runtime, skills=skills,
                    user_profile_text=(
                        profile.raw_resume_text if profile else None
                    ),
                ),
                name="offerguide_ambient_discovery",
            )
        try:
            yield
        finally:
            if bg_task is not None:
                bg_task.cancel()
                with _ctxlib.suppress(BaseException):
                    await bg_task

    app = FastAPI(
        title="OfferGuide", docs_url=None, redoc_url=None, lifespan=_lifespan,
    )
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

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
            "profile_loaded": profile is not None,
            "profile_chars": len(profile.raw_resume_text) if profile else 0,
            "nav": nav,
        }
        base.update(extra)
        return base

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> Any:
        return agent_page(request)

    @app.get("/today", response_class=HTMLResponse)
    def home_legacy(request: Request) -> Any:
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
        # Most recent harness run for the hero. Graceful: harness_runs may
        # not exist yet on a brand-new install (init_harness_schema runs on
        # first harness.run, not at web boot).
        row = None
        try:
            with store.connect() as conn:
                row = conn.execute(
                    "SELECT id, trigger_kind, trigger_detail, final_text, "
                    "       started_at, ended_at, iterations, status "
                    "FROM harness_runs WHERE final_text IS NOT NULL "
                    "  AND status = 'ok' "
                    "ORDER BY started_at DESC LIMIT 1"
                ).fetchone()
        except Exception:
            row = None
        latest_run = None
        if row:
            goal = _extract_trigger_goal(row[2])
            latency_ms: int | None = None
            if row[5] is not None and row[4] is not None:
                latency_ms = int((float(row[5]) - float(row[4])) * 86400 * 1000)
            latest_run = {
                "id": row[0], "goal": goal, "final_answer": row[3],
                "critic_score": None, "critic_notes": None,
                "latency_ms": latency_ms, "started_at": row[4],
                "status": row[7], "iterations": row[6],
                "trigger_kind": row[1],
            }

        # Pending agent_suggestion items
        suggestions = [
            i for i in inbox_mod.list_items(store, status="pending", limit=20)
            if i.kind == "agent_suggestion"
        ][:6]
        # W14.20 — pending questions agent asked user (separate, more urgent)
        pending_questions = [
            i for i in inbox_mod.list_items(store, status="pending", limit=20)
            if i.kind == "question"
        ][:3]

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
            # W14.11: state-aware next-step suggestion. Replaces the generic
            # "三步走" onboarding banner with a context-sensitive prompt
            # that points at the actual next thing to do.
            n_active_goals = conn.execute(
                "SELECT COUNT(*) FROM user_goals WHERE status='active'"
            ).fetchone()[0]
            latest_job_id = conn.execute(
                "SELECT id FROM jobs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            latest_job_id = latest_job_id[0] if latest_job_id else None

            # W14.12 — "agent 本周报告" data: what the autonomous daemons
            # actually accomplished in the last 7 days, so home reads as
            # "look what your agent did" rather than "tell the agent what
            # to do". Window is rolling 7 days from now.
            n_jobs_auto_found_week = conn.execute(
                "SELECT COUNT(*) FROM jobs "
                "WHERE source = 'agent_search' "
                "  AND created_at >= julianday('now') - 7"
            ).fetchone()[0]
            n_score_runs_week = conn.execute(
                "SELECT COUNT(*) FROM skill_runs "
                "WHERE skill_name = 'score_match' "
                "  AND created_at >= julianday('now') - 7"
            ).fetchone()[0]
            n_suggestions_week = conn.execute(
                "SELECT COUNT(*) FROM inbox_items "
                "WHERE kind = 'agent_suggestion' "
                "  AND created_at >= julianday('now') - 7"
            ).fetchone()[0]
            try:
                n_agent_runs_week = conn.execute(
                    "SELECT COUNT(*) FROM harness_runs "
                    "WHERE started_at >= julianday('now') - 7 "
                    "  AND status = 'ok'"
                ).fetchone()[0]
            except Exception:
                n_agent_runs_week = 0
            # cost burned by autonomous activity this week (skill_runs + harness_runs)
            cost_week_row = conn.execute(
                "SELECT "
                "  COALESCE(SUM(cost_usd), 0) "
                "FROM skill_runs WHERE created_at >= julianday('now') - 7"
            ).fetchone()
            cost_week = float(cost_week_row[0] or 0.0)

            # W14.13 — Mission Control daemon status. For each cron daemon,
            # read the most-recent daemon_runs row + count of recent runs.
            # Lets the home show "agent is doing X right now / last did Y at
            # T / next runs at Z" instead of a static brochure.
            # W14.18: now only wake_agent is on a cron. discover/score
            # are tools the agent calls itself when it judges they're
            # needed. We still expose them as "manual trigger" cards on
            # Mission Control so the user can force-run for impatience /
            # debugging.
            daemon_specs = [
                {
                    "name": "wake_agent",
                    "icon": "🤖",
                    "label": "中央 agent (心跳)",
                    "what": "每小时唤醒, 看全局自主决定干啥 (找 JD / score / "
                            "follow up / lay low) — 整个 OfferGuide 的大脑",
                    "schedule": "每小时 (08-22), 心跳唯一 cron",
                },
                {
                    "name": "discover_new_jobs",
                    "icon": "🔍",
                    "label": "找新 JD (子 agent)",
                    "what": "DiscoverySubAgent 用 9 个 verified 官方源 fetcher "
                            "(nowcoder / 腾讯 / 百度 / 字节 / 0voice / 实习僧) "
                            "找匹配 north star 的 JD. 平时由 harness 主 agent "
                            "自己调; 这里 ▶ 是手动触发, 等不及 cron 时用",
                    "schedule": "由 harness 主 agent 自主决定 (无独立 cron)",
                },
                {
                    "name": "score_unscored_jobs",
                    "icon": "📊",
                    "label": "评分 + 推到 inbox",
                    "what": "扫待评分 JD 跑 score_match, 高分预生成投递包推到 "
                            "inbox. 由中央 agent 自己调; 这里 ▶ 是手动触发",
                    "schedule": "由中央 agent 自主决定 (无独立 cron)",
                },
            ]
            # W14.19 — alias-aware lookup. The daemon_runs table has
            # historical rows under the OLD canonical names ("discover_jobs_via_search"
            # / "auto_score_new_jobs") from when those were independent crons,
            # plus the wake_agent → maintenance tool path that may write under
            # either old OR new names depending on which dispatch wrote it.
            # Without alias-aware query Mission Control shows "从未跑过" even
            # when the daemon really ran (W14.18 → user thought "失败了").
            DAEMON_NAME_ALIASES = {
                "discover_new_jobs": ("discover_new_jobs", "discover_jobs_via_search"),
                "score_unscored_jobs": ("score_unscored_jobs", "auto_score_new_jobs"),
                "wake_agent": ("wake_agent",),
            }
            daemon_status = []
            for spec in daemon_specs:
                names = DAEMON_NAME_ALIASES.get(spec["name"], (spec["name"],))
                placeholders = ",".join("?" * len(names))
                last_row = conn.execute(
                    f"SELECT id, started_at, ended_at, status, summary_json, error_text "
                    f"FROM daemon_runs WHERE job_name IN ({placeholders}) "
                    f"ORDER BY id DESC LIMIT 1",
                    names,
                ).fetchone()
                runs_24h = conn.execute(
                    f"SELECT COUNT(*) FROM daemon_runs "
                    f"WHERE job_name IN ({placeholders}) "
                    f"  AND started_at >= julianday('now') - 1",
                    names,
                ).fetchone()[0]
                last = None
                if last_row:
                    summary = {}
                    with contextlib.suppress(Exception):
                        summary = json.loads(last_row[4] or "{}")
                    last = {
                        "id": last_row[0],
                        "started_at": last_row[1],
                        "ended_at": last_row[2],
                        "status": last_row[3],
                        "summary": summary,
                        "error_text": last_row[5],
                    }
                daemon_status.append({**spec, "last": last, "runs_24h": runs_24h})

            # Recent daemon activity timeline (last 8 entries across all daemons)
            timeline_rows = conn.execute(
                "SELECT id, job_name, started_at, ended_at, status, "
                "       summary_json, error_text "
                "FROM daemon_runs ORDER BY id DESC LIMIT 8"
            ).fetchall()
            activity_timeline = []
            for r in timeline_rows:
                summary = {}
                with contextlib.suppress(Exception):
                    summary = json.loads(r[5] or "{}")
                activity_timeline.append({
                    "id": r[0], "job_name": r[1],
                    "started_at": r[2], "ended_at": r[3],
                    "status": r[4], "summary": summary,
                    "error_text": r[6],
                })

        # State-machine for the next-step card:
        #   no goal & no job  → set a goal OR paste a JD (parallel paths)
        #   goal but no job   → emphasize "add a JD now" (the actual blocker)
        #   job but no app    → "go generate a 投递包" (one click away)
        #   has applications  → "wake agent for review" (let model drive)
        next_step = None
        if not latest_run and (n_active_goals == 0 and n_jobs == 0):
            next_step = {
                "kind": "first_use",
                "title": "👋 第一次用? 这里有两条路, 哪条都行",
            }
        elif n_active_goals > 0 and n_jobs == 0:
            next_step = {
                "kind": "need_jd",
                "title": "🎯 Goal 设好了, 现在缺 JD — 30 秒粘一个就开始",
            }
        elif n_jobs > 0 and n_apps_active == 0 and n_apps_offer == 0:
            next_step = {
                "kind": "have_jd",
                "title": "📋 已经有 JD 在 pipeline, 去生成投递包 / 调简历",
                "latest_job_id": latest_job_id,
            }

        # W15.9 — surface the agent's worldview as the centerpiece of home.
        # This is the "心智窗口" — what the agent actually thinks about the
        # user / their job hunt right now. Read the markdown files the agent
        # owns; show snippets so the user can read agent's brain at a glance.
        worldview_summary: dict[str, str] = {}
        worldview_files: list[str] = []
        try:
            from ..harness import MemoryStore, default_worldview_dir
            wdir = default_worldview_dir(settings)
            mstore = MemoryStore(root=wdir)
            for fname in ("MEMORY.md", "candidate.md", "tracked-jobs.md", "upcoming-events.md"):
                fpath = wdir / fname
                if fpath.exists():
                    text = fpath.read_text(encoding="utf-8")
                    # First 60 lines is enough for at-a-glance — full file at /worldview/<fname>
                    lines = text.splitlines()[:60]
                    worldview_summary[fname] = "\n".join(lines)
            worldview_files = mstore.list_files()
        except Exception as e:
            log.debug("worldview load failed (non-fatal): %s", e)

        # Recent harness runs — 5 most recent for "agent 最近做了啥" strip
        harness_runs_recent: list[dict[str, Any]] = []
        try:
            with store.connect() as conn:
                rows = conn.execute(
                    "SELECT id, trigger_kind, started_at, ended_at, "
                    "       iterations, status, final_text, cost_usd "
                    "FROM harness_runs ORDER BY started_at DESC LIMIT 5"
                ).fetchall()
            for r in rows:
                harness_runs_recent.append({
                    "id": int(r[0]), "trigger_kind": r[1],
                    "started_at": r[2], "ended_at": r[3],
                    "iterations": int(r[4] or 0), "status": r[5],
                    "final_text": (r[6] or "")[:200],
                    "cost_usd": float(r[7] or 0.0),
                })
        except Exception:
            pass  # harness_runs table may not exist yet

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
                next_step=next_step,
                weekly={
                    "jobs_auto_found": n_jobs_auto_found_week,
                    "score_runs": n_score_runs_week,
                    "suggestions": n_suggestions_week,
                    "agent_runs": n_agent_runs_week,
                    "cost_usd": cost_week,
                },
                daemon_status=daemon_status,
                activity_timeline=activity_timeline,
                pending_questions=pending_questions,
                worldview_summary=worldview_summary,
                worldview_files=worldview_files,
                harness_runs_recent=harness_runs_recent,
                runtime_ready=runtime is not None and bool(settings.deepseek_api_key),
                active_tab="home",
            ),
        )

    @app.post("/api/scheduler/trigger/{job_name}", response_class=JSONResponse)
    def manual_trigger_daemon(job_name: str) -> Any:
        """W14.13 — let the user manually trigger a scheduler daemon now,
        instead of waiting for the next cron tick. Used by the Mission
        Control cards on home so "show me what discover_jobs would find
        right now" is a single click.

        Whitelist enforced — only daemons with a clear human-trigger
        meaning. Internal/maintenance jobs aren't exposed via this route.
        """
        # W14.18: accept both old (cron) names and new (agent-tool) names
        # so old links keep working + new home cards work.
        ALIASES = {
            "discover_new_jobs": "discover_jobs_via_search",
            "score_unscored_jobs": "auto_score_new_jobs",
        }
        canonical = ALIASES.get(job_name, job_name)
        VALID = {"discover_jobs_via_search", "auto_score_new_jobs", "wake_agent"}
        if canonical not in VALID:
            raise HTTPException(404, f"unknown daemon: {job_name}")
        job_name = canonical  # use canonical for downstream dispatch + recording
        if not settings.deepseek_api_key:
            raise HTTPException(400, "需要先配 LLM key (.env)")
        if runtime is None:
            raise HTTPException(400, "SkillRuntime 没初始化 (检查 LLM 配置)")

        # W15.7: route through the harness instead of the deleted W14
        # daemon helpers. The endpoint preserves the daemon_runs telemetry
        # contract (mission control timeline) but executes via harness.
        from ..agentic.search import build_default_search
        from ..harness import (
            HarnessDeps,
            MemoryStore,
            default_worldview_dir,
            make_user_input_trigger,
        )
        from ..harness import _schema as _harness_schema
        from ..harness import run as harness_run

        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        try:
            search = build_default_search()
        except Exception:
            search = None
        _harness_schema.init_harness_schema(store)
        deps = HarnessDeps(
            settings=settings,
            store=store,
            memory_store=MemoryStore(root=default_worldview_dir(settings)),
            llm=llm,
            runtime=runtime,
            skills=skills,
            search=search,
            notifier=notifier,
            user_profile_text=profile.raw_resume_text if profile else None,
        )

        # Record a daemon_runs row for this manual trigger so the
        # Mission Control activity timeline still shows it (UI legacy).
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO daemon_runs(job_name, status, summary_json) "
                "VALUES (?, 'running', '{}') RETURNING id",
                (job_name,),
            )
            run_id = int(cur.fetchone()[0])

        # Map old daemon names to a user-input goal the harness agent runs
        if job_name == "discover_jobs_via_search":
            user_msg = "(manual trigger) 帮我用 discover_jobs 找几个新岗位."
        elif job_name == "auto_score_new_jobs":
            user_msg = "(manual trigger) 给最近还没 score 的 jobs 跑 score_match."
        else:  # wake_agent
            user_msg = "(manual trigger) 看下当前状态自己决定干啥."

        try:
            res = harness_run(
                trigger=make_user_input_trigger(user_msg), deps=deps,
                max_iterations=15,
            )
            result = {
                "harness_run_id": res.run_id,
                "iterations": res.iterations,
                "finish": res.finish_reason,
                "cost_usd": round(res.cost_usd, 4),
            }
            with store.connect() as conn:
                conn.execute(
                    "UPDATE daemon_runs SET status='ok', "
                    "  ended_at=julianday('now'), summary_json=? WHERE id=?",
                    (json.dumps(result, ensure_ascii=False, default=str)[:4000], run_id),
                )
            return {"job": job_name, "result": result, "run_id": run_id}
        except Exception as e:
            with store.connect() as conn:
                conn.execute(
                    "UPDATE daemon_runs SET status='error', "
                    "  ended_at=julianday('now'), error_text=? WHERE id=?",
                    (str(e)[:500], run_id),
                )
            raise HTTPException(500, f"daemon failed: {e}") from None
        finally:
            with contextlib.suppress(Exception):
                llm.close()

    @app.post("/api/home/wake-agent", response_class=JSONResponse)
    async def home_wake_agent(request: Request) -> Any:
        """Trigger a harness run from the home page (manual user kickoff).

        User clicks "wake agent" on home page → harness runs one loop with
        a user_button trigger, returns the run summary.
        """
        if not settings.deepseek_api_key:
            raise HTTPException(400, "agent 不可用 — 缺 OFFERGUIDE_LLM_API_KEY")

        from ..harness import (
            HarnessDeps,
            MemoryStore,
            TriggerEvent,
            default_worldview_dir,
        )
        from ..harness import _schema as _hs
        from ..harness import run as harness_run

        _hs.init_harness_schema(store)
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
            deps = HarnessDeps(
                settings=settings, store=store,
                memory_store=MemoryStore(root=default_worldview_dir(settings)),
                llm=llm, runtime=runtime, skills=skills,
                search=_search, notifier=notifier,
                user_profile_text=profile.raw_resume_text if profile else None,
            )
            import asyncio
            result = await asyncio.to_thread(
                harness_run,
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
        """W15.9 — chat input on home → user_input trigger to harness.

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

        from ..harness import (
            HarnessDeps,
            MemoryStore,
            default_worldview_dir,
            make_user_input_trigger,
        )
        from ..harness import _schema as _hs
        from ..harness import run as harness_run

        _hs.init_harness_schema(store)
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
            deps = HarnessDeps(
                settings=settings, store=store,
                memory_store=MemoryStore(root=default_worldview_dir(settings)),
                llm=llm, runtime=runtime, skills=skills,
                search=_search, notifier=notifier,
                user_profile_text=profile.raw_resume_text if profile else None,
            )
            import asyncio as _asyncio
            res = await _asyncio.to_thread(
                harness_run,
                trigger=make_user_input_trigger(message),
                deps=deps,
                max_iterations=15,
            )
        finally:
            with contextlib.suppress(Exception):
                llm.close()
        return {
            "run_id": res.run_id,
            "iterations": res.iterations,
            "finish_reason": res.finish_reason,
            "final_text": res.final_text[:2000],
            "tool_calls": res.tool_call_log[-10:],
            "cost_usd": round(res.cost_usd, 4),
        }

    @app.post("/api/evaluate-job", response_class=JSONResponse)
    async def evaluate_job_endpoint(request: Request) -> Any:
        """W15.14 hero flow: paste a JD URL/text → structured evaluation.

        Bypasses agent loop for speed (~10-20s vs 30-90s) since this is
        a user-driven flow where we KNOW we want fetch + score + tailor.
        Returns structured JSON the frontend renders as cards (not a
        raw markdown blob from agent's final_text).
        """
        if not settings.deepseek_api_key:
            raise HTTPException(400, "需要先配 LLM key (.env)")
        if runtime is None:
            raise HTTPException(400, "SkillRuntime 未初始化")
        body = await request.json()
        url_or_text = (body.get("url_or_text") or "").strip()
        if not url_or_text:
            raise HTTPException(400, "url_or_text 不能为空")
        if len(url_or_text) > 50_000:
            raise HTTPException(400, "JD 文本太长 (max 50K 字符)")
        company_hint = (body.get("company_hint") or "").strip() or None
        title_hint = (body.get("title_hint") or "").strip() or None

        from ..harness import (
            HarnessDeps,
            MemoryStore,
            default_worldview_dir,
        )
        from ..harness import _schema as _hs
        from ..harness.evaluate import evaluate_job
        _hs.init_harness_schema(store)
        deps = HarnessDeps(
            settings=settings, store=store,
            memory_store=MemoryStore(root=default_worldview_dir(settings)),
            llm=LLMClient(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                default_model=settings.default_model,
            ),
            runtime=runtime, skills=skills,
            user_profile_text=profile.raw_resume_text if profile else None,
            notifier=notifier,
        )
        import asyncio as _asyncio
        try:
            result = await _asyncio.to_thread(
                evaluate_job,
                url_or_text=url_or_text, deps=deps,
                company_hint=company_hint, title_hint=title_hint,
            )
        finally:
            with contextlib.suppress(Exception):
                if deps.llm:
                    deps.llm.close()
        return result.to_dict()

    @app.get("/api/keywords", response_class=JSONResponse)
    def keywords_list() -> Any:
        """Current user-managed include/exclude keyword lists for /recommended."""
        from .. import user_keywords as _uk
        inc, exc = _uk.list_keywords(store)
        return {
            "includes": [{"id": k.id, "keyword": k.keyword} for k in inc],
            "excludes": [{"id": k.id, "keyword": k.keyword} for k in exc],
        }

    @app.post("/api/keywords", response_class=JSONResponse)
    async def keywords_add(request: Request) -> Any:
        """Add an include/exclude keyword. Body: ``{keyword, kind}``.

        ``kind='include'`` — ambient discovery loop will rotate this into
        the per-cycle keyword set (verified-official + shixiseng fetchers).
        ``kind='exclude'`` — /recommended hides cards whose JD text matches.
        """
        from .. import user_keywords as _uk
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "JSON body required") from None
        keyword = (body.get("keyword") or "").strip()
        kind = body.get("kind") or "include"
        if not keyword:
            raise HTTPException(400, "keyword 不能为空")
        if kind not in ("include", "exclude"):
            raise HTTPException(400, "kind must be 'include' or 'exclude'")
        try:
            was_new, kid = _uk.add_keyword(store, keyword=keyword, kind=kind)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        return {"id": kid, "keyword": keyword, "kind": kind, "added": was_new}

    @app.delete("/api/keywords/{kid}", response_class=JSONResponse)
    def keywords_remove(kid: int) -> Any:
        """Remove one user keyword by id."""
        from .. import user_keywords as _uk
        removed = _uk.remove_keyword(store, kid)
        if not removed:
            raise HTTPException(404, f"keyword#{kid} not found")
        return {"id": kid, "removed": True}

    @app.post("/api/jobs/{job_id}/report-dead", response_class=JSONResponse)
    async def report_dead_job(job_id: int, request: Request) -> Any:
        """User clicked "失效 →" on a /recommended card — mark this row dead.

        Background: ingest captures a JD URL at crawl-time; platforms (especially
        nowcoder) take down listings days later. The user discovers it as "查无
        此岗" and would otherwise be stuck seeing the dead card on every visit.

        We don't hard-delete (preserves score history + signal attribution),
        just stash a flag on jobs.extras_json. /recommended SQL hides any row
        with extras.dead = true.
        """
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        reason = (payload.get("reason") or "user reported 404 / dead url")[:200]

        with store.connect() as conn:
            row = conn.execute(
                "SELECT extras_json FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, f"job#{job_id} not found")
            try:
                extras = json.loads(row[0] or "{}")
                if not isinstance(extras, dict):
                    extras = {}
            except (json.JSONDecodeError, TypeError):
                extras = {}
            extras["dead"] = True
            extras["dead_reason"] = reason
            extras["dead_reported_at"] = (
                __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat()
            )
            conn.execute(
                "UPDATE jobs SET extras_json = ? WHERE id = ?",
                (json.dumps(extras, ensure_ascii=False), job_id),
            )

        # Log as a harness event so the agent learns 'this source returned a
        # dead URL' — input for downstream quality decisions.
        from ..harness import _schema as _hs
        _hs.init_harness_schema(store)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_events(kind, job_id, note, source) "
                "VALUES (?, ?, ?, ?)",
                ("dead_url_reported", job_id, reason, "user"),
            )
        return {"job_id": job_id, "marked_dead": True}


    @app.post("/api/jobs/{job_id}/track", response_class=JSONResponse)
    def track_job(job_id: int, request: Request) -> Any:
        """W15.14: user clicks "加入跟踪" on an evaluation result card.

        Adds an applications row with status='considered' so /jobs page
        shows it and agent sees it in worldview/tracked-jobs.md (next
        wake will reflect via record_event).

        Idempotent: if already tracked, returns existing application_id.
        """
        with store.connect() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, f"job#{job_id} not found")
            existing = conn.execute(
                "SELECT id FROM applications WHERE job_id = ?", (job_id,),
            ).fetchone()
            if existing:
                return {"application_id": int(existing[0]), "created": False}
            cur = conn.execute(
                "INSERT INTO applications(job_id, status) VALUES (?, 'considering') "
                "RETURNING id",
                (job_id,),
            )
            app_id = int(cur.fetchone()[0])

        # Also fire a harness event so agent knows next wake
        from ..harness import _schema as _hs
        _hs.init_harness_schema(store)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_events(kind, job_id, note, source) "
                "VALUES (?, ?, ?, ?)",
                ("user_tracked", job_id, "user clicked '加入跟踪'", "user"),
            )
        return {"application_id": app_id, "created": True}

    @app.post("/api/jobs/{job_id}/applied", response_class=JSONResponse)
    def mark_applied(job_id: int, request: Request) -> Any:
        """W15.14: user clicks "已投" on a job — track + transition status.

        Idempotent. Triggers a 7-day-out scheduled wake so agent
        proactively checks for response (true ambient ownership).
        """
        with store.connect() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, f"job#{job_id} not found")
            existing = conn.execute(
                "SELECT id, status FROM applications WHERE job_id = ?", (job_id,),
            ).fetchone()
            if existing:
                app_id = int(existing[0])
                conn.execute(
                    "UPDATE applications SET status = 'applied', "
                    "  applied_at = COALESCE(applied_at, julianday('now')) "
                    "WHERE id = ?", (app_id,),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO applications(job_id, status, applied_at) "
                    "VALUES (?, 'applied', julianday('now')) RETURNING id",
                    (job_id,),
                )
                app_id = int(cur.fetchone()[0])

        # Schedule a 7-day-out wake to check for response. This wires
        # the "agent ownership" promise: after user marks applied, the
        # agent will proactively wake to check status without user nag.
        from ..harness import _schema as _hs
        _hs.init_harness_schema(store)
        with store.connect() as conn:
            # 7 days = 7.0 in julianday delta
            conn.execute(
                "INSERT INTO harness_scheduled_wakes(fire_at, reason) "
                "VALUES (julianday('now') + 7, ?)",
                (f"check job#{job_id} response 7 days after user applied",),
            )
            conn.execute(
                "INSERT INTO harness_events(kind, job_id, note, source) "
                "VALUES (?, ?, ?, ?)",
                ("user_marked_applied", job_id, "user clicked '已投'", "user"),
            )

        # W20.4 — REAL signal (not LLM-judging-LLM): user actually acted on
        # this job → record follow_through=True for the score_match SKILL run
        # that recommended it. This is ground-truth, not opinion. Goes
        # straight into evolution_signals (weight 0.8).
        try:
            from .. import evolution as _evo
            with store.connect() as conn:
                # Find the score_match skill_run that scored this job
                # (most recent). harness_events.kind='scored' carries the
                # skill_run_id in the JSON note.
                row = conn.execute(
                    "SELECT json_extract(note, '$.skill_run_id') as srid "
                    "FROM harness_events "
                    "WHERE kind = 'scored' AND job_id = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (job_id,),
                ).fetchone()
                if row and row[0] is not None:
                    srid = int(row[0])
                    # Look up the SKILL version that ran
                    sr = conn.execute(
                        "SELECT skill_version FROM skill_runs WHERE id = ?",
                        (srid,),
                    ).fetchone()
                    if sr and sr[0]:
                        _evo.record_follow_through(
                            store,
                            skill_name="score_match",
                            skill_version=str(sr[0]),
                            skill_run_id=srid,
                            executed=True,
                            notes=f"user clicked '我投了' on job#{job_id}",
                        )
        except Exception as e:
            log_mod = __import__("logging").getLogger(__name__)
            log_mod.debug("follow_through signal write failed: %s", e)

        return {"application_id": app_id, "next_wake": "+7d"}

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

    def _link_skill_to_job_and_signal(
        *, skill_name: str, skill_run_id: int | None,
        job_id: int, signal_kind: str,
    ) -> None:
        """W20.5 — write 2 things in one go (idempotent, errors swallowed):

        1. ``harness_event`` linking this skill_run_id ↔ job_id (so later
           code can find "which apply_assistant call was for this job").
           Was missing pre-W20.5 for everything except score_match.
        2. ``evolution_signal`` of kind 'follow_through' (real signal: user
           opened this view = they're using the SKILL output).

        Skipped silently if skill_run_id is None (SKILL didn't run, e.g.,
        no LLM key — nothing to attribute).
        """
        if skill_run_id is None:
            return
        try:
            with store.connect() as conn:
                # Look up skill_version for accurate evolve attribution
                sr = conn.execute(
                    "SELECT skill_version FROM skill_runs WHERE id = ?",
                    (skill_run_id,),
                ).fetchone()
                if not sr or not sr[0]:
                    return
                skill_version = str(sr[0])
                conn.execute(
                    "INSERT INTO harness_events(kind, job_id, note, source) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        signal_kind, job_id,
                        json.dumps({
                            "skill_run_id": skill_run_id,
                            "skill_name": skill_name,
                            "skill_version": skill_version,
                        }, ensure_ascii=False),
                        "view_visit",
                    ),
                )
            from .. import evolution as _evo
            _evo.record_follow_through(
                store,
                skill_name=skill_name,
                skill_version=skill_version,
                skill_run_id=skill_run_id,
                executed=True,
                notes=f"user opened view for job#{job_id}",
            )
        except Exception as e:
            log.debug("link_skill_to_job_and_signal failed: %s", e)

    @app.get("/jobs/{job_id}/apply-pack", response_class=HTMLResponse)
    async def apply_pack_view(job_id: int, request: Request) -> Any:
        """投递包页面 — **先微调简历, 再生成投递话术**.

        CLAUDE.md 主流程:
            agent[找岗位] → /recommended → 用户挑 →
            [投递包: 微调简历 + 自我介绍 + 网申 QA + 内推码] →
            用户去官网投

        实现:
        - 并行跑 tailor_resume (针对此 JD 的简历) + apply_assistant
          (自我介绍 + Q&A) 两个 SKILL
        - 两个都走 use_cache=True, 第一次冷启动 ~30-40s, 之后秒开
        - 模板顶部先渲染 tailored 简历 + diff, 然后才是网申话术
        """
        with store.connect() as conn:
            row = conn.execute(
                "SELECT id, title, company, location, url, raw_text, source, extras_json "
                "FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"job#{job_id} 不存在")
        job = {
            "id": int(row[0]), "title": row[1] or "", "company": row[2] or "",
            "location": row[3] or "", "url": row[4] or "",
            "raw_text": row[5] or "",
            "source": row[6] or "", "extras_json": row[7] or "{}",
        }
        application_plan = build_application_plan(job)

        # Run tailor_resume + apply_assistant concurrently. Both are LLM
        # calls (~15-25s each cold) so gather() halves wall time. Cached
        # invocations short-circuit instantly so warm visits stay snappy.
        import asyncio as _asyncio

        from .. import project_vault as _pv
        from ..harness.tools import _format_jd_for_skill
        from ..skill_view import invoke_skill_for_view

        tailor_task = invoke_skill_for_view(
            skill_name="tailor_resume",
            inputs_builder=lambda _spec, p: {
                "master_resume": _pv.append_to_profile_text(
                    store, p.raw_resume_text, max_project_chars=3500,
                )[:9000],
                "job_text": _format_jd_for_skill(job)[:5000],
                "company": job["company"],
                "successful_profile_json": "{}",
            },
            settings=settings, profile=profile, runtime=runtime,
            skills=skills, store=store,
        )
        apply_task = invoke_skill_for_view(
            skill_name="apply_assistant",
            inputs_builder=lambda _spec, p: {
                "company": job["company"],
                "role_focus": job["title"],
                "job_text": _format_jd_for_skill(job)[:5000],
                "user_profile": p.raw_resume_text[:5000],
            },
            settings=settings, profile=profile, runtime=runtime,
            skills=skills, store=store,
        )
        tailor_result, apply_result = await _asyncio.gather(tailor_task, apply_task)

        # Link both SKILL runs to the job for evolution signal attribution.
        _link_skill_to_job_and_signal(
            skill_name="tailor_resume", skill_run_id=tailor_result.skill_run_id,
            job_id=job_id, signal_kind="apply_pack_generated",
        )
        _link_skill_to_job_and_signal(
            skill_name="apply_assistant", skill_run_id=apply_result.skill_run_id,
            job_id=job_id, signal_kind="apply_pack_generated",
        )
        return templates.TemplateResponse(
            request, "apply_pack.html",
            _ctx(request, job=job, application_plan=application_plan,
                 pack=apply_result.parsed, error=apply_result.error,
                 raw=apply_result.raw_text,
                 skill_run_id=apply_result.skill_run_id,
                 cost_usd=(apply_result.cost_usd or 0) + (tailor_result.cost_usd or 0),
                 duration_ms=max(apply_result.duration_ms or 0,
                                 tailor_result.duration_ms or 0),
                 tailored=tailor_result.parsed,
                 tailor_error=tailor_result.error,
                 tailor_run_id=tailor_result.skill_run_id,
                 active_tab="recommended"),
        )

    @app.get("/jobs/{job_id}/post-apply-pack", response_class=HTMLResponse)
    async def post_apply_pack_view(job_id: int, request: Request) -> Any:
        """W15.23 — 投后包页面 (用户点'我投了'后跳转过来).

        跑 prepare_interview SKILL 出: 公司画像 + 高频题 (校准过的概率) +
        备战重点 + 弱点. 也搜 interview_corpus 里的真实面经 (如果有抓过).

        SKILL 输入 verified W15.22: (company, job_text, user_profile, past_experiences).
        """
        with store.connect() as conn:
            row = conn.execute(
                "SELECT id, title, company, location, url, raw_text "
                "FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(404, f"job#{job_id} 不存在")
        job = {
            "id": int(row[0]), "title": row[1] or "", "company": row[2] or "",
            "location": row[3] or "", "url": row[4] or "",
            "raw_text": row[5] or "",
        }

        # Pull past experiences from interview_corpus (if any)
        past_experiences_text = ""
        try:
            from .. import interview_corpus
            experiences = interview_corpus.fetch_for_company(
                store, job["company"], limit=8,
            )
            if experiences:
                past_experiences_text = interview_corpus.render_snippets(
                    experiences, max_chars=4000,
                )
        except Exception as e:
            log.debug("post_apply_pack: interview_corpus lookup failed: %s", e)

        # W19 — invoke prepare_interview via reusable helper.
        # SKILL inputs verified W15.22:
        #   (company, job_text, user_profile, past_experiences)
        from .. import project_vault as _pv
        from ..harness.tools import _format_jd_for_skill
        from ..skill_view import invoke_skill_for_view

        result = await invoke_skill_for_view(
            skill_name="prepare_interview",
            inputs_builder=lambda _spec, p: {
                "company": job["company"],
                "job_text": _format_jd_for_skill(job)[:5000],
                "user_profile": _pv.append_to_profile_text(
                    store, p.raw_resume_text, max_project_chars=3500,
                )[:8500],
                "past_experiences": past_experiences_text or "(无过往面经)",
            },
            settings=settings, profile=profile, runtime=runtime,
            skills=skills, store=store,
        )
        # W20.5 — link SKILL→job + follow_through signal (user opened view).
        _link_skill_to_job_and_signal(
            skill_name="prepare_interview", skill_run_id=result.skill_run_id,
            job_id=job_id, signal_kind="post_apply_pack_generated",
        )
        return templates.TemplateResponse(
            request, "post_apply_pack.html",
            _ctx(request, job=job, pack=result.parsed,
                 error=result.error, raw=result.raw_text,
                 skill_run_id=result.skill_run_id,
                 cost_usd=result.cost_usd, duration_ms=result.duration_ms,
                 past_experiences_chars=len(past_experiences_text),
                 active_tab="recommended"),
        )

    @app.get("/recommended", response_class=HTMLResponse)
    def recommended_view(request: Request) -> Any:
        """W15.21 + W17 — ranked feed with recruit-type filter.

        W17: 应届校招用户(2027 届统计专硕) 主流程要分清:
        - **暑期实习** (5-9 月入职, 通常带转正) ← 5 月节奏的核心
        - **日常实习** (任何时间, 不一定转正)
        - **校招正式** (秋招主体, 9-10 月开)
        - **社招** (社会招聘, 应届生不该投)

        Filter via ?type=summer|daily|fulltime|social|all (default: intern =
        summer+daily, 5 月节奏的应届生默认看的就是这两个).
        """
        from ..recruit_type import (
            LABEL_ZH as RECRUIT_LABEL_ZH,
        )
        from ..recruit_type import (
            classify_recruit_type,
        )
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT j.id, j.title, j.company, j.location, j.url, "
                "       j.source, j.fetched_at, j.extras_json, "
                "       COALESCE(a.status, 'new') as app_status "
                "FROM jobs j "
                "LEFT JOIN applications a ON a.job_id = j.id "
                "WHERE a.status IS NULL "
                "   OR a.status IN ('considered', 'evaluated') "
                "ORDER BY j.fetched_at DESC "
                "LIMIT 400"
            ).fetchall()

            # W15.22 — Score lookup via harness_events kind='scored' (written
            # by tools._exec_score_match + evaluate.py after a successful
            # score_match run). Pre-W15.22 tried json_extract on input_json
            # for nonexistent job_title/job_company keys → always empty.
            # harness_events table may not exist in fresh fixtures — handle
            # OperationalError gracefully.
            scored_by_job: dict[int, dict[str, Any]] = {}
            try:
                score_rows = conn.execute(
                    "SELECT job_id, "
                    "       json_extract(note, '$.probability') as prob, "
                    "       json_extract(note, '$.skill_run_id') as srid, "
                    "       json_extract(note, '$.deal_breakers') as breakers "
                    "FROM harness_events "
                    "WHERE kind = 'scored' AND job_id IS NOT NULL "
                    "ORDER BY id DESC"
                ).fetchall()
                for sr in score_rows:
                    jid = int(sr[0])
                    if jid in scored_by_job:
                        continue  # keep the most recent (already iterated)
                    prob = sr[1]
                    if prob is None:
                        continue
                    breakers_raw = sr[3]
                    breakers: list[str] = []
                    if breakers_raw:
                        try:
                            parsed = json.loads(breakers_raw)
                            if isinstance(parsed, list):
                                breakers = [str(b)[:80] for b in parsed[:3]]
                        except (json.JSONDecodeError, TypeError):
                            pass
                    scored_by_job[jid] = {
                        "probability": float(prob),
                        "skill_run_id": sr[2],
                        "deal_breakers": breakers,
                    }
            except Exception:
                pass  # harness_events not initialized yet → no scores

            # User-managed include/exclude keywords (panel on /recommended)
            from .. import user_keywords as _uk
            user_kw_inc, user_kw_exc = _uk.list_keywords(store)

            candidates: list[dict[str, Any]] = []
            for r in rows:
                (
                    job_id, title, company, location, url, source,
                    _fetched_at, extras_json, app_status,
                ) = r
                # Skip rows the user reported as dead via /api/jobs/{id}/report-dead
                if extras_json:
                    try:
                        _extras_check = json.loads(extras_json)
                        if isinstance(_extras_check, dict) and _extras_check.get("dead"):
                            continue
                    except (json.JSONDecodeError, TypeError):
                        pass
                # Apply user exclude keywords — match against title+company
                # (cheap, no JD body fetch needed). The crawl already happened;
                # this is just a view-time filter.
                if user_kw_exc:
                    hay = ((title or "") + " " + (company or ""))
                    matched = _uk.matches_exclude(hay, user_kw_exc)
                    if matched:
                        continue
                meta = scored_by_job.get(int(job_id))
                # W17 — classify recruit_type from real platform fields
                recruit_type = classify_recruit_type({
                    "source": source or "",
                    "title": title or "",
                    "extras_json": extras_json or "{}",
                })
                # W18 — surface discovered_keyword (audit "哪个 keyword 引来的")
                # so user can see source diversity, not just "AI Agent" everywhere
                extras_obj: dict[str, Any] = {}
                if extras_json:
                    try:
                        extras_obj = json.loads(extras_json)
                        if not isinstance(extras_obj, dict):
                            extras_obj = {}
                    except (json.JSONDecodeError, TypeError):
                        extras_obj = {}
                discovered_keyword = extras_obj.get("discovered_keyword") or ""
                discovered_via = extras_obj.get("discovered_via") or ""
                # SKILL probability is 0-1; UI shows 0-100. Buckets per BOSS
                # cold-apply baseline: <5% industry avg, so >30% = strong fit.
                score: float | None = None
                gaps: list[str] = []
                sr_id = None
                if meta:
                    score = meta["probability"] * 100.0
                    gaps = meta["deal_breakers"]
                    sr_id = meta["skill_run_id"]

                color = "gray"
                verdict = "未评分"
                if score is not None:
                    if score >= 30:
                        color, verdict = "green", "值得投"
                    elif score >= 15:
                        color, verdict = "yellow", "可以试"
                    else:
                        color, verdict = "red", "性价比低"

                candidates.append({
                    "job_id": int(job_id), "title": title or "(未命名)",
                    "company": company or "(未知公司)",
                    "location": location or "",
                    "url": url, "source": source or "",
                    "recruit_type": recruit_type,
                    "recruit_type_label": RECRUIT_LABEL_ZH.get(recruit_type, recruit_type),
                    # W18
                    "discovered_keyword": discovered_keyword,
                    "discovered_via": discovered_via,
                    "application_plan": build_application_plan({
                        "source": source or "",
                        "url": url,
                        "title": title or "",
                        "company": company or "",
                        "location": location or "",
                    }),
                    "app_status": app_status, "score": score,
                    "reasoning": "", "gaps": gaps,
                    "color": color, "verdict": verdict,
                    "score_run_id": sr_id,
                    "has_score": score is not None,
                })

            # Daily chat budget — count today's greeting_drafted events as proxy
            # (harness_events.created_at is julianday REAL, not datetime).
            # Table is created lazily by init_harness_schema; treat absence as 0.
            today_chats = 0
            try:
                today_chats_row = conn.execute(
                    "SELECT COUNT(*) FROM harness_events "
                    "WHERE kind = 'greeting_drafted' "
                    "  AND created_at >= julianday('now', 'start of day')"
                ).fetchone()
                today_chats = int(today_chats_row[0]) if today_chats_row else 0
            except Exception:
                pass

        # W17 — recruit_type filter from query param. Default = intern (summer +
        # daily) which matches 5 月节奏的应届校招生 (用户当前情况).
        from ..recruit_type import (
            ALL_TYPES,
            CAMPUS_FULLTIME,
            DAILY_INTERN,
            SOCIAL,
            SUMMER_INTERN,
        )
        # Aliases: ?type=summer | daily | intern | fulltime | social | all | unknown
        filter_param = (request.query_params.get("type") or "intern").lower()
        type_alias_map = {
            "summer":   {SUMMER_INTERN},
            "daily":    {DAILY_INTERN},
            "intern":   {SUMMER_INTERN, DAILY_INTERN},
            "fulltime": {CAMPUS_FULLTIME},
            "social":   {SOCIAL},
            "all":      set(ALL_TYPES),
        }
        # Also accept exact canonical labels
        if filter_param in ALL_TYPES:
            allowed_types = {filter_param}
        else:
            allowed_types = type_alias_map.get(filter_param, set(ALL_TYPES))

        # Total pool by recruit_type — for filter buttons
        recruit_counts: dict[str, int] = {t: 0 for t in ALL_TYPES}
        for c in candidates:
            recruit_counts[c["recruit_type"]] += 1

        # W18 — source diversity stats (用户能看到 "不只大厂")
        source_counts: dict[str, int] = {}
        company_counts: dict[str, int] = {}
        for c in candidates:
            source_counts[c["source"]] = source_counts.get(c["source"], 0) + 1
            company_counts[c["company"]] = company_counts.get(c["company"], 0) + 1
        # Top non-大厂 companies (heuristic: those NOT in big_co_set get bonus)
        big_co_set = {"腾讯", "百度", "字节跳动", "阿里巴巴", "美团",
                      "京东", "拼多多", "网易", "华为", "小红书"}
        non_big_companies = [
            (co, n) for co, n in company_counts.items() if co not in big_co_set
        ]
        non_big_companies.sort(key=lambda x: -x[1])

        # Apply filter
        filtered = [c for c in candidates if c["recruit_type"] in allowed_types]

        # Sort: scored first by score desc, then unscored by recency
        scored = sorted(
            (c for c in filtered if c["has_score"]),
            key=lambda c: c["score"] or 0.0, reverse=True,
        )
        unscored = [c for c in filtered if not c["has_score"]]
        feed = scored[:50] + unscored[:30]

        # Bucket counts (for the top tile row) — based on filtered set
        green_n = sum(1 for c in scored if c["color"] == "green")
        yellow_n = sum(1 for c in scored if c["color"] == "yellow")
        red_n = sum(1 for c in scored if c["color"] == "red")

        # W18 — keywords daemon will use next cycle (transparency)
        try:
            from ..match_keywords import explain_match, extract_keywords
            from ..workers.ambient import _load_active_north_star
            cycle_keywords = extract_keywords(
                profile.raw_resume_text if profile else None,
                active_goal=_load_active_north_star(store)
                    if hasattr(store, "connect") else None,
                max_keywords=8,
            )
            keyword_explanations = [explain_match(h) for h in cycle_keywords]
        except Exception:
            keyword_explanations = []

        return templates.TemplateResponse(
            request, "recommended.html",
            _ctx(
                request,
                candidates=feed,
                total_pool=len(candidates),
                filtered_count=len(filtered),
                scored_count=len(scored),
                unscored_count=len(unscored),
                green_count=green_n,
                yellow_count=yellow_n,
                red_count=red_n,
                today_chats=today_chats,
                chat_daily_cap=80,
                chat_warn_threshold=70,
                runtime_ready=runtime is not None and bool(settings.deepseek_api_key),
                # W17 — recruit type filter state
                active_filter=filter_param,
                recruit_counts=recruit_counts,
                recruit_type_labels=RECRUIT_LABEL_ZH,
                # W18 — source / company diversity + cycle keywords
                source_counts=source_counts,
                non_big_companies=non_big_companies[:10],
                big_co_count=sum(1 for c in candidates if c["company"] in big_co_set),
                cycle_keywords=keyword_explanations,
                user_keyword_includes=user_kw_inc,
                user_keyword_excludes=user_kw_exc,
                active_tab="recommended",
            ),
        )

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
        from ..harness import _schema as _hs
        _hs.init_harness_schema(store)

        # Daemon status (same logic as home in W14, kept here)
        with store.connect() as conn:
            recent_daemons = conn.execute(
                "SELECT id, job_name, started_at, ended_at, status, "
                "       summary_json, error_text "
                "FROM daemon_runs ORDER BY started_at DESC LIMIT 30"
            ).fetchall()
            recent_harness = conn.execute(
                "SELECT id, trigger_kind, trigger_detail, started_at, "
                "       ended_at, iterations, status, final_text, "
                "       tool_calls_json, cost_usd, error_text "
                "FROM harness_runs ORDER BY started_at DESC LIMIT 30"
            ).fetchall()
            scheduled = conn.execute(
                "SELECT id, fire_at, reason, fired_at, requested_by_run_id "
                "FROM harness_scheduled_wakes ORDER BY fire_at DESC LIMIT 30"
            ).fetchall()
            events = conn.execute(
                "SELECT id, kind, job_id, note, source, created_at "
                "FROM harness_events ORDER BY id DESC LIMIT 30"
            ).fetchall()

        daemons_view = []
        for r in recent_daemons:
            summary = {}
            with contextlib.suppress(Exception):
                summary = json.loads(r[5] or "{}")
            daemons_view.append({
                "id": int(r[0]), "job_name": r[1],
                "started_at": r[2], "ended_at": r[3],
                "status": r[4], "summary": summary,
                "error_text": r[6],
            })
        harness_view = []
        for r in recent_harness:
            tcs = []
            with contextlib.suppress(Exception):
                tcs = json.loads(r[8] or "[]")
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
        scheduled_view = [
            {"id": int(r[0]), "fire_at": r[1], "reason": r[2],
             "fired_at": r[3], "requested_by_run_id": r[4]}
            for r in scheduled
        ]
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
                daemons=daemons_view,
                harness_runs=harness_view,
                scheduled_wakes=scheduled_view,
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

    @app.post("/api/pipeline/jobs/manual", response_class=HTMLResponse)
    def pipeline_add_manual_job(
        request: Request,
        raw_text: str = Form(...),
        title: str | None = Form(None),
        company: str | None = Form(None),
        location: str | None = Form(None),
        url: str | None = Form(None),
    ) -> Any:
        """W14.11: paste-a-JD bootstrap path. The W13.1 cleanup deleted
        /quick-eval thinking "扩展抓 + agent discover_jobs 自动" was enough,
        but real walk-through showed: fresh DB + no extension installed = no
        way to get a JD into the system at all → tailor / apply / agent all
        sit empty. This restores a single-form path: paste raw JD text →
        creates a jobs row → redirect to /apply/<id> so the user lands
        directly on "generate a 投递包".
        """
        from ..platforms import manual

        cleaned = (raw_text or "").strip()
        if len(cleaned) < 50:
            raise HTTPException(
                400, "JD 太短 (< 50 字), 没法生成有用的投递包",
            )
        try:
            rj = manual.from_text(
                cleaned,
                title=(title or None),
                company=(company or None),
                location=(location or None),
                url=(url or None),
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        was_new, job_id = scout.ingest(store, rj)
        # Whether new or duplicate, jump straight to /apply so the user
        # always sees forward motion (matches what they were after).
        return RedirectResponse(
            f"/apply/{job_id}", status_code=303,
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
        """List companies with ≥2 jobs; show comparison form for one company.

        W15.17 — surfaces source attribution for the app limit so the UI
        can render "this is just a community estimate, let agent research"
        instead of a confident number that's likely wrong.
        """
        from ..briefs import app_limit_with_attribution

        company_groups = _list_company_groups(store)

        target_jobs: list[dict] | None = None
        limit_answer = None
        if company:
            target_jobs = _list_jobs_for_company(store, company)
            limit_answer = app_limit_with_attribution(store, company)

        return templates.TemplateResponse(
            request,
            "compare.html",
            _ctx(
                request,
                company_groups=company_groups,
                selected_company=company,
                target_jobs=target_jobs,
                target_limit=limit_answer.limit if limit_answer else None,
                limit_answer=limit_answer,  # full structured answer for UI
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

        llm = runtime._llm if runtime is not None else None
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
        """SSE endpoint that runs the harness in a thread + streams events.

        Each event becomes one ``data: {...}\\n\\n`` SSE frame. The browser-
        side EventSource (in agent.html) appends each frame to the live
        panel as it arrives. The connection closes after the loop returns.

        Implementation note: harness.run is sync (each LLM call blocks), so
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

        from ..harness import (
            HarnessDeps,
            MemoryStore,
            TriggerEvent,
            default_worldview_dir,
        )
        from ..harness import _schema as _hs
        from ..harness import run as harness_run
        _hs.init_harness_schema(store)

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
        deps = HarnessDeps(
            settings=settings, store=store,
            memory_store=MemoryStore(root=default_worldview_dir(settings)),
            llm=llm, runtime=runtime, skills=skills,
            search=_search, notifier=notifier,
            user_profile_text=profile.raw_resume_text if profile else None,
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
            # harness.run calls this from its worker thread — bridge to async queue.
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
                result = harness_run(
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

    @app.get("/apply", response_class=HTMLResponse)
    def apply_index(request: Request) -> Any:
        """W14.11: /apply (no id) — used to be a 404 that confused fresh
        users who clicked "Apply" expecting to see a job picker. Now
        redirect to /pipeline where they can pick a job (or paste one)."""
        return RedirectResponse("/pipeline", status_code=303)

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
        application_plan = build_application_plan(job)

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
                application_plan=application_plan,
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

        # W20.5 — multi-SKILL outcome attribution.
        # Pre-W20.5: only apply_assistant got credit (hardcoded skill_version=0.1.0).
        # Real chain for an application:
        #   score_match (找到岗) → tailor_resume (改简历)
        #   → apply_assistant (写自我介绍) → prepare_interview (面试准备)
        # All 4 SKILLs participated. Outcome should fan to all of them via
        # the harness_events linking SKILL run_ids ↔ job_id (written by
        # _exec_score_match for score_match and _link_skill_to_job_and_signal
        # for apply_assistant + prepare_interview).
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
                # W20.5 — find all SKILL runs that touched this job via
                # harness_events (kind 'scored' / 'apply_pack_generated' /
                # 'post_apply_pack_generated'). harness_events table may not
                # exist in fresh fixtures — init it here defensively (idempotent)
                # and fall back to apply_assistant only-attribution if so.
                try:
                    from ..harness import _schema as _hs
                    _hs.init_harness_schema(store)
                except Exception:
                    pass
                with store.connect() as conn:
                    skill_rows = conn.execute(
                        "SELECT DISTINCT json_extract(note, '$.skill_run_id') as srid, "
                        "       json_extract(note, '$.skill_name') as sn "
                        "FROM harness_events "
                        "WHERE job_id = ? AND kind IN "
                        "  ('scored', 'apply_pack_generated', 'post_apply_pack_generated') "
                        "  AND json_extract(note, '$.skill_run_id') IS NOT NULL",
                        (job_id,),
                    ).fetchall()
                    # Map skill_run_id → skill_version
                    seen_runs: set[int] = set()
                    for srid_raw, sn_raw in skill_rows:
                        if srid_raw is None:
                            continue
                        srid = int(srid_raw)
                        if srid in seen_runs:
                            continue
                        seen_runs.add(srid)
                        sn = str(sn_raw) if sn_raw else None
                        # Look up version + (if we don't have skill_name) name
                        sr = conn.execute(
                            "SELECT skill_name, skill_version "
                            "FROM skill_runs WHERE id = ?",
                            (srid,),
                        ).fetchone()
                        if not sr:
                            continue
                        skill_name = sn or str(sr[0])
                        skill_version = str(sr[1])
                        try:
                            _evo.record_app_outcome(
                                store,
                                skill_name=skill_name,
                                skill_version=skill_version,
                                skill_run_id=srid,
                                outcome=outcome,  # type: ignore[arg-type]
                                weight=1.0,
                            )
                        except Exception as e:
                            log.debug(
                                "app_outcome write for %s#%d failed: %s",
                                skill_name, srid, e,
                            )
                # Fallback if NO SKILL events found (job created from
                # user paste with no SKILL chain) — at least keep old behavior
                # of attributing to apply_assistant
                if not skill_rows:
                    _evo.record_app_outcome(
                        store, skill_name="apply_assistant",
                        skill_version="0.1.0", skill_run_id=None,
                        outcome=outcome,  # type: ignore[arg-type]
                        weight=0.5,  # lower weight since attribution is fuzzy
                    )
        except Exception as e:
            log.debug("apply outcome signal write failed: %s", e)

        # W14 联动: outcome → user_facts. Terminal outcomes (offer/rejected) are
        # high-quality preference signals — extract them as user_facts so future
        # agent runs see them in the snapshot's user_facts section. This closes
        # the loop: agent suggests → user marks outcome → fact lands → agent
        # respects in future suggestions.
        try:
            from .. import user_facts as _uf
            with store.connect() as conn:
                row = conn.execute(
                    "SELECT j.company, j.title FROM applications a "
                    "LEFT JOIN jobs j ON j.id = a.job_id WHERE a.id = ?",
                    (app_id,),
                ).fetchone()
            if row and row[0]:
                company, title = row
                fact_text = None
                kind = None
                confidence = 0.85
                if status == "offer":
                    fact_text = f"用户拿到 offer: {company} {title or ''} 岗位"
                    kind = "experience"
                    confidence = 1.0
                elif status == "rejected":
                    fact_text = (
                        f"用户被 {company} {title or ''} 岗位拒了 "
                        f"(future agent 推类似岗位时降优先级)"
                    )
                    kind = "company_signal"
                    confidence = 0.85
                elif status == "interview":
                    fact_text = f"用户进入 {company} {title or ''} 面试阶段"
                    kind = "experience"
                    confidence = 0.95
                if fact_text and kind:
                    _uf.add_fact(
                        store,
                        fact_text=fact_text, kind=kind, confidence=confidence,
                        source_skill="apply_lifecycle",
                        entities=[company] + ([title] if title else []),
                    )
        except Exception as e:
            log_mod = __import__("logging").getLogger(__name__)
            log_mod.debug("apply outcome → user_facts failed: %s", e)

        return {"app_id": app_id, "status": status}

    @app.get("/agent/runs/{run_id}", response_class=HTMLResponse)
    def agent_run_detail(request: Request, run_id: int) -> Any:
        """Read a persisted harness_runs row + render its trajectory.

        Also surfaces inbox suggestions this run created (reverse link)
        and any user_thumbs signals that fed back into evolution_signals.
        """
        # Make sure the harness schema exists — first visit to /agent/runs/{id}
        # on a fresh install can land here before any harness.run has fired.
        from ..harness import _schema as _hs
        _hs.init_harness_schema(store)
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
                "WHERE note LIKE ? AND kind IN ("
                "  'tailor_resume_generated', 'interview_prep_generated', "
                "  'project_record_saved', 'project_assessed'"
                ") "
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
        """简历微调入口页 — 选 JD + 显示 master resume + 一键 tailor.

        W14: 同时列出 data/tailored/ 里已有的 docx 文件 (按 mtime 排倒序),
        让用户能直接预览 / 下载 / 重做, 不必每次重新跑 LLM。
        """
        from pathlib import Path as _Path

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
        latest_agent_tailor = _latest_agent_tailor_result(store)

        # W14: existing tailored docx history
        existing_tailored: list[dict] = []
        tailored_dir = _Path("data/tailored")
        if tailored_dir.exists():
            for f in sorted(
                tailored_dir.glob("*.docx"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:20]:
                existing_tailored.append({
                    "filename": f.name,
                    "size_kb": f.stat().st_size // 1024,
                    "mtime": f.stat().st_mtime,
                })

        return templates.TemplateResponse(
            request, "tailor.html",
            _ctx(
                request,
                jobs=jobs,
                selected_job_id=job_id,
                master_resume=master_resume_text,
                latest_agent_tailor=latest_agent_tailor,
                existing_tailored=existing_tailored,
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

        if profile is None or not profile.source_pdf:
            return templates.TemplateResponse(
                request, "_tailor_docx_result.html",
                _ctx(request, error="没加载简历——设 OFFERGUIDE_RESUME_PDF 后重启"),
            )
        # W14.8: direct attribute access (not getattr) so pyright narrows the
        # `str | None` to `str` after the truthy check above.
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

    @app.get("/api/tailor/preview/{filename}")
    def tailor_preview(filename: str) -> Any:
        """Legacy mammoth-HTML preview path — now redirects to libreoffice PDF.

        Mammoth flattens tab/space-based multi-column layout (master 用
        \\t 实现的"学校 ... 时间"两栏) into a single line, losing the
        resume's visual structure. The libreoffice PDF path renders
        the docx 100% faithfully (multi-column, headshot, fonts all
        preserved). Keep the URL alive for any saved links / bookmarks,
        but route through the faithful renderer.
        """
        from fastapi.responses import RedirectResponse

        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(400, "invalid filename")
        if not filename.endswith(".docx"):
            raise HTTPException(400, "not a docx")
        return RedirectResponse(url=f"/api/tailor/pdf/{filename}", status_code=307)

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
            from .. import project_vault as _pv
            result = runtime.invoke(
                spec,
                {
                    "master_resume": _pv.append_to_profile_text(
                        store, profile.raw_resume_text, max_project_chars=3500,
                    ),
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

        # W15.11 — feed user reaction into evolution_signals so GEPA learns
        # what kinds of suggestions the user accepts/rejects. This is the
        # learning loop that lets agent get more selective without if-else
        # rules ("don't push X" — agent learns by seeing X get rejected).
        try:
            from ..harness import feedback as _hfb
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

        return templates.TemplateResponse(
            request, "_inbox_list.html", _ctx(request, items=[item])
        )

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
            from ..harness import feedback as _hfb
            _hfb.on_question_answered(
                store, inbox_id=item_id, option_id=option_id,
                free_text=free_text,
            )
        except Exception as _e:
            log.warning("question feedback recording failed (non-fatal): %s", _e)

        return RedirectResponse("/", status_code=303)

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

    @app.post("/api/extension/bulk_ingest", response_class=JSONResponse)
    def extension_bulk_ingest(payload: ExtensionListPayload) -> dict:
        """W15.18 — accept BOSS 推荐列表 from extension. Bulk ingest N jobs.

        用户在 BOSS 自己刷岗位 → 扩展抓推荐列表 → 一键 sync 整页 N 个岗位
        到 OfferGuide. Agent 拿到后自动 score / 排序 / 主动通知.

        这是 W15.17 用户反馈"没自动找岗位项目就没用"的回应 — 因为国内
        BOSS/牛客 反爬严重不能 zero-touch crawl, 走"用户开 BOSS 我帮 sync"
        的合法路径 (用户自己账号, 自己看到的页面).

        每个 item 行 description 是空的 (列表页不展开 JD 全文), 只入
        title/company/salary/location/tags. 后续用户点感兴趣的可去 JD 详情
        页用 /api/extension/ingest 补 description.
        """
        items = payload.items or []
        if not items:
            raise HTTPException(400, "items 不能为空")
        if len(items) > 200:
            raise HTTPException(400, f"items 太多 ({len(items)}, max 200)")

        inserted = 0
        duplicate = 0
        new_job_ids: list[int] = []
        skipped_reasons: list[str] = []

        for item in items:
            if not (item.title and item.company):
                skipped_reasons.append(
                    f"缺 title/company: {item.title!r} / {item.company!r}"
                )
                continue
            # description 在列表模式是空的, 用 title + company + tags 当 raw_text
            # 这样 dedup 还能 work (scout.ingest 用 url + content_hash)
            raw_text_parts = [
                f"# {item.title}",
                f"公司: {item.company}",
            ]
            if item.salary:
                raw_text_parts.append(f"薪资: {item.salary}")
            if item.location:
                raw_text_parts.append(f"地点: {item.location}")
            if item.tags:
                raw_text_parts.append("标签: " + ", ".join(item.tags))
            raw_text_parts.append(
                "(从 BOSS 推荐列表抓的, 详情未展开 — 后续点 JD 详情可补全)"
            )
            raw_text = "\n".join(raw_text_parts)

            extras: dict = {"from_list_capture": True}
            if item.salary:
                extras["salary"] = item.salary
            if item.tags:
                extras["tags"] = item.tags

            rj = RawJob(
                source="boss_extension_list",
                source_id=_extract_boss_id(item.url) if item.url else None,
                url=item.url or None,
                title=item.title,
                company=item.company,
                location=item.location,
                raw_text=raw_text,
                extras=extras,
            )
            try:
                is_new, job_id = scout.ingest(store, rj)
            except Exception as e:
                skipped_reasons.append(f"{item.company}/{item.title}: {e}")
                continue
            if is_new:
                inserted += 1
                new_job_ids.append(job_id)
            else:
                duplicate += 1

        # Fire harness event so agent's next wake notices new jobs to score
        try:
            from ..harness import fire_event
            if inserted > 0:
                fire_event(
                    store,
                    event_kind="user_paste_jd",  # 借用现有 event kind
                    detail={
                        "note": f"BOSS 推荐列表 sync — {inserted} 新 + {duplicate} 重复",
                        "from_extension": True,
                        "page_url": payload.page_url,
                    },
                )
        except Exception:
            pass  # event firing is nice-to-have; ingest must succeed

        return {
            "inserted": inserted,
            "duplicate": duplicate,
            "total": len(items),
            "job_ids": new_job_ids[:30],
            "skipped_reasons": skipped_reasons[:5],
        }

    # ────────────────── W15.20 真半自动: 内联评分 + 一键开场白 ──────────────────
    #
    # 这两个 endpoint 是为 content script 设计的 — 用户在 BOSS 详情页浏览时
    # OfferGuide 浮窗自动调 score_inline (3-8s) 给即时评分; 用户点"写开场白"
    # 触发 greeting (5-10s) 把 200 字开场白写到剪贴板, 用户审核后粘到 BOSS
    # 沟通框. **不自动发送** — 半自动 = 用户在环.

    @app.post("/api/extension/score_inline", response_class=JSONResponse)
    async def extension_score_inline(payload: ExtensionJDPayload) -> Any:
        """W15.20 — content script 实时打分. 比 evaluate_job 更轻 (跳过 tailor).

        典型场景: 用户在 BOSS JD 详情页, content_script 自动抓 JD 调这个,
        3-8s 内拿到 score → 在页面右上角浮窗显示. 不阻塞用户浏览.

        返回 slim payload (score + top 3 gap + 简短 verdict), 不返回完整
        tailor (那是用户点"写开场白"时才需要).
        """
        if not settings.deepseek_api_key:
            return _ext_response(400, {"error": "需要先配 LLM key"})
        if runtime is None:
            return _ext_response(400, {"error": "SkillRuntime 未初始化"})
        if not (payload.description or "").strip():
            return _ext_response(400, {"error": "JD 描述为空"})
        if profile is None or not profile.raw_resume_text:
            return _ext_response(400, {"error": "未配简历, 去 /profile 上传"})

        from ..harness import (
            HarnessDeps,
            MemoryStore,
            default_worldview_dir,
        )
        from ..harness import _schema as _hs
        from ..harness.evaluate import _fetch_and_ingest, _invoke_skill, _safe_float
        _hs.init_harness_schema(store)

        deps = HarnessDeps(
            settings=settings, store=store,
            memory_store=MemoryStore(root=default_worldview_dir(settings)),
            llm=LLMClient(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                default_model=settings.default_model,
            ),
            runtime=runtime, skills=skills,
            user_profile_text=profile.raw_resume_text,
            notifier=notifier,
        )

        # Build raw text from extension payload
        raw_parts = [payload.description]
        if payload.title:
            raw_parts.insert(0, f"# {payload.title}")
        if payload.company:
            raw_parts.append(f"公司: {payload.company}")
        if payload.salary:
            raw_parts.append(f"薪资: {payload.salary}")
        if payload.tags:
            raw_parts.append("标签: " + ", ".join(payload.tags))
        raw_text = "\n".join(raw_parts).strip()

        import asyncio as _asyncio
        import time as _time
        t0 = _time.monotonic()
        try:
            # Reuse evaluate's fetch+ingest using the JD text path
            job_id, fetch_err = await _asyncio.to_thread(
                _fetch_and_ingest, raw_text, deps,
                company_hint=payload.company or "",
                title_hint=payload.title or "",
            )
            if fetch_err:
                return _ext_response(400, {"error": fetch_err})

            # Run only score_match (skip tailor for speed)
            score_spec = deps.find_skill("score_match")
            if score_spec is None:
                return _ext_response(500, {"error": "score_match SKILL 缺失"})

            # W15.22 — verified inputs/outputs against score_match SKILL.md
            # (inputs: job_text, user_profile; outputs: probability, reasoning,
            # dimensions, deal_breakers). Pre-W15.22 used wrong keys → ValueError
            # → 502 every call.
            from ..harness.tools import _format_jd_for_skill
            job_row = {
                "title": payload.title, "company": payload.company or "",
                "location": payload.location or "",
                "raw_text": payload.description,
            }
            sr = await _asyncio.to_thread(
                _invoke_skill, deps, score_spec,
                inputs={
                    "job_text": _format_jd_for_skill(job_row)[:4000],
                    "user_profile": profile.raw_resume_text[:4000],
                },
            )
            if sr is None or sr.parsed is None:
                raw_str = sr.raw_text[:200] if sr else "(no response)"
                return _ext_response(502, {
                    "error": "score 解析失败",
                    "job_id": job_id,
                    "raw": raw_str,
                })

            p = sr.parsed
            # SKILL outputs probability ∈ [0, 1]; convert to 0-100 for UI.
            prob_raw = _safe_float(p.get("probability"))
            score_val = (prob_raw * 100.0) if prob_raw is not None else None
            # SKILL outputs deal_breakers (hard-stop issues); use as top gaps.
            breakers = p.get("deal_breakers") or []
            if not isinstance(breakers, list):
                breakers = []
            top_gaps = [str(g)[:80] for g in breakers[:3]]
            # No strengths field in SKILL output — derive from dimensions
            dims = p.get("dimensions") or {}
            top_strengths: list[str] = []
            if isinstance(dims, dict):
                for dim_name, dim_val in dims.items():
                    try:
                        if float(dim_val) >= 0.7:
                            top_strengths.append(f"{dim_name}: {round(float(dim_val) * 100)}")
                    except (TypeError, ValueError):
                        continue
                top_strengths = top_strengths[:3]

            # Color-coded verdict for the badge
            if score_val is None:
                verdict = "评分缺失"
                color = "gray"
            elif score_val >= 30:
                # 30% reply rate = "强 fit" (BOSS 行业基线 < 5% for cold apply)
                verdict = "值得投"
                color = "green"
            elif score_val >= 15:
                verdict = "可以试"
                color = "yellow"
            else:
                verdict = "性价比低"
                color = "red"

            duration_ms = int((_time.monotonic() - t0) * 1000)
            return _ext_response(200, {
                "job_id": job_id,
                "score": round(score_val, 1) if score_val is not None else None,
                "probability": prob_raw,  # raw 0-1 for callers that want it
                "verdict": verdict,
                "color": color,
                "top_strengths": top_strengths,
                "top_gaps": top_gaps,
                "reasoning": (p.get("reasoning") or "")[:600],
                "dimensions": dims if isinstance(dims, dict) else {},
                "duration_ms": duration_ms,
                "cost_usd": round(sr.cost_usd or 0.0, 5),
            })
        finally:
            with contextlib.suppress(Exception):
                if deps.llm:
                    deps.llm.close()

    @app.post("/api/extension/greeting", response_class=JSONResponse)
    async def extension_greeting(request: Request) -> Any:
        """W15.20 — 一键生成 BOSS 沟通开场白 (200 字内, 用户审核后粘贴).

        Input: {job_id: int} 或 {jd_text, title, company} (前者更快, 后者
        独立可用). 返回开场白纯文本, 由 content_script 写到用户剪贴板.

        **不自动发送** — 用户必须自己粘到 BOSS 输入框 + 改抬头 + 点发送.
        OfferGuide 只代写文案, 决策权在用户.
        """
        if not settings.deepseek_api_key:
            return _ext_response(400, {"error": "需要先配 LLM key"})
        if profile is None or not profile.raw_resume_text:
            return _ext_response(400, {"error": "未配简历, 去 /profile 上传"})

        body = await request.json()
        job_id = body.get("job_id")
        jd_text = (body.get("jd_text") or "").strip()
        company = (body.get("company") or "").strip()
        title = (body.get("title") or "").strip()

        if job_id:
            with store.connect() as conn:
                row = conn.execute(
                    "SELECT title, company, raw_text FROM jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
            if row is None:
                return _ext_response(404, {"error": f"job#{job_id} 不存在"})
            title = title or (row[0] or "")
            company = company or (row[1] or "")
            jd_text = jd_text or (row[2] or "")

        if not jd_text:
            return _ext_response(400, {"error": "缺 jd_text 或 job_id"})

        # 直接调 LLM, 不走 SKILL — 这是单点小任务
        from ..llm import BudgetExceeded, enforce_daily_budget
        try:
            enforce_daily_budget(store)
        except BudgetExceeded as e:
            return _ext_response(429, {"error": str(e)})

        prompt = (
            "你是一个帮国内校招求职者写 BOSS 直聘开场白的助手. "
            "目标: 让 HR 愿意打开简历, 不让人觉得是模板. "
            "约束:\n"
            "- 200 字以内 (含标点)\n"
            "- 第一句别说'您好' — 直接点对方关注的事\n"
            "- 中段 1 个具体能匹配 JD 的项目/经历点 (从候选简历挑最对口的)\n"
            "- 末句 1 个轻问句 (不要『期待回复』『感谢』这种)\n"
            "- 不写薪资 / 工作时间 / 是否能转正这些事 (太敏感, 第一条别问)\n"
            "- 不要 emoji\n"
            "- 全中文\n\n"
            f"## 候选人简历 (摘选):\n{profile.raw_resume_text[:3000]}\n\n"
            f"## 目标岗位\n职位: {title}\n公司: {company}\n"
            f"JD:\n{jd_text[:2500]}\n\n"
            "直接输出开场白正文, 不要任何前言/解释/markdown 标记."
        )

        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        try:
            import asyncio as _asyncio
            resp = await _asyncio.to_thread(
                llm.chat,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                extra={"max_tokens": 400},
            )
        finally:
            with contextlib.suppress(Exception):
                llm.close()

        text = (resp.content or "").strip()
        # Trim if model added "好的, 这是开场白:" prefix
        for prefix in ("开场白:", "开场白：", "正文:", "正文："):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Hard cap at 240 chars (BOSS limit + buffer)
        if len(text) > 240:
            text = text[:237] + "..."

        # Try to record it as an event for trail
        try:
            with store.connect() as conn:
                conn.execute(
                    "INSERT INTO harness_events (kind, job_id, note, source) "
                    "VALUES (?, ?, ?, 'extension')",
                    (
                        "greeting_drafted",
                        job_id if isinstance(job_id, int) else None,
                        f"BOSS 开场白 ({len(text)} 字), 由 extension 拉",
                    ),
                )
                conn.commit()
        except Exception:
            pass  # event recording is nice-to-have

        return _ext_response(200, {
            "greeting": text,
            "length": len(text),
            "tip": "已生成. content_script 会写到剪贴板, 粘到 BOSS 沟通框, 改改抬头再发.",
            "cost_usd": round(resp.cost_usd or 0.0, 5),
        })

    @app.post("/api/extension/probe_dom", response_class=JSONResponse)
    async def extension_probe_dom(request: Request) -> Any:
        """W15.21 — DOM 校准探针. 用户在 BOSS 沟通框打开时点 content_script
        浮窗里的"🔍 抓 DOM"按钮, 把当前页面 + 沟通框区域 outerHTML 发回这里
        存档, 之后我们靠这些样本写出真正的 selector chain.

        这是元方法: 我没法在没有真账号的情况下凭空猜 BOSS 的 React 组件
        class 名 (它们是 hash 化的, 每次发版可能变). 让用户帮我抓真样本.

        Body: {url, snippet_kind: 'chat_box'|'job_card'|'send_button',
               outer_html: str, captured_at: ISO}
        """
        body = await request.json()
        url = (body.get("url") or "")[:500]
        kind = (body.get("snippet_kind") or "unknown")[:80]
        html = (body.get("outer_html") or "")
        if not html:
            return _ext_response(400, {"error": "outer_html 不能为空"})
        if len(html) > 200_000:
            return _ext_response(400, {"error": "html 太大 (max 200KB)"})

        # Save under .offerguide/probes/ for later inspection
        from pathlib import Path
        probe_dir = Path(".offerguide/probes")
        probe_dir.mkdir(parents=True, exist_ok=True)
        from datetime import UTC
        from datetime import datetime as _dt
        stamp = _dt.now(UTC).strftime("%Y%m%d_%H%M%S")
        fname = f"{stamp}_{kind}.html"
        try:
            (probe_dir / fname).write_text(
                f"<!-- url: {url} | kind: {kind} | captured_at: {body.get('captured_at')} -->\n"
                + html,
                encoding="utf-8",
            )
        except OSError as e:
            return _ext_response(500, {"error": f"写文件失败: {e}"})

        # Also record an event for trail
        try:
            with store.connect() as conn:
                conn.execute(
                    "INSERT INTO harness_events (kind, note, source) "
                    "VALUES (?, ?, 'extension')",
                    ("dom_probe", f"BOSS DOM probe ({kind}, {len(html)} bytes) → {fname}"),
                )
                conn.commit()
        except Exception:
            pass

        return _ext_response(200, {
            "saved_as": fname,
            "bytes": len(html),
            "tip": "感谢帮忙抓样本! 后续 W15.22 会用这些校准 selector",
        })

    return app


class ExtensionJDPayload(BaseModel):
    """Request body from the Boss browser extension (单个 JD 详情)."""

    url: str | None = None
    title: str = "(untitled)"
    company: str | None = None
    location: str | None = None
    salary: str | None = None
    description: str
    tags: list[str] = []


class ExtensionListItem(BaseModel):
    """W15.18 — 1 个岗位卡片 (从 BOSS 推荐列表抓的)."""

    url: str | None = None
    title: str
    company: str
    location: str | None = None
    salary: str | None = None
    tags: list[str] = []
    description: str = ""  # 列表页通常没展开, 默认空


class ExtensionListPayload(BaseModel):
    """W15.18 — 整页推荐列表的批量 ingest payload."""

    page_url: str | None = None
    items: list[ExtensionListItem]
    captured_at: str | None = None


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
        d = json_dumps  # ensure module imported; if not, parse manually
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
    from ..harness import _schema as _hs

    try:
        _hs.init_harness_schema(store)
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT id, kind, job_id, note, "
                "(julianday('now') - created_at) * 86400 AS age_seconds "
                "FROM harness_events "
                "WHERE kind IN ("
                "  'tailor_resume_generated', 'interview_prep_generated', "
                "  'project_record_saved', 'project_assessed'"
                ") "
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


def _latest_agent_tailor_result(store: Store) -> dict[str, Any] | None:
    """Latest tailor_resume artifact produced by Agent Chat, for /tailor."""
    from ..harness import _schema as _hs

    try:
        _hs.init_harness_schema(store)
        with store.connect() as conn:
            rows = conn.execute(
                "SELECT job_id, note FROM harness_events "
                "WHERE kind = 'tailor_resume_generated' "
                "ORDER BY created_at DESC, id DESC LIMIT 20"
            ).fetchall()
    except Exception:
        return None

    for job_id, note in rows:
        try:
            payload = json_loads(note or "{}")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        srid = payload.get("skill_run_id")
        if isinstance(srid, str) and srid.isdigit():
            srid = int(srid)
        if not isinstance(srid, int):
            continue
        try:
            with store.connect() as conn:
                row = conn.execute(
                    "SELECT output_json FROM skill_runs "
                    "WHERE id = ? AND skill_name = 'tailor_resume'",
                    (srid,),
                ).fetchone()
                job_row = conn.execute(
                    "SELECT title, company FROM jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
        except Exception:
            continue
        if row is None:
            continue
        try:
            tailored = json_loads(row[0] or "{}")
        except Exception:
            continue
        if not isinstance(tailored, dict):
            continue
        return {
            "run_id": srid,
            "job_title": (job_row[0] if job_row else "") or f"job#{job_id}",
            "job_company": (job_row[1] if job_row else "") or "",
            "tailored": tailored,
        }
    return None


def _artifact_from_event_row(row: Any) -> dict[str, Any] | None:
    event_id, kind, job_id, note = row[0], row[1], row[2], row[3]
    try:
        payload = json_loads(note or "{}")
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    if kind == "tailor_resume_generated":
        srid = payload.get("skill_run_id")
        return {
            "event_id": event_id,
            "kind": kind,
            "label": "简历微调",
            "title": f"job#{job_id} · skill_run#{srid}",
            "view": payload.get("view") or "/tailor",
            "detail": "Agent 生成了 truthful change_log 和定向简历产物",
        }
    if kind == "interview_prep_generated":
        srid = payload.get("skill_run_id")
        return {
            "event_id": event_id,
            "kind": kind,
            "label": "面试准备",
            "title": f"job#{job_id} · {payload.get('round') or '面试'} · skill_run#{srid}",
            "view": payload.get("view") or "/reflect",
            "detail": "Agent 生成了面试重点、预测问题和弱点清单",
        }
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

    # W14.12 — start the autonomous scheduler in-process so users get
    # ambient agent behavior just by running `python -m offerguide.ui.web`.
    # Previously the scheduler was a separate `python -m offerguide.autonomous`
    # process; users typically forgot to start it, leading to "agent doesn't
    # do anything" complaints. We spawn it on a background thread that lives
    # for the web process's lifetime. Disable with OFFERGUIDE_NO_SCHEDULER=1.
    sched_status = "disabled (OFFERGUIDE_NO_SCHEDULER=1)"
    if os.environ.get("OFFERGUIDE_NO_SCHEDULER") != "1" and settings.deepseek_api_key:
        try:
            import threading

            from ..autonomous.scheduler import build_agent_wake_scheduler

            def _run_scheduler() -> None:
                try:
                    sched = build_agent_wake_scheduler(settings=settings)
                    sched.run_blocking()  # AutonomousScheduler API
                except Exception as e:
                    log.exception("scheduler thread crashed: %s", e)

            t = threading.Thread(target=_run_scheduler, daemon=True, name="og-scheduler")
            t.start()
            sched_status = "running (in-process thread)"
        except Exception as e:
            sched_status = f"failed to start: {e}"

    print(f"\n✦ OfferGuide UI on http://{settings.web_host}:{settings.web_port}")
    print(f"  resume    = {settings.resume_pdf or '(none — set OFFERGUIDE_RESUME_PDF)'}")
    print(
        f"  llm       = {'configured' if settings.deepseek_api_key else 'NOT configured (set DEEPSEEK_API_KEY)'}"
    )
    print(f"  notify    = {settings.notify_channel} ({'ready' if settings.notify_ready() else 'fallback console'})")
    print(f"  scheduler = {sched_status}")
    uvicorn.run(app, host=settings.web_host, port=settings.web_port, log_level="info")


# `_` to silence unused-symbol lint in linters that don't read entry points
_ = RedirectResponse


if __name__ == "__main__":
    main()
