"""APScheduler-based autonomous task scheduler.

Architecture (borrowed from APScheduler's BlockingScheduler pattern):

- One scheduler instance owns the job registry
- Each job is a plain Python callable; the scheduler invokes it on a
  cron-like trigger with retry semantics
- Jobs receive a ``JobContext`` (store, llm, search, notifier) so they
  don't need to rebuild dependencies each run
- Jobs are *idempotent* — running a job twice in a row is safe
  (silence_check is idempotent by design via the
  ``_max_alerted_threshold`` check; corpus_refresh dedups via
  content_hash; brief_update is upsert)

Why not just cron? Two reasons:
1. We want the scheduler to carry our DB / LLM / search instances so
   each tick is cheap (no re-init)
2. APScheduler gives us misfire_grace_time + max_instances so a slow
   job doesn't trigger a re-entrant run

Run as a long-lived process: ``python -m offerguide.autonomous``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..llm import LLMClient
from ..memory import Store

log = logging.getLogger(__name__)


@dataclass
class JobContext:
    """Dependencies passed to every scheduled job. Built once at
    scheduler start; jobs treat it as read-only state."""

    settings: Settings
    store: Store
    llm: LLMClient | None
    """None when DEEPSEEK_API_KEY isn't set — jobs requiring LLM should
    short-circuit gracefully in that case."""

    search: Any | None = None
    """SearchBackend (from agentic.search) — None to disable corpus
    refresh."""

    notifier: Any | None = None

    # Optional — populated when the user has configured a resume so
    # SKILL-using jobs (discover_jobs auto-eval, etc.) can run.
    runtime: Any | None = None
    """SkillRuntime — None when LLM is not configured. Built once at
    scheduler start; passed to jobs that auto-invoke SKILLs."""

    skills: list[Any] = field(default_factory=list)
    """List of SkillSpec discovered at boot. Empty list means SKILL
    discovery wasn't run (e.g. headless tests)."""

    user_profile_text: str | None = None
    """Resume text loaded from OFFERGUIDE_RESUME_PDF. None when no
    resume is configured — jobs that need it should skip gracefully."""


@dataclass
class JobSpec:
    """One scheduled job + its trigger configuration."""

    name: str
    func: Callable[[JobContext], Any]
    """Callable that takes a ``JobContext`` and returns a dict of
    counters. Returning is for logs only — APScheduler ignores it."""

    trigger: str
    """``cron`` | ``interval`` — APScheduler trigger type."""

    trigger_kwargs: dict[str, Any] = field(default_factory=dict)
    """e.g. ``{'hour': 9, 'minute': 0}`` for a daily 09:00 trigger."""

    misfire_grace_time_s: int = 300
    """If the scheduler missed the trigger by < this many seconds (e.g.
    laptop was asleep), still run when it wakes up."""

    max_instances: int = 1
    """Don't fire concurrent runs of the same job."""


class AutonomousScheduler:
    """Wraps APScheduler with our JobSpec / JobContext sugar.

    Use as a context manager so shutdown is guaranteed:

        with AutonomousScheduler(ctx) as sched:
            sched.add(SILENCE_CHECK_JOB)
            sched.add(BRIEF_UPDATE_JOB)
            sched.run_blocking()  # blocks until SIGINT
    """

    def __init__(self, ctx: JobContext) -> None:
        self.ctx = ctx
        # Lazy import so the autonomous extra is only required when this
        # class is actually instantiated.
        from apscheduler.schedulers.blocking import BlockingScheduler

        self._scheduler = BlockingScheduler(
            timezone="Asia/Shanghai",  # most relevant TZ for user
        )
        self._jobs: list[JobSpec] = []

    def add(self, spec: JobSpec) -> None:
        """Register a job."""
        self._jobs.append(spec)
        self._scheduler.add_job(
            self._wrap(spec),
            trigger=spec.trigger,
            id=spec.name,
            name=spec.name,
            misfire_grace_time=spec.misfire_grace_time_s,
            max_instances=spec.max_instances,
            replace_existing=True,
            **spec.trigger_kwargs,
        )

    def _wrap(self, spec: JobSpec):
        """Wrap a job func in error handling + structured logging +
        daemon_runs persistence (so /dashboard can show health)."""
        def _run() -> None:
            log.info("autonomous job start: %s", spec.name)
            run_id = self._record_start(spec.name)
            try:
                result = spec.func(self.ctx)
                self._record_end(run_id, status="ok", summary=result)
                log.info("autonomous job done: %s → %s", spec.name, result)
            except Exception as e:
                self._record_end(
                    run_id, status="error", summary={}, error_text=str(e)[:500]
                )
                log.exception("autonomous job FAILED: %s: %s", spec.name, e)

        _run.__name__ = f"_run_{spec.name}"
        return _run

    def _record_start(self, job_name: str) -> int | None:
        """Insert daemon_runs row, return its id. Failure is silent —
        we don't want telemetry to crash the job."""
        try:
            import json as _json
            with self.ctx.store.connect() as conn:
                cur = conn.execute(
                    "INSERT INTO daemon_runs(job_name, status, summary_json) "
                    "VALUES (?, 'running', ?) RETURNING id",
                    (job_name, _json.dumps({})),
                )
                return int(cur.fetchone()[0])
        except Exception:
            return None

    def _record_end(
        self,
        run_id: int | None,
        *,
        status: str,
        summary: Any,
        error_text: str | None = None,
    ) -> None:
        if run_id is None:
            return
        try:
            import json as _json
            payload = _json.dumps(
                summary if isinstance(summary, dict) else {"result": str(summary)},
                ensure_ascii=False,
                default=str,
            )[:4000]
            with self.ctx.store.connect() as conn:
                conn.execute(
                    "UPDATE daemon_runs SET status = ?, ended_at = julianday('now'), "
                    "summary_json = ?, error_text = ? WHERE id = ?",
                    (status, payload, error_text, run_id),
                )
        except Exception:
            pass

    def run_blocking(self) -> None:
        """Block forever. Ctrl-C / SIGTERM stops it."""
        log.info(
            "autonomous scheduler starting with %d job(s): %s",
            len(self._jobs),
            ", ".join(j.name for j in self._jobs),
        )
        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("autonomous scheduler stopping (signal)")
            self.shutdown()
            raise

    def trigger_once(self, name: str) -> Any:
        """Run job *name* once immediately, returning its result.

        Used by the CLI ``run-once`` subcommand and by tests. Routes
        through ``_wrap`` so daemon_runs telemetry records the
        invocation just like a real scheduled tick — otherwise
        run-once would leave a hole in the health dashboard.
        """
        spec = next((j for j in self._jobs if j.name == name), None)
        if spec is None:
            raise KeyError(f"no job registered with name {name!r}")
        # Run through _wrap so daemon_runs gets the row, but capture
        # the result for the CLI / test caller (the wrapped runner
        # otherwise discards return value).
        run_id = self._record_start(name)
        try:
            result = spec.func(self.ctx)
            self._record_end(run_id, status="ok", summary=result)
            return result
        except Exception as e:
            self._record_end(
                run_id, status="error", summary={}, error_text=str(e)[:500],
            )
            raise

    def list_jobs(self) -> list[str]:
        return [j.name for j in self._jobs]

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def __enter__(self) -> AutonomousScheduler:
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()


# ── W15: harness-driven wake ────────────────────────────────────────
# W14.18 design: cron heartbeat → W14 central AgentLoop with a giant
# 3-tier prompt (`_AGENT_WAKE_GOAL`) that pre-organized tools for the
# model. User feedback (verbatim, W15 design session):
# > "不能堆叠过程式逻辑暴力模拟智能 — 庞大的规则树、节点图、链式提示词
# >  瀑布流 — 然后祈祷胶水代码涌现自主行为. Agency 是学出来的, 不是编出来的.
# >  我们做的是 harness, 心智是模型自己本身."
#
# W15 replaces that with the harness. Cron now does 2 things at each tick:
# 1. ``poll_pending`` — process scheduled wakes (agent's own self-scheduling)
#    and lifecycle events (user_paste_jd / marked_applied / etc.)
# 2. If nothing pending, fire a heartbeat trigger as a fallback safety net
#
# The agent itself decides everything else (what to do, when to next wake,
# what to update in worldview) via the agent runtime loop.


def build_agent_wake_scheduler(
    *,
    settings: Settings | None = None,
    cron_kwargs: dict[str, Any] | None = None,
) -> AutonomousScheduler:
    """Build a scheduler with ONE job: poll pending triggers + harness wake.

    Cron is **fallback** — the agent self-schedules via `schedule_next_wake`.
    Cron tick handles: scheduled wakes due, fresh user lifecycle events,
    or (if neither) a heartbeat poll so we don't drift > 1 hour out of sync.

    ``cron_kwargs`` overrides the default trigger (hourly, 08:00-22:00).
    """
    from ..agentic.search import build_default_search
    from ..profile import load_resume_pdf
    from ..skills import SkillRuntime, discover_skills
    from ..ui.notify import make_notifier

    settings = settings or Settings.from_env()
    store = Store(settings.db_path)
    store.init_schema()

    llm: LLMClient | None = None
    if settings.deepseek_api_key:
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )

    search = None
    try:
        search = build_default_search()
    except Exception as e:
        log.warning("search backend init failed: %s", e)

    from pathlib import Path
    skills_root = Path(__file__).parent.parent / "skills"
    skills = []
    try:
        skills = discover_skills(skills_root)
    except Exception as e:
        log.warning("skill discovery failed: %s", e)

    runtime = None
    if llm is not None:
        runtime = SkillRuntime(llm=llm, store=store)

    profile = None
    if settings.resume_pdf:
        try:
            profile = load_resume_pdf(settings.resume_pdf)
        except Exception as e:
            log.warning("resume load failed: %s", e)

    master_resume = profile.raw_resume_text if profile else None
    notifier = make_notifier(settings)

    ctx = JobContext(
        settings=settings, store=store, llm=llm,
        search=search, notifier=notifier,
        runtime=runtime, skills=skills,
        user_profile_text=master_resume,
    )

    def _wake_agent_job(_jc: JobContext) -> dict[str, Any]:
        """W15: poll pending triggers, run harness for each. Heartbeat fallback.

        Returns a summary the scheduler logs / surfaces in /debug.
        """
        if llm is None:
            return {"skipped": "no LLM configured"}

        # Lazy import to avoid hard dep at import time
        from ..agent_runtime import (
            AgentRuntimeDeps,
            MemoryStore,
            default_worldview_dir,
            make_cron_heartbeat,
            poll_pending,
        )
        from ..agent_runtime import _schema as _harness_schema
        from ..agent_runtime import run as agent_runtime_run

        _harness_schema.init_agent_runtime_schema(store)
        wdir = default_worldview_dir(settings)
        memory_store = MemoryStore(root=wdir)

        deps = AgentRuntimeDeps(
            settings=settings,
            store=store,
            memory_store=memory_store,
            llm=llm,
            runtime=runtime,
            skills=skills,
            search=search,
            notifier=notifier,
            user_profile_text=master_resume,
        )

        pending = poll_pending(store)
        runs: list[dict[str, Any]] = []
        if pending:
            for pt in pending[:5]:  # cap per-tick to bound cost
                # Bug 7 fix: cleanup MUST run regardless of agent_runtime_run
                # success/failure. Otherwise a scheduled wake that reliably
                # crashes the agent gets re-polled forever (infinite cost
                # loop). We track per-row failures via daemon_runs telemetry.
                run_ok = False
                run_summary: dict[str, Any] = {"source": pt.source}
                try:
                    res = agent_runtime_run(trigger=pt.trigger_event, deps=deps)
                    run_ok = True
                    run_summary.update({
                        "run_id": res.run_id,
                        "iterations": res.iterations,
                        "finish": res.finish_reason,
                        "cost_usd": round(res.cost_usd, 4),
                    })
                except Exception as e:
                    log.exception("pending trigger failed: %s", e)
                    run_summary["error"] = str(e)[:200]
                finally:
                    runs.append(run_summary)
                    # Always run cleanup (e.g. mark scheduled_wake fired_at)
                    # so the same trigger doesn't re-fire next tick. If
                    # cleanup itself fails, log + move on.
                    if pt.cleanup is not None:
                        try:
                            pt.cleanup()
                        except Exception:
                            log.exception(
                                "cleanup failed for source=%s (run_ok=%s)",
                                pt.source, run_ok,
                            )
        else:
            # Heartbeat fallback: agent looks around, no-ops if nothing to do
            res = agent_runtime_run(
                trigger=make_cron_heartbeat(), deps=deps,
            )
            runs.append({
                "source": "cron_heartbeat",
                "run_id": res.run_id,
                "iterations": res.iterations,
                "finish": res.finish_reason,
                "cost_usd": round(res.cost_usd, 4),
            })
        return {"runs": runs, "pending_count": len(pending)}

    # Hourly during waking hours (08:00-22:00). Frequency was already
    # turned up in W14.18 because per-tick cost is now bounded by the
    # agent's own decisioning ("nothing to do → done quickly").
    cron_kwargs = cron_kwargs or {"hour": "8-22"}
    wake_job = JobSpec(
        name="wake_agent",
        func=_wake_agent_job,
        trigger="cron",
        trigger_kwargs=cron_kwargs,
        misfire_grace_time_s=600,
        max_instances=1,
    )
    sched = AutonomousScheduler(ctx)
    sched.add(wake_job)
    return sched


# ── Backward-compat alias (deprecated) ────────────────────────────
# W15: removed `_discover_via_search_job` and `_auto_score_via_daemon`
# helpers — they were dead code (not registered as jobs after W14.18
# collapsed 3 daemons → 1). Discovery now happens via the agent's
# own `discover_jobs` tool inside the agent runtime loop.


def build_default_scheduler(
    *,
    settings: Settings | None = None,
) -> AutonomousScheduler:
    """W13.2 redirect: returns the agent-wake scheduler.

    The old 7-daemon design has been retired. Existing callers that
    expected ``build_default_scheduler`` get the new architecture.
    """
    log.info(
        "build_default_scheduler is deprecated since W13.2; "
        "use build_agent_wake_scheduler() directly"
    )
    return build_agent_wake_scheduler(settings=settings)
