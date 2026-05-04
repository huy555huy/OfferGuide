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


# ── W13.2: agent-wake scheduler ────────────────────────────────────
# The W4-W12 architecture registered 7 hardcoded daemons (discover at 06:30,
# enrich at 06:45, classify at 07:00...). Each had a fixed schedule + did its
# work blindly regardless of system state.
#
# W13.2 replaces that with ONE job: ``wake_agent``. It fires periodically
# (default: every 4 hours from 08:00 to 22:00) and runs the central AgentLoop
# with a "巡检" goal. The agent reads the snapshot (jobs queue, applications,
# user_facts, recent runs) and decides which maintenance tools to call.
#
# This is the architectural difference between "cron-driven" and "agent-driven":
# the agent SEES STATE before deciding. Cron sees nothing — it just runs.


_AGENT_WAKE_GOAL = """你被定时唤醒。当下时刻不一定有事可做。

先看 snapshot 想这两件事:
- 用户此刻可能在干啥 / 在意啥? (看时间、用户活跃度、user_facts 里最近变化)
- 系统里有没有真值得现在处理的事? (不是"数字 > 0", 是"再不处理用户会损失")

然后做你判断该做的, **包括"什么也不做"**。

如果你决定动手, 给 final 时说清楚:
- 做了啥 + 它对用户有什么实际价值
- 跳过了啥 + 为啥跳过 (说出你的判断, 不是"今天没必要")

如果你决定 lay low, final 写一句:
- 当前状态简评 (用户能一眼看懂)
- 为什么现在不动 (节奏 / 优先级 / 等待信号)

记住: 一次"什么也不做但判断准确"的唤醒, 比一次"忙忙叨叨调 5 个工具但没价值"的唤醒, critic 评分会高得多。
"""


def build_agent_wake_scheduler(
    *,
    settings: Settings | None = None,
    cron_kwargs: dict[str, Any] | None = None,
) -> AutonomousScheduler:
    """Build a scheduler with ONE job: wake the W13 central agent loop.

    The agent decides what maintenance to do; we just provide the heartbeat.

    ``cron_kwargs`` overrides the default trigger (every 4 hours, 08-22).
    Pass e.g. ``{'hour': '*/2'}`` to run every 2 hours instead.
    """
    from ..agent.loop import AgentLoop
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
        """The single job: wake the central agent + let it decide what to do."""
        if llm is None:
            return {"skipped": "no LLM configured"}
        if runtime is None:
            return {"skipped": "no SkillRuntime"}

        agent = AgentLoop(
            llm=llm, runtime=runtime, store=store,
            skills=skills, master_resume_text=master_resume,
            max_iterations=8, critic_enabled=True,
        )
        result = agent.run(
            goal=_AGENT_WAKE_GOAL,
            trigger_kind="cron_wake",
        )
        return {
            "agent_run_id": result.run_id,
            "iterations": result.iterations,
            "critic_score": result.critic_score,
            "latency_s": round(result.latency_ms / 1000.0, 1),
            "final": (result.final_answer or "")[:300],
        }

    cron_kwargs = cron_kwargs or {"hour": "8-22/4"}  # 08:00, 12:00, 16:00, 20:00
    wake_job = JobSpec(
        name="wake_agent",
        func=_wake_agent_job,
        trigger="cron",
        trigger_kwargs=cron_kwargs,
        misfire_grace_time_s=600,  # 10 min grace if laptop was sleeping
        max_instances=1,
    )

    sched = AutonomousScheduler(ctx)
    sched.add(wake_job)
    return sched


# ── Backward-compat alias (deprecated) ────────────────────────────


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
