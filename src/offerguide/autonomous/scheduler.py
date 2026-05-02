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


# ── Default scheduler factory ──────────────────────────────────────


def build_default_scheduler(
    *,
    settings: Settings | None = None,
) -> AutonomousScheduler:
    """Build a scheduler with the 4 default jobs registered:

    - discover_jobs       (daily 06:30) — spider sweep + auto-eval
    - silence_check       (daily 09:00) — tracker sweep
    - corpus_refresh      (weekly Mon 08:00) — agentic 面经
    - brief_update        (daily 23:00, after the day's events settled)
    """
    from ..agentic.search import build_default_search
    from ..profile import load_resume_pdf
    from ..skills import SkillRuntime, discover_skills
    from ..ui.notify import make_notifier
    from .jobs.brief_update import BRIEF_UPDATE_JOB
    from .jobs.corpus_classify import CORPUS_CLASSIFY_JOB
    from .jobs.corpus_refresh import CORPUS_REFRESH_JOB
    from .jobs.discover_jobs import DISCOVER_JOBS_JOB
    from .jobs.extract_facts import EXTRACT_FACTS_JOB
    from .jobs.jd_enrich import JD_ENRICH_JOB
    from .jobs.silence_check import SILENCE_CHECK_JOB

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
        log.warning("search backend init failed; corpus_refresh will skip: %s", e)

    # Discover SKILLs + load resume so discover_jobs can auto-eval.
    # Each is optional — discover_jobs degrades to "ingest only" if missing.
    from pathlib import Path
    skills_root = Path(__file__).parent.parent / "skills"
    skills = []
    try:
        skills = discover_skills(skills_root)
    except Exception as e:
        log.warning("skill discovery failed; auto-eval disabled: %s", e)

    runtime = None
    if llm is not None:
        runtime = SkillRuntime(llm=llm, store=store)

    profile = None
    if settings.resume_pdf:
        try:
            profile = load_resume_pdf(settings.resume_pdf)
        except Exception as e:
            log.warning("resume load failed; auto-eval disabled: %s", e)

    ctx = JobContext(
        settings=settings,
        store=store,
        llm=llm,
        search=search,
        notifier=make_notifier(settings),
        runtime=runtime,
        skills=skills,
        user_profile_text=profile.raw_resume_text if profile else None,
    )

    sched = AutonomousScheduler(ctx)
    sched.add(EXTRACT_FACTS_JOB)
    sched.add(DISCOVER_JOBS_JOB)
    sched.add(JD_ENRICH_JOB)
    sched.add(CORPUS_CLASSIFY_JOB)
    sched.add(SILENCE_CHECK_JOB)
    sched.add(CORPUS_REFRESH_JOB)
    sched.add(BRIEF_UPDATE_JOB)
    return sched
