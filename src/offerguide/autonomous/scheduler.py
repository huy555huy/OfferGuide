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


_AGENT_WAKE_GOAL = """你是 OfferGuide 的中央 ambient agent. 你每小时被心跳叫醒, 看完整
state 自己决定干什么 — 找新 JD / score / follow up / 写 suggestion / 问用户 /
跨 wake 留 todo / 还是什么都不做.

# 你拥有的工具 (按场景分)

## 找事 / 看事
- **discover_new_jobs**: 调子 agent 用 Tavily 找 3-5 个 JD (~3 分钟, ~$0.15)
- **score_unscored_jobs(limit)**: 给新 JD 跑 score_match, 高分自动推 inbox
- **read_job(job_id)** / **read_user_resume**: 看具体 JD / 简历

## 跟用户沟通
- **write_suggestion(title, body, ...)**: 单方面推荐. 用户 approve/reject 反馈
  到 evolution_signals 让 SKILL 演化.
- **ask_user_question(question, context, options)**: **主动问用户** (≥ 2 选项).
  用在: north star 跟简历方向不一致 / 多次 reject 后想确认 / 多个备选让用户挑.
  用户答完写到 user_facts, 你下次 wake 看到. 别 spam, 真不知道再问.

## 跨 wake 接力 (working memory)
- **write_note_to_self(body, kind)**: 留给未来的自己. 下次 wake 你在 snapshot
  顶部看到. 用在: "下次看 4 个 JD score 出来没" / "等用户答完那条 question 再
  decide 调 deeper" / "今天深夜了, 等明天早上 follow up".
- **clear_self_note(note_id, reason)**: 完成或不再 relevant 时调. 别让 snapshot
  越积越多.

## 维护 / 后台
- enrich_thin_jds / check_silent_applications / refresh_company_corpus /
  regenerate_company_brief / extract_facts_from_runs / classify_corpus

## 元认知
- meta_reflect: 看自己的 pattern, 写 self-observation
- + 11 个业务 SKILL (score_match / apply_assistant / tailor_resume / ...)

# 怎么决定干啥 — 一个流程提示

1. **先看 snapshot 顶部的 self_notes** (你上次给自己留的). 完成的 clear, 不再
   relevant 的也 clear.
2. **看 pending question** (你已经问的没答) — 别再问同类的, 等答案.
3. **看 north star + funnel + 时间** — 排出真正最该做的 1-2 件事.
4. 做完写 final + (如果有跨 wake 任务) write_note_to_self 给未来的自己.

**包括"什么都不做" 也是合理决定** — 深夜用户睡觉, 没新事件 = lay low. 真的.

# Final 写啥

- 做了啥 + 实际价值
- 跳过了啥 + 为啥
- 给自己留的 note (如果有, 提一下 note_id 让用户能看)
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
        """The single job: wake the central agent + let it decide what to do.

        W14: dynamic goal stitching — base wake goal + urgent context that
        agent should know about FIRST. Right now we surface "goals that look
        off-track" because those are the things the user would want the
        agent to think about before doing housekeeping.
        """
        if llm is None:
            return {"skipped": "no LLM configured"}
        if runtime is None:
            return {"skipped": "no SkillRuntime"}

        # Build the goal text. Start with base, then append any urgent context
        # so it's the LAST thing in the prompt → highest recency weight.
        goal_parts = [_AGENT_WAKE_GOAL]
        try:
            from .. import goals as _gmod
            active_goals = _gmod.list_active_goals(store)
            urgent_lines: list[str] = []
            for g in active_goals[:3]:
                progress = _gmod.compute_progress(store, g)
                if not progress.is_on_track:
                    urgent_lines.append(
                        f"- 「{g.title}」 off-track: "
                        + (f"剩 {progress.days_left} 天, "
                           if progress.days_left is not None else "")
                        + f"funnel {progress.apps_active}/{progress.apps_total}"
                          f", {progress.offers} offer"
                    )
                if progress.apps_silent_14d > 0:
                    urgent_lines.append(
                        f"- 「{g.title}」: {progress.apps_silent_14d} 个申请 14+ 天没回, 该判断 give up 还是最后催"
                    )
            if urgent_lines:
                goal_parts.append(
                    "\n\n# 此刻已知急事 (W14 注入, 应优先考虑):\n" + "\n".join(urgent_lines)
                )
        except Exception as e:
            log.debug("goal off-track injection failed (non-fatal): %s", e)

        agent = AgentLoop(
            llm=llm, runtime=runtime, store=store,
            skills=skills, master_resume_text=master_resume,
            max_iterations=8, critic_enabled=True,
            notifier=notifier,
        )
        result = agent.run(
            goal="".join(goal_parts),
            trigger_kind="cron_wake",
        )
        return {
            "agent_run_id": result.run_id,
            "iterations": result.iterations,
            "critic_score": result.critic_score,
            "latency_s": round(result.latency_ms / 1000.0, 1),
            "final": (result.final_answer or "")[:300],
        }

    # W14.18 — single ambient agent loop. Was 3 cron daemons (wake_agent +
    # discover_jobs_via_search + auto_score_new_jobs) — that was "3 微服务
    # + cron + DB 当总线" = 后端思想. User feedback (verbatim):
    # > "整体流程, 我们是在造 agent, 而不是一个接入了 LLM 的小程序, 你懂吗?
    # > 现在设计的还是老一套后端端思想"
    #
    # New design: 1 cron just for the heartbeat (laptop has no other way to
    # wake the agent). At each tick, the agent reads full state and decides
    # itself: 找新 JD? score? follow up silent app? sleep? — by calling its
    # tool set (which now includes discover_new_jobs and score_unscored_jobs
    # exposed via maintenance.py, replacing the dedicated daemons).
    #
    # Higher frequency than before (every hour vs 4h) because the agent now
    # owns priority — it can no-op cheaply when there's nothing to do, but
    # won't miss work for 4h after a state change.
    cron_kwargs = cron_kwargs or {"hour": "8-22"}  # hourly 08:00-22:00
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


def _discover_via_search_job(jc: JobContext) -> dict[str, Any]:
    """Cron-driven: drive a real ReAct agent loop to find JDs.

    W14.16: replaced JobCollector (BFS-with-LLM-as-classifier) with
    JobFinderAgent (LLM in the driver seat, calls web_search / fetch_url
    / extract_and_ingest_jd / done as it sees fit).
    """
    if jc.llm is None:
        return {"skipped": "no_llm"}
    if jc.search is None:
        return {"skipped": "no_search_backend"}
    try:
        from ..agentic.job_finder_agent import JobFinderAgent
    except Exception as e:
        return {"skipped": f"job_finder_agent import failed: {e}"}

    # Pull north star from active goals (most recent first)
    north_star = "拿 1 个 AI Agent 暑期实习 offer"
    try:
        from .. import goals as _gmod
        active = _gmod.list_active_goals(jc.store)
        if active:
            north_star = active[0].title
    except Exception:
        pass  # use default

    agent = JobFinderAgent(store=jc.store, llm=jc.llm, search=jc.search)
    try:
        result = agent.run(north_star=north_star)
    finally:
        agent.close()

    summary = {
        "north_star": north_star,
        "iterations": result.iterations,
        "inserted": result.inserted,
        "skipped_dup": result.skipped_dup,
        "new_job_ids": result.new_job_ids,
        "search_queries_used": result.search_queries[:8],
        "urls_visited": result.visited_urls[:10],
        "finish_reason": result.finish_reason,
        # Agent's per-step trace — surfaced in home Activity Timeline so
        # user can audit the LLM's actual decisions, not just count outputs.
        "notes": [n[:200] for n in result.notes[:25]],
    }
    if jc.notifier and result.inserted > 0:
        try:
            jc.notifier.notify(
                title=f"OfferGuide: agent 自己找到 {result.inserted} 个新 JD",
                body=f"基于「{north_star}」, agent 走了 {result.iterations} 步, "
                     f"自己 search + fetch + 抽 JD 入库 {result.inserted} 个。"
                     f"30 分钟内会自动 score + 推荐高匹配的到 inbox。",
                level="info",
            )
        except Exception:
            log.warning("discover_via_search: notify failed", exc_info=True)
    return summary


def _auto_score_via_daemon(jc: JobContext) -> dict[str, Any]:
    """Cron-driven: catch un-scored JDs, score, pre-gen apply pkg, inbox."""
    try:
        from .jobs import auto_score_jobs as _asj
    except Exception as e:
        return {"skipped": f"auto_score_jobs import failed: {e}"}
    return _asj.run(jc)


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
