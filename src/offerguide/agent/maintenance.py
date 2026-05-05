"""W13.2 maintenance action tools — daemon jobs as agent-callable tools.

The W4-W12 architecture hardcoded 7 cron daemons in ``autonomous/scheduler.py``,
each running on its own schedule (06:30 discover, 06:45 enrich, 07:00 classify,
09:00 silence, 23:00 brief_update, etc.). They couldn't reason about each other:
``discover_jobs`` would dump 30 thin JDs into the table, then ``jd_enrich``
would wake 15 minutes later and process them — but if a user only wanted to
look at one company that day, the daemons did the same exhaustive sweep.

W13.2 turns each daemon's ``run(ctx) -> dict`` function into an action tool
the agent can call directly. The agent looks at the snapshot ("8 thin JDs in
queue, 2 silent applications past 7 days, 5 new corpus items unclassified")
and decides which maintenance tools are worth running THIS tick.

The agent doesn't replace ALL the autonomy — there's still a wake-up cron
in ``autonomous/scheduler.py`` (W13.2 part 2), but it now schedules ONE
thing: an agent loop with a "巡检" goal. The agent reads the state and
picks which of these tools to invoke.

Why action tools (not SKILLs):
- These are pure orchestration: spider, classifier, silence-tracker.
  No LLM creative work. SKILL.md frontmatter would be empty.
- They mutate DB heavily (insert jobs, applications, signals). SKILLs are
  pure-LLM-call abstractions.
- No need to evolve their prompts (they don't have prompts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..llm import LLMClient
from ..memory import Store
from ..skills import SkillRuntime, SkillSpec

log = logging.getLogger(__name__)


@dataclass
class MaintenanceCtx:
    """Per-call context for maintenance tool execution.

    Built fresh for each agent invocation from the AgentLoop's already-
    constructed resources. Mirrors the autonomous scheduler's JobContext
    so we can reuse the daemon job functions verbatim.
    """
    store: Store
    llm: LLMClient
    runtime: SkillRuntime
    skills: list[SkillSpec]
    user_profile_text: str | None = None
    settings: Settings | None = None
    search: Any | None = None
    notifier: Any | None = None


# ─────────────────────────── tool schemas ───────────────────────────


MAINTENANCE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "discover_new_jobs",
            "description": (
                "运行所有 spider (Campus2026 + awesome_jobs 等) 抓新的 JD 入库。"
                "通常每天调一次。返回: 抓了几条, 入库几条, 重复几条。"
                "调用前看 snapshot: 如果今天已经 discover 过, 不要重复调。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enrich_thin_jds",
            "description": (
                "找 raw_text < 200 字的 jobs (spider 只抓到 metadata 没抓到 JD 全文), "
                "用 HTTP fetch + LLM 提取完整 JD 文本写回 jobs.raw_text。"
                "snapshot 里 'jd=N字' 字段提示哪些需要 enrich。N < 200 = 待 enrich。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_jobs": {
                        "type": "integer",
                        "description": "本次最多 enrich 多少个 (默认 10, 控制 LLM 成本)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_corpus",
            "description": (
                "对 quality_classified_at IS NULL 的 interview_experiences 跑质量分类器。"
                "只有分类完 (quality_score) 的面经才会被 successful_profile / "
                "prepare_interview 等 SKILL 用作 RAG 上下文。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_silent_applications",
            "description": (
                "扫所有 non-terminal applications (没拿 offer / 没被拒), "
                "对超过 7/14/30 天没事件的标 'silent_check' 事件 + 写 inbox 提示。"
                "幂等: 当天再调不会重复标。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refresh_company_corpus",
            "description": (
                "对一家公司用 web 搜索 (Tavily / Bing CN / DDG) 抓最新面经入库。"
                "跑一次 ~30-60s + 几分钱搜索成本。"
                "通常对 user 关注的公司 (snapshot user_facts 里有提到的) 调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string",
                        "description": "公司中文名 (字节跳动 / 腾讯 / ...)",
                    },
                    "role_hint": {
                        "type": "string",
                        "description": "可选的岗位线索 ('AI 算法' / '后端')",
                    },
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_facts_from_runs",
            "description": (
                "扫最近 N 个 skill_runs (默认 20), 用 LLM 提取 user_facts (用户经历 / 偏好 / "
                "项目细节) 写入长期记忆。"
                "通常每天 1 次, 不要短期内多次调 (会重复扫已处理的 run)。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_runs": {
                        "type": "integer",
                        "description": "扫多少最近 runs (默认 20)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regenerate_company_brief",
            "description": (
                "对一家公司基于最近的面经 / 申请事件 / JD 重新生成 company_brief "
                "(招聘节奏 / 面试风格 / 当前申请上限 / 趋势)。"
                "snapshot 里看 user_facts / applications 里这家公司有没有大变化, 有就调。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "公司中文名"},
                },
                "required": ["company"],
            },
        },
    },
]

MAINTENANCE_TOOL_NAMES = {sc["function"]["name"] for sc in MAINTENANCE_TOOL_SCHEMAS}


# ─────────────────────────── execution ───────────────────────────


def execute_maintenance_tool(name: str, arguments: dict[str, Any], ctx: MaintenanceCtx) -> str:
    """Dispatch a maintenance tool call to the underlying daemon-job function.

    Wraps each call: builds a daemon JobContext from MaintenanceCtx, invokes
    the job's ``run(ctx)``, formats the result dict as a human-readable string
    for the agent to reason about.
    """
    # Lazy import — daemon jobs pull in spiders/network code we don't need
    # unless the agent actually calls one of these tools.
    try:
        if name == "discover_new_jobs":
            return _run_with_job_context(ctx, _import_job("discover_jobs"))
        if name == "enrich_thin_jds":
            max_jobs = _safe_int(arguments.get("max_jobs"), default=10)
            return _run_with_job_context(
                ctx, _import_job("jd_enrich"), limit=max_jobs,
            )
        if name == "classify_corpus":
            return _run_with_job_context(ctx, _import_job("corpus_classify"))
        if name == "check_silent_applications":
            return _run_with_job_context(ctx, _import_job("silence_check"))
        if name == "extract_facts_from_runs":
            max_runs = _safe_int(arguments.get("max_runs"), default=20)
            return _run_with_job_context(
                ctx, _import_job("extract_facts"), limit=max_runs,
            )
        if name == "refresh_company_corpus":
            company = (arguments.get("company") or "").strip()
            if not company:
                return "ERROR: refresh_company_corpus 需要 company 参数"
            role_hint = (arguments.get("role_hint") or "").strip() or None
            return _run_corpus_refresh_one_company(ctx, company, role_hint)
        if name == "regenerate_company_brief":
            company = (arguments.get("company") or "").strip()
            if not company:
                return "ERROR: regenerate_company_brief 需要 company 参数"
            return _run_brief_for_company(ctx, company)
    except Exception as e:
        log.exception("maintenance tool %s crashed", name)
        return f"ERROR: {name} raised {type(e).__name__}: {e}"
    return f"ERROR: maintenance tool '{name}' is declared but not dispatched"


# ─────────────────────────── internals ───────────────────────────


def _import_job(module_name: str):
    """Lazy-import a daemon job module's ``run`` function."""
    import importlib
    mod = importlib.import_module(f"offerguide.autonomous.jobs.{module_name}")
    return mod.run


def _build_job_ctx(ctx: MaintenanceCtx):
    """Adapt MaintenanceCtx → autonomous.scheduler.JobContext."""
    from ..autonomous.scheduler import JobContext
    return JobContext(
        settings=ctx.settings or Settings.from_env(),
        store=ctx.store,
        llm=ctx.llm,
        search=ctx.search,
        notifier=ctx.notifier,
        runtime=ctx.runtime,
        skills=ctx.skills,
        user_profile_text=ctx.user_profile_text,
    )


def _run_with_job_context(
    ctx: MaintenanceCtx,
    run_func,
    **job_kwargs: Any,
) -> str:
    """Build a JobContext, run the job (forwarding any per-call kwargs),
    render the result dict as a human-readable string the agent can reason
    about.

    W14.9: previously this set os.environ vars before the call to pass
    per-call tunables like ``OFFERGUIDE_JD_ENRICH_MAX``. Two real bugs
    flowed from that:

      1) The daemon job modules never actually read those env vars — they
         used hardcoded module constants. So the agent's ``max_jobs=5``
         arg was silently ignored and the job ran the default (15).
      2) os.environ is process-global. Two concurrent agent runs (SSE +
         scheduler tick, or two SSE clients) would race on the env var,
         each clobbering the other's value mid-flight.

    Explicit kwargs fix both: the value flows directly into the job's
    own signature, and there's no shared global state to race on.
    """
    job_ctx = _build_job_ctx(ctx)
    result = run_func(job_ctx, **job_kwargs)

    if isinstance(result, dict):
        # Render dict as "key=value, key=value" for readability
        parts = [f"{k}={v}" for k, v in result.items()]
        return f"OK: {', '.join(parts)}"
    return f"OK: {result}"


def _run_corpus_refresh_one_company(
    ctx: MaintenanceCtx, company: str, role_hint: str | None,
) -> str:
    """Direct call to CorpusCollector for one company (skip the daemon's
    multi-company sweep). Mirrors what ``corpus_refresh.run`` does per company."""
    if ctx.llm is None:
        return "ERROR: LLM 没配, 无法跑 corpus refresh"
    try:
        from ..agentic.corpus_collector import CorpusCollector
        from ..agentic.search import build_default_search
    except Exception as e:
        return f"ERROR: agentic 模块加载失败: {e}"

    search = ctx.search or build_default_search()
    collector = CorpusCollector(
        store=ctx.store, llm=ctx.llm, search=search,
    )
    try:
        # W14.7-fix: CorpusCollector exposes `collect`, not `refresh_company`.
        # The previous name was inherited from an older draft and never existed.
        result = collector.collect(company=company, role_hint=role_hint)
    except Exception as e:
        return f"ERROR: collect({company}) 失败: {e}"
    return (
        f"OK: 公司={company} role={role_hint or '*'} "
        f"hits={result.hits_seen}, evaluated={result.hits_evaluated}, "
        f"inserted={result.inserted}, dup={result.skipped_dup}, "
        f"low_quality={result.skipped_low_quality}"
    )


def _run_brief_for_company(ctx: MaintenanceCtx, company: str) -> str:
    """Regenerate one company's brief — direct call instead of the daemon's
    multi-company sweep."""
    if ctx.llm is None:
        return "ERROR: LLM 没配, 无法生成 brief"
    try:
        from .. import briefs as briefs_mod
        result = briefs_mod.refresh_brief(
            ctx.store, company=company, llm=ctx.llm,
        )
    except Exception as e:
        return f"ERROR: refresh_brief({company}) 失败: {e}"
    if result is None:
        return f"OK: {company} 没有足够信号生成 brief (没有近期面经/JD)"
    # W14.7-fix: refresh_brief returns BriefRow (wrapper); the CompanyBrief
    # fields live under `result.brief`. Reading them off the wrapper directly
    # would AttributeError on every successful regeneration.
    return (
        f"OK: 公司={company} confidence={result.brief.confidence:.2f} "
        f"app_limit={result.brief.current_app_limit} "
        f"summary={result.brief.summary[:120]}..."
    )


def _safe_int(v: Any, *, default: int) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default
