"""Tool schemas + dispatch for the W15 harness.

Harness tools are **capabilities** the agent can choose; the agent decides
when/how to use them. No hardcoded "if condition X, call tool Y" logic — that's
the model's job, framed by ``instructions.md``.

The most important job-discovery distinction: verified official-source tools
return only sources we have actually probed, while generic web search remains
available for exploration.
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ..config import Settings
from ..llm import LLMClient
from ..memory import Store
from ..tools.registry import registry as _registry
from .memory import MEMORY_TOOL_SCHEMA, MemoryStore

if TYPE_CHECKING:
    from ..skills import SkillRuntime, SkillSpec
else:
    SkillRuntime = Any  # type: ignore[assignment,misc]
    SkillSpec = Any  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


_DESIGN_NOTE = """
Tool design note:
Each tool is a
**capability** the agent can choose; the agent decides when/how to use
them. No hardcoded "if condition X, call tool Y" logic — that's the
model's job, framed by ``instructions.md``.

Anthropic principle (verbatim from research):
> "Design 5-10 intentional tools targeting high-impact workflows."
> "Don't give raw SQL access; give search_jobs_by_criteria() wrapper."

Tool granularity: each tool is one verb of agent agency. ``score_match``
(SKILL wrapper) instead of ``invoke_skill(name=score_match)`` — agent
shouldn't have to know SKILL infra exists. Tools below all return
strings (success or 'ERROR: ...') so the agent reads them directly in
the next turn.
"""


# ── Dependency container ──────────────────────────────────────────────


@dataclass
class HarnessDeps:
    """Everything the dispatch functions need. Built once per agent run.

    Fields can be ``None`` for graceful degradation (e.g. ``llm=None``
    means LLM-using tools should return ERROR; the agent learns to
    avoid them).
    """

    settings: Settings
    store: Store
    memory_store: MemoryStore
    llm: LLMClient | None = None
    runtime: SkillRuntime | None = None
    skills: list[SkillSpec] = field(default_factory=list)
    search: Any = None
    notifier: Any = None
    user_profile_text: str | None = None
    current_run_id: int | None = None
    """Set by the loop; tools record references back to the run."""

    extra_cost_usd: float = 0.0
    """Sub-agent / external LLM cost sink (W15.12 Bug 5 fix). Tools that
    drive their own LLM calls (e.g. ``discover_jobs`` → DiscoverySubAgent)
    accumulate cost here so the master loop can include it in
    ``harness_runs.cost_usd``. Reset to 0 at the start of each run."""

    def find_skill(self, name: str) -> SkillSpec | None:
        for s in self.skills:
            if s.name == name:
                return s
        return None


# ── Tool schemas (OpenAI function-calling format) ──────────────────


_TOOL_DISCOVER_JOBS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "discover_jobs",
        "description": (
            "Actively find new JDs matching the user's criteria via web search. "
            "Drives a sub-agent (ReAct: web_search / fetch_url / extract). "
            "Pass criteria only from explicit evidence in worldview, active goals, "
            "or the user's current request. Do not invent missing preferences. "
            "Use this to do the chore the user dreads — manually scrolling job boards."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "string",
                    "description": (
                        "Natural-language description of what to look for, "
                        "e.g. 'AI agent / LLM application 暑期实习, prefer "
                        "BAT-tier or AI-native startups, exclude pure research labs.' "
                        "Must be grounded in candidate.md / MEMORY.md / active goals "
                        "or current user text."
                    ),
                },
            },
            "required": ["criteria"],
        },
    },
}


_TOOL_SEARCH_OFFICIAL_JOBS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_official_jobs",
        "description": (
            "Search verified official recruitment sources and ingest real JDs. "
            "Currently proven: Tencent campus/social JSON APIs and Baidu campus "
            "SSR list data. For ByteDance/Alibaba/Meituan/Xiaohongshu/BOSS, "
            "returns the observed limitation instead of inventing support."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "Company name; omit to search verified big-company sources.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Job keyword, e.g. AI Agent / LLM / RAG.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max jobs per source, default 5, max 20.",
                },
            },
            "required": ["keyword"],
        },
    },
}


_TOOL_FETCH_JD: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_jd",
        "description": (
            "User pasted a job URL or text — pull it into the jobs table. "
            "Returns the job_id you can then score / tailor / track."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url_or_text": {
                    "type": "string",
                    "description": "Job URL OR full JD text the user pasted.",
                },
                "company_hint": {"type": "string"},
                "title_hint": {"type": "string"},
            },
            "required": ["url_or_text"],
        },
    },
}


_TOOL_SCORE_MATCH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "score_match",
        "description": (
            "Score how well a job matches the user's profile. Returns a "
            "score (0-10), reasoning, and key gaps. Uses the score_match "
            "SKILL (which evolves via GEPA from user feedback)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
            },
            "required": ["job_id"],
        },
    },
}


_TOOL_TAILOR_ADVICE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "tailor_advice",
        "description": (
            "Given a specific job, output targeted resume modification "
            "advice (NOT a rewrite — bullet-level suggestions only). "
            "Call proactively only when the job is actually identified and has "
            "evidence of being worth applying to (score, explicit user interest, "
            "or clear goal match)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
            },
            "required": ["job_id"],
        },
    },
}


_TOOL_INTERVIEW_PREP: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "interview_prep",
        "description": (
            "User has an interview coming up — prepare them. Outputs "
            "predicted questions + answer skeletons + company intel. "
            "**Only call this when user asks** (do not push proactively — "
            "users handle interview prep themselves)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
                "round": {
                    "type": "string",
                    "description": "e.g. '一面' / '二面' / 'HR'",
                },
            },
            "required": ["job_id"],
        },
    },
}


_TOOL_REFLECT_OUTCOME: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "reflect_outcome",
        "description": (
            "User finished an interview — capture the outcome and learn. "
            "Triggers post-interview reflection SKILL + records lessons "
            "into worldview. Only call when user shares interview result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
                "outcome": {
                    "type": "string",
                    "description": "e.g. 'passed' / 'rejected' / 'pending' / user's free-text",
                },
                "user_notes": {"type": "string"},
            },
            "required": ["job_id", "outcome"],
        },
    },
}


_TOOL_RECORD_EVENT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record_event",
        "description": (
            "Record a job-hunt lifecycle event into the audit trail. "
            "Examples: applied / interview_received / interview_done / "
            "rejected / offer / followup_sent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string"},
                "job_id": {"type": "integer"},
                "note": {"type": "string"},
            },
            "required": ["kind"],
        },
    },
}


_TOOL_NOTIFY_USER: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "notify_user",
        "description": (
            "Push a message to the user's home page (and notification "
            "channel if configured). Use for: high-match jobs found / "
            "silent followup reminders / deadline alerts / important "
            "insights. **Be selective** and cite the concrete evidence; "
            "over-notification trains the user to ignore you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "related_job_id": {"type": "integer"},
                "priority": {
                    "type": "string",
                    "enum": ["info", "important", "urgent"],
                },
            },
            "required": ["title", "body"],
        },
    },
}


_TOOL_ASK_USER: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user a question. They'll see it in their inbox; "
            "answers come back to you on the next wake (read user_facts "
            "or check the question status). Use sparingly: only when "
            "you genuinely cannot proceed without their input."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {
                    "type": "string",
                    "description": "Why you're asking — shown to user.",
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "label": {"type": "string"},
                        },
                    },
                    "description": "≥2 button options the user clicks.",
                },
            },
            "required": ["question", "context", "options"],
        },
    },
}


_TOOL_SCHEDULE_NEXT_WAKE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "schedule_next_wake",
        "description": (
            "Tell the harness when to wake you again. **Use this to keep "
            "ownership** when there is a concrete event to follow up — e.g. "
            "after user marks 'applied X', call schedule_next_wake(7d, "
            "'check whether the system has recorded a response from X'). "
            "Do not schedule from a guessed user state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "integer",
                    "description": "Seconds from now. Min 60, max 30 days.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why — for logs and your own future reference.",
                },
            },
            "required": ["delay_seconds", "reason"],
        },
    },
}


_TOOL_WEB_SEARCH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web (Tavily). Returns ≤ 10 hits with url + title + snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
}


_TOOL_FETCH_URL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "HTTP GET a URL, return first 6000 chars of stripped text. "
            "Errors return 'ERROR: ...' (e.g. 404, anti-scraping)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
}


# ── Evolution tools — agent self-evolves its own SKILL prompts ──
#
# These close the GEPA-style self-evolution loop the user designed as the
# project's core differentiator. The agent reads its own fitness scores,
# detects SKILLs that are underperforming on real user signal (thumbs /
# app outcome / follow-through), generates new prompt variants via
# evolve_skill, and lets gray-release decide what wins. Without these
# three tools the agent can't drive its own evolution — only the human
# can, via the /evolution UI.

_TOOL_DETECT_EVOLUTION_CANDIDATES: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "detect_evolution_candidates",
        "description": (
            "Find SKILLs whose current live version is underperforming on "
            "real user signals (thumbs / app outcome / follow-through) and "
            "is past cooldown — i.e. ripe for a new prompt variant. "
            "Returns a list of {skill_name, current_version, fitness, "
            "sample_count, reason}. Call this when you suspect the agent "
            "is making low-quality outputs in some domain; the result tells "
            "you which SKILL prompt to evolve."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_TOOL_EVOLVE_SKILL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "evolve_skill",
        "description": (
            "Generate N candidate prompt variants for one SKILL, persisted "
            "as 'shadow' rows. Variants are NOT live yet — gray-release "
            "(run_release_cycle) promotes a winner after canary traffic. "
            "Call this on a SKILL surfaced by detect_evolution_candidates. "
            "Cost: ~$0.01 per call (one LLM-driven variant-generation pass)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Exact SKILL name (e.g. 'score_match').",
                },
                "num_variants": {
                    "type": "integer",
                    "description": "How many variants to generate (1-5, default 3).",
                },
            },
            "required": ["skill_name"],
        },
    },
}

_TOOL_RUN_RELEASE_CYCLE: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_release_cycle",
        "description": (
            "Advance the gray-release pipeline once: promote shadow→canary "
            "for SKILLs with new variants, judge canary→live (or fail) "
            "based on accumulated A/B signal. Idempotent — call after "
            "evolve_skill or when you suspect canary signals have ripened."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "If true, report actions but don't commit.",
                },
            },
        },
    },
}


# Master tool list — order matters for prompt clarity (related tools grouped).
# The pair (_TOOL_SCHEMA, _exec_func) is the single source of truth: it's
# registered into the global ToolRegistry as group="main" at module load
# (via ``_register_main_tools`` at file end), and the LLM-facing list +
# the dispatch path both come out of that one registration.
_MAIN_TOOL_ENTRIES: list[tuple[dict[str, Any], str]] = [
    # Memory comes first — it's the agent's brain
    (MEMORY_TOOL_SCHEMA, "_exec_memory"),
    # Active (agent should proactively call)
    (_TOOL_SEARCH_OFFICIAL_JOBS, "_exec_search_official_jobs"),
    (_TOOL_DISCOVER_JOBS, "_exec_discover_jobs"),
    (_TOOL_TAILOR_ADVICE, "_exec_tailor_advice"),
    (_TOOL_NOTIFY_USER, "_exec_notify_user"),
    # Reactive (agent calls when user brings it)
    (_TOOL_FETCH_JD, "_exec_fetch_jd"),
    (_TOOL_INTERVIEW_PREP, "_exec_interview_prep"),
    (_TOOL_REFLECT_OUTCOME, "_exec_reflect_outcome"),
    # Universal capabilities
    (_TOOL_SCORE_MATCH, "_exec_score_match"),
    (_TOOL_RECORD_EVENT, "_exec_record_event"),
    (_TOOL_ASK_USER, "_exec_ask_user"),
    (_TOOL_SCHEDULE_NEXT_WAKE, "_exec_schedule_next_wake"),
    (_TOOL_WEB_SEARCH, "_exec_web_search"),
    (_TOOL_FETCH_URL, "_exec_fetch_url"),
    # Self-evolution — agent drives SKILL improvement from real signals
    (_TOOL_DETECT_EVOLUTION_CANDIDATES, "_exec_detect_evolution_candidates"),
    (_TOOL_EVOLVE_SKILL, "_exec_evolve_skill"),
    (_TOOL_RUN_RELEASE_CYCLE, "_exec_run_release_cycle"),
]


# Preserve order for the LLM prompt — registry sorts by name, but the prompt
# benefits from grouping (memory first, evolution last, etc).
ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    schema for schema, _ in _MAIN_TOOL_ENTRIES
]


# ── Dispatch ──────────────────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any], deps: HarnessDeps) -> str:
    """Route a tool call through the global ToolRegistry.

    Handlers for the main agent's tools are registered by
    ``_register_main_tools`` at module load. This wrapper:
    1. Returns ``'ERROR: unknown tool ...'`` for missing tools (the harness
       loop matches on the literal "ERROR:" prefix to surface to the model)
    2. Injects ``deps`` as the registry runtime kwarg
    3. Translates uncaught handler exceptions into the same ERROR prefix
    """
    if _registry.get(name) is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        return _registry.dispatch(name, args, deps=deps)
    except Exception as e:
        log.exception("tool %s crashed: args=%s", name, args)
        return f"ERROR: {type(e).__name__}: {e}"


# ── Tool implementations ─────────────────────────────────────────────


def _exec_memory(args: dict[str, Any], deps: HarnessDeps) -> str:
    return deps.memory_store.execute(args)


def _exec_discover_jobs(args: dict[str, Any], deps: HarnessDeps) -> str:
    if deps.llm is None:
        return "ERROR: discover_jobs requires LLM (DEEPSEEK_API_KEY not set)"
    criteria = (args.get("criteria") or "").strip()
    if not criteria:
        return "ERROR: discover_jobs requires criteria (one sentence is fine)"

    from ..agents.base import register_universal_tools
    from ..agents.discovery import DiscoverySubAgent
    from ..tools import load_all_tools, registry

    load_all_tools()
    register_universal_tools(registry)

    sub = DiscoverySubAgent(
        llm=deps.llm,
        registry=registry,
        store=deps.store,
        settings=deps.settings,
        runtime=deps.runtime,
        skills=deps.skills,
        user_profile_text=deps.user_profile_text,
    )
    result = sub.run(goal=criteria)

    deps.extra_cost_usd += float(result.cost_usd or 0.0)
    summary = (
        f"OK discover_jobs done in {result.iterations} iters "
        f"({result.tool_calls_made} tool calls, ${result.cost_usd:.4f}). "
        f"Sub-agent summary: {result.final_answer[:600]}"
    )
    if result.error:
        summary += f"\nNote: sub-agent error: {result.error}"
    return summary


def _exec_search_official_jobs(args: dict[str, Any], deps: HarnessDeps) -> str:
    keyword = (args.get("keyword") or "").strip()
    if not keyword:
        return "ERROR: search_official_jobs requires keyword"
    company = (args.get("company") or "").strip() or None
    limit = max(1, min(int(args.get("limit") or 5), 20))

    from ..platforms.official_jobs import search_verified_official_jobs
    from ..workers import scout

    results = search_verified_official_jobs(
        company=company,
        keyword=keyword,
        limit=limit,
    )
    inserted_ids: list[int] = []
    dup_ids: list[int] = []
    lines = [
        "OK search_official_jobs completed",
        f"criteria: company={company or 'verified official sources'} keyword={keyword!r}",
    ]
    for source_result in results:
        lines.append(
            f"- {source_result.source}: {source_result.status}; "
            f"evidence={source_result.evidence_url or '(none)'}; "
            f"{source_result.note}"
        )
        for rj in source_result.jobs:
            was_new, job_id = scout.ingest(deps.store, rj)
            if was_new:
                inserted_ids.append(job_id)
                lines.append(f"  inserted job#{job_id}: {rj.company} · {rj.title}")
            else:
                dup_ids.append(job_id)
                lines.append(f"  duplicate job#{job_id}: {rj.company} · {rj.title}")
    lines.append(
        f"summary: inserted={inserted_ids[:10]}, duplicate={dup_ids[:10]}. "
        "Next for promising jobs: score_match(job_id)."
    )
    return "\n".join(lines)[:5000]


def _exec_fetch_jd(args: dict[str, Any], deps: HarnessDeps) -> str:
    raw = (args.get("url_or_text") or "").strip()
    if not raw:
        return "ERROR: fetch_jd requires url_or_text"
    company_hint = (args.get("company_hint") or "").strip()
    title_hint = (args.get("title_hint") or "").strip()

    from ..platforms import RawJob
    from ..workers import scout

    if raw.startswith(("http://", "https://")):
        # URL path: fetch + ingest
        try:
            r = httpx.get(raw, follow_redirects=True, timeout=12.0,
                          headers={"User-Agent": "Mozilla/5.0"})
        except httpx.HTTPError as e:
            return f"ERROR: fetch failed: {e}"
        if r.status_code != 200:
            return f"ERROR: HTTP {r.status_code}"
        body_text = _strip_html_to_text(r.text)
        if len(body_text) < 200:
            return (
                f"ERROR: page too short ({len(body_text)} chars), "
                "likely anti-scraping. Ask user to paste JD text directly."
            )
        rj = RawJob(
            source="user_paste_url",
            url=raw,
            title=title_hint or _guess_title(body_text) or "未识别岗位",
            company=company_hint or _guess_company(body_text, raw) or "未识别公司",
            location=None,
            raw_text=body_text[:10000],
            extras={"via": "fetch_jd_tool"},
        )
    else:
        # Text path: ingest with synthetic URL
        if len(raw) < 200:
            return f"ERROR: text too short ({len(raw)} chars) — paste full JD"
        # Bug 4 fix: hashlib.sha256 (stable across processes) instead of
        # builtin hash() (Python 3.3+ randomizes per-process — same JD pasted
        # twice in different runs gets different URL → dedup fails).
        import hashlib as _hl
        url_digest = _hl.sha256(raw.encode("utf-8")).hexdigest()[:16]
        synth_url = f"paste://{url_digest}"
        rj = RawJob(
            source="user_paste_text",
            url=synth_url,
            title=title_hint or "粘贴的 JD",
            company=company_hint or "未指定公司",
            location=None,
            raw_text=raw[:10000],
            extras={"via": "fetch_jd_tool"},
        )

    was_new, job_id = scout.ingest(deps.store, rj)
    if was_new:
        return (
            f"OK ingested as job#{job_id} ({rj.company} · {rj.title}). "
            f"Next: call score_match(job_id={job_id})."
        )
    return f"NOTE: this is dup of existing job#{job_id} ({rj.company} · {rj.title})"


def _exec_score_match(args: dict[str, Any], deps: HarnessDeps) -> str:
    if deps.runtime is None:
        return "ERROR: score_match requires SkillRuntime (no LLM configured?)"
    spec = deps.find_skill("score_match")
    if spec is None:
        return "ERROR: score_match SKILL not registered"
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return "ERROR: score_match requires integer job_id"
    job = _load_job(deps.store, job_id)
    if job is None:
        return f"ERROR: job {job_id} not found"
    if not deps.user_profile_text:
        return "ERROR: no user resume loaded — set OFFERGUIDE_RESUME_PDF"
    # W15.22 — verified against score_match SKILL.md (inputs: job_text + user_profile;
    # output: probability/reasoning/dimensions/deal_breakers). Pre-W15.22 used
    # `job_title/job_company/jd_text/candidate_resume` and `score/key_gaps` —
    # all wrong, SkillRuntime raised ValueError → silently swallowed → no scores
    # ever recorded since W14.
    inputs = {
        "job_text": _format_jd_for_skill(job)[:4000],
        "user_profile": deps.user_profile_text[:4000],
    }
    result = deps.runtime.invoke(spec, inputs)
    if result.parsed is None:
        return (
            f"WARN score_match for job#{job_id}: SKILL output not valid JSON. "
            f"Raw output (first 500 chars):\n{result.raw_text[:500]}\n"
            f"(skill_run_id={result.skill_run_id})"
        )
    parsed = result.parsed
    prob = parsed.get("probability")
    # Record an event so /recommended can rank by this score later
    try:
        _record_event_row(
            deps, kind="scored", job_id=job_id,
            note=_json.dumps({
                "probability": prob,
                "skill_run_id": result.skill_run_id,
                "deal_breakers": parsed.get("deal_breakers") or [],
            }, ensure_ascii=False),
        )
    except Exception as e:
        log.warning("score_match: record_event failed: %s", e)
    return (
        f"OK score_match for job#{job_id} ({job.get('company')} · {job.get('title')}):\n"
        f"  probability: {prob}\n"
        f"  reasoning: {(parsed.get('reasoning') or '')[:300]}\n"
        f"  dimensions: {parsed.get('dimensions', {})}\n"
        f"  deal_breakers: {parsed.get('deal_breakers', [])}\n"
        f"  (skill_run_id={result.skill_run_id} for GEPA feedback)"
    )


def _exec_tailor_advice(args: dict[str, Any], deps: HarnessDeps) -> str:
    if deps.runtime is None:
        return "ERROR: tailor_advice requires SkillRuntime"
    spec = deps.find_skill("tailor_resume")
    if spec is None:
        return "ERROR: tailor_resume SKILL not registered"
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return "ERROR: tailor_advice requires integer job_id"
    job = _load_job(deps.store, job_id)
    if job is None:
        return f"ERROR: job {job_id} not found"
    if not deps.user_profile_text:
        return "ERROR: no user resume loaded"
    # W15.22 — verified against tailor_resume SKILL.md (inputs: master_resume,
    # job_text, company, successful_profile_json). Pre-W15.22 passed
    # job_title/jd_text/current_resume — all wrong.
    inputs = {
        "master_resume": deps.user_profile_text[:6000],
        "job_text": _format_jd_for_skill(job)[:4000],
        "company": job.get("company", ""),
        "successful_profile_json": "{}",  # no successful_profile pipeline yet → empty
    }
    result = deps.runtime.invoke(spec, inputs)
    if result.parsed is None:
        return (
            f"WARN tailor_advice for job#{job_id}: SKILL output not valid JSON. "
            f"Raw (first 500 chars):\n{result.raw_text[:500]}\n"
            f"(skill_run_id={result.skill_run_id})"
        )
    return (
        f"OK tailor_advice for job#{job_id}:\n"
        f"  {_json.dumps(result.parsed, ensure_ascii=False, indent=2)[:1500]}\n"
        f"  (skill_run_id={result.skill_run_id})"
    )


def _exec_interview_prep(args: dict[str, Any], deps: HarnessDeps) -> str:
    if deps.runtime is None:
        return "ERROR: interview_prep requires SkillRuntime"
    spec = deps.find_skill("prepare_interview")
    if spec is None:
        return "ERROR: prepare_interview SKILL not registered"
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return "ERROR: interview_prep requires integer job_id"
    job = _load_job(deps.store, job_id)
    if job is None:
        return f"ERROR: job {job_id} not found"
    # W15.22 — verified against prepare_interview SKILL.md (inputs: company,
    # job_text, user_profile, past_experiences). Pre-W15.22 passed
    # job_title/jd_text/candidate_resume/round — all wrong.
    inputs = {
        "company": job.get("company", ""),
        "job_text": _format_jd_for_skill(job)[:4000],
        "user_profile": (deps.user_profile_text or "")[:4000],
        "past_experiences": (args.get("past_experiences") or "(无)"),
    }
    result = deps.runtime.invoke(spec, inputs)
    if result.parsed is None:
        return (
            f"WARN interview_prep for job#{job_id}: SKILL output not valid JSON. "
            f"Raw (first 500 chars):\n{result.raw_text[:500]}\n"
            f"(skill_run_id={result.skill_run_id})"
        )
    return (
        f"OK interview_prep for job#{job_id} (round: {args.get('round', '?')}):\n"
        f"  {_json.dumps(result.parsed, ensure_ascii=False, indent=2)[:2000]}\n"
        f"  (skill_run_id={result.skill_run_id})"
    )


def _exec_reflect_outcome(args: dict[str, Any], deps: HarnessDeps) -> str:
    if deps.runtime is None:
        return "ERROR: reflect_outcome requires SkillRuntime"
    spec = deps.find_skill("post_interview_reflection")
    if spec is None:
        return "ERROR: post_interview_reflection SKILL not registered"
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return "ERROR: reflect_outcome requires integer job_id"
    job = _load_job(deps.store, job_id)
    if job is None:
        return f"ERROR: job {job_id} not found"
    outcome = (args.get("outcome") or "").strip()
    if not outcome:
        return "ERROR: reflect_outcome requires outcome"

    # Record the event first (audit trail)
    _record_event_row(
        deps, kind=f"interview_{outcome}", job_id=job_id,
        note=(args.get("user_notes") or ""),
    )

    # W15.22 — verified against post_interview_reflection SKILL.md (inputs:
    # company, prep_questions_json, actual_transcript). Pre-W15.22 passed
    # job_title/outcome/user_notes/jd_text — all wrong (none of those are
    # declared inputs). actual_transcript is built from outcome + user_notes
    # since we don't have a real transcript without a real interview recorder.
    transcript_parts = [f"## 面试结果: {outcome}"]
    if args.get("user_notes"):
        transcript_parts.append(f"## 用户复盘记录\n{args['user_notes']}")
    inputs = {
        "company": job.get("company", ""),
        "prep_questions_json": args.get("prep_questions_json") or "[]",
        "actual_transcript": "\n\n".join(transcript_parts),
    }
    result = deps.runtime.invoke(spec, inputs)
    if result.parsed is None:
        return (
            f"WARN reflect_outcome for job#{job_id}: SKILL output not valid JSON. "
            f"Raw (first 500 chars):\n{result.raw_text[:500]}\n"
            f"(skill_run_id={result.skill_run_id}; event recorded)"
        )
    return (
        f"OK reflect_outcome for job#{job_id} ({outcome}):\n"
        f"  {_json.dumps(result.parsed, ensure_ascii=False, indent=2)[:1500]}\n"
        f"  (skill_run_id={result.skill_run_id}; event recorded)"
    )


def _exec_record_event(args: dict[str, Any], deps: HarnessDeps) -> str:
    kind = (args.get("kind") or "").strip()
    if not kind:
        return "ERROR: record_event requires kind"
    job_id = args.get("job_id")
    note = args.get("note", "")
    event_id = _record_event_row(
        deps,
        kind=kind,
        job_id=job_id if isinstance(job_id, int) else None,
        note=note,
    )
    return f"OK event#{event_id} recorded (kind={kind}, job_id={job_id})"


def _exec_notify_user(args: dict[str, Any], deps: HarnessDeps) -> str:
    title = (args.get("title") or "").strip()
    body = (args.get("body") or "").strip()
    if not title or not body:
        return "ERROR: notify_user requires both title and body"
    related_job_id = args.get("related_job_id")
    priority = args.get("priority", "info")

    from .. import inbox as _inbox

    payload: dict[str, Any] = {"priority": priority}
    if isinstance(related_job_id, int):
        payload["related_job_id"] = related_job_id
    item = _inbox.enqueue_agent_suggestion(
        deps.store,
        title=title[:200],
        body=body,
        source_agent_run_id=deps.current_run_id,
        payload=payload,
    )
    if deps.notifier is not None:
        try:
            deps.notifier.notify(title=title, body=body[:500], level=priority)
        except Exception as e:
            log.warning("external notify failed: %s", e)
    return (
        f"OK notify_user → inbox#{item.id} (title={title[:60]!r}, priority={priority})"
    )


def _exec_ask_user(args: dict[str, Any], deps: HarnessDeps) -> str:
    question = (args.get("question") or "").strip()
    context = (args.get("context") or "").strip()
    options = args.get("options") or []
    if not question:
        return "ERROR: ask_user requires question"
    if not context:
        return "ERROR: ask_user requires context (why are you asking)"
    if not isinstance(options, list) or len(options) < 2:
        return "ERROR: ask_user requires ≥2 options (the user clicks one)"

    from .. import inbox as _inbox

    item = _inbox.enqueue_question(
        deps.store,
        question=question,
        context=context,
        options=options,
        source_agent_run_id=deps.current_run_id,
    )
    return (
        f"OK ask_user → inbox#{item.id}. Wait for user answer (will appear "
        f"in user_facts on next wake)."
    )


def _exec_schedule_next_wake(args: dict[str, Any], deps: HarnessDeps) -> str:
    delay = args.get("delay_seconds")
    reason = (args.get("reason") or "").strip()
    if not isinstance(delay, int):
        return "ERROR: schedule_next_wake requires integer delay_seconds"
    if delay < 60:
        return "ERROR: delay_seconds must be ≥ 60"
    if delay > 30 * 86400:
        return "ERROR: delay_seconds must be ≤ 30 days (2592000)"
    if not reason:
        return "ERROR: schedule_next_wake requires reason"

    fire_at_jd = _seconds_from_now_julianday(delay)
    with deps.store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO harness_scheduled_wakes(fire_at, reason, requested_by_run_id) "
            "VALUES (?, ?, ?) RETURNING id",
            (fire_at_jd, reason, deps.current_run_id),
        )
        wake_id = int(cur.fetchone()[0])
    return (
        f"OK scheduled_wake#{wake_id}: harness will wake you in {delay}s "
        f"({delay // 60}min). Reason: {reason!r}"
    )


def _exec_web_search(args: dict[str, Any], deps: HarnessDeps) -> str:
    if deps.search is None:
        return "ERROR: web_search requires SearchBackend (TAVILY_API_KEY not set)"
    q = (args.get("query") or "").strip()
    if not q:
        return "ERROR: web_search requires query"
    max_n = args.get("max_results") or 6
    max_n = max(1, min(int(max_n), 10))
    try:
        hits = deps.search.search(q, max_results=max_n)
    except Exception as e:
        return f"ERROR: search failed: {e}"
    if not hits:
        return f"OK 0 hits for {q!r}"
    lines = [f"OK {len(hits)} hits for {q!r}:"]
    for i, h in enumerate(hits, 1):
        lines.append(
            f"  {i}. {h.title[:80]}\n"
            f"     {h.url}\n"
            f"     {h.snippet[:200]}"
        )
    return "\n".join(lines)[:3500]


def _exec_fetch_url(args: dict[str, Any], deps: HarnessDeps) -> str:
    url = (args.get("url") or "").strip()
    if not url:
        return "ERROR: fetch_url requires url"
    try:
        r = httpx.get(url, follow_redirects=True, timeout=12.0,
                      headers={"User-Agent": "Mozilla/5.0"})
    except httpx.HTTPError as e:
        return f"ERROR: fetch failed: {e}"
    if r.status_code != 200:
        return f"ERROR: HTTP {r.status_code}"
    text = _strip_html_to_text(r.text)
    if len(text) < 80:
        return f"ERROR: too short ({len(text)} chars) — likely anti-scraping"
    return f"OK ({len(text)} chars):\n{text[:6000]}"


# ── Dispatch table ────────────────────────────────────────────────────


def _exec_detect_evolution_candidates(args: dict[str, Any], deps: HarnessDeps) -> str:
    from ..evolution.fitness import detect_evolution_candidates
    try:
        triggers = detect_evolution_candidates(deps.store)
    except Exception as e:
        return f"ERROR: detect_evolution_candidates failed: {type(e).__name__}: {e}"
    if not triggers:
        return "OK no SKILLs ripe for evolution right now (all above threshold or in cooldown)."
    lines = ["Found {} candidates ripe for evolution:".format(len(triggers))]
    for t in triggers[:10]:
        lines.append(
            f"  · {t.skill_name} v{t.current_version} fitness={t.fitness:.2f} "
            f"samples={t.sample_count} — {t.reason}"
        )
    return "\n".join(lines)


def _exec_evolve_skill(args: dict[str, Any], deps: HarnessDeps) -> str:
    if deps.llm is None:
        return "ERROR: evolve_skill requires LLM (DEEPSEEK_API_KEY not set)"
    skill_name = (args.get("skill_name") or "").strip()
    if not skill_name:
        return "ERROR: evolve_skill requires skill_name"
    num_variants = max(1, min(int(args.get("num_variants") or 3), 5))
    from ..evolution.evolve import evolve_skill as _evolve
    try:
        result = _evolve(
            store=deps.store, llm=deps.llm,
            skill_name=skill_name, num_variants=num_variants,
        )
    except Exception as e:
        return f"ERROR: evolve_skill failed: {type(e).__name__}: {e}"
    return (
        f"OK evolve_skill {skill_name}: parent v{result.parent_version}, "
        f"generated {result.candidates_generated}, persisted "
        f"{result.candidates_persisted} as shadow variants "
        f"({result.variant_versions}). Note: {result.notes[:200]}"
    )


def _exec_run_release_cycle(args: dict[str, Any], deps: HarnessDeps) -> str:
    from ..evolution.release import run_release_cycle
    dry_run = bool(args.get("dry_run") or False)
    try:
        cycle = run_release_cycle(deps.store, dry_run=dry_run)
    except Exception as e:
        return f"ERROR: run_release_cycle failed: {type(e).__name__}: {e}"
    return cycle.render_summary()


def _register_main_tools() -> None:
    """Register all main-agent tools into the global ToolRegistry.

    Runs once at module load. The registry then drives both schema
    enumeration (``registry.get_schemas('main')``) and dispatch
    (``registry.dispatch(name, args, deps=...)``) — there is no second
    dispatch table that can drift. Handlers keep their ``(args, deps)``
    signature via the wrapper closure below.
    """
    import sys as _sys
    _this = _sys.modules[__name__]

    def _wrap(handler):
        def _impl(args: dict[str, Any], **rt: Any) -> str:
            return handler(args, rt["deps"])
        return _impl

    for schema, handler_name in _MAIN_TOOL_ENTRIES:
        handler = getattr(_this, handler_name)
        tool_name = schema["function"]["name"]
        # Re-register on hot-reload — registry.register overwrites with a warning.
        _registry.register(
            name=tool_name,
            group="main",
            schema=schema["function"],
            handler=_wrap(handler),
        )


# Run registration once at module load — all _exec_* are defined above.
_register_main_tools()


# ── helpers ────────────────────────────────────────────────────────────


def _format_jd_for_skill(job: dict[str, Any]) -> str:
    """Pack the structured job row back into one labeled text block.

    SKILLs declare a single `job_text` input (per their SKILL.md); we used to
    pass title/company/jd_text as separate keys, which the SkillRuntime then
    silently rejected with ValueError. This helper is the canonical 'flatten
    job row → SKILL job_text' bridge so all 4 score/tailor/interview/reflect
    call sites stay consistent.
    """
    parts: list[str] = []
    if job.get("title"):
        parts.append(f"# {job['title']}")
    if job.get("company"):
        parts.append(f"公司: {job['company']}")
    if job.get("location"):
        parts.append(f"地点: {job['location']}")
    if job.get("raw_text"):
        parts.append(job["raw_text"])
    return "\n".join(parts)


def _load_job(store: Store, job_id: int) -> dict[str, Any] | None:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, title, company, raw_text, location, url "
            "FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return dict(zip(["id", "title", "company", "raw_text", "location", "url"], row, strict=False))


def _record_event_row(
    deps: HarnessDeps, *, kind: str, job_id: int | None, note: str,
) -> int:
    with deps.store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO harness_events(kind, job_id, note, source) "
            "VALUES (?, ?, ?, ?) RETURNING id",
            (kind, job_id, note, "agent"),
        )
        return int(cur.fetchone()[0])


def _seconds_from_now_julianday(delay_seconds: int) -> float:
    """Convert a delay-from-now to a julianday timestamp (SQLite time format)."""
    import datetime as _dt
    target = _dt.datetime.now(_dt.UTC) + _dt.timedelta(seconds=delay_seconds)
    # julianday(0) = -4713-11-24 12:00:00 UTC
    epoch_ref = _dt.datetime(2000, 1, 1, 12, 0, 0, tzinfo=_dt.UTC)
    days_since_2000 = (target - epoch_ref).total_seconds() / 86400.0
    return 2451545.0 + days_since_2000  # julianday for 2000-01-01 12:00 UTC


_TAG = _re.compile(r"<[^>]+>")
_SPACE = _re.compile(r"[ \t\r]+")
_BLANK = _re.compile(r"\n{3,}")


def _strip_html_to_text(html: str) -> str:
    html = _re.sub(r"<script[^>]*>.*?</script>", "", html,
                   flags=_re.DOTALL | _re.IGNORECASE)
    html = _re.sub(r"<style[^>]*>.*?</style>", "", html,
                   flags=_re.DOTALL | _re.IGNORECASE)
    text = _TAG.sub(" ", html)
    text = _SPACE.sub(" ", text)
    text = _BLANK.sub("\n\n", text)
    return text.strip()


def _guess_title(body: str) -> str | None:
    """Heuristic: first non-trivial line."""
    for line in body.splitlines():
        line = line.strip()
        if 5 < len(line) < 80:
            return line
    return None


def _guess_company(body: str, url: str) -> str | None:
    """Heuristic: domain root, or first capitalized brand word."""
    m = _re.search(r"https?://(?:www\.)?([^/]+)", url)
    if m:
        host = m.group(1).split(":")[0]
        # strip TLD-like suffixes
        parts = host.split(".")
        if len(parts) >= 2:
            return parts[-2].title()
    return None


def _harness_dir() -> Path:
    return Path(__file__).parent
