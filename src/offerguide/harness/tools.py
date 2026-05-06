"""Tool schemas + dispatch for the W15 harness.

13 tools total (memory + 12 from this module). Each tool is a
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

from __future__ import annotations

import json as _json
import logging
import re as _re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ..config import Settings
from ..llm import LLMClient
from ..memory import Store
from .memory import MEMORY_TOOL_SCHEMA, MemoryStore

if TYPE_CHECKING:
    from ..skills import SkillRuntime, SkillSpec
else:
    SkillRuntime = Any  # type: ignore[assignment,misc]
    SkillSpec = Any  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


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
    drive their own LLM calls (e.g. ``discover_jobs`` → JobFinderAgent)
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
            "Pass criteria from your worldview understanding of the user. "
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
                        "Pulled from your candidate.md understanding."
                    ),
                },
            },
            "required": ["criteria"],
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
            "Call this proactively when you find a job worth applying to."
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
            "insights. **Be selective** — over-notification trains the "
            "user to ignore you."
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
            "ownership** — e.g. after user marks 'applied X', call "
            "schedule_next_wake(7d, 'check if X responded'). Cron heartbeat "
            "is just fallback; agent-driven scheduling is the main path."
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


# Master registry — order matters for prompt clarity (related tools grouped).
ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    # Memory comes first — it's the agent's brain
    MEMORY_TOOL_SCHEMA,
    # Active (agent should proactively call)
    _TOOL_DISCOVER_JOBS,
    _TOOL_TAILOR_ADVICE,
    _TOOL_NOTIFY_USER,
    # Reactive (agent calls when user brings it)
    _TOOL_FETCH_JD,
    _TOOL_INTERVIEW_PREP,
    _TOOL_REFLECT_OUTCOME,
    # Universal capabilities
    _TOOL_SCORE_MATCH,
    _TOOL_RECORD_EVENT,
    _TOOL_ASK_USER,
    _TOOL_SCHEDULE_NEXT_WAKE,
    _TOOL_WEB_SEARCH,
    _TOOL_FETCH_URL,
]


# ── Dispatch ──────────────────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any], deps: HarnessDeps) -> str:
    """Route a tool call to its implementation. Returns string for LLM.

    Errors are returned as 'ERROR: ...' strings rather than raised, so
    the model can read the error and self-correct.
    """
    handler = _DISPATCH_TABLE.get(name)
    if handler is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        return handler(args, deps)
    except Exception as e:
        log.exception("tool %s crashed: args=%s", name, args)
        return f"ERROR: {type(e).__name__}: {e}"


# ── Tool implementations ─────────────────────────────────────────────


def _exec_memory(args: dict[str, Any], deps: HarnessDeps) -> str:
    return deps.memory_store.execute(args)


def _exec_discover_jobs(args: dict[str, Any], deps: HarnessDeps) -> str:
    if deps.llm is None:
        return "ERROR: discover_jobs requires LLM (DEEPSEEK_API_KEY not set)"
    if deps.search is None:
        return "ERROR: discover_jobs requires SearchBackend (TAVILY_API_KEY not set)"
    criteria = (args.get("criteria") or "").strip()
    if not criteria:
        return "ERROR: discover_jobs requires criteria (one sentence is fine)"
    from ..agentic.job_finder_agent import JobFinderAgent
    agent = JobFinderAgent(store=deps.store, llm=deps.llm, search=deps.search)
    try:
        result = agent.run(north_star=criteria)
    finally:
        agent.close()
    # Bug 5 fix: propagate sub-agent cost to harness telemetry. Without this,
    # 5 discover_jobs calls × ~$0.15 = $0.75 invisible in harness_runs.
    sub_cost = float(getattr(result, "total_cost_usd", 0.0) or 0.0)
    deps.extra_cost_usd += sub_cost
    summary = (
        f"OK discover_jobs done in {result.iterations} iters (sub-agent cost ${sub_cost:.4f}). "
        f"Inserted {result.inserted} new JDs (job_ids: {result.new_job_ids[:8]}). "
        f"Skipped {result.skipped_dup} dups. Finish: {result.finish_reason}"
    )
    if result.notes:
        # Last few decision steps for the agent to learn from
        summary += "\nLast 3 sub-agent steps:\n" + "\n".join(
            f"  · {n[:140]}" for n in result.notes[-3:]
        )
    return summary


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
    inputs = {
        "job_title": job.get("title", ""),
        "job_company": job.get("company", ""),
        "jd_text": job.get("raw_text", "")[:4000],
        "candidate_resume": deps.user_profile_text[:4000],
    }
    result = deps.runtime.invoke(spec, inputs)
    if result.parsed is None:
        # Smell 3 fix: SKILL output failed JSON parse — show agent the raw
        # text so it can self-correct (e.g. "model returned markdown code block")
        return (
            f"WARN score_match for job#{job_id}: SKILL output not valid JSON. "
            f"Raw output (first 500 chars):\n{result.raw_text[:500]}\n"
            f"(skill_run_id={result.skill_run_id})"
        )
    parsed = result.parsed
    return (
        f"OK score_match for job#{job_id} ({job.get('company')} · {job.get('title')}):\n"
        f"  score: {parsed.get('score', '?')}\n"
        f"  reasoning: {(parsed.get('reasoning') or '')[:300]}\n"
        f"  key_gaps: {parsed.get('key_gaps', [])}\n"
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
    inputs = {
        "job_title": job.get("title", ""),
        "company": job.get("company", ""),
        "jd_text": job.get("raw_text", "")[:4000],
        "current_resume": deps.user_profile_text[:6000],
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
    inputs = {
        "company": job.get("company", ""),
        "job_title": job.get("title", ""),
        "jd_text": job.get("raw_text", "")[:4000],
        "candidate_resume": (deps.user_profile_text or "")[:4000],
        "round": args.get("round", "未指定"),
    }
    result = deps.runtime.invoke(spec, inputs)
    if result.parsed is None:
        return (
            f"WARN interview_prep for job#{job_id}: SKILL output not valid JSON. "
            f"Raw (first 500 chars):\n{result.raw_text[:500]}\n"
            f"(skill_run_id={result.skill_run_id})"
        )
    return (
        f"OK interview_prep for job#{job_id} (round: {inputs['round']}):\n"
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

    inputs = {
        "company": job.get("company", ""),
        "job_title": job.get("title", ""),
        "outcome": outcome,
        "user_notes": args.get("user_notes", ""),
        "jd_text": job.get("raw_text", "")[:3000],
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


_DISPATCH_TABLE: dict[str, Callable[[dict[str, Any], HarnessDeps], str]] = {
    "memory": _exec_memory,
    "discover_jobs": _exec_discover_jobs,
    "fetch_jd": _exec_fetch_jd,
    "score_match": _exec_score_match,
    "tailor_advice": _exec_tailor_advice,
    "interview_prep": _exec_interview_prep,
    "reflect_outcome": _exec_reflect_outcome,
    "record_event": _exec_record_event,
    "notify_user": _exec_notify_user,
    "ask_user": _exec_ask_user,
    "schedule_next_wake": _exec_schedule_next_wake,
    "web_search": _exec_web_search,
    "fetch_url": _exec_fetch_url,
}


# ── helpers ────────────────────────────────────────────────────────────


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
