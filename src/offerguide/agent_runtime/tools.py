"""Tool schemas + dispatch for the W15 application agent loop.

These tools are **capabilities** the agent can choose; the agent decides
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
from . import _schema
from .memory import MEMORY_TOOL_SCHEMA, MemoryStore

if TYPE_CHECKING:
    from ..resume import ResumeWorkspace
    from ..skills import SkillInvoker, SkillSpec
else:
    SkillInvoker = Any  # type: ignore[assignment,misc]
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

Tool granularity: each tool is one verb of agent agency. Domain research tools
delegate to their sole owning Agent instead of reimplementing search or
interview-source Q&A in this general runtime. Tools below all return strings
(success or 'ERROR: ...') so the agent reads them directly in the next turn.
"""


# ── Dependency container ──────────────────────────────────────────────


@dataclass
class AgentRuntimeDeps:
    """Everything the dispatch functions need. Built once per agent run.

    Fields can be ``None`` for graceful degradation (e.g. ``llm=None``
    means LLM-using tools should return ERROR; the agent learns to
    avoid them).
    """

    settings: Settings
    store: Store
    memory_store: MemoryStore
    llm: LLMClient | None = None
    runtime: SkillInvoker | None = None
    skills: list[SkillSpec] = field(default_factory=list)
    search: Any = None
    notifier: Any = None
    user_profile_text: str | None = None
    research_agents: Any = None
    """Sole production gateway to JobDiscoveryAgent and InterviewResearchAgent."""
    current_run_id: int | None = None
    """Set by the loop; tools record references back to the run."""

    extra_cost_usd: float = 0.0
    """External LLM cost sink retained for non-domain supporting tools."""

    def find_skill(self, name: str) -> SkillSpec | None:
        for s in self.skills:
            if s.name == name:
                return s
        return None


# ── Tool schemas (OpenAI function-calling format) ──────────────────


_TOOL_UPDATE_WORK_ITEM: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_work_item",
        "description": (
            "Update the durable work item you actually advanced in this run. "
            "Use it after real work, not as a planning ritual. Mark done when "
            "the item is genuinely resolved; waiting when you are waiting on "
            "user/external time; blocked when a missing fact prevents progress; "
            "dismissed when the item is no longer worth doing. For waiting or "
            "blocked, include the next concrete action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "work_item_id": {
                    "type": "integer",
                    "description": "ID from the Agent Work Items context block.",
                },
                "status": {
                    "type": "string",
                    "enum": [
                        "in_progress",
                        "waiting",
                        "blocked",
                        "done",
                        "dismissed",
                    ],
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "What changed because of your work. For done, state "
                        "the result; for blocked/waiting, state the current state."
                    ),
                },
                "next_action": {
                    "type": "string",
                    "description": (
                        "Required for waiting/blocked/in_progress: what must "
                        "happen next, by whom, and what evidence will unblock it."
                    ),
                },
                "evidence": {
                    "type": "object",
                    "description": (
                        "Concrete pointers: job_id, skill_run_id, event_id, "
                        "inbox_id, page path, source URL, or reason."
                    ),
                    "additionalProperties": True,
                },
                "due_seconds": {
                    "type": "integer",
                    "description": (
                        "Optional delay from now for ordering this item."
                    ),
                },
            },
            "required": ["work_item_id", "status", "summary"],
        },
    },
}


_TOOL_RESEARCH_JOBS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "research_jobs",
        "description": (
            "Delegate job research to the sole JobDiscoveryAgent. Pass intent only when "
            "the user explicitly stated or edited the current search direction; omit it "
            "to refresh the existing visible search context. This returns an invocation, "
            "not a second list of jobs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": (
                        "The user's explicit natural-language job-search intent. Do not "
                        "turn assumptions or a single rejected job into global constraints."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
}


_TOOL_FETCH_JD: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_jd",
        "description": (
            "User pasted a job URL or text — pull it into the jobs table. "
            "Returns the job_id you can then score or track."
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


_TOOL_PREPARE_APPLICATION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "prepare_application",
        "description": (
            "Create the selected job's one current resume and complete application package "
            "through the same ResumeWorkflow used by the apply-pack page. Call only after "
            "the user has chosen this job; this does not submit anything."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
                "selection_revision": {
                    "type": "integer",
                    "description": (
                        "Required for a job chosen from JobDiscoveryAgent results; use "
                        "the exact result_revision shown to the user. Omit only for a "
                        "JD the user explicitly added through the manual fallback."
                    ),
                },
                "job_evidence_id": {
                    "type": "integer",
                    "description": (
                        "Required with selection_revision; use the exact evidence id "
                        "from the same published selection item."
                    ),
                },
            },
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
}


_TOOL_REVISE_APPLICATION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "revise_application",
        "description": (
            "Apply the user's concrete feedback to the same current resume, render it, "
            "visually review it when configured, and refresh the matching application package."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "job_id": {"type": "integer"},
                "feedback": {"type": "string"},
            },
            "required": ["job_id", "feedback"],
        },
    },
}


_TOOL_RERENDER_APPLICATION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "rerender_application",
        "description": (
            "Re-render the same current ResumeDocument with the confirmed Typst template. "
            "This never calls a content model and never changes resume wording."
        ),
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
        },
    },
}


_TOOL_RESEARCH_INTERVIEW: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "research_interview",
        "description": (
            "Delegate interview research to the sole InterviewResearchAgent for the "
            "exact submitted workspace. It reads complete public pages, pools real "
            "questions from similar roles across companies, and answers only questions "
            "substantiated by the accepted source bodies. The same company affects search "
            "priority only."
        ),
        "parameters": {
            "type": "object",
            "properties": {"job_id": {"type": "integer"}},
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
}


_TOOL_CAPTURE_PROJECT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "capture_project",
        "description": (
            "Turn a free-form project description into a truthful Project Vault "
            "assessment. Use when the user asks to整理项目/项目入库/判断是不是 "
            "AI Agent 项目/为简历或复试准备项目素材. The tool returns whether "
            "the project is truly agent-like, missing facts, risk flags, next "
            "questions, and suggested fields. If facts are missing, ask_user "
            "instead of pretending the project is ready."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "raw_project_note": {
                    "type": "string",
                    "description": "User's free-form project description or pasted project notes.",
                },
            },
            "required": ["raw_project_note"],
        },
    },
}


_TOOL_SAVE_PROJECT_RECORD: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "save_project_record",
        "description": (
            "Persist a confirmed truthful project record into Project Vault. "
            "Call only after the user provided enough facts or explicitly asked "
            "to save. Do not invent missing metrics or ownership. If key facts "
            "are missing, call ask_user first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "mainstream_direction": {"type": "string"},
                "project_task": {"type": "string"},
                "my_work": {"type": "string"},
                "typical_problem": {"type": "string"},
                "method_route": {"type": "string"},
                "market_context": {"type": "string"},
                "reference_sources": {"type": "string"},
                "contribution_type": {
                    "type": "string",
                    "enum": [
                        "method_innovation",
                        "engineering_improvement",
                        "application_transfer",
                        "process_improvement",
                        "integration",
                        "reproduction",
                        "main_contribution",
                    ],
                },
                "contribution_detail": {"type": "string"},
                "key_difficulties": {"type": "string"},
                "resolution_process": {"type": "string"},
                "project_outputs": {"type": "string"},
                "evidence": {"type": "string"},
                "askable_points": {"type": "string"},
                "expression_boundary": {"type": "string"},
                "do_not_claim": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {"type": "number"},
            },
            "required": [
                "title",
                "mainstream_direction",
                "project_task",
                "my_work",
            ],
        },
    },
}


_TOOL_READ_ARTIFACT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_artifact",
        "description": (
            "Read the current job-specific application workspace, its current cited "
            "interview material, a historical skill run, or a Project Vault record. "
            "Current application and interview material are always resolved through "
            "the exact resume workspace for the requested job."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_kind": {
                    "type": "string",
                    "enum": [
                        "current_application",
                        "current_interview_material",
                        "latest_project",
                        "skill_run",
                        "project_record",
                    ],
                    "description": "Which artifact to retrieve.",
                },
                "artifact_id": {
                    "type": "integer",
                    "description": "Required for skill_run or project_record.",
                },
                "job_id": {
                    "type": "integer",
                    "description": (
                        "Required for current_application and current_interview_material."
                    ),
                },
            },
            "required": ["artifact_kind"],
        },
    },
}


_TOOL_RECORD_EVENT: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record_event",
        "description": (
            "Record a user-confirmed lifecycle event for one exact submitted "
            "application. This writes the application event log and, when research "
            "is configured, refreshes it from the frozen submitted workspace. Do not use "
            "this for internal notes, future plans, or to claim submission; actual "
            "submission is confirmed and frozen in the application workspace UI."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "application_id": {
                    "type": "integer",
                    "description": "Exact application id from the current frozen artifact.",
                },
                "kind": {
                    "type": "string",
                    "enum": [
                        "viewed",
                        "replied",
                        "assessment",
                        "interview",
                        "rejected",
                        "offer",
                        "withdrawn",
                    ],
                },
                "note": {
                    "type": "string",
                    "description": "The user's factual detail about this event.",
                },
                "round": {
                    "type": "string",
                    "description": (
                        "For kind=interview, the explicit round stated by the user, "
                        "for example 一面, 二面, 终面, or HR."
                    ),
                },
                "scheduled_at": {
                    "type": "string",
                    "description": (
                        "Optional interview appointment time exactly as supplied by "
                        "the user or calendar; do not guess a timezone."
                    ),
                },
            },
            "required": ["application_id", "kind"],
            "additionalProperties": False,
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
            "confirmed deadlines / high-match jobs / important "
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


_TOOL_WEB_SEARCH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": ("Search the web (Tavily). Returns ≤ 10 hits with url + title + snippet."),
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


# Master tool list — order matters for prompt clarity (related tools grouped).
# The pair (_TOOL_SCHEMA, _exec_func) is the single source of truth: it's
# registered into the global ToolRegistry as group="main" at module load
# (via ``_register_main_tools`` at file end), and the LLM-facing list +
# the dispatch path both come out of that one registration.
_MAIN_TOOL_ENTRIES: list[tuple[dict[str, Any], str]] = [
    # Memory comes first — it's the agent's brain
    (MEMORY_TOOL_SCHEMA, "_exec_memory"),
    # Durable ownership — close/defer/block real work, not formal decisions
    (_TOOL_UPDATE_WORK_ITEM, "_exec_update_work_item"),
    # Active (agent should proactively call)
    (_TOOL_RESEARCH_JOBS, "_exec_research_jobs"),
    (_TOOL_NOTIFY_USER, "_exec_notify_user"),
    # Reactive (agent calls when user brings it)
    (_TOOL_FETCH_JD, "_exec_fetch_jd"),
    (_TOOL_PREPARE_APPLICATION, "_exec_prepare_application"),
    (_TOOL_REVISE_APPLICATION, "_exec_revise_application"),
    (_TOOL_RERENDER_APPLICATION, "_exec_rerender_application"),
    (_TOOL_RESEARCH_INTERVIEW, "_exec_research_interview"),
    (_TOOL_CAPTURE_PROJECT, "_exec_capture_project"),
    (_TOOL_SAVE_PROJECT_RECORD, "_exec_save_project_record"),
    (_TOOL_READ_ARTIFACT, "_exec_read_artifact"),
    # Universal capabilities
    (_TOOL_RECORD_EVENT, "_exec_record_event"),
    (_TOOL_ASK_USER, "_exec_ask_user"),
    (_TOOL_WEB_SEARCH, "_exec_web_search"),
    (_TOOL_FETCH_URL, "_exec_fetch_url"),
]


# Preserve order for the LLM prompt — registry sorts by name, but the prompt
# benefits from grouping related job-search tools together.
ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [schema for schema, _ in _MAIN_TOOL_ENTRIES]


# ── Dispatch ──────────────────────────────────────────────────────────


def dispatch(name: str, args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    """Route a tool call through the global ToolRegistry.

    Handlers for the main agent's tools are registered by
    ``_register_main_tools`` at module load. This wrapper:
    1. Returns ``'ERROR: unknown tool ...'`` for missing tools (the runtime
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


def _exec_memory(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    return deps.memory_store.execute(args)


def _exec_update_work_item(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    item_id = args.get("work_item_id")
    if not isinstance(item_id, int):
        return "ERROR: update_work_item requires integer work_item_id"
    status = (args.get("status") or "").strip()
    allowed = {"in_progress", "waiting", "blocked", "done", "dismissed"}
    if status not in allowed:
        return (
            "ERROR: update_work_item status must be one of "
            "in_progress, waiting, blocked, done, dismissed"
        )
    summary = (args.get("summary") or "").strip()
    if not summary:
        return "ERROR: update_work_item requires summary"
    next_action = (args.get("next_action") or "").strip()
    if status in {"in_progress", "waiting", "blocked"} and not next_action:
        return f"ERROR: update_work_item status={status} requires next_action"
    evidence = args.get("evidence")
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, dict):
        return "ERROR: update_work_item evidence must be an object"

    due_at = None
    due_seconds = args.get("due_seconds")
    if due_seconds is not None:
        if not isinstance(due_seconds, int):
            return "ERROR: update_work_item due_seconds must be an integer"
        if due_seconds < 60:
            return "ERROR: update_work_item due_seconds must be ≥ 60"
        if due_seconds > 30 * 86400:
            return "ERROR: update_work_item due_seconds must be ≤ 30 days (2592000)"
        due_at = _seconds_from_now_julianday(due_seconds)

    try:
        item = _schema.update_work_item(
            deps.store,
            work_item_id=item_id,
            status=status,
            summary=summary,
            evidence=evidence,
            next_action=next_action,
            last_run_id=deps.current_run_id,
            due_at=due_at,
        )
    except KeyError:
        return f"ERROR: work item {item_id} not found"
    except ValueError as e:
        return f"ERROR: {e}"

    if item.status in {"done", "dismissed"}:
        return f"OK work_item#{item.id} {item.status}: closed. summary={item.summary[:220]!r}"
    due = f", due_at={item.due_at}" if item.due_at is not None else ""
    return f"OK work_item#{item.id} {item.status}{due}: next_action={item.next_action[:220]!r}"


def _exec_research_jobs(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    service = deps.research_agents
    if service is None:
        return "ERROR: JobDiscoveryAgent is not configured"
    intent = str(args.get("intent") or "").strip()
    try:
        context = (
            service.replace_job_search_context(intent)
            if intent
            else service.job_repository.get_search_context()
        )
        if context is None:
            return "ERROR: no current job-search intent; pass the user's explicit intent"
        invocation, created = service.enqueue_job_discovery(
            trigger_reason="main agent delegated job research"
        )
    except Exception as exc:
        return f"ERROR: could not delegate JobDiscoveryAgent: {exc}"
    return "OK JobDiscoveryAgent invocation accepted:\n" + _json.dumps(
        {
            "invocation_id": invocation.id,
            "status": invocation.status,
            "context_revision": context.revision,
            "created": created,
            "view": "/recommended",
        },
        ensure_ascii=False,
    )


def _exec_research_interview(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    service = deps.research_agents
    if service is None:
        return "ERROR: InterviewResearchAgent is not configured"
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return "ERROR: research_interview requires integer job_id"
    try:
        application_id, workspace = _workspace_for_job(
            deps.store, job_id, require_submitted=True
        )
        invocation, created = service.enqueue_interview_research(
            application_id=application_id,
            workspace_id=workspace.id,
            trigger_reason="main agent delegated interview research",
        )
    except Exception as exc:
        return f"ERROR: could not delegate InterviewResearchAgent: {exc}"
    return "OK InterviewResearchAgent invocation accepted:\n" + _json.dumps(
        {
            "invocation_id": invocation.id,
            "status": invocation.status,
            "application_id": application_id,
            "submitted_workspace_id": workspace.id,
            "created": created,
            "view": f"/jobs/{job_id}/post-apply-pack",
        },
        ensure_ascii=False,
    )


def _exec_fetch_jd(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    raw = (args.get("url_or_text") or "").strip()
    if not raw:
        return "ERROR: fetch_jd requires url_or_text"
    company_hint = (args.get("company_hint") or "").strip()
    title_hint = (args.get("title_hint") or "").strip()

    from ..manual_job import ManualJobIntakeError, intake_manual_job
    from ..research_agents.sources import SourceEvidenceStore

    service = deps.research_agents
    source_store = (
        getattr(service, "source_store", None) or SourceEvidenceStore(deps.store)
    )
    try:
        result = intake_manual_job(
            store=deps.store,
            source_store=source_store,
            source_reader=getattr(service, "source_reader", None),
            url_or_text=raw,
            company_hint=company_hint,
            title_hint=title_hint,
        )
    except ManualJobIntakeError as exc:
        return f"ERROR: {exc}"
    if result.is_new:
        return (
            f"OK ingested as job#{result.job_id} ({result.company} · {result.title}). "
            f"The user can review the one application workspace at "
            f"/jobs/{result.job_id}/apply-pack."
        )
    return (
        f"NOTE: updated existing job#{result.job_id} "
        f"({result.company} · {result.title})"
    )


def _exec_prepare_application(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    job_id = args.get("job_id")
    if not isinstance(job_id, int) or isinstance(job_id, bool):
        return "ERROR: prepare_application requires integer job_id"
    selection_revision = args.get("selection_revision")
    job_evidence_id = args.get("job_evidence_id")
    for name, value in (
        ("selection_revision", selection_revision),
        ("job_evidence_id", job_evidence_id),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            return f"ERROR: prepare_application requires integer {name}"
    try:
        existing = _existing_workspace_for_job(deps.store, job_id)
    except LookupError as exc:
        return f"ERROR: cannot prepare application: {exc}"
    if existing is not None:
        if selection_revision is not None or job_evidence_id is not None:
            try:
                _application_start_snapshot(
                    deps,
                    job_id=job_id,
                    selection_revision=selection_revision,
                    job_evidence_id=job_evidence_id,
                    allow_closed=True,
                )
            except (LookupError, ValueError) as exc:
                return f"ERROR: cannot open current application: {exc}"
        return "OK current application already exists:\n" + _json.dumps(
            {
                "job_id": job_id,
                "application_id": existing.application_id,
                "workspace_id": existing.id,
                "status": existing.status,
                "pdf_path": existing.pdf_path,
                "pdf_sha256": existing.pdf_sha256,
                "review_url": f"/jobs/{job_id}/apply-pack",
                "submitted": existing.is_submitted,
            },
            ensure_ascii=False,
            indent=2,
        )
    return _run_resume_operation(
        deps,
        job_id=job_id,
        operation="start",
        selection_revision=selection_revision,
        job_evidence_id=job_evidence_id,
    )


def _exec_revise_application(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    job_id = args.get("job_id")
    feedback = str(args.get("feedback") or "").strip()
    if not isinstance(job_id, int):
        return "ERROR: revise_application requires integer job_id"
    if not feedback:
        return "ERROR: revise_application requires non-blank feedback"
    return _run_resume_operation(
        deps,
        job_id=job_id,
        operation="revise",
        feedback=feedback,
    )


def _exec_rerender_application(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    job_id = args.get("job_id")
    if not isinstance(job_id, int):
        return "ERROR: rerender_application requires integer job_id"
    return _run_resume_operation(deps, job_id=job_id, operation="rerender")


def _run_resume_operation(
    deps: AgentRuntimeDeps,
    *,
    job_id: int,
    operation: str,
    feedback: str | None = None,
    selection_revision: int | None = None,
    job_evidence_id: int | None = None,
) -> str:
    start_snapshot: dict[str, Any] | None = None
    if operation == "start":
        try:
            start_snapshot = _application_start_snapshot(
                deps,
                job_id=job_id,
                selection_revision=selection_revision,
                job_evidence_id=job_evidence_id,
            )
        except (LookupError, ValueError) as exc:
            return f"ERROR: cannot prepare application: {exc}"
    if deps.llm is None or deps.runtime is None:
        return "ERROR: application preparation requires the configured LLM and SkillRuntime"
    resume_path = deps.settings.resume_pdf
    if resume_path is None:
        return "ERROR: master PDF is not configured; set OFFERGUIDE_RESUME_PDF"

    visual_llm: LLMClient | None = None
    try:
        from ..resume import (
            ResumeEditor,
            ResumeWorkflow,
            generate_application_package,
            load_resume_pdf,
        )

        master_source = load_resume_pdf(resume_path)
        visual_editor = None
        if (
            deps.settings.vision_api_key
            and deps.settings.vision_base_url
            and deps.settings.vision_model
        ):
            visual_llm = LLMClient(
                api_key=deps.settings.vision_api_key,
                base_url=deps.settings.vision_base_url,
                default_model=deps.settings.vision_model,
            )
            visual_editor = ResumeEditor(visual_llm)
        workflow = ResumeWorkflow(
            store=deps.store,
            master_source=master_source,
            editor=ResumeEditor(deps.llm),
            visual_editor=visual_editor,
            apply_pack_writer=lambda job, context, result: generate_application_package(
                runtime=deps.runtime,
                skills=deps.skills,
                job_snapshot=job,
                context=context,
                editor_result=result,
            ),
            artifact_root=deps.settings.db_path.expanduser().resolve().parent
            / "resume_workspaces",
        )
        if operation == "start":
            result = workflow.start(job_id, job_snapshot=start_snapshot)
        elif operation == "revise":
            result = workflow.revise(job_id, feedback=feedback or "")
        elif operation == "rerender":
            result = workflow.rerender(job_id)
        else:
            return f"ERROR: unsupported resume operation {operation}"
    except Exception as exc:
        return (
            f"ERROR: {operation} application for job#{job_id} failed: {exc}. "
            f"Review or confirm the master text at /jobs/{job_id}/apply-pack."
        )
    finally:
        if visual_llm is not None:
            visual_llm.close()

    return "OK current application updated:\n" + _json.dumps(
        {
            "job_id": job_id,
            "application_id": result.workspace.application_id,
            "workspace_id": result.workspace.id,
            "status": result.workspace.status,
            "pdf_path": result.workspace.pdf_path,
            "pdf_sha256": result.workspace.pdf_sha256,
            "visual_review": result.visual_review_status,
            "editor_note": result.editor_result.editor_note,
            "review_url": f"/jobs/{job_id}/apply-pack",
            "submitted": False,
        },
        ensure_ascii=False,
        indent=2,
    )


def _application_start_snapshot(
    deps: AgentRuntimeDeps,
    *,
    job_id: int,
    selection_revision: int | None,
    job_evidence_id: int | None,
    allow_closed: bool = False,
) -> dict[str, Any] | None:
    """Bind a new workspace to one publication, or prove it is manual."""
    from ..research_agents.job_discovery import JobDiscoveryRepository
    from ..resume import job_snapshot_from_evidence

    if (selection_revision is None) != (job_evidence_id is None):
        raise ValueError(
            "selection_revision and job_evidence_id must be provided together"
        )
    if selection_revision is None:
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT source FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"job#{job_id} does not exist")
        if str(row[0] or "") != "manual":
            raise ValueError(
                "a JobDiscoveryAgent result requires the exact selection_revision "
                "and job_evidence_id shown to the user; bare job_id is reserved "
                "for a user-provided manual JD"
            )
        return None

    assert job_evidence_id is not None
    repository = (
        getattr(deps.research_agents, "job_repository", None)
        or JobDiscoveryRepository(deps.store)
    )
    selection = repository.get_selection_revision(selection_revision)
    if selection is None:
        raise LookupError(f"job selection revision {selection_revision} does not exist")
    selected = next(
        (
            item
            for item in selection.items
            if item.job_evidence_id == job_evidence_id
            and item.job_evidence is not None
            and item.job_evidence.job_id == job_id
        ),
        None,
    )
    if selected is None or selected.job_evidence is None:
        raise ValueError(
            "job_id and job_evidence_id do not belong to the selected publication"
        )
    live = repository.get_job_evidence(job_evidence_id)
    if live is not None and live.job_id != job_id:
        raise ValueError("the current job evidence identity no longer matches")
    if not allow_closed and live is not None and live.source_status == "closed":
        raise ValueError("the job's latest evidence says it is closed")
    snapshot = job_snapshot_from_evidence(
        selected.job_evidence.model_dump(mode="json")
    )
    snapshot["selection_result_revision"] = selection.result_revision
    snapshot["job_evidence_id"] = job_evidence_id
    return snapshot


def _exec_capture_project(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    note = (args.get("raw_project_note") or "").strip()
    if not note:
        return "ERROR: capture_project requires raw_project_note"
    from .. import project_vault

    try:
        assessment = project_vault.run_intake_agent(
            raw_project_note=note,
            search=deps.search,
            llm=deps.llm,
        )
    except Exception as e:
        return f"ERROR: capture_project failed: {type(e).__name__}: {e}"

    payload = {
        "is_agent_project": assessment.is_agent_project,
        "agent_reason": assessment.agent_reason,
        "next_action": assessment.next_action,
        "agent_signals": assessment.agent_signals,
        "non_agent_signals": assessment.non_agent_signals,
        "missing_facts": assessment.missing_facts,
        "risk_flags": assessment.risk_flags,
        "next_questions": assessment.next_questions,
        "suggested_fields": assessment.suggested_fields,
        "market_context": assessment.market_context,
        "reference_sources": assessment.reference_sources,
        "action_trace": assessment.action_trace or [],
    }
    try:
        _record_event_row(
            deps,
            kind="project_assessed",
            job_id=None,
            note=_json.dumps(
                {
                    "is_agent_project": assessment.is_agent_project,
                    "next_action": assessment.next_action,
                    "missing_facts": assessment.missing_facts,
                    "risk_flags": assessment.risk_flags,
                    "agent_run_id": deps.current_run_id,
                },
                ensure_ascii=False,
            ),
        )
    except Exception as e:
        log.debug("capture_project: record_event failed: %s", e)
    return (
        "OK capture_project assessment. If next_action is ask_user, ask the "
        "user the next_questions before saving. If ready or user explicitly "
        "asked to save, call save_project_record with suggested_fields; otherwise "
        "summarize the draft and risk flags.\n"
        + _json.dumps(payload, ensure_ascii=False, indent=2)[:3500]
    )


def _exec_save_project_record(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    from .. import project_vault

    tags = args.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    try:
        record = project_vault.insert(
            deps.store,
            title=str(args.get("title") or ""),
            mainstream_direction=str(args.get("mainstream_direction") or ""),
            typical_problem=_none_if_blank_arg(args.get("typical_problem")),
            project_task=str(args.get("project_task") or ""),
            my_work=str(args.get("my_work") or ""),
            method_route=_none_if_blank_arg(args.get("method_route")),
            market_context=_none_if_blank_arg(args.get("market_context")),
            reference_sources=_none_if_blank_arg(args.get("reference_sources")),
            contribution_type=str(args.get("contribution_type") or "main_contribution"),
            contribution_detail=_none_if_blank_arg(args.get("contribution_detail")),
            key_difficulties=_none_if_blank_arg(args.get("key_difficulties")),
            resolution_process=_none_if_blank_arg(args.get("resolution_process")),
            project_outputs=_none_if_blank_arg(args.get("project_outputs")),
            evidence=_none_if_blank_arg(args.get("evidence")),
            askable_points=_none_if_blank_arg(args.get("askable_points")),
            expression_boundary=_none_if_blank_arg(args.get("expression_boundary")),
            do_not_claim=_none_if_blank_arg(args.get("do_not_claim")),
            tags=[str(t).strip() for t in tags if str(t).strip()],
            confidence=float(args.get("confidence") or 0.5),
        )
    except ValueError as e:
        return f"ERROR: save_project_record missing required truthful field: {e}"
    except Exception as e:
        return f"ERROR: save_project_record failed: {type(e).__name__}: {e}"

    try:
        _record_event_row(
            deps,
            kind="project_record_saved",
            job_id=None,
            note=_json.dumps(
                {
                    "project_id": record.id,
                    "title": record.title,
                    "direction": record.mainstream_direction,
                    "view": "/project-vault",
                    "agent_run_id": deps.current_run_id,
                },
                ensure_ascii=False,
            ),
        )
    except Exception as e:
        log.debug("save_project_record: record_event failed: %s", e)

    return (
        f"OK save_project_record stored project#{record.id}: {record.title}. "
        "View it on /project-vault. Next: tell the user what was saved, "
        "what remains risky, and how it can feed resume tailoring/interview prep."
    )


def _exec_read_artifact(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    kind = (args.get("artifact_kind") or "").strip()
    artifact_id = args.get("artifact_id")
    job_id = args.get("job_id") if isinstance(args.get("job_id"), int) else None

    if kind == "skill_run":
        if not isinstance(artifact_id, int):
            return "ERROR: read_artifact skill_run requires artifact_id"
        return _render_skill_run_artifact(deps.store, artifact_id)

    if kind == "project_record":
        if not isinstance(artifact_id, int):
            return "ERROR: read_artifact project_record requires artifact_id"
        return _render_project_artifact(deps.store, artifact_id)

    if kind == "current_application":
        if job_id is None:
            return "ERROR: read_artifact current_application requires job_id"
        try:
            application_id, workspace = _workspace_for_job(
                deps.store,
                job_id,
                require_submitted=False,
            )
            from ..resume import ResumeDocument, document_text

            resume_text = (
                document_text(ResumeDocument.model_validate(workspace.resume_document))
                if workspace.resume_document
                else ""
            )
        except (LookupError, ValueError) as exc:
            return f"ERROR: {exc}"
        return "OK current application:\n" + _json.dumps(
            {
                "job_id": job_id,
                "application_id": application_id,
                "workspace_id": workspace.id,
                "workspace_status": workspace.status,
                "company": workspace.job_snapshot.get("company"),
                "title": workspace.job_snapshot.get("title"),
                "pdf_path": workspace.pdf_path,
                "pdf_sha256": workspace.pdf_sha256,
                "resume": resume_text,
                "editor_note": workspace.apply_pack.get("editor_note"),
            },
            ensure_ascii=False,
            indent=2,
        )

    if kind == "current_interview_material":
        if job_id is None:
            return "ERROR: read_artifact current_interview_material requires job_id"
        try:
            application_id, workspace = _workspace_for_job(
                deps.store,
                job_id,
                require_submitted=True,
            )
        except LookupError as exc:
            return f"ERROR: {exc}"
        from ..interview_research import InterviewResearchRepository

        subject = InterviewResearchRepository(deps.store).ensure_subject(
            application_id, workspace.id
        )
        if subject.current_material is None:
            return (
                f"OK no current interview material for submitted workspace #{workspace.id}. "
                f"Delegate research with research_interview(job_id={job_id})."
            )
        return "OK current interview material:\n" + _json.dumps(
            {
                "job_id": job_id,
                "application_id": application_id,
                "workspace_id": workspace.id,
                "stale": subject.current_material_is_stale,
                "material": subject.current_material.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )

    if kind == "latest_project":
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT id FROM project_records ORDER BY updated_at DESC, id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return "OK no project records saved yet. Ask the user for project facts first."
        return _render_project_artifact(deps.store, int(row[0]))

    return (
        "ERROR: read_artifact artifact_kind must be one of "
        "current_application, current_interview_material, latest_project, "
        "skill_run, project_record"
    )


def _exec_record_event(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    application_id = args.get("application_id")
    kind = (args.get("kind") or "").strip()
    valid_kinds = {
        "viewed",
        "replied",
        "assessment",
        "interview",
        "rejected",
        "offer",
        "withdrawn",
    }
    if not isinstance(application_id, int) or isinstance(application_id, bool):
        return "ERROR: record_event requires integer application_id"
    if kind not in valid_kinds:
        return (
            "ERROR: record_event kind must be one of "
            + ", ".join(sorted(valid_kinds))
        )

    from .. import application_events
    from ..resume import ResumeWorkspaceRepository
    from ..state_machine import sync_status

    workspace = ResumeWorkspaceRepository(deps.store).get(application_id)
    if workspace is None:
        return f"ERROR: application#{application_id} has no resume workspace"
    if not workspace.is_submitted:
        return (
            f"ERROR: application#{application_id} is not actually submitted and frozen; "
            "confirm submission in its application workspace first"
        )
    note = str(args.get("note") or "").strip()
    payload = {"note": note} if note else {}
    round_name = str(args.get("round") or "").strip()
    scheduled_at = str(args.get("scheduled_at") or "").strip()
    if round_name and kind != "interview":
        return "ERROR: record_event round is valid only for kind=interview"
    if scheduled_at and kind != "interview":
        return "ERROR: record_event scheduled_at is valid only for kind=interview"
    if round_name:
        payload["round"] = round_name
    if scheduled_at:
        payload["scheduled_at"] = scheduled_at
    try:
        event = application_events.record(
            deps.store,
            application_id=application_id,
            kind=kind,  # type: ignore[arg-type]
            source="manual",
            payload=payload,
        )
        sync_status(deps.store, application_id, kind, payload)
    except Exception as exc:
        return f"ERROR: could not record application event: {exc}"

    return "OK application lifecycle updated:\n" + _json.dumps(
        {
            "application_event_id": event.id,
            "application_id": application_id,
            "workspace_id": workspace.id,
            "kind": kind,
        },
        ensure_ascii=False,
        indent=2,
    )


def _exec_notify_user(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
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
    return f"OK notify_user → inbox#{item.id} (title={title[:60]!r}, priority={priority})"


def _exec_ask_user(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
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


def _exec_web_search(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
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
        lines.append(f"  {i}. {h.title[:80]}\n     {h.url}\n     {h.snippet[:200]}")
    return "\n".join(lines)[:3500]


def _exec_fetch_url(args: dict[str, Any], deps: AgentRuntimeDeps) -> str:
    url = (args.get("url") or "").strip()
    if not url:
        return "ERROR: fetch_url requires url"
    try:
        r = httpx.get(
            url, follow_redirects=True, timeout=12.0, headers={"User-Agent": "Mozilla/5.0"}
        )
    except httpx.HTTPError as e:
        return f"ERROR: fetch failed: {e}"
    if r.status_code != 200:
        return f"ERROR: HTTP {r.status_code}"
    text = _strip_html_to_text(r.text)
    if len(text) < 80:
        return f"ERROR: too short ({len(text)} chars) — likely anti-scraping"
    return f"OK ({len(text)} chars):\n{text[:6000]}"


def _register_main_tools() -> None:
    """Register all main-agent tools into the global ToolRegistry.

    Runs once at module load. ``ALL_TOOL_SCHEMAS`` drives model-visible
    enumeration, while this registry drives dispatch from the same
    ``_MAIN_TOOL_ENTRIES`` source. Handlers keep their ``(args, deps)``
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


def _existing_workspace_for_job(store: Store, job_id: int) -> ResumeWorkspace | None:
    """Return one existing workspace without manufacturing an application."""
    from ..resume import ResumeWorkspaceRepository

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT a.id FROM applications AS a "
            "JOIN resume_workspaces AS rw ON rw.application_id = a.id "
            "WHERE a.job_id = ? ORDER BY rw.updated_at DESC, rw.id DESC",
            (job_id,),
        ).fetchall()
    if len(rows) > 1:
        raise LookupError(
            f"job#{job_id} has {len(rows)} resume workspaces; "
            "resolve duplicate applications before continuing"
        )
    if not rows:
        return None
    workspace = ResumeWorkspaceRepository(store).get(int(rows[0][0]))
    if workspace is None:
        raise LookupError(f"job#{job_id} workspace disappeared while it was read")
    return workspace


def _workspace_for_job(
    store: Store,
    job_id: int,
    *,
    require_submitted: bool,
) -> tuple[int, ResumeWorkspace]:
    """Resolve one exact application workspace without guessing between duplicates."""
    from ..resume import ResumeWorkspaceRepository

    where_status = "AND rw.status = 'submitted'" if require_submitted else ""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT a.id, rw.id FROM applications a "
            "JOIN resume_workspaces rw ON rw.application_id = a.id "
            f"WHERE a.job_id = ? {where_status} "
            "ORDER BY rw.updated_at DESC, rw.id DESC",
            (job_id,),
        ).fetchall()
    state = "submitted " if require_submitted else ""
    if not rows:
        raise LookupError(
            f"job#{job_id} has no {state}resume workspace; "
            "generate and review the application package, then confirm actual submission"
        )
    if len(rows) != 1:
        raise LookupError(
            f"job#{job_id} has {len(rows)} {state}resume workspaces; "
            "resolve duplicate applications before continuing"
        )
    application_id = int(rows[0][0])
    workspace = ResumeWorkspaceRepository(store).get(application_id)
    if workspace is None:
        raise LookupError(f"application#{application_id} has no resume workspace")
    return application_id, workspace


def _record_event_row(
    deps: AgentRuntimeDeps,
    *,
    kind: str,
    job_id: int | None,
    note: str,
) -> int:
    from . import _schema

    _schema.init_agent_runtime_schema(deps.store)
    with deps.store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO harness_events(kind, job_id, note, source) "
            "VALUES (?, ?, ?, ?) RETURNING id",
            (kind, job_id, note, "agent"),
        )
        return int(cur.fetchone()[0])


def _none_if_blank_arg(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _render_skill_run_artifact(store: Store, skill_run_id: int) -> str:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, skill_name, skill_version, input_json, output_json "
            "FROM skill_runs WHERE id = ?",
            (skill_run_id,),
        ).fetchone()
    if row is None:
        return f"ERROR: skill_run#{skill_run_id} not found"

    parsed = _json_obj(row[4])
    view = _view_for_skill(str(row[1]))
    summary = _summarize_skill_output(str(row[1]), parsed)
    return (f"OK artifact skill_run#{row[0]} ({row[1]} v{row[2]}). View: {view}.\n{summary}")[:4000]


def _render_project_artifact(store: Store, project_id: int) -> str:
    from .. import project_vault

    record = project_vault.get(store, project_id)
    if record is None:
        return f"ERROR: project_record#{project_id} not found"
    lines = [
        f"OK artifact project#{record.id}: {record.title}. View: /project-vault.",
        f"- 方向: {record.mainstream_direction}",
        f"- 任务: {record.project_task}",
        f"- 我的真实工作: {record.my_work}",
        f"- 贡献类型: {record.contribution_label}",
    ]
    optional = (
        ("方法路线", record.method_route),
        ("外部表达参考", record.market_context),
        ("关键难点", record.key_difficulties),
        ("解决过程", record.resolution_process),
        ("证据", record.evidence),
        ("表达边界", record.expression_boundary),
        ("不要写", record.do_not_claim),
    )
    for label, value in optional:
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)[:4000]


def _json_obj(raw: str | None) -> dict[str, Any]:
    try:
        obj = _json.loads(raw or "{}")
    except Exception:
        return {"raw_text": raw or ""}
    return obj if isinstance(obj, dict) else {"value": obj}


def _view_for_skill(_skill_name: str) -> str:
    return "/dashboard"


def _summarize_skill_output(skill_name: str, parsed: dict[str, Any]) -> str:
    _ = skill_name
    return _json.dumps(parsed, ensure_ascii=False, indent=2)[:2500]


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
    html = _re.sub(r"<script[^>]*>.*?</script>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
    html = _re.sub(r"<style[^>]*>.*?</style>", "", html, flags=_re.DOTALL | _re.IGNORECASE)
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


# Backward-compatible alias for old imports. New code should use
# AgentRuntimeDeps.
HarnessDeps = AgentRuntimeDeps
