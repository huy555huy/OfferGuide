"""In-product conversation agent loop.

Structure:
- ``instructions.md`` — agent's "soul" prompt (loaded by context.py)
- ``memory.py`` — Memory tool (6 commands on .offerguide/worldview/)
- ``context.py`` — context assembly + self-implemented compaction & clearing
- ``tools.py`` — job-hunt tool schemas + dispatch
- ``loop.py`` — single-threaded master loop (the heart)
- ``triggers.py`` — synchronous user-message framing
- ``feedback.py`` — bridge user reactions → GEPA evolution signals
- ``_schema.py`` — loop-owned DB tables (runs / events / work items)

Boundary note:
- This package is OfferGuide's in-product agent loop.
- **The loop is dumb on purpose.** It coordinates model decisions; it doesn't
  make them.
- **Agency is in-context.** Planning, reflection, communication judgment,
  causal reasoning — all done by the model in one master loop, not in
  separate sub-systems.
- **Persistence is file-based.** Worldview is markdown the agent owns; this
  loop just provides the I/O primitives.
- **Safety is at the boundary.** Permissions / hooks / sandboxing live
  outside the prompt; not encoded as "if X then Y" rules in the model
  context.

Entry points:
- ``agent_runtime.run_one(trigger, ...)`` — fire one loop with ready-built deps
- ``agent_runtime.build_deps(...)`` — build AgentRuntimeDeps for embedding contexts
- ``agent_runtime.make_user_input_trigger(...)`` — frame one synchronous user request
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import Settings
from ..memory import Store
from . import _schema, context, feedback, loop, memory, tools, triggers
from .context import SystemFacts, load_instructions
from .loop import RunResult, TriggerEvent, run
from .memory import MemoryStore
from .tools import AgentRuntimeDeps
from .triggers import make_user_input_trigger

_log = logging.getLogger(__name__)


def default_worldview_dir(settings: Settings | None = None) -> Path:
    """Resolve the on-disk location for the agent's worldview.

    Default: ``<repo>/.offerguide/worldview/``. Configurable via
    ``OFFERGUIDE_WORLDVIEW_DIR`` env var.
    """
    import os
    override = os.environ.get("OFFERGUIDE_WORLDVIEW_DIR")
    if override:
        return Path(override).expanduser().resolve()
    # Default — sibling of store.db
    return Path(".offerguide/worldview").resolve()


def build_deps(
    *,
    settings: Settings | None = None,
    worldview_dir: Path | None = None,
) -> AgentRuntimeDeps:
    """Build AgentRuntimeDeps for one-off agent runtime runs.

    Loads: Settings → Store → MemoryStore → LLMClient (if API key set)
    → SearchBackend (if Tavily key set) → SkillRuntime (if LLM set) →
    SKILLs (discovered) → Notifier (configured per env) → resume PDF.

    Anything missing → that field is None; tools relying on it return
    'ERROR: requires X' so the agent learns to avoid them.
    """
    from ..llm import LLMClient
    from ..ui.notify import make_notifier

    settings = settings or Settings.from_env()
    store = Store(settings.db_path)
    store.init_schema()
    _schema.init_agent_runtime_schema(store)

    wdir = worldview_dir or default_worldview_dir(settings)
    memory_store = MemoryStore(root=wdir)

    llm: LLMClient | None = None
    if settings.deepseek_api_key:
        llm = LLMClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )

    # Smell 5 fix (W15.12 review): log init failures instead of silent
    # except-pass. Without these warnings users see "tool requires X" errors
    # at runtime with no clue WHY init failed (network / missing dep / ...).
    search: Any | None = None
    try:
        from ..agentic.search import build_default_search
        search = build_default_search()
    except Exception as e:
        _log.warning("agent_runtime.build_deps: search backend init failed: %s", e)

    runtime = None
    skills: list[Any] = []
    if llm is not None:
        from ..skills import SkillRuntime, discover_skills
        runtime = SkillRuntime(llm=llm, store=store)
        try:
            skills_root = Path(__file__).parent.parent / "skills"
            skills = list(discover_skills(skills_root))
        except Exception as e:
            _log.warning("agent_runtime.build_deps: skill discovery failed: %s", e)
            skills = []

    research_agents = None
    if llm is not None:
        try:
            from ..research_agents.service import ResearchAgentService

            research_agents = ResearchAgentService(
                settings=settings,
                store=store,
                llm=llm,
            )
        except Exception as e:
            _log.warning("agent_runtime.build_deps: research agents init failed: %s", e)

    master_text: str | None = None
    resume_path = getattr(settings, "resume_pdf", None)
    if resume_path:
        try:
            from ..resume import ResumeWorkspaceRepository, load_resume_pdf
            master_source = load_resume_pdf(resume_path)
            master_text = ResumeWorkspaceRepository(store).effective_master_text(master_source)
        except Exception as e:
            _log.warning("agent_runtime.build_deps: resume load failed: %s", e)
            master_text = None

    notifier = None
    try:
        notifier = make_notifier(settings)
    except Exception as e:
        _log.warning("agent_runtime.build_deps: notifier init failed: %s", e)

    return AgentRuntimeDeps(
        settings=settings,
        store=store,
        memory_store=memory_store,
        llm=llm,
        runtime=runtime,
        skills=skills,
        search=search,
        notifier=notifier,
        user_profile_text=master_text,
        research_agents=research_agents,
    )


def run_one(
    trigger: TriggerEvent,
    *,
    settings: Settings | None = None,
    deps: AgentRuntimeDeps | None = None,
    max_iterations: int = 20,
    system_facts: SystemFacts | None = None,
    temperature: float | None = None,
) -> RunResult:
    """Convenience: build dependencies and run one conversation-agent turn.

    Pass ``deps=`` if you've already built them (e.g. long-running
    process). Otherwise we build fresh per call (cheap enough; LLMClient
    is just an httpx wrapper).

    Args:
        temperature: optional override of the loop default.
    """
    deps = deps or build_deps(settings=settings)
    kwargs: dict[str, Any] = {
        "trigger": trigger, "deps": deps,
        "max_iterations": max_iterations,
        "system_facts": system_facts,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    return run(**kwargs)


__all__ = [
    "AgentRuntimeDeps",
    "MemoryStore",
    "RunResult",
    "SystemFacts",
    "TriggerEvent",
    "build_deps",
    "context",
    "default_worldview_dir",
    "feedback",
    "load_instructions",
    "loop",
    "make_user_input_trigger",
    "memory",
    "run",
    "run_one",
    "tools",
    "triggers",
]
