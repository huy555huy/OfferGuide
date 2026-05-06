"""W15 harness — minimal scaffold + agent self-determination.

Structure:
- ``instructions.md`` — agent's "soul" prompt (loaded by context.py)
- ``memory.py`` — Memory tool (6 commands on .offerguide/worldview/)
- ``context.py`` — context assembly + self-implemented compaction & clearing
- ``tools.py`` — 13 tool schemas + dispatch (job-hunt specific)
- ``loop.py`` — single-threaded master loop (the heart)
- ``triggers.py`` — event-driven primary path + cron fallback
- ``feedback.py`` — bridge user reactions → GEPA evolution signals
- ``_schema.py`` — harness-owned DB tables (runs / events / scheduled wakes)

Design principle (Anthropic + W15 user direction):
- **Harness is dumb on purpose.** It coordinates Claude's decisions; it
  doesn't make them.
- **Agency is in-context.** Planning, reflection, communication judgment,
  causal reasoning — all done by the model in one master loop, not in
  separate sub-systems.
- **Persistence is file-based.** Worldview is markdown the agent owns;
  harness just provides the I/O primitives.
- **Safety is at the boundary.** Permissions / hooks / sandboxing live
  outside the prompt; not encoded as "if X then Y" rules in the model
  context.

Entry points:
- ``harness.run_one(trigger, ...)`` — fire one loop with ready-built deps
- ``harness.build_deps(...)`` — build HarnessDeps for embedding contexts
- ``harness.fire_event(...)`` — record a lifecycle event (triggers next wake)
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
from .tools import HarnessDeps
from .triggers import (
    USER_LIFECYCLE_EVENTS,
    PendingTrigger,
    fire_event,
    make_cron_heartbeat,
    make_user_input_trigger,
    poll_pending,
)

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
) -> HarnessDeps:
    """Build a HarnessDeps for one-off harness runs.

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
    _schema.init_harness_schema(store)

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
        _log.warning("harness.build_deps: search backend init failed: %s", e)

    runtime = None
    skills: list[Any] = []
    if llm is not None:
        from ..skills import SkillRuntime, discover_skills
        runtime = SkillRuntime(llm=llm, store=store)
        try:
            skills_root = Path(__file__).parent.parent / "skills"
            skills = list(discover_skills(skills_root))
        except Exception as e:
            _log.warning("harness.build_deps: skill discovery failed: %s", e)
            skills = []

    profile_text: str | None = None
    resume_path = getattr(settings, "resume_pdf", None)
    if resume_path:
        try:
            from ..profile import load_resume_pdf
            prof = load_resume_pdf(resume_path)
            profile_text = prof.raw_resume_text
        except Exception as e:
            _log.warning("harness.build_deps: resume load failed: %s", e)
            profile_text = None

    notifier = None
    try:
        notifier = make_notifier(settings)
    except Exception as e:
        _log.warning("harness.build_deps: notifier init failed: %s", e)

    return HarnessDeps(
        settings=settings,
        store=store,
        memory_store=memory_store,
        llm=llm,
        runtime=runtime,
        skills=skills,
        search=search,
        notifier=notifier,
        user_profile_text=profile_text,
    )


def run_one(
    trigger: TriggerEvent,
    *,
    settings: Settings | None = None,
    deps: HarnessDeps | None = None,
    max_iterations: int = 20,
    system_facts: SystemFacts | None = None,
    temperature: float | None = None,
) -> RunResult:
    """Convenience: build deps + run one loop. For CLI / cron / triggers.

    Pass ``deps=`` if you've already built them (e.g. long-running
    process). Otherwise we build fresh per call (cheap enough; LLMClient
    is just an httpx wrapper).

    Args:
        temperature: optional override of loop default. Pass higher
            (e.g. 0.5) for chat triggers where you want creative interpretation
            of user intent; lower (e.g. 0.3) for cron heartbeats where you
            want deterministic decisions.
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
    "USER_LIFECYCLE_EVENTS",
    "HarnessDeps",
    "MemoryStore",
    "PendingTrigger",
    "RunResult",
    "SystemFacts",
    "TriggerEvent",
    "build_deps",
    "context",
    "default_worldview_dir",
    "feedback",
    "fire_event",
    "load_instructions",
    "loop",
    "make_cron_heartbeat",
    "make_user_input_trigger",
    "memory",
    "poll_pending",
    "run",
    "run_one",
    "tools",
    "triggers",
]
