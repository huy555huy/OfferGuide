"""Tool layer — registry + tool modules for the agent system.

Architecture (locked):

- One ToolRegistry singleton, tools self-register at import time
- Each tool: name, schema (OpenAI function format), handler, group, doc
- Handlers return JSON strings (errors as ``{"error": "..."}`` JSON)
- Groups: 'main' (main agent tools) / 'discovery' / 'evaluation' /
  'outcome_review' (sub-agent tools)
- Modules in this dir register their tools at module-import time;
  ``load_all_tools()`` imports them in the right order to avoid circular
  imports
"""

from __future__ import annotations

from .registry import (
    ToolEntry,
    ToolRegistry,
    registry,
    tool_error,
    tool_result,
)

__all__ = [
    "ToolEntry",
    "ToolRegistry",
    "load_all_tools",
    "registry",
    "tool_error",
    "tool_result",
]


def load_all_tools() -> list[str]:
    """Import every tool module so its register() calls fire.

    Returns the list of imported module names. Call this once at process
    startup before any agent runs.
    """
    import importlib
    modules = [
        "offerguide.tools.delegate",
        "offerguide.tools.state_read",
        "offerguide.tools.state_write",
        "offerguide.tools.evolution",
        "offerguide.tools.discovery",
        "offerguide.tools.evaluation",
        "offerguide.tools.outcome_review",
        "offerguide.tools.self_inspect",
    ]
    imported: list[str] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
            imported.append(mod)
        except ImportError:
            # Module not yet implemented — skip
            pass
    return imported
