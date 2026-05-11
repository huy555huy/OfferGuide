"""Tool registry — singleton that collects tool schemas + handlers.

Design choices (locked):

1. **Singleton, module-level** — `from offerguide.tools.registry import registry`.
2. **Self-registration** — each tool module calls `registry.register(...)`
   at module-import time. ``load_all_tools()`` walks the modules in
   order. Avoids the spaghetti of "where do I plug this in?".
3. **OpenAI function-call format** — `schema` follows the standard
   `{"type": "function", "function": {...}}` shape so any compatible LLM
   API works (DeepSeek, Claude, OpenAI, OpenRouter).
4. **Groups, not toolsets** — each tool has a `group` ('main' /
   'discovery' / 'evaluation' / 'outcome_review'). Main agent + each
   sub-agent fetch their own group via `get_schemas(group=...)`.
5. **Handler returns str** — JSON-encoded result. Errors as
   `{"error": "..."}`. This matches OpenAI tool-result message format
   and lets the LLM see structured errors.

Inspired by hermes-agent's `tools/registry.py` pattern but trimmed for
our use case (no MCP, no toolset aliases, no async — we never need them).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

ToolGroup = Literal[
    "main",            # Tools the main agent uses directly
    "discovery",       # Discovery sub-agent's internal tools
    "evaluation",      # Evaluation sub-agent's internal tools
    "outcome_review",  # Outcome-review sub-agent's internal tools
    "shared",          # Used by multiple groups (e.g., read_job)
]


@dataclass
class ToolEntry:
    """One registered tool."""

    name: str
    group: ToolGroup
    schema: dict[str, Any]
    """OpenAI function schema — the ``function`` payload, not wrapped."""
    handler: Callable[..., str]
    """Sync callable. Takes (args_dict, **kwargs); returns JSON str."""
    description: str = ""
    deprecated_aliases: tuple[str, ...] = field(default_factory=tuple)
    """Old names that map to this tool (so renames don't break agents)."""


class ToolRegistry:
    """Module-level singleton for tool registration + dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}
        self._alias_to_name: dict[str, str] = {}

    # ── Registration ──────────────────────────────────────────

    def register(
        self,
        name: str,
        *,
        group: ToolGroup,
        schema: dict[str, Any],
        handler: Callable[..., str],
        description: str = "",
        deprecated_aliases: tuple[str, ...] = (),
    ) -> None:
        """Register one tool. Called at module-import time."""
        if name in self._tools:
            log.warning(
                "Tool '%s' already registered (group=%s); overwriting with group=%s",
                name, self._tools[name].group, group,
            )
        entry = ToolEntry(
            name=name, group=group, schema=schema,
            handler=handler,
            description=description or schema.get("description", ""),
            deprecated_aliases=deprecated_aliases,
        )
        self._tools[name] = entry
        for alias in deprecated_aliases:
            self._alias_to_name[alias] = name

    def deregister(self, name: str) -> None:
        entry = self._tools.pop(name, None)
        if entry:
            for alias in entry.deprecated_aliases:
                self._alias_to_name.pop(alias, None)

    def reset(self) -> None:
        """Wipe all registrations. **Only use in tests.**"""
        self._tools.clear()
        self._alias_to_name.clear()

    # ── Query ────────────────────────────────────────────────

    def get(self, name: str) -> ToolEntry | None:
        if name in self._tools:
            return self._tools[name]
        # Try alias
        canonical = self._alias_to_name.get(name)
        if canonical:
            return self._tools.get(canonical)
        return None

    def names_in_group(self, group: ToolGroup) -> list[str]:
        """Sorted tool names belonging to a group."""
        return sorted(
            entry.name for entry in self._tools.values()
            if entry.group == group
        )

    def names_for_agent(
        self, primary_group: ToolGroup,
        *, include_shared: bool = True,
    ) -> list[str]:
        """All tool names an agent in ``primary_group`` can call.

        Includes 'shared' tools by default (cross-cutting things like
        read_job that every agent might need).
        """
        groups = {primary_group}
        if include_shared:
            groups.add("shared")
        return sorted(
            entry.name for entry in self._tools.values()
            if entry.group in groups
        )

    def get_schemas(
        self, primary_group: ToolGroup,
        *, include_shared: bool = True,
    ) -> list[dict[str, Any]]:
        """OpenAI tool definitions for an agent's group + shared.

        Returns list of ``{"type": "function", "function": {...}}`` dicts
        ready to pass to ``llm.chat_with_tools(tools=...)``.
        """
        names = self.names_for_agent(primary_group, include_shared=include_shared)
        out: list[dict[str, Any]] = []
        for name in names:
            entry = self._tools[name]
            fn_schema = {**entry.schema}
            # Ensure schema has the 'name' field
            fn_schema.setdefault("name", entry.name)
            out.append({"type": "function", "function": fn_schema})
        return out

    def all_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def all_entries(self) -> list[ToolEntry]:
        return list(self._tools.values())

    # ── Dispatch ─────────────────────────────────────────────

    def dispatch(
        self, name: str, args: dict[str, Any],
        **runtime_kwargs: Any,
    ) -> str:
        """Execute a tool by name. Always returns JSON str.

        ``runtime_kwargs`` is for cross-cutting concerns like ``store``,
        ``settings``, ``runtime`` that handlers need but aren't model-
        visible parameters. The handler signature is
        ``handler(args: dict, **runtime_kwargs) -> str``.

        Tool exceptions are caught and turned into a JSON error string,
        so the LLM sees structured failures and can decide to retry.
        """
        entry = self.get(name)
        if entry is None:
            return tool_error(
                f"unknown tool: {name}",
                available=self.all_names()[:30],
            )
        try:
            result = entry.handler(args, **runtime_kwargs)
            if not isinstance(result, str):
                # Defensive: handler returned a dict by mistake
                return tool_result(result)
            return result
        except Exception as e:
            log.exception("tool %s dispatch raised", name)
            return tool_error(
                f"tool '{name}' raised {type(e).__name__}",
                detail=str(e)[:300],
            )


# Module-level singleton — imported as `from offerguide.tools.registry import registry`
registry = ToolRegistry()


# ── JSON response helpers ─────────────────────────────────────────


def tool_error(message: str, **extra: Any) -> str:
    """Standard JSON error string for tool handlers."""
    payload: dict[str, Any] = {"error": str(message)}
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def tool_result(data: Any = None, **kwargs: Any) -> str:
    """Standard JSON success string for tool handlers.

    Either pass a dict/list/value as ``data``, or use kwargs to build
    a dict inline. Not both.
    """
    if data is not None and kwargs:
        raise ValueError("tool_result: pass either data or kwargs, not both")
    if data is not None:
        return json.dumps(data, ensure_ascii=False, default=str)
    return json.dumps(kwargs, ensure_ascii=False, default=str)
