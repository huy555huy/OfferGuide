"""Small name-to-handler registry for the conversation agent's tools."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

ToolGroup = Literal["main"]


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


class ToolRegistry:
    """Module-level singleton for tool registration + dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    # ── Registration ──────────────────────────────────────────

    def register(
        self,
        name: str,
        *,
        group: ToolGroup,
        schema: dict[str, Any],
        handler: Callable[..., str],
        description: str = "",
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
        )
        self._tools[name] = entry

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)

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
