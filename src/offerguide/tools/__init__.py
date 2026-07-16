"""Registry primitives used by the in-product conversation agent."""

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
    "registry",
    "tool_error",
    "tool_result",
]
