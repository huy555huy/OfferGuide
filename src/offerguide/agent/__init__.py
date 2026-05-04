"""The OfferGuide agent layer — W13 central agent loop.

Model in the driver's seat: model decides which SKILL to call (via OpenAI
tool-calling), when to stop, how to compose tools. The W4 LangGraph
``build_graph`` was retired in W13.1 — the hardcoded ``requested_action``
enum routing was the canonical "model is a JSON formatter" pattern that
W13 set out to replace.
"""

from .loop import (
    DEFAULT_MAX_ITERATIONS,
    AgentEvent,
    AgentLoop,
    AgentRunResult,
    build_tool_schemas,
    snapshot_state,
)

__all__ = [
    "AgentEvent",
    "AgentLoop",
    "AgentRunResult",
    "DEFAULT_MAX_ITERATIONS",
    "build_tool_schemas",
    "snapshot_state",
]
