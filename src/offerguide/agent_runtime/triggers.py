"""Build the trigger used by a synchronous conversation request."""

from __future__ import annotations

from .loop import TriggerEvent


def make_user_input_trigger(message: str) -> TriggerEvent:
    """Frame one user message for the in-product conversation agent."""
    return TriggerEvent(kind="user_input", detail={"message": message[:2000]})
