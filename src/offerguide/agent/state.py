"""State shape for the OfferGuide single agent.

W4 expansion: the W1 placeholder fields are joined by skill-result slots and a
`requested_action` enum that the W4 graph routes on. The W5+ LLM-driven triage
will populate `requested_action` from natural-language messages; today the UI
sets it explicitly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

RequestedAction = Literal["score", "gaps", "score_and_gaps"]


class AgentState(TypedDict, total=False):
    """LangGraph state for the OfferGuide agent."""

    messages: Annotated[list[Any], add_messages]

    # ---- inputs ----------------------------------------------------------
    requested_action: RequestedAction | None
    """Set by the UI (W4) or by an LLM-triage node (W5+). Drives routing."""

    job_text: str | None
    """Canonical JD text — what the SKILLs see as `job_text`."""

    user_profile_text: str | None
    """Resume text passed as `user_profile` to SKILLs."""

    job_id: int | None
    """Optional foreign key into the `jobs` table — set when the JD came from
    Scout/manual ingestion. Inbox items can reference this for traceability."""

    # ---- outputs (filled by skill nodes) --------------------------------
    score_result: dict[str, Any] | None
    score_run_id: int | None

    gaps_result: dict[str, Any] | None
    gaps_run_id: int | None

    # ---- terminal ------------------------------------------------------
    final_response: str | None
    error: str | None
