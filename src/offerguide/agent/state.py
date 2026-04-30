"""State shape for the OfferGuide single agent.

W4 expansion: the W1 placeholder fields are joined by skill-result slots and a
`requested_action` enum that the W4 graph routes on. The W5+ LLM-driven triage
will populate `requested_action` from natural-language messages; today the UI
sets it explicitly.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

RequestedAction = Literal[
    "score",
    "gaps",
    "score_and_gaps",
    "prepare_interview",
    "deep_prep",
    "cover_letter",
    "everything",
]
"""Action selector. ``everything`` runs all 5 SKILLs sequentially."""


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

    company: str | None
    """Optional company name for prepare_interview. When the action requires
    it but it's not provided, the prep node returns an error rather than
    guessing — accuracy beats convenience here."""

    past_experiences: str | None
    """Pre-rendered 面经 snippets for prepare_interview. The ``prep_node``
    will retrieve from ``interview_corpus`` if this is empty and the agent
    has store access."""

    job_id: int | None
    """Optional foreign key into the `jobs` table — set when the JD came
    from Scout/manual ingestion."""

    # ---- outputs (filled by skill nodes) --------------------------------
    score_result: dict[str, Any] | None
    score_run_id: int | None

    gaps_result: dict[str, Any] | None
    gaps_run_id: int | None

    prep_result: dict[str, Any] | None
    prep_run_id: int | None
    prep_used_experiences: int | None
    """Count of 面经 snippets the prep node used (for transparency in UI)."""

    deep_prep_result: dict[str, Any] | None
    deep_prep_run_id: int | None

    cover_letter_result: dict[str, Any] | None
    cover_letter_run_id: int | None

    # ---- terminal ------------------------------------------------------
    final_response: str | None
    error: str | None
