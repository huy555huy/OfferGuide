"""Feedback channel — bridges user reactions into GEPA SKILL evolution.

Per W15 §1.3: "节制靠学不靠规则". The agent's calibration (what to push,
what to suppress, how often, what tone) is **learned** via GEPA, not
encoded in if-else.

This module is the **bridge**: when the user gives feedback (⊘ a
notification, accepts a suggestion, ignores N inbox items), we record
it into ``evolution_signals`` so the GEPA evolver picks it up on the
next run.

W14 already had ``evolution_signals`` infra (W6 schema). W15 just makes
sure the **right signals** flow from the **right user actions** with
**enough context** for GEPA to learn from.

Schema mapping (existing W6 evolution_signals):
- skill_name + skill_version + skill_run_id: identify which SKILL produced
  the artifact the user reacted to
- signal_kind: 'user_thumbs' (W6 standard), or 'user_question_answer' / 'user_ignored' (W15 new)
- signal_value: -1 .. +1 (negative for ⊘, positive for ✓, 0 for ignored)
- signal_weight: importance multiplier
- notes: free-text + JSON-encoded payload (option_id, free_text, etc.)
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass
from typing import Any

from ..memory import Store

log = logging.getLogger(__name__)


# Signal kinds — keep these stable; GEPA evolver discriminates on them.
SIGNAL_KIND_USER_THUMBS = "user_thumbs"
"""Used for accepted (+1) and rejected (-1) inbox items."""

SIGNAL_KIND_USER_ANSWERED_QUESTION = "user_question_answer"
"""User answered a question the agent asked."""

SIGNAL_KIND_USER_IGNORED = "user_ignored"
"""Soft negative — user saw notification but didn't act for N days."""


@dataclass
class FeedbackContext:
    """Everything the GEPA evolver needs about one feedback signal.

    The skill_run_id is the key linkage — it lets GEPA find the
    (inputs, output) pair that produced the thing the user reacted to.
    """

    signal_kind: str
    signal_value: float
    """-1 (strong negative) ... 0 (neutral) ... +1 (strong positive)."""

    skill_run_id: int | None = None
    """If the feedback is on a SKILL output (most cases), the skill_run_id
    that produced it. None for harness-level feedback (rare)."""

    related_inbox_id: int | None = None
    """Which inbox item the user reacted to. Recorded into notes JSON
    since the schema doesn't have a dedicated column."""

    user_text: str | None = None
    """Free-form user comment, if any."""

    metadata: dict[str, Any] | None = None
    """Extra context (option_id, days_ignored, etc.) — recorded into
    notes JSON."""

    signal_weight: float = 1.0


_NOTES_MAX_LEN = 2000


def record(store: Store, fb: FeedbackContext) -> int:
    """Record a feedback signal. Returns evolution_signals.id.

    Resolves skill_name / skill_version from skill_run_id when available.
    Without skill_run_id, signals get ('harness', 'v0') as a sentinel —
    GEPA will skip these (no SKILL to evolve), but the row stays for audit.
    """
    skill_name, skill_version = _lookup_skill_meta(store, fb.skill_run_id)
    notes_text = _build_notes_json(
        user_text=fb.user_text,
        metadata=fb.metadata,
        related_inbox_id=fb.related_inbox_id,
    )

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO evolution_signals("
            "  skill_name, skill_version, skill_run_id, "
            "  signal_kind, signal_value, signal_weight, notes"
            ") VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
            (
                skill_name, skill_version, fb.skill_run_id,
                fb.signal_kind, fb.signal_value, fb.signal_weight, notes_text,
            ),
        )
        sig_id = int(cur.fetchone()[0])
    log.info(
        "evolution signal #%d: %s value=%.2f (skill=%s/%s skill_run=%s inbox=%s)",
        sig_id, fb.signal_kind, fb.signal_value,
        skill_name, skill_version, fb.skill_run_id, fb.related_inbox_id,
    )
    return sig_id


def _build_notes_json(
    *,
    user_text: str | None,
    metadata: dict[str, Any] | None,
    related_inbox_id: int | None,
) -> str:
    """Produce a notes JSON string guaranteed to be valid + ≤ _NOTES_MAX_LEN.

    Bug 6 fix (W15.12 review): the old impl did
    ``json.dumps(payload)[:2000]`` which truncates mid-string when
    user_text/metadata is long → invalid JSON → GEPA evolver crashes
    reading the row. Now we shrink the inner fields BEFORE serializing.
    """
    payload: dict[str, Any] = {
        "user_text": (user_text or "")[:1200] if user_text else None,
        "metadata": metadata or {},
        "related_inbox_id": related_inbox_id,
    }
    encoded = _json.dumps(payload, ensure_ascii=False)
    if len(encoded) <= _NOTES_MAX_LEN:
        return encoded
    # Still too long — strip metadata down to a marker
    payload["metadata"] = {"_truncated": True, "_orig_keys": list((metadata or {}).keys())}
    encoded = _json.dumps(payload, ensure_ascii=False)
    if len(encoded) <= _NOTES_MAX_LEN:
        return encoded
    # Last resort: drop user_text too
    payload["user_text"] = None
    payload["_dropped"] = "user_text and metadata for length"
    return _json.dumps(payload, ensure_ascii=False)[:_NOTES_MAX_LEN]


def _lookup_skill_meta(
    store: Store, skill_run_id: int | None,
) -> tuple[str, str]:
    """Resolve (skill_name, skill_version) from a skill_run_id, or sentinel."""
    if skill_run_id is None:
        return ("harness", "v0")
    try:
        with store.connect() as conn:
            row = conn.execute(
                "SELECT skill_name, skill_version FROM skill_runs WHERE id = ?",
                (skill_run_id,),
            ).fetchone()
        if row:
            return (str(row[0]), str(row[1]))
    except Exception:
        pass
    return ("harness", "v0")


# ── Convenience entry points ──────────────────────────────────────────


def on_inbox_accepted(
    store: Store, *, inbox_id: int, skill_run_id: int | None,
    user_text: str | None = None,
) -> int:
    """User clicked "yes/accept/thumbs up" on an inbox item."""
    return record(store, FeedbackContext(
        signal_kind=SIGNAL_KIND_USER_THUMBS,
        signal_value=1.0,
        related_inbox_id=inbox_id,
        skill_run_id=skill_run_id,
        user_text=user_text,
    ))


def on_inbox_rejected(
    store: Store, *, inbox_id: int, skill_run_id: int | None,
    user_text: str | None = None,
) -> int:
    """User clicked "no/⊘/thumbs down" on an inbox item."""
    return record(store, FeedbackContext(
        signal_kind=SIGNAL_KIND_USER_THUMBS,
        signal_value=-1.0,
        related_inbox_id=inbox_id,
        skill_run_id=skill_run_id,
        user_text=user_text,
    ))


def on_question_answered(
    store: Store, *, inbox_id: int, option_id: str, free_text: str | None,
) -> int:
    """User answered a question the agent asked them."""
    return record(store, FeedbackContext(
        signal_kind=SIGNAL_KIND_USER_ANSWERED_QUESTION,
        signal_value=0.5,  # neutral-positive (the answer itself is data)
        related_inbox_id=inbox_id,
        user_text=free_text,
        metadata={"option_id": option_id},
    ))


def on_inbox_ignored(
    store: Store, *, inbox_id: int, days_ignored: int,
) -> int:
    """Soft-negative: agent pushed something, user never engaged.

    Caller (a periodic job) computes who's stale and calls this. GEPA
    treats this as weaker negative than explicit ⊘.
    """
    return record(store, FeedbackContext(
        signal_kind=SIGNAL_KIND_USER_IGNORED,
        signal_value=-0.3,  # weak negative
        related_inbox_id=inbox_id,
        metadata={"days_ignored": days_ignored},
        signal_weight=0.5,  # half-weight vs explicit thumbs
    ))
