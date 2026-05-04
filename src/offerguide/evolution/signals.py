"""W13.1 evolution signals — the multi-source feedback stream.

The W6 GEPA framework had a metric (``score_match_metric``) but no source
of ground truth — the trainset was hand-stitched from skill_runs without a
quality label. The W13.1 redesign replaces the single static metric with a
**signal stream**: every SKILL invocation can accumulate signals from many
sources, and ``fitness.compute_fitness`` aggregates them into one number
that drives evolution decisions.

Signal sources (each has a ``record_*`` function):

- **critic**: AgentLoop's self-critique LLM judges each agent run. Routed to
  the SKILLs that ran during the trajectory. Auto-written by AgentLoop.
- **user_thumbs**: When the user clicks 👍 / 👎 on an agent suggestion in
  the inbox. Carries the most weight because it's direct user signal.
- **app_outcome**: Application this run influenced reached an outcome
  (offer / interview / rejected / silent). Highest-quality signal but
  arrives weeks late.
- **follow_through**: Did the user actually execute the suggestion the
  agent made? Quick signal, less reliable than thumbs but always available.
- **eval_synthetic**: ``meta_evolve_skill`` running a candidate variant
  against recent real inputs and scoring with a critic LLM.

This module only WRITES signals. Reading + aggregation lives in
``fitness.py``. Generating new variants lives in ``evolve.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from ..memory import Store

log = logging.getLogger(__name__)

SignalKind = Literal[
    "critic",
    "user_thumbs",
    "app_outcome",
    "follow_through",
    "eval_synthetic",
]

# Default weights when fitness aggregates a SKILL's signals. The user_thumbs
# weight is highest because it's the most direct expression of preference;
# critic is mid because it's a model judging another model (good but biased);
# app_outcome is highest in theory but signal_weight is set per-record because
# attribution to a specific SKILL run is fuzzy (one application touches many
# SKILLs over weeks).
DEFAULT_WEIGHTS: dict[SignalKind, float] = {
    "critic": 1.0,
    "user_thumbs": 2.0,
    "app_outcome": 1.5,
    "follow_through": 0.8,
    "eval_synthetic": 0.6,
}


@dataclass(frozen=True)
class SignalRecord:
    """Lightweight read shape for a row in evolution_signals."""
    id: int
    skill_name: str
    skill_version: str
    skill_run_id: int | None
    signal_kind: SignalKind
    signal_value: float
    signal_weight: float
    notes: str | None
    created_at: float


# ─────────────────────────── write API ───────────────────────────


def record_critic_signal(
    store: Store,
    *,
    skill_name: str,
    skill_version: str,
    skill_run_id: int | None,
    score: float,
    notes: str | None = None,
) -> int | None:
    """Record a self-critique score (0..1) for one SKILL invocation.

    Called by AgentLoop after each ``critique`` event — but only for SKILLs
    that actually ran during the trajectory (we don't critique lookups).
    Returns the new evolution_signals.id, or None on failure.
    """
    if score is None:
        return None
    score = max(0.0, min(1.0, float(score)))
    return _insert_signal(
        store, skill_name=skill_name, skill_version=skill_version,
        skill_run_id=skill_run_id, kind="critic",
        value=score, weight=DEFAULT_WEIGHTS["critic"],
        notes=notes,
    )


def record_user_thumbs(
    store: Store,
    *,
    skill_name: str,
    skill_version: str,
    skill_run_id: int | None,
    thumbs: Literal[1, -1],
    notes: str | None = None,
) -> int | None:
    """Record a user 👍 (+1) or 👎 (-1) on a SKILL's output.

    Stored verbatim (-1 / +1, NOT mapped to 0..1) so fitness can detect
    "was this actively rejected" vs "was this just neutral".
    """
    if thumbs not in (1, -1):
        raise ValueError(f"thumbs must be -1 or +1, got {thumbs}")
    return _insert_signal(
        store, skill_name=skill_name, skill_version=skill_version,
        skill_run_id=skill_run_id, kind="user_thumbs",
        value=float(thumbs), weight=DEFAULT_WEIGHTS["user_thumbs"],
        notes=notes,
    )


def record_app_outcome(
    store: Store,
    *,
    skill_name: str,
    skill_version: str,
    skill_run_id: int | None,
    outcome: Literal["offer", "interview", "screening", "rejected", "silent"],
    weight: float | None = None,
) -> int | None:
    """Record that an application this SKILL run influenced reached a final state.

    Mapping outcome → numeric value:
      - offer / interview / screening → 1.0 (positive)
      - rejected / silent             → 0.0 (negative)

    The caller is expected to compute the weight based on attribution
    confidence (weight=1.0 if SKILL ran for *this exact* application;
    weight=0.3 if SKILL ran in a generic batch). Defaults to 1.0.
    """
    pos = {"offer", "interview", "screening"}
    neg = {"rejected", "silent"}
    if outcome in pos:
        value = 1.0
    elif outcome in neg:
        value = 0.0
    else:
        raise ValueError(f"unknown outcome '{outcome}'")
    return _insert_signal(
        store, skill_name=skill_name, skill_version=skill_version,
        skill_run_id=skill_run_id, kind="app_outcome",
        value=value,
        weight=weight if weight is not None else DEFAULT_WEIGHTS["app_outcome"],
        notes=f"outcome={outcome}",
    )


def record_follow_through(
    store: Store,
    *,
    skill_name: str,
    skill_version: str,
    skill_run_id: int | None,
    executed: bool,
    notes: str | None = None,
) -> int | None:
    """Record whether the user executed the agent's suggestion.

    Heuristic signal: if the suggestion sat in the inbox for > 7 days with
    no decision, we record executed=False (interpreted as "not useful").
    """
    return _insert_signal(
        store, skill_name=skill_name, skill_version=skill_version,
        skill_run_id=skill_run_id, kind="follow_through",
        value=1.0 if executed else 0.0,
        weight=DEFAULT_WEIGHTS["follow_through"],
        notes=notes,
    )


def record_synthetic_eval(
    store: Store,
    *,
    skill_name: str,
    skill_version: str,
    score: float,
    notes: str | None = None,
) -> int | None:
    """Record a meta_evolve_skill synthetic eval score on a candidate variant.

    Used during the variant generation step — meta_evolve_skill runs a
    proposed prompt against recent real inputs and judges with a critic.
    These signals attach to the candidate version (not yet live), so the
    promote/rollback logic can compare candidate fitness vs live fitness
    without polluting either with cross-version data.
    """
    score = max(0.0, min(1.0, float(score)))
    return _insert_signal(
        store, skill_name=skill_name, skill_version=skill_version,
        skill_run_id=None,  # synthetic; no real skill_run
        kind="eval_synthetic",
        value=score,
        weight=DEFAULT_WEIGHTS["eval_synthetic"],
        notes=notes,
    )


# ─────────────────────────── read helpers ───────────────────────────


def fetch_signals(
    store: Store,
    *,
    skill_name: str,
    skill_version: str | None = None,
    kind: SignalKind | None = None,
    since_julianday: float | None = None,
    limit: int = 1000,
) -> list[SignalRecord]:
    """Read signals matching filters. Used by fitness.compute_fitness."""
    where = ["skill_name = ?"]
    params: list = [skill_name]
    if skill_version is not None:
        where.append("skill_version = ?")
        params.append(skill_version)
    if kind is not None:
        where.append("signal_kind = ?")
        params.append(kind)
    if since_julianday is not None:
        where.append("created_at >= ?")
        params.append(since_julianday)
    sql = (
        "SELECT id, skill_name, skill_version, skill_run_id, signal_kind, "
        "       signal_value, signal_weight, notes, created_at "
        "FROM evolution_signals WHERE " + " AND ".join(where) +
        " ORDER BY created_at DESC LIMIT ?"
    )
    params.append(int(limit))
    with store.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        SignalRecord(
            id=r[0], skill_name=r[1], skill_version=r[2],
            skill_run_id=r[3], signal_kind=r[4],
            signal_value=r[5], signal_weight=r[6],
            notes=r[7], created_at=r[8],
        )
        for r in rows
    ]


# ─────────────────────────── internal ───────────────────────────


def _insert_signal(
    store: Store,
    *,
    skill_name: str,
    skill_version: str,
    skill_run_id: int | None,
    kind: SignalKind,
    value: float,
    weight: float,
    notes: str | None,
) -> int | None:
    """Single insert path so logging + error handling is centralized."""
    try:
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO evolution_signals("
                "  skill_name, skill_version, skill_run_id, "
                "  signal_kind, signal_value, signal_weight, notes"
                ") VALUES (?,?,?,?,?,?,?)",
                (skill_name, skill_version, skill_run_id,
                 kind, float(value), float(weight), notes),
            )
            return int(cur.lastrowid or 0)
    except Exception as e:
        log.warning(
            "evolution_signals INSERT failed (non-fatal): %s "
            "(skill=%s ver=%s kind=%s)",
            e, skill_name, skill_version, kind,
        )
        return None
