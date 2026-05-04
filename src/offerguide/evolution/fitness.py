"""W13.1 fitness — aggregate evolution_signals into a single score per SKILL version.

Why a separate module from signals.py: signals.py is the *write* path that
many places (AgentLoop, inbox routes, application_events) call into. fitness.py
is the *read* path called by:

- ``evolution.release.should_evolve(skill_name)`` — should this SKILL be
  evolved now? (cumulated fitness < threshold + enough samples + cooldown)
- ``evolution.release.compare_variants`` — given live + canary, who wins?
- ``/evolution`` UI — show the trend line

Fitness model:

    fitness(skill, version) = sum(value_i * weight_i) / sum(weight_i)

But signal_value lives on different scales:
- critic / app_outcome / follow_through / eval_synthetic: 0..1
- user_thumbs: -1 / +1

We normalize user_thumbs to 0..1 before aggregation: -1 → 0.0, +1 → 1.0.
That way fitness is always a clean 0..1 score (1.0 = perfect, 0.0 = useless).

The aggregation is INTENTIONALLY simple weighted-mean. Anything fancier
(time-decay, per-context fitness, etc.) is overengineering before we have
data. If we end up with 200+ signals per SKILL we can revisit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..memory import Store
from .signals import SignalKind, SignalRecord, fetch_signals

log = logging.getLogger(__name__)

# How many signals before we trust fitness as an evolution trigger.
# Less than this and we don't have enough data to make a stable judgment.
MIN_SIGNALS_FOR_TRIGGER = 10

# Below this fitness, the SKILL is candidate for evolution.
EVOLUTION_THRESHOLD = 0.55

# Don't re-evolve a SKILL more often than this (in days).
EVOLUTION_COOLDOWN_DAYS = 7


@dataclass(frozen=True)
class FitnessReport:
    """Result of compute_fitness — what fitness is + how confident we are."""
    skill_name: str
    skill_version: str | None
    """None means "across all versions" — used for total-SKILL view."""

    fitness: float | None
    """0..1; None when no signals at all."""

    sample_count: int
    """How many signals contributed."""

    by_kind: dict[str, tuple[float, int]]
    """Per-kind breakdown: {'critic': (avg_value, count), ...}"""


@dataclass(frozen=True)
class EvolutionTrigger:
    """Output of detect_evolution_candidates — is this SKILL ready to evolve?"""
    skill_name: str
    current_version: str
    fitness: float
    sample_count: int
    days_since_last_evolution: float | None
    reason: str
    """Human-readable explanation for the /evolution UI + agent's tool result."""


# ─────────────────────────── core scoring ───────────────────────────


def normalize_signal_value(kind: SignalKind, raw: float) -> float:
    """Map a signal_value into the unified 0..1 fitness scale.

    user_thumbs uses -1/+1 in storage; everything else is already 0..1.
    Anything outside the expected range gets clipped (defensive).
    """
    if kind == "user_thumbs":
        # -1 → 0.0, +1 → 1.0
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))
    return max(0.0, min(1.0, raw))


def compute_fitness(
    store: Store,
    *,
    skill_name: str,
    skill_version: str | None = None,
    since_julianday: float | None = None,
    limit: int = 500,
) -> FitnessReport:
    """Read all matching signals, return the weighted fitness + breakdown.

    ``skill_version=None`` aggregates across all versions (total view).
    ``skill_version='X'`` filters to one version (used for A/B comparison).
    """
    by_kind_sum: dict[str, float] = {}
    by_kind_n: dict[str, int] = {}
    weighted_sum = 0.0
    weight_sum = 0.0

    # Single fetch — even with limit=500 this is cheap on SQLite
    signals: list[SignalRecord] = fetch_signals(
        store, skill_name=skill_name, skill_version=skill_version,
        since_julianday=since_julianday, limit=limit,
    )

    for s in signals:
        normalized = normalize_signal_value(s.signal_kind, s.signal_value)
        weighted_sum += normalized * s.signal_weight
        weight_sum += s.signal_weight
        by_kind_sum[s.signal_kind] = by_kind_sum.get(s.signal_kind, 0.0) + normalized
        by_kind_n[s.signal_kind] = by_kind_n.get(s.signal_kind, 0) + 1

    fitness = (weighted_sum / weight_sum) if weight_sum > 0 else None
    by_kind = {
        k: (by_kind_sum[k] / by_kind_n[k], by_kind_n[k])
        for k in by_kind_sum
    }
    return FitnessReport(
        skill_name=skill_name,
        skill_version=skill_version,
        fitness=fitness,
        sample_count=len(signals),
        by_kind=by_kind,
    )


# ─────────────────────────── trigger detection ───────────────────────────


def detect_evolution_candidates(
    store: Store,
    *,
    min_samples: int = MIN_SIGNALS_FOR_TRIGGER,
    threshold: float = EVOLUTION_THRESHOLD,
    cooldown_days: float = EVOLUTION_COOLDOWN_DAYS,
) -> list[EvolutionTrigger]:
    """Scan all SKILLs that have a live variant; return the ones ripe for evolution.

    Conditions ALL must be true:
      1. SKILL has >= ``min_samples`` evolution_signals on its current live version
      2. live-version fitness < ``threshold``
      3. last evolution attempt for this SKILL > ``cooldown_days`` ago (or never)

    Used by:
      - ``meta_evolve_skill`` SKILL — agent calls this to find work
      - ``/evolution`` UI — show "ready to evolve" badge
    """
    triggers: list[EvolutionTrigger] = []

    # 1. Get all (skill_name, current_live_version) pairs from skill_variants
    #    For SKILLs with no live variant in skill_variants table, we fall back
    #    to whatever version is in skill_runs (the seed disk version).
    live_versions: dict[str, str] = {}
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT skill_name, version FROM skill_variants WHERE status = 'live'"
        ).fetchall()
        for name, ver in rows:
            live_versions[name] = ver
        # Add seed-only SKILLs that haven't been evolved yet (no live variant
        # row, just whatever's running off SKILL.md)
        seed_rows = conn.execute(
            "SELECT skill_name, skill_version, COUNT(*) AS n FROM skill_runs "
            "WHERE skill_name NOT IN (SELECT skill_name FROM skill_variants WHERE status='live') "
            "GROUP BY skill_name, skill_version "
        ).fetchall()
        for name, ver, _n in seed_rows:
            if name not in live_versions:
                live_versions[name] = ver

    # 2. For each (name, version), evaluate fitness + cooldown
    for skill_name, version in live_versions.items():
        report = compute_fitness(
            store, skill_name=skill_name, skill_version=version,
        )
        if report.sample_count < min_samples:
            continue
        if report.fitness is None or report.fitness >= threshold:
            continue

        # Check cooldown — when was the last evolution attempt?
        days_since = _days_since_last_evolution(store, skill_name)
        if days_since is not None and days_since < cooldown_days:
            continue

        reason = (
            f"fitness={report.fitness:.2f} < {threshold:.2f} "
            f"on {report.sample_count} signals"
        )
        if days_since is not None:
            reason += f"; last evolved {days_since:.1f} days ago"
        else:
            reason += "; never evolved before"

        triggers.append(EvolutionTrigger(
            skill_name=skill_name,
            current_version=version,
            fitness=report.fitness,
            sample_count=report.sample_count,
            days_since_last_evolution=days_since,
            reason=reason,
        ))

    # Worst fitness first (most urgent)
    triggers.sort(key=lambda t: t.fitness)
    return triggers


def _days_since_last_evolution(store: Store, skill_name: str) -> float | None:
    """How many days since the most recent EVOLUTION ATTEMPT for this SKILL.

    "Evolution attempt" = a variant with non-null parent_version (the seed
    has parent_version=null and doesn't count). Returns None when no
    evolution attempt has ever happened.

    This is what the cooldown gate checks — we don't want to re-evolve the
    same SKILL every day if last week's evolution failed.
    """
    with store.connect() as conn:
        row = conn.execute(
            "SELECT julianday('now') - MAX(created_at) "
            "FROM skill_variants "
            "WHERE skill_name = ? AND parent_version IS NOT NULL",
            (skill_name,),
        ).fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


# ─────────────────────────── A/B comparison ───────────────────────────


def compare_versions(
    store: Store,
    *,
    skill_name: str,
    version_a: str,
    version_b: str,
    min_signals_per_side: int = 5,
) -> dict[str, object]:
    """Compare two versions of a SKILL. Used by promote/rollback decisions.

    Returns dict with:
        ``a``, ``b``: FitnessReport for each version
        ``winner``: 'a' | 'b' | None (None when not enough data either side)
        ``delta``: fitness_b - fitness_a (positive = b is better)
        ``decisive``: True when delta > 0.05 AND both sides have enough samples
    """
    a = compute_fitness(store, skill_name=skill_name, skill_version=version_a)
    b = compute_fitness(store, skill_name=skill_name, skill_version=version_b)

    enough = (
        a.sample_count >= min_signals_per_side
        and b.sample_count >= min_signals_per_side
        and a.fitness is not None and b.fitness is not None
    )
    if not enough:
        return {
            "a": a, "b": b, "winner": None,
            "delta": None, "decisive": False,
        }

    # b - a so positive delta means "b is better"
    delta = b.fitness - a.fitness  # type: ignore[operator]
    decisive = abs(delta) > 0.05
    if not decisive:
        winner = None
    else:
        winner = "b" if delta > 0 else "a"
    return {
        "a": a, "b": b, "winner": winner,
        "delta": delta, "decisive": decisive,
    }
