"""W13.1 evolution.release — gray-release orchestration.

The promote/rollback state machine that ties evolve.py + fitness.py +
registry.py together. Designed to be called periodically (by the agent
loop or on a cron) and to make small, reversible changes.

State machine:

    shadow ──── promote_to_canary(20%) ────▶ canary
                                             │
                                  collect ≥ N signals as canary
                                             │
                                             ▼
                                    compare_versions(canary vs live)
                                       /             \\
                                      /               \\
                          decisive WIN              decisive LOSE
                                /                      \\
                               ▼                        ▼
                         promote_to_live              fail_variant
                         (live becomes retired)

Every transition is small, audited (skill_variants.notes), and reversible
(retired variants can be promoted back to live by manual operator action).

The agent calls ``run_release_cycle()`` periodically — it picks AT MOST
one shadow to promote and one canary to evaluate per cycle, so the
system changes one thing at a time (debuggability > throughput).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..memory import Store
from .fitness import compare_versions
from .registry import (
    Variant,
    fail_variant,
    get_canary_variants,
    get_live_variant,
    get_shadow_variants,
    list_all_variants,
    promote_to_canary,
    promote_to_live,
    update_fitness_score,
)

log = logging.getLogger(__name__)

# Default canary traffic share when promoting from shadow → canary.
# Conservative: only 20% of invocations route to the new version while
# we collect signals.
DEFAULT_CANARY_TRAFFIC_PCT = 0.2

# How many signals a canary needs before we'll judge it vs live.
# Lower = decide faster but with noisier signal; higher = slower
# rollouts but more stable promotions.
MIN_CANARY_SIGNALS = 8

# How many signals each side needs in compare_versions.
MIN_SIGNALS_PER_SIDE = 5


@dataclass
class ReleaseAction:
    """One thing the release cycle did (or would do in dry_run mode)."""
    skill_name: str
    action: str  # 'promote_to_canary' | 'promote_to_live' | 'fail_canary' | 'noop'
    version: str | None
    reason: str


@dataclass
class ReleaseCycleResult:
    """Summary of one run_release_cycle invocation."""
    actions: list[ReleaseAction]
    skipped: list[str]  # skill names with shadows/canaries but no decision yet

    def render_summary(self) -> str:
        """One-paragraph human summary for the agent's tool result."""
        if not self.actions and not self.skipped:
            return "No shadow/canary variants exist; nothing to release."
        parts = []
        if self.actions:
            parts.append(f"Took {len(self.actions)} actions:")
            for a in self.actions:
                parts.append(
                    f"  - {a.skill_name} {a.action}"
                    + (f" {a.version}" if a.version else "")
                    + f": {a.reason}"
                )
        if self.skipped:
            parts.append(f"\n{len(self.skipped)} SKILLs in observation period:")
            for name in self.skipped:
                parts.append(f"  - {name}")
        return "\n".join(parts)


# ─────────────────────────── per-skill cycle ───────────────────────────


def run_release_cycle_for_skill(
    store: Store,
    skill_name: str,
    *,
    canary_traffic_pct: float = DEFAULT_CANARY_TRAFFIC_PCT,
    min_canary_signals: int = MIN_CANARY_SIGNALS,
    dry_run: bool = False,
) -> ReleaseAction:
    """Run one promotion/rollback decision for ``skill_name``.

    Decision tree:
      - Has a canary that's accumulated >= min_canary_signals signals?
        Compare to live → promote winner / fail loser
      - Has a live + shadow but no canary?
        Promote oldest shadow to canary (start the A/B)
      - Otherwise: noop (waiting for more data)

    Set ``dry_run=True`` to plan without mutating skill_variants.
    """
    canaries = get_canary_variants(store, skill_name)
    live = get_live_variant(store, skill_name)
    shadows = get_shadow_variants(store, skill_name)

    # Case A: there's a canary — see if it's ready to be judged
    if canaries:
        canary = canaries[0]
        return _judge_canary(
            store, skill_name=skill_name, canary=canary, live=live,
            min_canary_signals=min_canary_signals, dry_run=dry_run,
        )

    # Case B: no canary, but a shadow exists and there's a live to A/B against
    if shadows and live:
        shadow = shadows[-1]  # oldest first (FIFO)
        if dry_run:
            return ReleaseAction(
                skill_name=skill_name, action="promote_to_canary",
                version=shadow.version,
                reason=f"would promote shadow {shadow.version} → canary {canary_traffic_pct:.0%}",
            )
        ok = promote_to_canary(
            store, skill_name=skill_name, version=shadow.version,
            traffic_pct=canary_traffic_pct,
        )
        return ReleaseAction(
            skill_name=skill_name,
            action="promote_to_canary" if ok else "noop",
            version=shadow.version,
            reason=(
                f"promoted shadow {shadow.version} → canary at {canary_traffic_pct:.0%} traffic"
                if ok else "DB UPDATE returned 0 rows (variant gone?)"
            ),
        )

    # Case C: shadow but no live — first-time promotion to live (no A/B possible)
    if shadows and not live:
        shadow = shadows[-1]
        if dry_run:
            return ReleaseAction(
                skill_name=skill_name, action="promote_to_live",
                version=shadow.version,
                reason="would promote first variant directly to live (no live to A/B against)",
            )
        ok = promote_to_live(store, skill_name=skill_name, version=shadow.version)
        return ReleaseAction(
            skill_name=skill_name,
            action="promote_to_live" if ok else "noop",
            version=shadow.version,
            reason="first-time promotion (no live to A/B against)",
        )

    return ReleaseAction(
        skill_name=skill_name, action="noop", version=None,
        reason="nothing to do (no shadow/canary)",
    )


def _judge_canary(
    store: Store,
    *,
    skill_name: str,
    canary: Variant,
    live: Variant | None,
    min_canary_signals: int,
    dry_run: bool,
) -> ReleaseAction:
    """The A/B comparison — should this canary be promoted, killed, or kept?"""
    # If there's no live, just promote the canary
    if live is None:
        if dry_run:
            return ReleaseAction(
                skill_name=skill_name, action="promote_to_live",
                version=canary.version,
                reason="would promote canary directly (no live exists)",
            )
        promote_to_live(store, skill_name=skill_name, version=canary.version)
        return ReleaseAction(
            skill_name=skill_name, action="promote_to_live",
            version=canary.version,
            reason="canary promoted (no live existed)",
        )

    cmp = compare_versions(
        store, skill_name=skill_name,
        version_a=live.version, version_b=canary.version,
        min_signals_per_side=min_canary_signals,
    )
    fit_a = cmp["a"]
    fit_b = cmp["b"]

    # Cache fitness on the variants for /evolution UI
    if not dry_run:
        if fit_a.fitness is not None:
            update_fitness_score(store, skill_name=skill_name,
                                  version=live.version, fitness=fit_a.fitness)
        if fit_b.fitness is not None:
            update_fitness_score(store, skill_name=skill_name,
                                  version=canary.version, fitness=fit_b.fitness)

    # Not enough data yet — keep observing
    if not cmp["decisive"]:
        return ReleaseAction(
            skill_name=skill_name, action="noop", version=canary.version,
            reason=(
                f"observing: live={fit_a.sample_count} signals, "
                f"canary={fit_b.sample_count} signals "
                f"(need {min_canary_signals}+ each side for decision)"
            ),
        )

    if cmp["winner"] == "b":
        # Canary wins → promote
        if dry_run:
            return ReleaseAction(
                skill_name=skill_name, action="promote_to_live",
                version=canary.version,
                reason=f"would promote canary (fitness {fit_b.fitness:.2f} > live {fit_a.fitness:.2f})",
            )
        promote_to_live(store, skill_name=skill_name, version=canary.version)
        return ReleaseAction(
            skill_name=skill_name, action="promote_to_live",
            version=canary.version,
            reason=(
                f"canary won: fitness {fit_b.fitness:.2f} > live {fit_a.fitness:.2f}, "
                f"delta={cmp['delta']:+.2f}"
            ),
        )

    # winner == 'a' → live wins → fail canary
    if dry_run:
        return ReleaseAction(
            skill_name=skill_name, action="fail_canary",
            version=canary.version,
            reason=f"would fail canary (fitness {fit_b.fitness:.2f} < live {fit_a.fitness:.2f})",
        )
    fail_variant(
        store, skill_name=skill_name, version=canary.version,
        reason=f"lost A/B vs live: {fit_b.fitness:.2f} < {fit_a.fitness:.2f}",
    )
    return ReleaseAction(
        skill_name=skill_name, action="fail_canary",
        version=canary.version,
        reason=(
            f"canary failed: fitness {fit_b.fitness:.2f} < live {fit_a.fitness:.2f}, "
            f"delta={cmp['delta']:+.2f}"
        ),
    )


# ─────────────────────────── all-skill cycle ───────────────────────────


def run_release_cycle(
    store: Store,
    *,
    canary_traffic_pct: float = DEFAULT_CANARY_TRAFFIC_PCT,
    min_canary_signals: int = MIN_CANARY_SIGNALS,
    dry_run: bool = False,
) -> ReleaseCycleResult:
    """Run a release cycle across every SKILL that has any non-seed variants.

    This is what the agent calls (via the ``run_gray_release`` action tool)
    or what a cron periodically calls. Per-cycle limits mean each tick
    moves AT MOST one variant per SKILL through the state machine.
    """
    actions: list[ReleaseAction] = []
    skipped: list[str] = []

    # Gather all SKILL names that have any variants in the table
    all_variants = list_all_variants(store, limit=1000)
    skill_names = sorted({v.skill_name for v in all_variants})

    for skill_name in skill_names:
        action = run_release_cycle_for_skill(
            store, skill_name,
            canary_traffic_pct=canary_traffic_pct,
            min_canary_signals=min_canary_signals,
            dry_run=dry_run,
        )
        if action.action == "noop":
            skipped.append(f"{skill_name} ({action.reason})")
        else:
            actions.append(action)

    return ReleaseCycleResult(actions=actions, skipped=skipped)
