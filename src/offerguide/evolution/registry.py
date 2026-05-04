"""W13.1 evolution.registry — read/write the skill_variants table.

This is the data-access layer between fitness/release/evolve modules and
the SQLite ``skill_variants`` table. Centralized so that:

- Tests can use one fixture set
- The ``status`` state machine is enforced in one place (shadow → canary →
  live, retired/failed are terminal)
- The ``select_variant_for_invoke`` routing logic is in one place — used
  by ``SkillRuntime`` for every SKILL invocation
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Literal

from ..memory import Store

log = logging.getLogger(__name__)

VariantStatus = Literal["shadow", "canary", "live", "retired", "failed"]


@dataclass(frozen=True)
class Variant:
    """One row from skill_variants."""
    id: int
    skill_name: str
    version: str
    parent_version: str | None
    body_md: str
    spec_overrides: dict
    status: VariantStatus
    canary_traffic_pct: float
    fitness_score: float | None
    notes: str | None
    created_at: float
    promoted_at: float | None


@dataclass(frozen=True)
class VariantSelection:
    """Result of select_variant_for_invoke — what version should we use?"""
    use_disk_seed: bool
    """True when no live/canary variants exist; SkillRuntime should use the
    SkillSpec it loaded from disk (the SKILL.md). False when we have at least
    one variant in the DB to route to."""

    selected_variant: Variant | None
    """The DB-stored variant whose body should override the disk seed.
    None when use_disk_seed=True."""

    selection_reason: str = ""
    """Human-readable: 'live' / 'canary 20% won the dice' / etc. Goes into
    skill_runs.input_json so we can audit which version produced each output."""


# ─────────────────────────── reads ───────────────────────────


def get_live_variant(store: Store, skill_name: str) -> Variant | None:
    """Return the current 'live' variant for a SKILL, or None if seed-only."""
    return _query_one(
        store,
        "SELECT * FROM skill_variants "
        "WHERE skill_name = ? AND status = 'live' "
        "ORDER BY promoted_at DESC LIMIT 1",
        (skill_name,),
    )


def get_canary_variants(store: Store, skill_name: str) -> list[Variant]:
    """Return all canary variants for a SKILL (usually 0 or 1)."""
    return _query_many(
        store,
        "SELECT * FROM skill_variants "
        "WHERE skill_name = ? AND status = 'canary' "
        "ORDER BY promoted_at DESC",
        (skill_name,),
    )


def get_shadow_variants(store: Store, skill_name: str) -> list[Variant]:
    """Return all shadow variants (generated but not yet in production)."""
    return _query_many(
        store,
        "SELECT * FROM skill_variants "
        "WHERE skill_name = ? AND status = 'shadow' "
        "ORDER BY created_at DESC",
        (skill_name,),
    )


def get_variant_by_version(store: Store, skill_name: str, version: str) -> Variant | None:
    return _query_one(
        store,
        "SELECT * FROM skill_variants WHERE skill_name = ? AND version = ?",
        (skill_name, version),
    )


def list_all_variants(
    store: Store,
    *,
    skill_name: str | None = None,
    status: VariantStatus | None = None,
    limit: int = 100,
) -> list[Variant]:
    where: list[str] = []
    params: list = []
    if skill_name:
        where.append("skill_name = ?")
        params.append(skill_name)
    if status:
        where.append("status = ?")
        params.append(status)
    sql = "SELECT * FROM skill_variants"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY skill_name, created_at DESC LIMIT ?"
    params.append(int(limit))
    return _query_many(store, sql, tuple(params))


# ─────────────────────────── writes ───────────────────────────


def insert_shadow_variant(
    store: Store,
    *,
    skill_name: str,
    version: str,
    parent_version: str,
    body_md: str,
    spec_overrides: dict | None = None,
    notes: str | None = None,
) -> int | None:
    """Persist a freshly-generated variant (status='shadow').

    The parent_version MUST be non-null — shadow variants are always
    derived from an existing version (the seed or a previous live).
    Use bump_version() to compute a unique version string.
    """
    import json
    try:
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO skill_variants("
                "  skill_name, version, parent_version, body_md, spec_json, "
                "  status, canary_traffic_pct, notes"
                ") VALUES (?,?,?,?,?,'shadow', 0.0, ?)",
                (skill_name, version, parent_version, body_md,
                 json.dumps(spec_overrides or {}, ensure_ascii=False), notes),
            )
            return int(cur.lastrowid or 0)
    except Exception as e:
        log.warning("insert_shadow_variant failed: %s", e)
        return None


def promote_to_canary(
    store: Store,
    *,
    skill_name: str,
    version: str,
    traffic_pct: float = 0.2,
) -> bool:
    """Move a shadow variant to canary status with given traffic share.

    Refuses if the variant doesn't exist or isn't currently 'shadow'.
    """
    if not (0.0 < traffic_pct <= 1.0):
        raise ValueError(f"traffic_pct must be in (0, 1], got {traffic_pct}")
    with store.connect() as conn:
        cur = conn.execute(
            "UPDATE skill_variants "
            "SET status='canary', canary_traffic_pct=?, promoted_at=julianday('now') "
            "WHERE skill_name=? AND version=? AND status='shadow'",
            (traffic_pct, skill_name, version),
        )
        return cur.rowcount > 0


def promote_to_live(
    store: Store,
    *,
    skill_name: str,
    version: str,
) -> bool:
    """Promote a canary to live; demote any existing live for this SKILL to retired.

    Atomic-ish (single connection, two updates).
    """
    with store.connect() as conn:
        # Demote existing live (if any) to retired
        conn.execute(
            "UPDATE skill_variants SET status='retired' "
            "WHERE skill_name=? AND status='live'",
            (skill_name,),
        )
        cur = conn.execute(
            "UPDATE skill_variants "
            "SET status='live', canary_traffic_pct=0.0, promoted_at=julianday('now') "
            "WHERE skill_name=? AND version=? AND status IN ('canary', 'shadow')",
            (skill_name, version),
        )
        return cur.rowcount > 0


def fail_variant(
    store: Store,
    *,
    skill_name: str,
    version: str,
    reason: str | None = None,
) -> bool:
    """Mark a variant as failed (canary lost the A/B, or shadow eval was bad)."""
    with store.connect() as conn:
        cur = conn.execute(
            "UPDATE skill_variants SET status='failed', "
            "  notes = COALESCE(notes, '') || ? "
            "WHERE skill_name=? AND version=? AND status IN ('shadow', 'canary')",
            (f"\nFAIL: {reason or '(no reason)'}", skill_name, version),
        )
        return cur.rowcount > 0


def update_fitness_score(
    store: Store, *, skill_name: str, version: str, fitness: float,
) -> bool:
    """Cache a computed fitness on the variant row (for UI display)."""
    with store.connect() as conn:
        cur = conn.execute(
            "UPDATE skill_variants SET fitness_score=? "
            "WHERE skill_name=? AND version=?",
            (float(fitness), skill_name, version),
        )
        return cur.rowcount > 0


# ─────────────────────────── invocation routing ───────────────────────────


def select_variant_for_invoke(
    store: Store,
    *,
    skill_name: str,
    rng: random.Random | None = None,
) -> VariantSelection:
    """Decide which version to use for one upcoming invocation.

    Routing rules (precedence):
      1. If a canary exists and the random dice rolls under canary_traffic_pct,
         use the canary (A/B traffic split)
      2. If a live exists, use it (overrides on-disk seed)
      3. Otherwise fall back to the on-disk SKILL.md (use_disk_seed=True)

    Shadow variants are NEVER selected for live traffic — they only get
    synthetic eval signals from meta_evolve_skill.
    """
    rng = rng or random
    canaries = get_canary_variants(store, skill_name)
    live = get_live_variant(store, skill_name)

    # 1. Canary traffic split (single canary supported for now; if multiple,
    #    pick the highest-traffic-pct one and use its split)
    if canaries:
        canary = canaries[0]  # most-recently-promoted
        if rng.random() < canary.canary_traffic_pct:
            return VariantSelection(
                use_disk_seed=False,
                selected_variant=canary,
                selection_reason=(
                    f"canary {canary.version} ({canary.canary_traffic_pct:.0%} traffic)"
                ),
            )

    # 2. Live overrides seed
    if live:
        return VariantSelection(
            use_disk_seed=False,
            selected_variant=live,
            selection_reason=f"live {live.version}",
        )

    # 3. Fall through to disk
    return VariantSelection(
        use_disk_seed=True,
        selected_variant=None,
        selection_reason="seed (no DB variants)",
    )


# ─────────────────────────── version helpers ───────────────────────────


def bump_version(parent_version: str, suffix: str = "") -> str:
    """Generate a child version string deterministically.

    Examples:
        bump_version("0.1.0")            -> "0.1.1"
        bump_version("0.1.0", "shadow")  -> "0.1.1-shadow"
        bump_version("0.1.5-shadow")     -> "0.1.6"   (drops the suffix)
    """
    base = parent_version.split("-", 1)[0]  # strip suffix from parent
    parts = base.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected semver X.Y.Z, got {parent_version!r}")
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as e:
        raise ValueError(f"non-numeric semver part in {parent_version!r}") from e
    new_base = f"{major}.{minor}.{patch + 1}"
    return f"{new_base}-{suffix}" if suffix else new_base


# ─────────────────────────── private helpers ───────────────────────────

import json as _json


_VARIANT_COLS = (
    "id, skill_name, version, parent_version, body_md, spec_json, status, "
    "canary_traffic_pct, fitness_score, notes, created_at, promoted_at"
)


def _row_to_variant(row: tuple) -> Variant:
    try:
        spec = _json.loads(row[5] or "{}")
    except _json.JSONDecodeError:
        spec = {}
    return Variant(
        id=row[0], skill_name=row[1], version=row[2], parent_version=row[3],
        body_md=row[4], spec_overrides=spec, status=row[6],
        canary_traffic_pct=row[7], fitness_score=row[8], notes=row[9],
        created_at=row[10], promoted_at=row[11],
    )


def _query_one(store: Store, sql: str, params: tuple) -> Variant | None:
    # If the caller passed `SELECT *`, replace with explicit cols for stability
    sql = sql.replace("SELECT *", f"SELECT {_VARIANT_COLS}", 1)
    with store.connect() as conn:
        row = conn.execute(sql, params).fetchone()
    return _row_to_variant(row) if row else None


def _query_many(store: Store, sql: str, params: tuple) -> list[Variant]:
    sql = sql.replace("SELECT *", f"SELECT {_VARIANT_COLS}", 1)
    with store.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_variant(r) for r in rows]
