"""W13.6 long-horizon goals + progress + self-observations.

A real agent has a north star, not just per-tick reactive tasks. This
module is the bridge between user intent ("我要 X 月前拿到 Y 个 offer")
and the agent's per-wake decisions ("with N days left + M apps in flight,
should I push enrich today or wait?").

Three concepts:

1. **Goals** — what the user is working toward (table: ``user_goals``)
2. **Progress** — current funnel state vs goal (computed from
   applications + application_events + interviews)
3. **Self-observations** — the agent's own notes about its behavior
   patterns (table: ``agent_self_observations``)

The agent reads all three as part of the snapshot and lets them shape
its decisions. Without a goal, every wake is reactive housekeeping;
with one, every wake aligns to "are we closer to or further from X?"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from .memory import Store

log = logging.getLogger(__name__)

GoalStatus = Literal["active", "paused", "achieved", "abandoned"]
PatternKind = Literal[
    "overreach",        # agent does too much, user pushes back
    "underreach",       # agent should have acted but didn't
    "tone",             # agent's voice / style is off
    "wrong_priority",   # agent picked wrong thing to focus on
    "repeated_mistake", # same error pattern N times
    "success_pattern",  # something agent does well that should continue
]


# ─────────────────────────── Goals ───────────────────────────


@dataclass(frozen=True)
class Goal:
    id: int
    title: str
    description: str | None
    target_date: date | None
    target_metric: str | None
    status: GoalStatus
    created_at: float
    updated_at: float
    achieved_at: float | None
    notes: str | None


def add_goal(
    store: Store,
    *,
    title: str,
    description: str | None = None,
    target_date: date | str | None = None,
    target_metric: str | None = None,
    notes: str | None = None,
) -> Goal:
    """Create a new active goal. Multiple active goals are allowed (agent
    will reason about all of them)."""
    target_date_str: str | None = None
    if isinstance(target_date, date):
        target_date_str = target_date.isoformat()
    elif isinstance(target_date, str):
        target_date_str = target_date

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO user_goals(title, description, target_date, "
            "  target_metric, status, notes) "
            "VALUES (?, ?, ?, ?, 'active', ?)",
            (title, description, target_date_str, target_metric, notes),
        )
        new_id = int(cur.lastrowid or 0)
    fetched = get_goal(store, new_id)
    assert fetched is not None
    return fetched


def get_goal(store: Store, goal_id: int) -> Goal | None:
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, title, description, target_date, target_metric, "
            "       status, created_at, updated_at, achieved_at, notes "
            "FROM user_goals WHERE id = ?",
            (goal_id,),
        ).fetchone()
    return _row_to_goal(row) if row else None


def list_active_goals(store: Store) -> list[Goal]:
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, title, description, target_date, target_metric, "
            "       status, created_at, updated_at, achieved_at, notes "
            "FROM user_goals WHERE status = 'active' "
            "ORDER BY target_date ASC, created_at ASC"
        ).fetchall()
    return [_row_to_goal(r) for r in rows]


def update_goal_status(
    store: Store, goal_id: int, *, status: GoalStatus, notes: str | None = None,
) -> bool:
    achieved_at_set = ", achieved_at = julianday('now')" if status == "achieved" else ""
    with store.connect() as conn:
        cur = conn.execute(
            f"UPDATE user_goals SET status = ?, updated_at = julianday('now'), "
            f"  notes = COALESCE(?, notes){achieved_at_set} "
            f"WHERE id = ?",
            (status, notes, goal_id),
        )
        return cur.rowcount > 0


def _row_to_goal(row: tuple) -> Goal:
    td: date | None = None
    if row[3]:
        try:
            td = date.fromisoformat(row[3])
        except ValueError:
            td = None
    return Goal(
        id=row[0], title=row[1], description=row[2],
        target_date=td, target_metric=row[4], status=row[5],
        created_at=row[6], updated_at=row[7], achieved_at=row[8],
        notes=row[9],
    )


# ─────────────────────────── Progress computation ─────────────────────


@dataclass(frozen=True)
class GoalProgress:
    """Snapshot of progress toward a goal — agent reads this on every wake."""
    goal: Goal
    days_left: int | None  # None for open-ended goals
    days_elapsed: int

    # Funnel counts
    apps_total: int
    apps_active: int
    apps_silent_7d: int
    apps_silent_14d: int
    interviews_scheduled: int
    interviews_done: int
    offers: int
    rejects: int

    def render_for_prompt(self) -> str:
        """Render as a compact prompt block. Agent sees this and reasons against it."""
        lines = [f"### Goal #{self.goal.id}: {self.goal.title}"]
        if self.goal.target_date:
            if self.days_left is not None:
                if self.days_left < 0:
                    lines.append(f"  ⚠ 已过期 {-self.days_left} 天 (target: {self.goal.target_date})")
                else:
                    lines.append(f"  剩余 {self.days_left} 天 (target: {self.goal.target_date})")
        if self.goal.target_metric:
            lines.append(f"  目标指标: {self.goal.target_metric}")
        if self.goal.description:
            lines.append(f"  说明: {self.goal.description[:200]}")

        lines.append(
            f"  Funnel: {self.apps_total} 投 → {self.apps_active} 进行 → "
            f"{self.interviews_scheduled} 面试排期 → {self.offers} offer "
            f"({self.rejects} 拒)"
        )
        if self.apps_silent_14d > 0:
            lines.append(f"  ⚠ {self.apps_silent_14d} 个申请超 14 天没回 (大概率挂了)")
        elif self.apps_silent_7d > 0:
            lines.append(f"  · {self.apps_silent_7d} 个申请 7-14 天没回")
        return "\n".join(lines)

    @property
    def is_on_track(self) -> bool:
        """Heuristic: enough apps in flight relative to time remaining?

        Rough rule: if user wants 1 offer and has 14 days left, need at least
        5 active apps (assuming ~20% reply rate). This is a starting heuristic
        the agent should question, not blindly trust.
        """
        if self.days_left is None:
            return True  # open-ended, always "on track"
        if self.offers > 0:
            return True
        if self.days_left > 0:
            ratio = self.apps_active / max(self.days_left, 1)
            return ratio >= 0.3  # need at least 1 app per 3 remaining days
        return False


def compute_progress(store: Store, goal: Goal) -> GoalProgress:
    """Compute funnel + time progress for one goal."""
    days_left: int | None = None
    if goal.target_date:
        days_left = (goal.target_date - date.today()).days
    days_elapsed = (date.today() - datetime.fromtimestamp(
        # julianday → unix epoch seconds (julianday 2440587.5 = 1970-01-01)
        max(0, (goal.created_at - 2440587.5) * 86400)
    ).date()).days

    # Count ALL current applications (not just those created after the goal).
    # The user might already have apps in flight when they set a goal — those
    # still count toward progress. Filtering by applied_at would create a
    # confusing "I have 5 apps but the goal page shows 0" experience.
    with store.connect() as conn:
        apps_total = conn.execute(
            "SELECT COUNT(*) FROM applications"
        ).fetchone()[0]
        apps_active = conn.execute(
            "SELECT COUNT(*) FROM applications "
            "WHERE status NOT IN ('offer','rejected','withdrawn')"
        ).fetchone()[0]
        apps_silent_7d = conn.execute(
            "SELECT COUNT(*) FROM applications "
            "WHERE status NOT IN ('offer','rejected','withdrawn') "
            "  AND last_status_change < julianday('now') - 7"
        ).fetchone()[0]
        apps_silent_14d = conn.execute(
            "SELECT COUNT(*) FROM applications "
            "WHERE status NOT IN ('offer','rejected','withdrawn') "
            "  AND last_status_change < julianday('now') - 14"
        ).fetchone()[0]
        interviews_scheduled = conn.execute(
            "SELECT COUNT(*) FROM interviews "
            "WHERE scheduled_at IS NOT NULL "
            "  AND scheduled_at > julianday('now')"
        ).fetchone()[0]
        interviews_done = conn.execute(
            "SELECT COUNT(*) FROM interviews "
            "WHERE scheduled_at IS NOT NULL "
            "  AND scheduled_at <= julianday('now')"
        ).fetchone()[0]
        offers = conn.execute(
            "SELECT COUNT(*) FROM applications WHERE status='offer'"
        ).fetchone()[0]
        rejects = conn.execute(
            "SELECT COUNT(*) FROM applications WHERE status='rejected'"
        ).fetchone()[0]

    return GoalProgress(
        goal=goal,
        days_left=days_left, days_elapsed=days_elapsed,
        apps_total=apps_total, apps_active=apps_active,
        apps_silent_7d=apps_silent_7d, apps_silent_14d=apps_silent_14d,
        interviews_scheduled=interviews_scheduled, interviews_done=interviews_done,
        offers=offers, rejects=rejects,
    )


# ─────────────────────────── Self-observations ───────────────────────


@dataclass(frozen=True)
class SelfObservation:
    id: int
    observation: str
    pattern_kind: PatternKind
    evidence: dict[str, Any]
    valid_until: float | None
    created_at: float
    superseded_by: int | None


def write_self_observation(
    store: Store,
    *,
    observation: str,
    pattern_kind: PatternKind,
    evidence: dict[str, Any] | None = None,
    valid_for_days: int | None = None,
) -> int:
    """Agent writes a note about its own behavior pattern. Future runs see it."""
    params: list = [observation, pattern_kind, json.dumps(evidence or {}, ensure_ascii=False)]
    # W14.7-fix: previous version used backslash-escaped quotes inside an
    # f-string (`{', julianday(\"now\") + ?'}`), which is a Python 3.12+
    # feature (PEP 701). pyproject targets py3.11, so this raised SyntaxError
    # at import time on the declared min version. Lift the conditional pieces
    # into named vars to keep the SQL string a plain (non-f) literal.
    extra_col = ", valid_until" if valid_for_days is not None else ""
    extra_val = ", julianday('now') + ?" if valid_for_days is not None else ""
    if valid_for_days is not None:
        params.append(int(valid_for_days))
    sql = (
        "INSERT INTO agent_self_observations("
        "  observation, pattern_kind, evidence_json" + extra_col + ") "
        "VALUES (?, ?, ?" + extra_val + ")"
    )
    with store.connect() as conn:
        cur = conn.execute(sql, params)
        return int(cur.lastrowid or 0)


def list_active_self_observations(
    store: Store, *, limit: int = 10,
) -> list[SelfObservation]:
    """Return current (un-superseded, not-expired) self-observations the
    agent has accumulated. Agent reads these as part of the snapshot to
    avoid repeating its own mistakes."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, observation, pattern_kind, evidence_json, "
            "       valid_until, created_at, superseded_by "
            "FROM agent_self_observations "
            "WHERE superseded_by IS NULL "
            "  AND (valid_until IS NULL OR valid_until > julianday('now')) "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        SelfObservation(
            id=r[0], observation=r[1], pattern_kind=r[2],
            evidence=json.loads(r[3]) if r[3] else {},
            valid_until=r[4], created_at=r[5], superseded_by=r[6],
        )
        for r in rows
    ]


def supersede_observation(store: Store, *, old_id: int, new_id: int) -> bool:
    """Mark old_id as superseded by new_id (chain of self-correction)."""
    with store.connect() as conn:
        cur = conn.execute(
            "UPDATE agent_self_observations SET superseded_by = ? WHERE id = ?",
            (new_id, old_id),
        )
        return cur.rowcount > 0
