"""Pipeline / Kanban view — applications grouped by lifecycle stage.

Where ``application_events.derive_status`` answers "what's the latest
state for one application", this module answers "show me everything
grouped into 5 stages so I can see the whole funnel at a glance".

The 5 stages map onto :data:`KANBAN_STAGES`:

- **scanned** — JD ingested but no submission event yet
- **applied** — submitted, no HR response yet
- **contacted** — HR viewed/replied/assessment
- **interview** — any interview round (1st/2nd/final)
- **terminal** — offer / rejected / withdrawn

Each card carries the data the kanban needs (company / title / latest
event / silence days), enough that no per-card fetch is needed when
the page renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory import Store

KANBAN_STAGES: list[tuple[str, str]] = [
    ("scanned",   "已扫描"),
    ("applied",   "已投递"),
    ("contacted", "HR 已联系"),
    ("interview", "面试中"),
    ("terminal",  "已结束"),
]
"""Stage key → Chinese label. Order is the visual left→right column order."""


@dataclass(frozen=True)
class KanbanCard:
    """One application's compact view, scaled for a kanban card."""
    application_id: int | None
    """None means this is a JD-only card (scanned but no application
    record yet). Click goes to /quick-eval pre-filled, not /applications."""

    job_id: int
    company: str
    title: str
    location: str
    source: str
    status: str
    """Latest event kind (or 'no_events' for scanned-only). The label
    surfaced on the card."""

    latest_event_kind: str | None
    silence_days: float | None
    """Days since latest non-inferred event. None if no events yet."""

    has_score: bool
    """Whether score_match has been run on this JD. Drives the 'score?'
    badge on scanned/applied cards."""


@dataclass
class PipelineView:
    """Full kanban payload — one bucket per stage."""
    columns: dict[str, list[KanbanCard]] = field(default_factory=dict)
    """Stage key → list of cards. Always has all 5 keys (empty lists OK)."""

    counts: dict[str, int] = field(default_factory=dict)
    """Stage key → card count, for the column headers."""

    total_active: int = 0
    """Cards in any non-terminal stage. Surfaced as the page-level stat."""


def build(store: Store) -> PipelineView:
    """Build the kanban view from the DB.

    Two-source aggregation:
    1. Every applications row → bucketed by derived status
    2. Every jobs row not yet in applications → 'scanned'

    'scanned' is intentionally noisy — it captures JDs the user dropped
    in but hasn't decided on. Once they hit "submit application", the
    row moves to 'applied' and disappears from 'scanned'.
    """
    columns: dict[str, list[KanbanCard]] = {k: [] for k, _ in KANBAN_STAGES}

    with store.connect() as conn:
        # Applications + their latest event (joined in SQL, not Python)
        app_rows = conn.execute(
            """
            WITH latest AS (
                SELECT application_id, MAX(occurred_at) AS last_at
                FROM application_events WHERE source != 'inferred'
                GROUP BY application_id
            ),
            latest_kind AS (
                SELECT ae.application_id, ae.kind
                FROM application_events ae
                JOIN latest ON latest.application_id = ae.application_id
                  AND latest.last_at = ae.occurred_at
                WHERE ae.source != 'inferred'
                GROUP BY ae.application_id
            )
            SELECT a.id, a.job_id, a.status,
                   j.company, j.title, j.location, j.source,
                   lk.kind,
                   (julianday('now') - latest.last_at) AS silence_days
            FROM applications a
            JOIN jobs j ON j.id = a.job_id
            LEFT JOIN latest      ON latest.application_id      = a.id
            LEFT JOIN latest_kind lk ON lk.application_id       = a.id
            ORDER BY a.last_status_change DESC, a.id DESC
            """
        ).fetchall()

        # Jobs without any application — 'scanned' bucket
        scanned_rows = conn.execute(
            """
            SELECT j.id, j.company, j.title, j.location, j.source,
                   (julianday('now') - j.fetched_at) AS age_days
            FROM jobs j
            LEFT JOIN applications a ON a.job_id = j.id
            WHERE a.id IS NULL
            ORDER BY j.fetched_at DESC, j.id DESC
            """
        ).fetchall()

        # Score lookup — for the 'has_score' badge
        score_rows = conn.execute(
            """
            SELECT DISTINCT json_extract(input_json, '$.job_id') AS jid
            FROM skill_runs
            WHERE skill_name = 'score_match'
              AND json_extract(input_json, '$.job_id') IS NOT NULL
            """
        ).fetchall()
    scored_job_ids: set[int] = {
        int(r[0]) for r in score_rows if r[0] is not None
    }

    # Place each application
    for r in app_rows:
        app_id = int(r[0])
        job_id = int(r[1])
        status = r[2] or "applied"
        company = r[3] or "(无公司名)"
        title = r[4] or "(无标题)"
        location = r[5] or ""
        source = r[6] or "?"
        latest_kind = r[7]
        silence_days = float(r[8]) if r[8] is not None else None

        stage = _stage_for_status(status)
        columns[stage].append(
            KanbanCard(
                application_id=app_id,
                job_id=job_id,
                company=company,
                title=title,
                location=location,
                source=source,
                status=status,
                latest_event_kind=latest_kind,
                silence_days=silence_days,
                has_score=job_id in scored_job_ids,
            )
        )

    # Place each scanned-only job
    for r in scanned_rows:
        job_id = int(r[0])
        columns["scanned"].append(
            KanbanCard(
                application_id=None,
                job_id=job_id,
                company=r[1] or "(无公司名)",
                title=r[2] or "(无标题)",
                location=r[3] or "",
                source=r[4] or "?",
                status="scanned",
                latest_event_kind=None,
                silence_days=None,
                has_score=job_id in scored_job_ids,
            )
        )

    counts = {k: len(v) for k, v in columns.items()}
    total_active = sum(
        n for k, n in counts.items() if k != "terminal"
    )
    return PipelineView(
        columns=columns,
        counts=counts,
        total_active=total_active,
    )


_STATUS_TO_STAGE: dict[str, str] = {
    "applied":         "applied",
    "considered":      "scanned",
    "viewed":          "contacted",
    "hr_replied":      "contacted",
    "screening":       "contacted",
    "written_test":    "contacted",
    "1st_interview":   "interview",
    "2nd_interview":   "interview",
    "final_interview": "interview",
    "offer":           "terminal",
    "rejected":        "terminal",
    "withdrawn":       "terminal",
}


def _stage_for_status(status: str) -> str:
    """Map a denormalized status string to one of the 5 kanban stages.

    Falls back to 'applied' for anything unknown — better to over-bucket
    than to silently drop applications from the view.
    """
    return _STATUS_TO_STAGE.get(status, "applied")


def stage_label(stage_key: str) -> str:
    """Human-readable label for a stage key."""
    for k, label in KANBAN_STAGES:
        if k == stage_key:
            return label
    return stage_key


def render_age(days: float | None) -> str:
    """Format silence-days for a card pill. Returns '—' for None."""
    if days is None:
        return "—"
    if days < 1:
        return "今天"
    if days < 7:
        return f"{int(days)}d"
    if days < 30:
        return f"{int(days // 7)}周"
    return f"{int(days // 30)}月"


def stage_color(stage_key: str) -> str:
    """CSS color class hint for a stage column."""
    return {
        "scanned":   "muted",
        "applied":   "primary",
        "contacted": "warning",
        "interview": "primary",
        "terminal":  "muted",
    }.get(stage_key, "muted")


def transition_options(stage_key: str) -> list[tuple[str, str]]:
    """Suggested next-event kinds the user can log from this stage.

    Returns ``[(kind, label), ...]`` for the inline action menu on each
    card. Empty list means terminal — no further events expected.
    """
    return _TRANSITIONS.get(stage_key, [])


_TRANSITIONS: dict[str, list[tuple[str, str]]] = {
    "scanned":   [("submitted", "标记已投递")],
    "applied":   [
        ("viewed", "HR 看了"),
        ("replied", "HR 回复"),
        ("rejected", "已拒"),
        ("withdrawn", "撤回"),
    ],
    "contacted": [
        ("assessment", "笔试"),
        ("interview", "面试"),
        ("rejected", "已拒"),
    ],
    "interview": [
        ("interview", "下一轮面试"),
        ("offer", "Offer"),
        ("rejected", "已拒"),
    ],
    "terminal":  [],
}

_ = Any  # keep typing import for downstream extension
