"""Daily standup brief — what should the user do *today*?

This module is the answer to "I just opened OfferGuide, what's important?".
Pure SQL derivations off the existing event log + jobs table; no LLM call,
no scheduled state. Always reflects current DB state when the home page
loads.

Surfaces four pillars:

- ``silent_followups``   applications stuck silent past the threshold
- ``upcoming_interviews`` interview events scheduled in the next 7 days
- ``unscored_jobs``       ingested JDs the user hasn't run score_match on
- ``stale_briefs``        company_briefs older than 14 days that need a refresh

The home page surfaces these as both stat cards (counts) and a punch list
(top 5 of each, with one-click action links). When a pillar is empty, we
say so honestly — nothing is more demoralizing than a dashboard pretending
to have work for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .memory import Store

SILENT_FOLLOWUP_DAYS = 7.0
"""How long silence (no non-inferred event) must last before we surface
the application as "needs follow-up". One week matches typical Chinese
campus-recruiting reply latency — under that, silence is normal; past
that, a polite 2-line nudge to the HR contact often reactivates the
thread."""

UPCOMING_WINDOW_DAYS = 7.0
"""Look-ahead window for interview events. Anything past this is too far
out to be "today's worry"."""

STALE_BRIEF_DAYS = 14.0
"""Company briefs older than this are considered stale. The autonomous
brief_update job refreshes weekly, so 14d is "we've missed at least one
refresh cycle". Surface them so the user knows *why* a brief feels off."""


@dataclass(frozen=True)
class SilentApplication:
    """Application that has gone silent past the threshold."""
    application_id: int
    company: str
    title: str
    status: str
    silence_days: float


@dataclass(frozen=True)
class UpcomingInterview:
    """Interview event scheduled within the look-ahead window."""
    application_id: int
    company: str
    title: str
    days_until: float
    when_human: str
    summary: str


@dataclass(frozen=True)
class UnscoredJob:
    """JD that's been ingested but never run through score_match."""
    job_id: int
    company: str | None
    title: str
    source: str
    age_days: float


@dataclass(frozen=True)
class StaleBrief:
    """Company brief older than the staleness threshold."""
    company: str
    age_days: float
    confidence: float


@dataclass(frozen=True)
class ActionItem:
    """One concrete thing the user should do, with a link.

    ``priority``: 'high' | 'medium' | 'low' — drives the visual stripe.
    ``action_url``: relative URL the click takes them to (e.g. ``/applications#42``).
    """
    title: str
    detail: str
    priority: str
    action_url: str


@dataclass
class DailyBrief:
    """The complete day-one view of the user's job-hunt state."""
    silent_followups: list[SilentApplication] = field(default_factory=list)
    upcoming_interviews: list[UpcomingInterview] = field(default_factory=list)
    unscored_jobs: list[UnscoredJob] = field(default_factory=list)
    stale_briefs: list[StaleBrief] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)


def build(store: Store, *, max_per_pillar: int = 5) -> DailyBrief:
    """Compute a full daily brief from current DB state.

    ``max_per_pillar`` caps each list — the home page shows top 5 per
    pillar; a dedicated drilldown page can show all.
    """
    silent = _silent_followups(store, limit=max_per_pillar)
    upcoming = _upcoming_interviews(store, limit=max_per_pillar)
    unscored = _unscored_jobs(store, limit=max_per_pillar)
    stale = _stale_briefs(store, limit=max_per_pillar)

    actions = _derive_action_items(
        silent_followups=silent,
        upcoming_interviews=upcoming,
        unscored_jobs=unscored,
        stale_briefs=stale,
    )

    return DailyBrief(
        silent_followups=silent,
        upcoming_interviews=upcoming,
        unscored_jobs=unscored,
        stale_briefs=stale,
        action_items=actions,
    )


def _silent_followups(store: Store, *, limit: int) -> list[SilentApplication]:
    """Applications whose latest non-inferred event is past the threshold.

    Excludes terminal statuses (offer/rejected/withdrawn) — those don't
    need follow-up. Sorted by silence_days desc so the longest-silent
    surfaces first.
    """
    with store.connect() as conn:
        rows = conn.execute(
            """
            WITH latest AS (
                SELECT application_id, MAX(occurred_at) AS last_at
                FROM application_events
                WHERE source != 'inferred'
                GROUP BY application_id
            )
            SELECT a.id, j.company, j.title, a.status,
                   (julianday('now') - latest.last_at) AS silence_days
            FROM applications a
            JOIN jobs j ON j.id = a.job_id
            JOIN latest ON latest.application_id = a.id
            WHERE a.status NOT IN ('rejected', 'offer', 'withdrawn')
              AND silence_days >= ?
            ORDER BY silence_days DESC
            LIMIT ?
            """,
            (SILENT_FOLLOWUP_DAYS, limit),
        ).fetchall()
    return [
        SilentApplication(
            application_id=int(r[0]),
            company=r[1] or "(无公司名)",
            title=r[2] or "(无标题)",
            status=r[3] or "applied",
            silence_days=float(r[4]),
        )
        for r in rows
    ]


def _upcoming_interviews(store: Store, *, limit: int) -> list[UpcomingInterview]:
    """Interview-kind events scheduled in the next 7 days.

    Pulls from application_events directly so calendar-derived interview
    events (source='calendar') with future occurred_at are captured.
    """
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT ae.application_id, j.company, j.title,
                   (ae.occurred_at - julianday('now')) AS days_until,
                   ae.payload_json
            FROM application_events ae
            JOIN applications a ON a.id = ae.application_id
            JOIN jobs j ON j.id = a.job_id
            WHERE ae.kind = 'interview'
              AND ae.occurred_at > julianday('now')
              AND ae.occurred_at < julianday('now', ?)
            ORDER BY ae.occurred_at ASC
            LIMIT ?
            """,
            (f"+{int(UPCOMING_WINDOW_DAYS)} days", limit),
        ).fetchall()

    out: list[UpcomingInterview] = []
    import json as _json
    for r in rows:
        days_until = float(r[3])
        payload: dict[str, Any] = {}
        try:
            payload = _json.loads(r[4]) if r[4] else {}
        except _json.JSONDecodeError:
            payload = {}
        when_human = _humanize_future(days_until)
        out.append(
            UpcomingInterview(
                application_id=int(r[0]),
                company=r[1] or "(无公司名)",
                title=r[2] or "(无标题)",
                days_until=days_until,
                when_human=when_human,
                summary=str(payload.get("summary") or "")[:120],
            )
        )
    return out


def _unscored_jobs(store: Store, *, limit: int) -> list[UnscoredJob]:
    """Jobs ingested but never passed through score_match.

    A job is "unscored" if no skill_runs row references its raw_text via
    a score_match invocation. We use a payload-free heuristic: jobs not
    referenced by any score_match run linked back via the app's
    application table.

    Keeping it simple: just check if this job has any application logged.
    No-application = no evaluation. (Score-without-application is the
    early-evaluation flow; the user can mark "I evaluated but won't apply"
    via inbox dismiss, which we treat as having scored it.)
    """
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT j.id, j.company, j.title, j.source,
                   (julianday('now') - j.fetched_at) AS age_days
            FROM jobs j
            LEFT JOIN applications a ON a.job_id = j.id
            LEFT JOIN inbox_items i ON
                json_extract(i.payload_json, '$.score_run_id') IS NOT NULL
                AND i.kind = 'consider_jd'
            WHERE a.id IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM inbox_items i2
                  WHERE i2.kind = 'consider_jd'
                    AND json_extract(i2.payload_json, '$.job_id') = j.id
              )
            ORDER BY j.fetched_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        UnscoredJob(
            job_id=int(r[0]),
            company=r[1],
            title=r[2] or "(无标题)",
            source=r[3] or "?",
            age_days=float(r[4] or 0),
        )
        for r in rows
    ]


def _stale_briefs(store: Store, *, limit: int) -> list[StaleBrief]:
    """Company briefs older than the staleness threshold.

    ``confidence`` lives inside ``brief_json`` (Pydantic CompanyBrief
    schema), so we extract it via SQLite's json_extract rather than a
    Python loop.
    """
    import json as _json

    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT company, brief_json,
                   (julianday('now') - last_updated_at) AS age_days
            FROM company_briefs
            WHERE age_days >= ?
            ORDER BY age_days DESC
            LIMIT ?
            """,
            (STALE_BRIEF_DAYS, limit),
        ).fetchall()
    out: list[StaleBrief] = []
    for r in rows:
        try:
            brief = _json.loads(r[1]) if r[1] else {}
        except _json.JSONDecodeError:
            brief = {}
        out.append(
            StaleBrief(
                company=r[0],
                age_days=float(r[2]),
                confidence=float(brief.get("confidence", 0.5)),
            )
        )
    return out


def _derive_action_items(
    *,
    silent_followups: list[SilentApplication],
    upcoming_interviews: list[UpcomingInterview],
    unscored_jobs: list[UnscoredJob],
    stale_briefs: list[StaleBrief],
) -> list[ActionItem]:
    """Translate the four pillars into concrete one-line action items.

    Order matters: highest-priority items go first. A user who sees the
    list top-down and stops after 3 items should still hit the most
    urgent things.

    Priority rules:
    - **high**: interview within 48h, silent >14 days, or 5+ unscored jobs
    - **medium**: interview within 7d, silent 7-14 days, 1-4 unscored
    - **low**: stale brief, distant interview
    """
    items: list[ActionItem] = []

    # Tomorrow's interview > everything else
    for up in upcoming_interviews:
        priority = "high" if up.days_until < 2 else "medium"
        prep_hint = "先跑 prepare_interview + deep_project_prep" if up.days_until < 3 else "排进本周备战"
        items.append(
            ActionItem(
                title=f"📅 {up.company} 面试 · {up.when_human}",
                detail=f"{up.title} — {prep_hint}",
                priority=priority,
                action_url=f"/applications#app-{up.application_id}",
            )
        )

    # Long-silent applications need a poke
    for s in silent_followups:
        if s.silence_days >= 14:
            priority = "high"
            verb = "立刻跟进"
        elif s.silence_days >= 10:
            priority = "medium"
            verb = "考虑跟进"
        else:
            priority = "low"
            verb = "可以观望"
        items.append(
            ActionItem(
                title=f"⏳ {s.company} 沉默 {int(s.silence_days)} 天",
                detail=f"{s.title} ({s.status}) — {verb}",
                priority=priority,
                action_url=f"/applications#app-{s.application_id}",
            )
        )

    # Pile of unscored JDs is a productivity smell
    if unscored_jobs:
        n = len(unscored_jobs)
        priority = "high" if n >= 5 else "medium"
        items.append(
            ActionItem(
                title=f"🎯 {n} 个 JD 没跑评估",
                detail="去快速评估面板挑高优先级先跑",
                priority=priority,
                action_url="/quick-eval",
            )
        )

    # Stale briefs are background hygiene
    for sb in stale_briefs[:2]:  # only top 2 in action list
        items.append(
            ActionItem(
                title=f"🔄 {sb.company} brief 过期 {int(sb.age_days)} 天",
                detail="跑 brief_update job 或在 sweep 里手动刷",
                priority="low",
                action_url=f"/dashboard#brief-{sb.company}",
            )
        )

    # Sort by priority weight
    weight = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda it: weight.get(it.priority, 3))
    return items


def _humanize_future(days: float) -> str:
    """Render a positive (future) days value as Chinese phrase."""
    if days < 1 / 24:
        return "马上"
    if days < 1:
        hours = int(days * 24)
        return f"{hours} 小时后"
    if days < 2:
        return "明天"
    if days < 3:
        return "后天"
    return f"{int(days)} 天后"
