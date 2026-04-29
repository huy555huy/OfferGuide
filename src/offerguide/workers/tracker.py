"""Tracker worker — ambient silence detection + notification.

Runs periodically (cron / APScheduler / manual) to:

1. Scan all active (non-terminal) applications
2. Compute silence age (days since last non-synthetic event)
3. Fire alerts for NEW threshold crossings (7 d → 14 d → 30 d)
4. Record inferred ``silent_check`` events (idempotency + analytics)
5. Enqueue inbox items + push notifications

**Idempotency**: each threshold fires at most once per application.
The sweep checks existing ``silent_check`` events' ``threshold_days``
payload field to decide which thresholds have already been alerted.

Usage (one-shot)::

    python -m offerguide.workers tracker run --once
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from .. import application_events as ae
from .. import inbox
from ..memory import Store
from ..ui.notify._base import Notifier, NotifyLevel

log = logging.getLogger(__name__)

# ── thresholds ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SilenceThreshold:
    """A silence duration that triggers an alert."""

    days: int
    level: NotifyLevel
    template: str  # {company}, {title}, {days} are available


DEFAULT_THRESHOLDS: tuple[SilenceThreshold, ...] = (
    SilenceThreshold(7, "info", "{company} · {title} — 已投递 {days:.0f} 天无回复"),
    SilenceThreshold(14, "warn", "{company} · {title} — 沉默 {days:.0f} 天，建议跟进或放弃"),
    SilenceThreshold(30, "high", "{company} · {title} — 沉默 {days:.0f} 天，大概率石沉大海"),
)

TERMINAL_STATUSES: frozenset[str] = frozenset({"rejected", "offer", "withdrawn"})

# ── data types ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class SilenceFinding:
    """An application that just crossed a *new* silence threshold."""

    application_id: int
    job_id: int
    title: str
    company: str | None
    silence_days: float
    threshold: SilenceThreshold
    last_real_event_kind: str


# ── sweep (pure query, no side-effects) ─────────────────────────────


def sweep_silences(
    store: Store,
    *,
    thresholds: tuple[SilenceThreshold, ...] = DEFAULT_THRESHOLDS,
    now: float | None = None,
) -> list[SilenceFinding]:
    """Find applications that have crossed a new silence threshold.

    Returns findings sorted by ``silence_days`` descending (most urgent
    first).  Idempotent: already-alerted thresholds are not re-raised.
    """
    sorted_thresholds = sorted(thresholds, key=lambda t: t.days)
    apps = _get_active_applications(store)

    findings: list[SilenceFinding] = []
    for app_id, job_id, title, company, _status in apps:
        age = ae.silence_age_days(store, app_id, now=now)
        if age is None:
            continue  # no events yet

        # Find highest applicable threshold
        applicable: SilenceThreshold | None = None
        for t in reversed(sorted_thresholds):
            if age >= t.days:
                applicable = t
                break
        if applicable is None:
            continue  # below the lowest threshold

        # Idempotency: skip if already alerted at this level (or higher)
        max_alerted = _max_alerted_threshold(store, app_id)
        if applicable.days <= max_alerted:
            continue

        last_kind = _last_real_event_kind(store, app_id)
        findings.append(
            SilenceFinding(
                application_id=app_id,
                job_id=job_id,
                title=title or "(untitled)",
                company=company,
                silence_days=age,
                threshold=applicable,
                last_real_event_kind=last_kind or "unknown",
            )
        )

    findings.sort(key=lambda f: f.silence_days, reverse=True)
    return findings


# ── tracker_run (side-effecty: writes events, inbox, notifies) ──────


def tracker_run(
    store: Store,
    *,
    notifier: Notifier,
    thresholds: tuple[SilenceThreshold, ...] = DEFAULT_THRESHOLDS,
    now: float | None = None,
) -> dict[str, int]:
    """One sweep pass: find silences → record events → inbox → notify.

    Returns counters for logging / testing.
    """
    findings = sweep_silences(store, thresholds=thresholds, now=now)
    counters: dict[str, int] = {
        "silences_found": len(findings),
        "events_recorded": 0,
        "inbox_enqueued": 0,
        "notify_ok": 0,
        "notify_failed": 0,
    }

    for finding in findings:
        # 1. Record inferred silent_check event
        ae.record(
            store,
            application_id=finding.application_id,
            kind="silent_check",
            source="inferred",
            occurred_at=now,
            payload={
                "threshold_days": finding.threshold.days,
                "silence_days": round(finding.silence_days, 1),
                "last_real_event": finding.last_real_event_kind,
            },
        )
        counters["events_recorded"] += 1

        # 2. Enqueue inbox item
        msg = finding.threshold.template.format(
            company=finding.company or "未知公司",
            title=finding.title,
            days=finding.silence_days,
        )
        inbox.enqueue(
            store,
            kind="ambient_alert",
            title=msg[:80],
            body=(
                f"上次状态: {finding.last_real_event_kind}，"
                f"已沉默 {finding.silence_days:.1f} 天"
            ),
            payload={
                "application_id": finding.application_id,
                "job_id": finding.job_id,
                "threshold_days": finding.threshold.days,
            },
        )
        counters["inbox_enqueued"] += 1

        # 3. Push notification
        notify_result = notifier.notify(
            title=f"OfferGuide: {finding.threshold.days}d 沉默提醒",
            body=msg,
            level=finding.threshold.level,
        )
        if notify_result.ok:
            counters["notify_ok"] += 1
        else:
            counters["notify_failed"] += 1
            log.warning(
                "notify failed for app %d: %s",
                finding.application_id,
                notify_result.error,
            )

    return counters


# ── internal helpers ────────────────────────────────────────────────


def _get_active_applications(store: Store) -> list[tuple[Any, ...]]:
    """(app_id, job_id, title, company, status) for non-terminal apps."""
    with store.connect() as conn:
        return conn.execute(
            "SELECT a.id, a.job_id, j.title, j.company, a.status "
            "FROM applications a JOIN jobs j ON j.id = a.job_id "
            "WHERE a.status NOT IN (?, ?, ?)",
            tuple(TERMINAL_STATUSES),
        ).fetchall()


def _max_alerted_threshold(store: Store, application_id: int) -> int:
    """Highest ``threshold_days`` value from existing silent_check events."""
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT payload_json FROM application_events "
            "WHERE application_id = ? AND kind = 'silent_check' AND source = 'inferred'",
            (application_id,),
        ).fetchall()
    max_t = 0
    for (payload_json,) in rows:
        payload = json.loads(payload_json) if payload_json else {}
        t = payload.get("threshold_days", 0)
        if isinstance(t, int | float) and t > max_t:
            max_t = int(t)
    return max_t


def _last_real_event_kind(store: Store, application_id: int) -> str | None:
    """Kind of the latest non-synthetic event."""
    with store.connect() as conn:
        row = conn.execute(
            "SELECT kind FROM application_events "
            "WHERE application_id = ? AND source != 'inferred' "
            "ORDER BY occurred_at DESC, id DESC LIMIT 1",
            (application_id,),
        ).fetchone()
    return row[0] if row else None
