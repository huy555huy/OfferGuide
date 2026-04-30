"""ICS calendar file parser → ``application_events.interview`` events.

Pure-Python (no ``icalendar`` dep) since RFC 5545 is simple enough to
parse the few fields we care about (DTSTART, SUMMARY, DESCRIPTION).

Use case: HR sends an interview invite with a `.ics` attachment. The
user uploads the file via /api/applications/{id}/events/ics; we
parse it, detect interview-shaped events, and record them with
``source='calendar'`` and ``payload`` containing the scheduled time
and any free-text summary the calendar gave us.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class IcsEvent:
    """One VEVENT extracted from an ICS file."""

    summary: str
    description: str
    dtstart_utc: datetime | None
    """Start time as UTC datetime; None if unparseable."""

    is_interview: bool
    """Heuristic: summary or description mentions interview-shaped words."""


_INTERVIEW_HINTS = (
    "面试", "interview", "技术面", "一面", "二面", "三面", "终面", "HR 面",
    "tech round", "screening", "笔试", "OA",
)


def parse_ics(text: str) -> list[IcsEvent]:
    """Parse RFC 5545 calendar text → list of IcsEvent.

    Tolerant of encoding artifacts and content-line folding (RFC 5545
    §3.1: continuation lines start with whitespace and should be
    re-joined with the previous line).
    """
    if not text:
        return []
    text = _unfold(text)

    events: list[IcsEvent] = []
    current: dict[str, str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "BEGIN:VEVENT":
            current = {}
            continue
        if line == "END:VEVENT":
            if current is not None:
                events.append(_to_event(current))
                current = None
            continue
        if current is None:
            continue
        # Property:value (handle params after semicolons in the property)
        if ":" not in line:
            continue
        prop_full, value = line.split(":", 1)
        prop = prop_full.split(";", 1)[0].upper()
        # Multi-property handling: take last write
        current[prop] = value

    return events


def _unfold(text: str) -> str:
    """RFC 5545 line unfolding — continuation lines start with WS."""
    return re.sub(r"\r?\n[ \t]", "", text)


def _to_event(props: dict[str, str]) -> IcsEvent:
    summary = props.get("SUMMARY", "")
    description = props.get("DESCRIPTION", "")
    dtstart = _parse_dt(props.get("DTSTART", ""))
    is_interview = any(
        h.lower() in (summary + " " + description).lower()
        for h in _INTERVIEW_HINTS
    )
    return IcsEvent(
        summary=summary,
        description=description,
        dtstart_utc=dtstart,
        is_interview=is_interview,
    )


def _parse_dt(raw: str) -> datetime | None:
    """Parse DTSTART value in any of the common forms.

    Forms supported:
    - 20260605T140000Z       (UTC)
    - 20260605T140000        (local, treated as UTC)
    - 20260605                (date-only — assumes 09:00 UTC)
    """
    raw = raw.strip()
    if not raw:
        return None
    # YYYYMMDDTHHMMSSZ
    m = re.fullmatch(r"(\d{8})T(\d{6})Z?", raw)
    if m:
        date_part, time_part = m.groups()
        try:
            return datetime.strptime(
                date_part + time_part, "%Y%m%d%H%M%S"
            ).replace(tzinfo=UTC)
        except ValueError:
            return None
    # Date-only YYYYMMDD
    m = re.fullmatch(r"(\d{8})", raw)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").replace(
                hour=9, tzinfo=UTC
            )
        except ValueError:
            return None
    return None


def select_first_interview(events: list[IcsEvent]) -> IcsEvent | None:
    """Pick the most likely interview-shaped event from a calendar."""
    interviews = [e for e in events if e.is_interview]
    if not interviews:
        return None
    # Prefer events with the earliest dtstart; events without dtstart sink
    return sorted(
        interviews,
        key=lambda e: (e.dtstart_utc is None, e.dtstart_utc or datetime.max),
    )[0]


def datetime_to_julianday(dt: datetime) -> float:
    """Convert UTC datetime to SQLite julianday."""
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = (
        dt.day
        + (153 * m + 2) // 5
        + 365 * y
        + y // 4
        - y // 100
        + y // 400
        - 32045
    )
    frac = (
        (dt.hour - 12) / 24
        + dt.minute / 1440
        + dt.second / 86400
    )
    return jdn + frac
