"""Parse RFC 5545 interview invitations into application events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

from icalendar import Calendar


@dataclass(frozen=True)
class IcsEvent:
    """One VEVENT with its calendar start semantics preserved."""

    summary: str
    description: str
    uid: str
    """RFC 5545 UID used to correlate invitation revisions."""

    sequence: int
    """RFC 5545 revision number; omitted SEQUENCE has the RFC default of zero."""

    status: str | None
    """VEVENT STATUS, normalized to uppercase when present."""

    method: str | None
    """VCALENDAR METHOD, normalized to uppercase when present."""

    dtstart_utc: datetime | None
    """Timed start converted to UTC, only when its timezone is known."""

    dtstart_local: datetime | None
    """Original wall-clock datetime, including its RFC 5545 timezone when present."""

    dtstart_date: date | None
    """Original VALUE=DATE start for an all-day event."""

    dtstart_tzid: str | None
    """The explicit DTSTART TZID parameter, if the invitation supplied one."""

    is_interview: bool
    """Whether the decoded summary or description looks interview-related."""

    @property
    def calendar_action(self) -> Literal["scheduled", "cancelled", "unsupported"]:
        """The lifecycle action represented by this calendar message."""
        if self.method in {"CANCEL", "CANCELED", "CANCELLED"} or self.status in {
            "CANCELED",
            "CANCELLED",
        }:
            return "cancelled"
        if self.method is None or self.method in {"REQUEST", "PUBLISH", "ADD"}:
            return "scheduled"
        return "unsupported"

    @property
    def round(self) -> str | None:
        """Explicit interview round found in organizer-authored event text."""
        return infer_interview_round(self.summary, self.description)


_INTERVIEW_HINTS = (
    "面试",
    "interview",
    "技术面",
    "一面",
    "二面",
    "三面",
    "终面",
    "HR面",
    "HR 面",
    "tech round",
    "screening",
    "笔试",
)

_OA_HINT = re.compile(r"(?<![A-Za-z0-9])oa(?![A-Za-z0-9])", re.IGNORECASE)

_ROUND_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "终面",
        (
            re.compile(r"终面"),
            re.compile(r"(?<![A-Za-z0-9])final\s+(?:round|interview)(?![A-Za-z0-9])", re.I),
        ),
    ),
    (
        "HR面",
        (
            re.compile(r"(?<![A-Za-z0-9])HR\s*面(?![A-Za-z0-9])", re.I),
            re.compile(
                r"(?<![A-Za-z0-9])HR\s+(?:round|interview)(?![A-Za-z0-9])",
                re.I,
            ),
        ),
    ),
    (
        "三面",
        (
            re.compile(r"三面|第三轮|第\s*3\s*轮"),
            re.compile(r"(?<![A-Za-z0-9])3rd\s+(?:round|interview)(?![A-Za-z0-9])", re.I),
            re.compile(r"(?<![A-Za-z0-9])(?:round|interview)\s*3(?![A-Za-z0-9])", re.I),
        ),
    ),
    (
        "二面",
        (
            re.compile(r"二面|第二轮|第\s*2\s*轮"),
            re.compile(r"(?<![A-Za-z0-9])2nd\s+(?:round|interview)(?![A-Za-z0-9])", re.I),
            re.compile(r"(?<![A-Za-z0-9])(?:round|interview)\s*2(?![A-Za-z0-9])", re.I),
        ),
    ),
    (
        "一面",
        (
            re.compile(r"一面|第一轮|第\s*1\s*轮"),
            re.compile(r"(?<![A-Za-z0-9])1st\s+(?:round|interview)(?![A-Za-z0-9])", re.I),
            re.compile(r"(?<![A-Za-z0-9])(?:round|interview)\s*1(?![A-Za-z0-9])", re.I),
        ),
    ),
)


def parse_ics(text: str) -> list[IcsEvent]:
    """Parse one or more RFC 5545 calendars without hand-parsing content lines."""
    if not text.strip():
        return []
    try:
        calendars = Calendar.from_ical(text, multiple=True)
    except (TypeError, ValueError):
        return []

    events: list[IcsEvent] = []
    for calendar in calendars:
        calendar_method = _optional_property_text(calendar.get("METHOD"))
        for component in calendar.walk("VEVENT"):
            summary = _property_text(component.get("SUMMARY"))
            description = _property_text(component.get("DESCRIPTION"))
            method = calendar_method or _optional_property_text(component.get("METHOD"))
            status = _optional_property_text(component.get("STATUS"))
            dtstart_utc, dtstart_local, dtstart_date, tzid = _decode_dtstart(
                component
            )
            events.append(
                IcsEvent(
                    summary=summary,
                    description=description,
                    uid=_property_text(component.get("UID")).strip(),
                    sequence=_sequence(component),
                    status=status.upper() if status is not None else None,
                    method=method.upper() if method is not None else None,
                    dtstart_utc=dtstart_utc,
                    dtstart_local=dtstart_local,
                    dtstart_date=dtstart_date,
                    dtstart_tzid=tzid,
                    is_interview=_looks_like_interview(summary, description),
                )
            )
    return events


def _property_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = value[-1] if value else ""
    return str(value)


def _optional_property_text(value: Any) -> str | None:
    text = _property_text(value).strip()
    return text or None


def _property_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("VEVENT SEQUENCE must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError("VEVENT SEQUENCE must be a non-negative integer")
    return parsed


def _sequence(component: Any) -> int:
    for property_name, _message in getattr(component, "errors", ()):
        if str(property_name).upper() == "SEQUENCE":
            raise ValueError("VEVENT SEQUENCE must be a non-negative integer")
    return _property_int(component.get("SEQUENCE"), default=0)


def _looks_like_interview(summary: str, description: str) -> bool:
    searchable = f"{summary} {description}"
    lowered = searchable.lower()
    return any(hint.lower() in lowered for hint in _INTERVIEW_HINTS) or bool(
        _OA_HINT.search(searchable)
    )


def infer_interview_round(summary: str, description: str = "") -> str | None:
    """Return a round only when the invitation text says it explicitly."""
    searchable = f"{summary} {description}"
    for label, patterns in _ROUND_PATTERNS:
        if any(pattern.search(searchable) for pattern in patterns):
            return label
    return None


def _decode_dtstart(
    component: Any,
) -> tuple[datetime | None, datetime | None, date | None, str | None]:
    prop = component.get("DTSTART")
    if prop is None:
        return None, None, None, None
    raw_tzid = prop.params.get("TZID")
    tzid = str(raw_tzid) if raw_tzid is not None else None
    try:
        decoded = component.decoded("DTSTART")
    except (TypeError, ValueError):
        return None, None, None, tzid

    if isinstance(decoded, datetime):
        local = decoded
        # Floating datetimes and unknown TZIDs decode without tzinfo. Their
        # wall-clock value is useful, but no UTC instant can be inferred.
        utc = decoded.astimezone(UTC) if decoded.tzinfo is not None else None
        return utc, local, None, tzid
    if isinstance(decoded, date):
        # VALUE=DATE has no time or timezone, so it is not an instant.
        return None, None, decoded, tzid
    return None, None, None, tzid


def select_first_interview(events: list[IcsEvent]) -> IcsEvent | None:
    """Pick the first interview-shaped VEVENT in calendar order.

    Floating, date-only, and timezone-aware starts are not mutually ordered:
    comparing them would require inventing a timezone. Calendar order is the
    only lossless deterministic choice.
    """
    return next(
        (
            event
            for event in events
            if event.is_interview and event.calendar_action != "unsupported"
        ),
        None,
    )


def datetime_to_julianday(dt: datetime) -> float:
    """Convert a datetime to SQLite Julian day after normalizing it to UTC."""
    if dt.tzinfo is None:
        raise ValueError("a floating datetime has no UTC Julian day")
    normalized = dt.astimezone(UTC)
    a = (14 - normalized.month) // 12
    y = normalized.year + 4800 - a
    m = normalized.month + 12 * a - 3
    jdn = (
        normalized.day
        + (153 * m + 2) // 5
        + 365 * y
        + y // 4
        - y // 100
        + y // 400
        - 32045
    )
    frac = (
        (normalized.hour - 12) / 24
        + normalized.minute / 1440
        + normalized.second / 86400
        + normalized.microsecond / 86_400_000_000
    )
    return jdn + frac


__all__ = [
    "IcsEvent",
    "datetime_to_julianday",
    "infer_interview_round",
    "parse_ics",
    "select_first_interview",
]
