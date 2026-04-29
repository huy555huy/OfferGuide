"""W5' (Phase 0) — application_events log: append, list, derive, silence age."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import offerguide
from offerguide import application_events as ae


def _store_with_application(tmp_path: Path, *, applied_at: float | None = None) -> tuple:
    store = offerguide.Store(tmp_path / "ae.db")
    store.init_schema()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, title, raw_text, content_hash) VALUES (?,?,?,?)",
            ("manual", "t", "body", "h"),
        )
        cur = conn.execute(
            "INSERT INTO applications(job_id, status, applied_at) VALUES (?,?,?)",
            (1, "considered", applied_at),
        )
        app_id = int(cur.lastrowid or 0)
    return store, app_id


def test_record_returns_event_with_id_and_timestamp(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    ev = ae.record(store, application_id=app_id, kind="submitted", source="manual")
    assert ev.id > 0
    assert ev.application_id == app_id
    assert ev.kind == "submitted"
    assert ev.source == "manual"
    assert ev.occurred_at > 0
    assert ev.payload == {}


def test_record_persists_payload(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    ev = ae.record(
        store,
        application_id=app_id,
        kind="interview",
        source="calendar",
        payload={"round": "二面", "interviewer": "tech-lead"},
    )
    refreshed = ae.list_events(store, app_id)
    assert refreshed[0].payload == {"round": "二面", "interviewer": "tech-lead"}
    assert refreshed[0].id == ev.id


def test_list_events_returns_lifecycle_order(tmp_path: Path) -> None:
    """Oldest first — matches how a recruiter / dashboard reads a timeline."""
    store, app_id = _store_with_application(tmp_path)
    # Inject events at known timestamps so the order is deterministic
    ae.record(store, application_id=app_id, kind="submitted", occurred_at=2460000.0)
    ae.record(store, application_id=app_id, kind="viewed", occurred_at=2460001.0)
    ae.record(store, application_id=app_id, kind="replied", occurred_at=2460002.0)

    kinds = [e.kind for e in ae.list_events(store, app_id)]
    assert kinds == ["submitted", "viewed", "replied"]


def test_latest_returns_most_recent(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    ae.record(store, application_id=app_id, kind="submitted", occurred_at=2460000.0)
    ae.record(store, application_id=app_id, kind="rejected", occurred_at=2460010.0)

    last = ae.latest(store, app_id)
    assert last is not None
    assert last.kind == "rejected"


def test_latest_returns_none_when_no_events(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    assert ae.latest(store, app_id) is None


def test_derive_status_uses_latest_event_kind(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    ae.record(store, application_id=app_id, kind="submitted", occurred_at=2460000.0)
    assert ae.derive_status(store, app_id) == "submitted"
    ae.record(store, application_id=app_id, kind="viewed", occurred_at=2460005.0)
    assert ae.derive_status(store, app_id) == "viewed"
    ae.record(store, application_id=app_id, kind="rejected", occurred_at=2460010.0)
    assert ae.derive_status(store, app_id) == "rejected"


def test_derive_status_no_events_returns_sentinel(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    assert ae.derive_status(store, app_id) == "no_events"


def test_silence_age_days_against_known_now(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    ae.record(store, application_id=app_id, kind="submitted", occurred_at=2460000.0)
    age = ae.silence_age_days(store, app_id, now=2460014.0)
    assert age == pytest.approx(14.0, abs=1e-6)


def test_silence_age_ignores_inferred_synthetic_events(tmp_path: Path) -> None:
    """A 'silent_check' synthetic event must NOT be counted as recent activity."""
    store, app_id = _store_with_application(tmp_path)
    ae.record(store, application_id=app_id, kind="submitted", occurred_at=2460000.0)
    # Cron job writes a silence check 14 days later
    ae.record(
        store,
        application_id=app_id,
        kind="silent_check",
        source="inferred",
        occurred_at=2460014.0,
    )
    # Asking again at day 20: silence age should still be 20 (since the original
    # submitted event), not 6 (since the synthetic silence_check)
    age = ae.silence_age_days(store, app_id, now=2460020.0)
    assert age == pytest.approx(20.0, abs=1e-6)


def test_silence_age_returns_none_when_no_events(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    assert ae.silence_age_days(store, app_id) is None


def test_record_rejects_unknown_kind(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    with pytest.raises(ValueError, match="unknown event kind"):
        ae.record(store, application_id=app_id, kind="rocket_launch")  # type: ignore[arg-type]


def test_record_rejects_unknown_source(tmp_path: Path) -> None:
    store, app_id = _store_with_application(tmp_path)
    with pytest.raises(ValueError, match="unknown event source"):
        ae.record(store, application_id=app_id, kind="submitted", source="raven")  # type: ignore[arg-type]


def test_foreign_key_blocks_unknown_application(tmp_path: Path) -> None:
    """SQLite FK enforcement must reject events referencing a non-existent application."""
    store, _ = _store_with_application(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        ae.record(store, application_id=99999, kind="submitted")
