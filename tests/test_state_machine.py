from __future__ import annotations

from pathlib import Path

import offerguide
from offerguide.state_machine import (
    EVENT_STATUS_MAP,
    next_status,
    status_for_event,
    sync_status,
)


def _application(tmp_path: Path, *, status: str = "considered") -> tuple[offerguide.Store, int]:
    store = offerguide.Store(tmp_path / "state-machine.db")
    store.init_schema()
    with store.connect() as conn:
        job_id = int(
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) "
                "VALUES ('manual', 'JD', 'state-machine-job') RETURNING id"
            ).fetchone()[0]
        )
        application_id = int(
            conn.execute(
                "INSERT INTO applications(job_id, status) VALUES (?, ?) RETURNING id",
                (job_id, status),
            ).fetchone()[0]
        )
    return store, application_id


def test_status_for_event_maps_real_lifecycle_events() -> None:
    for kind, expected in EVENT_STATUS_MAP.items():
        assert status_for_event(kind) == expected
    assert status_for_event("interview", {"round": "二面"}) == "2nd_interview"
    assert status_for_event("interview", {"round": "三面"}) == "final_interview"
    assert status_for_event("interview", {"round": "终面"}) == "final_interview"
    assert status_for_event("interview", {"round": "HR面"}) == "final_interview"
    assert status_for_event("interview", {"round": "HR 面"}) == "final_interview"
    assert status_for_event("interview_cancelled", {"round": "一面"}) is None
    assert status_for_event("unknown") is None


def test_sync_status_updates_denormalized_application_state(tmp_path: Path) -> None:
    store, application_id = _application(tmp_path)

    assert sync_status(store, application_id, "submitted") == "applied"

    with store.connect() as conn:
        status = conn.execute(
            "SELECT status FROM applications WHERE id = ?", (application_id,)
        ).fetchone()[0]
    assert status == "applied"


def test_lifecycle_status_never_regresses_confirmed_progress() -> None:
    assert next_status("2nd_interview", "interview", {}) == "2nd_interview"
    assert (
        next_status("final_interview", "interview", {"round": "一面"})
        == "final_interview"
    )
    assert next_status("final_interview", "viewed") == "final_interview"
    assert next_status("rejected", "interview", {"round": "三面"}) == "rejected"
