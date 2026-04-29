"""W7 — Tracker worker tests: state machine + silence sweep + tracker_run.

Tests cover:
- State machine: event→status mapping, interview round specialization
- Sweep: silence detection, idempotency, terminal skip, ordering
- Tracker run: end-to-end with stubbed notifier
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import offerguide
from offerguide import application_events as ae
from offerguide.state_machine import (
    EVENT_STATUS_MAP,
    TERMINAL_STATUSES,
    status_for_event,
    sync_status,
)
from offerguide.ui.notify._base import NotifyResult
from offerguide.workers.tracker import (
    SilenceThreshold,
    sweep_silences,
    tracker_run,
)

# ── fixtures ────────────────────────────────────────────────────────

T0 = 2_460_000.0  # arbitrary Julian day baseline


def _make_store(tmp_path: Path) -> offerguide.Store:
    store = offerguide.Store(tmp_path / "tracker.db")
    store.init_schema()
    return store


def _add_job(store: offerguide.Store, title: str = "AI Agent 实习", company: str = "字节") -> int:
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES (?,?,?,?,?)",
            ("manual", title, company, "jd body", f"hash_{title}_{company}"),
        )
        return int(cur.lastrowid or 0)


def _add_app(store: offerguide.Store, job_id: int, status: str = "applied") -> int:
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO applications(job_id, status) VALUES (?,?)",
            (job_id, status),
        )
        return int(cur.lastrowid or 0)


class _StubNotifier:
    """Records all calls; returns success by default."""

    name = "stub"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._fail = fail

    def notify(self, *, title: str, body: str, level: str = "info") -> NotifyResult:
        self.calls.append({"title": title, "body": body, "level": level})
        if self._fail:
            return NotifyResult(ok=False, channel="stub", error="simulated failure")
        return NotifyResult(ok=True, channel="stub")


# ═══════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════


class TestStatusForEvent:
    def test_submitted_maps_to_applied(self) -> None:
        assert status_for_event("submitted") == "applied"

    def test_all_basic_mappings_covered(self) -> None:
        for kind, expected in EVENT_STATUS_MAP.items():
            assert status_for_event(kind) == expected

    def test_interview_default_is_1st(self) -> None:
        assert status_for_event("interview") == "1st_interview"

    def test_interview_round_二面(self) -> None:
        assert status_for_event("interview", {"round": "二面"}) == "2nd_interview"

    def test_interview_round_终面(self) -> None:
        assert status_for_event("interview", {"round": "终面"}) == "final_interview"

    def test_interview_round_笔试(self) -> None:
        assert status_for_event("interview", {"round": "笔试"}) == "written_test"

    def test_silent_check_returns_none(self) -> None:
        assert status_for_event("silent_check") is None

    def test_unknown_kind_returns_none(self) -> None:
        assert status_for_event("rocket_launch") is None


class TestSyncStatus:
    def test_updates_applications_table(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id, status="considered")

        result = sync_status(store, app_id, "submitted")
        assert result == "applied"

        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()
        assert row[0] == "applied"

    def test_noop_for_silent_check(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id, status="applied")

        result = sync_status(store, app_id, "silent_check")
        assert result is None

        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()
        assert row[0] == "applied"  # unchanged


# ═══════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════


class TestSweepSilences:
    def test_finds_silence_at_7d(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id)
        ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)

        findings = sweep_silences(store, now=T0 + 8)
        assert len(findings) == 1
        assert findings[0].application_id == app_id
        assert findings[0].threshold.days == 7
        assert findings[0].silence_days == pytest.approx(8.0)

    def test_idempotent_does_not_re_alert(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id)
        ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)

        # First sweep at day 8 → fires 7d alert
        findings1 = sweep_silences(store, now=T0 + 8)
        assert len(findings1) == 1

        # Record the silent_check that tracker_run would write
        ae.record(
            store,
            application_id=app_id,
            kind="silent_check",
            source="inferred",
            occurred_at=T0 + 8,
            payload={"threshold_days": 7},
        )

        # Second sweep at day 10 → still 7d range, already alerted
        findings2 = sweep_silences(store, now=T0 + 10)
        assert len(findings2) == 0

    def test_alerts_at_higher_threshold(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id)
        ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)

        # Already alerted at 7d
        ae.record(
            store,
            application_id=app_id,
            kind="silent_check",
            source="inferred",
            occurred_at=T0 + 8,
            payload={"threshold_days": 7},
        )

        # Sweep at day 15 → 14d threshold is new
        findings = sweep_silences(store, now=T0 + 15)
        assert len(findings) == 1
        assert findings[0].threshold.days == 14

    def test_skips_terminal_statuses(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        for status in TERMINAL_STATUSES:
            job_id = _add_job(store, title=f"job_{status}", company=f"co_{status}")
            app_id = _add_app(store, job_id, status=status)
            ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)

        findings = sweep_silences(store, now=T0 + 30)
        assert len(findings) == 0

    def test_skips_apps_with_no_events(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        _add_app(store, job_id)  # no events recorded

        findings = sweep_silences(store, now=T0 + 30)
        assert len(findings) == 0

    def test_returns_most_urgent_first(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)

        j1 = _add_job(store, title="新投", company="A")
        a1 = _add_app(store, j1)
        ae.record(store, application_id=a1, kind="submitted", occurred_at=T0)

        j2 = _add_job(store, title="旧投", company="B")
        a2 = _add_app(store, j2)
        ae.record(store, application_id=a2, kind="submitted", occurred_at=T0 - 10)

        findings = sweep_silences(store, now=T0 + 8)
        assert len(findings) == 2
        # a2 (18d silence) should come before a1 (8d silence)
        assert findings[0].application_id == a2
        assert findings[1].application_id == a1

    def test_custom_thresholds(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id)
        ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)

        custom = (SilenceThreshold(3, "info", "3d: {title}"),)
        findings = sweep_silences(store, thresholds=custom, now=T0 + 4)
        assert len(findings) == 1
        assert findings[0].threshold.days == 3

    def test_real_event_resets_silence(self, tmp_path: Path) -> None:
        """An HR reply resets the silence clock; no alert if within threshold."""
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id)
        ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)
        ae.record(store, application_id=app_id, kind="replied", occurred_at=T0 + 5)

        # 3 days after reply — below 7d threshold
        findings = sweep_silences(store, now=T0 + 8)
        assert len(findings) == 0


# ═══════════════════════════════════════════════════════════════════
# TRACKER RUN (end-to-end with stub notifier)
# ═══════════════════════════════════════════════════════════════════


class TestTrackerRun:
    def test_records_event_inbox_and_notifies(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store, title="LLM 应用", company="阿里")
        app_id = _add_app(store, job_id)
        ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)

        notifier = _StubNotifier()
        counters = tracker_run(store, notifier=notifier, now=T0 + 8)

        assert counters["silences_found"] == 1
        assert counters["events_recorded"] == 1
        assert counters["inbox_enqueued"] == 1
        assert counters["notify_ok"] == 1

        # Verify the inferred event was recorded
        events = ae.list_events(store, app_id)
        silent_checks = [e for e in events if e.kind == "silent_check"]
        assert len(silent_checks) == 1
        assert silent_checks[0].source == "inferred"
        assert silent_checks[0].payload["threshold_days"] == 7

        # Verify inbox item was created
        from offerguide import inbox as ib

        items = ib.list_items(store, status="pending")
        assert len(items) == 1
        assert "阿里" in items[0].title
        assert items[0].kind == "ambient_alert"
        assert items[0].payload["application_id"] == app_id

        # Verify notifier was called
        assert len(notifier.calls) == 1
        assert "7d" in notifier.calls[0]["title"]

    def test_idempotent_second_run_no_duplicates(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id)
        ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)

        notifier = _StubNotifier()
        c1 = tracker_run(store, notifier=notifier, now=T0 + 8)
        c2 = tracker_run(store, notifier=notifier, now=T0 + 10)

        assert c1["silences_found"] == 1
        assert c2["silences_found"] == 0  # already alerted at 7d

    def test_continues_when_notifier_fails(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)

        j1 = _add_job(store, title="A", company="X")
        a1 = _add_app(store, j1)
        ae.record(store, application_id=a1, kind="submitted", occurred_at=T0)

        j2 = _add_job(store, title="B", company="Y")
        a2 = _add_app(store, j2)
        ae.record(store, application_id=a2, kind="submitted", occurred_at=T0)

        notifier = _StubNotifier(fail=True)
        counters = tracker_run(store, notifier=notifier, now=T0 + 8)

        # Both apps processed even though notifier fails
        assert counters["silences_found"] == 2
        assert counters["events_recorded"] == 2
        assert counters["inbox_enqueued"] == 2
        assert counters["notify_failed"] == 2
        assert counters["notify_ok"] == 0

    def test_empty_when_no_silences(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        notifier = _StubNotifier()
        counters = tracker_run(store, notifier=notifier, now=T0)
        assert counters == {
            "silences_found": 0,
            "events_recorded": 0,
            "inbox_enqueued": 0,
            "notify_ok": 0,
            "notify_failed": 0,
        }

    def test_multiple_thresholds_across_multiple_runs(self, tmp_path: Path) -> None:
        """Simulate: day 8 → 7d alert, day 15 → 14d alert, day 35 → 30d alert."""
        store = _make_store(tmp_path)
        job_id = _add_job(store)
        app_id = _add_app(store, job_id)
        ae.record(store, application_id=app_id, kind="submitted", occurred_at=T0)

        notifier = _StubNotifier()

        c1 = tracker_run(store, notifier=notifier, now=T0 + 8)
        assert c1["silences_found"] == 1
        assert notifier.calls[-1]["level"] == "info"

        c2 = tracker_run(store, notifier=notifier, now=T0 + 15)
        assert c2["silences_found"] == 1
        assert notifier.calls[-1]["level"] == "warn"

        c3 = tracker_run(store, notifier=notifier, now=T0 + 35)
        assert c3["silences_found"] == 1
        assert notifier.calls[-1]["level"] == "high"

        # No more thresholds to fire
        c4 = tracker_run(store, notifier=notifier, now=T0 + 50)
        assert c4["silences_found"] == 0

        assert len(notifier.calls) == 3
