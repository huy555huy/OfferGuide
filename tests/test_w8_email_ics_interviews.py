"""Email classifier and ICS parser integration tests.

Two features for closing the application-tracking gap:
- Pure-Python email pattern classifier (no IMAP, no LLM)
- RFC 5545 ICS parser → interview event with scheduled_at
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import application_events as ae
from offerguide import email_classifier as ec
from offerguide import ics_parser
from offerguide.config import Settings
from offerguide.skills import discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# EMAIL CLASSIFIER
# ═══════════════════════════════════════════════════════════════════


class TestEmailClassifyKinds:
    def test_offer_email(self) -> None:
        result = ec.classify(
            "主题: 录用通知 - 字节跳动\n非常荣幸通知您，您已被录用..."
        )
        assert result.kind == "offer"
        assert result.confidence > 0.5

    def test_interview_invite(self) -> None:
        result = ec.classify(
            "主题: 面试邀请\n邀请您参加 字节跳动 AI Agent 实习的一面，"
            "面试时间: 2026-05-20 14:00。"
        )
        assert result.kind == "interview"

    def test_assessment(self) -> None:
        result = ec.classify(
            "笔试通知 - 阿里巴巴\n您的在线笔试链接: https://test.alibaba.com/..."
        )
        assert result.kind == "assessment"

    def test_rejection(self) -> None:
        result = ec.classify(
            "感谢您对 字节跳动 的关注。很遗憾通知您，您未能进入下一轮筛选..."
        )
        assert result.kind == "rejected"

    def test_replied(self) -> None:
        result = ec.classify(
            "我们已收到您的简历，将在 5 个工作日内与您联系。"
        )
        assert result.kind == "replied"

    def test_unrelated_email(self) -> None:
        result = ec.classify(
            "Newsletter - This week in tech: 10 amazing AI startups to watch"
        )
        assert result.kind == "unrelated"
        assert result.confidence == 0.0

    def test_empty_text(self) -> None:
        result = ec.classify("")
        assert result.kind == "unrelated"


class TestEmailClassifyMatching:
    def test_matches_known_company(self) -> None:
        result = ec.classify(
            "字节跳动 面试邀请",
            known_companies=["字节跳动", "阿里巴巴"],
        )
        assert result.matched_company == "字节跳动"

    def test_prefers_longer_match(self) -> None:
        result = ec.classify(
            "阿里云面试邀请",
            known_companies=["阿里", "阿里云"],
        )
        assert result.matched_company == "阿里云"

    def test_application_id_when_unique(self) -> None:
        result = ec.classify(
            "字节跳动 面试邀请",
            known_companies=["字节跳动"],
            known_apps_by_company={"字节跳动": [42]},
        )
        assert result.matched_application_id == 42

    def test_application_id_none_when_ambiguous(self) -> None:
        result = ec.classify(
            "字节跳动 面试邀请",
            known_companies=["字节跳动"],
            known_apps_by_company={"字节跳动": [1, 2, 3]},
        )
        assert result.matched_application_id is None


class TestEmailDumpSplit:
    def test_split_mbox_style(self) -> None:
        blob = (
            "From: hr@bytedance.com\nSubject: 面试邀请\nBody...\n\n"
            "From: noreply@alibaba.com\nSubject: 笔试通知\nBody..."
        )
        chunks = ec.split_email_dump(blob)
        assert len(chunks) == 2
        assert "From: hr@bytedance.com" in chunks[0]
        assert "From: noreply@alibaba.com" in chunks[1]

    def test_split_blank_lines(self) -> None:
        blob = "Email 1 body\n\n\nEmail 2 body\n\n\nEmail 3 body"
        chunks = ec.split_email_dump(blob)
        assert len(chunks) == 3

    def test_no_separator_returns_one(self) -> None:
        chunks = ec.split_email_dump("just one email")
        assert chunks == ["just one email"]

    def test_empty_returns_empty(self) -> None:
        assert ec.split_email_dump("") == []


@pytest.fixture
def app_setup(tmp_path: Path, master_resume_source_factory):
    store = offerguide.Store(tmp_path / "ec.db")
    store.init_schema()
    profile = master_resume_source_factory("x")
    app = create_app(
        settings=Settings(), store=store, master_source=profile,
        skills=discover_skills(SKILLS_ROOT), runtime=None,
        notifier=ConsoleNotifier(),
    )
    return app, store


def _seed_app_for_company(store, company: str) -> int:
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('manual', 't', ?, 'jd', ?)",
            (company, f"hash_{company}"),
        )
        cur = conn.execute(
            "INSERT INTO applications(job_id, status) VALUES "
            "((SELECT id FROM jobs WHERE company = ? LIMIT 1), 'applied')",
            (company,),
        )
        return int(cur.lastrowid or 0)


class TestEmailClassifyEndpoint:
    def test_classify_single(self, app_setup) -> None:
        app, store = app_setup
        _seed_app_for_company(store, "字节跳动")
        resp = TestClient(app).post(
            "/api/email/classify",
            json={"text": "字节跳动 面试邀请", "batch": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["results"][0]["kind"] == "interview"
        assert data["results"][0]["matched_company"] == "字节跳动"

    def test_classify_batch(self, app_setup) -> None:
        app, store = app_setup
        _seed_app_for_company(store, "字节跳动")
        _seed_app_for_company(store, "阿里巴巴")
        blob = (
            "字节跳动 面试邀请\n\n\n"
            "阿里巴巴 笔试通知\n\n\n"
            "Spam email about NFTs"
        )
        resp = TestClient(app).post(
            "/api/email/classify",
            json={"text": blob, "batch": True},
        )
        data = resp.json()
        assert data["count"] == 3
        kinds = [r["kind"] for r in data["results"]]
        assert "interview" in kinds
        assert "assessment" in kinds
        assert "unrelated" in kinds

    def test_returns_application_id_when_resolvable(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        resp = TestClient(app).post(
            "/api/email/classify",
            json={"text": "字节跳动 面试邀请", "batch": False},
        )
        data = resp.json()
        assert data["results"][0]["matched_application_id"] == app_id


# ═══════════════════════════════════════════════════════════════════
# ICS PARSER
# ═══════════════════════════════════════════════════════════════════


_SAMPLE_ICS = """BEGIN:VCALENDAR
VERSION:2.0
METHOD:REQUEST
BEGIN:VEVENT
UID:abc-123
SEQUENCE:0
STATUS:CONFIRMED
SUMMARY:字节跳动 - AI Agent 一面
DTSTART:20260520T060000Z
DTEND:20260520T070000Z
DESCRIPTION:面试链接 https://meet.example.com/...
LOCATION:线上
END:VEVENT
END:VCALENDAR"""


_NON_INTERVIEW_ICS = """BEGIN:VCALENDAR
BEGIN:VEVENT
SUMMARY:Lunch with Bob
DTSTART:20260520T120000Z
END:VEVENT
END:VCALENDAR"""


class TestIcsParser:
    def test_parses_interview_event(self) -> None:
        events = ics_parser.parse_ics(_SAMPLE_ICS)
        assert len(events) == 1
        e = events[0]
        assert "字节跳动" in e.summary
        assert "一面" in e.summary  # '一面' is in interview hints, not '面试' literally
        assert e.is_interview is True
        assert e.dtstart_utc is not None
        assert e.dtstart_utc.year == 2026
        assert e.dtstart_utc.month == 5
        assert e.dtstart_utc.tzinfo is UTC
        assert e.uid == "abc-123"
        assert e.sequence == 0
        assert e.status == "CONFIRMED"
        assert e.method == "REQUEST"
        assert e.calendar_action == "scheduled"
        assert e.round == "一面"

    def test_non_interview_marked_false(self) -> None:
        events = ics_parser.parse_ics(_NON_INTERVIEW_ICS)
        assert len(events) == 1
        assert events[0].is_interview is False

    @pytest.mark.parametrize(
        ("method", "expected_action", "selected"),
        [
            (None, "scheduled", True),
            ("REQUEST", "scheduled", True),
            ("PUBLISH", "scheduled", True),
            ("ADD", "scheduled", True),
            ("CANCEL", "cancelled", True),
            ("REPLY", "unsupported", False),
            ("COUNTER", "unsupported", False),
            ("DECLINECOUNTER", "unsupported", False),
            ("REFRESH", "unsupported", False),
            ("X-UNKNOWN", "unsupported", False),
        ],
    )
    def test_calendar_method_controls_lifecycle_action(
        self,
        method: str | None,
        expected_action: str,
        selected: bool,
    ) -> None:
        method_line = f"METHOD:{method}\n" if method is not None else ""
        ics = _SAMPLE_ICS.replace("METHOD:REQUEST\n", method_line)

        event = ics_parser.parse_ics(ics)[0]

        assert event.method == method
        assert event.calendar_action == expected_action
        assert (ics_parser.select_first_interview([event]) is event) is selected

    def test_selector_skips_unsupported_event_and_uses_later_request(self) -> None:
        unsupported = _SAMPLE_ICS.replace("METHOD:REQUEST", "METHOD:REPLY")
        supported = _SAMPLE_ICS.replace("UID:abc-123", "UID:abc-456")

        selected = ics_parser.select_first_interview(
            [
                *ics_parser.parse_ics(unsupported),
                *ics_parser.parse_ics(supported),
            ]
        )

        assert selected is not None
        assert selected.uid == "abc-456"

    def test_cancelled_status_takes_precedence_over_non_scheduling_method(self) -> None:
        ics = _SAMPLE_ICS.replace("METHOD:REQUEST", "METHOD:REPLY").replace(
            "STATUS:CONFIRMED", "STATUS:CANCELLED"
        )

        event = ics_parser.parse_ics(ics)[0]

        assert event.method == "REPLY"
        assert event.status == "CANCELLED"
        assert event.calendar_action == "cancelled"
        assert ics_parser.select_first_interview([event]) is event

    def test_select_first_interview_picks_only_interview(self) -> None:
        events = ics_parser.parse_ics(_SAMPLE_ICS + "\n" + _NON_INTERVIEW_ICS)
        chosen = ics_parser.select_first_interview(events)
        assert chosen is not None
        assert "字节跳动" in chosen.summary

    def test_select_returns_none_when_no_interview(self) -> None:
        events = ics_parser.parse_ics(_NON_INTERVIEW_ICS)
        assert ics_parser.select_first_interview(events) is None

    def test_empty_returns_empty(self) -> None:
        assert ics_parser.parse_ics("") == []

    def test_dt_parser_handles_date_only(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\n"
            "BEGIN:VEVENT\nSUMMARY:面试 ALL DAY\nDTSTART:20260601\nEND:VEVENT\n"
            "END:VCALENDAR"
        )
        events = ics_parser.parse_ics(ics)
        assert events[0].dtstart_utc is None
        assert events[0].dtstart_date == date(2026, 6, 1)
        assert events[0].dtstart_local is None

    def test_floating_time_is_not_claimed_as_utc(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:floating\n"
            "SUMMARY:Interview\nDTSTART:20260601T140000\nEND:VEVENT\n"
            "END:VCALENDAR"
        )

        event = ics_parser.parse_ics(ics)[0]

        assert event.dtstart_utc is None
        assert event.dtstart_local is not None
        assert event.dtstart_local.isoformat() == "2026-06-01T14:00:00"
        assert event.dtstart_tzid is None

    def test_unknown_tzid_is_preserved_without_guessing_utc(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:unknown-zone\n"
            "SUMMARY:Interview\nDTSTART;TZID=Mars/Olympus:20260601T140000\n"
            "END:VEVENT\nEND:VCALENDAR"
        )

        event = ics_parser.parse_ics(ics)[0]

        assert event.dtstart_utc is None
        assert event.dtstart_local is not None
        assert event.dtstart_local.isoformat() == "2026-06-01T14:00:00"
        assert event.dtstart_tzid == "Mars/Olympus"

    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            ("Online OA assessment", True),
            ("在线OA通知", True),
            ("Roadmap planning", False),
            ("Coauthor sync", False),
        ],
    )
    def test_oa_uses_ascii_word_boundaries(
        self,
        summary: str,
        expected: bool,
    ) -> None:
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:oa-boundary\n"
            f"SUMMARY:{summary}\nEND:VEVENT\nEND:VCALENDAR"
        )

        assert ics_parser.parse_ics(ics)[0].is_interview is expected

    @pytest.mark.parametrize(
        ("summary", "expected_round"),
        [
            ("Example 三面", "三面"),
            ("Example HR面", "HR面"),
            ("Example HR 面", "HR面"),
            ("Example 3rd interview", "三面"),
        ],
    )
    def test_infers_later_interview_rounds(
        self,
        summary: str,
        expected_round: str,
    ) -> None:
        ics = (
            "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:later-round\n"
            f"SUMMARY:{summary}\nEND:VEVENT\nEND:VCALENDAR"
        )

        event = ics_parser.parse_ics(ics)[0]

        assert event.is_interview is True
        assert event.round == expected_round

    @pytest.mark.parametrize("sequence", ["-1", "not-an-integer", "1.5"])
    def test_rejects_invalid_sequence_instead_of_coercing_to_zero(
        self,
        sequence: str,
    ) -> None:
        ics = _SAMPLE_ICS.replace("SEQUENCE:0", f"SEQUENCE:{sequence}")

        with pytest.raises(ValueError, match="SEQUENCE must be a non-negative integer"):
            ics_parser.parse_ics(ics)

    def test_tzid_folded_and_escaped_values_follow_rfc5545(self) -> None:
        ics = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "BEGIN:VEVENT\r\n"
            "SUMMARY:示例科技\\, AI\\; Interview\r\n"
            "DTSTART;TZID=Asia/Shanghai:20260720T140000\r\n"
            "DESCRIPTION:面试第一行\\n第二行与\r\n"
            " 后半段\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

        event = ics_parser.parse_ics(ics)[0]

        assert event.summary == "示例科技, AI; Interview"
        assert event.description == "面试第一行\n第二行与后半段"
        assert event.dtstart_tzid == "Asia/Shanghai"
        assert event.dtstart_local is not None
        assert event.dtstart_local.hour == 14
        assert event.dtstart_utc is not None
        assert event.dtstart_utc.isoformat() == "2026-07-20T06:00:00+00:00"

    def test_julianday_conversion(self) -> None:
        from datetime import datetime
        dt = datetime(2026, 5, 20, 6, 0, 0, tzinfo=UTC)
        jd = ics_parser.datetime_to_julianday(dt)
        # 2026-05-20 06:00 UTC → JD ≈ 2461180.75
        assert 2461180.0 < jd < 2461181.5
        with pytest.raises(ValueError, match="floating datetime"):
            ics_parser.datetime_to_julianday(datetime(2026, 5, 20, 14, 0, 0))


class TestIcsEndpoint:
    def test_uploads_and_records_interview(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        with store.connect() as conn:
            window_start = float(conn.execute("SELECT julianday('now')").fetchone()[0])
        resp = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _SAMPLE_ICS},
        )
        with store.connect() as conn:
            window_end = float(conn.execute("SELECT julianday('now')").fetchone()[0])
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["application_id"] == app_id
        assert data["calendar_record_state"] == "recorded"
        assert data["recorded"] is True
        assert data["uid"] == "abc-123"
        assert data["sequence"] == 0
        assert data["round"] == "一面"
        assert "一面" in data["summary"] or "面试" in data["summary"]

        # The interview event was actually recorded
        events = ae.list_events(store, app_id)
        kinds = [e.kind for e in events]
        assert "interview" in kinds
        interview_event = next(event for event in events if event.kind == "interview")
        assert window_start <= interview_event.occurred_at <= window_end
        assert interview_event.payload["scheduled_at"] == "2026-05-20T06:00:00+00:00"
        assert interview_event.payload["scheduled_at_local"] == "2026-05-20T06:00:00+00:00"
        assert interview_event.payload["scheduled_date"] is None
        assert interview_event.payload["scheduled_tzid"] is None
        assert interview_event.payload["ics_uid"] == "abc-123"
        assert interview_event.payload["ics_sequence"] == 0
        assert interview_event.payload["ics_status"] == "CONFIRMED"
        assert interview_event.payload["ics_method"] == "REQUEST"
        assert interview_event.payload["calendar_action"] == "scheduled"
        assert interview_event.payload["round"] == "一面"
        # Status synced
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()
        assert row[0] == "1st_interview"

    @pytest.mark.parametrize(
        "method",
        ["REPLY", "COUNTER", "DECLINECOUNTER", "REFRESH", "X-UNKNOWN"],
    )
    def test_unsupported_calendar_method_cannot_create_interview_event(
        self,
        app_setup,
        method: str,
    ) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        unsupported = _SAMPLE_ICS.replace("METHOD:REQUEST", f"METHOD:{method}")

        response = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": unsupported},
        )

        assert response.status_code == 400
        assert ae.list_events(store, app_id) == []

    def test_duplicate_revision_is_idempotent(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        client = TestClient(app)

        first = client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _SAMPLE_ICS},
        )
        duplicate = client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _SAMPLE_ICS},
        )

        assert first.status_code == 200
        assert duplicate.status_code == 200
        assert duplicate.json()["calendar_record_state"] == "duplicate"
        assert duplicate.json()["recorded"] is False
        assert len(ae.list_events(store, app_id)) == 1

    def test_old_exact_revision_is_stale_after_a_newer_sequence(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        client = TestClient(app)
        newer = _SAMPLE_ICS.replace("SEQUENCE:0", "SEQUENCE:1").replace(
            "一面", "二面"
        )
        assert client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _SAMPLE_ICS},
        ).status_code == 200
        assert client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": newer},
        ).status_code == 200

        replay = client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _SAMPLE_ICS},
        )

        assert replay.status_code == 200
        assert replay.json()["calendar_record_state"] == "stale"
        assert len(ae.list_events(store, app_id)) == 2

    def test_higher_sequence_without_round_never_downgrades_interview_status(
        self,
        app_setup,
    ) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "Example")
        client = TestClient(app)
        second_round = _SAMPLE_ICS.replace("SEQUENCE:0", "SEQUENCE:1").replace(
            "字节跳动 - AI Agent 一面", "Example 二面"
        )
        no_round = second_round.replace("SEQUENCE:1", "SEQUENCE:2").replace(
            "Example 二面", "Example Interview update"
        )
        third_round = no_round.replace("SEQUENCE:2", "SEQUENCE:3").replace(
            "Example Interview update", "Example 三面"
        )
        final_without_round = third_round.replace("SEQUENCE:3", "SEQUENCE:4").replace(
            "Example 三面", "Example Interview time update"
        )

        for ics_text in (second_round, no_round, third_round, final_without_round):
            response = client.post(
                f"/api/applications/{app_id}/events/ics",
                data={"ics_text": ics_text},
            )
            assert response.status_code == 200

        with store.connect() as conn:
            status = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()[0]
        assert status == "final_interview"

    def test_calendar_event_and_application_status_are_one_transaction(
        self,
        app_setup,
    ) -> None:
        _app, store = app_setup
        app_id = _seed_app_for_company(store, "Example")
        with store.connect() as conn:
            conn.execute(
                "CREATE TRIGGER reject_interview_status BEFORE UPDATE ON applications "
                "WHEN NEW.status = '1st_interview' BEGIN "
                "SELECT RAISE(ABORT, 'status write failed'); END"
            )

        with pytest.raises(sqlite3.IntegrityError, match="status write failed"):
            ae.record_calendar_event(
                store,
                application_id=app_id,
                uid="atomic-calendar",
                sequence=0,
                action="scheduled",
                payload={"round": "一面"},
            )

        assert ae.list_events(store, app_id) == []
        with store.connect() as conn:
            status = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()[0]
        assert status == "applied"

    def test_concurrent_sequences_cannot_leave_status_at_the_older_round(
        self,
        app_setup,
    ) -> None:
        _app, store = app_setup
        app_id = _seed_app_for_company(store, "Example")
        barrier = threading.Barrier(2)

        def record(sequence: int, round_name: str):
            barrier.wait()
            return ae.record_calendar_event(
                store,
                application_id=app_id,
                uid="concurrent-calendar",
                sequence=sequence,
                action="scheduled",
                payload={"round": round_name},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            older = executor.submit(record, 0, "一面")
            newer = executor.submit(record, 1, "二面")
            outcomes = [older.result(), newer.result()]

        assert {outcome.state for outcome in outcomes} <= {"recorded", "stale"}
        assert any(outcome.state == "recorded" for outcome in outcomes)
        with store.connect() as conn:
            status = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()[0]
        assert status == "2nd_interview"

    def test_new_sequence_updates_round_and_cancel_is_a_distinct_action(
        self,
        app_setup,
    ) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        client = TestClient(app)
        assert client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _SAMPLE_ICS},
        ).status_code == 200
        second_round = _SAMPLE_ICS.replace("SEQUENCE:0", "SEQUENCE:1").replace(
            "一面", "二面"
        )

        updated = client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": second_round},
        )

        assert updated.status_code == 200
        assert updated.json()["calendar_record_state"] == "recorded"
        assert updated.json()["round"] == "二面"
        with store.connect() as conn:
            status = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()[0]
        assert status == "2nd_interview"

        stale_cancel = (
            _SAMPLE_ICS.replace("METHOD:REQUEST", "METHOD:CANCEL")
            .replace("STATUS:CONFIRMED", "STATUS:CANCELLED")
        )
        stale = client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": stale_cancel},
        )
        assert stale.status_code == 200
        assert stale.json()["calendar_record_state"] == "stale"
        assert len(ae.list_events(store, app_id)) == 2

        current_cancel = (
            second_round.replace("METHOD:REQUEST", "METHOD:CANCEL")
            .replace("STATUS:CONFIRMED", "STATUS:CANCELLED")
            .replace(
                "SUMMARY:字节跳动 - AI Agent 二面",
                "SUMMARY:Cancelled appointment",
            )
            .replace(
                "DESCRIPTION:面试链接 https://meet.example.com/...",
                "DESCRIPTION:Cancelled by organizer",
            )
        )
        cancelled = client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": current_cancel},
        )
        duplicate_cancel = client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": current_cancel},
        )

        assert cancelled.status_code == 200
        assert cancelled.json()["calendar_record_state"] == "recorded"
        assert cancelled.json()["event_kind"] == "interview_cancelled"
        assert duplicate_cancel.json()["calendar_record_state"] == "duplicate"
        recorded = ae.list_events(store, app_id)
        assert [event.kind for event in recorded] == [
            "interview",
            "interview",
            "interview_cancelled",
        ]
        with store.connect() as conn:
            status = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()[0]
        assert status == "2nd_interview"

        final_round = second_round.replace("SEQUENCE:1", "SEQUENCE:2").replace(
            "二面", "终面"
        )
        rescheduled = client.post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": final_round},
        )
        assert rescheduled.status_code == 200
        assert rescheduled.json()["calendar_record_state"] == "recorded"
        assert rescheduled.json()["round"] == "终面"
        with store.connect() as conn:
            status = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()[0]
        assert status == "final_interview"

    def test_cancel_without_request_does_not_advance_application(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        cancellation = (
            _SAMPLE_ICS.replace("METHOD:REQUEST", "METHOD:CANCEL")
            .replace("STATUS:CONFIRMED", "STATUS:CANCELLED")
        )

        response = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": cancellation},
        )

        assert response.status_code == 200
        assert response.json()["calendar_action"] == "cancelled"
        assert response.json()["event_kind"] == "interview_cancelled"
        events = ae.list_events(store, app_id)
        assert [event.kind for event in events] == ["interview_cancelled"]
        assert events[0].payload["ics_text"] == cancellation
        assert events[0].payload["ics_method"] == "CANCEL"
        assert events[0].payload["ics_status"] == "CANCELLED"
        with store.connect() as conn:
            status = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()[0]
        assert status == "applied"

    def test_route_preserves_unknown_timezone_and_floating_local_time(
        self,
        app_setup,
    ) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "Example")
        ics_text = """BEGIN:VCALENDAR
VERSION:2.0
METHOD:REQUEST
BEGIN:VEVENT
UID:unknown-timezone
SEQUENCE:3
SUMMARY:Example 一面
DTSTART;TZID=Mars/Olympus:20260720T140000
END:VEVENT
END:VCALENDAR"""

        response = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": ics_text},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["scheduled_at"] is None
        assert data["scheduled_at_local"] == "2026-07-20T14:00:00"
        assert data["scheduled_date"] is None
        assert data["scheduled_tzid"] == "Mars/Olympus"
        payload = ae.list_events(store, app_id)[0].payload
        assert payload["scheduled_at"] is None
        assert payload["scheduled_at_local"] == "2026-07-20T14:00:00"
        assert payload["scheduled_tzid"] == "Mars/Olympus"

    def test_route_preserves_value_date_without_inventing_time(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "Example")
        ics_text = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:date-only
SUMMARY:Example Interview
DTSTART;VALUE=DATE:20260720
END:VEVENT
END:VCALENDAR"""

        response = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": ics_text},
        )

        assert response.status_code == 200
        assert response.json()["scheduled_at"] is None
        assert response.json()["scheduled_at_local"] is None
        assert response.json()["scheduled_date"] == "2026-07-20"
        payload = ae.list_events(store, app_id)[0].payload
        assert payload["scheduled_at"] is None
        assert payload["scheduled_at_local"] is None
        assert payload["scheduled_date"] == "2026-07-20"

    def test_preserves_complete_long_ics_and_description(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        long_description = "面试安排：" + ("完整上下文不能被截断。" * 2000)
        ics_text = _SAMPLE_ICS.replace(
            "DESCRIPTION:面试链接 https://meet.example.com/...",
            f"DESCRIPTION:{long_description}",
        )

        resp = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": ics_text},
        )

        assert resp.status_code == 200
        recorded = ae.list_events(store, app_id)[-1]
        assert recorded.payload["ics_text"] == ics_text
        assert recorded.payload["description"] == long_description

    def test_ics_without_interview_returns_400(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "X")
        resp = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _NON_INTERVIEW_ICS},
        )
        assert resp.status_code == 400

    def test_interview_without_uid_returns_400(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "X")

        response = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _SAMPLE_ICS.replace("UID:abc-123\n", "")},
        )

        assert response.status_code == 400
        assert ae.list_events(store, app_id) == []

    @pytest.mark.parametrize("sequence", ["-1", "invalid"])
    def test_route_rejects_invalid_sequence(
        self,
        app_setup,
        sequence: str,
    ) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "X")
        invalid = _SAMPLE_ICS.replace("SEQUENCE:0", f"SEQUENCE:{sequence}")

        response = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": invalid},
        )

        assert response.status_code == 400
        assert "SEQUENCE must be a non-negative integer" in response.json()["detail"]
        assert ae.list_events(store, app_id) == []
