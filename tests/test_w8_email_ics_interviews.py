"""W8'' — Email classifier + ICS parser + 面经 paste UI.

Three small but real features for closing the application-tracking gap:
- Pure-Python email pattern classifier (no IMAP, no LLM)
- RFC 5545 ICS parser → interview event with scheduled_at
- 面经 manual-paste UI writing to interview_experiences table
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import application_events as ae
from offerguide import email_classifier as ec
from offerguide import ics_parser, interview_corpus
from offerguide.config import Settings
from offerguide.profile import UserProfile
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
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "ec.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="x")
    app = create_app(
        settings=Settings(), store=store, profile=profile,
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
BEGIN:VEVENT
UID:abc-123
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

    def test_non_interview_marked_false(self) -> None:
        events = ics_parser.parse_ics(_NON_INTERVIEW_ICS)
        assert len(events) == 1
        assert events[0].is_interview is False

    def test_select_first_interview_picks_only_interview(self) -> None:
        events = ics_parser.parse_ics(_SAMPLE_ICS + _NON_INTERVIEW_ICS)
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
        assert events[0].dtstart_utc is not None
        assert events[0].dtstart_utc.day == 1

    def test_julianday_conversion(self) -> None:
        from datetime import datetime
        dt = datetime(2026, 5, 20, 6, 0, 0, tzinfo=UTC)
        jd = ics_parser.datetime_to_julianday(dt)
        # 2026-05-20 06:00 UTC → JD ≈ 2461180.75
        assert 2461180.0 < jd < 2461181.5


class TestIcsEndpoint:
    def test_uploads_and_records_interview(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "字节跳动")
        resp = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _SAMPLE_ICS},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["application_id"] == app_id
        assert "一面" in data["summary"] or "面试" in data["summary"]

        # The interview event was actually recorded
        events = ae.list_events(store, app_id)
        kinds = [e.kind for e in events]
        assert "interview" in kinds
        # Status synced
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()
        assert row[0] in ("1st_interview", "2nd_interview", "final_interview")

    def test_ics_without_interview_returns_400(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app_for_company(store, "X")
        resp = TestClient(app).post(
            f"/api/applications/{app_id}/events/ics",
            data={"ics_text": _NON_INTERVIEW_ICS},
        )
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════
# 面经 paste UI
# ═══════════════════════════════════════════════════════════════════


class TestInterviewPaste:
    def test_get_interviews_page(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/interviews")
        assert resp.status_code == 200
        assert "面经库" in resp.text
        # Sources select
        assert "xiaohongshu" in resp.text
        assert "1point3acres" in resp.text

    def test_paste_inserts_into_corpus(self, app_setup) -> None:
        app, store = app_setup
        resp = TestClient(app).post(
            "/api/interviews/paste",
            data={
                "company": "字节跳动",
                "raw_text": "面经 · 字节 AI Agent 一面: 问了 attention 缩放...",
                "source": "xiaohongshu",
                "role_hint": "AI Agent 实习",
                "source_url": "https://www.xiaohongshu.com/...",
            },
        )
        assert resp.status_code == 200

        rows = interview_corpus.fetch_for_company(store, "字节跳动")
        assert len(rows) == 1
        assert rows[0].source == "xiaohongshu"
        assert rows[0].role_hint == "AI Agent 实习"

    def test_paste_dedup_preserves_first(self, app_setup) -> None:
        app, store = app_setup
        client = TestClient(app)
        text = "重复面经原文..."
        client.post("/api/interviews/paste",
                    data={"company": "字节", "raw_text": text, "source": "manual_paste"})
        client.post("/api/interviews/paste",
                    data={"company": "字节", "raw_text": text, "source": "manual_paste"})
        rows = interview_corpus.fetch_for_company(store, "字节")
        assert len(rows) == 1  # second was deduped

    def test_paste_empty_returns_4xx(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).post(
            "/api/interviews/paste",
            data={"company": "字节", "raw_text": ""},
        )
        # Either 400 from our explicit check or 422 from FastAPI validation
        assert resp.status_code in (400, 422)


def test_topbar_has_面经_link(app_setup) -> None:
    app, _ = app_setup
    resp = TestClient(app).get("/")
    assert 'href="/interviews"' in resp.text
    assert "面经" in resp.text
