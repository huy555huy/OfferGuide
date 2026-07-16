"""W8' — /applications page + 1-click event logging.

Tests:
- /applications renders empty state when no apps
- /applications lists apps + their events
- POST /api/applications/{id}/event records the event AND syncs status
- Invalid kind returns 400
- Invalid app_id returns 400 (FK constraint failure surfaces)
- Status pill class reflects the synced state
- Elapsed time is factual and has no reminder threshold
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import application_events as ae
from offerguide.config import Settings
from offerguide.skills import discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def app_setup(tmp_path: Path, master_resume_source_factory):
    store = offerguide.Store(tmp_path / "apps.db")
    store.init_schema()
    profile = master_resume_source_factory("x")
    skills = discover_skills(SKILLS_ROOT)
    app = create_app(
        settings=Settings(),
        store=store,
        master_source=profile,
        skills=skills,
        runtime=None,
        notifier=ConsoleNotifier(),
    )
    return app, store


def _seed_app(store, *, title: str, company: str, status: str = "applied") -> int:
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('manual', ?, ?, 'jd', ?)",
            (title, company, f"hash_{title}"),
        )
        cur = conn.execute(
            "INSERT INTO applications(job_id, status) VALUES "
            "((SELECT id FROM jobs WHERE title = ? LIMIT 1), ?)",
            (title, status),
        )
        return int(cur.lastrowid or 0)


def _seed_submitted_workspace(store, app_id: int) -> int:
    with store.connect() as conn:
        job_id = int(
            conn.execute(
                "SELECT job_id FROM applications WHERE id = ?", (app_id,)
            ).fetchone()[0]
        )
        return int(
            conn.execute(
                "INSERT INTO resume_workspaces("
                "application_id, status, job_snapshot_json, master_source_sha256, "
                "context_json, resume_document_json, pdf_path, pdf_sha256, "
                "apply_pack_json, submitted_at"
                ") VALUES (?, 'submitted', ?, ?, '{}', '{}', ?, ?, '{}', "
                "julianday('now')) RETURNING id",
                (
                    app_id,
                    json.dumps({"job_id": job_id}),
                    "a" * 64,
                    f"/tmp/application-{app_id}.pdf",
                    "b" * 64,
                ),
            ).fetchone()[0]
        )


# ═══════════════════════════════════════════════════════════════════
# /applications page
# ═══════════════════════════════════════════════════════════════════


class TestApplicationsPage:
    def test_empty_state(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/applications")
        assert resp.status_code == 200
        assert "应用追踪" in resp.text
        assert "还没有投递记录" in resp.text

    def test_lists_applications_with_events(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app(store, title="AI Agent 实习", company="字节跳动")
        ae.record(store, application_id=app_id, kind="submitted", source="manual")
        ae.record(store, application_id=app_id, kind="viewed",   source="email")

        resp = TestClient(app).get("/applications")
        assert "AI Agent 实习" in resp.text
        assert "字节跳动" in resp.text
        # Event timeline pills
        assert "submitted" in resp.text
        assert "viewed" in resp.text

    def test_each_application_keeps_its_apply_pack_entry(self, app_setup) -> None:
        app, store = app_setup
        first_id = _seed_app(store, title="Agent A", company="A")
        second_id = _seed_app(store, title="Agent B", company="B")
        with store.connect() as conn:
            job_ids = {
                int(row[0])
                for row in conn.execute(
                    "SELECT job_id FROM applications WHERE id IN (?, ?)",
                    (first_id, second_id),
                ).fetchall()
            }

        resp = TestClient(app).get("/applications")

        assert resp.text.count("查看投递材料") == 2
        for job_id in job_ids:
            assert f'href="/jobs/{job_id}/apply-pack"' in resp.text

    def test_submitted_application_keeps_interview_entry_after_event_swap(
        self, app_setup
    ) -> None:
        app, store = app_setup
        app_id = _seed_app(store, title="Agent C", company="C")
        _seed_submitted_workspace(store, app_id)
        with store.connect() as conn:
            job_id = int(
                conn.execute(
                    "SELECT job_id FROM applications WHERE id = ?", (app_id,)
                ).fetchone()[0]
            )
        client = TestClient(app)

        initial = client.get("/applications")
        swapped = client.post(
            f"/api/applications/{app_id}/event", data={"kind": "viewed"}
        )

        for response in (initial, swapped):
            assert response.status_code == 200
            assert f'href="/jobs/{job_id}/apply-pack"' in response.text
            assert f'href="/jobs/{job_id}/post-apply-pack"' in response.text
            assert "查看投递材料" in response.text
            assert "data-interview-status=" in response.text

    def test_active_terminal_counts(self, app_setup) -> None:
        app, store = app_setup
        _seed_app(store, title="A", company="X", status="applied")
        _seed_app(store, title="B", company="Y", status="rejected")
        _seed_app(store, title="C", company="Z", status="offer")

        resp = TestClient(app).get("/applications")
        # Header shows "活跃 1 · 终态 2"
        assert "活跃 1" in resp.text
        assert "终态 2" in resp.text

    def test_elapsed_time_renders_without_a_silence_verdict(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app(store, title="老投递", company="X")
        # Record submitted at a julianday computed by SQLite itself, minus N days
        with store.connect() as conn:
            now_jd = conn.execute("SELECT julianday('now')").fetchone()[0]
        ae.record(
            store, application_id=app_id, kind="submitted",
            source="manual", occurred_at=now_jd - 16,
        )

        resp = TestClient(app).get("/applications")
        assert "距上次进展" in resp.text
        import re as _re
        assert _re.search(r"距上次进展\s*\d+d", resp.text)


# ═══════════════════════════════════════════════════════════════════
# POST /api/applications/{id}/event
# ═══════════════════════════════════════════════════════════════════


class TestEventLogging:
    def test_log_viewed_event_records_and_syncs_status(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app(store, title="t", company="c")
        client = TestClient(app)

        resp = client.post(
            f"/api/applications/{app_id}/event",
            data={"kind": "viewed"},
        )
        assert resp.status_code == 200

        # Event recorded
        events = ae.list_events(store, app_id)
        assert any(e.kind == "viewed" and e.source == "manual" for e in events)

        # Status synced (viewed → "viewed")
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()
        assert row[0] == "viewed"

    def test_log_offer_syncs_to_terminal_status(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app(store, title="t", company="c", status="hr_replied")
        client = TestClient(app)

        client.post(f"/api/applications/{app_id}/event", data={"kind": "offer"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE id = ?", (app_id,)
            ).fetchone()
        assert row[0] == "offer"

    def test_invalid_kind_returns_400(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app(store, title="t", company="c")
        client = TestClient(app)

        resp = client.post(
            f"/api/applications/{app_id}/event",
            data={"kind": "rocket_launch"},
        )
        assert resp.status_code == 400

    def test_unknown_app_id_returns_4xx(self, app_setup) -> None:
        app, _ = app_setup
        client = TestClient(app)
        resp = client.post(
            "/api/applications/9999/event", data={"kind": "viewed"}
        )
        # Either FK violation → 400, or "not found after event" → 404
        assert resp.status_code in (400, 404)

    def test_response_body_is_swap_in_card_fragment(self, app_setup) -> None:
        app, store = app_setup
        app_id = _seed_app(store, title="t", company="c")
        client = TestClient(app)

        resp = client.post(f"/api/applications/{app_id}/event", data={"kind": "viewed"})
        # The HTMX swap target is the card with id app-row-{N}
        assert f'id="app-row-{app_id}"' in resp.text
        # And contains the new event pill
        assert "viewed" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Navigation includes the applications tab
# ═══════════════════════════════════════════════════════════════════


def test_topbar_has_applications_link(app_setup) -> None:
    """Mission Control keeps application tracking reachable through Pipeline."""
    app, _ = app_setup
    resp = TestClient(app).get("/")
    assert 'href="/pipeline"' in resp.text
    assert "Pipeline" in resp.text


def test_active_tab_highlights_applications(app_setup) -> None:
    app, _ = app_setup
    resp = TestClient(app).get("/applications")
    assert 'aria-label="Pipeline"' in resp.text
    assert 'href="/pipeline"' in resp.text
    assert 'item active' in resp.text
