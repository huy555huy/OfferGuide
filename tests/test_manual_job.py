from __future__ import annotations

import json

from fastapi.testclient import TestClient

from offerguide.config import Settings
from offerguide.manual_job import intake_manual_job
from offerguide.memory import Store
from offerguide.research_agents.sources import SourceEvidenceStore
from offerguide.ui.web import create_app


def test_manual_intake_preserves_complete_text_and_reuses_one_job(tmp_path) -> None:
    store = Store(tmp_path / "manual.db")
    store.init_schema()
    source_store = SourceEvidenceStore(store)
    text = "MANUAL_JD_HEAD\n" + ("完整岗位正文，不允许截断。\n" * 80) + "MANUAL_JD_TAIL"

    first = intake_manual_job(
        store=store,
        source_store=source_store,
        url_or_text=text,
        company_hint="Example",
        title_hint="Agent Intern",
        location_hint="Shanghai",
        source_url="https://jobs.example.com/manual/1",
    )
    second = intake_manual_job(
        store=store,
        source_store=source_store,
        url_or_text=text,
        company_hint="Example",
        title_hint="Agent Intern",
        location_hint="Shanghai",
        source_url="https://jobs.example.com/manual/1",
    )

    assert first.is_new is True
    assert second.is_new is False
    assert second.job_id == first.job_id
    evidence = source_store.get(first.source_evidence_id)
    assert evidence.text_content == text
    assert evidence.provenance == "user_provided"
    with store.connect() as conn:
        row = conn.execute(
            "SELECT raw_text, extras_json FROM jobs WHERE id = ?", (first.job_id,)
        ).fetchone()
    assert row[0] == text
    assert json.loads(row[1])["source_evidence_id"] == first.source_evidence_id


def test_manual_job_api_saves_without_scoring_or_generating_a_resume(tmp_path) -> None:
    store = Store(tmp_path / "manual-web.db")
    store.init_schema()
    app = create_app(
        settings=Settings(db_path=tmp_path / "manual-web.db"),
        store=store,
        master_source=None,
        skills=[],
        runtime=None,
    )
    client = TestClient(app)
    text = "API_MANUAL_JD_HEAD\n" + ("完整岗位正文。\n" * 80) + "API_MANUAL_JD_TAIL"

    response = client.post(
        "/api/jobs/manual",
        json={
            "url_or_text": text,
            "company_hint": "API Company",
            "title_hint": "API Role",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["apply_pack_url"] == f"/jobs/{payload['job_id']}/apply-pack"
    assert "score" not in payload
    assert "analysis" not in payload
    page = client.get(payload["apply_pack_url"])
    assert page.status_code == 200
    assert "API_MANUAL_JD_HEAD" in page.text
    assert "API_MANUAL_JD_TAIL" in page.text
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM resume_workspaces").fetchone()[0] == 0


def test_pipeline_manual_job_form_uses_the_unified_intake_api(tmp_path) -> None:
    store = Store(tmp_path / "manual-pipeline.db")
    store.init_schema()
    app = create_app(
        settings=Settings(db_path=tmp_path / "manual-pipeline.db"),
        store=store,
        master_source=None,
        skills=[],
        runtime=None,
    )

    page = TestClient(app).get("/pipeline")

    assert page.status_code == 200
    assert 'action="/api/jobs/manual"' in page.text
    assert 'fetch("/api/jobs/manual"' in page.text
    assert "/api/pipeline/jobs/manual" not in page.text
    assert "window.location.assign(data.apply_pack_url)" in page.text
