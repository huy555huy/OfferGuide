from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from offerguide.memory import Store
from offerguide.resume import MasterResumeSource
from offerguide.resume.workspace import (
    ArtifactIntegrityError,
    ResumeWorkspaceRepository,
    WorkspaceConflictError,
    WorkspaceNotFoundError,
    WorkspaceNotReadyError,
    WorkspaceSubmittedError,
)


def _file(path: Path, body: bytes) -> tuple[Path, str]:
    path.write_bytes(body)
    return path, hashlib.sha256(body).hexdigest()


def _application(tmp_path: Path) -> tuple[Store, ResumeWorkspaceRepository, int, int, str]:
    store = Store(tmp_path / "workspace.db")
    store.init_schema()
    repo = ResumeWorkspaceRepository(store)
    source, source_sha = _file(tmp_path / "master.pdf", b"master resume evidence")
    repo.save_master(
        source_path=source,
        source_sha256=source_sha,
        extracted_text="Candidate\nProject experience",
        semantic_document={
            "source_sha256": source_sha,
            "semantic_text": "Candidate\nProject experience",
            "confirmed_by_user": True,
        },
        confirmed=True,
    )
    with store.connect() as conn:
        job_id = int(
            conn.execute(
                "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
                "VALUES ('manual', 'Agent Intern', 'Example', 'Build an agent', 'job-1') "
                "RETURNING id"
            ).fetchone()[0]
        )
        application_id = int(
            conn.execute(
                "INSERT INTO applications(job_id, status) VALUES (?, 'considered') RETURNING id",
                (job_id,),
            ).fetchone()[0]
        )
    return store, repo, job_id, application_id, source_sha


def _job_snapshot(job_id: int) -> dict:
    return {
        "job_id": job_id,
        "company": "Example",
        "title": "Agent Intern",
        "raw_text": "Build an agent",
    }


def _context(
    repo: ResumeWorkspaceRepository,
    *,
    job_id: int,
    source_sha: str,
) -> dict:
    master = repo.get_master()
    assert master is not None
    return {
        "job": {
            "job_id": job_id,
            "company": "Example",
            "title": "Agent Intern",
            "jd_text": "Build an agent",
        },
        "master_source": {
            "source_path": master.source_path,
            "sha256": source_sha,
            "extracted_text": master.extracted_text,
        },
        "master_document": master.semantic_document,
        "project_facts": [],
        "feedback": [],
    }


def _resume_document() -> dict:
    return {
        "header": {
            "name": {"spans": [{"text": "Candidate"}]},
            "lines": [],
        },
        "sections": [
            {
                "title": {"spans": [{"text": "Projects"}]},
                "entries": [
                    {
                        "rows": [{"left": {"spans": [{"text": "Agent runtime"}]}}],
                        "blocks": [],
                    }
                ],
            }
        ],
    }


def _ready(
    tmp_path: Path,
    repo: ResumeWorkspaceRepository,
    *,
    job_id: int,
    application_id: int,
    source_sha: str,
):
    pdf, pdf_sha = _file(tmp_path / "resume.pdf", b"rendered resume")
    workspace = repo.save_draft(
        application_id,
        job_snapshot=_job_snapshot(job_id),
        master_source_sha256=source_sha,
        context=_context(repo, job_id=job_id, source_sha=source_sha),
        resume_document=_resume_document(),
        pdf_path=pdf,
        pdf_sha256=pdf_sha,
        apply_pack={"assistant": {"message": "Relevant application message"}},
    )
    return workspace, pdf


def test_master_and_one_draft_workspace_are_persisted(tmp_path: Path) -> None:
    store, repo, job_id, application_id, source_sha = _application(tmp_path)
    master = repo.get_master()
    assert master is not None
    assert master.source_sha256 == source_sha
    assert master.semantic_status == "confirmed"
    assert master.confirmed_at is not None

    first, pdf = _ready(
        tmp_path,
        repo,
        job_id=job_id,
        application_id=application_id,
        source_sha=source_sha,
    )
    retried = repo.save_draft(
        application_id,
        job_snapshot=first.job_snapshot,
        master_source_sha256=first.master_source_sha256,
        context=first.context,
        resume_document=first.resume_document,
        pdf_path=pdf,
        pdf_sha256=first.pdf_sha256 or "",
        apply_pack=first.apply_pack,
    )
    assert retried.id == first.id
    assert retried.resume_document == first.resume_document
    assert retried.apply_pack == first.apply_pack
    assert first.status == "draft"

    ready = first
    updated = repo.save_draft(
        application_id,
        job_snapshot=ready.job_snapshot,
        master_source_sha256=ready.master_source_sha256,
        context=ready.context,
        resume_document=ready.resume_document,
        pdf_path=pdf,
        pdf_sha256=ready.pdf_sha256 or "",
        apply_pack={"assistant": {"message": "I built a relevant runtime."}},
    )
    assert updated.id == first.id
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM resume_workspaces").fetchone()[0] == 1
        assert json.loads(
            conn.execute(
                "SELECT apply_pack_json FROM resume_workspaces WHERE application_id = ?",
                (application_id,),
            ).fetchone()[0]
        )["assistant"]["message"]
def test_master_and_context_hashes_must_match_their_recorded_source(tmp_path: Path) -> None:
    _, repo, job_id, application_id, source_sha = _application(tmp_path)
    master = repo.get_master()
    assert master is not None
    different_sha = "b" * 64

    with pytest.raises(WorkspaceConflictError, match="semantic document"):
        repo.save_master(
            source_path=master.source_path,
            source_sha256=source_sha,
            extracted_text=master.extracted_text,
            semantic_document={
                "source_sha256": different_sha,
                "semantic_text": master.extracted_text,
                "confirmed_by_user": True,
            },
            confirmed=True,
        )

    ready, pdf = _ready(
        tmp_path,
        repo,
        job_id=job_id,
        application_id=application_id,
        source_sha=source_sha,
    )
    mismatched_context = dict(ready.context)
    mismatched_context["master_source"] = {
        **ready.context["master_source"],
        "sha256": different_sha,
    }
    mismatched_context["master_document"] = {
        **ready.context["master_document"],
        "source_sha256": different_sha,
    }
    with pytest.raises(WorkspaceConflictError, match="workspace master source"):
        repo.save_draft(
            application_id,
            job_snapshot=ready.job_snapshot,
            master_source_sha256=ready.master_source_sha256,
            context=mismatched_context,
            resume_document=ready.resume_document,
            pdf_path=pdf,
            pdf_sha256=ready.pdf_sha256 or "",
            apply_pack=ready.apply_pack,
        )

    assert repo.get(application_id) == ready


def test_submit_freezes_workspace_and_lifecycle_in_one_transaction(tmp_path: Path) -> None:
    store, repo, job_id, application_id, source_sha = _application(tmp_path)
    draft, pdf = _ready(
        tmp_path,
        repo,
        job_id=job_id,
        application_id=application_id,
        source_sha=source_sha,
    )
    draft = repo.save_draft(
        application_id,
        job_snapshot=draft.job_snapshot,
        master_source_sha256=draft.master_source_sha256,
        context=draft.context,
        resume_document=draft.resume_document,
        pdf_path=pdf,
        pdf_sha256=draft.pdf_sha256 or "",
        apply_pack={"assistant": {"message": "Why this role"}},
    )

    submitted = repo.submit(application_id)
    assert submitted.id == draft.id
    assert submitted.status == "submitted"
    assert submitted.submitted_at is not None
    assert repo.submit(application_id) == submitted

    with store.connect() as conn:
        app = conn.execute(
            "SELECT status, applied_at FROM applications WHERE id = ?",
            (application_id,),
        ).fetchone()
        assert app[0] == "applied" and app[1] is not None
        events = conn.execute(
            "SELECT source, payload_json FROM application_events "
            "WHERE application_id = ? AND kind = 'submitted'",
            (application_id,),
        ).fetchall()
        assert len(events) == 1
        assert events[0][0] == "manual"
        assert json.loads(events[0][1]) == {"workspace_id": submitted.id}

    with pytest.raises(WorkspaceSubmittedError):
        repo.save_draft(
            application_id,
            job_snapshot=submitted.job_snapshot,
            master_source_sha256=submitted.master_source_sha256,
            context=submitted.context,
            resume_document=submitted.resume_document,
            pdf_path=pdf,
            pdf_sha256=submitted.pdf_sha256 or "",
            apply_pack={"assistant": {"message": "changed"}},
        )
    with (
        store.connect() as conn,
        pytest.raises(sqlite3.IntegrityError, match="submitted resume workspace is immutable"),
    ):
        conn.execute(
            "UPDATE resume_workspaces SET context_json = '{}' WHERE id = ?",
            (submitted.id,),
        )
    with (
        store.connect() as conn,
        pytest.raises(sqlite3.IntegrityError, match="submitted resume workspace is immutable"),
    ):
        conn.execute("DELETE FROM resume_workspaces WHERE id = ?", (submitted.id,))


def test_failed_submit_rolls_back_workspace_application_and_event(tmp_path: Path) -> None:
    store, repo, job_id, application_id, source_sha = _application(tmp_path)
    _ready(
        tmp_path,
        repo,
        job_id=job_id,
        application_id=application_id,
        source_sha=source_sha,
    )
    with store.connect() as conn:
        conn.executescript(
            """
            CREATE TRIGGER abort_workspace_submit_event
            BEFORE INSERT ON application_events
            FOR EACH ROW WHEN NEW.kind = 'submitted'
            BEGIN
                SELECT RAISE(ABORT, 'forced event failure');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced event failure"):
        repo.submit(application_id)

    workspace = repo.get(application_id)
    assert workspace is not None and workspace.status == "draft"
    with store.connect() as conn:
        assert conn.execute(
            "SELECT status, applied_at FROM applications WHERE id = ?",
            (application_id,),
        ).fetchone() == ("considered", None)
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM application_events WHERE application_id = ?",
                (application_id,),
            ).fetchone()[0]
            == 0
        )


def test_submit_requires_intact_rendered_pdf(tmp_path: Path) -> None:
    _, repo, job_id, application_id, source_sha = _application(tmp_path)
    with pytest.raises(WorkspaceNotFoundError):
        repo.submit(application_id)

    _, pdf = _ready(
        tmp_path,
        repo,
        job_id=job_id,
        application_id=application_id,
        source_sha=source_sha,
    )
    pdf.write_bytes(b"tampered after render")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        repo.submit(application_id)
    assert repo.get(application_id).status == "draft"  # type: ignore[union-attr]


def test_incomplete_application_package_never_creates_a_workspace(tmp_path: Path) -> None:
    store, repo, job_id, application_id, source_sha = _application(tmp_path)
    pdf, pdf_sha = _file(tmp_path / "resume.pdf", b"rendered resume")

    with pytest.raises(WorkspaceNotReadyError, match="application package"):
        repo.save_draft(
            application_id,
            job_snapshot=_job_snapshot(job_id),
            master_source_sha256=source_sha,
            context=_context(repo, job_id=job_id, source_sha=source_sha),
            resume_document=_resume_document(),
            pdf_path=pdf,
            pdf_sha256=pdf_sha,
            apply_pack={"assistant": {"_error": "model failed"}},
        )

    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM resume_workspaces").fetchone()[0] == 0


def test_database_rejects_two_workspaces_for_the_same_job(tmp_path: Path) -> None:
    store, repo, job_id, application_id, source_sha = _application(tmp_path)
    _ready(
        tmp_path,
        repo,
        job_id=job_id,
        application_id=application_id,
        source_sha=source_sha,
    )
    with store.connect() as conn:
        second_application = int(
            conn.execute(
                "INSERT INTO applications(job_id, status) VALUES (?, 'considered') RETURNING id",
                (job_id,),
            ).fetchone()[0]
        )
    second_pdf, second_sha = _file(tmp_path / "second.pdf", b"second resume")

    with pytest.raises(sqlite3.IntegrityError, match="only one resume workspace"):
        repo.save_draft(
            second_application,
            job_snapshot=_job_snapshot(job_id),
            master_source_sha256=source_sha,
            context=_context(repo, job_id=job_id, source_sha=source_sha),
            resume_document=_resume_document(),
            pdf_path=second_pdf,
            pdf_sha256=second_sha,
            apply_pack={"assistant": {"message": "second"}},
        )


def test_job_level_workspace_resolution_rejects_multiple_applications(
    tmp_path: Path,
) -> None:
    store, repo, job_id, application_id, _source_sha = _application(tmp_path)
    with store.connect() as conn:
        second_application = int(
            conn.execute(
                "INSERT INTO applications(job_id, status) "
                "VALUES (?, 'considered') RETURNING id",
                (job_id,),
            ).fetchone()[0]
        )

    with pytest.raises(WorkspaceConflictError, match="multiple application records"):
        repo.application_for_job(job_id, create=False)

    assert second_application != application_id


def test_schema_migration_removes_deleted_package_field_references(tmp_path: Path) -> None:
    store = Store(tmp_path / "migration.db")
    store.init_schema()
    with store.connect() as conn:
        job_id = int(
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) "
                "VALUES ('manual', 'JD', 'migration-job') RETURNING id"
            ).fetchone()[0]
        )
        application_id = int(
            conn.execute(
                "INSERT INTO applications(job_id, status) VALUES (?, 'considered') RETURNING id",
                (job_id,),
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO resume_workspaces("
            "application_id, job_snapshot_json, master_source_sha256, apply_pack_json"
            ") VALUES (?, ?, ?, ?)",
            (
                application_id,
                json.dumps({"job_id": job_id}),
                "a" * 64,
                json.dumps(
                    {
                        "assistant": {
                            "message": "Current application message",
                            "form_answers": [],
                            "pre_submit_checks": [
                                "Confirm the attached PDF.",
                                "Copy from self_intro_snippet.",
                            ],
                        }
                    }
                ),
            ),
        )

    store.init_schema()

    with store.connect() as conn:
        raw = conn.execute(
            "SELECT apply_pack_json FROM resume_workspaces WHERE application_id = ?",
            (application_id,),
        ).fetchone()[0]
    assistant = json.loads(raw)["assistant"]
    assert assistant["pre_submit_checks"] == ["Confirm the attached PDF."]


def test_effective_master_text_prefers_user_confirmed_correction(tmp_path: Path) -> None:
    store = Store(tmp_path / "master-text.db")
    store.init_schema()
    repo = ResumeWorkspaceRepository(store)
    source_path, source_sha = _file(tmp_path / "master.pdf", b"master evidence")
    source = MasterResumeSource(
        source_path=str(source_path),
        sha256=source_sha,
        extracted_text="PDF extraction with a typo",
    )

    assert repo.effective_master_text(source) == "PDF extraction with a typo"

    repo.save_master(
        source_path=source_path,
        source_sha256=source_sha,
        extracted_text=source.extracted_text,
        semantic_document={
            "source_sha256": source_sha,
            "semantic_text": "User-confirmed corrected text",
            "confirmed_by_user": True,
        },
        confirmed=True,
    )

    assert repo.effective_master_text(source) == "User-confirmed corrected text"
