from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from offerguide.memory import Store
from offerguide.resume import (
    ArtifactFile,
    MasterResumeSource,
    ResumeBlock,
    ResumeDocument,
    ResumeEditorResult,
    ResumeEntry,
    ResumeEntryRow,
    ResumeHeader,
    ResumePage,
    ResumeRenderError,
    ResumeRenderResult,
    ResumeSection,
    ResumeWorkflow,
    ResumeWorkspaceRepository,
    RichText,
    RichTextSpan,
    WorkspaceConflictError,
    WorkspaceNotReadyError,
    WorkspaceSubmittedError,
    document_text,
)
from offerguide.resume import workflow as workflow_module


def _text(value: str) -> RichText:
    return RichText(spans=[RichTextSpan(text=value)])


def _document(label: str) -> ResumeDocument:
    return ResumeDocument(
        header=ResumeHeader(name=_text("Candidate")),
        sections=[
            ResumeSection(
                title=_text("Projects"),
                entries=[
                    ResumeEntry(
                        rows=[ResumeEntryRow(left=_text(label))],
                        blocks=[ResumeBlock(kind="bullet", content=_text(f"{label} evidence"))],
                    )
                ],
            )
        ],
    )


def _source(tmp_path: Path, name: str, extracted_text: str) -> MasterResumeSource:
    path = tmp_path / name
    payload = f"%PDF-1.4\n{name}\n".encode()
    path.write_bytes(payload)
    return MasterResumeSource(
        source_path=str(path.resolve()),
        sha256=hashlib.sha256(payload).hexdigest(),
        extracted_text=extracted_text,
    )


def _store_with_job(tmp_path: Path) -> tuple[Store, int]:
    store = Store(tmp_path / "workflow.db")
    store.init_schema()
    with store.connect() as conn:
        job_id = int(
            conn.execute(
                "INSERT INTO jobs(source, title, company, url, raw_text, content_hash) "
                "VALUES ('manual', 'Agent Intern', 'Example', "
                "'https://example.test/job', 'Complete JD', 'workflow-job') RETURNING id"
            ).fetchone()[0]
        )
    return store, job_id


def _application_id(store: Store, job_id: int) -> int:
    with store.connect() as conn:
        return int(
            conn.execute(
                "SELECT id FROM applications WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
        )


class _Editor:
    def __init__(self, document: ResumeDocument) -> None:
        self.document = document
        self.calls: list[Any] = []
        self.fail = False

    def edit(self, context: Any, *, user_feedback: str | None = None) -> ResumeEditorResult:
        self.calls.append((context, user_feedback))
        if self.fail:
            raise RuntimeError("forced editor failure")
        return ResumeEditorResult(document=self.document, editor_note="edited")


class _VisualEditor:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.fail = False
        self.documents: list[ResumeDocument] = []

    def review_pages(self, **kwargs: Any) -> ResumeEditorResult:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("forced visual review failure")
        current = kwargs["current"]
        if self.documents:
            return current.model_copy(update={"document": self.documents.pop(0)})
        return current


class _ApplyPackWriter:
    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.fail = False

    def __call__(self, job: Any, context: Any, result: ResumeEditorResult) -> dict[str, Any]:
        self.calls.append((job, context, result))
        if self.fail:
            raise RuntimeError("forced apply-pack failure")
        return {
            "message": document_text(result.document),
            "form_answers": [],
            "pre_submit_checks": [],
        }


class _Renderer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail = False
        self.template_revision = 1

    def __call__(
        self,
        *,
        document: ResumeDocument,
        company: str,
        role: str,
        output_dir: str | Path,
        portrait_source_pdf: str | Path | None,
    ) -> ResumeRenderResult:
        self.calls.append(
            {
                "document": document,
                "company": company,
                "role": role,
                "portrait_source_pdf": str(portrait_source_pdf),
            }
        )
        if self.fail:
            raise ResumeRenderError("forced render failure")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        document_sha = hashlib.sha256(
            document.model_dump_json(exclude_none=True).encode()
        ).hexdigest()
        template_sha = hashlib.sha256(str(self.template_revision).encode()).hexdigest()
        render_key = hashlib.sha256(f"{document_sha}:{template_sha}".encode()).hexdigest()
        pdf_path = destination / f"resume-{render_key[:12]}.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            + f"template:{self.template_revision}\n".encode()
            + document_text(document).encode()
            + b"\n%%EOF\n"
        )
        page_path = destination / f"page-{render_key[:12]}.png"
        page_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 128)
        pdf = ArtifactFile(
            path=pdf_path,
            sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            size_bytes=pdf_path.stat().st_size,
        )
        image = ArtifactFile(
            path=page_path,
            sha256=hashlib.sha256(page_path.read_bytes()).hexdigest(),
            size_bytes=page_path.stat().st_size,
        )
        return ResumeRenderResult(
            generated_on=date(2026, 7, 13),
            document_sha256=document_sha,
            template_sha256=template_sha,
            portrait_sha256=None,
            render_key=render_key,
            visible_characters=len(document_text(document)),
            pdf=pdf,
            pages=(ResumePage(page_number=1, image=image, extracted_text="rendered"),),
        )


def _workflow(
    *,
    store: Store,
    source: MasterResumeSource,
    editor: _Editor,
    renderer: _Renderer,
    tmp_path: Path,
    visual_editor: _VisualEditor | None = None,
    apply_pack_writer: _ApplyPackWriter | None = None,
    monkeypatch: pytest.MonkeyPatch,
) -> ResumeWorkflow:
    monkeypatch.setattr(workflow_module, "render_resume", renderer)
    ResumeWorkspaceRepository(store).save_master(
        source_path=source.source_path,
        source_sha256=source.sha256,
        extracted_text=source.extracted_text,
        semantic_document={
            "source_sha256": source.sha256,
            "semantic_text": source.extracted_text,
            "confirmed_by_user": True,
        },
        confirmed=True,
    )
    return ResumeWorkflow(
        store=store,
        master_source=source,
        editor=editor,  # type: ignore[arg-type]
        visual_editor=visual_editor,  # type: ignore[arg-type]
        apply_pack_writer=apply_pack_writer or _ApplyPackWriter(),
        artifact_root=tmp_path / "artifacts",
    )


def test_rerender_never_calls_content_or_visual_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _store_with_job(tmp_path)
    source = _source(tmp_path, "master.pdf", "Candidate master evidence")
    editor = _Editor(_document("original"))
    visual = _VisualEditor()
    writer = _ApplyPackWriter()
    renderer = _Renderer()
    workflow = _workflow(
        store=store,
        source=source,
        editor=editor,
        visual_editor=visual,
        apply_pack_writer=writer,
        renderer=renderer,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    initial = workflow.start(job_id).workspace
    initial_document = initial.resume_document
    initial_context = initial.context
    initial_assistant = initial.apply_pack["assistant"]
    assert len(editor.calls) == len(visual.calls) == len(writer.calls) == 1

    renderer.template_revision = 2
    rerendered = workflow.rerender(job_id)

    assert len(editor.calls) == len(visual.calls) == len(writer.calls) == 1
    assert rerendered.workspace.resume_document == initial_document
    assert rerendered.workspace.context == initial_context
    assert rerendered.workspace.apply_pack["assistant"] == initial_assistant
    assert rerendered.visual_review_status == "not_run_after_rerender"
    assert rerendered.workspace.pdf_sha256 != initial.pdf_sha256


@pytest.mark.parametrize("failing_stage", ["editor", "render", "visual", "apply_pack"])
def test_failed_revision_preserves_the_last_complete_draft(
    failing_stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _store_with_job(tmp_path)
    source = _source(tmp_path, "master.pdf", "Candidate master evidence")
    editor = _Editor(_document("original"))
    visual = _VisualEditor()
    writer = _ApplyPackWriter()
    renderer = _Renderer()
    workflow = _workflow(
        store=store,
        source=source,
        editor=editor,
        visual_editor=visual,
        apply_pack_writer=writer,
        renderer=renderer,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    workflow.start(job_id)
    application_id = _application_id(store, job_id)
    repo = ResumeWorkspaceRepository(store)
    before = repo.get(application_id)
    assert before is not None

    editor.document = _document("uncommitted revision")
    failure_target = {
        "editor": editor,
        "render": renderer,
        "visual": visual,
        "apply_pack": writer,
    }[failing_stage]
    failure_target.fail = True
    with pytest.raises((RuntimeError, ResumeRenderError)):
        workflow.revise(job_id, feedback="make this revision")

    assert repo.get(application_id) == before
    assert Path(before.pdf_path or "").read_bytes()


def test_revision_keeps_the_workspace_master_even_if_the_global_master_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _store_with_job(tmp_path)
    original_source = _source(tmp_path, "original.pdf", "Original master evidence")
    renderer = _Renderer()
    first_editor = _Editor(_document("original"))
    first_workflow = _workflow(
        store=store,
        source=original_source,
        editor=first_editor,
        renderer=renderer,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    first_workflow.start(job_id)

    replacement_source = _source(tmp_path, "replacement.pdf", "Replacement evidence")
    replacement_editor = _Editor(_document("replacement"))
    replacement_workflow = _workflow(
        store=store,
        source=replacement_source,
        editor=replacement_editor,
        renderer=renderer,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    ResumeWorkspaceRepository(store).save_master(
        source_path=replacement_source.source_path,
        source_sha256=replacement_source.sha256,
        extracted_text=replacement_source.extracted_text,
        semantic_document={
            "source_sha256": replacement_source.sha256,
            "semantic_text": replacement_source.extracted_text,
            "confirmed_by_user": True,
        },
        confirmed=True,
    )
    replacement_workflow.revise(job_id, feedback="revise the existing workspace")

    application_id = _application_id(store, job_id)
    saved = ResumeWorkspaceRepository(store).get(application_id)
    assert saved is not None
    assert saved.master_source_sha256 == original_source.sha256
    assert saved.context["master_source"]["sha256"] == original_source.sha256
    assert replacement_editor.calls[-1][0].master_source.sha256 == original_source.sha256
    assert renderer.calls[-1]["portrait_source_pdf"] == original_source.source_path


def test_submitted_workspace_rejects_rerender_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _store_with_job(tmp_path)
    source = _source(tmp_path, "master.pdf", "Candidate master evidence")
    editor = _Editor(_document("submitted"))
    renderer = _Renderer()
    workflow = _workflow(
        store=store,
        source=source,
        editor=editor,
        renderer=renderer,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    workflow.start(job_id)
    ResumeWorkspaceRepository(store).submit(_application_id(store, job_id))
    render_calls = len(renderer.calls)

    with pytest.raises(WorkspaceSubmittedError):
        workflow.rerender(job_id)

    assert len(renderer.calls) == render_calls


def test_rerender_rejects_a_master_pdf_that_no_longer_matches_its_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _store_with_job(tmp_path)
    source = _source(tmp_path, "master.pdf", "Candidate master evidence")
    editor = _Editor(_document("current"))
    renderer = _Renderer()
    workflow = _workflow(
        store=store,
        source=source,
        editor=editor,
        renderer=renderer,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    workflow.start(job_id)
    application_id = _application_id(store, job_id)
    repo = ResumeWorkspaceRepository(store)
    before = repo.get(application_id)
    assert before is not None
    Path(source.source_path).write_bytes(b"tampered master")
    render_calls = len(renderer.calls)

    with pytest.raises(WorkspaceConflictError, match="recorded hash"):
        workflow.rerender(job_id)

    assert len(renderer.calls) == render_calls
    assert repo.get(application_id) == before


def test_start_requires_an_explicitly_confirmed_master_and_leaves_no_skeleton(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _store_with_job(tmp_path)
    source = _source(tmp_path, "master.pdf", "Candidate master evidence")
    editor = _Editor(_document("unused"))
    renderer = _Renderer()
    monkeypatch.setattr(workflow_module, "render_resume", renderer)
    workflow = ResumeWorkflow(
        store=store,
        master_source=source,
        editor=editor,  # type: ignore[arg-type]
        apply_pack_writer=_ApplyPackWriter(),
        artifact_root=tmp_path / "artifacts",
    )

    with pytest.raises(WorkspaceNotReadyError, match="reviewed and confirmed"):
        workflow.start(job_id)

    assert editor.calls == []
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM resume_workspaces").fetchone()[0] == 0


def test_visual_review_approves_the_actual_final_render_and_prunes_intermediate_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _store_with_job(tmp_path)
    source = _source(tmp_path, "master.pdf", "Candidate master evidence")
    editor = _Editor(_document("initial"))
    visual = _VisualEditor()
    visual.documents = [_document("visual revision"), _document("visual revision")]
    renderer = _Renderer()
    workflow = _workflow(
        store=store,
        source=source,
        editor=editor,
        visual_editor=visual,
        renderer=renderer,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    result = workflow.start(job_id)

    assert len(visual.calls) == 2
    assert len(renderer.calls) == 2
    assert result.visual_review_status == "reviewed"
    assert result.workspace.apply_pack["visual_review"]["passes"] == 2
    assert result.workspace.resume_document == _document("visual revision").model_dump(mode="json")
    artifact_dir = tmp_path / "artifacts" / f"app_{result.workspace.application_id}"
    assert list(artifact_dir.glob("*.pdf")) == [Path(result.workspace.pdf_path or "")]
    assert len(list(artifact_dir.glob("*.png"))) == 1
