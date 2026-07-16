"""End-to-end orchestration for the one resume workspace attached to a job."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..job_text import format_job_for_skill
from ..memory import Store
from ..project_vault import list_all as list_project_records
from .context import build_resume_context
from .editor import ResumeEditor, ResumeEditorError, ResumeEditorResult, VisualReviewUnavailable
from .master import MasterResumeDocument, MasterResumeSource
from .models import (
    ApplicationPackage,
    ProjectFact,
    ResumeContext,
    ResumeDocument,
    ResumeJobContext,
)
from .render import ResumeRenderResult, document_text, render_resume
from .workspace import (
    ResumeWorkspace,
    ResumeWorkspaceRepository,
    WorkspaceConflictError,
    WorkspaceNotFoundError,
    WorkspaceNotReadyError,
    WorkspaceSubmittedError,
)

ApplyPackWriter = Callable[[dict[str, Any], ResumeContext, ResumeEditorResult], Mapping[str, Any]]
log = logging.getLogger(__name__)
_MAX_VISUAL_REVIEW_PASSES = 4


def generate_application_package(
    *,
    runtime: Any,
    skills: Sequence[Any],
    job_snapshot: dict[str, Any],
    context: ResumeContext,
    editor_result: ResumeEditorResult,
) -> dict[str, Any]:
    """Generate and validate the one application package used by every entry point."""
    if runtime is None:
        raise WorkspaceNotReadyError("投递包模型未配置")
    spec = next((item for item in skills if item.name == "apply_assistant"), None)
    if spec is None:
        raise WorkspaceNotReadyError("apply_assistant 未加载")
    try:
        result = runtime.invoke(
            spec,
            {
                "company": str(job_snapshot.get("company") or ""),
                "role_focus": str(job_snapshot.get("title") or ""),
                "job_text": format_job_for_skill(job_snapshot),
                "user_profile": json.dumps(
                    {
                        "final_resume": document_text(editor_result.document),
                        "master_resume": context.master_document.semantic_text,
                        "project_facts": [
                            item.model_dump(mode="json") for item in context.project_facts
                        ],
                        "preparation_notes": [
                            item.model_dump(mode="json")
                            for item in editor_result.preparation_notes
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
            consult_variant_registry=False,
            inject_long_term_memory=False,
        )
    except Exception as exc:
        raise WorkspaceNotReadyError(f"投递话术生成失败: {exc}") from exc
    if result.parsed is None:
        raise WorkspaceNotReadyError("投递话术模型没有返回可用 JSON")

    try:
        package = ApplicationPackage.model_validate(result.parsed)
    except Exception as exc:
        raise WorkspaceNotReadyError(f"投递话术结构不完整: {exc}") from exc
    return package.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class ResumeOperation:
    workspace: ResumeWorkspace
    render: ResumeRenderResult
    editor_result: ResumeEditorResult
    visual_review_status: str
    visual_review_error: str | None = None


class ResumeWorkflow:
    """Create and update one draft without exposing alternate resume paths."""

    def __init__(
        self,
        *,
        store: Store,
        master_source: MasterResumeSource,
        editor: ResumeEditor,
        artifact_root: str | Path,
        visual_editor: ResumeEditor | None = None,
        apply_pack_writer: ApplyPackWriter | None = None,
    ) -> None:
        self.store = store
        self.master_source = master_source
        self.editor = editor
        self.visual_editor = visual_editor
        self.apply_pack_writer = apply_pack_writer
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.workspaces = ResumeWorkspaceRepository(store)

    def start(
        self,
        job_id: int,
        *,
        job_snapshot: Mapping[str, Any] | None = None,
    ) -> ResumeOperation:
        """Explicitly create and edit a workspace; GET handlers must not call this."""
        application_id, application_created = self.workspaces.application_for_job(
            job_id, create=True
        )
        assert application_id is not None
        existing = self.workspaces.get(application_id)
        if existing is not None:
            if existing.is_submitted:
                raise WorkspaceSubmittedError("the submitted resume is immutable")
            if existing.resume_document and existing.pdf_path:
                return self._operation_from_workspace(existing)
            context = _context_from_workspace(existing)
            return self._edit_and_render(
                application_id=application_id,
                job_snapshot=existing.job_snapshot,
                context=context,
            )

        try:
            master_document = self._ensure_master()
            job_snapshot = (
                self._job_snapshot(job_id)
                if job_snapshot is None
                else _normalize_supplied_job_snapshot(job_id, job_snapshot)
            )
            context = self._build_context(
                job_snapshot=job_snapshot,
                master_document=master_document,
            )
            return self._edit_and_render(
                application_id=application_id,
                job_snapshot=job_snapshot,
                context=context,
            )
        except Exception:
            if application_created:
                self._delete_empty_application(application_id)
            raise

    def revise(self, job_id: int, *, feedback: str) -> ResumeOperation:
        feedback = str(feedback or "").strip()
        if not feedback:
            raise ValueError("resume feedback must not be blank")
        application_id, _ = self.workspaces.application_for_job(job_id, create=False)
        if application_id is None:
            raise WorkspaceNotFoundError(f"job {job_id} has no application")
        workspace = self.workspaces.get(application_id)
        if workspace is None:
            raise WorkspaceNotFoundError(f"job {job_id} has no resume workspace")
        if workspace.is_submitted:
            raise WorkspaceSubmittedError("the submitted resume is immutable")
        if not workspace.resume_document:
            raise WorkspaceNotFoundError(f"job {job_id} has no resume document")
        current = ResumeDocument.model_validate(workspace.resume_document)
        previous_context = _context_from_workspace(workspace)
        context = self._build_context(
            job_snapshot=workspace.job_snapshot,
            master_source=previous_context.master_source,
            master_document=previous_context.master_document,
            feedback=[*previous_context.feedback, feedback],
            current_resume=current,
        )
        return self._edit_and_render(
            application_id=application_id,
            job_snapshot=workspace.job_snapshot,
            context=context,
            user_feedback=feedback,
        )

    def rerender(self, job_id: int) -> ResumeOperation:
        """Apply the current template without calling an editor or changing content."""
        application_id, _ = self.workspaces.application_for_job(job_id, create=False)
        if application_id is None:
            raise WorkspaceNotFoundError(f"job {job_id} has no application")
        workspace = self.workspaces.get(application_id)
        if workspace is None or not workspace.resume_document:
            raise WorkspaceNotFoundError(f"job {job_id} has no resume document")
        if workspace.is_submitted:
            raise WorkspaceSubmittedError("the submitted resume is immutable")
        context = _context_from_workspace(workspace)
        document = ResumeDocument.model_validate(workspace.resume_document)
        previous = _editor_result_from_workspace(workspace, document)
        try:
            rendered = self._render(
                application_id,
                workspace.job_snapshot,
                document,
                master_source=context.master_source,
            )
            apply_pack = dict(workspace.apply_pack)
            apply_pack["render"] = rendered.as_dict()
            apply_pack["visual_review"] = {
                "status": "not_run_after_rerender",
                "error": None,
                "passes": 0,
            }
            workspace = self.workspaces.save_draft(
                application_id,
                job_snapshot=workspace.job_snapshot,
                master_source_sha256=workspace.master_source_sha256,
                context=context.model_dump(mode="json", exclude_none=True),
                resume_document=document.model_dump(mode="json"),
                pdf_path=rendered.pdf.path,
                pdf_sha256=rendered.pdf.sha256,
                apply_pack=apply_pack,
            )
        except Exception:
            self._prune_workspace_artifacts(application_id, self.workspaces.get(application_id))
            raise
        self._prune_workspace_artifacts(application_id, workspace)
        return ResumeOperation(
            workspace=workspace,
            render=rendered,
            editor_result=previous,
            visual_review_status="not_run_after_rerender",
        )

    def get_for_job(self, job_id: int) -> ResumeWorkspace | None:
        application_id, _ = self.workspaces.application_for_job(job_id, create=False)
        return self.workspaces.get(application_id) if application_id is not None else None

    def _edit_and_render(
        self,
        *,
        application_id: int,
        job_snapshot: dict[str, Any],
        context: ResumeContext,
        user_feedback: str | None = None,
    ) -> ResumeOperation:
        initial = self.editor.edit(context, user_feedback=user_feedback)
        return self._render_review_and_persist(
            application_id=application_id,
            job_snapshot=job_snapshot,
            context=context,
            initial=initial,
            user_feedback=user_feedback,
        )

    def _render_review_and_persist(
        self,
        *,
        application_id: int,
        job_snapshot: dict[str, Any],
        context: ResumeContext,
        initial: ResumeEditorResult,
        user_feedback: str | None = None,
    ) -> ResumeOperation:
        try:
            rendered = self._render(
                application_id,
                job_snapshot,
                initial.document,
                master_source=context.master_source,
            )
            final = initial
            visual_status = "not_configured"
            visual_error: str | None = None
            visual_passes = 0
            if self.visual_editor is not None:
                review_context = context.model_copy(update={"current_resume": None})
                for pass_number in range(1, _MAX_VISUAL_REVIEW_PASSES + 1):
                    try:
                        reviewed = self.visual_editor.review_pages(
                            context=review_context,
                            current=final,
                            render=rendered,
                            user_feedback=user_feedback,
                        )
                    except VisualReviewUnavailable as exc:
                        if pass_number > 1:
                            raise ResumeEditorError(
                                "visual review became unavailable before the final PDF was approved"
                            ) from exc
                        visual_status = "unavailable"
                        visual_error = str(exc)
                        break
                    visual_passes = pass_number
                    previous_document = final.document
                    final = reviewed
                    if final.document == previous_document:
                        visual_status = "reviewed"
                        break
                    rendered = self._render(
                        application_id,
                        job_snapshot,
                        final.document,
                        master_source=context.master_source,
                    )
                else:
                    raise ResumeEditorError(
                        "visual review did not approve a stable final PDF after "
                        f"{_MAX_VISUAL_REVIEW_PASSES} passes"
                    )

            apply_pack = self._write_apply_pack(job_snapshot, context, final)
            workspace = self.workspaces.save_draft(
                application_id,
                job_snapshot=job_snapshot,
                master_source_sha256=context.master_source.sha256,
                context=context.model_dump(mode="json", exclude_none=True),
                resume_document=final.document.model_dump(mode="json"),
                pdf_path=rendered.pdf.path,
                pdf_sha256=rendered.pdf.sha256,
                apply_pack={
                    "assistant": apply_pack,
                    "editor_note": final.editor_note,
                    "preparation_notes": [
                        item.model_dump(mode="json") for item in final.preparation_notes
                    ],
                    "render": rendered.as_dict(),
                    "visual_review": {
                        "status": visual_status,
                        "error": visual_error,
                        "passes": visual_passes,
                    },
                },
            )
        except Exception:
            self._prune_workspace_artifacts(application_id, self.workspaces.get(application_id))
            raise
        self._prune_workspace_artifacts(application_id, workspace)
        return ResumeOperation(
            workspace=workspace,
            render=rendered,
            editor_result=final,
            visual_review_status=visual_status,
            visual_review_error=visual_error,
        )

    def _render(
        self,
        application_id: int,
        job_snapshot: Mapping[str, Any],
        document: ResumeDocument,
        *,
        master_source: MasterResumeSource,
    ) -> ResumeRenderResult:
        source_path = _verified_master_source_path(master_source)
        return render_resume(
            document=document,
            company=str(job_snapshot.get("company") or "目标公司"),
            role=str(job_snapshot.get("title") or "目标岗位"),
            output_dir=self.artifact_root / f"app_{application_id}",
            portrait_source_pdf=source_path,
        )

    def _write_apply_pack(
        self,
        job_snapshot: dict[str, Any],
        context: ResumeContext,
        editor_result: ResumeEditorResult,
    ) -> dict[str, Any]:
        if self.apply_pack_writer is None:
            raise WorkspaceNotReadyError("application package writer is not configured")
        result = dict(self.apply_pack_writer(job_snapshot, context, editor_result))
        if not result or result.get("_error"):
            raise WorkspaceNotReadyError(
                str(result.get("_error") or "application package writer returned no package")
            )
        return result

    def _ensure_master(self) -> MasterResumeDocument:
        stored = self.workspaces.get_master()
        if stored is None:
            raise WorkspaceNotReadyError(
                "master resume text has not been reviewed and confirmed by the user"
            )
        if stored.source_sha256 != self.master_source.sha256:
            raise WorkspaceConflictError(
                "the configured master PDF changed and must be reviewed again"
            )
        if stored.semantic_status != "confirmed":
            raise WorkspaceNotReadyError(
                "master resume text has not been reviewed and confirmed by the user"
            )
        semantic = MasterResumeDocument.model_validate(stored.semantic_document)
        if not semantic.confirmed_by_user:
            raise WorkspaceNotReadyError("master resume confirmation is invalid")
        return semantic

    def _build_context(
        self,
        *,
        job_snapshot: Mapping[str, Any],
        master_source: MasterResumeSource | None = None,
        master_document: MasterResumeDocument,
        feedback: Sequence[str] = (),
        current_resume: ResumeDocument | None = None,
    ) -> ResumeContext:
        return build_resume_context(
            job=_job_context(job_snapshot),
            master_source=master_source or self.master_source,
            master_document=master_document,
            project_facts=_project_facts(self.store),
            feedback=feedback,
            current_resume=current_resume,
        )

    def _delete_empty_application(self, application_id: int) -> None:
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM applications WHERE id = ? "
                "AND NOT EXISTS (SELECT 1 FROM resume_workspaces WHERE application_id = ?) "
                "AND NOT EXISTS (SELECT 1 FROM application_events WHERE application_id = ?)",
                (application_id, application_id, application_id),
            )
        path = self.artifact_root / f"app_{application_id}"
        if path.is_dir():
            shutil.rmtree(path)

    def _prune_workspace_artifacts(
        self,
        application_id: int,
        workspace: ResumeWorkspace | None,
    ) -> None:
        """Keep only files referenced by the current draft or frozen submission."""
        root = (self.artifact_root / f"app_{application_id}").resolve()
        if not root.is_dir() or self.artifact_root not in root.parents:
            return
        keep_files: set[Path] = set()
        keep_page_dirs: set[Path] = set()
        if workspace is not None and workspace.pdf_path:
            keep_files.add(Path(workspace.pdf_path).expanduser().resolve())
        if workspace is not None:
            render = workspace.apply_pack.get("render")
            if isinstance(render, Mapping):
                pages = render.get("pages")
                if isinstance(pages, list):
                    for page in pages:
                        if not isinstance(page, Mapping):
                            continue
                        image = page.get("image")
                        if isinstance(image, Mapping) and image.get("path"):
                            image_path = Path(str(image["path"])).expanduser().resolve()
                            keep_files.add(image_path)
                            keep_page_dirs.add(image_path.parent)
        try:
            for artifact in root.iterdir():
                if (
                    artifact.is_file()
                    and artifact.suffix.casefold() in {".pdf", ".png", ".jpg", ".jpeg"}
                    and artifact.resolve() not in keep_files
                ):
                    artifact.unlink()
            pages_root = root / "pages"
            if pages_root.is_dir():
                for page_dir in pages_root.iterdir():
                    if page_dir.is_dir() and page_dir.resolve() not in keep_page_dirs:
                        shutil.rmtree(page_dir)
                if not any(pages_root.iterdir()):
                    pages_root.rmdir()
            if workspace is None and not any(root.iterdir()):
                root.rmdir()
        except OSError as exc:
            log.warning("resume artifact cleanup failed for application %s: %s", application_id, exc)

    def _job_snapshot(self, job_id: int) -> dict[str, Any]:
        with self.store.connect() as conn:
            conn.row_factory = __import__("sqlite3").Row
            row = conn.execute(
                "SELECT id, source, source_id, url, title, company, location, "
                "raw_text, extras_json, content_hash, fetched_at, created_at "
                "FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise WorkspaceNotFoundError(f"job {job_id} not found")
        try:
            extras = json.loads(row["extras_json"] or "{}")
        except json.JSONDecodeError:
            extras = {"unparsed_extras": str(row["extras_json"] or "")}
        return {
            "job_id": int(row["id"]),
            "source": str(row["source"] or ""),
            "source_id": row["source_id"],
            "url": str(row["url"] or ""),
            "title": str(row["title"] or ""),
            "company": str(row["company"] or ""),
            "location": str(row["location"] or ""),
            "raw_text": str(row["raw_text"] or ""),
            "extras": extras if isinstance(extras, dict) else {"value": extras},
            "content_hash": str(row["content_hash"] or ""),
            "fetched_at": row["fetched_at"],
            "created_at": row["created_at"],
        }

    def _operation_from_workspace(self, workspace: ResumeWorkspace) -> ResumeOperation:
        render_data = workspace.apply_pack.get("render")
        if not isinstance(render_data, dict):
            raise WorkspaceNotFoundError("workspace has no render metadata")
        _context_from_workspace(workspace)
        document = ResumeDocument.model_validate(workspace.resume_document)
        editor_result = _editor_result_from_workspace(workspace, document)
        render = _render_result_from_dict(render_data)
        visual = workspace.apply_pack.get("visual_review")
        visual = visual if isinstance(visual, dict) else {}
        return ResumeOperation(
            workspace=workspace,
            render=render,
            editor_result=editor_result,
            visual_review_status=str(visual.get("status") or "unknown"),
            visual_review_error=str(visual.get("error")) if visual.get("error") else None,
        )


def _job_context(snapshot: Mapping[str, Any]) -> ResumeJobContext:
    extras = snapshot.get("extras")
    verified: list[str] = []
    if isinstance(extras, Mapping):
        verified.append(json.dumps(dict(extras), ensure_ascii=False, sort_keys=True))
    return ResumeJobContext(
        job_id=int(snapshot.get("job_id") or snapshot.get("id") or 0),
        company=str(snapshot.get("company") or ""),
        title=str(snapshot.get("title") or ""),
        jd_text=str(snapshot.get("raw_text") or ""),
        source_url=str(snapshot.get("url") or "") or None,
        verified_information=verified,
    )


def job_snapshot_from_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one published job-evidence snapshot into the resume input contract."""
    raw = dict(evidence)
    raw_job_id = raw.get("job_id")
    if raw_job_id is None or isinstance(raw_job_id, bool):
        raise WorkspaceConflictError("published job evidence has no valid job id")
    try:
        job_id = int(raw_job_id)
    except (TypeError, ValueError) as exc:
        raise WorkspaceConflictError("published job evidence has no valid job id") from exc
    snapshot = {
        "job_id": job_id,
        "source": str(raw.get("source_name") or ""),
        "source_id": raw.get("source_job_id"),
        "url": str(raw.get("canonical_url") or ""),
        "title": str(raw.get("title") or ""),
        "company": str(raw.get("company") or ""),
        "location": str(raw.get("location") or ""),
        "raw_text": str(raw.get("jd_text") or ""),
        "extras": {
            "source_evidence_id": raw.get("source_evidence_id"),
            "recruitment_type": raw.get("recruitment_type"),
            "page_time_information": list(raw.get("page_time_information") or []),
            "source_status": raw.get("source_status") or "unknown",
        },
        "content_hash": str(raw.get("content_sha256") or ""),
        "fetched_at": raw.get("checked_at"),
        "created_at": raw.get("last_seen_at"),
        "job_evidence": raw,
    }
    return _normalize_supplied_job_snapshot(job_id, snapshot)


def _normalize_supplied_job_snapshot(
    job_id: int,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an immutable upstream snapshot before the resume workspace owns it."""
    value = dict(snapshot)
    snapshot_job_id = value.get("job_id", value.get("id"))
    if snapshot_job_id is None or isinstance(snapshot_job_id, bool):
        raise WorkspaceConflictError("job snapshot has no valid job id")
    try:
        resolved_job_id = int(snapshot_job_id)
    except (TypeError, ValueError) as exc:
        raise WorkspaceConflictError("job snapshot has no valid job id") from exc
    if resolved_job_id != job_id:
        raise WorkspaceConflictError("job snapshot does not match the selected job")

    value["job_id"] = resolved_job_id
    value["raw_text"] = str(value.get("raw_text") or value.get("jd_text") or "")
    value["url"] = str(value.get("url") or value.get("canonical_url") or "")
    value["source"] = str(value.get("source") or value.get("source_name") or "")
    value["source_id"] = value.get("source_id", value.get("source_job_id"))
    if not value["raw_text"].strip():
        raise WorkspaceConflictError("job snapshot has no complete JD")
    return value


def _context_from_workspace(workspace: ResumeWorkspace) -> ResumeContext:
    try:
        context = ResumeContext.model_validate(workspace.context)
    except Exception as exc:
        raise WorkspaceConflictError("resume workspace contains an invalid context") from exc
    if context.master_source.sha256 != workspace.master_source_sha256:
        raise WorkspaceConflictError("resume workspace context does not match its master source")
    snapshot_job_id = workspace.job_snapshot.get("job_id", workspace.job_snapshot.get("id"))
    if snapshot_job_id is None or str(context.job.job_id) != str(snapshot_job_id):
        raise WorkspaceConflictError("resume workspace context does not match its job snapshot")
    return context


def _verified_master_source_path(source: MasterResumeSource) -> Path:
    path = Path(source.source_path).expanduser().resolve()
    if not path.is_file():
        raise WorkspaceConflictError(f"master resume PDF is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != source.sha256:
        raise WorkspaceConflictError("master resume PDF no longer matches its recorded hash")
    return path


def _project_facts(store: Store) -> list[ProjectFact]:
    facts: list[ProjectFact] = []
    with store.connect() as conn:
        record_count = int(conn.execute("SELECT COUNT(*) FROM project_records").fetchone()[0])
    for record in list_project_records(store, limit=max(1, record_count)):
        fields = (
            ("项目方向", record.mainstream_direction),
            ("典型问题", record.typical_problem),
            ("项目任务", record.project_task),
            ("我的真实工作", record.my_work),
            ("方法路线", record.method_route),
            ("贡献类型", record.contribution_label),
            ("贡献细节", record.contribution_detail),
            ("关键困难", record.key_difficulties),
            ("解决过程", record.resolution_process),
            ("项目产出", record.project_outputs),
            ("证据", record.evidence),
            ("可追问点", record.askable_points),
            ("表达边界", record.expression_boundary),
        )
        facts.append(
            ProjectFact(
                project_id=record.id,
                title=record.title,
                facts=[f"{label}: {value}" for label, value in fields if value],
                do_not_claim=[record.do_not_claim] if record.do_not_claim else [],
            )
        )
    return facts


def _editor_result_from_workspace(
    workspace: ResumeWorkspace,
    document: ResumeDocument,
) -> ResumeEditorResult:
    return ResumeEditorResult.model_validate(
        {
            "document": document.model_dump(mode="json"),
            "editor_note": workspace.apply_pack.get("editor_note") or "沿用当前简历内容。",
            "preparation_notes": workspace.apply_pack.get("preparation_notes") or [],
        }
    )


def _render_result_from_dict(value: Mapping[str, Any]) -> ResumeRenderResult:
    from datetime import date

    from .render import ArtifactFile, ResumePage

    pdf = value.get("pdf")
    pages = value.get("pages")
    if not isinstance(pdf, Mapping) or not isinstance(pages, list):
        raise WorkspaceNotFoundError("workspace render metadata is incomplete")
    return ResumeRenderResult(
        generated_on=date.fromisoformat(str(value["generated_on"])),
        document_sha256=str(value["document_sha256"]),
        template_sha256=str(value["template_sha256"]),
        portrait_sha256=(str(value["portrait_sha256"]) if value.get("portrait_sha256") else None),
        render_key=str(value["render_key"]),
        visible_characters=int(value["visible_characters"]),
        pdf=ArtifactFile(
            path=Path(str(pdf["path"])),
            sha256=str(pdf["sha256"]),
            size_bytes=int(pdf["size_bytes"]),
        ),
        pages=tuple(
            ResumePage(
                page_number=int(page["page_number"]),
                image=ArtifactFile(
                    path=Path(str(page["image"]["path"])),
                    sha256=str(page["image"]["sha256"]),
                    size_bytes=int(page["image"]["size_bytes"]),
                ),
                extracted_text=str(page.get("extracted_text") or ""),
            )
            for page in pages
            if isinstance(page, Mapping) and isinstance(page.get("image"), Mapping)
        ),
    )


__all__ = [
    "ApplyPackWriter",
    "ResumeOperation",
    "ResumeWorkflow",
    "document_text",
    "generate_application_package",
]
