"""The one application path: explicit workspace -> feedback -> submit -> prep."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from offerguide.agent_runtime import _schema as runtime_schema
from offerguide.agent_runtime.memory import MemoryStore
from offerguide.agent_runtime.tools import (
    AgentRuntimeDeps,
    _exec_prepare_application,
    _exec_read_artifact,
    _exec_record_event,
    _exec_rerender_application,
    _exec_research_interview,
    _exec_revise_application,
)
from offerguide.config import Settings
from offerguide.interview_research import (
    EvidenceDocument,
    GroundingReference,
    InterviewAnswerSet,
    InterviewQuestionAnswer,
    InterviewResearchAgent,
    InterviewResearchRepository,
    InterviewSourceAssessment,
    PublicInterviewSourceProvider,
    SourceCitation,
)
from offerguide.llm import LLMResponse
from offerguide.memory import Store
from offerguide.research_agents import SourceEvidenceStore, SourceScope
from offerguide.research_agents.job_discovery import (
    CandidateEvidence,
    JobDiscoveryRepository,
    JobPostingEvidence,
    JobSelectionItem,
    SelectionGroundingQuote,
)
from offerguide.resume import MasterResumeSource, ResumeEditor, ResumeWorkspaceRepository
from offerguide.skills import discover_skills
from offerguide.ui.web import create_app


def _document() -> dict:
    def text(value: str, emphasis: bool = False) -> dict:
        return {"spans": [{"text": value, "emphasis": emphasis}]}

    return {
        "header": {
            "name": text("Candidate"),
            "lines": [text("candidate@example.com")],
        },
        "sections": [
            {
                "title": text("Experience"),
                "entries": [
                    {
                        "rows": [{"left": text("Agent Runtime", True), "right": text("2026")}],
                        "blocks": [
                            {
                                "kind": "bullet",
                                "content": text("SUBMITTED_RESUME_TOKEN", True),
                            }
                        ],
                    }
                ],
            }
        ],
    }


class StubLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def chat(self, messages, **_kwargs):
        self.calls.append(messages)
        payload = {
            "document": _document(),
            "editor_note": f"edit {len(self.calls)}",
            "preparation_notes": [],
        }
        return LLMResponse(content=json.dumps(payload), model="stub")


def _client(
    tmp_path,
    master_resume_source_factory,
    *,
    runtime=None,
    skills=None,
    confirm_master: bool = True,
    research_service_factory=None,
):
    store = Store(tmp_path / "flow.db")
    store.init_schema()
    runtime_schema.init_agent_runtime_schema(store)
    with store.connect() as conn:
        job_id = int(
            conn.execute(
                "INSERT INTO jobs(source, title, company, location, url, raw_text, "
                "content_hash) VALUES ('manual', 'Agent Intern', 'Example', 'Shanghai', "
                "'https://example.test/job', ?, 'flow-job') RETURNING id",
                ("Complete JD tail sentinel. " * 20,),
            ).fetchone()[0]
        )
    llm = StubLLM()
    master_source = master_resume_source_factory("MASTER_TAIL_SENTINEL")
    if confirm_master:
        ResumeWorkspaceRepository(store).save_master(
            source_path=master_source.source_path,
            source_sha256=master_source.sha256,
            extracted_text=master_source.extracted_text,
            semantic_document={
                "source_sha256": master_source.sha256,
                "semantic_text": master_source.extracted_text,
                "confirmed_by_user": True,
            },
            confirmed=True,
        )
    active_runtime = runtime or StubSkillRuntime()
    active_skills = list(skills or [])
    if not any(spec.name == "apply_assistant" for spec in active_skills):
        active_skills.append(_skill("apply_assistant"))
    research_agents = (
        research_service_factory(store) if research_service_factory is not None else None
    )
    app = create_app(
        settings=Settings(
            deepseek_api_key="stub",
            db_path=tmp_path / "flow.db",
            disable_background_agents=True,
        ),
        store=store,
        master_source=master_source,
        skills=active_skills,
        runtime=active_runtime,
        resume_editor=ResumeEditor(llm),
        research_agents=research_agents,
    )
    return TestClient(app), store, llm, job_id


class StubSkillRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.invoke_options: list[dict] = []

    def invoke(self, spec, inputs, **kwargs):
        self.calls.append((spec.name, inputs))
        self.invoke_options.append(kwargs)
        if spec.name == "apply_assistant":
            parsed = {
                "message": "Relevant application message",
                "form_answers": [],
                "pre_submit_checks": ["Confirm contact details"],
            }
        else:
            raise AssertionError(f"unexpected skill: {spec.name}")
        return SimpleNamespace(
            parsed=parsed,
            raw_text=json.dumps(parsed),
            skill_run_id=321,
            cost_usd=0.01,
            latency_ms=12,
        )


class StubResearchAgentService:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.source_store = SourceEvidenceStore(store)
        self.source_store.init_schema()
        self.interview_repository = InterviewResearchRepository(store)
        self.interview_source_provider = PublicInterviewSourceProvider(
            self.source_store,
            api_key="",
        )
        self.interview_agent = InterviewResearchAgent(
            store=store,
            llm=object(),
            source_provider=self.interview_source_provider,
            source_store=self.source_store,
        )
        self.interview_agent.init_schema()
        self.invocations: list[SimpleNamespace] = []
        self.enqueue_reasons: list[str] = []

    def ensure_interview_subject(self, *, application_id: int, workspace_id: int):
        return self.interview_repository.ensure_subject(application_id, workspace_id)

    def enqueue_interview_research(
        self,
        *,
        application_id: int,
        workspace_id: int,
        trigger_reason: str,
    ):
        subject = self.ensure_interview_subject(
            application_id=application_id,
            workspace_id=workspace_id,
        )
        self.enqueue_reasons.append(trigger_reason)
        with self.store.connect() as conn:
            now = float(conn.execute("SELECT julianday('now')").fetchone()[0])
        invocation = SimpleNamespace(
            id=f"invocation-{len(self.invocations) + 1}",
            status="queued",
            subject_revision=subject.agent_subject_revision,
            message="等待 Agent 运行",
            error_text=None,
            unresolved=(),
            updated_at=now,
        )
        self.invocations.append(invocation)
        return invocation, True

    def latest_interview_invocation(
        self,
        *,
        application_id: int,
        workspace_id: int,
    ):
        return self.invocations[-1] if self.invocations else None

    def add_interview_source(
        self,
        *,
        application_id: int,
        workspace_id: int,
        text: str,
        title: str,
        source_url: str | None = None,
    ) -> int:
        return self.interview_agent.add_user_source(
            application_id=application_id,
            submitted_workspace_id=workspace_id,
            text=text,
            title=title,
            source_url=source_url,
        )

    def latest_job_invocation(self):
        return None

    def close(self) -> None:
        self.interview_source_provider.close()


def _skill(name: str):
    root = Path(__file__).parent.parent / "src/offerguide/skills"
    return next(spec for spec in discover_skills(root) if spec.name == name)


def test_explicit_single_workspace_is_the_only_resume_and_submit_path(
    tmp_path,
    master_resume_source_factory,
) -> None:
    client, store, llm, job_id = _client(
        tmp_path,
        master_resume_source_factory,
        confirm_master=False,
    )

    page = client.get(f"/jobs/{job_id}/apply-pack")
    assert page.status_code == 200
    assert "确认 master 简历文本" in page.text
    assert llm.calls == []
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM resume_workspaces").fetchone()[0] == 0

    started = client.post(
        f"/api/jobs/{job_id}/resume-workspace/confirm-master",
        data={"semantic_text": "MASTER_TAIL_SENTINEL"},
        follow_redirects=False,
    )
    assert started.status_code == 303
    assert len(llm.calls) == 1
    with store.connect() as conn:
        application_id = int(
            conn.execute("SELECT id FROM applications WHERE job_id = ?", (job_id,)).fetchone()[0]
        )
    repo = ResumeWorkspaceRepository(store)
    draft = repo.get(application_id)
    assert draft is not None and draft.status == "draft"
    assert draft.pdf_path and Path(draft.pdf_path).is_file()
    assert (
        draft.resume_document["sections"][0]["entries"][0]["blocks"][0]["content"]["spans"][0][
            "text"
        ]
        == "SUBMITTED_RESUME_TOKEN"
    )

    reviewed = client.get(f"/jobs/{job_id}/apply-pack")
    assert reviewed.status_code == 200
    assert f"/api/resume-workspaces/{draft.id}/pdf" in reviewed.text
    assert '<iframe class="resume-frame"' in reviewed.text
    assert "Complete JD tail sentinel" in reviewed.text
    assert "tailored markdown" not in reviewed.text.lower()
    assert len(llm.calls) == 1
    pdf_response = client.get(f"/api/resume-workspaces/{draft.id}/pdf")
    assert pdf_response.status_code == 200
    assert pdf_response.headers["content-type"] == "application/pdf"
    assert pdf_response.headers["content-disposition"].startswith("inline;")

    revised = client.post(
        f"/api/jobs/{job_id}/resume-workspace/revise",
        data={"feedback": "Keep the same resume but improve emphasis."},
        follow_redirects=False,
    )
    assert revised.status_code == 303
    updated = repo.get(application_id)
    assert updated is not None and updated.id == draft.id
    assert updated.context["feedback"] == ["Keep the same resume but improve emphasis."]
    assert len(llm.calls) == 2

    submitted = client.post(f"/api/jobs/{job_id}/applied")
    assert submitted.status_code == 200
    frozen = repo.get(application_id)
    assert frozen is not None and frozen.status == "submitted"
    with store.connect() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM application_events "
                "WHERE application_id = ? AND kind = 'submitted'",
                (application_id,),
            ).fetchone()[0]
            == 1
        )

    assert client.get(f"/jobs/{job_id}/post-apply-pack").status_code == 200
    assert (
        client.post(
            f"/api/jobs/{job_id}/resume-workspace/revise",
            data={"feedback": "rewrite frozen"},
        ).status_code
        == 409
    )

    for legacy in ("/tailor", "/apply", f"/apply/{job_id}", "/profile/Example"):
        assert client.get(legacy).status_code == 404


def test_retired_pipeline_submit_route_cannot_bypass_workspace_submission(
    tmp_path,
    master_resume_source_factory,
) -> None:
    client, store, _llm, job_id = _client(tmp_path, master_resume_source_factory)
    response = client.post(
        f"/api/pipeline/jobs/{job_id}/submit",
        follow_redirects=False,
    )
    assert response.status_code == 404
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM application_events").fetchone()[0] == 0


def test_published_job_snapshot_stays_stable_through_apply_pack(
    tmp_path,
    master_resume_source_factory,
) -> None:
    client, store, _llm, _manual_job_id = _client(
        tmp_path,
        master_resume_source_factory,
    )
    repository = JobDiscoveryRepository(store)
    context = repository.replace_search_context(
        intent="寻找真实参与 Agent 产品交付的实习岗位",
        expected_revision=None,
    )
    published_evidence = repository.save_job_evidence(
        JobPostingEvidence(
            source_evidence_id=101,
            source_name="official-example",
            source_job_id="stable-position",
            canonical_url="https://jobs.example.com/stable-position",
            company="Snapshot Company",
            title="Published Role",
            location="Shanghai",
            recruitment_type="intern",
            page_time_information=["published page time"],
            jd_text="PUBLISHED_JD_SENTINEL " + ("complete evidence " * 20),
            source_status="open",
        )
    )
    assert published_evidence.job_id is not None
    assert published_evidence.id is not None
    published = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=published_evidence.id,
                why_worth_attention="真实参与 Agent 产品交付，与当前目标直接相关",
                grounding_quotes=[
                    SelectionGroundingQuote(
                        reference="search-context:intent",
                        quote="真实参与 Agent 产品交付",
                    )
                ],
                concerns=[],
                unknowns=[],
            )
        ],
        candidate_evidence=CandidateEvidence(),
        coverage_summary="已核验当前岗位",
        evidence_gaps=[],
    )

    repository.save_job_evidence(
        published_evidence.model_copy(
            update={
                "source_evidence_id": 102,
                "title": "Later Live Role",
                "jd_text": "MUTATED_LIVE_JD_SENTINEL " + ("later evidence " * 20),
            }
        )
    )

    recommended = client.get("/recommended")
    assert recommended.status_code == 200
    assert "Published Role" in recommended.text
    assert "Later Live Role" not in recommended.text

    job_id = published_evidence.job_id
    started = client.post(
        f"/api/jobs/{job_id}/resume-workspace/start",
        data={
            "selection_revision": str(published.result_revision),
            "job_evidence_id": str(published_evidence.id),
        },
    )
    assert started.status_code == 200
    application_id, _ = ResumeWorkspaceRepository(store).application_for_job(
        job_id,
        create=False,
    )
    assert application_id is not None
    workspace = ResumeWorkspaceRepository(store).get(application_id)
    assert workspace is not None
    assert workspace.job_snapshot["title"] == "Published Role"
    assert "PUBLISHED_JD_SENTINEL" in workspace.job_snapshot["raw_text"]
    assert "MUTATED_LIVE_JD_SENTINEL" not in workspace.job_snapshot["raw_text"]

    repository.save_job_evidence(
        published_evidence.model_copy(
            update={
                "source_evidence_id": 103,
                "title": "Newest Live Role",
                "jd_text": "NEWEST_LIVE_JD_SENTINEL " + ("newest evidence " * 20),
            }
        )
    )
    existing_page = client.get(f"/jobs/{job_id}/apply-pack")
    assert existing_page.status_code == 200
    assert "Published Role" in existing_page.text
    assert "PUBLISHED_JD_SENTINEL" in existing_page.text
    assert "Newest Live Role" not in existing_page.text
    assert "NEWEST_LIVE_JD_SENTINEL" not in existing_page.text

    client.post(f"/api/jobs/{job_id}/track")
    with store.connect() as conn:
        application_id = int(
            conn.execute("SELECT id FROM applications WHERE job_id = ?", (job_id,)).fetchone()[0]
        )
    response = client.post(
        f"/api/pipeline/applications/{application_id}/event",
        data={"kind": "submitted"},
    )
    assert response.status_code == 400
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM application_events").fetchone()[0] == 0


def test_recommended_action_uses_the_selection_revision_the_user_saw(
    tmp_path,
    master_resume_source_factory,
) -> None:
    client, store, _llm, _manual_job_id = _client(
        tmp_path,
        master_resume_source_factory,
    )
    repository = JobDiscoveryRepository(store)
    context = repository.replace_search_context(
        intent="寻找真实岗位",
        expected_revision=None,
    )
    evidence = repository.save_job_evidence(
        JobPostingEvidence(
            source_evidence_id=201,
            source_name="official-example",
            source_job_id="revision-bound-position",
            canonical_url="https://jobs.example.com/revision-bound-position",
            company="Revision Company",
            title="USER_SAW_THIS_ROLE",
            jd_text="USER_SAW_THIS_JD " + ("complete evidence " * 20),
            source_status="open",
        )
    )
    assert evidence.id is not None
    assert evidence.job_id is not None
    first = repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=evidence.id,
                why_worth_attention="寻找真实岗位：第一轮选择。",
                grounding_quotes=[
                    SelectionGroundingQuote(
                        reference="search-context:intent",
                        quote="寻找真实岗位",
                    )
                ],
            )
        ],
        candidate_evidence=CandidateEvidence(),
        coverage_summary="第一轮发布。",
        evidence_gaps=["团队信息未知"],
    )
    recommended = client.get("/recommended")
    assert recommended.status_code == 200
    assert (
        f"selection_revision={first.result_revision}&amp;job_evidence_id={evidence.id}"
        in recommended.text
    )

    repository.save_job_evidence(
        evidence.model_copy(
            update={
                "source_evidence_id": 202,
                "title": "BACKGROUND_REFRESH_ROLE",
                "jd_text": "BACKGROUND_REFRESH_JD " + ("changed evidence " * 20),
            }
        )
    )
    repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=first.result_revision,
        items=[
            JobSelectionItem(
                job_evidence_id=evidence.id,
                why_worth_attention="寻找真实岗位：后台刷新后的选择。",
                grounding_quotes=[
                    SelectionGroundingQuote(
                        reference="search-context:intent",
                        quote="寻找真实岗位",
                    )
                ],
            )
        ],
        candidate_evidence=CandidateEvidence(),
        coverage_summary="后台发布。",
        evidence_gaps=["团队信息未知"],
    )

    preview = client.get(
        f"/jobs/{evidence.job_id}/apply-pack",
        params={
            "selection_revision": first.result_revision,
            "job_evidence_id": evidence.id,
        },
    )
    assert preview.status_code == 200
    assert "USER_SAW_THIS_ROLE" in preview.text
    assert "BACKGROUND_REFRESH_ROLE" not in preview.text
    started = client.post(
        f"/api/jobs/{evidence.job_id}/resume-workspace/start",
        data={
            "selection_revision": str(first.result_revision),
            "job_evidence_id": str(evidence.id),
        },
    )
    assert started.status_code == 200
    application_id, _ = ResumeWorkspaceRepository(store).application_for_job(
        evidence.job_id,
        create=False,
    )
    assert application_id is not None
    workspace = ResumeWorkspaceRepository(store).get(application_id)
    assert workspace is not None
    assert workspace.job_snapshot["selection_result_revision"] == first.result_revision
    assert workspace.job_snapshot["title"] == "USER_SAW_THIS_ROLE"
    assert "USER_SAW_THIS_JD" in workspace.job_snapshot["raw_text"]
    assert "BACKGROUND_REFRESH_JD" not in workspace.job_snapshot["raw_text"]
    reopened = _exec_prepare_application(
        {"job_id": evidence.job_id},
        AgentRuntimeDeps(
            settings=Settings(db_path=store.db_path),
            store=store,
            memory_store=MemoryStore(tmp_path / "existing-workspace-memory"),
        ),
    )
    assert reopened.startswith("OK current application already exists")
    assert f'"workspace_id": {workspace.id}' in reopened


def test_agent_edits_the_same_application_workspace_as_the_web(
    tmp_path,
    master_resume_source_factory,
    monkeypatch,
) -> None:
    runtime = StubSkillRuntime()
    client, store, llm, job_id = _client(
        tmp_path,
        master_resume_source_factory,
        runtime=runtime,
    )
    del client
    master = ResumeWorkspaceRepository(store).get_master()
    assert master is not None
    source = MasterResumeSource(
        source_path=master.source_path,
        sha256=master.source_sha256,
        extracted_text=master.extracted_text,
    )
    monkeypatch.setattr("offerguide.resume.load_resume_pdf", lambda _path: source)
    deps = AgentRuntimeDeps(
        settings=Settings(db_path=store.db_path, resume_pdf=Path(master.source_path)),
        store=store,
        memory_store=MemoryStore(tmp_path / "agent-memory"),
        llm=llm,  # type: ignore[arg-type]
        runtime=runtime,  # type: ignore[arg-type]
        skills=[_skill("apply_assistant")],
    )

    started = _exec_prepare_application({"job_id": job_id}, deps)
    revised = _exec_revise_application(
        {"job_id": job_id, "feedback": "Keep facts, improve emphasis."},
        deps,
    )
    calls_before_rerender = (len(llm.calls), len(runtime.calls))
    rerendered = _exec_rerender_application({"job_id": job_id}, deps)

    assert started.startswith("OK current application updated")
    assert revised.startswith("OK current application updated")
    assert rerendered.startswith("OK current application updated")
    assert (len(llm.calls), len(runtime.calls)) == calls_before_rerender
    application_id, _ = ResumeWorkspaceRepository(store).application_for_job(
        job_id,
        create=False,
    )
    assert application_id is not None
    workspace = ResumeWorkspaceRepository(store).get(application_id)
    assert workspace is not None
    assert workspace.context["feedback"] == ["Keep facts, improve emphasis."]
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM resume_workspaces").fetchone()[0] == 1


def test_main_agent_event_binds_the_submitted_application(
    tmp_path,
    master_resume_source_factory,
) -> None:
    client, store, _llm, job_id = _client(
        tmp_path,
        master_resume_source_factory,
    )
    assert client.post(f"/api/jobs/{job_id}/resume-workspace/start").status_code == 200
    assert client.post(f"/api/jobs/{job_id}/applied").status_code == 200
    with store.connect() as conn:
        application_id = int(
            conn.execute("SELECT id FROM applications WHERE job_id = ?", (job_id,)).fetchone()[0]
        )

    result = _exec_record_event(
        {
            "application_id": application_id,
            "kind": "interview",
            "note": "用户确认收到二面通知",
            "round": "二面",
            "scheduled_at": "2026-07-20 14:00 Asia/Shanghai",
        },
        AgentRuntimeDeps(
            settings=Settings(db_path=store.db_path),
            store=store,
            memory_store=MemoryStore(tmp_path / "event-memory"),
        ),
    )

    assert result.startswith("OK application lifecycle updated")
    with store.connect() as conn:
        row = conn.execute(
            "SELECT application_id, kind, payload_json FROM application_events "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert tuple(row[:2]) == (application_id, "interview")
    payload = json.loads(row[2])
    assert payload["note"] == "用户确认收到二面通知"
    assert payload["round"] == "二面"
    assert payload["scheduled_at"] == "2026-07-20 14:00 Asia/Shanghai"
    with store.connect() as conn:
        status = conn.execute(
            "SELECT status FROM applications WHERE id = ?", (application_id,)
        ).fetchone()[0]
    assert status == "2nd_interview"


def test_main_agent_delegates_interview_research_for_the_frozen_submission(
    tmp_path,
    master_resume_source_factory,
) -> None:
    holder: dict[str, StubResearchAgentService] = {}

    def research_factory(store: Store) -> StubResearchAgentService:
        service = StubResearchAgentService(store)
        holder["service"] = service
        return service

    client, store, _llm, job_id = _client(
        tmp_path,
        master_resume_source_factory,
        research_service_factory=research_factory,
    )
    frozen_jd = "FROZEN_JD_HEAD " + ("J" * 5000) + " FROZEN_JD_TAIL"
    with store.connect() as conn:
        conn.execute("UPDATE jobs SET raw_text = ? WHERE id = ?", (frozen_jd, job_id))

    assert client.post(f"/api/jobs/{job_id}/resume-workspace/start").status_code == 200
    with store.connect() as conn:
        application_id = int(
            conn.execute("SELECT id FROM applications WHERE job_id = ?", (job_id,)).fetchone()[0]
        )
    repo = ResumeWorkspaceRepository(store)
    draft = repo.get(application_id)
    assert draft is not None and draft.pdf_path and draft.pdf_sha256
    context = json.loads(json.dumps(draft.context))
    context["project_facts"] = [
        {
            "project_id": "frozen-project",
            "title": "Frozen Project",
            "facts": ["FROZEN_PROJECT_FACT"],
            "do_not_claim": ["FROZEN_PROJECT_BOUNDARY"],
        }
    ]
    apply_pack = json.loads(json.dumps(draft.apply_pack))
    apply_pack["preparation_notes"] = [{"claim": "FROZEN_PREP_CLAIM", "note": "FROZEN_PREP_NOTE"}]
    repo.save_draft(
        application_id,
        job_snapshot=draft.job_snapshot,
        master_source_sha256=draft.master_source_sha256,
        context=context,
        resume_document=draft.resume_document,
        pdf_path=draft.pdf_path,
        pdf_sha256=draft.pdf_sha256,
        apply_pack=apply_pack,
    )
    assert client.post(f"/api/jobs/{job_id}/applied").status_code == 200

    with store.connect() as conn:
        conn.execute(
            "UPDATE jobs SET company = 'LIVE_COMPANY', title = 'LIVE_ROLE', "
            "raw_text = 'LIVE_JD_SHOULD_NOT_APPEAR' WHERE id = ?",
            (job_id,),
        )

    service = holder["service"]
    deps = AgentRuntimeDeps(
        settings=Settings(db_path=store.db_path),
        store=store,
        memory_store=MemoryStore(tmp_path / "agent-memory"),
        research_agents=service,
        user_profile_text="LIVE_MASTER_SHOULD_NOT_APPEAR",
    )
    result = _exec_research_interview({"job_id": job_id}, deps)

    assert "OK InterviewResearchAgent invocation accepted" in result
    assert service.enqueue_reasons[-1] == "main agent delegated interview research"
    submitted = service.interview_repository.load_subject(application_id, draft.id)
    assert submitted.frozen_submission["job_snapshot"]["raw_text"].endswith("FROZEN_JD_TAIL")
    assert "LIVE_JD_SHOULD_NOT_APPEAR" not in json.dumps(
        submitted.frozen_submission, ensure_ascii=False
    )
    assert "SUBMITTED_RESUME_TOKEN" in json.dumps(submitted.frozen_submission, ensure_ascii=False)
    assert "FROZEN_PROJECT_FACT" in json.dumps(submitted.frozen_submission, ensure_ascii=False)
    assert "FROZEN_PREP_CLAIM" in json.dumps(submitted.frozen_submission, ensure_ascii=False)
    question = "你如何设计 Agent Runtime 的任务执行与失败恢复？"
    evidence_record = service.source_store.save_user_provided(
        scope=SourceScope(
            run_id="main-agent-material",
            agent_name="interview_research_agent",
            subject_kind="interview_research",
            subject_id=f"application:{application_id}:workspace:{draft.id}",
            subject_revision=submitted.agent_subject_revision,
        ),
        text=f"这次一面实际问了：{question}",
        title="同岗位候选人面经",
        source_url="https://example.test/agent-runtime-interview",
    )
    evidence = EvidenceDocument(
        evidence_id=str(evidence_record.id),
        url=evidence_record.final_url,
        title=evidence_record.title,
        text=evidence_record.text_content,
        fetched_at=evidence_record.fetched_at,
        complete=True,
        attached_to_subject=True,
    )
    service.interview_repository.assess_source(
        application_id,
        draft.id,
        evidence=evidence,
        assessment=InterviewSourceAssessment(
            decision="accepted",
            source_kind="firsthand_interview",
            rationale="当前投递的同公司同岗位一手面经",
            actual_questions=[question],
        ),
        expected_subject_token=submitted.subject_token,
    )
    service.interview_repository.publish(
        application_id,
        draft.id,
        answer_set=InterviewAnswerSet(
            answers=[
                InterviewQuestionAnswer(
                    question=question,
                    answer="结合 FROZEN_PROJECT_FACT 说明，并补充 FROZEN_PREP_CLAIM。",
                    source_citations=[
                        SourceCitation(
                            evidence_id=str(evidence_record.id),
                            quote=question,
                        )
                    ],
                    grounding=[
                        GroundingReference(
                            kind="submitted_resume",
                            quote="SUBMITTED_RESUME_TOKEN",
                        )
                    ],
                )
            ],
        ),
        expected_subject_token=submitted.subject_token,
        expected_result_revision=0,
        evidence_loader=lambda requested_id: (
            evidence if requested_id == str(evidence_record.id) else None
        ),
    )

    current = _exec_read_artifact(
        {"artifact_kind": "current_application", "job_id": job_id},
        deps,
    )
    assert '"workspace_status": "submitted"' in current
    assert "SUBMITTED_RESUME_TOKEN" in current
    assert "LIVE_MASTER_SHOULD_NOT_APPEAR" not in current

    prep = _exec_read_artifact(
        {"artifact_kind": "current_interview_material", "job_id": job_id},
        deps,
    )
    assert question in prep
    assert "FROZEN_PROJECT_FACT" in prep
    assert '"status": "answered"' in prep
    assert f'"workspace_id": {draft.id}' in prep
    assert "LIVE_JD_SHOULD_NOT_APPEAR" not in prep


def test_main_agent_interview_research_rejects_an_unsubmitted_workspace(
    tmp_path,
    master_resume_source_factory,
) -> None:
    holder: dict[str, StubResearchAgentService] = {}

    def research_factory(store: Store) -> StubResearchAgentService:
        service = StubResearchAgentService(store)
        holder["service"] = service
        return service

    client, store, _llm, job_id = _client(
        tmp_path,
        master_resume_source_factory,
        research_service_factory=research_factory,
    )
    assert client.post(f"/api/jobs/{job_id}/resume-workspace/start").status_code == 200
    deps = AgentRuntimeDeps(
        settings=Settings(db_path=store.db_path),
        store=store,
        memory_store=MemoryStore(tmp_path / "agent-memory"),
        research_agents=holder["service"],
        user_profile_text="master",
    )

    result = _exec_research_interview({"job_id": job_id}, deps)

    assert result.startswith("ERROR:")
    assert "submitted resume workspace" in result
    assert holder["service"].enqueue_reasons == []


def test_interview_research_web_keeps_grounded_answers_visible_during_refresh(
    tmp_path,
    master_resume_source_factory,
) -> None:
    holder: dict[str, StubResearchAgentService] = {}

    def research_factory(store: Store) -> StubResearchAgentService:
        service = StubResearchAgentService(store)
        holder["service"] = service
        return service

    client, _store, _llm, job_id = _client(
        tmp_path,
        master_resume_source_factory,
        research_service_factory=research_factory,
    )
    assert client.post(f"/api/jobs/{job_id}/resume-workspace/start").status_code == 200
    submitted = client.post(f"/api/jobs/{job_id}/applied")
    assert submitted.status_code == 200
    assert submitted.json()["interview_research_status"] == "queued"

    service = holder["service"]
    assert service.enqueue_reasons == ["application submitted"]
    application_id = int(submitted.json()["application_id"])
    workspace_id = int(submitted.json()["workspace_id"])
    question = "你在项目中如何权衡 Agent 的执行效果与失败恢复成本？"
    source_text = f"候选人记录：一面原题是“{question}”"

    pasted = client.post(
        f"/interview-research/{application_id}/sources",
        data={
            "title": "用户找到的同岗位一手面经",
            "source_url": "https://example.test/interview-note",
            "text": source_text,
        },
        follow_redirects=False,
    )
    assert pasted.status_code == 303
    subject = service.interview_repository.load_subject(application_id, workspace_id)
    assert subject.context_updates[-1]["kind"] == "user_provided_source"
    assert service.enqueue_reasons[-1] == "user provided interview source"

    evidence_id = int(subject.context_updates[-1]["content"]["evidence_id"])
    record = service.source_store.get(evidence_id)
    evidence = EvidenceDocument(
        evidence_id=str(record.id),
        url=record.final_url,
        title=record.title,
        text=record.text_content,
        fetched_at=record.fetched_at,
        complete=True,
        attached_to_subject=True,
    )
    service.interview_repository.assess_source(
        application_id,
        workspace_id,
        evidence=evidence,
        assessment=InterviewSourceAssessment(
            decision="accepted",
            source_kind="firsthand_interview",
            rationale="用户确认这是当前公司和岗位的一手面经",
            actual_questions=[question],
        ),
        expected_subject_token=subject.subject_token,
    )
    answer_text = (
        "我会先定义任务成功标准，再用幂等检查点和有限重试隔离失败，"
        "最后结合实际错误分布调整恢复成本。"
    )
    answer_set = InterviewAnswerSet(
        answers=[
            InterviewQuestionAnswer(
                question=question,
                answer=answer_text,
                source_citations=[
                    SourceCitation(
                        evidence_id=str(evidence_id),
                        quote=question,
                    )
                ],
                grounding=[
                    GroundingReference(
                        kind="job_description",
                        quote="Complete JD tail sentinel.",
                    )
                ],
            )
        ]
    )
    published = service.interview_repository.publish(
        application_id,
        workspace_id,
        answer_set=answer_set,
        expected_subject_token=subject.subject_token,
        expected_result_revision=subject.result_revision,
        evidence_loader=lambda requested_id: evidence if requested_id == str(evidence_id) else None,
    )
    assert published.answer_set == answer_set

    source_url = (
        f"/interview-research/{application_id}/workspaces/{workspace_id}/sources/{evidence_id}"
    )
    page = client.get(f"/jobs/{job_id}/post-apply-pack")
    assert page.status_code == 200
    assert "真实面经原题与回答" in page.text
    assert question in page.text
    assert answer_text in page.text
    assert "用户找到的同岗位一手面经" in page.text
    assert "与当前岗位直接相关" not in page.text
    assert "针对当前投递的回答" in page.text
    assert "原题证据" in page.text
    assert "回答依据" in page.text
    assert "冻结 JD" in page.text
    assert source_url in page.text

    source_page = client.get(source_url)
    assert source_page.status_code == 200
    assert source_text in source_page.text
    assert "用户提供" in source_page.text
    assert (
        client.get(
            f"/interview-research/{application_id + 1}/workspaces/"
            f"{workspace_id}/sources/{evidence_id}"
        ).status_code
        == 404
    )

    additional_source = client.post(
        f"/interview-research/{application_id}/sources",
        data={
            "title": "新找到的补充面经",
            "source_url": "https://example.test/follow-up-interview-note",
            "text": "另一位候选人补充了后续轮次的真实面试原题。",
        },
        follow_redirects=False,
    )
    assert additional_source.status_code == 303
    stale_subject = service.interview_repository.load_subject(
        application_id,
        workspace_id,
    )
    assert stale_subject.current_material_is_stale is True
    assert stale_subject.current_material is not None
    assert stale_subject.current_material.answer_set == answer_set
    assert stale_subject.context_updates[-1]["kind"] == "user_provided_source"
    assert service.enqueue_reasons[-1] == "user provided interview source"

    refreshed = client.post(
        f"/interview-research/{application_id}/refresh",
        follow_redirects=False,
    )
    assert refreshed.status_code == 303
    assert service.enqueue_reasons[-1] == "user requested interview research refresh"
    refreshing_page = client.get(f"/jobs/{job_id}/post-apply-pack")
    assert "当前问答正在更新" in refreshing_page.text
    assert question in refreshing_page.text
    assert answer_text in refreshing_page.text
    assert "用户找到的同岗位一手面经" in refreshing_page.text
    assert source_url in refreshing_page.text
