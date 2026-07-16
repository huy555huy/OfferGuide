from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from offerguide.config import Settings
from offerguide.interview_research import InterviewResearchRepository
from offerguide.memory import Store
from offerguide.research_agents.job_discovery import (
    CandidateEvidence,
    JobDiscoveryRepository,
    JobDiscoveryRevisionConflict,
    JobPostingEvidence,
    JobSelectionItem,
    SelectionGroundingQuote,
    init_job_discovery_schema,
)
from offerguide.research_agents.service import ResearchAgentService
from offerguide.research_agents.sources import SourceEvidenceStore
from offerguide.ui.web import create_app


class _ResearchService:
    def __init__(self, store: Store) -> None:
        init_job_discovery_schema(store)
        self.job_repository = JobDiscoveryRepository(store)
        self.interview_repository = InterviewResearchRepository(store)
        self.source_store = SourceEvidenceStore(store)
        self.source_store.init_schema()
        self.enqueue_reasons: list[str] = []
        self.invocation: SimpleNamespace | None = None

    def latest_job_invocation(self):
        return self.invocation

    def replace_job_search_context(
        self,
        intent: str,
        *,
        hard_constraints: list[str] | None = None,
        expected_revision: int | None = None,
    ):
        current = self.job_repository.get_search_context()
        return self.job_repository.replace_search_context(
            intent=intent,
            hard_constraints=(
                hard_constraints
                if hard_constraints is not None
                else (current.hard_constraints if current else [])
            ),
            feedback=current.feedback if current else [],
            expected_revision=expected_revision,
        )

    def append_job_feedback(self, feedback: str, *, expected_revision: int):
        current = self.job_repository.get_search_context()
        assert current is not None
        return self.job_repository.replace_search_context(
            intent=current.intent,
            hard_constraints=current.hard_constraints,
            feedback=[*current.feedback, feedback],
            expected_revision=expected_revision,
        )

    def update_job_feedback(
        self,
        index: int,
        feedback: str,
        *,
        expected_revision: int,
    ):
        current = self.job_repository.get_search_context()
        assert current is not None
        values = list(current.feedback)
        values[index] = feedback
        return self.job_repository.replace_search_context(
            intent=current.intent,
            hard_constraints=current.hard_constraints,
            feedback=values,
            expected_revision=expected_revision,
        )

    def remove_job_feedback(self, index: int, *, expected_revision: int):
        current = self.job_repository.get_search_context()
        assert current is not None
        values = list(current.feedback)
        del values[index]
        return self.job_repository.replace_search_context(
            intent=current.intent,
            hard_constraints=current.hard_constraints,
            feedback=values,
            expected_revision=expected_revision,
        )

    def enqueue_job_discovery(self, *, trigger_reason: str):
        self.enqueue_reasons.append(trigger_reason)
        return SimpleNamespace(status="queued"), True


def _published_client(tmp_path: Path):
    store = Store(tmp_path / "selection-ui.db")
    store.init_schema()
    service = _ResearchService(store)
    context = service.job_repository.replace_search_context(
        intent="寻找真实 Agent 实习岗位",
        hard_constraints=["地点必须在上海或支持远程"],
        feedback=["少看纯运营岗位"],
        expected_revision=None,
    )
    evidence = service.job_repository.save_job_evidence(
        JobPostingEvidence(
            source_evidence_id=17,
            source_name="official-example",
            source_job_id="agent-intern-17",
            canonical_url="https://jobs.example.com/agent-intern-17",
            company="Example AI",
            title="Agent 实习生",
            location="上海",
            recruitment_type="实习",
            page_time_information=["页面显示 2026-07-14 更新"],
            jd_text="完整 JD " + ("负责真实 Agent 产品交付。" * 30),
            source_status="open",
        )
    )
    assert evidence.id is not None and evidence.job_id is not None
    selection = service.job_repository.publish_selection(
        expected_context_revision=context.revision,
        expected_result_revision=0,
        items=[
            JobSelectionItem(
                job_evidence_id=evidence.id,
                why_worth_attention="Agent 实习岗位与当前方向直接一致。",
                grounding_quotes=[
                    SelectionGroundingQuote(
                        reference="search-context:intent",
                        quote="Agent 实习岗位",
                    )
                ],
                unknowns=["团队规模未公开"],
            )
        ],
        candidate_evidence=CandidateEvidence(),
        coverage_summary="已读取并比较完整岗位页。",
        evidence_gaps=["没有找到该团队公开的面试流程"],
    )
    service.invocation = SimpleNamespace(
        id="failed-run",
        status="failed",
        message="来源读取失败",
        error_text="browser login expired",
        unresolved=("登录态来源尚未读取",),
        updated_at=selection.published_at,
    )
    app = create_app(
        settings=Settings(db_path=store.db_path),
        store=store,
        master_source=None,
        skills=[],
        runtime=None,
        research_agents=service,  # type: ignore[arg-type]
    )
    return TestClient(app), store, service, context, evidence, selection


def test_recommended_and_mission_control_preserve_complete_selection_binding(
    tmp_path: Path,
) -> None:
    client, _store, _service, _context, evidence, selection = _published_client(
        tmp_path
    )
    expected_query = (
        f"selection_revision={selection.result_revision}"
        f"&amp;job_evidence_id={evidence.id}"
    )

    recommended = client.get("/recommended")
    mission = client.get("/")

    assert recommended.status_code == 200
    assert mission.status_code == 200
    for value in (
        "地点必须在上海或支持远程",
        "少看纯运营岗位",
        "实习",
        "页面显示 2026-07-14 更新",
        "最近检查",
        "最近成功读取",
        "没有找到该团队公开的面试流程",
        "browser login expired",
        "登录态来源尚未读取",
        "Agent 实习岗位",
    ):
        assert value in recommended.text
    assert expected_query in recommended.text
    assert expected_query in mission.text


def test_pipeline_only_shows_current_actionable_candidates(
    tmp_path: Path,
) -> None:
    client, store, _service, _context, evidence, selection = _published_client(
        tmp_path
    )
    with store.connect() as conn:
        for index in range(12):
            conn.execute(
                "INSERT INTO jobs(source, title, company, url, raw_text, content_hash) "
                "VALUES ('legacy-crawl', ?, '旧来源', ?, ?, ?)",
                (
                    f"历史抓取岗位 {index}",
                    f"https://legacy.example/{index}",
                    "旧链路中没有当前证据绑定的 JD" * 20,
                    f"legacy-pipeline-{index}",
                ),
            )

    page = client.get("/pipeline")

    assert page.status_code == 200
    assert "待投递" in page.text
    assert "Agent 实习生" in page.text
    assert "历史抓取岗位" not in page.text
    assert "历史线索 · 未进入当前选择集" not in page.text
    assert (
        f"/jobs/{evidence.job_id}/apply-pack?selection_revision="
        f"{selection.result_revision}&amp;job_evidence_id={evidence.id}"
    ) in page.text


def test_job_feedback_is_visible_editable_removable_and_revision_safe(
    tmp_path: Path,
) -> None:
    client, _store, service, context, _evidence, _selection = _published_client(
        tmp_path
    )

    edited = client.post(
        "/job-search/feedback/0",
        data={"context_revision": str(context.revision), "feedback": "多找平台工程岗位"},
        follow_redirects=False,
    )
    assert edited.status_code == 303
    updated = service.job_repository.get_search_context()
    assert updated is not None and updated.feedback == ["多找平台工程岗位"]

    stale = client.post(
        "/job-search/feedback/0",
        data={"context_revision": str(context.revision), "feedback": "覆盖新编辑"},
    )
    assert stale.status_code == 409
    assert service.job_repository.get_search_context() == updated

    removed = client.post(
        "/job-search/feedback/0/remove",
        data={"context_revision": str(updated.revision)},
        follow_redirects=False,
    )
    assert removed.status_code == 303
    final = service.job_repository.get_search_context()
    assert final is not None and final.feedback == []


def test_closed_evidence_blocks_new_workspace_but_keeps_existing_workspace_viewable(
    tmp_path: Path,
) -> None:
    client, store, service, _context, evidence, selection = _published_client(tmp_path)
    assert evidence.id is not None
    binding = {
        "selection_revision": str(selection.result_revision),
        "job_evidence_id": str(evidence.id),
    }

    reported = client.post(
        f"/job-search/jobs/{evidence.job_id}/report-closed",
        data=binding,
        follow_redirects=False,
    )
    assert reported.status_code == 303
    updated_evidence = service.job_repository.get_job_evidence(evidence.id)
    assert updated_evidence is not None and updated_evidence.source_status == "closed"
    assert client.post(f"/api/jobs/{evidence.job_id}/report-dead").status_code == 404

    page = client.get(f"/jobs/{evidence.job_id}/apply-pack", params=binding)
    blocked = client.post(
        f"/api/jobs/{evidence.job_id}/resume-workspace/start", data=binding
    )
    assert page.status_code == 200
    assert "不能新建投递工作区" in page.text
    assert blocked.status_code == 409
    assert "最近一次检查显示已关闭" in blocked.json()["detail"]

    with store.connect() as conn:
        application_id = int(
            conn.execute(
                "INSERT INTO applications(job_id, status) "
                "VALUES (?, 'considered') RETURNING id",
                (evidence.job_id,),
            ).fetchone()[0]
        )
        conn.execute(
            "INSERT INTO resume_workspaces("
            "application_id, job_snapshot_json, master_source_sha256, context_json, "
            "resume_document_json, apply_pack_json"
            ") VALUES (?, ?, ?, '{}', '{}', '{}')",
            (
                application_id,
                json.dumps(
                    {
                        "id": evidence.job_id,
                        "job_id": evidence.job_id,
                        "title": evidence.title,
                        "company": evidence.company,
                        "raw_text": evidence.jd_text,
                        "url": evidence.canonical_url,
                    }
                ),
                "0" * 64,
            ),
        )

    existing = client.get(f"/jobs/{evidence.job_id}/apply-pack")
    assert existing.status_code == 200
    assert "完整 JD" in existing.text


def test_non_manual_bare_job_id_cannot_enter_apply_pack(tmp_path: Path) -> None:
    client, _store, _service, _context, evidence, _selection = _published_client(
        tmp_path
    )

    response = client.get(f"/jobs/{evidence.job_id}/apply-pack")

    assert response.status_code == 409
    assert "必须从带版本的岗位选择入口进入" in response.json()["detail"]


def test_production_service_feedback_operations_use_compare_and_swap(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "service-feedback.db")
    store.init_schema()
    init_job_discovery_schema(store)
    repository = JobDiscoveryRepository(store)
    first = repository.replace_search_context(
        intent="寻找真实岗位",
        feedback=["原反馈"],
        expected_revision=None,
    )
    service = object.__new__(ResearchAgentService)
    service.job_repository = repository

    second = service.update_job_feedback(
        0,
        "新反馈",
        expected_revision=first.revision,
    )
    assert second.feedback == ["新反馈"]
    with pytest.raises(JobDiscoveryRevisionConflict):
        service.remove_job_feedback(0, expected_revision=first.revision)
    final = service.remove_job_feedback(0, expected_revision=second.revision)
    assert final.feedback == []
