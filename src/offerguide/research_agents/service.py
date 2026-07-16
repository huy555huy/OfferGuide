"""Production composition for OfferGuide's two domain research agents."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .. import project_vault
from ..agentic.search import build_search_backends
from ..config import Settings
from ..interview_research import (
    InterviewResearchAgent,
    InterviewResearchRepository,
    InterviewResearchSubject,
    PublicInterviewSourceProvider,
    init_interview_research_schema,
)
from ..llm import LLMClient
from ..memory import Store
from ..resume import ResumeWorkspaceRepository
from .browser_bridge import (
    AuthenticatedBrowserBridgeClient,
    AuthenticatedBrowserBridgeStore,
)
from .coordinator import AgentInvocation, ResearchAgentCoordinator
from .job_discovery import (
    CandidateEvidence,
    CandidateEvidenceDocument,
    JobDiscoveryAgent,
    JobDiscoveryRepository,
    JobDiscoveryRevisionConflict,
    JobSearchContext,
    StoredJobSourceEvidenceVerifier,
    build_job_source_tools,
    init_job_discovery_schema,
    standard_source_dependencies,
)
from .job_discovery.platform_tools import build_platform_job_tools
from .runner import AgentRunner, AgentRunResult, AgentRunStatus
from .sources import SearchExecutor, SourceEvidenceStore, SourceReader


class ResearchAgentServiceError(RuntimeError):
    pass


class ResearchAgentService:
    """The only production composition used by Web, background, and delegation."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: Store,
        llm: Any | None = None,
        search_executor: SearchExecutor | None = None,
        source_reader: SourceReader | None = None,
        interview_source_provider: PublicInterviewSourceProvider | None = None,
        coordinator: ResearchAgentCoordinator | None = None,
    ) -> None:
        if llm is None and not settings.deepseek_api_key:
            raise ResearchAgentServiceError("research agents require an LLM API key")
        self.settings = settings
        self.store = store
        self.llm = llm or LLMClient(
            api_key=str(settings.deepseek_api_key),
            base_url=settings.deepseek_base_url,
            default_model=settings.default_model,
        )
        self._owns_llm = llm is None
        self.source_store = SourceEvidenceStore(store)
        self.source_store.init_schema()
        self.browser_bridge_store = AuthenticatedBrowserBridgeStore(store)
        self.browser_bridge_store.init_schema()
        self.authenticated_browser_client = AuthenticatedBrowserBridgeClient(
            self.browser_bridge_store
        )
        init_job_discovery_schema(store)
        init_interview_research_schema(store)
        self.job_repository = JobDiscoveryRepository(store)
        self.interview_repository = InterviewResearchRepository(store)
        self.coordinator = coordinator or ResearchAgentCoordinator(store)
        self._owns_coordinator = coordinator is None

        if search_executor is None:
            self._search_backends = build_search_backends()
            search_executor = SearchExecutor(self.source_store, self._search_backends)
        else:
            self._search_backends = []
        self.search_executor = search_executor
        self.source_reader = source_reader or SourceReader(
            self.source_store,
            authenticated_browser_client=self.authenticated_browser_client,
        )
        self._owns_source_reader = source_reader is None
        self.interview_source_provider = (
            interview_source_provider
            or PublicInterviewSourceProvider(self.source_store)
        )
        self._owns_interview_source_provider = interview_source_provider is None

        source_dependencies = standard_source_dependencies(
            evidence_store=self.source_store,
            search_executor=self.search_executor,
            source_reader=self.source_reader,
        )
        self.job_agent = JobDiscoveryAgent(
            repository=self.job_repository,
            runner=AgentRunner(self.llm),
            candidate_evidence_loader=self._candidate_evidence,
            source_tools=(
                *build_job_source_tools(),
                *build_platform_job_tools(),
            ),
            source_evidence_verifier=StoredJobSourceEvidenceVerifier(self.source_store),
            dependencies=source_dependencies,
            model=settings.default_model,
        )
        self.interview_agent = InterviewResearchAgent(
            store=store,
            llm=self.llm,
            source_provider=self.interview_source_provider,
            source_store=self.source_store,
            model=settings.interview_research_model or settings.default_model,
        )
        self.interview_agent.init_schema()

    def replace_job_search_context(
        self,
        intent: str,
        *,
        hard_constraints: list[str] | None = None,
        expected_revision: int | None = None,
    ) -> JobSearchContext:
        """Replace the visible semantic document with a revision-safe write.

        ``expected_revision`` is supplied by browser forms. The main Agent's
        explicit-intent path omits it and binds the write to the revision it
        just read here, so neither path can silently overwrite a concurrent
        user edit.
        """
        current = self.job_repository.get_search_context()
        effective_revision = (
            expected_revision
            if expected_revision is not None
            else (current.revision if current else None)
        )
        return self.job_repository.replace_search_context(
            intent=intent,
            hard_constraints=(
                hard_constraints
                if hard_constraints is not None
                else (current.hard_constraints if current else [])
            ),
            feedback=current.feedback if current else [],
            expected_revision=effective_revision,
        )

    def append_job_feedback(
        self,
        feedback: str,
        *,
        expected_revision: int | None = None,
    ) -> JobSearchContext:
        value = feedback.strip()
        if not value:
            raise ValueError("job-search feedback must not be blank")
        current = self.job_repository.get_search_context()
        if current is None:
            raise ResearchAgentServiceError("create a job-search context first")
        self._require_job_context_revision(current, expected_revision)
        if value in current.feedback:
            return current
        return self.job_repository.replace_search_context(
            intent=current.intent,
            hard_constraints=current.hard_constraints,
            feedback=[*current.feedback, value],
            expected_revision=current.revision,
        )

    def update_job_feedback(
        self,
        index: int,
        feedback: str,
        *,
        expected_revision: int,
    ) -> JobSearchContext:
        value = feedback.strip()
        if not value:
            raise ValueError("job-search feedback must not be blank")
        current = self.job_repository.get_search_context()
        if current is None:
            raise ResearchAgentServiceError("create a job-search context first")
        self._require_job_context_revision(current, expected_revision)
        if not 0 <= index < len(current.feedback):
            raise IndexError("job-search feedback no longer exists")
        feedback_values = list(current.feedback)
        feedback_values[index] = value
        return self.job_repository.replace_search_context(
            intent=current.intent,
            hard_constraints=current.hard_constraints,
            feedback=feedback_values,
            expected_revision=current.revision,
        )

    def remove_job_feedback(
        self,
        index: int,
        *,
        expected_revision: int,
    ) -> JobSearchContext:
        current = self.job_repository.get_search_context()
        if current is None:
            raise ResearchAgentServiceError("create a job-search context first")
        self._require_job_context_revision(current, expected_revision)
        if not 0 <= index < len(current.feedback):
            raise IndexError("job-search feedback no longer exists")
        feedback_values = list(current.feedback)
        del feedback_values[index]
        return self.job_repository.replace_search_context(
            intent=current.intent,
            hard_constraints=current.hard_constraints,
            feedback=feedback_values,
            expected_revision=current.revision,
        )

    @staticmethod
    def _require_job_context_revision(
        current: JobSearchContext,
        expected_revision: int | None,
    ) -> None:
        if expected_revision is not None and expected_revision != current.revision:
            raise JobDiscoveryRevisionConflict(
                f"search context is revision {current.revision}, not {expected_revision}"
            )

    def enqueue_job_discovery(self, *, trigger_reason: str) -> tuple[AgentInvocation, bool]:
        context = self.job_repository.get_search_context()
        if context is None:
            raise ResearchAgentServiceError("create a job-search context first")
        expected_revision = context.revision

        def _call() -> AgentRunResult:
            current = self.job_repository.get_search_context()
            if current is None or current.revision != expected_revision:
                return _stale_result(
                    agent_name="JobDiscoveryAgent",
                    subject_kind="job_search",
                    subject_id="current",
                    subject_revision=expected_revision,
                    reason="job-search context changed before the agent started",
                )
            return self.job_agent.run(trigger_reason=trigger_reason)

        return self.coordinator.enqueue(
            agent_name="JobDiscoveryAgent",
            subject_kind="job_search",
            subject_id="current",
            subject_revision=expected_revision,
            trigger_reason=trigger_reason,
            call=_call,
        )

    def latest_job_invocation(self) -> AgentInvocation | None:
        return self.coordinator.latest(
            agent_name="JobDiscoveryAgent",
            subject_kind="job_search",
            subject_id="current",
        )

    def ensure_interview_subject(
        self,
        *,
        application_id: int,
        workspace_id: int,
    ) -> InterviewResearchSubject:
        return self.interview_repository.ensure_subject(application_id, workspace_id)

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

    def enqueue_interview_research(
        self,
        *,
        application_id: int,
        workspace_id: int,
        trigger_reason: str,
    ) -> tuple[AgentInvocation, bool]:
        subject = self.interview_repository.ensure_subject(application_id, workspace_id)
        expected_revision = subject.agent_subject_revision
        subject_id = _interview_subject_id(application_id, workspace_id)

        def _call() -> AgentRunResult:
            current = self.interview_repository.ensure_subject(application_id, workspace_id)
            if current.agent_subject_revision != expected_revision:
                return _stale_result(
                    agent_name="interview_research_agent",
                    subject_kind="interview_research",
                    subject_id=subject_id,
                    subject_revision=expected_revision,
                    reason="interview context changed before the agent started",
                )
            return self.interview_agent.run(
                application_id=application_id,
                submitted_workspace_id=workspace_id,
                trigger_reason=trigger_reason,
            )

        return self.coordinator.enqueue(
            agent_name="interview_research_agent",
            subject_kind="interview_research",
            subject_id=subject_id,
            subject_revision=expected_revision,
            trigger_reason=trigger_reason,
            call=_call,
        )

    def latest_interview_invocation(
        self,
        *,
        application_id: int,
        workspace_id: int,
    ) -> AgentInvocation | None:
        return self.coordinator.latest(
            agent_name="interview_research_agent",
            subject_kind="interview_research",
            subject_id=_interview_subject_id(application_id, workspace_id),
        )

    def close(self) -> None:
        if self._owns_coordinator:
            self.coordinator.shutdown(wait=True)
        if self._owns_source_reader:
            close = getattr(self.source_reader.client, "close", None)
            if callable(close):
                close()
        if self._owns_interview_source_provider:
            self.interview_source_provider.close()
        for backend in self._search_backends:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        if self._owns_llm:
            close = getattr(self.llm, "close", None)
            if callable(close):
                close()

    def _candidate_evidence(self) -> CandidateEvidence:
        documents: list[CandidateEvidenceDocument] = []
        master = ResumeWorkspaceRepository(self.store).get_master()
        if master is not None:
            semantic_text = str(master.semantic_document.get("semantic_text") or "").strip()
            text = semantic_text if master.semantic_status == "confirmed" else master.extracted_text
            if text.strip():
                documents.append(CandidateEvidenceDocument(
                    reference=f"master:{master.source_sha256}",
                    kind="confirmed_master_resume" if master.semantic_status == "confirmed" else "master_resume_evidence",
                    title="当前候选人简历",
                    text=text,
                ))

        with self.store.connect() as conn:
            project_count = int(conn.execute("SELECT COUNT(*) FROM project_records").fetchone()[0])
        for record in project_vault.list_all(self.store, limit=max(project_count, 1)):
            documents.append(CandidateEvidenceDocument(
                reference=f"project:{record.id}",
                kind="project_vault",
                title=record.title,
                text=json.dumps(asdict(record), ensure_ascii=False, default=str),
            ))
        return CandidateEvidence(documents=documents)


def _interview_subject_id(application_id: int, workspace_id: int) -> str:
    return f"application:{application_id}:workspace:{workspace_id}"


def _stale_result(
    *,
    agent_name: str,
    subject_kind: str,
    subject_id: str,
    subject_revision: int,
    reason: str,
) -> AgentRunResult:
    return AgentRunResult(
        run_id="preflight-stale",
        agent_name=agent_name,
        subject_kind=subject_kind,
        subject_id=subject_id,
        subject_revision=subject_revision,
        starting_result_revision=0,
        status=AgentRunStatus.STALE,
        iterations=0,
        final_text="",
        terminal_tool=None,
        tool_calls=(),
        cost_usd=0.0,
        latency_ms=0,
        reason=reason,
    )


__all__ = ["ResearchAgentService", "ResearchAgentServiceError"]
