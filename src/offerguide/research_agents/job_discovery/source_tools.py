"""Explicit shared-source tools exposed to JobDiscoveryAgent."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from ..runner import AgentExecutionContext, AgentTool, AgentToolResult
from ..sources import (
    EvidenceNotFoundError,
    SearchExecutor,
    SourceEvidenceStore,
    SourceReader,
    SourceScope,
)
from .models import JobPostingEvidence
from .quality import require_substantive_job_description

SEARCH_EXECUTOR_DEPENDENCY = "job_source_search_executor"
SOURCE_READER_DEPENDENCY = "job_source_reader"
SOURCE_STORE_DEPENDENCY = "job_source_evidence_store"
PLATFORM_DERIVED_JOB_PURPOSE = "verified platform job detail"


class StoredJobSourceEvidenceVerifier:
    """Resolve ordinary pages and verify deterministic platform adapter results."""

    def __init__(self, evidence_store: SourceEvidenceStore) -> None:
        self.evidence_store = evidence_store

    def resolve_generic_job_page(
        self,
        *,
        source_evidence_id: int,
        company: str,
        title: str,
        location: str | None,
        recruitment_type: str | None,
        page_time_information: list[str],
        source_status: str,
        subject_kind: str,
        subject_id: str | int,
        subject_revision: int,
    ) -> JobPostingEvidence:
        evidence = self._attached_evidence(
            source_evidence_id=source_evidence_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            subject_revision=subject_revision,
        )
        if self._is_platform_derived_job(
            source_evidence_id=source_evidence_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            subject_revision=subject_revision,
        ):
            raise ValueError(
                "platform-derived evidence must be recorded by its platform_job_ref"
            )
        haystack = _normalized_source_text(
            f"{evidence.title}\n{evidence.text_content}"
        )
        facts = {
            "company": company,
            "title": title,
            "location": location,
            "recruitment_type": recruitment_type,
        }
        for label, value in facts.items():
            normalized_value = _normalized_source_text(value) if value else ""
            if value and (not normalized_value or normalized_value not in haystack):
                raise ValueError(f"the proposed {label} is not present in the source page")
        for value in page_time_information:
            normalized_value = _normalized_source_text(value)
            if not normalized_value or normalized_value not in haystack:
                raise ValueError(
                    "page_time_information must contain exact text from the source page"
                )
        if source_status != "unknown":
            raise ValueError(
                "a generic fetched page must use source_status='unknown'; "
                "only deterministic platform adapters may assert open or closed"
            )
        parsed = urlsplit(evidence.canonical_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("generic job evidence must have a saved HTTP(S) source URL")
        require_substantive_job_description(
            evidence.text_content,
            identity_values=(
                company,
                title,
                location,
                recruitment_type,
                *page_time_information,
            ),
        )
        return JobPostingEvidence(
            evidence_kind="web",
            source_evidence_id=evidence.id,
            source_name=parsed.hostname.lower(),
            source_job_id=None,
            canonical_url=evidence.canonical_url,
            company=company,
            title=title,
            location=location,
            recruitment_type=recruitment_type,
            page_time_information=page_time_information,
            jd_text=evidence.text_content.strip(),
            source_status="unknown",
        )

    def verify_platform_job_page(
        self,
        evidence: JobPostingEvidence,
        *,
        subject_kind: str,
        subject_id: str | int,
        subject_revision: int,
    ) -> None:
        if evidence.evidence_kind != "platform_adapter":
            raise ValueError("platform job evidence must come from a deterministic adapter")
        self._attached_evidence(
            source_evidence_id=evidence.source_evidence_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            subject_revision=subject_revision,
        )
        if not self._is_platform_derived_job(
            source_evidence_id=evidence.source_evidence_id,
            subject_kind=subject_kind,
            subject_id=subject_id,
            subject_revision=subject_revision,
        ):
            raise ValueError(
                "platform job evidence is not attached by a deterministic adapter"
            )

    def _attached_evidence(
        self,
        *,
        source_evidence_id: int,
        subject_kind: str,
        subject_id: str | int,
        subject_revision: int,
    ):
        evidence = self.evidence_store.get(source_evidence_id)
        if not evidence.is_readable:
            raise ValueError(f"source evidence {source_evidence_id} has no readable body")
        if not self.evidence_store.is_attached(
            subject_kind=subject_kind,
            subject_id=subject_id,
            subject_revision=subject_revision,
            evidence_id=source_evidence_id,
        ):
            raise ValueError(
                f"source evidence {source_evidence_id} is not attached to this search revision"
            )
        return evidence

    def _is_platform_derived_job(
        self,
        *,
        source_evidence_id: int,
        subject_kind: str,
        subject_id: str | int,
        subject_revision: int,
    ) -> bool:
        """Identify evidence whose JD is produced by a deterministic adapter.

        Platform APIs often return structured fields or several jobs in one response.
        Their normalized JD is therefore not necessarily a literal response substring.
        The attachment purpose is written only by the verified platform tool, while
        ordinary fetched pages retain exact-passage verification.
        """
        with self.evidence_store.store.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM source_subject_evidence "
                "WHERE subject_kind = ? AND subject_id = ? AND subject_revision = ? "
                "AND evidence_id = ? AND purpose = ? LIMIT 1",
                (
                    subject_kind,
                    str(subject_id),
                    subject_revision,
                    source_evidence_id,
                    PLATFORM_DERIVED_JOB_PURPOSE,
                ),
            ).fetchone()
        return row is not None


def build_job_source_tools() -> tuple[AgentTool, ...]:
    """Build the real search, fetch, and paginated evidence-reading tools."""

    return (
        AgentTool(
            name="search_job_sources",
            description=(
                "Execute the query against every configured search backend and persist "
                "the actual attempts. Results are unverified clues only; open selected "
                "URLs with fetch_job_source or fetch_authenticated_job_source before "
                "treating them as evidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "max_results_per_backend": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_search_sources,
        ),
        AgentTool(
            name="fetch_job_source",
            description=(
                "Fetch one public HTTP(S) clue URL, validate redirects/content type, "
                "and save the complete raw response and readable text as untrusted "
                "source evidence. A failed or restricted fetch remains a visible failure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 1},
                    "purpose": {"type": "string"},
                    "title_hint": {"type": "string"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            handler=_fetch_source,
        ),
        AgentTool(
            name="fetch_authenticated_job_source",
            description=(
                "Use the user's connected signed-in browser for one public HTTP(S) URL "
                "when the normal fetch is blocked by login or client-side rendering. The "
                "bridge saves the complete rendered HTML and readable text as untrusted "
                "evidence. Missing browser session, login wall, timeout, and size limits "
                "are explicit failures; this tool never clicks, submits, or returns cookies."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 1},
                    "purpose": {"type": "string"},
                    "title_hint": {"type": "string"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            handler=_fetch_authenticated_source,
        ),
        AgentTool(
            name="read_job_source",
            description=(
                "Read a saved source by evidence id without losing the full body. Use "
                "next_offset until complete when a page exceeds max_chars. Only evidence "
                "attached to this exact search-context revision can be read."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "integer", "minimum": 1},
                    "offset": {"type": "integer", "minimum": 0},
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50_000,
                    },
                },
                "required": ["evidence_id"],
                "additionalProperties": False,
            },
            handler=_read_source,
        ),
    )


def standard_source_dependencies(
    *,
    evidence_store: SourceEvidenceStore,
    search_executor: SearchExecutor,
    source_reader: SourceReader,
) -> dict[str, Any]:
    return {
        SEARCH_EXECUTOR_DEPENDENCY: search_executor,
        SOURCE_READER_DEPENDENCY: source_reader,
        SOURCE_STORE_DEPENDENCY: evidence_store,
    }


def _scope(execution: AgentExecutionContext) -> SourceScope:
    return SourceScope(
        run_id=execution.run_id,
        agent_name=execution.agent_name,
        subject_kind=execution.subject.subject_kind,
        subject_id=execution.subject.subject_id,
        subject_revision=execution.subject.subject_revision,
    )


def _search_sources(args: Mapping[str, Any], execution: AgentExecutionContext) -> AgentToolResult:
    query = str(args.get("query") or "").strip()
    if not query:
        return AgentToolResult({"error": "query must not be blank"}, is_error=True)
    raw_max = args.get("max_results_per_backend", 10)
    if isinstance(raw_max, bool):
        return AgentToolResult(
            {"error": "max_results_per_backend must be an integer"}, is_error=True
        )
    try:
        max_results = int(raw_max)
        executor: SearchExecutor = execution.dependency(SEARCH_EXECUTOR_DEPENDENCY)
        result = executor.execute(
            query,
            scope=_scope(execution),
            max_results_per_backend=max_results,
        )
    except (TypeError, ValueError) as exc:
        return AgentToolResult({"error": str(exc)}, is_error=True)
    all_failed = bool(result.attempts) and all(
        attempt.status == "failed" for attempt in result.attempts
    )
    return AgentToolResult(result.as_tool_result(), is_error=all_failed)


def _fetch_source(args: Mapping[str, Any], execution: AgentExecutionContext) -> AgentToolResult:
    url = str(args.get("url") or "").strip()
    if not url:
        return AgentToolResult({"error": "url must not be blank"}, is_error=True)
    reader: SourceReader = execution.dependency(SOURCE_READER_DEPENDENCY)
    result = reader.fetch(
        url,
        scope=_scope(execution),
        purpose=str(args.get("purpose") or "job_detail").strip(),
        title_hint=str(args.get("title_hint") or "").strip(),
    )
    if result.evidence is None:
        repository = execution.dependencies.get("job_discovery_repository")
        if repository is not None:
            repository.record_source_check(
                canonical_url=result.requested_url,
                source_status=(
                    "closed" if result.http_status in {404, 410} else "unknown"
                ),
            )
    return AgentToolResult(result.as_tool_result(), is_error=result.evidence is None)


def _fetch_authenticated_source(
    args: Mapping[str, Any], execution: AgentExecutionContext
) -> AgentToolResult:
    url = str(args.get("url") or "").strip()
    if not url:
        return AgentToolResult({"error": "url must not be blank"}, is_error=True)
    reader: SourceReader = execution.dependency(SOURCE_READER_DEPENDENCY)
    result = reader.fetch_authenticated(
        url,
        scope=_scope(execution),
        purpose=str(args.get("purpose") or "authenticated job detail").strip(),
        title_hint=str(args.get("title_hint") or "").strip(),
    )
    return AgentToolResult(result.as_tool_result(), is_error=result.evidence is None)


def _read_source(args: Mapping[str, Any], execution: AgentExecutionContext) -> AgentToolResult:
    raw_evidence_id = args.get("evidence_id")
    if raw_evidence_id is None:
        return AgentToolResult(
            {"error": "evidence_id is required"},
            is_error=True,
        )
    try:
        evidence_id = int(raw_evidence_id)
        offset = int(args.get("offset", 0))
        max_chars = int(args.get("max_chars", 12_000))
    except (TypeError, ValueError):
        return AgentToolResult(
            {"error": "evidence_id, offset, and max_chars must be integers"},
            is_error=True,
        )
    store: SourceEvidenceStore = execution.dependency(SOURCE_STORE_DEPENDENCY)
    if not store.is_attached(
        subject_kind=execution.subject.subject_kind,
        subject_id=execution.subject.subject_id,
        subject_revision=execution.subject.subject_revision,
        evidence_id=evidence_id,
    ):
        return AgentToolResult(
            {"error": "source evidence is not attached to this search revision"},
            is_error=True,
        )
    try:
        page = store.read_page(
            evidence_id,
            offset=offset,
            max_chars=max_chars,
        )
    except (EvidenceNotFoundError, ValueError) as exc:
        return AgentToolResult({"error": str(exc)}, is_error=True)
    return AgentToolResult(page.as_tool_result())


def _normalized_source_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalized)
