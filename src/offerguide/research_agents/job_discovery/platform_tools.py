"""Verified platform tools available to JobDiscoveryAgent.

These tools expose existing, actually probed adapters as evidence-producing
actions. The Agent decides whether and when to call them; there is no fixed
platform order or keyword list in this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx

from ...platforms._spec import RawJob
from ...platforms.official_jobs import search_verified_official_jobs
from ...platforms.shixiseng import (
    DETAIL_URL_TEMPLATE,
    fetch_detail,
    fetch_list_tokens,
    to_raw_job,
)
from ..runner import AgentExecutionContext, AgentTool, AgentToolResult
from ..sources import SourceEvidenceStore, SourceScope
from .models import platform_job_reference
from .quality import ThinJobDescriptionError, require_substantive_job_description
from .source_tools import PLATFORM_DERIVED_JOB_PURPOSE, SOURCE_STORE_DEPENDENCY


def build_platform_job_tools() -> tuple[AgentTool, ...]:
    return (
        AgentTool(
            name="search_verified_official_jobs",
            description=(
                "Search the verified public recruitment adapters for the model-chosen "
                "keyword and optional company. Successful results already contain full "
                "official normalized JD evidence and an opaque platform_job_ref. Record a "
                "chosen result by passing only that ref to record_verified_platform_job."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "minLength": 1},
                    "company": {"type": ["string", "null"]},
                    "limit_per_source": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
            handler=_search_official,
        ),
        AgentTool(
            name="search_shixiseng_jobs",
            description=(
                "Search one model-chosen 实习僧 keyword/page and read each selected "
                "detail page into complete job evidence. Use for internship discovery; "
                "results with missing JD bodies are reported but not verified. Record a "
                "chosen result by passing only its ref to record_verified_platform_job."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "minLength": 1},
                    "page": {"type": "integer", "minimum": 1, "maximum": 20},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["keyword"],
                "additionalProperties": False,
            },
            handler=_search_shixiseng,
        ),
    )


def _scope(execution: AgentExecutionContext) -> SourceScope:
    return SourceScope(
        run_id=execution.run_id,
        agent_name=execution.agent_name,
        subject_kind=execution.subject.subject_kind,
        subject_id=execution.subject.subject_id,
        subject_revision=execution.subject.subject_revision,
    )


def _search_official(args: Mapping[str, Any], execution: AgentExecutionContext) -> AgentToolResult:
    keyword = str(args.get("keyword") or "").strip()
    if not keyword:
        return AgentToolResult({"error": "keyword must not be blank"}, is_error=True)
    raw_company = args.get("company")
    company = str(raw_company).strip() if raw_company is not None else None
    company = company or None
    try:
        limit = _bounded_integer(args, "limit_per_source", default=8, minimum=1, maximum=20)
    except ValueError as exc:
        return AgentToolResult({"error": str(exc)}, is_error=True)
    evidence_store: SourceEvidenceStore = execution.dependency(SOURCE_STORE_DEPENDENCY)
    scope = _scope(execution)
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as base_client:
            client = _RecordingClient(base_client)
            results = search_verified_official_jobs(
                company=company,
                keyword=keyword,
                limit=limit,
                client=cast(httpx.Client, client),
            )
    except Exception as exc:
        evidence_store.record_search(
            scope=scope,
            backend="verified_official",
            query=_query_label(keyword, company),
            status="failed",
            results=(),
            error_text=f"{type(exc).__name__}: {exc}",
        )
        return AgentToolResult(
            {"error": f"verified official search failed: {type(exc).__name__}: {exc}"},
            is_error=True,
        )

    saved: list[dict[str, Any]] = []
    source_attempts: list[dict[str, Any]] = []
    successful_attempts = 0
    for result in results:
        clues = [
            {"title": job.title, "url": job.url or "", "snippet": job.company or ""}
            for job in result.jobs
            if job.url
        ]
        saved_for_source = 0
        rejected_details: list[str] = []
        for raw_job in result.jobs:
            item = _save_platform_job(
                evidence_store,
                scope,
                raw_job,
                source_response=client.response_for(raw_job),
            )
            if item is not None:
                saved.append(item)
                saved_for_source += 1
                continue
            error = (
                f"{raw_job.source_id or raw_job.title}: detail response is missing "
                "identity fields or a substantive JD body"
            )
            rejected_details.append(error)
            _record_platform_detail_failure(
                evidence_store,
                scope,
                url=str(raw_job.url or result.evidence_url),
                error_code="incomplete_job_detail",
                error_text=error,
            )

        if result.status == "ok" and saved_for_source:
            status = "succeeded"
        elif (
            result.status == "ok"
            and not result.jobs
            and "detail_errors=" not in result.note
        ):
            status = "empty"
        else:
            status = "failed"
        if status in {"succeeded", "empty"}:
            successful_attempts += 1
        error_parts = []
        if status == "failed" or "detail_errors=" in result.note:
            error_parts.append(result.note)
        error_parts.extend(rejected_details)
        error_text = " | ".join(error_parts) or None
        evidence_store.record_search(
            scope=scope,
            backend=result.source,
            query=_query_label(keyword, company),
            status=status,
            results=clues,
            error_text=error_text,
        )
        source_attempts.append(
            {
                "source": result.source,
                "status": status,
                "adapter_status": result.status,
                "note": result.note,
                "jobs": len(result.jobs),
                "verified_jobs": saved_for_source,
                "detail_errors": rejected_details,
            }
        )

    return AgentToolResult(
        {
            "query": _query_label(keyword, company),
            "source_attempts": source_attempts,
            "verified_jobs": saved,
            "instruction": (
                "Each verified job already includes the adapter-owned complete normalized "
                "JD. Record a selected result with record_verified_platform_job using only "
                "its returned platform_job_ref; do not repeat source or identity fields."
            ),
        },
        is_error=successful_attempts == 0,
    )


def _search_shixiseng(args: Mapping[str, Any], execution: AgentExecutionContext) -> AgentToolResult:
    keyword = str(args.get("keyword") or "").strip()
    if not keyword:
        return AgentToolResult({"error": "keyword must not be blank"}, is_error=True)
    try:
        page = _bounded_integer(args, "page", default=1, minimum=1, maximum=20)
        limit = _bounded_integer(args, "limit", default=8, minimum=1, maximum=20)
    except ValueError as exc:
        return AgentToolResult({"error": str(exc)}, is_error=True)
    evidence_store: SourceEvidenceStore = execution.dependency(SOURCE_STORE_DEPENDENCY)
    scope = _scope(execution)
    saved: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as base_client:
            client = _RecordingClient(base_client)
            tokens = fetch_list_tokens(
                keyword,
                page=page,
                client=cast(httpx.Client, client),
            )[:limit]
            evidence_store.record_search(
                scope=scope,
                backend="shixiseng",
                query=f"{keyword} page={page}",
                status="succeeded" if tokens else "empty",
                results=[
                    {
                        "title": token,
                        "url": DETAIL_URL_TEMPLATE.format(token=token),
                        "snippet": "detail token",
                    }
                    for token in tokens
                ],
            )
            for token in tokens:
                try:
                    parsed = fetch_detail(token, client=cast(httpx.Client, client))
                    if parsed is None or not parsed.body.strip():
                        error = f"{token}: detail page has no readable JD body"
                        errors.append(error)
                        _record_platform_detail_failure(
                            evidence_store,
                            scope,
                            url=DETAIL_URL_TEMPLATE.format(token=token),
                            error_code="empty_job_detail",
                            error_text=error,
                        )
                        continue
                    saved_item = _save_platform_job(
                        evidence_store,
                        scope,
                        to_raw_job(parsed, discovered_keyword=keyword),
                        source_response=client.response_for_url(parsed.url),
                    )
                    if saved_item is not None:
                        saved.append(saved_item)
                    else:
                        error = (
                            f"{token}: detail page is missing job identity fields "
                            "or a substantive JD body"
                        )
                        errors.append(error)
                        _record_platform_detail_failure(
                            evidence_store,
                            scope,
                            url=parsed.url,
                            error_code="incomplete_job_detail",
                            error_text=error,
                        )
                except Exception as exc:
                    error = f"{token}: {type(exc).__name__}: {exc}"
                    errors.append(error)
                    _record_platform_detail_failure(
                        evidence_store,
                        scope,
                        url=DETAIL_URL_TEMPLATE.format(token=token),
                        error_code="platform_detail_error",
                        error_text=error,
                    )
    except Exception as exc:
        evidence_store.record_search(
            scope=scope,
            backend="shixiseng",
            query=f"{keyword} page={page}",
            status="failed",
            results=(),
            error_text=f"{type(exc).__name__}: {exc}",
        )
        return AgentToolResult(
            {"error": f"shixiseng search failed: {type(exc).__name__}: {exc}"},
            is_error=True,
        )

    return AgentToolResult(
        {
            "query": f"{keyword} page={page}",
            "verified_jobs": saved,
            "detail_errors": errors,
            "instruction": (
                "Each verified job already includes the adapter-owned complete normalized "
                "JD. Record a selected result with record_verified_platform_job using only "
                "its returned platform_job_ref; do not repeat source or identity fields."
            ),
        },
        is_error=bool(tokens and not saved),
    )


def _save_platform_job(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
    raw_job: RawJob,
    *,
    source_response: _CapturedResponse | None,
) -> dict[str, Any] | None:
    url = str(raw_job.url or "").strip()
    company = str(raw_job.company or "").strip()
    title = str(raw_job.title or "").strip()
    jd_text = str(raw_job.raw_text or "").strip()
    if not url or not company or not title or not jd_text or source_response is None:
        return None
    try:
        require_substantive_job_description(
            jd_text,
            identity_values=(company, title, raw_job.location),
        )
    except ThinJobDescriptionError:
        return None
    evidence, status, _fetch_id = evidence_store.save_response(
        scope=scope,
        requested_url=source_response.requested_url,
        final_url=source_response.final_url,
        title=f"{company} · {title}",
        media_type=source_response.media_type,
        charset=source_response.charset,
        raw_content=source_response.raw_content,
        text_content=source_response.text_content,
        http_status=source_response.http_status,
        purpose=PLATFORM_DERIVED_JOB_PURPOSE,
        provenance="web",
    )
    page_time = []
    for key in ("refresh_date", "project_type", "projectName", "lastmod"):
        value = raw_job.extras.get(key)
        if value:
            page_time.append(f"{key}: {value}")
    item = {
        "evidence_status": status,
        "evidence_kind": "platform_adapter",
        "source_evidence_id": evidence.id,
        "source_document_url": evidence.final_url,
        "source_name": raw_job.source,
        "source_job_id": raw_job.source_id,
        "canonical_url": url,
        "company": company,
        "title": title,
        "location": raw_job.location,
        "recruitment_type": (
            raw_job.extras.get("project_type")
            or raw_job.extras.get("projectName")
            or raw_job.extras.get("recruit_type")
        ),
        "page_time_information": page_time,
        "jd_text": jd_text,
        "source_status": "open",
    }
    item["platform_job_ref"] = platform_job_reference(
        source_evidence_id=evidence.id,
        source_name=raw_job.source,
        source_job_id=raw_job.source_id,
        canonical_url=url,
    )
    return item


@dataclass(frozen=True, slots=True)
class _CapturedResponse:
    requested_url: str
    final_url: str
    media_type: str
    charset: str | None
    raw_content: bytes
    text_content: str
    http_status: int | None


class _RecordingClient:
    """Transparent adapter client that keeps the exact responses parsers consumed."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.responses: list[_CapturedResponse] = []

    def get(self, url: str, **kwargs: Any) -> Any:
        response = self.client.get(url, **kwargs)
        self._record(url, kwargs.get("params"), response)
        return response

    def post(self, url: str, **kwargs: Any) -> Any:
        response = self.client.post(url, **kwargs)
        self._record(url, kwargs.get("params"), response)
        return response

    def response_for(self, raw_job: RawJob) -> _CapturedResponse | None:
        desired = str(
            raw_job.extras.get("detail_evidence_url")
            or raw_job.extras.get("evidence_url")
            or raw_job.url
            or ""
        ).strip()
        matches = [
            item
            for item in self.responses
            if _without_query(item.requested_url) == _without_query(desired)
        ]
        source_id = str(raw_job.source_id or "").strip()
        if source_id:
            identified = [
                item
                for item in matches
                if source_id in item.requested_url or source_id.encode("utf-8") in item.raw_content
            ]
            if identified:
                return identified[-1]
        return matches[-1] if matches else None

    def response_for_url(self, url: str) -> _CapturedResponse | None:
        canonical = _without_query(url)
        matches = [
            item for item in self.responses if _without_query(item.requested_url) == canonical
        ]
        return matches[-1] if matches else None

    def _record(self, url: str, params: Any, response: Any) -> None:
        requested_url = _url_with_params(url, params)
        response_url = getattr(response, "url", None)
        final_url = str(response_url) if response_url else requested_url
        headers = getattr(response, "headers", {}) or {}
        content_type = str(headers.get("content-type") or "").strip()
        media_type = content_type.partition(";")[0]
        media_type = media_type.strip().lower() or _response_media_type(response)
        charset = _content_type_charset(content_type)
        raw_content = _response_bytes(response)
        resolved_charset = charset or str(getattr(response, "encoding", "") or "utf-8")
        self.responses.append(
            _CapturedResponse(
                requested_url=requested_url,
                final_url=final_url,
                media_type=media_type,
                charset=resolved_charset,
                raw_content=raw_content,
                text_content=_response_text(response, raw_content, resolved_charset),
                http_status=getattr(response, "status_code", None),
            )
        )


def _response_bytes(response: Any) -> bytes:
    raw = getattr(response, "content", None)
    if isinstance(raw, bytes) and raw:
        return raw
    text = str(getattr(response, "text", "") or "")
    if text:
        return text.encode("utf-8")
    try:
        return json.dumps(response.json(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    except Exception as exc:
        raise ValueError("platform response body is unavailable") from exc


def _response_media_type(response: Any) -> str:
    try:
        response.json()
    except Exception:
        return "text/html"
    return "application/json"


def _content_type_charset(content_type: str) -> str | None:
    for parameter in content_type.split(";")[1:]:
        name, separator, value = parameter.partition("=")
        if separator and name.strip().lower() == "charset":
            return value.strip().strip('"') or None
    return None


def _response_text(response: Any, raw_content: bytes, charset: str) -> str:
    text = str(getattr(response, "text", "") or "")
    if text:
        return text
    try:
        return raw_content.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return raw_content.decode("utf-8", errors="replace")


def _url_with_params(url: str, params: Any) -> str:
    if not isinstance(params, Mapping) or not params:
        return url
    parts = urlsplit(url)
    query = urlencode([(str(key), str(value)) for key, value in params.items()])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _without_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _query_label(keyword: str, company: str | None) -> str:
    return f"{company + ' ' if company else ''}{keyword}".strip()


def _bounded_integer(
    args: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw = args.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _record_platform_detail_failure(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
    *,
    url: str,
    error_code: str,
    error_text: str,
) -> None:
    evidence_store.record_fetch_failure(
        scope=scope,
        requested_url=url,
        final_url=url,
        status="failed",
        error_code=error_code,
        error_text=error_text,
    )


__all__ = ["build_platform_job_tools"]
