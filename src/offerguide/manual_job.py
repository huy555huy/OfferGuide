"""One shared fallback for a user-provided job URL or complete JD text."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .memory import Store
from .research_agents.sources import SourceEvidenceStore, SourceReader, SourceScope


class ManualJobIntakeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ManualJobIntakeResult:
    job_id: int
    is_new: bool
    company: str
    title: str
    source_evidence_id: int


def intake_manual_job(
    *,
    store: Store,
    source_store: SourceEvidenceStore,
    url_or_text: str,
    company_hint: str = "",
    title_hint: str = "",
    location_hint: str = "",
    source_url: str | None = None,
    source_reader: SourceReader | None = None,
) -> ManualJobIntakeResult:
    """Preserve one complete user-selected JD and materialize its application identity."""
    value = str(url_or_text or "").strip()
    if not value:
        raise ManualJobIntakeError("url_or_text 不能为空")
    company = str(company_hint or "").strip() or "未指定公司"
    title = str(title_hint or "").strip() or "手动粘贴的岗位"
    location = str(location_hint or "").strip() or None
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    scope = SourceScope(
        run_id=f"manual-job-{digest[:16]}",
        agent_name="manual_job_intake",
        subject_kind="manual_job",
        subject_id=digest,
        subject_revision=0,
    )
    source_store.init_schema()

    if value.startswith(("http://", "https://")):
        owns_reader = source_reader is None
        reader = source_reader or SourceReader(source_store)
        try:
            fetched = reader.fetch(
                value,
                scope=scope,
                purpose="user-selected manual JD",
                title_hint=title,
            )
        finally:
            if owns_reader:
                reader.client.close()
        if fetched.evidence is None:
            raise ManualJobIntakeError(
                fetched.error_text or "无法读取该岗位页面，请直接粘贴完整 JD"
            )
        evidence = fetched.evidence
        raw_text = evidence.text_content
        resolved_url = evidence.final_url
        if title == "手动粘贴的岗位" and evidence.title:
            title = evidence.title
        provenance = "fetched_url"
    else:
        if len(value) < 200:
            raise ManualJobIntakeError("请粘贴完整 JD（至少 200 字）")
        try:
            evidence = source_store.save_user_provided(
                scope=scope,
                text=value,
                title=f"{company} · {title}",
                source_url=source_url,
                purpose="user-selected manual JD",
            )
        except ValueError as exc:
            raise ManualJobIntakeError(str(exc)) from exc
        raw_text = value
        resolved_url = evidence.final_url if source_url else None
        provenance = "user_provided_text"

    job_id, is_new = _materialize_manual_job(
        store=store,
        digest=digest,
        source_evidence_id=evidence.id,
        company=company,
        title=title,
        location=location,
        url=resolved_url,
        raw_text=raw_text,
        provenance=provenance,
    )
    return ManualJobIntakeResult(
        job_id=job_id,
        is_new=is_new,
        company=company,
        title=title,
        source_evidence_id=evidence.id,
    )


def _materialize_manual_job(
    *,
    store: Store,
    digest: str,
    source_evidence_id: int,
    company: str,
    title: str,
    location: str | None,
    url: str | None,
    raw_text: str,
    provenance: str,
) -> tuple[int, bool]:
    extras = json.dumps(
        {
            "source_evidence_id": source_evidence_id,
            "manual_fallback": True,
            "provenance": provenance,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id FROM jobs WHERE source = 'manual' AND source_id = ? "
            "ORDER BY id ASC LIMIT 1",
            (digest,),
        ).fetchone()
        if row is not None:
            job_id = int(row[0])
            conn.execute(
                "UPDATE jobs SET url = ?, title = ?, company = ?, location = ?, "
                "raw_text = ?, extras_json = ?, fetched_at = julianday('now') "
                "WHERE id = ?",
                (url, title, company, location, raw_text, extras, job_id),
            )
            return job_id, False
        job_id = int(
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, location, "
                "raw_text, extras_json, content_hash) "
                "VALUES ('manual', ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (
                    digest,
                    url,
                    title,
                    company,
                    location,
                    raw_text,
                    extras,
                    digest,
                ),
            ).fetchone()[0]
        )
    return job_id, True


__all__ = [
    "ManualJobIntakeError",
    "ManualJobIntakeResult",
    "intake_manual_job",
]
