"""Interview-experience corpus — the 面经 RAG store fed to `prepare_interview`.

Sources of 面经 supported (today / planned):

- `manual_paste`     — user pastes a 面经 they read elsewhere; W5 supports this
- `nowcoder_discuss` — crawled from `/discuss/<id>` URLs in nowcoder's main
                       sitemap; W7 will wire the crawler. The schema is ready.
- `1point3acres`     — placeholder for an offline-format export; not implemented

Storage shape mirrors `jobs`: source-scoped UNIQUE on content_hash so the same
discussion can't be ingested twice. Retrieval is keyword-on-company, ordered by
recency — good enough at our corpus size (will grow to thousands not millions).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .memory import Store


@dataclass(frozen=True)
class InterviewExperience:
    id: int
    company: str
    role_hint: str | None
    raw_text: str
    source: str
    source_url: str | None
    content_hash: str
    created_at: float


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def insert(
    store: Store,
    *,
    company: str,
    raw_text: str,
    source: str,
    role_hint: str | None = None,
    source_url: str | None = None,
) -> tuple[bool, int]:
    """Insert a 面经. Returns (was_new, id). Idempotent on (source, content_hash)."""
    if not company.strip():
        raise ValueError("company is required")
    if not raw_text.strip():
        raise ValueError("raw_text is required")
    h = _hash(raw_text)
    with store.connect() as conn:
        existing = conn.execute(
            "SELECT id FROM interview_experiences WHERE source = ? AND content_hash = ?",
            (source, h),
        ).fetchone()
        if existing:
            return False, int(existing[0])
        cur = conn.execute(
            "INSERT INTO interview_experiences(company, role_hint, raw_text, source, "
            "source_url, content_hash) VALUES (?,?,?,?,?,?)",
            (company.strip(), role_hint, raw_text.strip(), source, source_url, h),
        )
        return True, int(cur.lastrowid or 0)


def fetch_for_company(
    store: Store, company: str, *, limit: int = 5
) -> list[InterviewExperience]:
    """Return the most recent 面经 for `company`, up to `limit`. Case-insensitive prefix match."""
    if not company.strip():
        return []
    pattern = f"%{company.strip()}%"
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, company, role_hint, raw_text, source, source_url, content_hash, created_at "
            "FROM interview_experiences WHERE company LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (pattern, limit),
        ).fetchall()
    return [InterviewExperience(*r) for r in rows]


def render_snippets(experiences: list[InterviewExperience], *, max_chars: int = 4000) -> str:
    """Concatenate experiences into a labelled blob suitable for the LLM prompt.

    Returns empty string if `experiences` is empty — the SKILL prompt knows how
    to handle "no past experiences" gracefully.
    """
    if not experiences:
        return ""
    out: list[str] = []
    used = 0
    for i, e in enumerate(experiences, start=1):
        block = f"### 面经 {i} · {e.company}{f' · {e.role_hint}' if e.role_hint else ''}\n{e.raw_text.strip()}\n"
        if used + len(block) > max_chars:
            out.append(f"…（还有 {len(experiences) - i + 1} 篇面经因长度限制省略）")
            break
        out.append(block)
        used += len(block)
    return "\n".join(out).strip()
