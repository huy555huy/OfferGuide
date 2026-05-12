"""User-managed search keywords.

The ambient discovery loop derives keywords from the resume + active goal
via ``match_keywords.extract_keywords``. That works for the common case
but doesn't let the user say "actually also look for 'RAG 工程师'" or
"never show me '量化交易' jobs". This module is the persistence layer
behind the ``/recommended`` keyword chips UI:

- ``list_keywords(store)`` → ``(includes, excludes)`` lists
- ``add_keyword(store, ...)`` / ``remove_keyword(store, kid)`` — UI ops
- ``apply_user_excludes(text, store)`` — used by /recommended filter

``ambient.py`` consults includes when assembling the per-cycle keyword
list (user includes outrank vocab matches). Excludes apply at /recommended
render time so users can iterate without re-crawling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .memory import Store

log = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_keywords (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword     TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('include', 'exclude')),
    created_at  REAL NOT NULL DEFAULT (julianday('now')),
    UNIQUE(keyword, kind)
);
CREATE INDEX IF NOT EXISTS idx_user_keywords_kind ON user_keywords(kind);
"""


def _ensure_schema(store: Store) -> None:
    with store.connect() as conn:
        conn.executescript(_SCHEMA)


@dataclass(frozen=True)
class UserKeyword:
    id: int
    keyword: str
    kind: str  # 'include' | 'exclude'


def list_keywords(store: Store) -> tuple[list[UserKeyword], list[UserKeyword]]:
    """Return ``(includes, excludes)`` ordered by created_at desc."""
    _ensure_schema(store)
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT id, keyword, kind FROM user_keywords ORDER BY created_at DESC",
        ).fetchall()
    includes = [UserKeyword(int(r[0]), r[1], r[2]) for r in rows if r[2] == "include"]
    excludes = [UserKeyword(int(r[0]), r[1], r[2]) for r in rows if r[2] == "exclude"]
    return includes, excludes


def add_keyword(store: Store, *, keyword: str, kind: str) -> tuple[bool, int]:
    """Insert one keyword. Returns ``(was_new, id)``.

    Duplicates (same keyword+kind) are silently no-op: the existing row's
    id is returned with ``was_new=False``. Empty / overlong inputs raise.
    """
    if kind not in ("include", "exclude"):
        raise ValueError(f"kind must be 'include' or 'exclude', got {kind!r}")
    cleaned = (keyword or "").strip()
    if not cleaned:
        raise ValueError("keyword must be non-empty")
    if len(cleaned) > 80:
        raise ValueError("keyword too long (max 80 chars)")

    _ensure_schema(store)
    with store.connect() as conn:
        existing = conn.execute(
            "SELECT id FROM user_keywords WHERE keyword = ? AND kind = ?",
            (cleaned, kind),
        ).fetchone()
        if existing:
            return False, int(existing[0])
        cur = conn.execute(
            "INSERT INTO user_keywords(keyword, kind) VALUES (?, ?) RETURNING id",
            (cleaned, kind),
        )
        return True, int(cur.fetchone()[0])


def remove_keyword(store: Store, kid: int) -> bool:
    """Delete by id. Returns True if a row was removed."""
    _ensure_schema(store)
    with store.connect() as conn:
        cur = conn.execute("DELETE FROM user_keywords WHERE id = ?", (kid,))
        return cur.rowcount > 0


def matches_exclude(text: str, excludes: list[UserKeyword]) -> str | None:
    """Return the first excluded keyword that appears in ``text``, or None.

    Substring match (case-insensitive). Used at /recommended render time
    so excluded keywords filter live without re-crawling.
    """
    lower = (text or "").lower()
    for x in excludes:
        if x.keyword.lower() in lower:
            return x.keyword
    return None
