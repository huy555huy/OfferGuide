"""sqlite-vec integration — single-user local-first vector storage.

Why sqlite-vec over Qdrant: see ATTRIBUTION.md. Short version: zero deployment
+ zero dependencies + lives next to our relational tables in the same SQLite
file. Trade-off: brute-force only (no ANN), so this won't scale to millions of
vectors — fine for one user's job-history corpus.

The vec0 virtual table requires a fixed embedding dimension. We don't pick the
embedding model until W2, so this module exposes `init_vec_schema(conn, dim)`
to be called once the dim is known. W1's `init_schema()` does NOT create vec
tables — only verifies that the extension loads.
"""

from __future__ import annotations

import sqlite3

import sqlite_vec


def attach_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into an open connection."""
    try:
        conn.enable_load_extension(True)
    except (sqlite3.NotSupportedError, AttributeError) as e:
        raise RuntimeError(
            "Your Python's sqlite3 was built without extension loading. "
            "On macOS Anaconda this usually means installing Python via pyenv "
            "or Homebrew instead. See https://github.com/asg017/sqlite-vec/issues "
            "for workarounds."
        ) from e
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def vec_version(conn: sqlite3.Connection) -> str:
    """Return the loaded sqlite-vec version, e.g. 'v0.1.9'."""
    (v,) = conn.execute("SELECT vec_version()").fetchone()
    return v


def init_vec_schema(conn: sqlite3.Connection, *, embedding_dim: int) -> None:
    """Create the vector tables. Call once `embedding_dim` is known (W2+).

    Two tables, joined by rowid:

    - `vec_embeddings` (vec0 virtual): the actual float vectors, brute-force searched
    - `vec_metadata` (regular): which entity each rowid refers to (job / resume_chunk / etc.)
    """
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
            embedding float[{embedding_dim}]
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vec_metadata (
            rowid      INTEGER PRIMARY KEY,
            kind       TEXT NOT NULL,
            ref_id     INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            created_at REAL DEFAULT (julianday('now'))
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vec_meta_kind ON vec_metadata(kind, ref_id)")
