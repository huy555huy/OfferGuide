"""Atomic revision checks used by domain publication tools."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar


@dataclass(frozen=True, slots=True)
class RevisionSnapshot:
    subject_revision: int
    result_revision: int

    def __post_init__(self) -> None:
        if self.subject_revision < 0 or self.result_revision < 0:
            raise ValueError("revisions must be non-negative")


class StaleRevisionError(RuntimeError):
    def __init__(self, *, expected: RevisionSnapshot, current: RevisionSnapshot) -> None:
        self.expected = expected
        self.current = current
        super().__init__(
            "agent run is stale: expected subject/result revisions "
            f"{expected.subject_revision}/{expected.result_revision}, found "
            f"{current.subject_revision}/{current.result_revision}"
        )


class ConnectionProvider(Protocol):
    def connect(self) -> Any: ...


T = TypeVar("T")
RevisionLoader = Callable[[sqlite3.Connection], RevisionSnapshot]
CASWriter = Callable[[sqlite3.Connection], T]


class RevisionGuard(Generic[T]):
    """Run a domain write only while both bound revisions still match.

    The loader is domain-owned because each agent has a different source of
    truth.  ``compare_and_swap`` takes the SQLite write lock before loading the
    revisions, then performs the caller's write in that same transaction.
    """

    def __init__(self, load_current: RevisionLoader) -> None:
        self._load_current = load_current

    def assert_current(
        self,
        conn: sqlite3.Connection,
        *,
        expected_subject_revision: int,
        expected_result_revision: int,
    ) -> RevisionSnapshot:
        expected = RevisionSnapshot(expected_subject_revision, expected_result_revision)
        current = self._load_current(conn)
        if not isinstance(current, RevisionSnapshot):
            raise TypeError("revision loader must return RevisionSnapshot")
        if current != expected:
            raise StaleRevisionError(expected=expected, current=current)
        return current

    def compare_and_swap(
        self,
        store: ConnectionProvider,
        *,
        expected_subject_revision: int,
        expected_result_revision: int,
        write: CASWriter[T],
    ) -> T:
        with store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.assert_current(
                conn,
                expected_subject_revision=expected_subject_revision,
                expected_result_revision=expected_result_revision,
            )
            return write(conn)
