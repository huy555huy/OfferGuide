from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from offerguide.interview_research import init_interview_research_schema
from offerguide.memory import Store

RETIRED_TABLES = {
    "post_apply_materials",
    "interview_experiences",
    "company_briefs",
    "user_keywords",
}
RETIRED_DERIVED_TABLES = {"daemon_runs"}
RETIRED_CONTROL_TABLES = {"agent_self_notes"}


def _create_legacy_tables(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        for table in RETIRED_TABLES:
            conn.execute(f'CREATE TABLE "{table}" (payload TEXT)')
        conn.execute('CREATE TABLE "daemon_runs" (payload TEXT)')
        conn.execute(
            'INSERT INTO "daemon_runs" (payload) VALUES (?)',
            ("obsolete scheduler telemetry",),
        )
        conn.execute('CREATE TABLE "agent_self_notes" (body TEXT NOT NULL)')
        conn.execute(
            'INSERT INTO "agent_self_notes" (body) VALUES (?)',
            ("obsolete agent-authored continuation instruction",),
        )
        conn.executescript(
            """
            CREATE TRIGGER trg_post_apply_requires_submitted_workspace_insert
            BEFORE INSERT ON post_apply_materials
            BEGIN
                SELECT 1;
            END;
            CREATE TRIGGER trg_post_apply_requires_submitted_workspace_update
            BEFORE UPDATE ON post_apply_materials
            BEGIN
                SELECT 1;
            END;
            """
        )


def _tables(store: Store) -> set[str]:
    with store.connect() as conn:
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }


def _triggers(store: Store) -> set[str]:
    with store.connect() as conn:
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }


def test_fresh_schema_does_not_create_retired_tables(tmp_path: Path) -> None:
    store = Store(tmp_path / "fresh.db")

    store.init_schema()

    assert RETIRED_TABLES.isdisjoint(_tables(store))
    assert RETIRED_DERIVED_TABLES.isdisjoint(_tables(store))
    assert RETIRED_CONTROL_TABLES.isdisjoint(_tables(store))
    assert RETIRED_TABLES.isdisjoint(store.health_check())
    assert "interviews" in store.health_check()


def test_empty_retired_tables_are_removed_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _create_legacy_tables(path)
    store = Store(path)

    store.init_schema()
    store.init_schema()

    assert RETIRED_TABLES.isdisjoint(_tables(store))
    assert RETIRED_DERIVED_TABLES.isdisjoint(_tables(store))
    assert RETIRED_CONTROL_TABLES.isdisjoint(_tables(store))
    assert not any(name.startswith("trg_post_apply_") for name in _triggers(store))


def test_legacy_interview_prep_revisions_are_replaced_by_answer_sets(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "legacy-interview.db")
    store.init_schema()
    with store.connect() as conn:
        conn.execute(
            "CREATE TABLE interview_prep_material_revisions("
            "id INTEGER PRIMARY KEY, material_json TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO interview_prep_material_revisions(material_json) VALUES (?)",
            ('{"blocks":[{"evidence_class":"inference"}]}',),
        )

    init_interview_research_schema(store)
    init_interview_research_schema(store)

    tables = _tables(store)
    assert "interview_prep_material_revisions" not in tables
    assert "interview_answer_set_revisions" in tables
    with store.connect() as conn:
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(interview_answer_set_revisions)"
            ).fetchall()
        }
    assert "answer_set_json" in columns
    assert "material_json" not in columns


@pytest.mark.parametrize("populated_table", sorted(RETIRED_TABLES))
def test_nonempty_retired_table_blocks_all_destructive_migration(
    tmp_path: Path,
    populated_table: str,
) -> None:
    path = tmp_path / "legacy.db"
    _create_legacy_tables(path)
    with sqlite3.connect(path) as conn:
        conn.execute(
            f'INSERT INTO "{populated_table}" (payload) VALUES (?)',
            ("preserve me",),
        )
    store = Store(path)

    with pytest.raises(
        RuntimeError,
        match=rf"refusing to drop retired tables.*{populated_table}=1",
    ):
        store.init_schema()

    tables = _tables(store)
    assert tables >= RETIRED_TABLES
    assert tables >= RETIRED_DERIVED_TABLES
    assert "jobs" not in tables
    assert any(name.startswith("trg_post_apply_") for name in _triggers(store))
    with store.connect() as conn:
        assert conn.execute(
            f'SELECT payload FROM "{populated_table}"'
        ).fetchone() == ("preserve me",)
