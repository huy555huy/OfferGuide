"""Scout worker tests — write into the jobs table, dedup, no network."""

from __future__ import annotations

from pathlib import Path

import offerguide
from offerguide.platforms import RawJob, manual
from offerguide.workers import scout


def test_ingest_inserts_new_job(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rj = manual.from_text("Backend SWE\n岗位职责...", company="ByteDance")

    was_new, job_id = scout.ingest(store, rj)
    assert was_new is True
    assert job_id > 0

    with store.connect() as conn:
        rows = conn.execute("SELECT title, company, source FROM jobs").fetchall()
    assert rows == [("Backend SWE", "ByteDance", "manual")]


def test_ingest_is_idempotent_on_same_content(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rj = manual.from_text("X\n岗位职责...", company="A")

    new1, id1 = scout.ingest(store, rj)
    new2, id2 = scout.ingest(store, rj)

    assert new1 is True
    assert new2 is False
    assert id1 == id2
    with store.connect() as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
    assert n == 1


def test_ingest_distinguishes_different_sources(tmp_path: Path) -> None:
    """Same title+text but different `source` must be allowed (UNIQUE is per-source)."""
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()

    a = RawJob(source="manual", title="X", raw_text="body")
    b = RawJob(source="nowcoder", title="X", raw_text="body", source_id="999")

    assert scout.ingest(store, a)[0] is True
    assert scout.ingest(store, b)[0] is True

    with store.connect() as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
    assert n == 2


def test_ingest_persists_extras_in_structured_column(tmp_path: Path) -> None:
    """W5' fix: extras live in jobs.extras_json (queryable), NOT concatenated to raw_text."""
    import json

    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rj = RawJob(
        source="nowcoder",
        title="t",
        raw_text="body",
        extras={"salaryMin": 15, "salaryMax": 35, "avgProcessRate": 38},
    )
    scout.ingest(store, rj)
    with store.connect() as conn:
        text, extras_json = conn.execute(
            "SELECT raw_text, extras_json FROM jobs"
        ).fetchone()
    # raw_text stays clean — no extras pollution
    assert text == "body"
    assert "[extras]" not in text
    assert "salaryMin" not in text
    # extras_json is queryable JSON
    extras = json.loads(extras_json)
    assert extras["salaryMin"] == 15
    assert extras["avgProcessRate"] == 38


def test_ingest_extras_queryable_via_json_extract(tmp_path: Path) -> None:
    """The structured column means SQL can filter / aggregate on platform fields."""
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    scout.ingest(
        store,
        RawJob(
            source="nowcoder",
            title="hot",
            raw_text="body1",
            extras={"avgProcessRate": 50},
        ),
    )
    scout.ingest(
        store,
        RawJob(
            source="nowcoder",
            title="cold",
            raw_text="body2",
            extras={"avgProcessRate": 10},
        ),
    )
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT title FROM jobs "
            "WHERE CAST(json_extract(extras_json, '$.avgProcessRate') AS INTEGER) >= 30"
        ).fetchall()
    assert rows == [("hot",)]


def test_ingest_handles_empty_extras(tmp_path: Path) -> None:
    """RawJob.extras = {} should still produce a valid extras_json row."""
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    scout.ingest(store, RawJob(source="manual", title="t", raw_text="body"))
    with store.connect() as conn:
        (extras_json,) = conn.execute("SELECT extras_json FROM jobs").fetchone()
    assert extras_json == "{}"
