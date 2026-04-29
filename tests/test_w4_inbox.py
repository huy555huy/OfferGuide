"""Inbox queue — enqueue, list, decide, immutability after decision."""

from __future__ import annotations

from pathlib import Path

import pytest

import offerguide
from offerguide import inbox


def _store(tmp_path: Path) -> offerguide.Store:
    s = offerguide.Store(tmp_path / "i.db")
    s.init_schema()
    return s


def test_enqueue_returns_pending_item(tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = inbox.enqueue(
        store,
        kind="consider_jd",
        title="AI Agent @ ByteDance",
        body="prob=0.72",
        payload={"job_id": 1, "score_run_id": 5},
    )
    assert item.id > 0
    assert item.kind == "consider_jd"
    assert item.status == "pending"
    assert item.payload == {"job_id": 1, "score_run_id": 5}


def test_list_filters_by_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = inbox.enqueue(store, kind="consider_jd", title="A")
    inbox.enqueue(store, kind="consider_jd", title="B")
    inbox.decide(store, a.id, decision="approved")

    pending = inbox.list_items(store, status="pending")
    approved = inbox.list_items(store, status="approved")
    assert {i.title for i in pending} == {"B"}
    assert {i.title for i in approved} == {"A"}

    everything = inbox.list_items(store, status=None)
    assert {i.title for i in everything} == {"A", "B"}


def test_list_orders_newest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    inbox.enqueue(store, kind="consider_jd", title="first")
    inbox.enqueue(store, kind="consider_jd", title="second")
    items = inbox.list_items(store, status="pending")
    assert [i.title for i in items] == ["second", "first"]


def test_decide_marks_status_and_timestamp(tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = inbox.enqueue(store, kind="consider_jd", title="X")
    decided = inbox.decide(store, item.id, decision="approved", note="going for it")
    assert decided.status == "approved"
    assert decided.decided_at is not None
    assert decided.decision_note == "going for it"


def test_decide_twice_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    item = inbox.enqueue(store, kind="consider_jd", title="X")
    inbox.decide(store, item.id, decision="approved")
    with pytest.raises(ValueError, match="already decided"):
        inbox.decide(store, item.id, decision="rejected")


def test_decide_unknown_id_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(KeyError):
        inbox.decide(store, 999, decision="approved")


def test_get_returns_none_for_missing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert inbox.get(store, 999) is None
