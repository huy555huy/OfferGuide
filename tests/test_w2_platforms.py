"""Tests for the source-neutral job payload shared by current adapters."""

from __future__ import annotations

from offerguide.platforms import RawJob, canonical_text, content_hash


def test_canonical_text_includes_label_lines() -> None:
    rj = RawJob(
        source="manual",
        title="数据科学实习生",
        company="ByteDance",
        location="Shanghai",
        raw_text="负责数据 ETL",
    )
    txt = canonical_text(rj)
    assert "标题: 数据科学实习生" in txt
    assert "公司: ByteDance" in txt
    assert "地点: Shanghai" in txt
    assert "负责数据 ETL" in txt


def test_content_hash_stable_for_same_inputs() -> None:
    a = RawJob(source="x", title="t", raw_text="body")
    b = RawJob(source="x", title="t", raw_text="body")
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_when_text_changes() -> None:
    a = RawJob(source="x", title="t", raw_text="body")
    b = RawJob(source="x", title="t", raw_text="body!")
    assert content_hash(a) != content_hash(b)
