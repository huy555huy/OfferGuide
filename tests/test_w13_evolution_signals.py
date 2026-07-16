"""W13.1 evolution_signals — write API + read filters."""

from __future__ import annotations

import pytest

import offerguide
from offerguide.evolution.signals import (
    fetch_signals,
    record_app_outcome,
    record_critic_signal,
    record_follow_through,
    record_synthetic_eval,
    record_user_thumbs,
)


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "evo.db")
    s.init_schema()
    return s


class TestRecordCritic:
    def test_basic_insert(self, store):
        sid = record_critic_signal(
            store, skill_name="example_skill", skill_version="0.1.0",
            skill_run_id=42, score=0.85, notes="trajectory clean",
        )
        assert sid is not None and sid > 0
        recs = fetch_signals(store, skill_name="example_skill")
        assert len(recs) == 1
        assert recs[0].signal_kind == "critic"
        assert recs[0].signal_value == 0.85
        assert recs[0].skill_run_id == 42

    def test_clips_score_to_unit_interval(self, store):
        record_critic_signal(store, skill_name="x", skill_version="v",
                             skill_run_id=None, score=1.5)
        record_critic_signal(store, skill_name="x", skill_version="v",
                             skill_run_id=None, score=-0.3)
        recs = fetch_signals(store, skill_name="x")
        values = sorted(r.signal_value for r in recs)
        assert values == [0.0, 1.0]

    def test_none_score_skipped(self, store):
        sid = record_critic_signal(store, skill_name="x", skill_version="v",
                                    skill_run_id=None, score=None)
        assert sid is None
        assert fetch_signals(store, skill_name="x") == []


class TestRecordUserThumbs:
    def test_thumbs_stored_verbatim(self, store):
        record_user_thumbs(store, skill_name="x", skill_version="v",
                           skill_run_id=1, thumbs=1)
        record_user_thumbs(store, skill_name="x", skill_version="v",
                           skill_run_id=2, thumbs=-1)
        recs = fetch_signals(store, skill_name="x")
        values = sorted(r.signal_value for r in recs)
        assert values == [-1.0, 1.0]

    def test_invalid_thumbs_raises(self, store):
        with pytest.raises(ValueError):
            record_user_thumbs(store, skill_name="x", skill_version="v",
                               skill_run_id=None, thumbs=0)  # type: ignore[arg-type]


class TestRecordAppOutcome:
    def test_positive_outcomes_score_one(self, store):
        for o in ("offer", "interview", "screening"):
            record_app_outcome(store, skill_name="x", skill_version="v",
                               skill_run_id=None, outcome=o)
        recs = fetch_signals(store, skill_name="x")
        assert all(r.signal_value == 1.0 for r in recs)
        assert all(r.signal_kind == "app_outcome" for r in recs)

    def test_negative_outcomes_score_zero(self, store):
        for o in ("rejected", "silent"):
            record_app_outcome(store, skill_name="x", skill_version="v",
                               skill_run_id=None, outcome=o)
        recs = fetch_signals(store, skill_name="x")
        assert all(r.signal_value == 0.0 for r in recs)

    def test_unknown_outcome_raises(self, store):
        with pytest.raises(ValueError, match="unknown outcome"):
            record_app_outcome(store, skill_name="x", skill_version="v",
                               skill_run_id=None, outcome="invalid")  # type: ignore[arg-type]

    def test_custom_weight(self, store):
        record_app_outcome(store, skill_name="x", skill_version="v",
                           skill_run_id=None, outcome="offer", weight=0.3)
        recs = fetch_signals(store, skill_name="x")
        assert recs[0].signal_weight == 0.3


class TestRecordFollowThrough:
    def test_executed_vs_ignored(self, store):
        record_follow_through(store, skill_name="x", skill_version="v",
                              skill_run_id=1, executed=True)
        record_follow_through(store, skill_name="x", skill_version="v",
                              skill_run_id=2, executed=False)
        recs = fetch_signals(store, skill_name="x")
        values = sorted(r.signal_value for r in recs)
        assert values == [0.0, 1.0]


class TestRecordSyntheticEval:
    def test_for_candidate_variant(self, store):
        record_synthetic_eval(store, skill_name="example_skill",
                              skill_version="0.2.0-shadow", score=0.78,
                              notes="meta_evolve_skill iter 3 of 5")
        recs = fetch_signals(store, skill_name="example_skill")
        assert len(recs) == 1
        assert recs[0].signal_kind == "eval_synthetic"
        assert recs[0].skill_version == "0.2.0-shadow"
        assert recs[0].signal_value == 0.78
        assert recs[0].skill_run_id is None  # synthetic, no real run


class TestFetchSignals:
    def test_filter_by_kind(self, store):
        record_critic_signal(store, skill_name="x", skill_version="v",
                             skill_run_id=None, score=0.5)
        record_user_thumbs(store, skill_name="x", skill_version="v",
                           skill_run_id=None, thumbs=1)
        critic = fetch_signals(store, skill_name="x", kind="critic")
        thumbs = fetch_signals(store, skill_name="x", kind="user_thumbs")
        assert len(critic) == 1 and critic[0].signal_kind == "critic"
        assert len(thumbs) == 1 and thumbs[0].signal_kind == "user_thumbs"

    def test_filter_by_version(self, store):
        record_critic_signal(store, skill_name="x", skill_version="0.1.0",
                             skill_run_id=None, score=0.4)
        record_critic_signal(store, skill_name="x", skill_version="0.2.0",
                             skill_run_id=None, score=0.8)
        v1 = fetch_signals(store, skill_name="x", skill_version="0.1.0")
        v2 = fetch_signals(store, skill_name="x", skill_version="0.2.0")
        assert len(v1) == 1 and v1[0].signal_value == 0.4
        assert len(v2) == 1 and v2[0].signal_value == 0.8

    def test_filter_by_recency(self, store):
        # Insert 2 signals, one with old created_at by direct SQL
        record_critic_signal(store, skill_name="x", skill_version="v",
                             skill_run_id=None, score=0.5)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO evolution_signals(skill_name, skill_version, "
                "  signal_kind, signal_value, signal_weight, created_at) "
                "VALUES (?,?,?,?,?, julianday('now') - 30)",
                ("x", "v", "critic", 0.3, 1.0),
            )
        # only-recent should return just the new one
        recent = fetch_signals(
            store, skill_name="x",
            since_julianday=None,  # no filter, get all
        )
        assert len(recent) == 2

        # Now filter to last 7 days
        with store.connect() as conn:
            cutoff = conn.execute(
                "SELECT julianday('now') - 7"
            ).fetchone()[0]
        recent_only = fetch_signals(
            store, skill_name="x", since_julianday=cutoff,
        )
        assert len(recent_only) == 1

    def test_returns_empty_when_no_signals(self, store):
        assert fetch_signals(store, skill_name="nothing_here") == []
