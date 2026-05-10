"""W13.1 evolution fitness — aggregate signals + detect evolution candidates."""

from __future__ import annotations

import pytest

import offerguide
from offerguide.evolution.fitness import (
    MIN_SIGNALS_FOR_TRIGGER,
    compare_versions,
    compute_fitness,
    detect_evolution_candidates,
    normalize_signal_value,
)
from offerguide.evolution.signals import (
    record_app_outcome,
    record_critic_signal,
    record_user_thumbs,
)


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "evo_fit.db")
    s.init_schema()
    return s


# ═══════════════════════════════════════════════════════════════════
# normalize
# ═══════════════════════════════════════════════════════════════════


class TestNormalize:
    def test_thumbs_neg_one_to_zero(self):
        assert normalize_signal_value("user_thumbs", -1.0) == 0.0
    def test_thumbs_pos_one_to_one(self):
        assert normalize_signal_value("user_thumbs", 1.0) == 1.0
    def test_critic_passthrough(self):
        assert normalize_signal_value("critic", 0.7) == 0.7
    def test_clip_above_one(self):
        assert normalize_signal_value("critic", 1.5) == 1.0
    def test_clip_below_zero(self):
        assert normalize_signal_value("critic", -0.3) == 0.0


# ═══════════════════════════════════════════════════════════════════
# compute_fitness
# ═══════════════════════════════════════════════════════════════════


class TestComputeFitness:
    def test_no_signals_returns_none(self, store):
        r = compute_fitness(store, skill_name="x")
        assert r.fitness is None
        assert r.sample_count == 0
        assert r.by_kind == {}

    def test_single_critic_signal(self, store):
        record_critic_signal(store, skill_name="x", skill_version="v",
                             skill_run_id=None, score=0.8)
        r = compute_fitness(store, skill_name="x")
        assert r.fitness == pytest.approx(0.8)
        assert r.sample_count == 1
        assert "critic" in r.by_kind

    def test_thumbs_normalized_correctly(self, store):
        record_user_thumbs(store, skill_name="x", skill_version="v",
                           skill_run_id=None, thumbs=1)
        record_user_thumbs(store, skill_name="x", skill_version="v",
                           skill_run_id=None, thumbs=-1)
        # Both thumbs equal weight → fitness ≈ 0.5
        r = compute_fitness(store, skill_name="x")
        assert r.fitness == pytest.approx(0.5)
        assert r.sample_count == 2

    def test_weighted_aggregation(self, store):
        # critic weight=1.0, thumbs weight=2.0 (per DEFAULT_WEIGHTS)
        # critic 0.5, thumbs +1 → 1.0 normalized
        # weighted = (0.5*1 + 1.0*2) / (1+2) = 2.5/3 ≈ 0.833
        record_critic_signal(store, skill_name="x", skill_version="v",
                             skill_run_id=None, score=0.5)
        record_user_thumbs(store, skill_name="x", skill_version="v",
                           skill_run_id=None, thumbs=1)
        r = compute_fitness(store, skill_name="x")
        assert r.fitness == pytest.approx(2.5 / 3, abs=0.01)

    def test_filter_by_version(self, store):
        record_critic_signal(store, skill_name="x", skill_version="0.1.0",
                             skill_run_id=None, score=0.4)
        record_critic_signal(store, skill_name="x", skill_version="0.2.0",
                             skill_run_id=None, score=0.9)
        r1 = compute_fitness(store, skill_name="x", skill_version="0.1.0")
        r2 = compute_fitness(store, skill_name="x", skill_version="0.2.0")
        assert r1.fitness == pytest.approx(0.4)
        assert r2.fitness == pytest.approx(0.9)

    def test_app_outcome_negative_pulls_fitness_down(self, store):
        # 5 critic at 0.9 + 5 app_outcome rejections (value=0)
        for _ in range(5):
            record_critic_signal(store, skill_name="x", skill_version="v",
                                 skill_run_id=None, score=0.9)
        for _ in range(5):
            record_app_outcome(store, skill_name="x", skill_version="v",
                               skill_run_id=None, outcome="rejected")
        # critic weight 1.0 * 5 * 0.9 = 4.5
        # app_outcome weight 1.5 * 5 * 0.0 = 0.0
        # total weight = 5 + 7.5 = 12.5
        # fitness = 4.5 / 12.5 = 0.36
        r = compute_fitness(store, skill_name="x")
        assert r.fitness == pytest.approx(0.36, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# detect_evolution_candidates
# ═══════════════════════════════════════════════════════════════════


class TestDetectCandidates:
    def test_no_skills_no_candidates(self, store):
        assert detect_evolution_candidates(store) == []

    def test_skill_with_too_few_signals_skipped(self, store):
        # Only 3 signals, threshold needs 10
        for _ in range(3):
            record_critic_signal(store, skill_name="x", skill_version="0.1.0",
                                 skill_run_id=None, score=0.2)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status) "
                "VALUES (?,?,?,'live')",
                ("x", "0.1.0", "seed body"),
            )
        candidates = detect_evolution_candidates(
            store, min_samples=MIN_SIGNALS_FOR_TRIGGER,
        )
        assert candidates == []

    def test_skill_with_high_fitness_skipped(self, store):
        for _ in range(15):
            record_critic_signal(store, skill_name="x", skill_version="0.1.0",
                                 skill_run_id=None, score=0.9)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status) "
                "VALUES (?,?,?,'live')",
                ("x", "0.1.0", "seed"),
            )
        assert detect_evolution_candidates(store) == []

    def test_low_fitness_with_enough_signals_triggers(self, store):
        # 12 signals all at 0.3 → fitness 0.3 < 0.55 threshold + enough samples
        for _ in range(12):
            record_critic_signal(store, skill_name="bad_skill",
                                 skill_version="0.1.0",
                                 skill_run_id=None, score=0.3)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, status) "
                "VALUES (?,?,?,'live')",
                ("bad_skill", "0.1.0", "seed"),
            )
        candidates = detect_evolution_candidates(store)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.skill_name == "bad_skill"
        assert c.fitness == pytest.approx(0.3)
        assert c.sample_count == 12
        assert "fitness=0.30" in c.reason

    def test_seed_skills_picked_up_via_skill_runs(self, store):
        """SKILL with no skill_variants row should still be detected via skill_runs."""
        # Insert into skill_runs (the seed runs)
        with store.connect() as conn:
            for _ in range(12):
                conn.execute(
                    "INSERT INTO skill_runs(skill_name, skill_version, "
                    "  input_hash, input_json, output_json) VALUES (?,?,?,?,?)",
                    ("seed_skill", "0.1.0", "h", "{}", "{}"),
                )
        for _ in range(12):
            record_critic_signal(store, skill_name="seed_skill",
                                 skill_version="0.1.0",
                                 skill_run_id=None, score=0.4)
        candidates = detect_evolution_candidates(store)
        names = [c.skill_name for c in candidates]
        assert "seed_skill" in names

    def test_cooldown_skips_recently_evolved(self, store):
        for _ in range(15):
            record_critic_signal(store, skill_name="x", skill_version="0.1.0",
                                 skill_run_id=None, score=0.3)
        # Insert a variant CREATED RECENTLY → should be in cooldown
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, "
                "  status, created_at) VALUES (?,?,?,'live', julianday('now') - 2)",
                ("x", "0.1.0", "seed"),
            )
            # And a recent shadow variant
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, parent_version, "
                "  body_md, status, created_at) "
                "VALUES (?,?,?,?,'shadow', julianday('now') - 2)",
                ("x", "0.2.0-shadow", "0.1.0", "evolved"),
            )
        # cooldown=7, last evolution 2 days ago → blocked
        candidates = detect_evolution_candidates(store, cooldown_days=7.0)
        assert candidates == []

    def test_cooldown_passes_when_old_enough(self, store):
        for _ in range(15):
            record_critic_signal(store, skill_name="x", skill_version="0.1.0",
                                 skill_run_id=None, score=0.3)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, body_md, "
                "  status, created_at) VALUES (?,?,?,'live', julianday('now') - 30)",
                ("x", "0.1.0", "seed"),
            )
            # Old shadow variant
            conn.execute(
                "INSERT INTO skill_variants(skill_name, version, parent_version, "
                "  body_md, status, created_at) "
                "VALUES (?,?,?,?,'failed', julianday('now') - 30)",
                ("x", "0.1.5-shadow", "0.1.0", "old failed try"),
            )
        candidates = detect_evolution_candidates(store, cooldown_days=7.0)
        assert len(candidates) == 1
        assert candidates[0].days_since_last_evolution > 7.0


# ═══════════════════════════════════════════════════════════════════
# compare_versions (A/B)
# ═══════════════════════════════════════════════════════════════════


class TestCompareVersions:
    def test_not_enough_data_either_side(self, store):
        record_critic_signal(store, skill_name="x", skill_version="v1",
                             skill_run_id=None, score=0.5)
        record_critic_signal(store, skill_name="x", skill_version="v2",
                             skill_run_id=None, score=0.7)
        out = compare_versions(store, skill_name="x", version_a="v1", version_b="v2",
                                min_signals_per_side=5)
        assert out["winner"] is None
        assert out["decisive"] is False

    def test_b_clearly_wins(self, store):
        for _ in range(6):
            record_critic_signal(store, skill_name="x", skill_version="v1",
                                 skill_run_id=None, score=0.4)
        for _ in range(6):
            record_critic_signal(store, skill_name="x", skill_version="v2",
                                 skill_run_id=None, score=0.85)
        out = compare_versions(store, skill_name="x", version_a="v1", version_b="v2",
                                min_signals_per_side=5)
        assert out["winner"] == "b"
        assert out["decisive"] is True
        assert out["delta"] > 0.4

    def test_a_clearly_wins_returns_a(self, store):
        for _ in range(6):
            record_critic_signal(store, skill_name="x", skill_version="v1",
                                 skill_run_id=None, score=0.9)
        for _ in range(6):
            record_critic_signal(store, skill_name="x", skill_version="v2",
                                 skill_run_id=None, score=0.3)
        out = compare_versions(store, skill_name="x", version_a="v1", version_b="v2",
                                min_signals_per_side=5)
        assert out["winner"] == "a"
        assert out["delta"] < -0.5

    def test_close_call_no_winner(self, store):
        for _ in range(6):
            record_critic_signal(store, skill_name="x", skill_version="v1",
                                 skill_run_id=None, score=0.5)
        for _ in range(6):
            record_critic_signal(store, skill_name="x", skill_version="v2",
                                 skill_run_id=None, score=0.52)
        out = compare_versions(store, skill_name="x", version_a="v1", version_b="v2",
                                min_signals_per_side=5)
        assert out["winner"] is None
        assert out["decisive"] is False
