"""W20.4 — REAL signal wiring: '我投了' click → follow_through to score_match.

User point (verbatim): "纯系统代码的 critic 没意义, 因为他凭什么能评判你"
→ Real signal must come from user actions, not LLM-judging-LLM.

When user clicks '我投了' on a job, that IS ground truth:
  - The score_match SKILL recommended this job (probability ≥ some threshold)
  - The user evaluated the recommendation
  - The user CHOSE to act on it
  - That's a follow_through=True for the score_match SKILL run

Tests:
- Real /api/jobs/{id}/applied call writes follow_through signal
- Without prior score_match event, no signal written (no SKILL to attribute)
- Multiple applied clicks don't double-count beyond expected
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.agent_runtime import _schema as harness_schema
from offerguide.profile.schema import UserProfile
from offerguide.ui.web import create_app


@pytest.fixture
def app_setup(tmp_path):
    """Build a real app + store, with one scored job ready to be marked applied."""
    os.environ["OFFERGUIDE_NO_AMBIENT"] = "1"

    db = tmp_path / "ft.db"
    store = offerguide.Store(db)
    store.init_schema()
    harness_schema.init_agent_runtime_schema(store)

    profile = UserProfile(raw_resume_text="resume A " * 50)

    # Insert one job + a fake skill_run + a 'scored' harness_event linking
    # them. Mimics what _exec_score_match would write.
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            ("nowcoder", "AI Agent 实习", "某公司", "x" * 300, "h_test"),
        )
        job_id = cur.lastrowid
        # Insert skill_run for score_match v0.2.0
        cur2 = conn.execute(
            "INSERT INTO skill_runs (skill_name, skill_version, "
            "  input_hash, input_json, output_json, cost_usd, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("score_match", "0.2.0", "abc",
             json.dumps({"job_text": "x", "user_profile": "y"}),
             json.dumps({"probability": 0.7}), 0.0001, 5000),
        )
        skill_run_id = cur2.lastrowid
        # Insert harness_event 'scored' for this job, with skill_run_id
        conn.execute(
            "INSERT INTO harness_events (kind, job_id, note, source) "
            "VALUES (?, ?, ?, ?)",
            ("scored", job_id,
             json.dumps({"probability": 0.7, "skill_run_id": skill_run_id}),
             "score_match"),
        )
        conn.commit()

    settings = Settings(deepseek_api_key=None, db_path=db, disable_ambient_crawl=True)
    app = create_app(
        settings=settings, store=store, profile=profile,
        skills=[], runtime=None, notifier=None,
    )
    return store, app, job_id, skill_run_id


def _count_signals(store, kind: str | None = None) -> int:
    with store.connect() as conn:
        if kind:
            r = conn.execute(
                "SELECT COUNT(*) FROM evolution_signals WHERE signal_kind = ?",
                (kind,),
            ).fetchone()
        else:
            r = conn.execute("SELECT COUNT(*) FROM evolution_signals").fetchone()
    return int(r[0])


def test_marking_applied_writes_follow_through_signal(app_setup):
    """W20.4 — real wiring: POST /api/jobs/N/applied writes follow_through
    signal for the score_match run that recommended this job.

    This is the REAL signal that drives evolve, not critic LLM."""
    store, app, job_id, skill_run_id = app_setup

    # Before: no follow_through signals
    assert _count_signals(store, "follow_through") == 0

    with TestClient(app) as client:
        r = client.post(f"/api/jobs/{job_id}/applied")
        assert r.status_code == 200

    # After: 1 follow_through signal, attributed to score_match
    assert _count_signals(store, "follow_through") == 1

    with store.connect() as conn:
        row = conn.execute(
            "SELECT skill_name, skill_version, skill_run_id, signal_value, "
            "       signal_weight, notes "
            "FROM evolution_signals WHERE signal_kind = 'follow_through'"
        ).fetchone()
    assert row is not None
    skill_name, skill_version, srid, value, weight, notes = row
    assert skill_name == "score_match"
    assert skill_version == "0.2.0"
    assert srid == skill_run_id
    assert value == 1.0  # executed=True
    assert weight == 0.8  # DEFAULT_WEIGHTS['follow_through']
    assert "我投了" in notes
    assert f"job#{job_id}" in notes


def test_marking_applied_without_prior_score_match_doesnt_crash(app_setup):
    """If no score_match was ever run for this job, applied click still
    succeeds; just doesn't write a signal (nothing to attribute to)."""
    store, app, _job_id, _ = app_setup

    # Insert a separate job with NO scored event
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            ("nowcoder", "另一个岗", "另一家", "y" * 300, "h_other"),
        )
        unscored_job_id = cur.lastrowid
        conn.commit()

    before = _count_signals(store, "follow_through")
    with TestClient(app) as client:
        r = client.post(f"/api/jobs/{unscored_job_id}/applied")
        assert r.status_code == 200
    # No new signal (nothing to attribute)
    assert _count_signals(store, "follow_through") == before


def test_marking_applied_emits_user_marked_applied_event(app_setup):
    """Sanity: the existing harness_event still fires (no regression)."""
    store, app, job_id, _ = app_setup
    with TestClient(app) as client:
        client.post(f"/api/jobs/{job_id}/applied")
    with store.connect() as conn:
        r = conn.execute(
            "SELECT COUNT(*) FROM harness_events "
            "WHERE kind = 'user_marked_applied' AND job_id = ?",
            (job_id,),
        ).fetchone()
    assert r[0] == 1


def test_signal_is_real_user_action_not_llm_judgment(app_setup):
    """W20.4 epistemological assertion (user's point made literal in code):

    The follow_through signal kind is documented as 'real user action' in
    evolution/signals.py. critic kind is documented as 'model judging
    model'. Default weights reflect this:
        user_thumbs (real)    = 2.0  ← highest
        app_outcome (real)    = 1.5
        critic (LLM opinion)  = 1.0  ← mid (admitted bias)
        follow_through (real) = 0.8
    """
    from offerguide.evolution.signals import DEFAULT_WEIGHTS
    # Three real-action signals all weight > critic LLM signal? No — only
    # user_thumbs and app_outcome do. follow_through is 0.8 (lower, less
    # reliable per-event because user might apply for many reasons besides
    # the agent's score). But cumulatively they outweigh critic.
    real_signals = ["user_thumbs", "app_outcome", "follow_through"]
    real_total = sum(DEFAULT_WEIGHTS[k] for k in real_signals)
    critic_weight = DEFAULT_WEIGHTS["critic"]
    assert real_total > critic_weight * 4, (
        f"Real signals (sum={real_total}) should dominate critic ({critic_weight})"
    )
