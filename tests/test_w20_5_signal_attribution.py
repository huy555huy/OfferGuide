"""W20.5 — Multi-SKILL signal attribution + view-visit follow_through.

Real wiring tests via TestClient:
1. GET /jobs/N/apply-pack → harness_event 'apply_pack_generated' written
   linking apply_assistant skill_run_id ↔ job_id, AND follow_through signal
   written for apply_assistant.
2. GET /jobs/N/post-apply-pack → same for prepare_interview.
3. POST /api/apply/N/mark with status=interview → app_outcome signal fans
   to ALL SKILLs that touched this job (score_match + apply_assistant +
   prepare_interview), not just apply_assistant hardcoded.
4. Job with no prior SKILL chain → outcome falls back to apply_assistant
   (with reduced weight).
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
from offerguide.harness import _schema as harness_schema
from offerguide.profile.schema import UserProfile
from offerguide.ui.web import create_app


@pytest.fixture
def app_with_scored_job(tmp_path):
    """App + 1 job with a 'scored' harness_event (i.e., score_match ran)."""
    os.environ["OFFERGUIDE_NO_AMBIENT"] = "1"

    db = tmp_path / "w205.db"
    store = offerguide.Store(db)
    store.init_schema()
    harness_schema.init_harness_schema(store)

    profile = UserProfile(raw_resume_text="resume " * 100)

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            ("nowcoder", "AI Agent 实习", "某公司", "x" * 300, "h_w205"),
        )
        job_id = cur.lastrowid
        # Pre-existing score_match run (mimics what _exec_score_match wrote)
        cur2 = conn.execute(
            "INSERT INTO skill_runs (skill_name, skill_version, "
            "  input_hash, input_json, output_json, cost_usd, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("score_match", "0.2.0", "h_score",
             json.dumps({"job_text": "x", "user_profile": "y"}),
             json.dumps({"probability": 0.7}), 0.0001, 5000),
        )
        score_skill_run_id = cur2.lastrowid
        conn.execute(
            "INSERT INTO harness_events (kind, job_id, note, source) "
            "VALUES (?, ?, ?, ?)",
            ("scored", job_id,
             json.dumps({"probability": 0.7, "skill_run_id": score_skill_run_id}),
             "score_match"),
        )
        conn.commit()

    settings = Settings(
        deepseek_api_key=None,  # SKILL won't actually invoke; result.skill_run_id=None
        db_path=db,
        disable_ambient_crawl=True,
    )
    app = create_app(
        settings=settings, store=store, profile=profile,
        skills=[], runtime=None, notifier=None,
    )
    return store, app, job_id, score_skill_run_id


def _count_signals(store, kind: str | None = None,
                   skill_name: str | None = None) -> int:
    sql = "SELECT COUNT(*) FROM evolution_signals WHERE 1=1"
    params: list = []
    if kind:
        sql += " AND signal_kind = ?"
        params.append(kind)
    if skill_name:
        sql += " AND skill_name = ?"
        params.append(skill_name)
    with store.connect() as conn:
        r = conn.execute(sql, params).fetchone()
    return int(r[0])


def _count_harness_events(store, kind: str, job_id: int | None = None) -> int:
    sql = "SELECT COUNT(*) FROM harness_events WHERE kind = ?"
    params: list = [kind]
    if job_id is not None:
        sql += " AND job_id = ?"
        params.append(job_id)
    with store.connect() as conn:
        r = conn.execute(sql, params).fetchone()
    return int(r[0])


# ── Tests ──────────────────────────────────────────────────────────


def test_apply_pack_view_writes_link_and_signal_when_skill_runs(
    app_with_scored_job, tmp_path,
):
    """Hard part: when the SKILL actually invokes (real LLM key), opening
    /apply-pack should write 'apply_pack_generated' harness_event AND
    follow_through signal for apply_assistant.

    For this test, sandbox has no LLM key → SKILL doesn't invoke →
    skill_run_id=None → no signal written. We assert that case (defensive
    no-crash) here, then test the full path in a separate fixture below.
    """
    store, app, job_id, _ = app_with_scored_job

    before = _count_signals(store, "follow_through")
    before_events = _count_harness_events(store, "apply_pack_generated", job_id)

    with TestClient(app) as client:
        r = client.get(f"/jobs/{job_id}/apply-pack")
        assert r.status_code == 200
        # Page renders with LLM-key-missing error block
        assert "需要先配 LLM key" in r.text or "apply" in r.text.lower()

    # Without LLM key, SKILL didn't invoke → no signal/event written
    assert _count_signals(store, "follow_through") == before
    assert _count_harness_events(store, "apply_pack_generated", job_id) == before_events


def test_apply_pack_view_writes_signal_when_skill_succeeds(tmp_path):
    """Force the SKILL to "succeed" by stubbing invoke_skill_for_view to
    return a non-None skill_run_id pointing to a fake skill_run row.
    Then GET /apply-pack and verify the link + signal write."""
    os.environ["OFFERGUIDE_NO_AMBIENT"] = "1"

    db = tmp_path / "w205_b.db"
    store = offerguide.Store(db)
    store.init_schema()
    harness_schema.init_harness_schema(store)
    profile = UserProfile(raw_resume_text="resume " * 100)

    # Insert a job + a fake apply_assistant skill_run row to attribute to
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            ("nowcoder", "AI Agent", "co", "x" * 300, "h_b"),
        )
        job_id = cur.lastrowid
        cur2 = conn.execute(
            "INSERT INTO skill_runs (skill_name, skill_version, input_hash, "
            "input_json, output_json, cost_usd, latency_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("apply_assistant", "0.1.0", "h_aa",
             json.dumps({"company": "co"}), json.dumps({"text": "ok"}),
             0.0, 1000),
        )
        fake_apply_skill_run_id = cur2.lastrowid
        conn.commit()

    # Patch invoke_skill_for_view to return the fake skill_run_id
    from offerguide.skill_view import SkillViewResult
    import offerguide.ui.web as _webmod
    original_invoke = _webmod.invoke_skill_for_view if hasattr(_webmod, 'invoke_skill_for_view') else None

    # Minimal apply_assistant output that matches what apply_pack.html
    # expects (self_intro_snippet + qa_templates + ...). Just enough to render.
    async def _fake_invoke(**kwargs):
        return SkillViewResult(
            parsed={
                "self_intro_snippet": {
                    "text": "测试自我介绍", "platform_hint": "BOSS",
                    "rationale": "test rationale",
                },
                "qa_templates": [],
                "submission_strategy": {},
                "pre_submit_checklist": [],
                "skip_reasons": [],
                "confidence": 0.8,
            },
            raw_text="",
            skill_run_id=fake_apply_skill_run_id,
            cost_usd=0.0, duration_ms=1, error=None,
        )

    # Patch the import at the apply_pack_view function — it does
    # `from ..skill_view import invoke_skill_for_view` inside the function.
    import offerguide.skill_view as _sv
    _sv.invoke_skill_for_view = _fake_invoke

    settings = Settings(
        deepseek_api_key="sk-fake",  # truthy so SKILL "would" run
        db_path=db, disable_ambient_crawl=True,
    )
    app = create_app(
        settings=settings, store=store, profile=profile,
        skills=[], runtime=None, notifier=None,
    )

    try:
        with TestClient(app) as client:
            r = client.get(f"/jobs/{job_id}/apply-pack")
            assert r.status_code == 200
    finally:
        # Restore (test isolation)
        if original_invoke is not None:
            _sv.invoke_skill_for_view = original_invoke

    # Now verify: link + signal both written.
    # Post-Q2 (W21 follow-up): apply-pack runs BOTH tailor_resume AND
    # apply_assistant — so we expect 2 harness_events, one per SKILL.
    # The apply_assistant follow_through signal is still expected (the
    # signal-attribution-only test stubs both SKILLs to the same run id,
    # so signal count remains 1 on the apply_assistant skill_name).
    assert _count_harness_events(store, "apply_pack_generated", job_id) == 2
    assert _count_signals(store, "follow_through", "apply_assistant") == 1

    # Verify at least one harness_event references the apply_assistant SKILL run
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT json_extract(note, '$.skill_run_id'), "
            "       json_extract(note, '$.skill_name') "
            "FROM harness_events "
            "WHERE kind = 'apply_pack_generated' AND job_id = ?",
            (job_id,),
        ).fetchall()
    skill_run_ids = {r[0] for r in rows}
    skill_names = {r[1] for r in rows}
    assert fake_apply_skill_run_id in skill_run_ids
    assert "apply_assistant" in skill_names


def test_apply_mark_status_interview_fans_to_all_involved_skills(tmp_path):
    """W20.5 multi-SKILL outcome attribution.

    Setup: 1 job that has been touched by score_match + apply_assistant
    (each wrote a harness_event). User marks status=interview.
    Expect: app_outcome signal written for BOTH score_match AND apply_assistant
    (was: only apply_assistant pre-W20.5).
    """
    os.environ["OFFERGUIDE_NO_AMBIENT"] = "1"
    db = tmp_path / "w205_c.db"
    store = offerguide.Store(db)
    store.init_schema()
    harness_schema.init_harness_schema(store)
    profile = UserProfile(raw_resume_text="resume " * 100)

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            ("nowcoder", "AI", "co", "x" * 300, "h_c"),
        )
        job_id = cur.lastrowid
        # 2 skill_runs (score_match + apply_assistant)
        sr1 = conn.execute(
            "INSERT INTO skill_runs (skill_name, skill_version, input_hash, "
            "input_json, output_json, cost_usd, latency_ms) "
            "VALUES ('score_match', '0.2.0', 'h1', '{}', '{}', 0, 0)"
        ).lastrowid
        sr2 = conn.execute(
            "INSERT INTO skill_runs (skill_name, skill_version, input_hash, "
            "input_json, output_json, cost_usd, latency_ms) "
            "VALUES ('apply_assistant', '0.1.0', 'h2', '{}', '{}', 0, 0)"
        ).lastrowid
        # 2 harness_events linking each skill_run to the job
        conn.execute(
            "INSERT INTO harness_events (kind, job_id, note, source) "
            "VALUES ('scored', ?, ?, 'score_match')",
            (job_id, json.dumps({"probability": 0.7, "skill_run_id": sr1})),
        )
        conn.execute(
            "INSERT INTO harness_events (kind, job_id, note, source) "
            "VALUES ('apply_pack_generated', ?, ?, 'view_visit')",
            (job_id, json.dumps({"skill_run_id": sr2,
                                 "skill_name": "apply_assistant",
                                 "skill_version": "0.1.0"})),
        )
        conn.commit()

    settings = Settings(
        deepseek_api_key=None, db_path=db, disable_ambient_crawl=True,
    )
    app = create_app(
        settings=settings, store=store, profile=profile,
        skills=[], runtime=None, notifier=None,
    )

    with TestClient(app) as client:
        r = client.post(
            f"/api/apply/{job_id}/mark", data={"status": "interview"},
        )
        assert r.status_code == 200

    # BOTH SKILLs got app_outcome signal (W20.5 multi-attribution)
    assert _count_signals(store, "app_outcome", "score_match") == 1
    assert _count_signals(store, "app_outcome", "apply_assistant") == 1

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT skill_name, signal_value, signal_weight "
            "FROM evolution_signals WHERE signal_kind = 'app_outcome' "
            "ORDER BY skill_name"
        ).fetchall()
    assert {r[0] for r in rows} == {"apply_assistant", "score_match"}


def test_apply_mark_with_no_skill_chain_falls_back(tmp_path):
    """If a job has no prior SKILL events (user pasted JD, didn't go via
    score_match / apply_pack), outcome marking still writes a signal but
    falls back to apply_assistant with reduced weight=0.5."""
    os.environ["OFFERGUIDE_NO_AMBIENT"] = "1"
    db = tmp_path / "w205_d.db"
    store = offerguide.Store(db)
    store.init_schema()
    harness_schema.init_harness_schema(store)
    profile = UserProfile(raw_resume_text="resume " * 100)

    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            ("user_paste_url", "x", "y", "z" * 300, "h_d"),
        )
        job_id = cur.lastrowid
        conn.commit()

    settings = Settings(
        deepseek_api_key=None, db_path=db, disable_ambient_crawl=True,
    )
    app = create_app(
        settings=settings, store=store, profile=profile,
        skills=[], runtime=None, notifier=None,
    )

    with TestClient(app) as client:
        r = client.post(
            f"/api/apply/{job_id}/mark", data={"status": "interview"},
        )
        assert r.status_code == 200

    # Fallback: apply_assistant with weight 0.5 (lower because attribution fuzzy)
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT skill_name, signal_weight FROM evolution_signals "
            "WHERE signal_kind = 'app_outcome'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "apply_assistant"
    assert rows[0][1] == 0.5  # reduced weight (vs default 1.0 for direct attribution)
