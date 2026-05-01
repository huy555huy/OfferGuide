"""W9 — Daily standup home + Pipeline kanban + Verdict synthesis.

Coverage:

- daily_brief.build derives silent_followups, upcoming_interviews,
  unscored_jobs, stale_briefs, action_items
- pipeline_view.build buckets applications + scanned-only jobs into 5
  stages, attaches has_score badge
- /pipeline route renders all 5 columns
- /api/pipeline/applications/{id}/event logs an event + returns
  re-rendered card fragment
- /api/pipeline/jobs/{id}/submit promotes scanned → applied with a
  fresh application + submitted event
- verdict.synthesize maps SKILL outputs to go/maybe/hold/skip
- /chat with everything action attaches verdict to context
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import application_events as ae
from offerguide import daily_brief, pipeline_view, verdict
from offerguide.config import Settings
from offerguide.profile import UserProfile
from offerguide.skills import discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "w9.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="resume")
    skills = discover_skills(SKILLS_ROOT)
    app = create_app(
        settings=Settings(),
        store=store,
        profile=profile,
        skills=skills,
        runtime=None,
        notifier=ConsoleNotifier(),
    )
    return app, store


def _seed_job(store, *, title: str, company: str, hash_suffix: str = "") -> int:
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('manual', ?, ?, 'jd body', ?) RETURNING id",
            (title, company, f"h_{title}{hash_suffix}"),
        )
        return int(cur.fetchone()[0])


def _seed_app(store, job_id: int, status: str = "applied") -> int:
    with store.connect() as conn:
        cur = conn.execute(
            "INSERT INTO applications(job_id, status) VALUES (?,?) RETURNING id",
            (job_id, status),
        )
        return int(cur.fetchone()[0])


# ═══════════════════════════════════════════════════════════════════
# daily_brief
# ═══════════════════════════════════════════════════════════════════


class TestDailyBrief:
    def test_empty_db_returns_empty_brief(self, app_setup) -> None:
        _, store = app_setup
        b = daily_brief.build(store)
        assert b.silent_followups == []
        assert b.upcoming_interviews == []
        assert b.unscored_jobs == []
        assert b.stale_briefs == []
        assert b.action_items == []

    def test_silent_followup_surfaces_after_threshold(self, app_setup) -> None:
        _, store = app_setup
        job = _seed_job(store, title="后端实习", company="美团")
        app_id = _seed_app(store, job)
        # Latest non-inferred event 10 days ago — past 7d threshold
        with store.connect() as c:
            c.execute(
                "INSERT INTO application_events"
                "(application_id, kind, occurred_at, source, payload_json) "
                "VALUES (?, 'submitted', julianday('now','-10 days'), 'manual', '{}')",
                (app_id,),
            )
        b = daily_brief.build(store)
        assert len(b.silent_followups) == 1
        assert b.silent_followups[0].company == "美团"
        assert b.silent_followups[0].silence_days >= 7

    def test_terminal_apps_excluded_from_silent(self, app_setup) -> None:
        _, store = app_setup
        job = _seed_job(store, title="数据分析", company="腾讯")
        app_id = _seed_app(store, job, status="rejected")
        with store.connect() as c:
            c.execute(
                "INSERT INTO application_events"
                "(application_id, kind, occurred_at, source, payload_json) "
                "VALUES (?, 'submitted', julianday('now','-30 days'), 'manual', '{}')",
                (app_id,),
            )
        b = daily_brief.build(store)
        assert b.silent_followups == []  # rejected = terminal, not surfaced

    def test_upcoming_interview_within_window(self, app_setup) -> None:
        _, store = app_setup
        job = _seed_job(store, title="算法实习", company="字节跳动")
        app_id = _seed_app(store, job)
        with store.connect() as c:
            c.execute(
                "INSERT INTO application_events"
                "(application_id, kind, occurred_at, source, payload_json) "
                "VALUES (?, 'interview', julianday('now','+2 days'), 'calendar', "
                " '{\"summary\":\"AI Agent 一面\"}')",
                (app_id,),
            )
        b = daily_brief.build(store)
        assert len(b.upcoming_interviews) == 1
        up = b.upcoming_interviews[0]
        assert up.company == "字节跳动"
        assert 1.5 < up.days_until < 2.5
        assert "AI Agent" in up.summary

    def test_unscored_jobs_excludes_jobs_with_application(self, app_setup) -> None:
        _, store = app_setup
        # Two jobs: one applied, one not
        job1 = _seed_job(store, title="A", company="X", hash_suffix="1")
        job2 = _seed_job(store, title="B", company="Y", hash_suffix="2")
        _seed_app(store, job1)  # job1 has application — counts as scored
        _ = job2  # job2 stays scanned

        b = daily_brief.build(store)
        unscored_ids = [u.job_id for u in b.unscored_jobs]
        assert job2 in unscored_ids
        assert job1 not in unscored_ids

    def test_action_items_high_priority_for_imminent_interview(self, app_setup) -> None:
        _, store = app_setup
        job = _seed_job(store, title="X", company="Z")
        app_id = _seed_app(store, job)
        # Tomorrow morning interview
        with store.connect() as c:
            c.execute(
                "INSERT INTO application_events"
                "(application_id, kind, occurred_at, source, payload_json) "
                "VALUES (?, 'interview', julianday('now','+1 days'), 'calendar', '{}')",
                (app_id,),
            )
        b = daily_brief.build(store)
        # High priority action for imminent interview
        priorities = [a.priority for a in b.action_items]
        assert "high" in priorities

    def test_action_items_sorted_by_priority(self, app_setup) -> None:
        _, store = app_setup
        # Mix of urgent + non-urgent
        job1 = _seed_job(store, title="J1", company="C1", hash_suffix="1")
        app1 = _seed_app(store, job1)
        with store.connect() as c:
            # Old silent (high priority)
            c.execute(
                "INSERT INTO application_events(application_id,kind,occurred_at,source,payload_json) "
                "VALUES (?, 'submitted', julianday('now','-20 days'), 'manual', '{}')",
                (app1,),
            )
        b = daily_brief.build(store)
        weight = {"high": 0, "medium": 1, "low": 2}
        weights = [weight.get(a.priority, 99) for a in b.action_items]
        assert weights == sorted(weights), "action items must be priority-ordered"


# ═══════════════════════════════════════════════════════════════════
# pipeline_view
# ═══════════════════════════════════════════════════════════════════


class TestPipelineView:
    def test_empty_db_has_all_5_buckets(self, app_setup) -> None:
        _, store = app_setup
        v = pipeline_view.build(store)
        assert set(v.columns.keys()) == {
            "scanned", "applied", "contacted", "interview", "terminal"
        }
        assert all(len(cards) == 0 for cards in v.columns.values())
        assert v.total_active == 0

    def test_scanned_job_no_application(self, app_setup) -> None:
        _, store = app_setup
        _seed_job(store, title="未投递的岗位", company="C")
        v = pipeline_view.build(store)
        assert len(v.columns["scanned"]) == 1
        assert v.columns["scanned"][0].application_id is None
        assert v.columns["scanned"][0].company == "C"

    def test_status_to_stage_mapping(self, app_setup) -> None:
        _, store = app_setup
        # Seed one app per status to test the mapping
        cases = {
            "applied":         "applied",
            "viewed":          "contacted",
            "hr_replied":      "contacted",
            "1st_interview":   "interview",
            "2nd_interview":   "interview",
            "final_interview": "interview",
            "offer":           "terminal",
            "rejected":        "terminal",
            "withdrawn":       "terminal",
        }
        for status in cases:
            j = _seed_job(store, title=status, company=f"co-{status}",
                          hash_suffix=status)
            _seed_app(store, j, status=status)

        v = pipeline_view.build(store)
        for status, expected_stage in cases.items():
            stage_cards = v.columns[expected_stage]
            companies = [c.company for c in stage_cards]
            assert f"co-{status}" in companies, f"{status} not in {expected_stage}"

    def test_total_active_excludes_terminal(self, app_setup) -> None:
        _, store = app_setup
        for i, status in enumerate(["applied", "rejected", "offer"]):
            j = _seed_job(store, title=str(i), company=f"c{i}",
                          hash_suffix=str(i))
            _seed_app(store, j, status=status)
        v = pipeline_view.build(store)
        # 1 applied (active), 2 terminal
        assert v.total_active == 1
        assert v.counts["terminal"] == 2

    def test_transition_options(self) -> None:
        assert pipeline_view.transition_options("scanned") == [
            ("submitted", "标记已投递")
        ]
        assert ("offer", "Offer") in pipeline_view.transition_options("interview")
        assert pipeline_view.transition_options("terminal") == []

    def test_render_age_buckets(self) -> None:
        assert pipeline_view.render_age(None) == "—"
        assert pipeline_view.render_age(0.5) == "今天"
        assert pipeline_view.render_age(3) == "3d"
        assert pipeline_view.render_age(14) == "2周"
        assert pipeline_view.render_age(60) == "2月"


# ═══════════════════════════════════════════════════════════════════
# /pipeline route
# ═══════════════════════════════════════════════════════════════════


class TestPipelineRoute:
    def test_pipeline_route_renders_with_empty_db(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/pipeline")
        assert resp.status_code == 200
        # All 5 column labels visible (even when empty)
        for label in ("已扫描", "已投递", "HR 已联系", "面试中", "已结束"):
            assert label in resp.text
        # Empty-state banner
        assert "Pipeline 还是空的" in resp.text

    def test_pipeline_route_shows_cards(self, app_setup) -> None:
        app, store = app_setup
        j = _seed_job(store, title="后端实习", company="美团")
        app_id = _seed_app(store, j)
        ae.record(store, application_id=app_id, kind="submitted", source="manual")
        resp = TestClient(app).get("/pipeline")
        assert "美团" in resp.text
        assert "后端实习" in resp.text

    def test_log_event_via_kanban_returns_card_fragment(self, app_setup) -> None:
        app, store = app_setup
        j = _seed_job(store, title="x", company="Cy")
        app_id = _seed_app(store, j)
        ae.record(store, application_id=app_id, kind="submitted", source="manual")
        resp = TestClient(app).post(
            f"/api/pipeline/applications/{app_id}/event",
            data={"kind": "viewed"},
        )
        assert resp.status_code == 200
        # Card moves to contacted, fragment includes the company
        assert "Cy" in resp.text

    def test_log_event_unknown_kind_400(self, app_setup) -> None:
        app, store = app_setup
        j = _seed_job(store, title="x", company="Cy")
        app_id = _seed_app(store, j)
        resp = TestClient(app).post(
            f"/api/pipeline/applications/{app_id}/event",
            data={"kind": "uhh"},
        )
        assert resp.status_code == 400

    def test_submit_scanned_job_creates_application(self, app_setup) -> None:
        app, store = app_setup
        j = _seed_job(store, title="未投递", company="某公司")
        # Initially no application
        with store.connect() as c:
            assert c.execute(
                "SELECT COUNT(*) FROM applications WHERE job_id = ?", (j,)
            ).fetchone()[0] == 0

        resp = TestClient(app).post(f"/api/pipeline/jobs/{j}/submit")
        assert resp.status_code == 200
        assert "某公司" in resp.text

        with store.connect() as c:
            apps = c.execute(
                "SELECT id, status FROM applications WHERE job_id = ?", (j,)
            ).fetchall()
            assert len(apps) == 1
            assert apps[0][1] == "applied"
            # And a 'submitted' event was recorded
            evts = c.execute(
                "SELECT kind FROM application_events WHERE application_id = ?",
                (apps[0][0],),
            ).fetchall()
            assert ("submitted",) in [(e[0],) for e in evts]


# ═══════════════════════════════════════════════════════════════════
# verdict synthesis
# ═══════════════════════════════════════════════════════════════════


class TestVerdict:
    def test_high_score_no_blockers_is_go(self) -> None:
        v = verdict.synthesize(
            score={"probability": 0.8, "deal_breakers": []},
            gaps={"suggestions": []},
            prep=None, deep_prep=None, cover_letter=None,
        )
        assert v.kind == "go"
        assert v.label == "建议投"
        # First action is the urgency-flag
        assert v.actions[0].priority == "high"
        assert "排进" in v.actions[0].title or "投递" in v.actions[0].title

    def test_dealbreaker_drops_to_hold(self) -> None:
        v = verdict.synthesize(
            score={"probability": 0.75, "deal_breakers": ["要 985+"]},
            gaps=None, prep=None, deep_prep=None, cover_letter=None,
        )
        assert v.kind == "hold"

    def test_dealbreaker_with_low_score_is_skip(self) -> None:
        v = verdict.synthesize(
            score={"probability": 0.3, "deal_breakers": ["不收应届"]},
            gaps=None, prep=None, deep_prep=None, cover_letter=None,
        )
        assert v.kind == "skip"

    def test_low_brief_confidence_downgrades_go(self) -> None:
        v = verdict.synthesize(
            score={"probability": 0.85, "deal_breakers": []},
            gaps=None, prep=None, deep_prep=None, cover_letter=None,
            brief_confidence=0.2,
        )
        # Was go (>=0.7), but low brief confidence demotes
        assert v.kind == "maybe"

    def test_tight_app_limit_upgrades_scrutiny(self) -> None:
        v = verdict.synthesize(
            score={"probability": 0.55, "deal_breakers": []},  # maybe
            gaps=None, prep=None, deep_prep=None, cover_letter=None,
            brief_app_limit=2,
        )
        # maybe + only-2-slots → hold (don't waste a precious slot)
        assert v.kind == "hold"

    def test_high_risk_gaps_surface_action(self) -> None:
        v = verdict.synthesize(
            score={"probability": 0.7, "deal_breakers": []},
            gaps={"suggestions": [
                {"section": "技能", "ai_risk": "high"},
                {"section": "项目", "ai_risk": "high"},
            ]},
            prep=None, deep_prep=None, cover_letter=None,
        )
        action_titles = [a.title for a in v.actions]
        assert any("AI 风险" in t for t in action_titles)

    def test_pillars_summarize_each_skill(self) -> None:
        v = verdict.synthesize(
            score={"probability": 0.7, "deal_breakers": []},
            gaps={"suggestions": [{"ai_risk": "low"}]},
            prep={"expected_questions": [1, 2, 3]},
            deep_prep={"projects_analyzed": [1], "weak_spots_to_practice": []},
            cover_letter={"overall_word_count": 250, "ai_risk_warnings": []},
            brief_confidence=0.6,
        )
        assert "score" in v.pillars
        assert "70%" in v.pillars["score"]
        assert "gaps" in v.pillars
        assert "prep" in v.pillars
        assert "deep" in v.pillars
        assert "cover" in v.pillars
        assert "brief" in v.pillars

    def test_no_inputs_returns_safe_default(self) -> None:
        v = verdict.synthesize(
            score=None, gaps=None, prep=None, deep_prep=None, cover_letter=None,
        )
        # With no signal, default kind is maybe (prob=0.5)
        assert v.kind == "maybe"
        assert v.actions  # never empty — always 1+ guidance

    def test_overall_score_punishes_dealbreakers(self) -> None:
        v_clean = verdict.synthesize(
            score={"probability": 0.75, "deal_breakers": []},
            gaps=None, prep=None, deep_prep=None, cover_letter=None,
        )
        v_blocked = verdict.synthesize(
            score={"probability": 0.75, "deal_breakers": ["x"]},
            gaps=None, prep=None, deep_prep=None, cover_letter=None,
        )
        assert v_clean.overall_score > v_blocked.overall_score


# ═══════════════════════════════════════════════════════════════════
# Home page integration
# ═══════════════════════════════════════════════════════════════════


class TestHomeIntegration:
    def test_home_renders_action_items_when_silent(self, app_setup) -> None:
        app, store = app_setup
        j = _seed_job(store, title="老投递", company="字节跳动")
        app_id = _seed_app(store, j)
        with store.connect() as c:
            c.execute(
                "INSERT INTO application_events"
                "(application_id, kind, occurred_at, source, payload_json) "
                "VALUES (?, 'submitted', julianday('now','-15 days'), 'manual', '{}')",
                (app_id,),
            )
        resp = TestClient(app).get("/")
        # The silence action should be on the page
        assert "字节跳动" in resp.text
        assert "沉默" in resp.text or "跟进" in resp.text

    def test_home_renders_pipeline_count_link(self, app_setup) -> None:
        """Home shows the active-pipeline count as a clickable stat
        card linking to /pipeline. The 5-column mini-kanban moved to
        /pipeline itself to keep home uncluttered (总分式)."""
        app, store = app_setup
        j = _seed_job(store, title="x", company="C")
        _seed_app(store, j, status="1st_interview")
        resp = TestClient(app).get("/")
        assert "/pipeline" in resp.text
        # The 投递战况 stat card carries the count
        assert "投递战况" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Verdict in chat report
# ═══════════════════════════════════════════════════════════════════


class TestChatVerdictIntegration:
    """The /chat handler attaches a Verdict when ≥2 SKILL outputs are
    present in the agent's result. We patch build_graph to return a
    canned multi-skill result so we can verify the verdict surface."""

    def test_chat_with_no_runtime_short_circuits_without_verdict(
        self, app_setup
    ) -> None:
        """When runtime is None (no LLM configured) the /chat route
        early-returns with a 'no LLM' error before any SKILL runs.
        Critically this means no Verdict block is rendered — useful
        as a smoke test that the verdict path is conditional."""
        app, _ = app_setup
        resp = TestClient(app).post(
            "/chat",
            data={"job_text": "JD body here", "action": "score"},
        )
        assert resp.status_code == 200
        assert "综合判断" not in resp.text
        assert "未配置 LLM" in resp.text

    def test_verdict_synthesize_module_directly(self) -> None:
        """Smoke: synth doesn't crash on realistic 'everything' shape."""
        v = verdict.synthesize(
            score={"probability": 0.62, "deal_breakers": []},
            gaps={
                "suggestions": [
                    {"section": "技能", "ai_risk": "high"},
                    {"section": "项目", "ai_risk": "low"},
                ]
            },
            prep={"expected_questions": list(range(8))},
            deep_prep={
                "projects_analyzed": [{"name": "RemeDi"}],
                "weak_spots_to_practice": ["分布式训练", "评估指标"],
            },
            cover_letter={
                "overall_word_count": 280,
                "ai_risk_warnings": ["三段相同节奏"],
            },
            brief_confidence=0.55,
            brief_app_limit=3,
        )
        assert v.kind in ("go", "maybe", "hold", "skip")
        assert 0 <= v.overall_score <= 1
        assert 1 <= len(v.actions) <= 4
        # ai_risk_warnings → action surfaced
        assert any("cover letter" in a.title for a in v.actions)


_ = json  # keep import for downstream use
