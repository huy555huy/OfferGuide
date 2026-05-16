"""W8''''' — Autonomous layer: company_briefs + scheduler + 3 jobs.

Tests:
- company_briefs schema migration on fresh + pre-existing DB
- briefs.refresh_brief / get_brief / list_briefs / effective_app_limit
- briefs.gather_observations gathers all signal sources
- AutonomousScheduler lifecycle + trigger_once
- silence_check / corpus_refresh / brief_update jobs each invoke their
  underlying tools
- Dashboard renders briefs section
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import application_events as ae
from offerguide import briefs, interview_corpus
from offerguide.autonomous.scheduler import AutonomousScheduler, JobContext, JobSpec
from offerguide.briefs import CompanyBrief
from offerguide.config import Settings
from offerguide.profile import UserProfile
from offerguide.skills import discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ── stubs ──────────────────────────────────────────────────────────


class _StubLLM:
    def __init__(self, response: str = "{}") -> None:
        self.calls: list = []
        self._response = response

    def chat(self, *, messages, model=None, temperature=0.0, json_mode=False, extra=None):
        self.calls.append({"messages": messages, "temperature": temperature})
        return type("Resp", (), {"content": self._response, "raw": {}})()


def _make_store(tmp_path: Path) -> offerguide.Store:
    store = offerguide.Store(tmp_path / "auto.db")
    store.init_schema()
    return store


def _seed_company_data(store, company: str = "字节跳动") -> None:
    """Seed enough data for refresh_brief to have something to chew on."""
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('manual', 't', ?, 'jd', ?)",
            (company, f"hash_{company}"),
        )
        cur = conn.execute(
            "INSERT INTO applications(job_id, status) VALUES "
            "((SELECT id FROM jobs WHERE company = ? LIMIT 1), 'applied')",
            (company,),
        )
        app_id = cur.lastrowid
    ae.record(store, application_id=app_id, kind="submitted", source="manual")
    interview_corpus.insert(
        store, company=company, source="manual_paste",
        raw_text=f"{company} 一面 30 min · 项目深挖 RemeDi · GRPO vs PPO ·"
                 f" attention 缩放 · DeepSpeed ZeRO-2/3 选型",
    )


# ═══════════════════════════════════════════════════════════════════
# SCHEMA — company_briefs table is created on init
# ═══════════════════════════════════════════════════════════════════


class TestSchema:
    def test_company_briefs_in_health_check(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        counts = store.health_check()
        assert "company_briefs" in counts
        assert counts["company_briefs"] == 0


# ═══════════════════════════════════════════════════════════════════
# briefs.gather_observations
# ═══════════════════════════════════════════════════════════════════


class TestGatherObservations:
    def test_collects_all_signals(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")

        obs = briefs.gather_observations(store, "字节跳动")
        assert obs["jobs_count"] >= 1
        assert obs["applications_by_status"] == {"applied": 1}
        assert len(obs["recent_面经"]) == 1
        assert any(e["kind"] == "submitted" for e in obs["recent_events"])

    def test_empty_company_returns_empty_signals(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        obs = briefs.gather_observations(store, "无人公司")
        assert obs["jobs_count"] == 0
        assert obs["recent_面经"] == []


# ═══════════════════════════════════════════════════════════════════
# briefs.refresh_brief
# ═══════════════════════════════════════════════════════════════════


_GOOD_BRIEF_JSON = json.dumps({
    "summary": "字节跳动 AI 应用方向头部公司，最近 Seed 实验室扩招。",
    "current_app_limit": 2,
    "interview_style": "项目深挖 + 技术细节强，从面经看 GRPO 是必问点。",
    "recent_signals": [
        "最近 1 篇面经显示 GRPO vs PPO 必问",
        "DeepSpeed ZeRO 选型也常考",
    ],
    "hiring_trend": "expanding",
    "confidence": 0.7,
}, ensure_ascii=False)


class TestRefreshBrief:
    def test_refresh_inserts_new(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")
        llm = _StubLLM(response=_GOOD_BRIEF_JSON)
        result = briefs.refresh_brief(store, llm, "字节跳动")  # type: ignore[arg-type]

        assert result is not None
        assert result.brief.current_app_limit == 2
        assert result.brief.confidence == 0.7
        assert result.update_count == 1

    def test_refresh_updates_existing_increments_count(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")
        llm = _StubLLM(response=_GOOD_BRIEF_JSON)
        briefs.refresh_brief(store, llm, "字节跳动")  # type: ignore[arg-type]
        result = briefs.refresh_brief(store, llm, "字节跳动")  # type: ignore[arg-type]
        assert result is not None
        assert result.update_count == 2

    def test_refresh_returns_none_when_no_observations(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        # No seed
        llm = _StubLLM(response=_GOOD_BRIEF_JSON)
        result = briefs.refresh_brief(store, llm, "无人公司")  # type: ignore[arg-type]
        assert result is None
        # And LLM was never called (early bail)
        assert llm.calls == []

    def test_refresh_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")
        llm = _StubLLM(response="not valid json {")
        result = briefs.refresh_brief(store, llm, "字节跳动")  # type: ignore[arg-type]
        assert result is None

    def test_refresh_returns_none_on_schema_violation(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")
        # Missing required fields
        bad = json.dumps({"summary": "x", "extra_field": "rogue"})
        llm = _StubLLM(response=bad)
        result = briefs.refresh_brief(store, llm, "字节跳动")  # type: ignore[arg-type]
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# briefs.get_brief / list_briefs / effective_app_limit
# ═══════════════════════════════════════════════════════════════════


class TestBriefReads:
    def test_get_brief_round_trips(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")
        llm = _StubLLM(response=_GOOD_BRIEF_JSON)
        briefs.refresh_brief(store, llm, "字节跳动")  # type: ignore[arg-type]

        loaded = briefs.get_brief(store, "字节跳动")
        assert loaded is not None
        assert loaded.brief.current_app_limit == 2

    def test_list_briefs_orders_by_recency(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        for c in ("字节跳动", "阿里巴巴"):
            _seed_company_data(store, c)
            llm = _StubLLM(response=_GOOD_BRIEF_JSON)
            briefs.refresh_brief(store, llm, c)  # type: ignore[arg-type]

        rows = briefs.list_briefs(store)
        assert len(rows) == 2
        # Newest first
        assert rows[0].last_updated_at >= rows[1].last_updated_at

    def test_effective_app_limit_uses_brief_when_high_confidence(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")
        # High confidence brief says limit=5
        brief_high_conf = json.dumps({
            "summary": "x", "current_app_limit": 5,
            "interview_style": "x", "recent_signals": [],
            "hiring_trend": "expanding", "confidence": 0.85,
        })
        llm = _StubLLM(response=brief_high_conf)
        briefs.refresh_brief(store, llm, "字节跳动")  # type: ignore[arg-type]

        limit, source = briefs.effective_app_limit(store, "字节跳动")
        assert limit == 5
        assert source == "brief"

    def test_effective_app_limit_falls_back_when_low_confidence(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")
        brief_low_conf = json.dumps({
            "summary": "x", "current_app_limit": 5,
            "interview_style": "x", "recent_signals": [],
            "hiring_trend": "unknown", "confidence": 0.3,  # < 0.6
        })
        llm = _StubLLM(response=brief_low_conf)
        briefs.refresh_brief(store, llm, "字节跳动")  # type: ignore[arg-type]

        limit, source = briefs.effective_app_limit(store, "字节跳动")
        # Falls back to hardcoded (字节跳动 = 2)
        assert limit == 2
        assert source == "hardcoded"

    def test_effective_app_limit_falls_back_when_no_brief(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        limit, source = briefs.effective_app_limit(store, "字节跳动")
        assert limit == 2
        assert source == "hardcoded"


# ═══════════════════════════════════════════════════════════════════
# AutonomousScheduler — without actually running APScheduler
# ═══════════════════════════════════════════════════════════════════


class TestScheduler:
    def test_register_jobs_and_list(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ctx = JobContext(settings=Settings(), store=store, llm=None)
        sched = AutonomousScheduler(ctx)

        sched.add(JobSpec(
            name="dummy", func=lambda c: {"ok": True},
            trigger="cron", trigger_kwargs={"hour": 9},
        ))
        sched.add(JobSpec(
            name="dummy2", func=lambda c: {"ok": True},
            trigger="cron", trigger_kwargs={"hour": 10},
        ))

        assert sched.list_jobs() == ["dummy", "dummy2"]

    def test_trigger_once_calls_job_func(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ctx = JobContext(settings=Settings(), store=store, llm=None)
        sched = AutonomousScheduler(ctx)

        seen: list[JobContext] = []
        sched.add(JobSpec(
            name="probe",
            func=lambda c: (seen.append(c), {"called": True})[1],
            trigger="cron", trigger_kwargs={"hour": 9},
        ))

        result = sched.trigger_once("probe")
        assert result == {"called": True}
        assert len(seen) == 1
        assert seen[0] is ctx

    def test_trigger_once_unknown_job_raises(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        ctx = JobContext(settings=Settings(), store=store, llm=None)
        sched = AutonomousScheduler(ctx)
        with pytest.raises(KeyError):
            sched.trigger_once("nonexistent")


# ═══════════════════════════════════════════════════════════════════
# Jobs — silence_check / corpus_refresh / brief_update
# ═══════════════════════════════════════════════════════════════════


class TestSilenceCheckJob:
    def test_runs_tracker_run(self, tmp_path: Path) -> None:
        from offerguide.autonomous.jobs.silence_check import run

        store = _make_store(tmp_path)
        ctx = JobContext(
            settings=Settings(), store=store, llm=None,
            notifier=ConsoleNotifier(),
        )
        result = run(ctx)
        # Tracker counters are returned
        assert "silences_found" in result
        assert "events_recorded" in result


class TestCorpusRefreshJob:
    def test_skipped_when_no_llm(self, tmp_path: Path) -> None:
        from offerguide.autonomous.jobs.corpus_refresh import run

        store = _make_store(tmp_path)
        ctx = JobContext(settings=Settings(), store=store, llm=None)
        result = run(ctx)
        assert result == {"skipped": "no_llm"}

    def test_skipped_when_no_search(self, tmp_path: Path) -> None:
        from offerguide.autonomous.jobs.corpus_refresh import run

        store = _make_store(tmp_path)
        ctx = JobContext(
            settings=Settings(), store=store,
            llm=_StubLLM(),  # type: ignore[arg-type]
            search=None,
        )
        result = run(ctx)
        assert result == {"skipped": "no_search"}


class TestBriefUpdateJob:
    def test_skipped_when_no_llm(self, tmp_path: Path) -> None:
        from offerguide.autonomous.jobs.brief_update import run

        store = _make_store(tmp_path)
        ctx = JobContext(settings=Settings(), store=store, llm=None)
        result = run(ctx)
        assert result == {"skipped": "no_llm"}

    def test_refreshes_active_companies(self, tmp_path: Path) -> None:
        from offerguide.autonomous.jobs.brief_update import run

        store = _make_store(tmp_path)
        _seed_company_data(store, "字节跳动")
        _seed_company_data(store, "阿里巴巴")
        ctx = JobContext(
            settings=Settings(), store=store,
            llm=_StubLLM(response=_GOOD_BRIEF_JSON),  # type: ignore[arg-type]
        )
        result = run(ctx)
        assert result["refreshed"] == 2
        # Both companies got a brief stored
        assert len(briefs.list_briefs(store)) == 2


# ═══════════════════════════════════════════════════════════════════
# Briefs remain data-backed; /dashboard itself is merged into Mission Control.
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_setup(tmp_path: Path):
    store = _make_store(tmp_path)
    profile = UserProfile(raw_resume_text="x")
    skills = discover_skills(SKILLS_ROOT)
    app = create_app(
        settings=Settings(), store=store, profile=profile,
        skills=skills, runtime=None, notifier=ConsoleNotifier(),
    )
    return app, store


class TestDashboardBriefs:
    def test_briefs_list_still_returns_seeded_brief(self, app_setup) -> None:
        app, store = app_setup
        # Seed a brief
        brief = CompanyBrief(
            summary="字节最近 Seed 扩招",
            current_app_limit=2,
            interview_style="项目深挖",
            recent_signals=["最近面经显示 GRPO 必问"],
            hiring_trend="expanding",
            confidence=0.75,
        )
        briefs._upsert(store, "字节跳动", brief)

        rows = briefs.list_briefs(store, limit=10)
        assert rows[0].company == "字节跳动"
        assert "Seed 扩招" in rows[0].brief.summary
        assert any("GRPO 必问" in signal for signal in rows[0].brief.recent_signals)

    def test_dashboard_redirects_to_mission_control(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/dashboard", follow_redirects=False)
        assert resp.status_code == 301
        assert resp.headers["location"] == "/"


# ═══════════════════════════════════════════════════════════════════
# CLI smoke
# ═══════════════════════════════════════════════════════════════════


class TestAutonomousCLI:
    """W13.2: scheduler now registers ONE job (wake_agent), not 7 daemons."""

    def test_list_command_shows_wake_agent(self, tmp_path: Path, monkeypatch, capsys) -> None:
        from offerguide.autonomous.__main__ import main

        monkeypatch.setenv("OFFERGUIDE_DB", str(tmp_path / "auto.db"))
        rc = main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        # Old 7-daemon names gone; the only job is wake_agent
        assert "wake_agent" in out
        assert "silence_check" not in out  # was a job in pre-W13.2

    def test_run_once_wake_agent_skips_when_no_llm(self, tmp_path: Path, monkeypatch, capsys) -> None:
        """Without OFFERGUIDE_LLM_API_KEY, wake_agent should skip gracefully."""
        from offerguide.autonomous.__main__ import main

        monkeypatch.setenv("OFFERGUIDE_DB", str(tmp_path / "auto.db"))
        # Strip any inherited LLM env so the job sees None
        monkeypatch.delenv("OFFERGUIDE_LLM_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("TOKEN", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        rc = main(["run-once"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "wake_agent" in out
        # Should report the skip reason (no LLM)
        assert "skip" in out.lower() or "no LLM" in out

    def test_run_once_no_subcommand_fails_argparse(self, tmp_path: Path, monkeypatch) -> None:
        from offerguide.autonomous.__main__ import main

        monkeypatch.setenv("OFFERGUIDE_DB", str(tmp_path / "auto.db"))
        with pytest.raises(SystemExit):
            main([])
