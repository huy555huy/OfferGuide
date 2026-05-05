"""W14.12 — ambient agent: real auto-discover, auto-score, intent-preview UI.

Pinning the contract for the rebuild from "tool-mode dashboard" to
"ambient agent that does work in the background and reports to home".

The 3 pillars exercised here:

1. JobCollector: search-driven JD discovery (Tavily + LLM extraction)
   replaces the company-directory awesome_jobs spider.
2. auto_score_jobs daemon: catches un-scored JDs, scores them, pre-gens
   the apply package for high-match ones, enqueues Intent Preview
   suggestions to inbox.
3. Home rewrites: "Agent 本周自动做了…" hero + Intent Preview cards
   with a 1-click "→ 看投递包 + 决定投不投" path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.llm import LLMResponse
from offerguide.profile import UserProfile
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# JobCollector — search-driven JD discovery
# ═══════════════════════════════════════════════════════════════════


class TestJobCollector:
    def test_collect_inserts_real_jds_via_search(self, tmp_path):
        """End-to-end with stubbed search + LLM. Verifies the full pipeline:
        search → domain filter → fetch fail → snippet fallback → LLM verdict
        → scout.ingest → row in jobs table."""
        from offerguide.agentic.job_collector import JobCollector
        from offerguide.agentic.search import SearchHit

        store = offerguide.Store(tmp_path / "jc.db")
        store.init_schema()

        good_snippet = (
            "字节跳动 AI Agent 暑期实习招聘 - 北京\n"
            "工作内容: 1. 设计并实现 LLM-driven agent;\n"
            "2. evolve prompts 通过 user feedback;\n"
            "岗位要求: Python / LangGraph 经验; 在校学生"
        ) + ("." * 200)  # ensure > 200 chars

        class _StubSearch:
            name = "stub"
            def search(self, q, *, max_results=10):
                return [
                    SearchHit(
                        title="字节 AI 实习",
                        url="https://nowcoder.com/jobs/abc",
                        snippet=good_snippet,
                    ),
                ]

        class _StubLLM:
            def chat(self, messages, **kw):
                # Return a positive verdict
                import json
                return LLMResponse(
                    content=json.dumps({
                        "is_real_jd": True,
                        "is_actively_hiring": True,
                        "matches_north_star": True,
                        "company": "字节跳动",
                        "title": "AI Agent 暑期实习",
                        "location": "北京",
                        "jd_body_clean": good_snippet,
                        "rationale": "明确实习岗 + 匹配 AI Agent 方向",
                    }),
                    model="stub",
                )

        coll = JobCollector(store=store, llm=_StubLLM(), search=_StubSearch())
        try:
            result = coll.collect(north_star="AI Agent 暑期实习")
        finally:
            coll.close()

        assert result.inserted == 1
        assert len(result.new_job_ids) == 1
        # Verify the row really landed with real JD body, not a 150-char directory
        with store.connect() as conn:
            row = conn.execute(
                "SELECT company, title, length(raw_text), source FROM jobs",
            ).fetchone()
        assert row[0] == "字节跳动"
        assert "AI Agent" in row[1]
        assert row[2] >= 200  # real JD body
        assert row[3] == "agent_search"  # provenance preserved

    def test_collect_rejects_low_quality_via_llm(self, tmp_path):
        """LLM verdict is_real_jd=False → row should NOT land."""
        from offerguide.agentic.job_collector import JobCollector
        from offerguide.agentic.search import SearchHit

        store = offerguide.Store(tmp_path / "jc.db")
        store.init_schema()

        class _StubSearch:
            name = "stub"
            def search(self, q, *, max_results=10):
                return [
                    SearchHit(
                        title="某公司",
                        url="https://nowcoder.com/companies/xyz",
                        snippet="公司目录: 字节、腾讯、阿里" + ("." * 250),
                    ),
                ]

        class _StubLLM:
            def chat(self, messages, **kw):
                import json
                return LLMResponse(content=json.dumps({
                    "is_real_jd": False,
                    "is_actively_hiring": False,
                    "matches_north_star": False,
                    "company": "",
                    "title": "",
                    "location": None,
                    "jd_body_clean": "",
                    "rationale": "公司目录, 不是具体岗位",
                }), model="stub")

        coll = JobCollector(store=store, llm=_StubLLM(), search=_StubSearch())
        try:
            result = coll.collect(north_star="AI 实习")
        finally:
            coll.close()

        assert result.inserted == 0
        assert result.skipped_low_quality >= 1
        with store.connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert n == 0

    def test_dedup_via_content_hash(self, tmp_path):
        """Re-running on the same DB does NOT double-insert."""
        from offerguide.agentic.job_collector import JobCollector
        from offerguide.agentic.search import SearchHit

        store = offerguide.Store(tmp_path / "jc.db")
        store.init_schema()

        body = "A具体岗位 JD body, 这是一段很长的内容用来通过长度阈值. " + ("内容" * 200)

        class _StubSearch:
            name = "stub"
            def search(self, q, *, max_results=10):
                return [SearchHit(title="A", url="https://nowcoder.com/x", snippet=body)]

        class _StubLLM:
            def chat(self, messages, **kw):
                import json
                return LLMResponse(content=json.dumps({
                    "is_real_jd": True, "is_actively_hiring": True,
                    "matches_north_star": True,
                    "company": "A公司", "title": "AI 实习",
                    "location": "北京",
                    "jd_body_clean": body,
                    "rationale": "ok",
                }), model="stub")

        coll = JobCollector(store=store, llm=_StubLLM(), search=_StubSearch())
        try:
            r1 = coll.collect(north_star="AI 实习")
            r2 = coll.collect(north_star="AI 实习")
        finally:
            coll.close()

        assert r1.inserted == 1
        assert r2.inserted == 0  # dedup
        assert r2.skipped_dup >= 1


# ═══════════════════════════════════════════════════════════════════
# auto_score_jobs daemon — picks up un-scored JDs + Intent Preview
# ═══════════════════════════════════════════════════════════════════


class TestAutoScoreJobsDaemon:
    def test_skips_when_no_llm(self, tmp_path):
        from offerguide.autonomous.jobs import auto_score_jobs
        from offerguide.autonomous.scheduler import JobContext

        store = offerguide.Store(tmp_path / "asj.db")
        store.init_schema()

        ctx = JobContext(
            settings=Settings(deepseek_api_key=""),
            store=store, llm=None, runtime=None,
            skills=[], user_profile_text=None,
        )
        out = auto_score_jobs.run(ctx)
        assert out.get("skipped") == "no_llm"

    def test_skips_when_no_profile(self, tmp_path):
        from offerguide.autonomous.jobs import auto_score_jobs
        from offerguide.autonomous.scheduler import JobContext

        store = offerguide.Store(tmp_path / "asj.db")
        store.init_schema()

        class _StubLLM:
            def chat(self, *a, **kw): return LLMResponse(content="{}", model="stub")

        ctx = JobContext(
            settings=Settings(deepseek_api_key="x"),
            store=store, llm=_StubLLM(),
            runtime=SkillRuntime(llm=_StubLLM(), store=store),
            skills=discover_skills(SKILLS_ROOT),
            user_profile_text=None,  # missing
        )
        out = auto_score_jobs.run(ctx)
        assert "missing" in out.get("skipped", "")

    def test_only_processes_jobs_above_min_text_length(self, tmp_path):
        from offerguide.autonomous.jobs import auto_score_jobs
        from offerguide.autonomous.scheduler import JobContext
        from offerguide.auto_pipeline import MIN_TEXT_FOR_AUTO_EVAL

        store = offerguide.Store(tmp_path / "asj.db")
        store.init_schema()

        # Insert one short JD (below threshold) and one long JD
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, 'h_short')",
                ("short JD",),
            )
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, 'h_long')",
                ("x" * (MIN_TEXT_FOR_AUTO_EVAL + 50),),
            )

        # LLM that returns score=0.3 (below INBOX_PROBABILITY_THRESHOLD)
        # so we test the score path without the suggest path
        import json
        class _StubLLM:
            def chat(self, *a, **kw):
                return LLMResponse(content=json.dumps({
                    "probability": 0.3,
                    "reasoning": "uncertain",
                    "dimensions": {"tech": 0.4, "exp": 0.2, "company_tier": 0.3},
                    "deal_breakers": [],
                }), model="stub")

        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        ctx = JobContext(
            settings=Settings(deepseek_api_key="x"),
            store=store, llm=_StubLLM(),
            runtime=runtime, skills=skills,
            user_profile_text="some resume text",
        )
        out = auto_score_jobs.run(ctx)
        # Only the long JD should have been considered (short one filtered by SQL)
        assert out["candidates"] == 1
        assert out["scored"] == 1
        # Below threshold → no auto suggestion
        assert out["auto_suggested"] == 0

    def test_high_match_creates_intent_preview_suggestion(self, tmp_path):
        """High-score JD → enqueue agent_suggestion with Intent Preview body."""
        from offerguide.autonomous.jobs import auto_score_jobs
        from offerguide.autonomous.scheduler import JobContext
        from offerguide.auto_pipeline import MIN_TEXT_FOR_AUTO_EVAL

        store = offerguide.Store(tmp_path / "asj.db")
        store.init_schema()

        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash, company, title, url) "
                "VALUES ('agent_search', ?, 'h_a', '字节', 'AI Agent 实习', 'https://x.com/jd1')",
                ("x" * (MIN_TEXT_FOR_AUTO_EVAL + 100),),
            )

        # LLM returns high score (above INBOX_PROBABILITY_THRESHOLD=0.55)
        # apply_assistant LLM call also returns OK so pre-gen path runs.
        import json
        class _StubLLM:
            def chat(self, *a, **kw):
                return LLMResponse(content=json.dumps({
                    "probability": 0.83,
                    "reasoning": "强匹配: AI Agent 经验对得上 + 在校学生身份符合",
                    "dimensions": {"tech": 0.9, "exp": 0.7, "company_tier": 0.9},
                    "deal_breakers": [],
                    # apply_assistant fields (since LLM is stubbed, same shape works)
                    "self_intro_snippet": "stub",
                    "qa_templates": [],
                    "submission_strategy": "stub",
                    "checklist": [],
                    "skip_reasons": [],
                }), model="stub")

        skills = discover_skills(SKILLS_ROOT)
        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        ctx = JobContext(
            settings=Settings(deepseek_api_key="x"),
            store=store, llm=_StubLLM(),
            runtime=runtime, skills=skills,
            user_profile_text="resume content"
        )
        out = auto_score_jobs.run(ctx)
        assert out["scored"] == 1
        assert out["auto_suggested"] == 1
        # Suggestion should be an agent_suggestion with attribution
        with store.connect() as conn:
            row = conn.execute(
                "SELECT title, source_skill_name, source_skill_version, kind FROM inbox_items"
            ).fetchone()
        assert row is not None
        assert "字节" in row[0] or "AI Agent" in row[0]
        # Intent Preview format: title encodes match %
        assert "%" in row[0]
        assert row[1] == "score_match"  # attributed to the SKILL that scored it
        assert row[2]  # version recorded
        assert row[3] == "agent_suggestion"


# ═══════════════════════════════════════════════════════════════════
# Scheduler — defaults to including ambient jobs
# ═══════════════════════════════════════════════════════════════════


class TestSchedulerHasAmbientJobs:
    def test_default_scheduler_has_three_jobs(self):
        from offerguide.autonomous.scheduler import build_agent_wake_scheduler

        sched = build_agent_wake_scheduler(
            settings=Settings(deepseek_api_key="", db_path=":memory:")
        )
        names = sched.list_jobs()
        # The 3 ambient pillars
        assert "wake_agent" in names              # central agent loop
        assert "discover_jobs_via_search" in names  # Tavily-driven
        assert "auto_score_new_jobs" in names      # score + Intent Preview
        sched.shutdown()


# ═══════════════════════════════════════════════════════════════════
# Home rewrite — "agent did this week" hero + Intent Preview cards
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_client(tmp_path):
    store = offerguide.Store(tmp_path / "home.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)
    s = Settings(deepseek_api_key="x", default_model="stub")

    class _StubLLM:
        def chat(self, messages, **kw):
            return LLMResponse(content="{}", model="stub")

    runtime = SkillRuntime(llm=_StubLLM(), store=store)
    profile = UserProfile(raw_resume_text="x", source_pdf="/tmp/x.pdf")
    app = create_app(
        settings=s, store=store, profile=profile,
        skills=skills, runtime=runtime, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestHomeAmbientHero:
    def test_empty_db_shows_ambient_pitch(self, app_client):
        client, _ = app_client
        resp = client.get("/")
        # Ambient mode pitch — sells what scheduler will do
        assert "scheduler" in resp.text or "Tavily" in resp.text or "准备就绪" in resp.text

    def test_with_weekly_activity_shows_report(self, app_client):
        """Insert some agent runs / skill runs / suggestions / agent_search jobs
        from this week and verify hero shows the count breakdown."""
        client, store = app_client
        with store.connect() as conn:
            # 2 agent runs in last 7 days
            for _ in range(2):
                conn.execute(
                    "INSERT INTO agent_runs(trigger_kind, goal, status, "
                    " iterations, started_at, ended_at) "
                    "VALUES ('cron_wake','x','ok',1,julianday('now'),julianday('now'))"
                )
            # 5 score_match runs
            for i in range(5):
                conn.execute(
                    "INSERT INTO skill_runs(skill_name, skill_version, "
                    " input_hash, input_json, output_json, created_at) "
                    "VALUES ('score_match', '0.1.0', ?, '{}', '{}', julianday('now'))",
                    (f"h_{i}",),
                )
            # 3 jobs from agent_search
            for i in range(3):
                conn.execute(
                    "INSERT INTO jobs(source, raw_text, content_hash) "
                    "VALUES ('agent_search', ?, ?)",
                    ("x" * 250, f"hash_{i}"),
                )
            # 4 inbox suggestions
            for i in range(4):
                conn.execute(
                    "INSERT INTO inbox_items(kind, title, body, "
                    " payload_json, created_at) "
                    "VALUES ('agent_suggestion', ?, '', '{}', julianday('now'))",
                    (f"sug{i}",),
                )

        resp = client.get("/")
        # Shows all 4 metrics in the hero
        assert "本周自动" in resp.text
        # Counts visible
        assert ">3<" in resp.text or "3</div>" in resp.text  # 3 jobs auto-found
        assert ">5<" in resp.text or "5</div>" in resp.text  # 5 score_match
        assert ">4<" in resp.text or "4</div>" in resp.text  # 4 suggestions
        assert ">2<" in resp.text or "2</div>" in resp.text  # 2 wakes


class TestIntentPreviewSuggestionCards:
    def test_high_match_suggestion_links_to_apply(self, app_client):
        """Suggestions with payload.job_id show a "→ 看投递包" CTA pointing at /apply/<id>."""
        from offerguide import inbox as inbox_mod
        client, store = app_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash, company, title) "
                "VALUES ('m', ?, 'h_x', '字节', 'AI Agent 实习')",
                ("x" * 250,),
            )
            job_id = cur.lastrowid
        inbox_mod.enqueue_agent_suggestion(
            store, title="考虑投: 字节 (匹配 83%)",
            body="**为什么推荐**: 自动 score_match 算出 83% 匹配。",
            source_skill_name="score_match",
            source_skill_version="0.1.0",
            payload={"job_id": job_id, "probability": 0.83, "apply_run_id": 999},
        )
        resp = client.get("/")
        assert f'/apply/{job_id}' in resp.text
        assert "看投递包" in resp.text
        # Match % pill displayed
        assert "83%" in resp.text or "%" in resp.text

    def test_dismiss_button_present(self, app_client):
        from offerguide import inbox as inbox_mod
        client, store = app_client
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="t", body="b",
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        resp = client.get("/")
        # The "不感兴趣" dismiss button posts to /inbox/<id>/decide with dismissed
        assert f'/inbox/{item.id}/decide' in resp.text
        assert "不感兴趣" in resp.text or "dismissed" in resp.text
