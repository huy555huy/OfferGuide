"""W14.11 — bootstrap path real-walkthrough fixes.

User report: "输入 goal 后不知道干什么了, 自动找岗位也没有, 调简历也没有,
投递我也没看到 — 你到底有没有把这些功能放上去?"

Root cause: W13.1 删了 /quick-eval 想着扩展 + agent discover_jobs 自动会
够。但 fresh-DB + no-extension 用户撞死: 没 jobs → tailor / apply / agent
全空, 看不到入口。本轮恢复一条手动粘贴 JD 路径 + home 的下一步引导
随 DB state 变化。
"""

from __future__ import annotations

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


@pytest.fixture
def app_client(tmp_path):
    store = offerguide.Store(tmp_path / "boot.db")
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


# ═══════════════════════════════════════════════════════════════════
# Manual JD entry — the bootstrap path that was missing
# ═══════════════════════════════════════════════════════════════════


class TestManualJDEntry:
    def test_paste_creates_job_and_redirects_to_apply(self, app_client):
        client, store = app_client
        jd = (
            "AI Agent 暑期实习 - 上海\n"
            "工作内容: 1. 设计并实现 LLM-driven agent loop;\n"
            "2. evolve prompts 通过 user feedback;\n"
            "岗位要求: 1. 熟悉 Python; 2. LangGraph / LangChain 经验加分。"
        )
        # follow_redirects=False so we can inspect the 303 location
        resp = client.post(
            "/api/pipeline/jobs/manual",
            data={
                "raw_text": jd,
                "company": "字节跳动",
                "title": "AI Agent 暑期实习",
                "location": "上海",
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303
        # Should redirect to /apply/<the new job_id>
        assert resp.headers["location"].startswith("/apply/")
        # And the row landed in the DB
        with store.connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        assert n == 1

    def test_paste_too_short_rejected(self, app_client):
        client, _ = app_client
        resp = client.post(
            "/api/pipeline/jobs/manual",
            data={"raw_text": "太短"},
            follow_redirects=False,
        )
        assert resp.status_code == 400
        assert "JD 太短" in resp.text or "JD 至少" in resp.text

    def test_pipeline_page_shows_paste_form(self, app_client):
        client, _ = app_client
        resp = client.get("/pipeline")
        # The paste-JD form is on the pipeline page itself
        assert 'action="/api/pipeline/jobs/manual"' in resp.text
        assert 'name="raw_text"' in resp.text
        assert "粘贴 JD" in resp.text

    def test_pipeline_paste_form_open_by_default_when_empty(self, app_client):
        """Fresh users should see the form open without an extra click."""
        client, _ = app_client
        resp = client.get("/pipeline")
        # The <details> wrapping the form should have `open` when empty
        assert 'action="/api/pipeline/jobs/manual"' in resp.text
        # Find the <details> that contains the form action and check it has `open`
        idx = resp.text.find('action="/api/pipeline/jobs/manual"')
        before = resp.text[:idx]
        last_details = before.rfind("<details")
        assert "open" in resp.text[last_details:idx], (
            "form's <details> wrapper should be open when pipeline is empty"
        )


# ═══════════════════════════════════════════════════════════════════
# /apply (no id) redirects gracefully — no more 404 surprise
# ═══════════════════════════════════════════════════════════════════


class TestApplyIndexRedirect:
    def test_apply_with_no_id_redirects_to_pipeline(self, app_client):
        client, _ = app_client
        resp = client.get("/apply", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers["location"] == "/pipeline"


# ═══════════════════════════════════════════════════════════════════
# Home next-step card — state-aware, not a generic onboarding banner
# ═══════════════════════════════════════════════════════════════════


class TestStateAwareNextStep:
    def test_first_use_shows_two_paths(self, app_client):
        """No goal + no job = brand new. Show both A (paste JD) and B (set goal)."""
        client, _ = app_client
        resp = client.get("/")
        assert "👋 第一次用" in resp.text
        # Both calls-to-action should appear
        assert 'href="/pipeline"' in resp.text
        assert 'href="/goals"' in resp.text

    def test_goal_set_but_no_job_emphasizes_jd(self, app_client):
        """Goal exists but pipeline empty — push the user toward JD entry."""
        from offerguide import goals as _goals
        client, store = app_client
        _goals.add_goal(store, title="2026 暑期 AI Agent offer")
        resp = client.get("/")
        assert "Goal 设好了" in resp.text or "缺 JD" in resp.text
        # Should not show first-use two-path layout anymore
        assert "👋 第一次用" not in resp.text

    def test_have_jd_no_app_points_to_apply(self, app_client):
        """JD exists but no application yet — point at /apply for that job."""
        client, store = app_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, 'h1')",
                ("x" * 250,),
            )
            new_job_id = cur.lastrowid
        resp = client.get("/")
        # Should mention "投递包" and link to the actual job's /apply page
        assert "投递包" in resp.text
        assert f"/apply/{new_job_id}" in resp.text

    def test_no_next_step_card_when_already_progressed(self, app_client):
        """When user has applications, the next-step nudge gets out of the
        way — agent / inbox become primary."""
        client, store = app_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, 'h1')",
                ("x" * 250,),
            )
            job_id = cur.lastrowid
            conn.execute(
                "INSERT INTO applications(job_id, status, applied_at) "
                "VALUES (?, 'submitted', julianday('now'))", (job_id,),
            )
        resp = client.get("/")
        assert "👋 第一次用" not in resp.text
        assert "Goal 设好了" not in resp.text
        assert "去生成投递包" not in resp.text
