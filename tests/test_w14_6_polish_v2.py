"""W14.6 — second-pass UI polish: dead links / .env-leaks / unified empty states.

Driven by a real "fresh-user simulation" walk-through: spinning up a clean DB,
hitting every nav route via curl, and noting where the UI still felt like a
toy (dead links, env-var names leaking to users, mismatched empty-state styling).
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
def fresh_client(tmp_path):
    """Empty DB + no profile + no LLM key — simulates a friend's first launch."""
    store = offerguide.Store(tmp_path / "fresh.db")
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


@pytest.fixture
def fresh_client_no_llm(tmp_path):
    """Empty DB, no LLM runtime, no profile — most pessimistic friend scenario."""
    store = offerguide.Store(tmp_path / "fresh.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)
    s = Settings(deepseek_api_key=None, default_model="stub")
    app = create_app(
        settings=s, store=store, profile=None,
        skills=skills, runtime=None, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


# ═══════════════════════════════════════════════════════════════════
# Dead links (P0): /quick-eval and "Chat 页" must be gone everywhere.
# ═══════════════════════════════════════════════════════════════════


class TestDeadLinks:
    """The W13.1 cleanup deleted /quick-eval and /chat routes but left
    references in templates. A friend clicking these gets a 404."""

    @pytest.mark.parametrize("path", [
        "/", "/agent", "/goals", "/inbox", "/applications", "/pipeline",
        "/tailor", "/funnel", "/portfolio", "/interviews", "/compare",
        "/stories", "/reflect", "/mock", "/evolution", "/dashboard",
    ])
    def test_no_quick_eval_link_in_any_page(self, fresh_client, path):
        client, _ = fresh_client
        resp = client.get(path)
        assert resp.status_code == 200
        assert "/quick-eval" not in resp.text, (
            f"{path} still references the dead /quick-eval route"
        )

    def test_applications_no_longer_mentions_chat_page(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/applications")
        assert "Chat 页" not in resp.text, (
            "applications.html empty hint still says 'Chat 页' (route removed in W13)"
        )


# ═══════════════════════════════════════════════════════════════════
# .env / OFFERGUIDE_* names should not surface to non-technical friends.
# ═══════════════════════════════════════════════════════════════════


class TestEnvVarNamesHidden:
    """Friends won't know what OFFERGUIDE_LLM_API_KEY is. Visible-to-user
    error/empty messages should use plain Chinese; env names belong only
    in tooltips (title="...") if at all."""

    def test_home_no_naked_env_var_in_warning(self, fresh_client_no_llm):
        client, _ = fresh_client_no_llm
        resp = client.get("/")
        # The big LLM-not-ready banner must NOT show env var name in body text.
        # It's OK if it appears inside a tooltip title="..." attribute.
        body_after_strip_titles = "".join(
            line for line in resp.text.split("\n")
            if 'title="' not in line
        )
        assert "OFFERGUIDE_LLM_API_KEY" not in body_after_strip_titles
        # New friendly copy should be there
        assert "还没配 LLM" in resp.text or "看 README" in resp.text

    def test_agent_page_uses_friendly_hint(self, fresh_client_no_llm):
        client, _ = fresh_client_no_llm
        resp = client.get("/agent")
        # New copy
        assert "Agent 还没接通 LLM" in resp.text
        # Old scary error class shouldn't be how we say it
        assert "Agent 不可用" not in resp.text

    def test_apply_page_uses_friendly_hint(self, fresh_client_no_llm):
        client, store = fresh_client_no_llm
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, 'h1')",
                ("x" * 250,),
            )
        resp = client.get("/apply/1")
        # Should not show raw env var to user
        assert "缺 OFFERGUIDE_LLM_API_KEY" not in resp.text
        assert "缺 OFFERGUIDE_RESUME_PDF" not in resp.text
        # Should use friendly copy
        assert "还没接通 LLM" in resp.text or "还没传简历" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Empty states should all use the W14.5 .empty-{icon,title,hint,action} pattern.
# ═══════════════════════════════════════════════════════════════════


class TestUnifiedEmptyStates:
    def test_applications_empty_uses_v2_structure(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/applications")
        assert "empty-icon" in resp.text
        assert "empty-title" in resp.text
        assert "还没有投递记录" in resp.text

    def test_tailor_jd_empty_uses_v2_structure(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/tailor")
        # The "no JDs to tailor" empty state
        assert "还没有可微调的 JD" in resp.text
        assert "empty-title" in resp.text

    def test_pipeline_empty_uses_v2_structure(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/pipeline")
        # W14.11: copy updated to surface the 3 entry paths upfront
        assert "Pipeline 空" in resp.text or "Pipeline 还是空的" in resp.text
        assert "empty-icon" in resp.text
        assert "empty-title" in resp.text

    def test_interviews_empty_uses_v2_structure(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/interviews")
        # paste-in form is always visible; only the recent-list is empty
        assert "还没有面经" in resp.text
        assert "empty-title" in resp.text

    def test_compare_empty_uses_v2_structure(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/compare")
        assert "empty-title" in resp.text
        assert "empty-hint" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Home onboarding (P1): truly-empty users get a 3-step "first time?" banner.
# ═══════════════════════════════════════════════════════════════════


class TestHomeOnboarding:
    def test_empty_db_shows_first_time_banner(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/")
        assert "👋 第一次用" in resp.text
        # Three concrete steps with real route links
        assert 'href="/goals"' in resp.text
        assert 'href="/"' in resp.text

    def test_non_empty_db_hides_first_time_banner(self, fresh_client):
        client, store = fresh_client
        # W14.11: next-step is now state-aware. Insert a job so we leave
        # the "first_use" branch and don't show the "👋 第一次用" header.
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, 'h1')",
                ("x" * 250,),
            )
        resp = client.get("/")
        assert "👋 第一次用" not in resp.text


# ═══════════════════════════════════════════════════════════════════
# Quick-link grid: 8 cards including the entry-point ones (W14.6).
# ═══════════════════════════════════════════════════════════════════


class TestHomeQuickLinks:
    def test_quick_links_now_include_goals_and_funnel(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/")
        # The four critical ones the previous nav grid was missing
        for needed in ['href="/goals"', 'href="/funnel"',
                       'href="/pipeline"', 'href="/portfolio"']:
            assert needed in resp.text, f"home.html quick-link grid is missing {needed}"


# ═══════════════════════════════════════════════════════════════════
# Status strip: friendlier when un-configured (no scary warn dot + env name).
# ═══════════════════════════════════════════════════════════════════


class TestStatusStrip:
    def test_no_resume_uses_dim_dot_not_warn(self, fresh_client_no_llm):
        client, _ = fresh_client_no_llm
        resp = client.get("/")
        # The header strip should use the new neutral "dim" dot, not warn (yellow).
        # Yellow on every page makes the friend think the system is broken.
        assert "status-dot dim" in resp.text
        # Friendly copy
        assert "简历未配置" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Status-dot CSS class .dim must be defined so the new dot color works.
# ═══════════════════════════════════════════════════════════════════


class TestStatusDotDimClass:
    def test_status_dot_dim_class_is_defined(self, fresh_client):
        client, _ = fresh_client
        resp = client.get("/")
        assert ".status-dot.dim" in resp.text
