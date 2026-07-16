"""W13.8 — Browser extension API + funnel UI + portfolio page + prompt caching."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.llm import LLMResponse
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# Extension API
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_client(tmp_path, master_resume_source_factory):
    store = offerguide.Store(tmp_path / "ext.db")
    store.init_schema()
    skills = discover_skills(SKILLS_ROOT)
    s = Settings(deepseek_api_key="x", default_model="stub")

    class _StubLLM:
        def chat(self, messages, **kw):
            return LLMResponse(content="{}", model="stub")

    runtime = SkillRuntime(llm=_StubLLM(), store=store)
    profile = master_resume_source_factory("x")
    app = create_app(
        settings=s,
        store=store,
        master_source=profile,
        skills=skills,
        runtime=runtime,
        notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestExtensionAPI:
    def test_ping(self, app_client):
        client, _ = app_client
        resp = client.get("/api/extension/ping")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert "version" in resp.json()
        # CORS header must be present (extension calls cross-origin)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_get_package_no_company_returns_404(self, app_client):
        client, _ = app_client
        resp = client.get("/api/extension/package?company=")
        assert resp.status_code == 404

    def test_get_package_no_match_returns_404_with_hint(self, app_client):
        client, _ = app_client
        resp = client.get("/api/extension/package?company=NeverHeardOfThem")
        assert resp.status_code == 404
        body = resp.json()
        assert "hint" in body
        assert "NeverHeardOfThem" in body["hint"]
        assert "具体岗位" in body["hint"]

    def test_get_package_returns_exact_resume_workspace(self, app_client):
        client, store = app_client
        package = {
            "message": "current application message",
            "form_answers": [],
            "pre_submit_checks": [],
        }
        with store.connect() as conn:
            job_id = int(
                conn.execute(
                    "INSERT INTO jobs(source, source_id, url, title, company, "
                    "raw_text, content_hash) VALUES ('m','j1','x','t','字节跳动','x','h1') "
                    "RETURNING id"
                ).fetchone()[0]
            )
            application_id = int(
                conn.execute(
                    "INSERT INTO applications(job_id, status) VALUES (?, 'considered') RETURNING id",
                    (job_id,),
                ).fetchone()[0]
            )
            workspace_id = int(
                conn.execute(
                    "INSERT INTO resume_workspaces("
                    "application_id, job_snapshot_json, master_source_sha256, apply_pack_json"
                    ") VALUES (?, ?, ?, ?) RETURNING id",
                    (
                        application_id,
                        json.dumps({"job_id": job_id, "company": "字节跳动"}),
                        "a" * 64,
                        json.dumps({"assistant": package}, ensure_ascii=False),
                    ),
                ).fetchone()[0]
            )

        resp = client.get(f"/api/extension/package?company=字节跳动&job_id={job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["package"]["message"] == "current application message"
        assert data["company"] == "字节跳动"
        assert data["job_id"] == job_id
        assert data["workspace_id"] == workspace_id
        assert data["workspace_status"] == "draft"
        assert set(data["package"]) == {"message", "form_answers", "pre_submit_checks"}
        with store.connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM skill_runs").fetchone()[0] == 0

    def test_multiple_company_workspaces_return_choices_instead_of_guessing(
        self,
        app_client,
    ) -> None:
        client, store = app_client
        package = {
            "message": "Current message",
            "form_answers": [],
            "pre_submit_checks": [],
        }
        with store.connect() as conn:
            for index, title in enumerate(("Agent 实习", "模型实习"), start=1):
                job_id = int(
                    conn.execute(
                        "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
                        "VALUES ('manual', ?, '同一公司', 'JD', ?) RETURNING id",
                        (title, f"same-company-{index}"),
                    ).fetchone()[0]
                )
                application_id = int(
                    conn.execute(
                        "INSERT INTO applications(job_id, status) "
                        "VALUES (?, 'considered') RETURNING id",
                        (job_id,),
                    ).fetchone()[0]
                )
                conn.execute(
                    "INSERT INTO resume_workspaces("
                    "application_id, job_snapshot_json, master_source_sha256, apply_pack_json"
                    ") VALUES (?, ?, ?, ?)",
                    (
                        application_id,
                        json.dumps({"job_id": job_id}),
                        "a" * 64,
                        json.dumps({"assistant": package}, ensure_ascii=False),
                    ),
                )

        response = client.get("/api/extension/package?company=同一公司")

        assert response.status_code == 409
        matches = response.json()["matches"]
        assert {match["title"] for match in matches} == {"Agent 实习", "模型实习"}
        assert all(set(match) == {"workspace_id", "job_id", "title"} for match in matches)


# ═══════════════════════════════════════════════════════════════════
# Funnel UI
# ═══════════════════════════════════════════════════════════════════


class TestFunnelView:
    def test_renders_empty_funnel(self, app_client):
        client, _ = app_client
        resp = client.get("/funnel")
        assert resp.status_code == 200
        assert "Pipeline · 转化概览" in resp.text
        # All zero counts
        assert "投递" in resp.text

    def test_funnel_counts_real_apps_and_events(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO jobs(source, raw_text, content_hash) VALUES ('m', ?, ?)",
                    ("x" * 250, f"h{i}"),
                )
                conn.execute(
                    "INSERT INTO applications(job_id, status) VALUES (?, 'submitted')",
                    (i + 1,),
                )
            # 3 of them got 'viewed'
            for app_id in [1, 2, 3]:
                conn.execute(
                    "INSERT INTO application_events(application_id, kind, source) "
                    "VALUES (?, 'viewed', 'manual')",
                    (app_id,),
                )
            # 1 got 'replied'
            conn.execute(
                "INSERT INTO application_events(application_id, kind, source) "
                "VALUES (1, 'replied', 'manual')"
            )
        resp = client.get("/funnel")
        assert resp.status_code == 200
        # Should mention 5 投递 and the conversion %
        assert "5" in resp.text  # total apps

    def test_per_company_breakdown(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            for company in ["字节", "字节", "腾讯"]:
                conn.execute(
                    "INSERT INTO jobs(source, company, raw_text, content_hash) "
                    "VALUES ('m', ?, ?, ?)",
                    (
                        company,
                        "x" * 250,
                        f"h_{company}_{conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]}",
                    ),
                )
            # 3 apps total
            for jid in [1, 2, 3]:
                conn.execute(
                    "INSERT INTO applications(job_id, status) VALUES (?, 'applied')",
                    (jid,),
                )
        resp = client.get("/funnel")
        assert "字节" in resp.text
        assert "腾讯" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Portfolio page
# ═══════════════════════════════════════════════════════════════════


class TestPortfolioPage:
    def test_renders_with_zero_data(self, app_client):
        client, _ = app_client
        resp = client.get("/portfolio")
        assert resp.status_code == 200
        assert "OfferGuide" in resp.text
        assert "Build" in resp.text or "Portfolio" in resp.text

    def test_shows_agent_run_count(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            for _ in range(7):
                conn.execute(
                    "INSERT INTO agent_runs(trigger_kind, goal, status, "
                    "  iterations, critic_score, cost_usd) "
                    "VALUES ('cron_wake', 'x', 'ok', 1, 0.85, 0.02)"
                )
        resp = client.get("/portfolio")
        assert resp.status_code == 200
        assert "7" in resp.text
        assert "real feedback" in resp.text
        assert "LLM 自评" in resp.text

    def test_no_user_data_leaked(self, app_client):
        """Portfolio must NOT show resume / user_facts / company names from apps."""
        client, store = app_client
        with store.connect() as conn:
            # Seed user_facts + applications with sensitive info
            conn.execute(
                "INSERT INTO user_facts(fact_text, kind) "
                "VALUES ('REAL_NAME private fact', 'profile')"
            )
            conn.execute(
                "INSERT INTO jobs(source, company, raw_text, content_hash) "
                "VALUES ('m', 'PrivateCorp', 'x' * 250, 'h1')"
            )
            conn.execute("INSERT INTO applications(job_id, status) VALUES (1, 'applied')")
        resp = client.get("/portfolio")
        # None of these should leak to portfolio
        assert "REAL_NAME" not in resp.text
        assert "PrivateCorp" not in resp.text
        assert "private fact" not in resp.text

    def test_per_skill_fitness_appears_when_data_present(self, app_client):
        """Per-SKILL fitness shows when ≥3 user_thumbs signals exist.

        (Pre-W21 this used 'critic' signals; the LLM-self-critique source
        was retired and the portfolio now reads from user_thumbs which is
        the real-signal channel.)
        """
        client, store = app_client
        from offerguide.evolution.signals import record_user_thumbs

        for _ in range(3):
            record_user_thumbs(
                store,
                skill_name="example_skill",
                skill_version="0.1.0",
                skill_run_id=None,
                thumbs=1,
            )
        resp = client.get("/portfolio")
        assert "example_skill" in resp.text


# ═══════════════════════════════════════════════════════════════════
# Prompt caching marker on system message
# ═══════════════════════════════════════════════════════════════════


class TestPromptCaching:
    def test_system_message_gets_cache_control_block(self, monkeypatch):
        """When chat_with_tools is called, the system message should be
        converted to a content-block list with cache_control marker."""
        from offerguide.llm import LLMClient

        captured_body: dict = {}

        class _StubResp:
            def __init__(self):
                self.status_code = 200

            def json(self):
                return {
                    "choices": [
                        {"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    "model": "claude-sonnet-4-6",
                }

            text = ""

        def fake_post(url, headers, json):
            captured_body.update(json)
            return _StubResp()

        client = LLMClient(
            api_key="x", base_url="https://example.com", default_model="claude-sonnet-4-6"
        )
        monkeypatch.setattr(client._http, "post", fake_post)

        client.chat_with_tools(
            messages=[
                {"role": "system", "content": "system prompt body"},
                {"role": "user", "content": "user msg"},
            ],
            tools=[],
        )
        sys_msg = captured_body["messages"][0]
        # System content is now a content-block list (Anthropic format)
        assert isinstance(sys_msg["content"], list)
        assert sys_msg["content"][0]["type"] == "text"
        assert sys_msg["content"][0]["text"] == "system prompt body"
        # The cache_control marker must be present
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_disabled_keeps_string_content(self, monkeypatch):
        from offerguide.llm import LLMClient

        captured_body: dict = {}

        class _StubResp:
            status_code = 200
            text = ""

            def json(self):
                return {
                    "choices": [
                        {"message": {"content": "x", "tool_calls": []}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "model": "x",
                }

        def fake_post(url, headers, json):
            captured_body.update(json)
            return _StubResp()

        client = LLMClient(api_key="x", base_url="https://example.com", default_model="x")
        monkeypatch.setattr(client._http, "post", fake_post)

        client.chat_with_tools(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u"},
            ],
            tools=[],
            cache_system_prompt=False,
        )
        sys_msg = captured_body["messages"][0]
        assert sys_msg["content"] == "sys"  # unchanged plain string
