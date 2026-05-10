"""W13.8 — Browser extension API + funnel UI + portfolio page + prompt caching."""

from __future__ import annotations

import json
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
# Extension API
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_client(tmp_path):
    store = offerguide.Store(tmp_path / "ext.db")
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
        assert "apply_assistant" in body["hint"]

    def test_get_package_returns_existing_skill_run(self, app_client):
        client, store = app_client
        # Seed a skill_run that looks like an apply_assistant output
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs(source, source_id, url, title, company, "
                "  raw_text, content_hash) VALUES ('m','j1','x','t','字节跳动','x','h1')"
            )
            output = json.dumps({
                "company": "字节跳动",
                "self_intro_snippet": {"text": "self intro text", "platform_hint": "boss_zhipin", "rationale": "x"},
                "qa_templates": [],
                "submission_strategy": {"best_time_window": "x", "follow_up_plan": "y", "expected_response_window_days": 7},
                "pre_submit_checklist": [],
                "skip_reasons": [],
                "confidence": 0.85,
            }, ensure_ascii=False)
            inputs = json.dumps({"company": "字节跳动", "job_text": "x", "user_profile": "y"}, ensure_ascii=False)
            conn.execute(
                "INSERT INTO skill_runs(skill_name, skill_version, input_hash, "
                "  input_json, output_json) VALUES (?,?,?,?,?)",
                ("apply_assistant", "0.1.0", "h", inputs, output),
            )
        resp = client.get("/api/extension/package?company=字节跳动")
        assert resp.status_code == 200
        data = resp.json()
        assert data["package"]["company"] == "字节跳动"
        assert data["company"] == "字节跳动"
        assert data["job_id"] == 1
        assert "self_intro_snippet" in data["package"]


# ═══════════════════════════════════════════════════════════════════
# Funnel UI
# ═══════════════════════════════════════════════════════════════════


class TestFunnelView:
    def test_renders_empty_funnel(self, app_client):
        client, _ = app_client
        resp = client.get("/funnel")
        assert resp.status_code == 200
        assert "转化漏斗" in resp.text
        # All zero counts
        assert "投递" in resp.text

    def test_funnel_counts_real_apps_and_events(self, app_client):
        client, store = app_client
        with store.connect() as conn:
            for i in range(5):
                conn.execute(
                    "INSERT INTO jobs(source, raw_text, content_hash) "
                    "VALUES ('m', ?, ?)", ("x" * 250, f"h{i}"),
                )
                conn.execute(
                    "INSERT INTO applications(job_id, status) VALUES (?, 'submitted')",
                    (i + 1,),
                )
            # 3 of them got 'viewed'
            for app_id in [1, 2, 3]:
                conn.execute(
                    "INSERT INTO application_events(application_id, kind, source) "
                    "VALUES (?, 'viewed', 'manual')", (app_id,),
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
                    (company, "x" * 250, f"h_{company}_{conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]}"),
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
        assert "0.85" in resp.text or "0.85" in resp.text  # avg critic

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
            conn.execute(
                "INSERT INTO applications(job_id, status) VALUES (1, 'applied')"
            )
        resp = client.get("/portfolio")
        # None of these should leak to portfolio
        assert "REAL_NAME" not in resp.text
        assert "PrivateCorp" not in resp.text
        assert "private fact" not in resp.text

    def test_per_skill_critic_appears_when_data_present(self, app_client):
        """Per-SKILL avg critic shows when ≥3 critic signals exist."""
        client, store = app_client
        from offerguide.evolution.signals import record_critic_signal
        for _ in range(3):
            record_critic_signal(
                store, skill_name="score_match", skill_version="0.1.0",
                skill_run_id=None, score=0.85,
            )
        resp = client.get("/portfolio")
        assert "score_match" in resp.text


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
                    "choices": [{"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                    "model": "claude-sonnet-4-6",
                }
            text = ""

        def fake_post(url, headers, json):
            captured_body.update(json)
            return _StubResp()

        client = LLMClient(api_key="x", base_url="https://example.com", default_model="claude-sonnet-4-6")
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
                    "choices": [{"message": {"content": "x", "tool_calls": []}, "finish_reason": "stop"}],
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
