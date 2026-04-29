"""FastAPI web UI — TestClient drives chat / inbox routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import inbox as inbox_mod
from offerguide.config import Settings
from offerguide.profile import UserProfile
from offerguide.skills import SkillResult, discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._next_id = 200

    def invoke(self, spec, inputs, **_):
        self._next_id += 1
        self.calls.append((spec.name, inputs))
        parsed: dict[str, Any]
        if spec.name == "score_match":
            parsed = {
                "probability": 0.65,
                "reasoning": "中等匹配",
                "dimensions": {"tech": 0.7, "exp": 0.5, "company_tier": 0.7},
                "deal_breakers": [],
            }
        else:
            parsed = {
                "summary": "中等匹配，缺少 X",
                "keyword_gaps": [],
                "suggestions": [],
                "do_not_add": [],
                "ai_detection_warnings": [],
            }
        return SkillResult(
            raw_text=json.dumps(parsed),
            parsed=parsed,
            skill_name=spec.name,
            skill_version=spec.version,
            skill_run_id=self._next_id,
            input_hash="x",
            cost_usd=0.0,
            latency_ms=1,
        )


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "web.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="胡阳，统计学专硕")
    skills = discover_skills(SKILLS_ROOT)
    runtime = _FakeRuntime()
    notifier = ConsoleNotifier()
    settings = Settings()
    app = create_app(
        settings=settings,
        store=store,
        profile=profile,
        skills=skills,
        runtime=runtime,  # type: ignore[arg-type]
        notifier=notifier,
    )
    return app, store, runtime


def test_home_renders_with_profile_status(app_setup) -> None:
    app, _, _ = app_setup
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "OfferGuide" in resp.text
    assert "简历已加载" in resp.text


def test_home_renders_warning_when_no_profile(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "x.db")
    store.init_schema()
    app = create_app(
        settings=Settings(),
        store=store,
        profile=None,
        skills=discover_skills(SKILLS_ROOT),
        runtime=None,
        notifier=ConsoleNotifier(),
    )
    client = TestClient(app)
    resp = client.get("/")
    assert "未加载简历" in resp.text


def test_inbox_view_renders_pending_items(app_setup) -> None:
    app, store, _ = app_setup
    inbox_mod.enqueue(store, kind="consider_jd", title="AI Agent @ ByteDance")
    client = TestClient(app)
    resp = client.get("/inbox")
    assert resp.status_code == 200
    assert "AI Agent @ ByteDance" in resp.text
    assert "pending" in resp.text


def test_chat_dispatches_score_and_gaps_and_renders_report(app_setup) -> None:
    app, _, runtime = app_setup
    client = TestClient(app)
    resp = client.post(
        "/chat",
        data={"job_text": "前端实习生 React TypeScript", "action": "score_and_gaps"},
    )
    assert resp.status_code == 200
    # Both skills called
    assert {c[0] for c in runtime.calls} == {"score_match", "analyze_gaps"}
    # Rendered summary contains both sections
    assert "## 匹配评分" in resp.text
    assert "## 差距与建议" in resp.text
    # The "考虑投递" CTA must be present so user can enqueue to inbox
    assert "考虑投递" in resp.text


def test_chat_with_no_profile_returns_error_fragment(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "x.db")
    store.init_schema()
    app = create_app(
        settings=Settings(),
        store=store,
        profile=None,
        skills=discover_skills(SKILLS_ROOT),
        runtime=None,
        notifier=ConsoleNotifier(),
    )
    client = TestClient(app)
    resp = client.post("/chat", data={"job_text": "x", "action": "score"})
    assert resp.status_code == 200
    assert "未加载简历" in resp.text


def test_chat_with_no_runtime_returns_error_fragment(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "x.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="x")
    app = create_app(
        settings=Settings(),
        store=store,
        profile=profile,
        skills=discover_skills(SKILLS_ROOT),
        runtime=None,
        notifier=ConsoleNotifier(),
    )
    client = TestClient(app)
    resp = client.post("/chat", data={"job_text": "x", "action": "score"})
    assert "未配置 LLM" in resp.text


def test_inbox_from_report_enqueues_and_returns_list(app_setup) -> None:
    app, store, _ = app_setup
    client = TestClient(app)
    resp = client.post(
        "/inbox/from-report",
        data={
            "title": "考虑投递: Frontend Intern",
            "body": "match=0.65",
            "job_text": "...",
            "score_run_id": "201",
            "gaps_run_id": "",
        },
    )
    assert resp.status_code == 200
    items = inbox_mod.list_items(store, status="pending")
    assert len(items) == 1
    assert items[0].title == "考虑投递: Frontend Intern"
    assert items[0].payload.get("score_run_id") == 201


def test_inbox_decide_marks_status_and_returns_swap_fragment(app_setup) -> None:
    app, store, _ = app_setup
    item = inbox_mod.enqueue(store, kind="consider_jd", title="X")
    client = TestClient(app)
    resp = client.post(f"/inbox/{item.id}/decide", data={"decision": "approved"})
    assert resp.status_code == 200
    assert "approved" in resp.text
    refreshed = inbox_mod.get(store, item.id)
    assert refreshed is not None
    assert refreshed.status == "approved"


def test_inbox_decide_unknown_id_returns_404(app_setup) -> None:
    app, _, _ = app_setup
    client = TestClient(app)
    resp = client.post("/inbox/9999/decide", data={"decision": "approved"})
    assert resp.status_code == 404


def test_inbox_decide_invalid_decision_returns_400(app_setup) -> None:
    app, store, _ = app_setup
    item = inbox_mod.enqueue(store, kind="consider_jd", title="X")
    client = TestClient(app)
    resp = client.post(f"/inbox/{item.id}/decide", data={"decision": "ignored"})
    assert resp.status_code == 400


def test_inbox_decide_already_decided_returns_409(app_setup) -> None:
    app, store, _ = app_setup
    item = inbox_mod.enqueue(store, kind="consider_jd", title="X")
    inbox_mod.decide(store, item.id, decision="approved")
    client = TestClient(app)
    resp = client.post(f"/inbox/{item.id}/decide", data={"decision": "rejected"})
    assert resp.status_code == 409
