from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from offerguide.config import Settings
from offerguide.memory import Store
from offerguide.ui.web import create_app

TEMPLATE_ROOT = (
    Path(__file__).resolve().parent.parent / "src" / "offerguide" / "ui" / "templates"
)


@pytest.mark.parametrize("template_name", ["recommended.html", "post_apply_pack.html"])
def test_running_agent_pages_poll_without_repeated_full_page_refresh(
    template_name: str,
) -> None:
    template = (TEMPLATE_ROOT / template_name).read_text(encoding="utf-8")

    assert 'id="agent-run-state"' in template
    assert 'role="status"' in template
    assert 'aria-live="polite"' in template
    assert 'data-poll-url="/api/research-agent-invocations/{{ agent_run.id }}"' in template
    assert "fetch(state.dataset.pollUrl" in template
    assert "delay = Math.min(delay * 2, 30000)" in template
    assert "if (payload.terminal)" in template
    assert "if (payload.result_changed === true)" in template
    assert "window.location.reload();" in template
    assert "window.setTimeout(function () { window.location.reload(); }, 2500)" not in template
    assert "<noscript>" in template


def test_invocation_status_endpoint_is_durable_and_not_cached(tmp_path: Path) -> None:
    store = Store(tmp_path / "status.db")
    store.init_schema()
    app = create_app(
        settings=Settings(db_path=store.db_path, disable_background_agents=True),
        store=store,
        master_source=None,
        skills=[],
        runtime=None,
    )
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO research_agent_invocations("
            "id, agent_name, subject_kind, subject_id, subject_revision, status, "
            "trigger_reason, message) "
            "VALUES ('poll-me', 'JobDiscoveryAgent', 'job_search', 'current', 1, "
            "'running', 'user request', 'Agent 正在检查真实来源')"
        )
    client = TestClient(app)

    running = client.get("/api/research-agent-invocations/poll-me")
    assert running.status_code == 200
    assert running.headers["cache-control"] == "no-store, max-age=0"
    assert running.json() == {
        "id": "poll-me",
        "agent_name": "JobDiscoveryAgent",
        "subject_kind": "job_search",
        "status": "running",
        "status_label": "Agent 正在研究",
        "message": "Agent 正在检查真实来源",
        "error": None,
        "unresolved": [],
        "updated_at": running.json()["updated_at"],
        "terminal": False,
        "result_changed": False,
        "result_url": "/recommended",
    }

    with store.connect() as conn:
        conn.execute(
            "UPDATE research_agent_invocations SET status = 'published', "
            "message = 'Agent 已发布新的当前结果', updated_at = julianday('now'), "
            "finished_at = julianday('now') WHERE id = 'poll-me'"
        )
    published = client.get("/api/research-agent-invocations/poll-me")
    assert published.json()["terminal"] is True
    assert published.json()["result_changed"] is True
    assert published.json()["result_url"] == "/recommended"
    assert client.get("/api/research-agent-invocations/missing").status_code == 404
