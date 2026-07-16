from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.ui.web import create_app


def test_manual_apply_pack_get_shows_job_without_creating_materials(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    app = create_app(
        settings=Settings(deepseek_api_key="", db_path=tmp_path / "ui.db"),
        store=store,
        master_source=None,
        skills=[],
        runtime=None,
    )
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, url, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "manual",
                "https://careers.example.com/jobs/123",
                "算法工程师",
                "样例科技",
                "完整 JD 全文" * 40,
                "apply-pack-entry",
            ),
        )

    response = TestClient(app).get("/jobs/1/apply-pack")

    assert response.status_code == 200
    assert "完整 JD 全文" in response.text
    assert "缺少 master PDF" in response.text
    with store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM resume_workspaces").fetchone()[0] == 0
