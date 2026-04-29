"""W7 — Browser extension ingest endpoint tests.

Tests the ``POST /api/extension/ingest`` endpoint that receives
JD data from the Boss browser extension and calls ``scout.ingest()``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.ui.web import _extract_boss_id, create_app


def _make_client(tmp_path: Path) -> TestClient:
    settings = Settings(db_path=tmp_path / "ext.db")
    store = offerguide.Store(settings.db_path)
    store.init_schema()
    skills = []
    app = create_app(
        settings=settings,
        store=store,
        profile=None,
        skills=skills,
        runtime=None,
    )
    return TestClient(app)


class TestExtensionIngest:
    def test_ingests_new_job(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        payload = {
            "url": "https://www.zhipin.com/job_detail/abc123.html",
            "title": "AI Agent 实习",
            "company": "字节跳动",
            "location": "北京",
            "salary": "200-400元/天",
            "description": "负责 AI Agent 平台开发，熟悉 LangChain / LangGraph",
            "tags": ["Python", "LLM", "实习"],
        }
        resp = client.post("/api/extension/ingest", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_new"] is True
        assert data["job_id"] > 0

    def test_dedup_returns_existing(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        payload = {
            "url": "https://www.zhipin.com/job_detail/xyz.html",
            "title": "后端开发",
            "company": "阿里",
            "description": "负责微服务架构设计与开发",
        }
        r1 = client.post("/api/extension/ingest", json=payload)
        r2 = client.post("/api/extension/ingest", json=payload)
        assert r1.json()["is_new"] is True
        assert r2.json()["is_new"] is False
        assert r1.json()["job_id"] == r2.json()["job_id"]

    def test_rejects_empty_description(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        payload = {"title": "x", "description": "   "}
        resp = client.post("/api/extension/ingest", json=payload)
        assert resp.status_code == 400

    def test_rejects_missing_description(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        payload = {"title": "x"}
        resp = client.post("/api/extension/ingest", json=payload)
        assert resp.status_code == 422  # Pydantic validation error

    def test_stores_salary_and_tags_in_extras(self, tmp_path: Path) -> None:
        settings = Settings(db_path=tmp_path / "ext.db")
        store = offerguide.Store(settings.db_path)
        store.init_schema()
        app = create_app(
            settings=settings, store=store, profile=None, skills=[], runtime=None
        )
        client = TestClient(app)

        payload = {
            "title": "SDE",
            "company": "腾讯",
            "salary": "15-25K",
            "description": "C++ 后台开发",
            "tags": ["C++", "后台"],
        }
        resp = client.post("/api/extension/ingest", json=payload)
        job_id = resp.json()["job_id"]

        import json

        with store.connect() as conn:
            row = conn.execute(
                "SELECT extras_json, source, source_id FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        extras = json.loads(row[0])
        assert extras["salary"] == "15-25K"
        assert extras["tags"] == ["C++", "后台"]
        assert row[1] == "boss_extension"

    def test_extracts_boss_id_from_url(self, tmp_path: Path) -> None:
        settings = Settings(db_path=tmp_path / "ext.db")
        store = offerguide.Store(settings.db_path)
        store.init_schema()
        app = create_app(
            settings=settings, store=store, profile=None, skills=[], runtime=None
        )
        client = TestClient(app)

        payload = {
            "url": "https://www.zhipin.com/job_detail/f6a94d1cc3cbab571.html",
            "title": "前端",
            "description": "React 开发",
        }
        resp = client.post("/api/extension/ingest", json=payload)
        job_id = resp.json()["job_id"]

        with store.connect() as conn:
            row = conn.execute(
                "SELECT source_id FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        assert row[0] == "f6a94d1cc3cbab571"

    def test_tags_appended_to_raw_text(self, tmp_path: Path) -> None:
        settings = Settings(db_path=tmp_path / "ext.db")
        store = offerguide.Store(settings.db_path)
        store.init_schema()
        app = create_app(
            settings=settings, store=store, profile=None, skills=[], runtime=None
        )
        client = TestClient(app)

        payload = {
            "title": "ML",
            "description": "机器学习工程师",
            "tags": ["PyTorch", "推荐系统"],
        }
        resp = client.post("/api/extension/ingest", json=payload)
        job_id = resp.json()["job_id"]

        with store.connect() as conn:
            row = conn.execute(
                "SELECT raw_text FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        assert "标签: PyTorch, 推荐系统" in row[0]


class TestExtractBossId:
    def test_extracts_from_standard_url(self) -> None:
        assert _extract_boss_id(
            "https://www.zhipin.com/job_detail/f6a94d1cc3cbab571.html"
        ) == "f6a94d1cc3cbab571"

    def test_returns_none_for_non_boss_url(self) -> None:
        assert _extract_boss_id("https://www.nowcoder.com/jobs/123") is None

    def test_returns_none_for_none(self) -> None:
        assert _extract_boss_id(None) is None

    def test_handles_trailing_query_params(self) -> None:
        assert _extract_boss_id(
            "https://www.zhipin.com/job_detail/abc123.html?ka=search_list_1"
        ) == "abc123"
