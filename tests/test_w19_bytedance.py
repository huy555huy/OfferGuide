"""W19+ — ByteDance jobs.bytedance.com API adapter tests."""
from __future__ import annotations

import json as _json

import pytest

import offerguide
from offerguide.application_plan import build_application_plan
from offerguide.agent_runtime import _schema as harness_schema
from offerguide.platforms.official_jobs import (
    BYTEDANCE_SEARCH_URL,
    SOURCE_LANDSCAPE,
    raw_job_from_bytedance,
    search_bytedance_jobs,
)
from offerguide.recruit_type import SOCIAL, classify_recruit_type

# ────────────────── source landscape sanity ──────────────────


def test_bytedance_status_updated_to_verified_public_social_only():
    """W19+ — codex 之前标 unverified_js_shell, 实测后 W19+ 改为 verified_public_social_only."""
    st = SOURCE_LANDSCAPE["bytedance"]
    assert st.status == "verified_public_social_only"
    assert "jobs.bytedance.com/api/v1/search/job/posts" in st.evidence_url
    assert "1334" in st.note or "公开" in st.note


def test_alibaba_status_updated_to_verified_via_aggregator():
    """W19+ — 阿里通过 0voice repo 间接覆盖, 不再是 unverified."""
    st = SOURCE_LANDSCAPE["alibaba"]
    assert st.status == "verified_via_aggregator"
    assert "0voice" in st.note or "campus-talent.alibaba.com" in st.evidence_url


# ────────────────── raw_job_from_bytedance ──────────────────


def test_raw_job_from_bytedance_full_fields():
    """Real API field shape (probed 2026-05-11)."""
    item = {
        "id": "7545051719739115783",
        "title": "后端/资深后端研发工程师（AI Agent方向）-TikTok Shop",
        "description": "1、负责跨境电商业务Agent应用开发\n2、优化Agent应用",
        "requirement": "1、掌握至少一种主流编程语言\n2、熟悉AI Agent技术",
        "city_info": {"name": "杭州", "code": "CT_52"},
        "city_list": [{"name": "杭州", "code": "CT_52"}],
        "recruit_type": {"id": "101", "name": "正式"},
        "code": "A39102C",
        "publish_time": 1756719479167,
    }
    rj = raw_job_from_bytedance(item)
    assert rj.source == "bytedance_jobs"
    assert rj.source_id == "7545051719739115783"
    assert rj.url == "https://jobs.bytedance.com/experienced/position/7545051719739115783/detail"
    assert rj.title == "后端/资深后端研发工程师（AI Agent方向）-TikTok Shop"
    assert rj.company == "字节跳动"
    assert rj.location == "杭州"
    # raw_text contains 岗位职责 + 任职要求
    assert "岗位职责" in rj.raw_text
    assert "任职要求" in rj.raw_text
    assert "Agent" in rj.raw_text
    # extras stamps source as verified API
    assert rj.extras["source_verified"] is True
    assert rj.extras["source_kind"] == "official_json_api"
    assert rj.extras["evidence_url"] == BYTEDANCE_SEARCH_URL
    assert rj.extras["recruit_type_name"] == "正式"


def test_raw_job_handles_missing_city_info():
    """Defensive: city_info missing or None falls back to city_list[0]."""
    item = {
        "id": "999", "title": "test", "city_info": {},
        "city_list": [{"name": "深圳"}],
        "recruit_type": {"name": "正式"},
        "code": "X1", "description": "x", "requirement": "y",
    }
    rj = raw_job_from_bytedance(item)
    assert rj.location == "深圳"


def test_raw_job_handles_missing_post_id():
    """Defensive: empty id → URL falls back to root."""
    item = {
        "id": "", "title": "no id job", "city_info": {"name": "北京"},
        "recruit_type": {"name": "正式"}, "code": "",
        "description": "d", "requirement": "r",
    }
    rj = raw_job_from_bytedance(item)
    assert rj.url == "https://jobs.bytedance.com/"


# ────────────────── recruit_type classifier ──────────────────


def test_bytedance_jobs_source_classifies_as_social():
    """All bytedance API results are recruit_type='正式' (verified empirically).
    classifier returns SOCIAL so default ?type=intern hides them."""
    rt = classify_recruit_type({
        "source": "bytedance_jobs",
        "title": "AI Agent 高级工程师",
        "extras_json": '{"recruit_type_name": "正式"}',
    })
    assert rt == SOCIAL


# ────────────────── search_bytedance_jobs network behavior ──────────────────


class TestSearchBytedance:
    def test_search_with_fake_client_returns_jobs(self):
        captured = {}

        class _FakeClient:
            def post(self, url, json=None, headers=None):
                captured["url"] = url
                captured["payload"] = json
                captured["headers"] = headers
                class R:
                    status_code = 200
                    def json(self):
                        return {
                            "message": "ok",
                            "data": {
                                "count": 100,
                                "job_post_list": [
                                    {
                                        "id": "111", "title": "AI Agent 工程师",
                                        "city_info": {"name": "上海"},
                                        "recruit_type": {"name": "正式"},
                                        "code": "B1",
                                        "description": "build agents",
                                        "requirement": "python",
                                    },
                                ],
                            },
                        }
                return R()

        result = search_bytedance_jobs(
            keyword="AI Agent", limit=3, client=_FakeClient(),  # type: ignore[arg-type]
        )
        assert captured["url"] == BYTEDANCE_SEARCH_URL
        assert captured["payload"]["keyword"] == "AI Agent"
        assert captured["payload"]["limit"] == 3
        assert captured["payload"]["portal_type"] == 6
        assert "Referer" in captured["headers"]
        assert result.status == "ok"
        assert result.source == "bytedance_jobs"
        assert len(result.jobs) == 1
        assert result.jobs[0].company == "字节跳动"
        assert result.jobs[0].title == "AI Agent 工程师"
        assert "1" in result.note or "100" in result.note  # mentions count

    def test_search_handles_http_error(self):
        class _FailClient:
            def post(self, url, json=None, headers=None):
                class R:
                    status_code = 500
                    def json(self):
                        return {"message": "internal error"}
                return R()

        r = search_bytedance_jobs(
            keyword="x", limit=3, client=_FailClient(),  # type: ignore[arg-type]
        )
        assert r.status == "error"
        assert "HTTP 500" in r.note

    def test_search_handles_network_exception(self):
        class _CrashClient:
            def post(self, url, json=None, headers=None):
                raise RuntimeError("conn reset")

        r = search_bytedance_jobs(
            keyword="x", limit=3, client=_CrashClient(),  # type: ignore[arg-type]
        )
        assert r.status == "error"
        assert "conn reset" in r.note

    def test_search_caps_limit_at_20(self):
        captured = {}
        class _C:
            def post(self, url, json=None, headers=None):
                captured["limit"] = json["limit"]
                class R:
                    status_code = 200
                    def json(self):
                        return {"message": "ok", "data": {"job_post_list": []}}
                return R()
        search_bytedance_jobs(
            keyword="x", limit=99, client=_C(),  # type: ignore[arg-type]
        )
        assert captured["limit"] == 20  # capped


# ────────────────── application_plan integration ──────────────────


def test_bytedance_jobs_source_routes_to_bytedance_plan():
    """Source name is enough — host detection isn't needed for the
    new bytedance_jobs source."""
    plan = build_application_plan({
        "source": "bytedance_jobs",
        "url": "https://jobs.bytedance.com/experienced/position/7545051719739115783/detail",
        "title": "AI Agent 工程师",
        "company": "字节跳动",
        "extras_json": _json.dumps({"source_verified": True}),
    })
    assert plan.platform == "bytedance"
    assert "字节跳动" in plan.platform_label


# ────────────────── ambient.py wiring ──────────────────


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "bd.db")
    s.init_schema()
    harness_schema.init_agent_runtime_schema(s)
    return s


def test_bytedance_source_in_unscored_discovered_ids(store):
    """bytedance_jobs source must be in the ambient daemon's unscored list,
    or those jobs would never get score_match."""
    from offerguide.workers.ambient import _load_unscored_discovered_ids
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            ("bytedance_jobs", "AI Agent 工程师", "字节跳动", "x" * 300, "h_bd_test"),
        )
        conn.commit()
    ids = _load_unscored_discovered_ids(store)
    assert len(ids) == 1
