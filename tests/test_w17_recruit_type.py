"""W17 — recruit_type classifier + /recommended filter tests.

Empirical-driven (probed real APIs 2026-05-10):
  腾讯校招 projectName='应届实习' 当前 596 条全是这一个值
  百度校招 GRADUATE → projectType ∈ {'校招', 'AIDU项目', '管培生项目'}
  百度校招 INTERN   → projectType ∈ {'暑期实习项目', '日常实习项目'}
"""
from __future__ import annotations

import json as _json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.harness import _schema as harness_schema
from offerguide.recruit_type import (
    CAMPUS_FULLTIME,
    DAILY_INTERN,
    LABEL_ZH,
    SOCIAL,
    SUMMER_INTERN,
    UNKNOWN,
    classify_recruit_type,
    is_intern,
)
from offerguide.ui.web import create_app

# ─────────────────── Classifier unit tests ───────────────────

class TestClassifier:
    def test_baidu_intern_summer_via_flat_project_type(self):
        """W17 baidu raw_job adds flat 'project_type' for fast lookup."""
        job = {
            "source": "baidu_intern", "title": "agent训练工程环境研发（J99851）",
            "extras_json": _json.dumps({"project_type": "暑期实习项目"}),
        }
        assert classify_recruit_type(job) == SUMMER_INTERN

    def test_baidu_intern_daily(self):
        job = {
            "source": "baidu_intern", "title": "内容营销实习生（百度健康）",
            "extras_json": _json.dumps({"project_type": "日常实习项目"}),
        }
        assert classify_recruit_type(job) == DAILY_INTERN

    def test_baidu_campus_fulltime(self):
        for pt in ("校招", "AIDU项目", "管培生项目"):
            job = {
                "source": "baidu_campus", "title": f"测试-{pt}(J123)",
                "extras_json": _json.dumps({"project_type": pt}),
            }
            assert classify_recruit_type(job) == CAMPUS_FULLTIME, f"projectType={pt}"

    def test_baidu_falls_back_to_nested_raw_row(self):
        """codex's W16 stored projectType inside extras['raw_row']; classifier
        should still find it for backward compat."""
        job = {
            "source": "baidu_campus", "title": "x",
            "extras_json": _json.dumps({"raw_row": {"projectType": "暑期实习项目"}}),
        }
        assert classify_recruit_type(job) == SUMMER_INTERN

    def test_tencent_campus_应届实习_is_summer_intern(self):
        """腾讯当前 596 条全是 '应届实习' → 都是暑期可转正实习."""
        job = {
            "source": "tencent_campus", "title": "AI-HR培训生（创意方向）",
            "extras_json": _json.dumps({
                "project_name": "应届实习", "recruit_label": "应届实习",
            }),
        }
        assert classify_recruit_type(job) == SUMMER_INTERN

    def test_tencent_social_is_always_social(self):
        """腾讯社招源全部 social, 即使 title 含'实习生'."""
        job = {
            "source": "tencent_social", "title": "AI Agent 应用架构工程师",
            "extras_json": "{}",
        }
        assert classify_recruit_type(job) == SOCIAL

    def test_nowcoder_summer_intern_by_title(self):
        job = {
            "source": "nowcoder", "title": "字节跳动 2026 暑期实习 - AI Agent",
            "extras_json": "{}",
        }
        assert classify_recruit_type(job) == SUMMER_INTERN

    def test_nowcoder_daily_intern_by_title(self):
        job = {"source": "nowcoder", "title": "日常实习 - 算法", "extras_json": "{}"}
        assert classify_recruit_type(job) == DAILY_INTERN

    def test_nowcoder_intern_alone_defaults_to_daily(self):
        """大厂暑期实习 always 标 '暑期'; '实习' 没标暑期 → 日常 by convention."""
        job = {"source": "nowcoder", "title": "AI 算法实习生", "extras_json": "{}"}
        assert classify_recruit_type(job) == DAILY_INTERN

    def test_nowcoder_campus_fulltime_by_title(self):
        job = {
            "source": "nowcoder", "title": "2026 校招 - 运维工程师",
            "extras_json": "{}",
        }
        assert classify_recruit_type(job) == CAMPUS_FULLTIME

    def test_unknown_when_no_signal(self):
        job = {"source": "manual", "title": "xyz", "extras_json": "{}"}
        assert classify_recruit_type(job) == UNKNOWN

    def test_robust_to_missing_extras(self):
        """Doesn't crash on bad/missing extras_json."""
        for raw in (None, "", "not json", "{garbage", '"a string"'):
            job = {"source": "baidu_campus", "title": "暑期实习生", "extras_json": raw}
            # Falls through to title heuristic
            assert classify_recruit_type(job) == SUMMER_INTERN

    def test_is_intern_helper(self):
        assert is_intern(SUMMER_INTERN)
        assert is_intern(DAILY_INTERN)
        assert not is_intern(CAMPUS_FULLTIME)
        assert not is_intern(SOCIAL)
        assert not is_intern(UNKNOWN)

    def test_label_zh_covers_all_types(self):
        from offerguide.recruit_type import ALL_TYPES
        for t in ALL_TYPES:
            assert t in LABEL_ZH
            assert LABEL_ZH[t]  # non-empty


# ─────────────────── Baidu INTERN endpoint ───────────────────

class TestBaiduIntern:
    def test_search_baidu_intern_calls_correct_url(self, monkeypatch):
        """search_baidu_intern_jobs should hit recruitType=INTERN, not GRADUATE."""
        from offerguide.platforms.official_jobs import search_baidu_intern_jobs

        captured = {}

        class FakeClient:
            def get(self, url, headers=None):
                captured["url"] = url
                # Minimal __INITIAL_DATA__ with one INTERN row
                payload = _json.dumps({
                    "listData": {
                        "recruitType": "INTERN",
                        "listDetailData": [{
                            "name": "测试-暑期实习生(J99999)",
                            "postId": "abc123",
                            "projectType": "暑期实习项目",
                            "workPlace": "北京",
                        }],
                    },
                })
                html = f"<script>window.__INITIAL_DATA__ = {payload};</script>"
                class R:
                    status_code = 200
                    text = html
                return R()

        result = search_baidu_intern_jobs(
            keyword="AI", limit=5, client=FakeClient(),  # type: ignore[arg-type]
        )
        assert "recruitType=INTERN" in captured["url"]
        assert result.status == "ok"
        assert result.source == "baidu_intern"
        assert len(result.jobs) == 1
        assert result.jobs[0].extras["project_type"] == "暑期实习项目"
        assert classify_recruit_type({
            "source": result.jobs[0].source,
            "title": result.jobs[0].title,
            "extras_json": _json.dumps(result.jobs[0].extras, ensure_ascii=False),
        }) == SUMMER_INTERN


# ─────────────────── /recommended filter integration ───────────────────

@pytest.fixture
def w17_client(tmp_path):
    """Web client + a store seeded with one job per recruit_type."""
    from offerguide.skills import discover_skills
    from offerguide.ui.notify import ConsoleNotifier

    store = offerguide.Store(tmp_path / "w17.db")
    store.init_schema()
    harness_schema.init_harness_schema(store)
    skills = discover_skills(Path(__file__).parent.parent / "src/offerguide/skills")
    s = Settings(deepseek_api_key="", default_model="stub")
    app = create_app(
        settings=s, store=store, profile=None, skills=skills,
        runtime=None, notifier=ConsoleNotifier(),
    )
    # Seed one job per recruit_type, all unscored, all unapplied
    seeds = [
        ("baidu_intern", "agent训练工程环境(J99851)", "百度",
            {"project_type": "暑期实习项目", "source_verified": True}),
        ("baidu_intern", "内容营销实习生", "百度",
            {"project_type": "日常实习项目", "source_verified": True}),
        ("baidu_campus", "校招-工程师", "百度",
            {"project_type": "校招", "source_verified": True}),
        ("tencent_social", "AI Agent 应用架构师", "腾讯", {}),
        ("nowcoder",       "未分类岗位", "某公司", {}),
    ]
    with store.connect() as conn:
        for i, (src, title, company, extras) in enumerate(seeds):
            conn.execute(
                "INSERT INTO jobs (source, title, company, raw_text, extras_json, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (src, title, company, "x" * 200,
                 _json.dumps(extras, ensure_ascii=False), f"h_w17_{i}"),
            )
        conn.commit()
    return TestClient(app), store


class TestRecommendedFilter:
    def test_default_filter_shows_only_interns(self, w17_client):
        """Default ?type=intern → 暑期 + 日常 only, hides campus/social/unknown."""
        client, _ = w17_client
        resp = client.get("/recommended")
        assert resp.status_code == 200
        text = resp.text
        # Filter buttons render
        assert "🌟 实习总览" in text
        assert "☀ 暑期实习" in text
        assert "📅 日常实习" in text
        assert "🎓 校招正式" in text
        assert "💼 社招" in text
        # Pills on cards
        assert "rt-pill summer_intern" in text
        assert "rt-pill daily_intern" in text
        # Default filter = intern: campus/social cards hidden
        assert "校招-工程师" not in text
        assert "AI Agent 应用架构师" not in text
        # Intern cards visible
        assert "agent训练工程环境" in text
        assert "内容营销实习生" in text

    def test_filter_summer_only(self, w17_client):
        client, _ = w17_client
        resp = client.get("/recommended?type=summer")
        assert resp.status_code == 200
        text = resp.text
        assert "agent训练工程环境" in text  # 暑期实习项目
        assert "内容营销实习生" not in text  # 日常实习项目, filtered out

    def test_filter_daily_only(self, w17_client):
        client, _ = w17_client
        resp = client.get("/recommended?type=daily")
        assert resp.status_code == 200
        text = resp.text
        assert "内容营销实习生" in text
        assert "agent训练工程环境" not in text

    def test_filter_fulltime(self, w17_client):
        client, _ = w17_client
        resp = client.get("/recommended?type=fulltime")
        assert resp.status_code == 200
        text = resp.text
        assert "校招-工程师" in text
        assert "agent训练工程环境" not in text

    def test_filter_social(self, w17_client):
        client, _ = w17_client
        resp = client.get("/recommended?type=social")
        assert resp.status_code == 200
        assert "AI Agent 应用架构师" in resp.text

    def test_filter_all_shows_everything(self, w17_client):
        client, _ = w17_client
        resp = client.get("/recommended?type=all")
        assert resp.status_code == 200
        text = resp.text
        for needle in (
            "agent训练工程环境", "内容营销实习生",
            "校招-工程师", "AI Agent 应用架构师", "未分类岗位",
        ):
            assert needle in text, f"'{needle}' missing in ?type=all"

    def test_active_button_marked(self, w17_client):
        client, _ = w17_client
        resp = client.get("/recommended?type=summer")
        # The active pill should have class 'active'
        assert 'href="/recommended?type=summer" class="active"' in resp.text

    def test_invalid_filter_falls_back_to_all(self, w17_client):
        """Garbage ?type= shouldn't 500."""
        client, _ = w17_client
        resp = client.get("/recommended?type=zzz_invalid")
        assert resp.status_code == 200
