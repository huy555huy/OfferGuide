from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import offerguide
from offerguide.application_plan import build_application_plan
from offerguide.config import Settings
from offerguide.harness import _schema as harness_schema
from offerguide.ui.web import create_app
from offerguide.workers.ambient import _load_unscored_discovered_ids


def test_boss_application_plan_prepares_chat_path() -> None:
    plan = build_application_plan({
        "source": "boss_extension",
        "url": "https://www.zhipin.com/job_detail/abc.html",
        "title": "AI Agent 实习",
        "company": "字节",
    })

    assert plan.platform == "boss_zhipin"
    assert plan.action_label == "打开 BOSS 沟通"
    assert any(f.label == "沟通开场白" and f.copyable for f in plan.fields)
    assert any("点击发送" in step for step in plan.steps)


def test_official_application_plan_prepares_form_fields() -> None:
    plan = build_application_plan({
        "source": "user_paste_url",
        "url": "https://careers.example.com/jobs/123",
        "title": "算法工程师",
        "company": "样例科技",
    })

    assert plan.platform == "official_site"
    assert plan.action_label == "打开官网网申"
    labels = [f.label for f in plan.fields]
    assert "姓名 / 手机 / 邮箱" in labels
    assert "求职动机 / Why us" in labels
    assert plan.verified_source is False


def test_verified_official_application_plan_shows_evidence() -> None:
    plan = build_application_plan({
        "source": "tencent_social",
        "url": "https://careers.tencent.com/jobdesc.html?postId=2035",
        "title": "AI Agent 应用架构工程师",
        "company": "腾讯",
        "extras_json": (
            '{"source_verified": true, '
            '"evidence_url": "https://careers.tencent.com/tencentcareer/api/post/Query"}'
        ),
    })

    assert plan.platform == "official_site"
    assert plan.verified_source is True
    assert "已核验来源" in plan.platform_label
    assert plan.evidence_url == "https://careers.tencent.com/tencentcareer/api/post/Query"


def test_apply_pack_no_llm_still_renders_real_application_plan(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    app = create_app(
        settings=Settings(deepseek_api_key="", db_path=tmp_path / "ui.db"),
        store=store,
        profile=None,
        skills=[],
        runtime=None,
    )
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, url, title, company, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                "user_paste_url",
                "https://careers.example.com/jobs/123",
                "算法工程师",
                "样例科技",
                "完整 JD 全文" * 40,
                "h_official_plan",
            ),
        )

    resp = TestClient(app).get("/jobs/1/apply-pack")

    assert resp.status_code == 200
    assert "投递路径核验" in resp.text
    assert "官网 / ATS 网申" in resp.text
    assert "页面要填的东西" in resp.text
    assert "姓名 / 手机 / 邮箱" in resp.text


def test_ambient_unscored_queue_includes_agent_search_and_skips_thin(tmp_path: Path) -> None:
    store = offerguide.Store(tmp_path / "ambient.db")
    store.init_schema()
    harness_schema.init_harness_schema(store)
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('baidu_campus', '官网岗', '样例科技', ?, 'h_baidu_campus')",
            ("x" * 220,),
        )
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('boss_extension_list', '列表薄岗', '样例科技', ?, 'h_thin_list')",
            ("x" * 80,),
        )
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('manual', '手动岗', '样例科技', ?, 'h_manual_skip')",
            ("x" * 220,),
        )

    ids = _load_unscored_discovered_ids(store)

    assert ids == [1]
