"""W22 — job discovery quality gates."""
from __future__ import annotations

import json as _json

import offerguide
from offerguide.job_quality import (
    normalize_known_job_url,
    quality_verdict_for_job,
)
from offerguide.platforms import RawJob
from offerguide.workers import scout


def test_tencent_campus_old_jobdesc_url_normalizes_to_live_detail_page():
    old = "https://join.qq.com/jobdesc.html?postId=1216462959547938816"
    assert normalize_known_job_url(source="tencent_campus", url=old) == (
        "https://join.qq.com/post_detail.html?postid=1216462959547938816"
    )


def test_init_schema_repairs_existing_tencent_campus_urls(tmp_path):
    store = offerguide.Store(tmp_path / "quality.db")
    store.init_schema()
    old = "https://join.qq.com/jobdesc.html?postId=1234"
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, source_id, url, title, raw_text, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("tencent_campus", "1234", old, "AI 实习", "x" * 300, "h_old_tencent"),
        )

    store.init_schema()

    with store.connect() as conn:
        url, extras_json = conn.execute(
            "SELECT url, extras_json FROM jobs WHERE source_id = '1234'",
        ).fetchone()
    assert url == "https://join.qq.com/post_detail.html?postid=1234"
    assert "repair_tencent_campus_url" in extras_json


def test_nowcoder_off_target_platform_job_is_rejected(tmp_path):
    store = offerguide.Store(tmp_path / "quality.db")
    store.init_schema()
    rj = RawJob(
        source="nowcoder",
        source_id="447310",
        url="https://www.nowcoder.com/jobs/detail/447310?urlSource=sitemap",
        title="考研/保研辅导",
        company="魁星科技",
        raw_text="职业方向: 辅导教师\n行业: 企业服务\n岗位职责: 负责学生辅导",
        extras={"careerJobName": "辅导教师", "industryName": "企业服务"},
    )

    was_new, job_id = scout.ingest(store, rj)

    assert was_new is False
    assert job_id == 0
    with store.connect() as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
    assert count == 0


def test_nowcoder_tech_job_with_soft_block_word_is_kept():
    verdict = quality_verdict_for_job(
        source="nowcoder",
        title="智能客服算法工程师",
        url="https://www.nowcoder.com/jobs/detail/1",
        raw_text="负责大模型对话算法、RAG、推荐策略和 Python 工程实现",
        extras_json=_json.dumps({
            "careerJobName": "算法工程师",
            "industryName": "互联网",
        }, ensure_ascii=False),
    )
    assert verdict.usable is True


def test_zerovoice_wechat_article_is_not_recommendable():
    verdict = quality_verdict_for_job(
        source="zerovoice_repo",
        title="AI Agent 高级研发工程师",
        url="https://mp.weixin.qq.com/s?__biz=test",
        extras_json=_json.dumps({"link_type": "wechat_article"}),
    )
    assert verdict.usable is False
    assert verdict.reason == "not_direct_apply_link"
