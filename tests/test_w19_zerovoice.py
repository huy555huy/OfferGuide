"""W19+ — 0voice GitHub repo aggregator tests."""
from __future__ import annotations

import json as _json

import pytest

import offerguide
from offerguide.application_plan import build_application_plan
from offerguide.harness import _schema as harness_schema
from offerguide.platforms.zerovoice import (
    ParsedJob,
    crawl_zerovoice,
    link_is_official_ats,
    parse_readme,
    to_raw_job,
)

# ─────────────────── parser tests (deterministic, on fixture markdown) ───────────────────

# Mini fixture matching real 0voice README format (verified 2026-05-11)
_FIXTURE_MD = """\
# 2026 Spring Recruitment

## 26春招岗
## 大厂岗

## <h3 id="5">阿里巴巴</h3>
|NO.|工作岗位|详细内容|
|--|--|--|
|1|AI应用算法工程师|[点击查看](https://campus-talent.alibaba.com/campus/position/199903540003)|
|2|Agent Infra工程师|[点击查看](https://campus-talent.alibaba.com/campus/position/199903480006)|
|3|研发工程师JAVA|[点击查看](https://mp.weixin.qq.com/s?__biz=Mzk0MzY5NjI3MA==&mid=2247496603)|

## <h3 id="1">腾讯</h3>
|NO.|工作岗位|详细内容|
|--|--|--|
|1|后端开发工程师|[点击查看](https://join.qq.com/post.html?postId=1234)|
|2|应届实习|[点击查看](https://mp.weixin.qq.com/s?__biz=test)|

## <h3 id="33">商汤科技</h3>
|NO.|工作岗位|详细内容|
|--|--|--|
|1|算法工程师|[点击查看](https://app.mokahr.com/campus_apply/sensetime/8765)|
"""


class TestParser:
    def test_parses_3_companies_6_jobs(self):
        jobs = parse_readme(_FIXTURE_MD)
        assert len(jobs) == 6
        companies = {j.company for j in jobs}
        assert companies == {"阿里巴巴", "腾讯", "商汤科技"}

    def test_alibaba_section_first_3_jobs_intact(self):
        jobs = parse_readme(_FIXTURE_MD)
        ali = [j for j in jobs if j.company == "阿里巴巴"]
        assert len(ali) == 3
        assert ali[0].title == "AI应用算法工程师"
        assert ali[0].url == "https://campus-talent.alibaba.com/campus/position/199903540003"
        assert ali[0].section_no == 1

    def test_handles_empty_section_between_headers(self):
        md = """## <h3 id="1">EmptyCo</h3>
no table here, just text

## <h3 id="2">RealCo</h3>
|NO.|工作岗位|详细内容|
|--|--|--|
|1|实习|[点击查看](https://realco.com/123)|
"""
        jobs = parse_readme(md)
        assert len(jobs) == 1
        assert jobs[0].company == "RealCo"

    def test_skips_separator_row(self):
        """The |--|--|--| markdown table separator should not become a job row."""
        jobs = parse_readme(_FIXTURE_MD)
        # No job has all-dash title
        assert all(not all(c == "-" for c in j.title) for j in jobs)

    def test_real_readme_structure_smoke(self):
        """Sanity smoke vs the real README — just check parse doesn't crash."""
        # Use a slightly-larger synthetic that mirrors the real README's
        # "目录导航" links pattern (not job tables) BEFORE the first section
        nav_then_jobs = (
            "## 📂 目录导航\n\n"
            "[**26春招岗**](#26春招岗)\n"
            "- [大厂岗](#大厂岗)\n"
            "  * [阿里巴巴](#5)\n\n"
            + _FIXTURE_MD
        )
        jobs = parse_readme(nav_then_jobs)
        assert len(jobs) == 6  # nav links don't accidentally get parsed


# ─────────────────── link classifier ───────────────────


class TestLinkClassifier:
    def test_alibaba_campus_talent_is_ats(self):
        assert link_is_official_ats(
            "https://campus-talent.alibaba.com/campus/position/199903540003"
        )

    def test_mokahr_is_ats(self):
        assert link_is_official_ats("https://app.mokahr.com/campus_apply/sensetime/8765")

    def test_tencent_join_is_ats(self):
        assert link_is_official_ats("https://join.qq.com/post.html?postId=1234")

    def test_wechat_article_is_not_ats(self):
        assert not link_is_official_ats(
            "https://mp.weixin.qq.com/s?__biz=Mzk0MzY5NjI3MA==&mid=2247496603"
        )

    def test_random_blog_is_not_ats(self):
        assert not link_is_official_ats("https://www.csdn.net/article/123")


# ─────────────────── to_raw_job + ingest ───────────────────


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "zv.db")
    s.init_schema()
    harness_schema.init_harness_schema(s)
    return s


class TestToRawJob:
    def test_ats_link_marks_source_verified(self):
        rj = to_raw_job(ParsedJob(
            company="阿里巴巴", title="AI应用算法工程师",
            url="https://campus-talent.alibaba.com/campus/position/199903540003",
            section_no=1,
        ))
        assert rj.source == "zerovoice_repo"
        assert rj.extras["source_verified"] is True
        assert rj.extras["link_type"] == "official_ats"
        assert rj.extras["discovered_via"] == "zerovoice_aggregator"
        assert "阿里巴巴 岗位汇总" in rj.extras["discovered_keyword"]
        assert "github.com" in rj.extras["evidence_url"]
        assert rj.url == "https://campus-talent.alibaba.com/campus/position/199903540003"
        # raw_text mentions it's an ATS link
        assert "真官方 ATS" in rj.raw_text

    def test_wechat_link_does_not_mark_verified(self):
        rj = to_raw_job(ParsedJob(
            company="腾讯", title="应届实习",
            url="https://mp.weixin.qq.com/s?__biz=test",
            section_no=2,
        ))
        assert rj.extras["source_verified"] is False
        assert rj.extras["link_type"] == "wechat_article"
        assert "公众号汇总文章" in rj.raw_text


class TestCrawlZerovoice:
    def test_crawl_with_fake_readme_ingests(self, monkeypatch, store):
        """End-to-end crawl with patched fetch — ingests the 6 fixture jobs."""
        from offerguide.platforms import zerovoice as zv
        monkeypatch.setattr(zv, "fetch_readme", lambda timeout_s=20.0: _FIXTURE_MD)

        result = crawl_zerovoice(store)
        assert result.parsed_total == 6
        assert result.inserted == 6
        assert result.duplicate == 0
        assert "阿里巴巴" in result.by_company

        # Re-run → should all be duplicates
        result2 = crawl_zerovoice(store)
        assert result2.inserted == 0
        assert result2.duplicate == 6

    def test_crawl_max_jobs_caps_intake(self, monkeypatch, store):
        from offerguide.platforms import zerovoice as zv
        monkeypatch.setattr(zv, "fetch_readme", lambda timeout_s=20.0: _FIXTURE_MD)

        result = crawl_zerovoice(store, max_jobs=2)
        # parsed_total = full README (6), but cap limits ingest to 2
        assert result.parsed_total == 6
        assert result.inserted == 2

    def test_crawl_handles_fetch_failure_gracefully(self, monkeypatch, store):
        from offerguide.platforms import zerovoice as zv

        def _boom(timeout_s=20.0):
            raise RuntimeError("network down")
        monkeypatch.setattr(zv, "fetch_readme", _boom)

        result = crawl_zerovoice(store)
        assert result.inserted == 0
        assert any("network down" in e for e in result.errors)

    def test_ingested_jobs_have_extras_in_db(self, monkeypatch, store):
        """After ingest, the discovered_via attribution lands in extras_json
        so /recommended cards can show '↳ 由 0voice repo 找到'."""
        from offerguide.platforms import zerovoice as zv
        monkeypatch.setattr(zv, "fetch_readme", lambda timeout_s=20.0: _FIXTURE_MD)

        crawl_zerovoice(store)
        with store.connect() as conn:
            row = conn.execute(
                "SELECT extras_json FROM jobs "
                "WHERE source = 'zerovoice_repo' AND company = '阿里巴巴' "
                "ORDER BY id LIMIT 1"
            ).fetchone()
        assert row is not None
        extras = _json.loads(row[0])
        assert extras.get("discovered_via") == "zerovoice_aggregator"
        assert extras.get("link_type") in ("official_ats", "wechat_article")


# ─────────────────── application_plan integration ───────────────────


class TestApplicationPlanIntegration:
    def test_alibaba_campus_talent_url_routes_to_alibaba_plan(self):
        """0voice ingest with campus-talent.alibaba.com URL → alibaba plan
        (host detection picks it up even though host is different from
        codex's W16 talent.alibaba.com)."""
        plan = build_application_plan({
            "source": "zerovoice_repo",
            "url": "https://campus-talent.alibaba.com/campus/position/199903540003",
            "title": "AI应用算法工程师", "company": "阿里巴巴",
            "extras_json": _json.dumps({
                "source_verified": True,
                "link_type": "official_ats",
            }),
        })
        assert plan.platform == "alibaba"
        assert "阿里巴巴" in plan.platform_label
        assert plan.verified_source is True

    def test_mokahr_url_routes_to_mokahr_plan(self):
        plan = build_application_plan({
            "source": "zerovoice_repo",
            "url": "https://app.mokahr.com/campus_apply/sensetime/8765",
            "title": "算法工程师", "company": "商汤科技",
            "extras_json": _json.dumps({
                "source_verified": True, "link_type": "official_ats",
            }),
        })
        assert plan.platform == "mokahr"
        assert "商汤科技" in plan.platform_label
        assert "北森 SaaS" in plan.platform_label
        assert "微信" in plan.steps[0] or "手机号" in plan.steps[0]
        assert "5MB" in " ".join(plan.material_checklist)
