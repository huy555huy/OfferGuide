"""Platform adapter tests — pure functions over fixture HTML/XML, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from offerguide.platforms import RawJob, canonical_text, content_hash, manual, nowcoder

FIXTURES = Path(__file__).parent / "fixtures" / "nowcoder"


# ---- nowcoder sitemap parsing --------------------------------------------


def test_parse_sitemap_root_lists_sub_sitemaps() -> None:
    locs = nowcoder.parse_sitemap_locs((FIXTURES / "sitemap_root.xml").read_text())
    assert locs == [
        "https://www.nowcoder.com/sitemap1.xml",
        "https://www.nowcoder.com/sitemap2.xml",
    ]


def test_parse_sitemap_nowpick_lists_sub_files() -> None:
    locs = nowcoder.parse_sitemap_locs((FIXTURES / "sitemap_nowpick.xml").read_text())
    assert locs == ["https://www.nowcoder.com/nowpick/sitemap1.xml"]


def test_parse_nowpick1_yields_jd_and_enterprise_urls() -> None:
    locs = nowcoder.parse_sitemap_locs((FIXTURES / "sitemap_nowpick1.xml").read_text())
    assert len(locs) == 891
    jds = nowcoder.filter_jd_urls(locs)
    # We expect roughly half/half — but at minimum >= 100 JD URLs in the snapshot.
    assert len(jds) >= 100
    # All filtered URLs must match the JD pattern
    for u in jds:
        assert nowcoder.JD_URL_RE.match(u), u


def test_jd_url_to_id_extracts_numeric_id() -> None:
    assert (
        nowcoder.jd_url_to_id("https://www.nowcoder.com/jobs/detail/446211?urlSource=sitemap")
        == "446211"
    )
    assert nowcoder.jd_url_to_id("https://www.nowcoder.com/discuss/123") is None


# ---- nowcoder JD page parsing --------------------------------------------


def test_parse_jd_html_extracts_structured_fields() -> None:
    html = (FIXTURES / "jd_446211.html").read_text(encoding="utf-8")
    rj = nowcoder.parse_jd_html(html, url="https://www.nowcoder.com/jobs/detail/446211")

    assert rj.source == "nowcoder"
    assert rj.source_id == "446211"
    assert rj.title == "AI Agent"
    assert rj.location == "北京"
    assert rj.url == "https://www.nowcoder.com/jobs/detail/446211"

    # raw_text must include both responsibilities and requirements verbatim
    assert "## 岗位职责" in rj.raw_text
    assert "## 任职要求" in rj.raw_text
    assert "Transformer" in rj.raw_text
    assert "本科及以上学历" in rj.raw_text

    # extras carry the platform-native useful metadata
    assert rj.extras["salaryMin"] == 15
    assert rj.extras["salaryMax"] == 35
    assert rj.extras["salaryMonth"] == 16
    assert rj.extras["careerJobName"] == "算法工程师"
    assert rj.extras["industryName"] == "电商"
    assert rj.extras["graduationYear"] == "2027届"
    assert "avgProcessRate" in rj.extras  # 平台公布的回复率 — 后续 reply rate prior 用


def test_parse_jd_html_raises_when_initial_state_missing() -> None:
    with pytest.raises(ValueError, match="__INITIAL_STATE__"):
        nowcoder.parse_jd_html("<html><body>no SPA state here</body></html>")


def test_parse_jd_html_raises_when_state_lacks_detail() -> None:
    fake = (
        "<html><script>"
        'window.__INITIAL_STATE__ = {"store": {"jobDetail": {}}};'
        "</script></html>"
    )
    with pytest.raises(ValueError, match="store.jobDetail.detail missing"):
        nowcoder.parse_jd_html(fake)


# ---- manual paste --------------------------------------------------------


def test_manual_from_text_wraps_pasted_jd() -> None:
    rj = manual.from_text("Backend Intern\n\n负责后端 API 开发", company="ByteDance", location="Shanghai")
    assert rj.source == "manual"
    assert rj.title == "Backend Intern"
    assert rj.company == "ByteDance"
    assert rj.location == "Shanghai"
    assert "API 开发" in rj.raw_text


def test_manual_from_text_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Empty"):
        manual.from_text("   \n  ")


def test_manual_from_text_guesses_title_when_omitted() -> None:
    rj = manual.from_text("数据科学实习生\n岗位职责：...")
    assert rj.title == "数据科学实习生"


def test_manual_from_url_routes_unknown_host_to_error() -> None:
    with pytest.raises(NotImplementedError):
        manual.from_url("https://www.example.com/some-jd")


# ---- canonical text + content hash ---------------------------------------


def test_canonical_text_includes_label_lines() -> None:
    rj = RawJob(
        source="manual",
        title="数据科学实习生",
        company="ByteDance",
        location="Shanghai",
        raw_text="负责数据 ETL",
    )
    txt = canonical_text(rj)
    assert "标题: 数据科学实习生" in txt
    assert "公司: ByteDance" in txt
    assert "地点: Shanghai" in txt
    assert "负责数据 ETL" in txt


def test_content_hash_stable_for_same_inputs() -> None:
    a = RawJob(source="x", title="t", raw_text="body")
    b = RawJob(source="x", title="t", raw_text="body")
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_when_text_changes() -> None:
    a = RawJob(source="x", title="t", raw_text="body")
    b = RawJob(source="x", title="t", raw_text="body!")
    assert content_hash(a) != content_hash(b)
