"""W20 — 实习僧 adapter tests.

Mix of:
- Pure parser tests with frozen real-HTML fixtures (no network)
- 1 真 network smoke test (skipped if offline) to verify the live page still
  matches our regexes — catches their HTML structure changes early
"""

from __future__ import annotations

import os

import pytest

from offerguide.platforms.shixiseng import (
    DETAIL_URL_TEMPLATE,
    ParsedDetail,
    fetch_list_tokens,
    parse_detail,
    to_raw_job,
)

# Frozen real-HTML fragment from /intern/inn_0j40loi4axcb (probed 2026-05-11).
# Trimmed to just the parts our regex looks at, avoid 500KB test file.
_REAL_DETAIL_HTML_FRAGMENT = """
<html><head>
<title>ai动画师实习招聘-耀漫文化实习生招聘-实习僧</title>
</head><body>
<div class="job-header">
<div class="new_job_name" data-v-eba2f908><span data-v-eba2f908>ai动画师</span></div>
<div class="job_date" data-v-eba2f908>
<span class="cutom_font" data-v-eba2f908>2026-05-02 19:43:00</span>
<span data-v-eba2f908>刷新</span>
</div>
<div class="job_msg" data-v-eba2f908>
<span class="job_money cutom_font" data-v-eba2f908>100-200/天</span>
<span title="杭州" class="job_position" data-v-eba2f908>杭州</span>
</div>
</div>
<div class="job-about">
<div class="con-job">
<div class="com_intro">
<a href="javascript:;" class="com-name" data-v-eba2f908>耀漫文化</a>
</div>
</div>
</div>
<div class="job_detail" data-v-eba2f908>
岗位职责： 1. 负责AI动态漫/短剧的AI生帧、画面修正与动画制作；
2. 熟练使用主流AI动画/绘图工具，解决生成画面的穿模、变形、连贯性问题。
任职要求： ✅ 会用AI生图/生帧；✅ 懂镜头、有审美；
大专学历及以上，接受实习生
</div>
</body></html>
"""


# Frozen list-page snippet — just the bit with intern tokens.
_REAL_LIST_HTML_FRAGMENT = """
<html><body>
<a href="/intern/inn_0j40loi4axcb">job1</a>
<a href="/intern/inn_0kk2tg23is4p">job2</a>
<a href="/intern/inn_14lnrnziaols">job3</a>
<a href="/intern/inn_0j40loi4axcb">dup</a>
</body></html>
"""


# ── parser tests ────────────────────────────────────────────────────────


class TestParseDetail:
    def test_parses_real_detail_html_full(self):
        p = parse_detail(_REAL_DETAIL_HTML_FRAGMENT, token="inn_0j40loi4axcb")
        assert p is not None
        assert p.token == "inn_0j40loi4axcb"
        assert p.url == "https://www.shixiseng.com/intern/inn_0j40loi4axcb"
        # Title comes from <title> split: "ai动画师-耀漫文化实习生招聘-实习僧"
        assert p.title == "ai动画师"
        assert p.company == "耀漫文化"
        assert p.location == "杭州"
        assert p.salary == "100-200/天"
        assert p.refresh_date == "2026-05-02 19:43:00"
        assert "AI动态漫" in p.body
        assert "任职要求" in p.body
        assert "大专学历" in p.body

    def test_parses_when_title_tag_missing_falls_back_to_div(self):
        html = '<div class="new_job_name"><span>fallback岗位</span></div>'
        p = parse_detail(html, token="inn_xxxxxxxx")
        assert p is not None
        assert p.title == "fallback岗位"
        assert p.company is None  # no com-name in this fragment
        assert p.location is None

    def test_parses_when_title_doesnt_match_pattern_falls_back(self):
        # <title> doesn't match the splittable pattern → fall back to
        # new_job_name + com-name divs.
        html = (
            "<title>不匹配的纯文字标题</title>"
            '<div class="new_job_name"><span>fallback岗</span></div>'
            '<a class="com-name">独立公司名</a>'
        )
        p = parse_detail(html, token="inn_x")
        assert p is not None
        assert p.title == "fallback岗"
        assert p.company == "独立公司名"

    def test_parses_when_no_title_at_all_returns_none(self):
        html = "<div>nothing useful here</div>"
        p = parse_detail(html, token="inn_x")
        assert p is None

    def test_parses_title_only_no_company(self):
        # Real title format: <岗位>实习招聘-<公司>实习生招聘-实习僧
        html = "<title>大模型算法实习招聘-某AI公司实习生招聘-实习僧</title>"
        p = parse_detail(html, token="inn_x")
        assert p is not None
        assert p.title == "大模型算法"
        assert p.company == "某AI公司"

    def test_parses_strips_html_tags_in_body(self):
        html = (
            "<title>测试岗实习招聘-测试公司实习生招聘-实习僧</title>"
            '<div class="job_detail">'
            "<p>第一段 <strong>重点</strong></p>"
            "<br>"
            "<ul><li>条目 1</li><li>条目 2</li></ul>"
            "</div>"
        )
        p = parse_detail(html, token="inn_x")
        assert p is not None
        assert "重点" in p.body
        assert "条目 1" in p.body
        assert "<p>" not in p.body
        assert "<strong>" not in p.body


# ── list parsing ────────────────────────────────────────────────────────


class TestListTokenExtraction:
    """fetch_list_tokens uses regex against /intern/inn_xxx — test
    that the regex picks them up + dedups + preserves order."""

    def test_token_dedup_and_order(self):
        # Use a fake httpx.Client to avoid network
        class _C:
            def get(self, url, params=None):
                class R:
                    status_code = 200
                    text = _REAL_LIST_HTML_FRAGMENT

                    def raise_for_status(self):
                        pass

                return R()

        tokens = fetch_list_tokens("AI", page=1, client=_C())  # type: ignore[arg-type]
        # 4 anchors but only 3 unique, in original order
        assert tokens == [
            "inn_0j40loi4axcb",
            "inn_0kk2tg23is4p",
            "inn_14lnrnziaols",
        ]


# ── to_raw_job ──────────────────────────────────────────────────────────


def test_to_raw_job_full_fields():
    p = ParsedDetail(
        token="inn_test",
        url=DETAIL_URL_TEMPLATE.format(token="inn_test"),
        title="LLM 应用实习",
        company="某 AI 公司",
        location="北京",
        salary="200-300/天",
        refresh_date="2026-05-10 09:00:00",
        body="负责开发 RAG 系统\n要求 Python 熟练",
    )
    rj = to_raw_job(p, discovered_keyword="AI Agent")
    assert rj.source == "shixiseng"
    assert rj.source_id == "inn_test"
    assert rj.url == "https://www.shixiseng.com/intern/inn_test"
    assert rj.title == "LLM 应用实习"
    assert rj.company == "某 AI 公司"
    assert rj.location == "北京"
    assert "RAG" in rj.raw_text
    assert "200-300/天" in rj.raw_text
    assert "投递入口" in rj.raw_text
    assert rj.extras["discovered_via"] == "shixiseng_list_search"
    assert rj.extras["discovered_keyword"] == "AI Agent"
    assert rj.extras["salary"] == "200-300/天"
    assert rj.extras["source_verified"] is True
    assert "shixiseng.com/interns?keyword=AI Agent" in rj.extras["evidence_url"]


def test_to_raw_job_handles_missing_optional_fields():
    p = ParsedDetail(
        token="inn_min",
        url="https://www.shixiseng.com/intern/inn_min",
        title="最小岗",
        company=None,
        location=None,
        salary=None,
        refresh_date=None,
        body="",
    )
    rj = to_raw_job(p, discovered_keyword="x")
    assert rj.source == "shixiseng"
    assert rj.title == "最小岗"
    assert "JD body 未抓到" in rj.raw_text  # body fallback note


# ── Live network smoke test (skipped offline) ────────────────────────────


@pytest.mark.skipif(
    os.environ.get("OFFERGUIDE_RUN_NETWORK_TESTS") != "1",
    reason="Live shixiseng network test (set OFFERGUIDE_RUN_NETWORK_TESTS=1 to run)",
)
def test_live_list_fetch_returns_tokens():
    """Real network: confirms 实习僧 list page still returns SSR HTML
    with /intern/<token> links matching our regex. Catches their HTML
    structure changes."""
    tokens = fetch_list_tokens("AI", page=1)
    assert len(tokens) >= 5, f"expected ≥5 tokens, got {len(tokens)}"
    for t in tokens:
        assert t.startswith("inn_")
        assert len(t) >= 12
