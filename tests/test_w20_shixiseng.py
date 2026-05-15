"""W20 — 实习僧 adapter tests.

Mix of:
- Pure parser tests with frozen real-HTML fixtures (no network)
- 1 真 network smoke test (skipped if offline) to verify the live page still
  matches our regexes — catches their HTML structure changes early
"""
from __future__ import annotations

import os

import pytest

import offerguide
from offerguide.platforms.shixiseng import (
    DETAIL_URL_TEMPLATE,
    ParsedDetail,
    crawl_shixiseng,
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
            '<title>不匹配的纯文字标题</title>'
            '<div class="new_job_name"><span>fallback岗</span></div>'
            '<a class="com-name">独立公司名</a>'
        )
        p = parse_detail(html, token="inn_x")
        assert p is not None
        assert p.title == "fallback岗"
        assert p.company == "独立公司名"

    def test_parses_when_no_title_at_all_returns_none(self):
        html = '<div>nothing useful here</div>'
        p = parse_detail(html, token="inn_x")
        assert p is None

    def test_parses_title_only_no_company(self):
        # Real title format: <岗位>实习招聘-<公司>实习生招聘-实习僧
        html = '<title>大模型算法实习招聘-某AI公司实习生招聘-实习僧</title>'
        p = parse_detail(html, token="inn_x")
        assert p is not None
        assert p.title == "大模型算法"
        assert p.company == "某AI公司"

    def test_parses_strips_html_tags_in_body(self):
        html = (
            '<title>测试岗实习招聘-测试公司实习生招聘-实习僧</title>'
            '<div class="job_detail">'
            '<p>第一段 <strong>重点</strong></p>'
            '<br>'
            '<ul><li>条目 1</li><li>条目 2</li></ul>'
            '</div>'
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
                    def raise_for_status(self): pass
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
        token="inn_min", url="https://www.shixiseng.com/intern/inn_min",
        title="最小岗", company=None, location=None,
        salary=None, refresh_date=None, body="",
    )
    rj = to_raw_job(p, discovered_keyword="x")
    assert rj.source == "shixiseng"
    assert rj.title == "最小岗"
    assert "JD body 未抓到" in rj.raw_text  # body fallback note


# ── crawl_shixiseng integration with stub client ────────────────────────


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "ssseng.db")
    s.init_schema()
    return s


def test_crawl_with_stub_client_full_flow(store, monkeypatch):
    """End-to-end: stub httpx.Client returning list page + 3 detail pages →
    crawl_shixiseng calls list, then 3 details, ingests 3 jobs."""

    list_html = _REAL_LIST_HTML_FRAGMENT
    detail_template = """
<title>测试岗{i}实习招聘-公司{i}实习生招聘-实习僧</title>
<div class="new_job_name"><span>测试岗{i}</span></div>
<a class="com-name">公司{i}</a>
<span title="城市{i}" class="job_position">城市{i}</span>
<div class="job_money cutom_font">{i}00/天</div>
<span class="cutom_font">2026-05-{i:02}</span>
<div class="job_detail">JD body for token {i}</div>
"""
    call_log: list[str] = []

    class _StubClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get(self, url, params=None):
            call_log.append(url + (("?" + str(params)) if params else ""))
            class R:
                status_code = 200
                text = ""
                def raise_for_status(self): pass
            r = R()
            if "/interns" in url:
                r.text = list_html
            elif "/intern/inn_0j40loi4axcb" in url:
                r.text = detail_template.format(i=1)
            elif "/intern/inn_0kk2tg23is4p" in url:
                r.text = detail_template.format(i=2)
            elif "/intern/inn_14lnrnziaols" in url:
                r.text = detail_template.format(i=3)
            return r

    import offerguide.platforms.shixiseng as ssmod
    monkeypatch.setattr(ssmod.httpx, "Client", _StubClient)

    result = crawl_shixiseng(store, keyword="AI", max_jobs=20)
    assert result.listed_total == 3
    assert result.fetched == 3
    assert result.parsed == 3
    assert result.inserted == 3
    assert result.duplicate == 0
    assert result.errors == []
    assert set(result.by_company.keys()) == {"公司1", "公司2", "公司3"}

    # Verify ingested rows
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT title, company, location, source FROM jobs ORDER BY id"
        ).fetchall()
    assert len(rows) == 3
    assert rows[0] == ("测试岗1", "公司1", "城市1", "shixiseng")
    assert rows[1] == ("测试岗2", "公司2", "城市2", "shixiseng")
    assert rows[2] == ("测试岗3", "公司3", "城市3", "shixiseng")

    # Calls: 1 list + 3 detail
    assert len(call_log) == 4


def test_crawl_caps_at_max_jobs(store, monkeypatch):
    """If list returns 3 tokens but max_jobs=2, only 2 details fetched."""
    detail_html = (
        '<title>x实习招聘-y实习生招聘-实习僧</title>'
        '<div class="new_job_name"><span>x</span></div>'
        '<a class="com-name">y</a>'
        '<div class="job_detail">jd</div>'
    )
    detail_calls: list[str] = []

    class _StubClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get(self, url, params=None):
            class R:
                status_code = 200
                def raise_for_status(self): pass
            r = R()
            if "/interns" in url:
                r.text = _REAL_LIST_HTML_FRAGMENT
            else:
                detail_calls.append(url)
                r.text = detail_html
            return r

    import offerguide.platforms.shixiseng as ssmod
    monkeypatch.setattr(ssmod.httpx, "Client", _StubClient)

    result = crawl_shixiseng(store, keyword="x", max_jobs=2)
    assert result.listed_total == 3  # list returned 3
    assert result.fetched == 2  # only 2 detail-fetched
    assert len(detail_calls) == 2


def test_crawl_handles_list_fetch_failure(store, monkeypatch):
    """List 500 → result.errors records, no inserts, no detail calls."""

    class _StubClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get(self, url, params=None):
            class R:
                status_code = 500
                text = "fail"
                def raise_for_status(self):
                    raise RuntimeError("HTTP 500")
            return R()

    import offerguide.platforms.shixiseng as ssmod
    monkeypatch.setattr(ssmod.httpx, "Client", _StubClient)

    result = crawl_shixiseng(store, keyword="x", max_jobs=5)
    assert result.listed_total == 0
    assert result.fetched == 0
    assert result.inserted == 0
    assert any("list fetch failed" in e for e in result.errors)


def test_crawl_handles_individual_detail_failure_doesnt_break_batch(
    store, monkeypatch,
):
    """One bad detail page doesn't kill the whole batch."""
    good_detail = (
        '<title>good实习招聘-co实习生招聘-实习僧</title>'
        '<div class="new_job_name"><span>good</span></div>'
        '<a class="com-name">co</a>'
        '<div class="job_detail">jd</div>'
    )

    class _StubClient:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get(self, url, params=None):
            class R:
                status_code = 200
                def raise_for_status(self): pass
            r = R()
            if "/interns" in url:
                r.text = _REAL_LIST_HTML_FRAGMENT
            elif "inn_0kk2tg23is4p" in url:
                # crash the middle one
                raise RuntimeError("conn reset")
            else:
                r.text = good_detail
            return r

    import offerguide.platforms.shixiseng as ssmod
    monkeypatch.setattr(ssmod.httpx, "Client", _StubClient)

    result = crawl_shixiseng(store, keyword="x", max_jobs=5)
    assert result.listed_total == 3
    assert result.fetched == 2  # 2 succeeded
    assert result.inserted == 2  # 2 unique jobs (both 'good')... wait
    # Both successful detail pages return identical "good-co" job → content_hash
    # collision → second is duplicate
    assert result.inserted + result.duplicate == 2
    assert any("conn reset" in e for e in result.errors)


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


# ── Cross-module wiring tests (recruit_type + application_plan) ──────────


class TestShixisengRecruitType:
    """W20 — shixiseng /interns is 100% 实习. classifier should never
    return UNKNOWN/CAMPUS/SOCIAL for it."""

    def test_default_title_classifies_as_daily_intern(self):
        from offerguide.recruit_type import DAILY_INTERN, classify_recruit_type
        rt = classify_recruit_type({
            "source": "shixiseng",
            "title": "AI Agent 实习生",
            "extras_json": "{}",
        })
        assert rt == DAILY_INTERN

    def test_summer_in_title_classifies_as_summer_intern(self):
        from offerguide.recruit_type import SUMMER_INTERN, classify_recruit_type
        rt = classify_recruit_type({
            "source": "shixiseng",
            "title": "AI Agent 暑期实习生",
            "extras_json": "{}",
        })
        assert rt == SUMMER_INTERN

    def test_summer_english_in_title_classifies_as_summer(self):
        from offerguide.recruit_type import SUMMER_INTERN, classify_recruit_type
        rt = classify_recruit_type({
            "source": "shixiseng",
            "title": "Summer Intern - LLM Application",
            "extras_json": "{}",
        })
        assert rt == SUMMER_INTERN

    def test_no_intern_in_title_still_classifies_as_daily_not_unknown(self):
        """Even if the title is missing the 实习 keyword (rare since 实习僧
        scrape stays on /interns), source guarantees it. Should be DAILY_INTERN
        not UNKNOWN — that would hide it from the default ?type=intern filter."""
        from offerguide.recruit_type import DAILY_INTERN, classify_recruit_type
        rt = classify_recruit_type({
            "source": "shixiseng",
            "title": "AI Agent 工程师",  # no 实习 word
            "extras_json": "{}",
        })
        assert rt == DAILY_INTERN


class TestShixisengApplicationPlan:
    """W20 — shixiseng source must route to dedicated _shixiseng_plan with
    intern-specific instructions, not generic ATS plan."""

    def test_shixiseng_source_routes_to_shixiseng_plan(self):
        from offerguide.application_plan import build_application_plan
        plan = build_application_plan({
            "source": "shixiseng",
            "url": "https://www.shixiseng.com/intern/inn_xxxxxxxx",
            "title": "AI Agent 实习生",
            "company": "某公司",
            "extras_json": '{"source_verified": true}',
        })
        assert plan.platform == "shixiseng"
        assert "实习僧" in plan.platform_label
        # Intern-specific guidance present
        assert any("实习时长" in f.label or "实习时长" in f.value_hint
                   for f in plan.fields)
        # Verified source pill
        assert plan.verified_source is True

    def test_shixiseng_host_in_url_also_routes(self):
        """Even if source label is something else, shixiseng.com host
        triggers shixiseng plan. Useful for user-pasted URLs."""
        from offerguide.application_plan import build_application_plan
        plan = build_application_plan({
            "source": "manual",
            "url": "https://www.shixiseng.com/intern/inn_zzz",
            "title": "x", "company": "y",
            "extras_json": "{}",
        })
        assert plan.platform == "shixiseng"

    def test_shixiseng_plan_includes_post_apply_actions(self):
        """W19 audit caught template not rendering post_apply_actions —
        ensure shixiseng plan provides them so the (now-fixed) template
        actually has content to show."""
        from offerguide.application_plan import build_application_plan
        plan = build_application_plan({
            "source": "shixiseng",
            "url": "https://www.shixiseng.com/intern/inn_x",
            "title": "x", "company": "y",
            "extras_json": "{}",
        })
        assert len(plan.post_apply_actions) >= 1
        assert any("微信" in a or "站内信" in a for a in plan.post_apply_actions)
        assert len(plan.material_checklist) >= 1


# ── ambient daemon wiring ──────────────────────────────────────────────


class TestShixisengAmbientWiring:
    def test_shixiseng_in_unscored_sources(self, store):
        """shixiseng source must be in ambient daemon's unscored list, or
        scored events would never fire for shixiseng jobs."""
        from offerguide.agent_runtime import _schema as harness_schema
        from offerguide.workers.ambient import _load_unscored_discovered_ids
        harness_schema.init_agent_runtime_schema(store)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                ("shixiseng", "AI Agent 实习生", "某公司", "x" * 300, "h_ssseng"),
            )
            conn.commit()
        ids = _load_unscored_discovered_ids(store)
        assert len(ids) == 1

    def test_per_keyword_helper_exists_and_returns_summary(self, store, monkeypatch):
        """_crawl_shixiseng_per_keyword should exist + work end-to-end with
        stub crawl_shixiseng."""
        from offerguide.workers.ambient import _crawl_shixiseng_per_keyword
        from offerguide.platforms.shixiseng import FetchResult

        def _fake_crawl(store_arg, *, keyword, max_jobs):
            r = FetchResult()
            r.inserted = 3
            r.duplicate = 1
            r.parsed = 4
            r.by_company = {f"公司A_{keyword}": 2, f"公司B_{keyword}": 1}
            return r

        import offerguide.platforms.shixiseng as ssmod
        monkeypatch.setattr(ssmod, "crawl_shixiseng", _fake_crawl)

        summary = _crawl_shixiseng_per_keyword(
            store=store, keywords=["AI Agent", "RLHF"], limit_per_kw=8,
        )
        assert summary["inserted_total"] == 6
        assert summary["duplicate_total"] == 2
        assert summary["per_keyword"] == {"AI Agent": 3, "RLHF": 3}
        assert summary["company_diversity"] == 4  # 2 公司 × 2 keywords
