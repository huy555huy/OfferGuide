"""W12-fix(b) — Bing CN + Tavily + ChainedSearch + /api/search/test.

Audit 残留: corpus_collector 默认 search backend 是 DDG, 国内大概率不通。
修复: 加 Bing CN (零配置) + Tavily (有 key 升级) + ChainedSearch fallback。
本组测试不依赖网络 (用 stub backend 验证 chain 行为 + helper 函数)。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.agentic.search import (
    BingCNSearch,
    ChainedSearch,
    DuckDuckGoSearch,
    SearchHit,
    StubSearch,
    TavilySearch,
    _parse_bing_html,
    build_default_search,
    health_check,
)
from offerguide.config import Settings
from offerguide.profile import UserProfile
from offerguide.skills import discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import _search_guidance_message, create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ═══════════════════════════════════════════════════════════════════
# Bing HTML parser
# ═══════════════════════════════════════════════════════════════════


SAMPLE_BING_BLOCK = """
<ol>
  <li class="b_algo">
    <div class="b_tpcn"><a class="tilk">junk inner anchor</a></div>
    <h2 class="">
      <a target="_blank" href="https://nowcoder.com/discuss/123">字节跳动 暑期实习面经 - 牛客</a>
    </h2>
    <div class="b_caption">
      <p class="b_lineclamp2">面试官先问 LangGraph state 设计...</p>
    </div>
  </li>
  <li class="b_algo">
    <h2><a href="https://zhihu.com/q/456">字节跳动面经 (附攻略) - 知乎</a></h2>
    <div class="b_caption"><p class="b_lineclamp1">本人 985 应届...</p></div>
  </li>
</ol>
"""


class TestBingHTMLParser:
    def test_extracts_two_results(self) -> None:
        hits = _parse_bing_html(SAMPLE_BING_BLOCK, max_results=10)
        assert len(hits) == 2

    def test_picks_h2_anchor_not_inner(self) -> None:
        # The b_tpcn inner anchor should NOT be picked as the title
        hits = _parse_bing_html(SAMPLE_BING_BLOCK, max_results=10)
        first = hits[0]
        assert first.url == "https://nowcoder.com/discuss/123"
        assert "字节跳动" in first.title
        assert "junk" not in first.title

    def test_max_results_caps(self) -> None:
        hits = _parse_bing_html(SAMPLE_BING_BLOCK, max_results=1)
        assert len(hits) == 1

    def test_snippet_truncated_to_300(self) -> None:
        big = SAMPLE_BING_BLOCK.replace(
            "面试官先问 LangGraph state 设计...",
            "x" * 500,
        )
        hits = _parse_bing_html(big, max_results=10)
        assert len(hits[0].snippet) <= 300


# ═══════════════════════════════════════════════════════════════════
# ChainedSearch
# ═══════════════════════════════════════════════════════════════════


class _FakeBackend:
    def __init__(self, name: str, hits: list[SearchHit] | None = None,
                 raises: bool = False):
        self.name = name
        self._hits = hits or []
        self._raises = raises
        self.call_count = 0

    def search(self, query: str, *, max_results: int = 10) -> list[SearchHit]:
        self.call_count += 1
        if self._raises:
            raise RuntimeError(f"{self.name} simulated failure")
        return self._hits[:max_results]


class TestChainedSearch:
    def test_first_hit_wins(self) -> None:
        a = _FakeBackend("a", hits=[
            SearchHit(title="from_a", url="https://a.com", snippet=""),
        ])
        b = _FakeBackend("b", hits=[
            SearchHit(title="from_b", url="https://b.com", snippet=""),
        ])
        chain = ChainedSearch([a, b])
        hits = chain.search("test query")
        assert len(hits) == 1
        assert hits[0].title == "from_a"
        assert b.call_count == 0  # never called because a succeeded

    def test_falls_through_empty_to_next(self) -> None:
        empty = _FakeBackend("empty", hits=[])
        good = _FakeBackend("good", hits=[
            SearchHit(title="found", url="https://x.com", snippet=""),
        ])
        chain = ChainedSearch([empty, good])
        hits = chain.search("test")
        assert len(hits) == 1
        assert hits[0].title == "found"
        assert empty.call_count == 1
        assert good.call_count == 1

    def test_swallows_exception_and_tries_next(self) -> None:
        boom = _FakeBackend("boom", raises=True)
        good = _FakeBackend("good", hits=[
            SearchHit(title="found", url="https://x.com", snippet=""),
        ])
        chain = ChainedSearch([boom, good])
        hits = chain.search("test")
        assert hits[0].title == "found"

    def test_all_empty_returns_empty(self) -> None:
        chain = ChainedSearch([
            _FakeBackend("a", hits=[]),
            _FakeBackend("b", hits=[]),
        ])
        assert chain.search("test") == []

    def test_requires_at_least_one_backend(self) -> None:
        with pytest.raises(ValueError, match="≥1 backend"):
            ChainedSearch([])


# ═══════════════════════════════════════════════════════════════════
# build_default_search factory
# ═══════════════════════════════════════════════════════════════════


class TestFactory:
    def test_default_returns_chain(self, monkeypatch) -> None:
        monkeypatch.delenv("OFFERGUIDE_SEARCH_BACKEND", raising=False)
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        backend = build_default_search()
        assert backend.name == "chain"

    def test_explicit_bing(self, monkeypatch) -> None:
        monkeypatch.setenv("OFFERGUIDE_SEARCH_BACKEND", "bing")
        backend = build_default_search()
        assert isinstance(backend, BingCNSearch)

    def test_explicit_ddg(self, monkeypatch) -> None:
        monkeypatch.setenv("OFFERGUIDE_SEARCH_BACKEND", "duckduckgo")
        backend = build_default_search()
        assert isinstance(backend, DuckDuckGoSearch)

    def test_explicit_stub(self, monkeypatch) -> None:
        monkeypatch.setenv("OFFERGUIDE_SEARCH_BACKEND", "stub")
        backend = build_default_search()
        assert isinstance(backend, StubSearch)

    def test_tavily_requires_key(self, monkeypatch) -> None:
        monkeypatch.setenv("OFFERGUIDE_SEARCH_BACKEND", "tavily")
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            build_default_search()


# ═══════════════════════════════════════════════════════════════════
# health_check
# ═══════════════════════════════════════════════════════════════════


class TestHealthCheck:
    def test_health_check_ok(self) -> None:
        backend = _FakeBackend("ok", hits=[
            SearchHit(title="t1", url="https://x.com/1", snippet="s"),
        ])
        result = health_check(backend)
        assert result["ok"] is True
        assert result["hit_count"] == 1
        assert result["sample_titles"] == ["t1"]
        assert result["error_str"] is None

    def test_health_check_empty_returns_not_ok(self) -> None:
        backend = _FakeBackend("empty", hits=[])
        result = health_check(backend)
        assert result["ok"] is False
        assert result["hit_count"] == 0

    def test_health_check_exception_caught(self) -> None:
        backend = _FakeBackend("boom", raises=True)
        result = health_check(backend)
        assert result["ok"] is False
        assert "simulated failure" in result["error_str"]


# ═══════════════════════════════════════════════════════════════════
# UI route + guidance helper
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="x")
    skills = discover_skills(SKILLS_ROOT)
    app = create_app(
        settings=Settings(), store=store, profile=profile,
        skills=skills, runtime=None, notifier=ConsoleNotifier(),
    )
    return app, store


class TestSearchTestRoute:
    def test_route_returns_4_backend_keys(self, app_setup, monkeypatch) -> None:
        app, _ = app_setup
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        # Force chain → all-empty (StubSearch) so route returns fast
        monkeypatch.setenv("OFFERGUIDE_SEARCH_BACKEND", "stub")
        # Need to also make BingCNSearch and DDG fast — patch their
        # search methods to return [] without networking
        monkeypatch.setattr(
            "offerguide.agentic.search.BingCNSearch.search",
            lambda self, q, max_results=10: [],
        )
        monkeypatch.setattr(
            "offerguide.agentic.search.DuckDuckGoSearch.search",
            lambda self, q, max_results=10: [],
        )

        resp = TestClient(app).get("/api/search/test")
        assert resp.status_code == 200
        data = resp.json()
        assert {"default_chain", "bing_cn", "duckduckgo", "tavily", "guidance"} <= set(data)
        # Tavily 'ok' should be None when no key configured
        assert data["tavily"]["ok"] is None
        assert "TAVILY_API_KEY" in data["tavily"]["error_str"]

    def test_dashboard_renders_search_card(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/dashboard")
        assert resp.status_code == 200
        assert "面经搜索后端 健康" in resp.text
        assert "/api/search/test" in resp.text

    def test_guidance_message_no_tavily(self, monkeypatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        msg = _search_guidance_message()
        assert "强烈推荐" in msg
        assert "tavily.com" in msg

    def test_guidance_message_with_tavily(self, monkeypatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fake")
        msg = _search_guidance_message()
        assert "已配 Tavily" in msg


_ = TavilySearch  # keep import (TavilySearch needs network/key — covered indirectly)
