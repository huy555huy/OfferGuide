from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from offerguide.interview_research.public_sources import (
    PublicInterviewSourceProvider,
    PublicSourceError,
    parse_nowcoder_discussion_html,
)
from offerguide.memory import Store
from offerguide.research_agents import SourceEvidenceStore, SourceScope


@pytest.fixture
def evidence_store(tmp_path: Path) -> SourceEvidenceStore:
    store = SourceEvidenceStore(Store(tmp_path / "public-sources.db"))
    store.init_schema()
    return store


@pytest.fixture
def scope() -> SourceScope:
    return SourceScope(
        run_id="interview-run-7",
        agent_name="interview_research_agent",
        subject_kind="interview_research",
        subject_id="233:1",
        subject_revision=4,
    )


def _nowcoder_html() -> str:
    return """
    <!doctype html>
    <html><head><title>页面标题不应覆盖原帖标题</title></head><body>
      <header><span class="name-text">导航栏用户</span></header>
      <section class="main-user-top">
        <span class="name-text">真实作者</span>
        <span class="time-text">2026-07-10 18:30</span>
      </section>
      <section class="post-content-box">
        <div class="content-post-title"><h1>AI 产品经理一面复盘</h1></div>
        <div class="nc-post-content">
          <p>一面问题：</p>
          <ol>
            <li value="1"><span>介绍你的项目</span></li>
            <li value="2"><span>为什么选择这个评测指标？</span></li>
          </ol>
          <script>题库自动答案：不属于原帖</script>
        </div>
      </section>
      <div class="js-comment-container">评论里的补充问题</div>
      <section class="recommend-text">相关推荐</section>
      <div class="content-post-title"><h1>推荐帖标题</h1></div>
      <div class="nc-post-content">推荐帖题库答案</div>
    </body></html>
    """


def _nowcoder_initial_state_html() -> str:
    state = {
        "prefetchData": {
            "1": {"userInfo": {}},
            "2": {
                "contentId": "6ecea7cfdbec4aac8c2dc12a3f7bbcc6",
                "contentType": 74,
                "ssrCommonData": {
                    "contentData": {
                        "id": 2798883,
                        "uuid": "6ecea7cfdbec4aac8c2dc12a3f7bbcc6",
                        "userBrief": {"nickname": "真实 SSR 作者"},
                        "title": "AI Infra 实习一面",
                        "content": (
                            "<p>一面问题：</p><ol>"
                            '<li value="1">介绍 FlashAttention</li>'
                            '<li value="2">如何设计显存分配器？</li>'
                            "</ol>"
                        ),
                        "createdAt": 1716188510000,
                    }
                },
                "similarRecommend": [
                    {
                        "contentData": {
                            "title": "推荐帖标题",
                            "content": "推荐帖自动题库答案",
                        }
                    }
                ],
            },
        }
    }
    return f"""
    <!doctype html>
    <html><body>
      <div class="content-post-title"><h1>DOM 中的错误标题</h1></div>
      <div class="nc-post-content">DOM 中的推荐内容</div>
      <script> window.__INITIAL_STATE__ = {json.dumps(state)}; </script>
    </body></html>
    """


def _legacy_nowcoder_html() -> str:
    return """
    <!doctype html>
    <html><body>
      <a class="post-name">旧版作者</a>
      <span class="post-time">发布于 2024-05-20 15:01</span>
      <h1 class="discuss-title"><span class="js-post-title post-title">旧版面经</span></h1>
      <div class="post-topic-des nc-post-content">
        <p>一面问题：</p>
        <ol><li value="1">介绍上一段实习</li></ol>
        <script>评论区题库，不属于原帖</script>
      </div>
    </body></html>
    """


def test_nowcoder_parser_keeps_only_original_post_region() -> None:
    post = parse_nowcoder_discussion_html(_nowcoder_html())

    assert post.title == "AI 产品经理一面复盘"
    assert post.author == "真实作者"
    assert post.published_at == "2026-07-10 18:30"
    assert post.body == ("一面问题：\n1. 介绍你的项目\n2. 为什么选择这个评测指标？")
    assert "导航栏用户" not in post.text_content
    assert "评论" not in post.text_content
    assert "推荐" not in post.text_content
    assert "题库自动答案" not in post.text_content


def test_nowcoder_parser_prefers_ssr_content_data_over_page_recommendations() -> None:
    post = parse_nowcoder_discussion_html(_nowcoder_initial_state_html())

    assert post.title == "AI Infra 实习一面"
    assert post.author == "真实 SSR 作者"
    assert post.published_at == "2024-05-20 15:01"
    assert post.body == ("一面问题：\n1. 介绍 FlashAttention\n2. 如何设计显存分配器？")
    assert "推荐帖" not in post.text_content
    assert "DOM 中" not in post.text_content


def test_nowcoder_parser_falls_back_to_legacy_discussion_dom() -> None:
    post = parse_nowcoder_discussion_html(_legacy_nowcoder_html())

    assert post.title == "旧版面经"
    assert post.author == "旧版作者"
    assert post.published_at == "2024-05-20 15:01"
    assert post.body == "一面问题：\n1. 介绍上一段实习"
    assert "评论区题库" not in post.text_content


def test_search_uses_advanced_domains_and_retains_full_raw_content_in_run_cache(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
) -> None:
    full_body = "正文" * 12_000
    full_snippet = "搜索摘要" * 1_000
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/search"
        payload = json.loads(request.content)
        assert payload["search_depth"] == "advanced"
        assert payload["include_domains"] == ["nowcoder.com", "maimai.cn"]
        assert payload["include_raw_content"] is True
        assert payload["include_answer"] is False
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "一手面经",
                        "url": "https://example.com/interview/7#comments",
                        "content": full_snippet,
                        "raw_content": full_body,
                    }
                ]
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handle),
        cookies={"session": "must-not-leak"},
    )
    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="tvly-test",
        tavily_client=client,
    )

    result = provider.search(
        "AI 产品经理 面经",
        scope=scope,
        domains=("nowcoder.com", "maimai.cn"),
    )

    candidate = result.candidates[0]
    assert candidate.snippet == full_snippet
    assert candidate.raw_content == full_body
    assert provider.cached_candidate(candidate.url, scope=scope) == candidate
    clue = result.as_tool_result()["clues"][0]
    assert len(clue["snippet"]) == 1_200
    assert clue["snippet"].endswith("...")
    assert "cookie" not in requests[0].headers
    assert requests[0].headers["authorization"] == "Bearer tvly-test"

    # Loading the selected URL reuses the complete search body. No second
    # Tavily request is made and the body is never replaced with its snippet.
    document = provider.load(candidate.url, scope=scope)
    assert len(requests) == 1
    assert document.text_content == full_body
    assert document.raw_content.decode() == full_body
    assert document.content_scope == "unscoped_page"

    saved = provider.persist_document(document, scope=scope)
    assert saved.status == "saved"
    assert saved.content_scope == "unscoped_page"
    assert saved.evidence is not None
    assert saved.evidence.text_content == full_body
    with evidence_store.store.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_evidence").fetchone()[0] == 1


def test_search_can_be_web_wide_and_skips_only_malformed_tavily_items(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        assert "include_domains" not in payload
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "", "url": "https://example.com/missing-title"},
                    {"title": "坏 URL", "url": "javascript:alert(1)"},
                    {
                        "title": "可用面经",
                        "url": "https://interview.example.org/post/7",
                        "content": "真实问题列表",
                    },
                ]
            },
        )

    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="tvly-test",
        tavily_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )

    result = provider.search("大模型推理优化 面经", scope=scope, domains=())

    assert len(requests) == 1
    assert result.domains == ()
    assert [candidate.title for candidate in result.candidates] == [
        "https://example.com/missing-title",
        "可用面经",
    ]
    assert result.as_tool_result()["clues"][1]["url"] == ("https://interview.example.org/post/7")


def test_known_url_uses_tavily_extract_when_no_search_body_is_cached(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/extract"
        payload = json.loads(request.content)
        assert payload == {
            "urls": ["https://blog.example.com/interview"],
            "extract_depth": "advanced",
            "format": "markdown",
            "include_images": False,
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://blog.example.com/interview",
                        "raw_content": "# 面经\n\n完整正文",
                    }
                ]
            },
        )

    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="tvly-test",
        tavily_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )

    document = provider.load(
        "https://blog.example.com/interview",
        scope=scope,
        title_hint="候选面经",
    )
    cached_document = provider.load(
        "https://blog.example.com/interview#ignored-fragment",
        scope=scope,
        title_hint="不会触发第二次提取",
    )

    assert len(requests) == 1
    assert cached_document is document
    assert document.source_backend == "tavily_extract"
    assert document.text_content == "# 面经\n\n完整正文"
    assert document.title == "候选面经"
    assert document.content_scope == "unscoped_page"

    saved = provider.persist_document(document, scope=scope)
    assert saved.status == "saved"
    assert saved.evidence is not None
    assert saved.evidence.text_content == "# 面经\n\n完整正文"


def test_nowcoder_fetch_ignores_cached_page_dump_and_sends_no_cookie(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
) -> None:
    url = "https://www.nowcoder.com/discuss/622084529531961344"
    tavily_requests: list[httpx.Request] = []
    public_requests: list[httpx.Request] = []

    def tavily_handle(request: httpx.Request) -> httpx.Response:
        tavily_requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "AI 产品经理一面复盘",
                        "url": url,
                        "content": "摘要",
                        "raw_content": "原帖 + 相关推荐 + 自动题库答案",
                    }
                ]
            },
        )

    def public_handle(request: httpx.Request) -> httpx.Response:
        public_requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=_nowcoder_html().encode(),
        )

    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="tvly-test",
        tavily_client=httpx.Client(transport=httpx.MockTransport(tavily_handle)),
        public_client=httpx.Client(
            transport=httpx.MockTransport(public_handle),
            cookies={"NOWCODER_SESSION": "must-not-leak"},
        ),
    )
    provider.search(
        "AI 产品经理 面经",
        scope=scope,
        domains=("nowcoder.com",),
    )

    fetched = provider.fetch(
        url,
        scope=scope,
        purpose="firsthand interview experience",
    )

    assert len(tavily_requests) == 1
    assert len(public_requests) == 1
    assert "cookie" not in public_requests[0].headers
    assert fetched.status == "saved"
    assert fetched.evidence is not None
    evidence = fetched.evidence
    assert evidence.raw_content == _nowcoder_html().encode()
    assert "介绍你的项目" in evidence.text_content
    assert "相关推荐" not in evidence.text_content
    assert "题库答案" not in evidence.text_content
    assert evidence.final_url == url


def test_nowcoder_direct_failure_uses_complete_public_search_body(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
) -> None:
    url = "https://www.nowcoder.com/feed/main/detail/public-post"

    def tavily_handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "AI Infra 面经",
                        "url": url,
                        "content": "线索摘要",
                        "raw_content": "完整公开正文：FlashAttention 为什么减少显存访问？",
                    }
                ]
            },
        )

    def public_handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, headers={"Content-Type": "text/html"})

    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="tvly-test",
        tavily_client=httpx.Client(transport=httpx.MockTransport(tavily_handle)),
        public_client=httpx.Client(transport=httpx.MockTransport(public_handle)),
    )

    provider.search("AI Infra 面经", scope=scope, domains=())
    document = provider.load(url, scope=scope)

    assert document.text_content == "完整公开正文：FlashAttention 为什么减少显存访问？"
    assert document.content_scope == "unscoped_page"
    assert document.source_backend == "tavily_search"


@pytest.mark.parametrize(
    "url",
    (
        "https://www.nowcoder.com/feed/main/detail/6ecea7cfdbec4aac8c2dc12a3f7bbcc6",
        "https://www.nowcoder.com/discuss/622084529531961344",
    ),
)
def test_nowcoder_current_feed_and_discuss_routes_use_public_ssr(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
    url: str,
) -> None:
    calls: list[httpx.Request] = []

    def public_handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=_nowcoder_initial_state_html(),
        )

    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="",
        public_client=httpx.Client(
            transport=httpx.MockTransport(public_handle),
            cookies={"NOWCODER_SESSION": "must-not-leak"},
        ),
    )

    fetched = provider.fetch(url, scope=scope)
    cached_document = provider.load(url, scope=scope)

    assert fetched.status == "saved"
    assert fetched.evidence is not None
    assert fetched.evidence.title == "AI Infra 实习一面"
    assert cached_document.title == "AI Infra 实习一面"
    assert len(calls) == 1
    assert "cookie" not in calls[0].headers


def test_nowcoder_legacy_ac_redirect_stays_public_and_cookie_free(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
) -> None:
    calls: list[httpx.Request] = []

    def public_handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert "cookie" not in request.headers
        if len(calls) == 1:
            return httpx.Response(
                301,
                headers={
                    "Location": "/discuss/1309363",
                    "Set-Cookie": "NOWCODERUID=must-not-follow",
                },
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=_legacy_nowcoder_html(),
        )

    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="",
        public_client=httpx.Client(
            transport=httpx.MockTransport(public_handle),
            cookies={"NOWCODER_SESSION": "must-not-leak"},
        ),
    )

    fetched = provider.fetch(
        "https://ac.nowcoder.com/discuss/622084529531961344",
        scope=scope,
    )

    assert fetched.status == "saved"
    assert fetched.evidence is not None
    assert fetched.evidence.title == "旧版面经"
    assert fetched.final_url == "https://ac.nowcoder.com/discuss/1309363"
    assert len(calls) == 2


def test_nowcoder_cross_host_redirect_uses_public_extract(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
) -> None:
    public_calls: list[httpx.Request] = []
    extract_calls: list[httpx.Request] = []

    def public_handle(request: httpx.Request) -> httpx.Response:
        public_calls.append(request)
        return httpx.Response(302, headers={"Location": "https://example.com/post/1"})

    def tavily_handle(request: httpx.Request) -> httpx.Response:
        extract_calls.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/post/1",
                        "title": "公开面经",
                        "raw_content": "一次真实面试的完整正文",
                    }
                ]
            },
        )

    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="tvly-test",
        tavily_client=httpx.Client(transport=httpx.MockTransport(tavily_handle)),
        public_client=httpx.Client(transport=httpx.MockTransport(public_handle)),
    )

    document = provider.load(
        "https://www.nowcoder.com/feed/main/detail/example",
        scope=scope,
    )

    assert len(public_calls) == 1
    assert len(extract_calls) == 1
    assert document.url == "https://example.com/post/1"
    assert document.text_content == "一次真实面试的完整正文"
    assert document.content_scope == "unscoped_page"


def test_no_api_key_is_explicit_and_nowcoder_pages_are_read_before_assessment(
    evidence_store: SourceEvidenceStore,
    scope: SourceScope,
) -> None:
    public_calls: list[str] = []

    def public_handle(request: httpx.Request) -> httpx.Response:
        public_calls.append(str(request.url))
        if request.url.path == "/search":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                text="<html><body>搜索结果页，没有原帖正文</body></html>",
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text=_nowcoder_html(),
        )

    provider = PublicInterviewSourceProvider(
        evidence_store,
        api_key="",
        public_client=httpx.Client(transport=httpx.MockTransport(public_handle)),
    )

    with pytest.raises(PublicSourceError) as caught:
        provider.search(
            "AI 产品经理 面经",
            scope=scope,
            domains=("nowcoder.com",),
        )
    assert caught.value.code == "tavily_unavailable"

    direct = provider.fetch(
        "https://www.nowcoder.com/discuss/123",
        scope=scope,
    )
    assert direct.status == "saved"
    assert len(public_calls) == 1

    page = provider.load(
        "https://www.nowcoder.com/search?query=面经",
        scope=scope,
    )
    assert page.content_scope == "unscoped_page"
    assert page.text_content == "搜索结果页，没有原帖正文"
    assert len(public_calls) == 2
