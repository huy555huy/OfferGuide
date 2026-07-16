"""Account-free discovery and extraction of public interview experiences."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin, urlsplit

import httpx

from ..research_agents.sources import (
    SourceEvidenceStore,
    SourceFetchResult,
    SourceScope,
    canonicalize_http_url,
)

_SEARCH_URL = "https://api.tavily.com/search"
_EXTRACT_URL = "https://api.tavily.com/extract"
_NOWCODER_HOSTS = {"nowcoder.com", "www.nowcoder.com", "ac.nowcoder.com"}
_MAX_CLUE_SNIPPET_CHARS = 1_200
_DOMAIN = re.compile(r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z0-9-]+$")

ContentScope = Literal["original_post", "unscoped_page"]


class PublicSourceError(RuntimeError):
    def __init__(self, code: str, message: str, *, http_status: int | None = None) -> None:
        self.code = code
        self.http_status = http_status
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PublicSourceCandidate:
    title: str
    url: str
    snippet: str
    raw_content: str | None
    content_scope: ContentScope = "unscoped_page"

    @property
    def has_full_content(self) -> bool:
        return bool(self.raw_content and self.raw_content.strip())

    def as_clue(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": _bounded_clue_text(self.snippet),
            "raw_content_available": self.has_full_content,
            "content_scope": self.content_scope,
            "evidence": False,
        }


@dataclass(frozen=True, slots=True)
class PublicSourceSearchResult:
    query: str
    domains: tuple[str, ...]
    candidates: tuple[PublicSourceCandidate, ...]
    search_execution_id: int

    def as_tool_result(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "domains": list(self.domains),
            "results_are_unverified_clues": True,
            "clues": [item.as_clue() for item in self.candidates],
            "search_execution_id": self.search_execution_id,
        }


@dataclass(frozen=True, slots=True)
class PublicSourceDocument:
    requested_url: str
    url: str
    title: str
    raw_content: bytes
    text_content: str
    media_type: str
    charset: str | None
    source_backend: str
    content_scope: ContentScope
    http_status: int | None = None


@dataclass(frozen=True, slots=True)
class NowcoderPost:
    title: str
    author: str | None
    published_at: str | None
    body: str

    @property
    def text_content(self) -> str:
        lines = [f"标题：{self.title}"]
        if self.author:
            lines.append(f"作者：{self.author}")
        if self.published_at:
            lines.append(f"发布时间：{self.published_at}")
        return "\n".join((*lines, "正文：", self.body))


class PublicInterviewSourceProvider:
    """Tavily discovery plus platform-scoped, login-free public readers."""

    def __init__(
        self,
        evidence_store: SourceEvidenceStore,
        *,
        api_key: str | None = None,
        tavily_client: httpx.Client | None = None,
        public_client: httpx.Client | None = None,
        timeout_s: float = 30.0,
        max_response_bytes: int = 5_000_000,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.evidence_store = evidence_store
        self.api_key = (
            os.environ.get("TAVILY_API_KEY", "") if api_key is None else api_key
        ).strip()
        self.tavily_client = tavily_client or httpx.Client(timeout=timeout_s)
        self.public_client = public_client or httpx.Client(
            timeout=timeout_s,
            follow_redirects=False,
            headers={
                "User-Agent": "OfferGuide/1.0 (+public interview research)",
                "Accept": "text/html,application/xhtml+xml;q=0.9",
            },
        )
        self.max_response_bytes = max_response_bytes
        self._candidate_cache: dict[tuple[str, str], PublicSourceCandidate] = {}
        self._document_cache: dict[tuple[str, str], PublicSourceDocument] = {}
        self._load_error_cache: dict[tuple[str, str], PublicSourceError] = {}

    @property
    def tavily_available(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        query: str,
        *,
        scope: SourceScope,
        domains: Sequence[str],
        max_results: int = 10,
    ) -> PublicSourceSearchResult:
        query = str(query or "").strip()
        if not query:
            raise ValueError("search query must not be blank")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        domains = _normalize_domains(domains)
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            "topic": "general",
            "max_results": max_results,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": True,
        }
        if domains:
            payload["include_domains"] = list(domains)
        try:
            response = self._tavily_post(_SEARCH_URL, payload)
            raw_results = response.get("results")
            if not isinstance(raw_results, list):
                raise PublicSourceError("invalid_response", "Tavily search has no results array")
            parsed_candidates: list[PublicSourceCandidate] = []
            for item in raw_results[:max_results]:
                try:
                    parsed_candidates.append(_candidate(item))
                except (PublicSourceError, ValueError):
                    continue
            candidates = tuple(parsed_candidates)
            cache_scope = _cache_scope(scope)
            for item in candidates:
                self._candidate_cache[(cache_scope, item.url)] = item
            execution_id = self.evidence_store.record_search(
                scope=scope,
                backend="tavily_public",
                query=query,
                status="succeeded" if candidates else "empty",
                results=[item.as_clue() for item in candidates],
            )
        except Exception as exc:
            error = _source_error(exc)
            self.evidence_store.record_search(
                scope=scope,
                backend="tavily_public",
                query=query,
                status="failed",
                results=(),
                error_text=f"{type(error).__name__}: {error}",
            )
            raise error from exc
        return PublicSourceSearchResult(query, domains, candidates, execution_id)

    def cached_candidate(self, url: str, *, scope: SourceScope) -> PublicSourceCandidate | None:
        return self._candidate_cache.get((_cache_scope(scope), canonicalize_http_url(url)))

    def load(
        self,
        url: str,
        *,
        scope: SourceScope,
        title_hint: str = "",
    ) -> PublicSourceDocument:
        """Load readable public content without authenticated browser state."""
        requested_url = canonicalize_http_url(url)
        cache_key = (_cache_scope(scope), requested_url)
        cached_document = self._document_cache.get(cache_key)
        if cached_document is not None:
            return cached_document
        cached_error = self._load_error_cache.get(cache_key)
        if cached_error is not None:
            raise cached_error
        try:
            if _is_nowcoder_url(requested_url):
                try:
                    document = self._load_nowcoder(requested_url, title_hint)
                except PublicSourceError as direct_error:
                    cached = self.cached_candidate(requested_url, scope=scope)
                    if cached and cached.has_full_content:
                        document = _unscoped_document(
                            requested_url,
                            cached.url,
                            cached.title or title_hint,
                            cached.raw_content or "",
                            "tavily_search",
                        )
                    elif self.tavily_available:
                        try:
                            document = self._extract(requested_url, title_hint)
                        except PublicSourceError:
                            raise direct_error from None
                    else:
                        raise
            else:
                cached = self.cached_candidate(requested_url, scope=scope)
                if cached and cached.has_full_content:
                    document = _unscoped_document(
                        requested_url,
                        cached.url,
                        cached.title or title_hint,
                        cached.raw_content or "",
                        "tavily_search",
                    )
                else:
                    document = self._extract(requested_url, title_hint)
            self._document_cache[cache_key] = document
            return document
        except Exception as exc:
            error = _source_error(exc)
            self._load_error_cache[cache_key] = error
            self.evidence_store.record_fetch_failure(
                scope=scope,
                requested_url=requested_url,
                final_url=requested_url,
                status="failed",
                error_code=error.code,
                error_text=str(error),
                http_status=error.http_status,
            )
            raise error from exc

    def persist_document(
        self,
        document: PublicSourceDocument,
        *,
        scope: SourceScope,
        purpose: str = "",
    ) -> SourceFetchResult:
        """Save readable content so the research agent can assess its relevance."""
        evidence, status, fetch_id = self.evidence_store.save_response(
            scope=scope,
            requested_url=document.requested_url,
            final_url=document.url,
            title=document.title or document.url,
            media_type=document.media_type,
            charset=document.charset,
            raw_content=document.raw_content,
            text_content=document.text_content,
            http_status=document.http_status,
            purpose=purpose,
        )
        return SourceFetchResult(
            status,
            document.requested_url,
            document.url,
            evidence,
            fetch_id,
            http_status=document.http_status,
            content_scope=document.content_scope,
        )

    def fetch(
        self,
        url: str,
        *,
        scope: SourceScope,
        purpose: str = "",
        title_hint: str = "",
    ) -> SourceFetchResult:
        document = self.load(url, scope=scope, title_hint=title_hint)
        return self.persist_document(document, scope=scope, purpose=purpose)

    def close(self) -> None:
        self.tavily_client.close()
        if self.public_client is not self.tavily_client:
            self.public_client.close()

    def _tavily_post(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.api_key:
            raise PublicSourceError(
                "tavily_unavailable",
                "TAVILY_API_KEY is not configured; automatic public search is unavailable",
            )
        request = self.tavily_client.build_request(
            "POST",
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=dict(payload),
        )
        request.headers.pop("cookie", None)
        response = self.tavily_client.send(request)
        if not 200 <= response.status_code < 300:
            raise PublicSourceError(
                "tavily_http_status",
                f"Tavily returned HTTP {response.status_code}",
                http_status=response.status_code,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise PublicSourceError("invalid_response", "Tavily returned invalid JSON") from exc
        if not isinstance(data, Mapping):
            raise PublicSourceError("invalid_response", "Tavily returned a non-object response")
        return data

    def _extract(self, url: str, title_hint: str) -> PublicSourceDocument:
        response = self._tavily_post(
            _EXTRACT_URL,
            {
                "urls": [url],
                "extract_depth": "advanced",
                "format": "markdown",
                "include_images": False,
            },
        )
        results = response.get("results")
        if not isinstance(results, list) or not results or not isinstance(results[0], Mapping):
            raise PublicSourceError("extract_failed", "Tavily returned no extracted page")
        item = results[0]
        content = item.get("raw_content")
        if not isinstance(content, str) or not content.strip():
            raise PublicSourceError("empty_source", "Tavily returned no readable content")
        final_url = canonicalize_http_url(str(item.get("url") or url))
        title = str(item.get("title") or title_hint or final_url).strip()
        return _unscoped_document(url, final_url, title, content, "tavily_extract")

    def _load_nowcoder(
        self,
        requested_url: str,
        title_hint: str = "",
    ) -> PublicSourceDocument:
        current_url = requested_url
        for redirect in range(4):
            if not _is_nowcoder_url(current_url):
                return self._extract(current_url, title_hint)
            request = self.public_client.build_request("GET", current_url)
            request.headers.pop("cookie", None)
            response = self.public_client.send(request)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location or redirect == 3:
                    raise PublicSourceError("redirect_rejected", "Nowcoder redirect was rejected")
                current_url = canonicalize_http_url(urljoin(current_url, location))
                if not _is_nowcoder_url(current_url):
                    return self._extract(current_url, title_hint)
                continue
            if response.status_code != 200:
                raise PublicSourceError(
                    "public_http_status",
                    f"Nowcoder returned HTTP {response.status_code}",
                    http_status=response.status_code,
                )
            media_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if media_type not in {"text/html", "application/xhtml+xml"}:
                raise PublicSourceError("unsupported_content_type", "Nowcoder did not return HTML")
            body = response.content
            if len(body) > self.max_response_bytes:
                raise PublicSourceError("source_too_large", "Nowcoder page is too large")
            charset = (
                response.encoding or _charset(response.headers.get("content-type", "")) or "utf-8"
            )
            html = body.decode(charset, errors="replace")
            try:
                post = parse_nowcoder_discussion_html(html)
            except PublicSourceError as exc:
                if exc.code != "missing_original_post":
                    raise
                if self.tavily_available:
                    try:
                        return self._extract(current_url, title_hint)
                    except PublicSourceError:
                        pass
                page_title, page_text = _html_page_text(html)
                if not page_text:
                    raise
                return PublicSourceDocument(
                    requested_url,
                    current_url,
                    page_title or title_hint or current_url,
                    body,
                    page_text,
                    media_type,
                    charset,
                    "nowcoder_public_html",
                    "unscoped_page",
                    response.status_code,
                )
            return PublicSourceDocument(
                requested_url,
                current_url,
                post.title,
                body,
                post.text_content,
                media_type,
                charset,
                "nowcoder_public_html",
                "original_post",
                response.status_code,
            )
        raise AssertionError("redirect loop ended unexpectedly")


def parse_nowcoder_discussion_html(html: str) -> NowcoderPost:
    """Extract the SSR original post, with legacy DOM parsing as a fallback."""
    html = str(html or "")
    structured = _parse_nowcoder_initial_state(html)
    if structured is not None:
        return structured

    parser = _NowcoderParser()
    parser.feed(html)
    parser.close()
    title = _inline("".join(parser.title))
    body = parser.body.finish()
    if not title or not body:
        raise PublicSourceError("missing_original_post", "Nowcoder original post is missing")
    return NowcoderPost(
        title,
        _optional_inline("".join(parser.author)),
        _normalize_published_at("".join(parser.published_at)),
        body,
    )


def _parse_nowcoder_initial_state(html: str) -> NowcoderPost | None:
    marker = re.compile(r"window\s*\.\s*__INITIAL_STATE__\s*=")
    decoder = json.JSONDecoder()
    for match in marker.finditer(html):
        try:
            state, _ = decoder.raw_decode(html[match.end() :].lstrip())
        except (TypeError, ValueError):
            continue
        content_data = _primary_nowcoder_content_data(state)
        if content_data is None:
            continue
        title = _first_text(content_data, "newTitle", "title")
        content = _first_text(content_data, "newContent", "content", "richText")
        body = _html_fragment_text(content)
        if not title or not body:
            continue
        user = content_data.get("userBrief")
        author = (
            _optional_inline(str(user.get("nickname") or "")) if isinstance(user, Mapping) else None
        )
        return NowcoderPost(
            title,
            author,
            _structured_published_at(content_data),
            body,
        )
    return None


def _primary_nowcoder_content_data(state: Any) -> Mapping[str, Any] | None:
    if not isinstance(state, Mapping):
        return None
    roots: list[Mapping[str, Any]] = [state]
    prefetch = state.get("prefetchData")
    if isinstance(prefetch, Mapping):
        roots.extend(item for item in prefetch.values() if isinstance(item, Mapping))
    for root in roots:
        common = root.get("ssrCommonData")
        if isinstance(common, Mapping):
            content_data = common.get("contentData")
            if isinstance(content_data, Mapping):
                return content_data
        content_data = root.get("contentData")
        if isinstance(content_data, Mapping):
            return content_data
    return None


def _first_text(values: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _structured_published_at(content_data: Mapping[str, Any]) -> str | None:
    for key in ("createdAt", "createTime", "showTime"):
        value = content_data.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_published_at(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            seconds = float(value) / 1000 if value > 10_000_000_000 else float(value)
            try:
                china = timezone(timedelta(hours=8))
                return datetime.fromtimestamp(seconds, tz=china).strftime("%Y-%m-%d %H:%M")
            except (OverflowError, OSError, ValueError):
                continue
    return None


def _normalize_published_at(value: str) -> str | None:
    normalized = _inline(value)
    normalized = re.sub(r"^(?:发布于|发表于)\s*", "", normalized)
    return normalized or None


@dataclass(slots=True)
class _Frame:
    tag: str
    roles: frozenset[str]
    body_root: bool = False


class _BodyText:
    _BLOCKS: ClassVar[set[str]] = {
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ol",
        "p",
        "pre",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        self.parts: list[str] = []

    def start(self, tag: str, attrs: Mapping[str, str]) -> None:
        if tag in self._BLOCKS:
            self.parts.append("\n")
        if tag == "li":
            value = attrs.get("value", "")
            self.parts.append(f"{value}. " if value.isdigit() else "- ")

    def end(self, tag: str) -> None:
        if tag in self._BLOCKS:
            self.parts.append("\n")

    def finish(self) -> str:
        lines = [_inline(line) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


class _FragmentTextParser(HTMLParser):
    _IGNORED: ClassVar[set[str]] = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.body = _BodyText()
        self.title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        ignored = (self.stack[-1][1] if self.stack else False) or tag in self._IGNORED
        self.stack.append((tag, ignored))
        if not ignored:
            self.body.start(tag, {key.lower(): value or "" for key, value in attrs})

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        match = next(
            (i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag),
            None,
        )
        if match is None:
            return
        while len(self.stack) > match:
            frame_tag, ignored = self.stack.pop()
            if not ignored:
                self.body.end(frame_tag)

    def handle_data(self, data: str) -> None:
        if not self.stack or not self.stack[-1][1]:
            self.body.parts.append(data)
            if self.stack and self.stack[-1][0] == "title":
                self.title.append(data)


def _html_fragment_text(content: str) -> str:
    parser = _FragmentTextParser()
    parser.feed(content)
    parser.close()
    return parser.body.finish()


def _html_page_text(content: str) -> tuple[str, str]:
    parser = _FragmentTextParser()
    parser.feed(content)
    parser.close()
    return _inline("".join(parser.title)), parser.body.finish()


class _NowcoderParser(HTMLParser):
    _IGNORED: ClassVar[set[str]] = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Frame] = []
        self.title: list[str] = []
        self.author: list[str] = []
        self.published_at: list[str] = []
        self.body = _BodyText()
        self.body_complete = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = {key.lower(): value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        roles = set(self.stack[-1].roles) if self.stack else set()
        if tag in self._IGNORED:
            roles.add("ignored")
        body_root = False
        if "ignored" not in roles:
            if "main-user-top" in classes:
                roles.add("user")
            if "user" in roles and "name-text" in classes and not self.author:
                roles.add("author")
            if "user" in roles and "time-text" in classes and not self.published_at:
                roles.add("time")
            if "post-name" in classes and not self.author:
                roles.add("author")
            if "post-time" in classes and not self.published_at:
                roles.add("time")
            if (
                classes.intersection({"content-post-title", "discuss-title", "js-post-title"})
                and not self.title
            ):
                roles.add("title")
            if "nc-post-content" in classes and not self.body_complete and "body" not in roles:
                roles.add("body")
                body_root = True
            if "body" in roles:
                self.body.start(tag, values)
        self.stack.append(_Frame(tag, frozenset(roles), body_root))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        match = next(
            (i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i].tag == tag.lower()),
            None,
        )
        if match is None:
            return
        while len(self.stack) > match:
            frame = self.stack.pop()
            if "body" in frame.roles and "ignored" not in frame.roles:
                self.body.end(frame.tag)
            if frame.body_root:
                self.body_complete = True

    def handle_data(self, data: str) -> None:
        roles = self.stack[-1].roles if self.stack else frozenset()
        if "ignored" in roles:
            return
        if "title" in roles:
            self.title.append(data)
        if "author" in roles:
            self.author.append(data)
        if "time" in roles:
            self.published_at.append(data)
        if "body" in roles:
            self.body.parts.append(data)


def _candidate(item: Any) -> PublicSourceCandidate:
    if not isinstance(item, Mapping):
        raise PublicSourceError("invalid_response", "invalid Tavily search result")
    url = str(item.get("url") or "").strip()
    if not url:
        raise PublicSourceError("invalid_response", "Tavily result has no URL")
    canonical_url = canonicalize_http_url(url)
    title = str(item.get("title") or "").strip() or canonical_url
    raw = item.get("raw_content")
    return PublicSourceCandidate(
        title,
        canonical_url,
        str(item.get("content") or "").strip(),
        str(raw) if raw is not None else None,
    )


def _bounded_clue_text(value: str) -> str:
    value = str(value or "").strip()
    if len(value) <= _MAX_CLUE_SNIPPET_CHARS:
        return value
    return value[: _MAX_CLUE_SNIPPET_CHARS - 3].rstrip() + "..."


def _unscoped_document(
    requested_url: str,
    final_url: str,
    title: str,
    content: str,
    backend: str,
) -> PublicSourceDocument:
    return PublicSourceDocument(
        requested_url,
        final_url,
        title or final_url,
        content.encode(),
        content,
        "text/markdown",
        "utf-8",
        backend,
        "unscoped_page",
    )


def _normalize_domains(domains: Sequence[str]) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(item or "").strip().lower().strip(".") for item in domains))
    if any(not _DOMAIN.fullmatch(item) for item in result):
        raise ValueError("search domains must be valid public domains")
    if len(result) > 20:
        raise ValueError("at most 20 domains are allowed")
    return result


def _cache_scope(scope: SourceScope) -> str:
    return (
        f"{scope.run_id}\x1f{scope.subject_kind}\x1f{scope.subject_id}\x1f{scope.subject_revision}"
    )


def _is_nowcoder_url(url: str) -> bool:
    return (urlsplit(url).hostname or "").lower() in _NOWCODER_HOSTS


def _charset(content_type: str) -> str | None:
    for part in content_type.split(";")[1:]:
        key, separator, value = part.partition("=")
        if separator and key.strip().lower() == "charset":
            return value.strip().strip("\"'") or None
    return None


def _inline(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def _optional_inline(value: str) -> str | None:
    return _inline(value) or None


def _source_error(exc: Exception) -> PublicSourceError:
    if isinstance(exc, PublicSourceError):
        return exc
    return PublicSourceError("public_source_error", f"{type(exc).__name__}: {exc}")


__all__ = [
    "ContentScope",
    "NowcoderPost",
    "PublicInterviewSourceProvider",
    "PublicSourceCandidate",
    "PublicSourceDocument",
    "PublicSourceError",
    "PublicSourceSearchResult",
    "parse_nowcoder_discussion_html",
]
