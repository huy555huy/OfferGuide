"""W8'''' — Agentic layer: LLM email classifier + 面经 collector + sweep.

Tests:
- LLMEmailClassifier returns structured kind/extracted via stub LLM
- DuckDuckGo HTML parser (offline against canned HTML)
- CorpusCollector orchestrates search + LLM + ingest with stubs
- /api/agent/sweep endpoint composes everything
- /api/email/classify mode=auto picks regex when no API key
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import interview_corpus
from offerguide.agentic.corpus_collector import CorpusCollector
from offerguide.agentic.email_classifier_llm import (
    classify_email_batch_llm,
    classify_email_llm,
)
from offerguide.agentic.meta_agent import sweep_company
from offerguide.agentic.search import (
    SearchHit,
    StubSearch,
    _parse_ddg_html,
    _strip_html,
    _unwrap_ddg_redirect,
)
from offerguide.config import Settings
from offerguide.profile import UserProfile
from offerguide.skills import discover_skills
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


# ── stubs ──────────────────────────────────────────────────────────


class _StubLLM:
    """Stand-in for LLMClient.chat — returns canned JSON per call."""

    def __init__(self, *, response_per_call: list[str] | None = None) -> None:
        self.calls: list[Any] = []
        self._responses = list(response_per_call or [])
        self._idx = 0

    def chat(self, *, messages, model=None, temperature=0.0, json_mode=False, extra=None):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        if self._idx < len(self._responses):
            content = self._responses[self._idx]
            self._idx += 1
        else:
            content = "{}"
        # Mimic LLMResponse just enough
        return type("Resp", (), {"content": content, "model": "stub", "raw": {}})()


# ═══════════════════════════════════════════════════════════════════
# LLM EMAIL CLASSIFIER
# ═══════════════════════════════════════════════════════════════════


class TestLLMEmailClassifier:
    def test_extracts_structured_info(self) -> None:
        llm = _StubLLM(response_per_call=[
            json.dumps({
                "kind": "interview",
                "confidence": 0.95,
                "matched_company": "字节跳动",
                "extracted": {
                    "interview_time": "2026-05-20 14:00",
                    "contact_name": "张 HR",
                    "referenced_role": "AI Agent 实习",
                    "interview_round": "一面",
                    "assessment_link": None,
                    "deadline": None,
                },
                "evidence": [
                    "邀请您参加 字节跳动 AI Agent 实习的一面",
                    "面试时间: 2026-05-20 14:00",
                ],
            })
        ])
        result = classify_email_llm(
            "字节跳动 AI Agent 一面邀请...",
            llm=llm,  # type: ignore[arg-type]
            known_companies=["字节跳动"],
        )
        assert result.kind == "interview"
        assert result.confidence == 0.95
        assert result.matched_company == "字节跳动"
        assert result.extracted["interview_time"] == "2026-05-20 14:00"
        assert result.extracted["contact_name"] == "张 HR"
        assert result.extracted["interview_round"] == "一面"
        assert len(result.evidence) == 2

    def test_application_id_when_unique(self) -> None:
        llm = _StubLLM(response_per_call=[
            json.dumps({
                "kind": "rejected",
                "confidence": 0.9,
                "matched_company": "字节跳动",
                "extracted": {},
                "evidence": ["很遗憾"],
            })
        ])
        result = classify_email_llm(
            "字节跳动 拒信",
            llm=llm,  # type: ignore[arg-type]
            known_companies=["字节跳动"],
            known_apps_by_company={"字节跳动": [42]},
        )
        assert result.matched_application_id == 42

    def test_invalid_kind_falls_to_unrelated(self) -> None:
        llm = _StubLLM(response_per_call=[
            json.dumps({"kind": "rocket_launch", "confidence": 0.5,
                        "matched_company": None, "extracted": {}, "evidence": []})
        ])
        result = classify_email_llm("x", llm=llm)  # type: ignore[arg-type]
        assert result.kind == "unrelated"

    def test_non_json_response_falls_to_unrelated(self) -> None:
        llm = _StubLLM(response_per_call=["not valid json {"])
        result = classify_email_llm("x", llm=llm)  # type: ignore[arg-type]
        assert result.kind == "unrelated"
        assert "non-JSON" in result.evidence[0]

    def test_empty_input_short_circuits(self) -> None:
        llm = _StubLLM(response_per_call=[])
        result = classify_email_llm("", llm=llm)  # type: ignore[arg-type]
        assert result.kind == "unrelated"
        # LLM should NOT have been called
        assert len(llm.calls) == 0

    def test_batch(self) -> None:
        llm = _StubLLM(response_per_call=[
            json.dumps({"kind": "offer", "confidence": 0.95,
                        "matched_company": None, "extracted": {}, "evidence": []}),
            json.dumps({"kind": "rejected", "confidence": 0.9,
                        "matched_company": None, "extracted": {}, "evidence": []}),
        ])
        results = classify_email_batch_llm(
            ["你被录用了", "感谢您但不合适"],
            llm=llm,  # type: ignore[arg-type]
        )
        assert [r.kind for r in results] == ["offer", "rejected"]


# ═══════════════════════════════════════════════════════════════════
# SEARCH BACKEND
# ═══════════════════════════════════════════════════════════════════


_DDG_HTML_SAMPLE = """
<html><body>
<div class="result">
  <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nowcoder.com%2Fdiscuss%2F12345">字节 AI Agent 一面经验</a></h2>
  <a class="result__snippet" href="...">字节跳动 AI Agent 实习一面，30 分钟，主要是项目深挖。</a>
</div>
<div class="result">
  <h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F999">字节实习面经</a></h2>
  <a class="result__snippet" href="...">面试官问了 attention 缩放…</a>
</div>
</body></html>
"""


class TestDuckDuckGoParser:
    def test_parses_two_results(self) -> None:
        hits = _parse_ddg_html(_DDG_HTML_SAMPLE, max_results=10)
        assert len(hits) == 2
        assert "nowcoder.com" in hits[0].url
        assert "zhihu.com" in hits[1].url

    def test_unwraps_redirect(self) -> None:
        url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nowcoder.com%2Fdiscuss%2Fxyz"
        out = _unwrap_ddg_redirect(url)
        assert out == "https://www.nowcoder.com/discuss/xyz"

    def test_strip_html_removes_tags(self) -> None:
        assert _strip_html("<b>hello</b> &amp; world") == "hello & world"

    def test_max_results_cap(self) -> None:
        hits = _parse_ddg_html(_DDG_HTML_SAMPLE, max_results=1)
        assert len(hits) == 1


class TestStubSearch:
    def test_returns_canned_hits(self) -> None:
        s = StubSearch({
            "字节 面经": [
                SearchHit(title="t", url="https://www.nowcoder.com/x", snippet="s")
            ]
        })
        hits = s.search("字节 面经")
        assert len(hits) == 1
        assert hits[0].url == "https://www.nowcoder.com/x"

    def test_empty_for_unknown_query(self) -> None:
        s = StubSearch()
        assert s.search("anything") == []


# ═══════════════════════════════════════════════════════════════════
# CORPUS COLLECTOR
# ═══════════════════════════════════════════════════════════════════


class _RecordingFetchHTTP:
    """Stand-in for httpx.Client used inside CorpusCollector.

    Returns canned HTML per URL.
    """

    def __init__(self, content_per_url: dict[str, str]) -> None:
        self._content = content_per_url

    def get(self, url: str, *, follow_redirects: bool = False, **_):
        if url in self._content:
            return type(
                "_R",
                (),
                {
                    "status_code": 200,
                    "headers": {"content-type": "text/html"},
                    "text": self._content[url],
                },
            )()
        return type(
            "_R404",
            (),
            {"status_code": 404, "headers": {}, "text": ""},
        )()

    def close(self) -> None:
        pass


@pytest.fixture
def corpus_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "corpus.db")
    store.init_schema()
    return store


class TestCorpusCollector:
    def test_collects_genuine_面经(self, corpus_setup) -> None:
        store = corpus_setup
        # Search backend returns 1 nowcoder hit
        search = StubSearch({
            "字节跳动 面经 校招 2026": [
                SearchHit(title="字节面经", url="https://www.nowcoder.com/d/123", snippet="s"),
            ]
        })
        # LLM evaluator OK's it
        llm = _StubLLM(response_per_call=[
            json.dumps({
                "is_genuine_interview_exp": True,
                "company_match": True,
                "role_hint": "AI Agent 实习",
                "year_guess": "2026",
                "clean_raw_text": (
                    "字节跳动 AI Agent 实习一面 · 30 分钟 · 项目深挖。"
                    "面试官先让我介绍 RemeDi 项目，然后问 GRPO vs PPO 区别，"
                    "DeepSpeed ZeRO-2 vs ZeRO-3 选型理由，attention 缩放推导，"
                    "为什么不直接用 Llama-3 base。简单 leetcode 中等题。"
                    "整体偏基础 + 项目细节，HR 反馈 1-2 周内出结果。"
                ),
                "rationale": "确实是 2026 字节实习一面",
            })
        ])
        collector = CorpusCollector(store=store, llm=llm, search=search)  # type: ignore[arg-type]
        # Replace the http fetcher with our stub
        collector._http = _RecordingFetchHTTP({
            "https://www.nowcoder.com/d/123": "<html>字节面经原文...30 分钟项目深挖...</html>"
        })  # type: ignore[assignment]

        result = collector.collect("字节跳动")
        assert result.inserted == 1
        assert result.hits_seen >= 1
        # 面经 actually in DB
        rows = interview_corpus.fetch_for_company(store, "字节跳动")
        assert len(rows) == 1
        assert rows[0].source == "agent_search"

    def test_rejects_low_quality(self, corpus_setup) -> None:
        store = corpus_setup
        search = StubSearch({
            "字节跳动 面经 校招 2026": [
                SearchHit(title="random", url="https://www.nowcoder.com/x", snippet="s"),
            ]
        })
        llm = _StubLLM(response_per_call=[
            json.dumps({
                "is_genuine_interview_exp": False,
                "company_match": False,
                "role_hint": None,
                "year_guess": None,
                "clean_raw_text": "",
                "rationale": "this is a marketing page, not 面经",
            })
        ])
        collector = CorpusCollector(store=store, llm=llm, search=search)  # type: ignore[arg-type]
        collector._http = _RecordingFetchHTTP({
            "https://www.nowcoder.com/x": "<html>...</html>"
        })  # type: ignore[assignment]
        result = collector.collect("字节跳动")
        assert result.inserted == 0
        assert result.skipped_low_quality >= 1

    def test_skips_non_preferred_domains(self, corpus_setup) -> None:
        store = corpus_setup
        # All hits are from random sites (not in _PREFERRED_DOMAINS)
        search = StubSearch({
            "字节跳动 面经 校招 2026": [
                SearchHit(title="t", url="https://example.com/x", snippet="s"),
                SearchHit(title="t", url="https://random.cn/y", snippet="s"),
            ]
        })
        llm = _StubLLM()  # never called
        collector = CorpusCollector(store=store, llm=llm, search=search)  # type: ignore[arg-type]
        result = collector.collect("字节跳动")
        # No fetches because no preferred-domain candidates
        assert result.hits_evaluated == 0
        assert result.inserted == 0


# ═══════════════════════════════════════════════════════════════════
# META-AGENT SWEEP
# ═══════════════════════════════════════════════════════════════════


def _seed_app(store, company: str, status: str = "applied") -> int:
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO jobs(source, title, company, raw_text, content_hash) "
            "VALUES ('manual', 't', ?, 'jd', ?)",
            (company, f"hash_{company}_{status}"),
        )
        cur = conn.execute(
            "INSERT INTO applications(job_id, status) VALUES "
            "((SELECT id FROM jobs WHERE company = ? ORDER BY id DESC LIMIT 1), ?)",
            (company, status),
        )
        return int(cur.lastrowid or 0)


class TestSweep:
    def test_sweep_with_no_llm_summarizes_only(self, corpus_setup) -> None:
        store = corpus_setup
        _seed_app(store, "字节跳动", status="applied")
        _seed_app(store, "字节跳动", status="rejected")
        result = sweep_company(
            "字节跳动", store=store, llm=None, search=None,
        )
        assert result.application_summary["total"] == 2
        assert result.application_summary["active"] == 1
        assert result.interview_corpus is None
        # Notes mention skipped agentic step
        assert any("skipped" in n.lower() for n in result.notes)

    def test_sweep_with_llm_runs_corpus_collection(self, corpus_setup) -> None:
        store = corpus_setup
        search = StubSearch()  # always empty results — easy/safe path
        llm = _StubLLM()
        result = sweep_company(
            "字节跳动", store=store,
            llm=llm,  # type: ignore[arg-type]
            search=search,
        )
        # corpus result is present (even though it found nothing)
        assert result.interview_corpus is not None
        assert result.interview_corpus.inserted == 0


# ═══════════════════════════════════════════════════════════════════
# /api/email/classify mode handling
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "ag.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="x")
    app = create_app(
        settings=Settings(),  # no DEEPSEEK_API_KEY
        store=store, profile=profile,
        skills=discover_skills(SKILLS_ROOT), runtime=None,
        notifier=ConsoleNotifier(),
    )
    return app, store


class TestEmailClassifyMode:
    def test_auto_falls_to_regex_when_no_api_key(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).post(
            "/api/email/classify",
            json={"text": "面试邀请", "batch": False, "mode": "auto"},
        )
        assert resp.status_code == 200
        assert resp.json()["mode"] == "regex"

    def test_explicit_regex_mode(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).post(
            "/api/email/classify",
            json={"text": "面试邀请", "batch": False, "mode": "regex"},
        )
        assert resp.json()["mode"] == "regex"

    def test_extracted_present_in_response(self, app_setup) -> None:
        """Even regex mode includes the 'extracted' key (empty dict) for
        UI consistency."""
        app, _ = app_setup
        resp = TestClient(app).post(
            "/api/email/classify",
            json={"text": "面试邀请", "batch": False, "mode": "regex"},
        )
        assert "extracted" in resp.json()["results"][0]


class TestSweepEndpoint:
    def test_sweep_returns_summary_no_llm(self, app_setup) -> None:
        app, store = app_setup
        _seed_app(store, "字节跳动")
        resp = TestClient(app).post(
            "/api/agent/sweep",
            json={"company": "字节跳动", "do_corpus": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["company"] == "字节跳动"
        assert data["application_summary"]["total"] == 1
        assert data["interview_corpus"] is None

    def test_sweep_skips_corpus_when_do_corpus_false(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).post(
            "/api/agent/sweep",
            json={"company": "腾讯", "do_corpus": False},
        )
        data = resp.json()
        assert data["interview_corpus"] is None
