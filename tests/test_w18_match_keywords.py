"""W18 — match_keywords + multi-keyword ambient dispatch tests."""
from __future__ import annotations

import json as _json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.agent_runtime import _schema as harness_schema
from offerguide.match_keywords import (
    ANCHOR_KEYWORDS,
    DEFAULT_KEYWORDS_PER_CYCLE,
    KEYWORD_VOCAB,
    explain_match,
    extract_keywords,
    to_search_phrases,
)

# ─────────────────── Keyword extractor tests ───────────────────

class TestKeywordExtractor:
    def test_empty_resume_returns_anchors_only(self):
        hits = extract_keywords(None, max_keywords=8)
        keywords = to_search_phrases(hits)
        assert all(a in keywords for a in ANCHOR_KEYWORDS)

    def test_real_user_resume_finds_niche_keywords(self):
        """Sanity check: real user resume content yields niche AI keywords,
        not just generic 'AI'."""
        # Synthesized minimal slice mirroring the real resume content
        resume = (
            "AI Agent 实习 | Deep Research Agent 项目 | "
            "扩散模型 + GRPO 强化学习 + SFT 微调 LoRA | "
            "PyTorch HuggingFace TRL claude code"
        )
        hits = extract_keywords(resume, max_keywords=8)
        kws = set(to_search_phrases(hits))
        # All these should appear because aliases match real resume content
        assert "AI Agent" in kws
        assert "Deep Research" in kws
        assert "Diffusion 模型" in kws
        assert "RLHF" in kws  # GRPO is alias
        assert "大模型微调" in kws  # SFT/LoRA are aliases
        assert "PyTorch 实习" in kws

    def test_active_goal_takes_priority(self):
        """User's explicit active goal beats any vocab match."""
        hits = extract_keywords("pytorch python", active_goal="字节大模型实习", max_keywords=5)
        kws = to_search_phrases(hits)
        assert "字节大模型实习" in kws
        # And it should be at the top (weight=10 > vocab matches)
        assert hits[0].keyword == "字节大模型实习"

    def test_priority_order_niche_wins_over_generic(self):
        """When all match equally, niche keywords ('AI Agent') come before
        generic ones ('PyTorch 实习')."""
        # "ai agent" matches AI Agent (1 alias), "pytorch" matches PyTorch (1 alias)
        # Both weight=1; vocab order should put AI Agent first (niche-first design)
        hits = extract_keywords("AI Agent pytorch", max_keywords=8)
        kws = to_search_phrases(hits)
        ai_idx = kws.index("AI Agent")
        py_idx = kws.index("PyTorch 实习")
        assert ai_idx < py_idx, "AI Agent (niche) should come before PyTorch (generic)"

    def test_max_keywords_respected(self):
        # A resume that hits everything
        resume = " ".join(alias for _, aliases in KEYWORD_VOCAB for alias in aliases)
        hits = extract_keywords(resume, max_keywords=3)
        assert len(hits) == 3

    def test_anchors_added_when_resume_has_few_hits(self):
        """If resume only matches 2 vocab entries, anchors fill to max_keywords."""
        hits = extract_keywords("pytorch", max_keywords=5)
        kws = to_search_phrases(hits)
        # PyTorch hit + at least one anchor
        assert "PyTorch 实习" in kws
        assert any(a in kws for a in ANCHOR_KEYWORDS)

    def test_default_per_cycle_constant_sane(self):
        assert 1 <= DEFAULT_KEYWORDS_PER_CYCLE <= 10

    def test_explain_match_is_human_readable(self):
        hits = extract_keywords("pytorch ai agent", max_keywords=2)
        for h in hits:
            text = explain_match(h)
            assert isinstance(text, str) and len(text) > 0


# ─────────────────── Multi-keyword ambient dispatch ───────────────────

class TestAmbientMultiKeywordDispatch:
    def test_crawl_verified_official_per_keyword_dedups_across_kw(self, tmp_path, monkeypatch):
        """Same job (same content_hash) found via 2 different keywords →
        ingested only once. Counters reflect this."""
        from offerguide.platforms import RawJob
        from offerguide.platforms.official_jobs import SourceSearchResult
        from offerguide.workers.ambient import _crawl_verified_official_per_keyword

        store = offerguide.Store(tmp_path / "amb_kw.db")
        store.init_schema()
        harness_schema.init_agent_runtime_schema(store)

        # Build a fake job that's identical across both keyword calls
        identical_job = RawJob(
            source="tencent_campus",
            source_id="post123", url="https://join.qq.com/jobdesc.html?postId=123",
            title="AI Agent 实习", company="腾讯", location="深圳",
            raw_text=("职位名: AI Agent 实习\n公司: 腾讯\n## 岗位职责\n" + "x" * 200),
            extras={"source_verified": True, "post_id": "post123"},
        )

        # Patch search_verified_official_jobs to return same job for any keyword
        call_log = []
        def _fake_search(*, company, keyword, limit):
            call_log.append(keyword)
            return [SourceSearchResult(
                source="tencent_campus", status="ok",
                evidence_url="https://join.qq.com/api/v1/position/searchPosition",
                jobs=[identical_job], note=f"fake for kw={keyword}",
            )]
        monkeypatch.setattr(
            "offerguide.workers.ambient.search_verified_official_jobs", _fake_search,
            raising=False,
        )
        # Have to monkeypatch where _crawl_verified_official_per_keyword imports it
        import offerguide.platforms.official_jobs as oj_mod
        monkeypatch.setattr(oj_mod, "search_verified_official_jobs", _fake_search)

        result = _crawl_verified_official_per_keyword(
            store=store, keywords=["AI Agent", "Deep Research"], limit_per_kw=3,
        )
        assert call_log == ["AI Agent", "Deep Research"]
        # First keyword inserts, second is duplicate
        assert result["inserted_total"] == 1
        assert result["duplicate_total"] == 1
        assert result["per_keyword"]["AI Agent"] == 1
        assert result["per_keyword"]["Deep Research"] == 0

    def test_discovered_keyword_attribution_on_ingested_job(self, tmp_path, monkeypatch):
        """When ingested via verified_official, the discovered_keyword is
        recorded in extras_json so /recommended can show '↳ 由 keyword X 找到'."""
        from offerguide.platforms import RawJob
        from offerguide.platforms.official_jobs import SourceSearchResult
        from offerguide.workers.ambient import _crawl_verified_official_per_keyword

        store = offerguide.Store(tmp_path / "amb_attr.db")
        store.init_schema()
        harness_schema.init_agent_runtime_schema(store)

        rj = RawJob(
            source="tencent_campus", source_id="p9", url="https://join.qq.com/x?postId=9",
            title="测试", company="腾讯", location="北京",
            raw_text="职位名: 测试\n公司: 腾讯\n## 内容\n" + "y" * 300,
            extras={},
        )
        def _fake_search(*, company, keyword, limit):
            return [SourceSearchResult(
                source="tencent_campus", status="ok", evidence_url="x",
                jobs=[rj], note="",
            )]
        import offerguide.platforms.official_jobs as oj_mod
        monkeypatch.setattr(oj_mod, "search_verified_official_jobs", _fake_search)

        _crawl_verified_official_per_keyword(
            store=store, keywords=["AI Agent"], limit_per_kw=3,
        )
        # Check the row's extras_json now has discovered_keyword
        with store.connect() as conn:
            row = conn.execute(
                "SELECT extras_json FROM jobs WHERE title = '测试'"
            ).fetchone()
        assert row is not None
        extras = _json.loads(row[0])
        assert extras.get("discovered_keyword") == "AI Agent"
        assert extras.get("discovered_via") == "verified_official"


# ─────────────────── /recommended UI shows W18 stuff ───────────────────

@pytest.fixture
def w18_client(tmp_path):
    from offerguide.skills import discover_skills
    from offerguide.ui.notify import ConsoleNotifier
    from offerguide.ui.web import create_app

    store = offerguide.Store(tmp_path / "w18.db")
    store.init_schema()
    harness_schema.init_agent_runtime_schema(store)
    skills = discover_skills(Path(__file__).parent.parent / "src/offerguide/skills")
    s = Settings(deepseek_api_key="", default_model="stub")
    app = create_app(
        settings=s, store=store, profile=None, skills=skills,
        runtime=None, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestRecommendedShowsW18:
    def test_renders_keyword_strip(self, w18_client):
        client, _ = w18_client
        resp = client.get("/recommended?type=all")
        assert resp.status_code == 200
        # Anchor keywords always present
        assert "agent 用这些 keyword 找岗位" in resp.text
        # Anchor keyword shows up (since no profile, only anchors)
        assert any(a in resp.text for a in ANCHOR_KEYWORDS)

    def test_renders_diversity_bar_with_jobs(self, w18_client):
        client, store = w18_client
        # Insert 1 大厂 job + 2 中小厂 jobs
        with store.connect() as conn:
            for i, (src, title, company) in enumerate([
                ("baidu_intern", "AI 实习生", "百度"),
                ("agent_search", "Agent 工程师", "智谱AI"),
                ("agent_search", "LLM 算法实习", "月之暗面"),
            ]):
                conn.execute(
                    "INSERT INTO jobs (source, title, company, raw_text, extras_json, content_hash) "
                    "VALUES (?, ?, ?, ?, '{}', ?)",
                    (src, title, company, "x" * 200, f"h_w18_{i}"),
                )
            conn.commit()
        resp = client.get("/recommended?type=all")
        assert resp.status_code == 200
        # Diversity stats render
        assert "中小厂" in resp.text
        # Both 智谱AI and 月之暗面 are 中小厂
        assert "智谱AI" in resp.text
        assert "月之暗面" in resp.text

    def test_card_shows_discovered_keyword(self, w18_client):
        client, store = w18_client
        with store.connect() as conn:
            extras = _json.dumps({
                "discovered_keyword": "Diffusion 模型",
                "discovered_via": "agent_search",
            })
            conn.execute(
                "INSERT INTO jobs (source, title, company, raw_text, extras_json, content_hash) "
                "VALUES ('agent_search', '扩散模型实习生', '某AI创业', ?, ?, 'h_w18_kw')",
                ("x" * 200, extras),
            )
            conn.commit()
        resp = client.get("/recommended?type=all")
        assert "由 keyword" in resp.text
        assert "Diffusion 模型" in resp.text
