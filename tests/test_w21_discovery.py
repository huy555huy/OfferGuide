"""W21 Phase 3 — Discovery tools + Discovery sub-agent tests.

Tools (in offerguide.tools.discovery) self-register at module import.
Tests use real Store + stubbed httpx clients / fake adapters to avoid
network calls.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import offerguide
from offerguide.agents.base import register_universal_tools
from offerguide.agents.discovery import DiscoverySubAgent
from offerguide.llm.client import LLMResponse, ToolCall
from offerguide.tools.registry import ToolRegistry


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "disc.db")
    s.init_schema()
    return s


@pytest.fixture
def reg_with_discovery():
    """Fresh registry that has discovery tools loaded (no global pollution)."""
    r = ToolRegistry()
    register_universal_tools(r)
    # Manually re-register discovery tools against THIS registry
    # (since they registered against the global at module import).
    # The cleanest way: copy entries.
    import offerguide.tools.discovery  # noqa: F401  # loads to global
    from offerguide.tools.registry import registry as global_reg
    for name in global_reg.names_in_group("discovery"):
        entry = global_reg.get(name)
        if entry is not None and r.get(name) is None:
            r.register(
                name=entry.name, group=entry.group,
                schema=entry.schema, handler=entry.handler,
                description=entry.description,
                deprecated_aliases=entry.deprecated_aliases,
            )
    return r


# ── Schema registration sanity ────────────────────────────────────


class TestDiscoveryToolRegistration:
    def test_all_8_discovery_tools_registered(self, reg_with_discovery):
        names = reg_with_discovery.names_in_group("discovery")
        expected = {
            "fetch_nowcoder", "fetch_tencent_campus", "fetch_tencent_social",
            "fetch_baidu_grad", "fetch_baidu_intern", "fetch_bytedance",
            "fetch_zerovoice", "fetch_shixiseng", "read_last_fetch_times",
        }
        assert set(names) == expected

    def test_each_tool_has_valid_schema(self, reg_with_discovery):
        for name in reg_with_discovery.names_in_group("discovery"):
            entry = reg_with_discovery.get(name)
            assert entry is not None
            schema = entry.schema
            assert "name" in schema or "description" in schema
            assert "parameters" in schema
            params = schema["parameters"]
            assert params["type"] == "object"


# ── fetch_nowcoder tool (real handler wrapping scout.crawl_nowcoder) ──


class TestFetchNowcoder:
    def test_fetch_nowcoder_handler_with_stubbed_scout(
        self, store, reg_with_discovery, monkeypatch,
    ):
        """Verify the tool wrapper extracts the right fields from FetchResult."""
        from offerguide.workers import scout

        class _FakeResult:
            inserted = 5
            duplicate = 2
            errors = []

        monkeypatch.setattr(scout, "crawl_nowcoder", lambda store_, limit: _FakeResult())

        out = reg_with_discovery.dispatch("fetch_nowcoder", {"limit": 10}, store=store)
        data = json.loads(out)
        assert data["source"] == "nowcoder"
        assert data["inserted"] == 5
        assert data["duplicates"] == 2

    def test_fetch_nowcoder_missing_store_returns_error(self, reg_with_discovery):
        out = reg_with_discovery.dispatch("fetch_nowcoder", {})
        d = json.loads(out)
        assert "error" in d
        assert "store not provided" in d["error"]

    def test_fetch_nowcoder_handler_catches_exception(
        self, store, reg_with_discovery, monkeypatch,
    ):
        from offerguide.workers import scout

        def _crash(store_, limit):
            raise RuntimeError("network down")

        monkeypatch.setattr(scout, "crawl_nowcoder", _crash)
        out = reg_with_discovery.dispatch("fetch_nowcoder", {"limit": 5}, store=store)
        d = json.loads(out)
        assert "error" in d
        assert "RuntimeError" in d["error"] or "network down" in d["error"]


# ── fetch_zerovoice ──────────────────────────────────────────────


class TestFetchZerovoice:
    def test_fetch_zerovoice_returns_company_top10(
        self, store, reg_with_discovery, monkeypatch,
    ):
        from offerguide.platforms.zerovoice import FetchResult

        fake = FetchResult(
            parsed_total=475, inserted=80, duplicate=395,
            errors=[],
            by_company={
                "腾讯": 30, "阿里巴巴": 25, "字节跳动": 15, "百度": 10,
                "美团": 5, "网易": 3, "拼多多": 2,
            },
        )
        monkeypatch.setattr(
            "offerguide.platforms.zerovoice.crawl_zerovoice",
            lambda s, *, max_jobs, verify_urls=False: fake,
        )
        out = reg_with_discovery.dispatch("fetch_zerovoice", {"max_jobs": 80}, store=store)
        d = json.loads(out)
        assert d["inserted"] == 80
        assert d["parsed_total"] == 475
        # top10 — 7 entries, all sorted desc
        assert "腾讯" in d["by_company_top10"]
        assert d["by_company_top10"]["腾讯"] == 30


# ── fetch_shixiseng ─────────────────────────────────────────────


class TestFetchShixiseng:
    def test_fetch_shixiseng_wraps_correctly(
        self, store, reg_with_discovery, monkeypatch,
    ):
        from offerguide.platforms.shixiseng import FetchResult as SsFetchResult

        fake = SsFetchResult(
            listed_total=20,
            fetched=8,
            parsed=8,
            inserted=8,
            duplicate=0,
            errors=[],
            by_company={"行知启新": 1, "香巴拉科技": 1, "百度": 1},
        )
        monkeypatch.setattr(
            "offerguide.platforms.shixiseng.crawl_shixiseng",
            lambda s, *, keyword, max_jobs: fake,
        )
        out = reg_with_discovery.dispatch(
            "fetch_shixiseng", {"keyword": "AI Agent", "max_jobs": 8},
            store=store,
        )
        d = json.loads(out)
        assert d["source"] == "shixiseng"
        assert d["inserted"] == 8
        assert "by_company" in d


# ── read_last_fetch_times ──────────────────────────────────────────


class TestReadLastFetchTimes:
    def test_reads_real_jobs_table(self, store, reg_with_discovery):
        # Insert 3 jobs from 2 sources
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (source, raw_text, content_hash) "
                "VALUES (?, ?, ?)", ("nowcoder", "x" * 300, "h1"),
            )
            conn.execute(
                "INSERT INTO jobs (source, raw_text, content_hash) "
                "VALUES (?, ?, ?)", ("nowcoder", "x" * 300, "h2"),
            )
            conn.execute(
                "INSERT INTO jobs (source, raw_text, content_hash) "
                "VALUES (?, ?, ?)", ("shixiseng", "x" * 300, "h3"),
            )
            conn.commit()
        out = reg_with_discovery.dispatch(
            "read_last_fetch_times", {}, store=store,
        )
        d = json.loads(out)
        by_src = d["by_source"]
        assert by_src["nowcoder"]["total_jobs"] == 2
        assert by_src["shixiseng"]["total_jobs"] == 1
        # Just inserted, hours_since_last should be near 0
        assert by_src["nowcoder"]["hours_since_last"] < 1.0


# ── DiscoverySubAgent end-to-end with stub LLM + fake fetcher ─────


class TestDiscoverySubAgentEndToEnd:
    def test_sub_agent_calls_fetcher_then_done(
        self, store, reg_with_discovery, monkeypatch,
    ):
        """LLM picks fetch_shixiseng, gets result, calls done(). Sub-agent
        returns clean summary."""

        # Stub the fetcher so we don't hit network
        from offerguide.platforms.shixiseng import FetchResult as SsFetchResult
        monkeypatch.setattr(
            "offerguide.platforms.shixiseng.crawl_shixiseng",
            lambda s, *, keyword, max_jobs: SsFetchResult(
                listed_total=8, fetched=8, parsed=8, inserted=8,
                duplicate=0, errors=[],
                by_company={"行知启新": 1, "香巴拉科技": 1},
            ),
        )

        # Stub LLM: turn 1 calls fetch_shixiseng, turn 2 calls done
        responses = [
            LLMResponse(
                content="先抓实习僧的实习岗", model="stub", cost_usd=0.001,
                tool_calls=[ToolCall(
                    id="t1", name="fetch_shixiseng",
                    arguments={"keyword": "AI Agent", "max_jobs": 8},
                    arguments_raw='{"keyword": "AI Agent", "max_jobs": 8}',
                )],
            ),
            LLMResponse(
                content="拉完了", model="stub", cost_usd=0.001,
                tool_calls=[ToolCall(
                    id="t2", name="done",
                    arguments={"summary": "实习僧拉了 8 个 AI Agent 实习"},
                    arguments_raw='{"summary": "实习僧拉了 8 个 AI Agent 实习"}',
                )],
            ),
        ]
        llm = MagicMock()
        llm.chat_with_tools = MagicMock(side_effect=responses)

        sub = DiscoverySubAgent(
            llm=llm, registry=reg_with_discovery, store=store, max_iter=4,
        )
        result = sub.run(goal="今天找 AI Agent 暑期实习")
        assert result.error is None
        assert result.iterations == 2
        assert result.tool_calls_made == 2
        assert "实习僧" in result.final_answer

    def test_sub_agent_sees_only_discovery_tools(
        self, store, reg_with_discovery,
    ):
        """Verify the sub-agent's tool_schemas passed to LLM are
        discovery + shared only."""
        captured = {}

        def _spy(messages, tools, temperature):
            captured["tools"] = tools
            return LLMResponse(content="ok", model="stub", tool_calls=[])

        llm = MagicMock()
        llm.chat_with_tools = _spy

        sub = DiscoverySubAgent(
            llm=llm, registry=reg_with_discovery, store=store,
        )
        sub.run(goal="x")

        tool_names = {t["function"]["name"] for t in captured["tools"]}
        # Should include all 8 discovery + done
        assert "fetch_nowcoder" in tool_names
        assert "fetch_shixiseng" in tool_names
        assert "fetch_zerovoice" in tool_names
        assert "done" in tool_names
