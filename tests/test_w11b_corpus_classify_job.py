"""W11(b) — corpus_classify daemon job + awesome_jobs stale-date filter.

Coverage:

- corpus_classify daemon job
  * runs classify_pending and returns counters
  * graceful with no LLM (deterministic prefilter only)
  * notifier called only when items processed
- awesome_jobs.AwesomeJobsSpider
  * max_stale_days drops entries older than threshold
  * empty/unparseable update_date keeps the entry (no filter applied)
  * max_stale_days=None disables filter entirely
- corpus_collector inline classification
  * (smoke) _classify_inline doesn't crash on classifier failure
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

import offerguide
from offerguide.spiders.awesome_jobs import AwesomeJobsSpider

# ═══════════════════════════════════════════════════════════════════
# corpus_classify daemon job
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def store(tmp_path: Path):
    s = offerguide.Store(tmp_path / "w11b.db")
    s.init_schema()
    return s


def _seed_pending(store, raw: str, source: str = "s1") -> int:
    """Insert a corpus row in 'pending classification' state."""
    h = hashlib.sha256(raw.encode()).hexdigest()
    with store.connect() as c:
        cur = c.execute(
            "INSERT INTO interview_experiences"
            "(company, raw_text, source, content_hash) "
            "VALUES (?,?,?,?) RETURNING id",
            ("X", raw, source, h),
        )
        return int(cur.fetchone()[0])


@dataclass
class _NoopNotifier:
    notified: list[tuple[str, str]] = None

    def __post_init__(self):
        self.notified = []

    def notify(self, *, title, body, level="info"):
        self.notified.append((title, body))


@dataclass
class _Ctx:
    store: object
    settings: object = None
    llm: object = None
    notifier: object = None


class TestCorpusClassifyJob:
    def test_runs_without_llm(self, store) -> None:
        from offerguide.autonomous.jobs import corpus_classify

        _seed_pending(
            store,
            raw=(
                "加微信 wx 训练营 包过 大厂内推 资料包 限免 "
                "1v1 辅导 课程 999 加我 dm"
            ),
        )
        notifier = _NoopNotifier()
        result = corpus_classify.run(_Ctx(store=store, notifier=notifier))

        assert result["processed"] == 1
        assert result["marketer"] == 1
        # Notifier called because items were processed
        assert len(notifier.notified) == 1
        assert "语料分类完成" in notifier.notified[0][0]

    def test_skips_notifier_when_nothing_to_process(self, store) -> None:
        from offerguide.autonomous.jobs import corpus_classify

        notifier = _NoopNotifier()
        result = corpus_classify.run(_Ctx(store=store, notifier=notifier))

        assert result["processed"] == 0
        assert notifier.notified == []  # nothing to notify about

    def test_jobspec_metadata(self) -> None:
        from offerguide.autonomous.jobs.corpus_classify import (
            CORPUS_CLASSIFY_JOB,
        )
        assert CORPUS_CLASSIFY_JOB.name == "corpus_classify"
        assert CORPUS_CLASSIFY_JOB.trigger == "cron"
        assert CORPUS_CLASSIFY_JOB.trigger_kwargs.get("hour") == 7


# ═══════════════════════════════════════════════════════════════════
# awesome_jobs stale-date filter
# ═══════════════════════════════════════════════════════════════════


_SAMPLE_MD_MIXED = """\
## 校招正式批

### 互联网 && AI

| 公司   | 投递链接 | 更新日期   | 地点 | 备注 |
| ------ | -------- | ---------- | ---- | ---- |
| 字节跳动 | [校招](https://x.com/a) | 2026/4/20 | 北京 | recent  |
| 阿里巴巴 | [校招](https://x.com/b) | 2024/3/1  | 上海 | stale   |
| 腾讯     | [校招](https://x.com/c) |            | 深圳 | unknown |
"""


@pytest.fixture
def patched_get(monkeypatch):
    """Stub rate_limited_get to serve the mixed-date sample."""
    @dataclass
    class _Resp:
        text: str
        status_code: int = 200

    monkeypatch.setattr(
        "offerguide.spiders.awesome_jobs.rate_limited_get",
        lambda url, **kw: _Resp(text=_SAMPLE_MD_MIXED),
    )
    return None


class TestStaleDateFilter:
    def test_default_filter_drops_old(self, patched_get) -> None:
        sp = AwesomeJobsSpider(sources=[("t/r", "README.md")])
        result = sp.run(max_items=10)
        companies = [rj.company for rj in result.raw_jobs]
        assert "字节跳动" in companies     # recent
        assert "腾讯" in companies         # unknown date kept
        assert "阿里巴巴" not in companies  # stale, dropped

    def test_filter_disabled_keeps_all(self, patched_get) -> None:
        sp = AwesomeJobsSpider(
            sources=[("t/r", "README.md")], max_stale_days=None,
        )
        result = sp.run(max_items=10)
        companies = [rj.company for rj in result.raw_jobs]
        assert "阿里巴巴" in companies  # stale-but-kept

    def test_aggressive_filter(self, patched_get) -> None:
        # Cut to 30 days — even 2026/4/20 (~10-30 days back)
        # might fall outside on edge dates
        sp = AwesomeJobsSpider(
            sources=[("t/r", "README.md")], max_stale_days=1,
        )
        result = sp.run(max_items=10)
        companies = [rj.company for rj in result.raw_jobs]
        # Only the one with no parseable date survives the 1-day cutoff
        assert companies == ["腾讯"]


# ═══════════════════════════════════════════════════════════════════
# corpus_collector inline classification (smoke)
# ═══════════════════════════════════════════════════════════════════


class TestInlineClassify:
    def test_classify_inline_doesnt_crash_without_llm(self, store, monkeypatch) -> None:
        """The inline classifier should swallow exceptions and let
        ingest succeed even if classify_one blows up."""
        from offerguide.agentic.corpus_collector import CorpusCollector

        # Build a minimal collector with stub LLM + search
        class _StubLLM:
            pass

        class _StubSearch:
            def search(self, q, max_results=10):
                return []

        cc = CorpusCollector(
            store=store, llm=_StubLLM(),  # type: ignore[arg-type]
            search=_StubSearch(),  # type: ignore[arg-type]
        )

        # Make corpus_quality.classify_one throw
        def _boom(**kwargs):
            raise RuntimeError("classifier exploded")

        monkeypatch.setattr(
            "offerguide.corpus_quality.classify_one", _boom,
        )

        # Seed a row to classify
        item_id = _seed_pending(store, raw="some content")

        # Should NOT raise
        cc._classify_inline(
            item_id=item_id, text="x", company="X", role_hint=None,
        )

        cc.close()
