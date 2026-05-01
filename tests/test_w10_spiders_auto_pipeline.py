"""W10 — Spider framework + auto-eval pipeline.

Coverage:

- spiders._base
  * rate_limited_get sleeps when called too fast against same host
  * SpiderError raised on transport failures
- spiders.awesome_jobs
  * parse_markdown_tables extracts company/url/date/location/note rows
  * Section-context tracking (H2/H3 carried into rows)
  * Section filter respects sections_keep_substrings
- auto_pipeline.run_spider_sweep
  * Ingests candidates, dedups on second run
  * Skips auto-eval when no runtime/skills/profile
  * Skips auto-eval for short raw_text (metadata-only entries)
  * Pushes to inbox when probability >= threshold
- discover_jobs job
  * Returns ``skipped`` cleanly when no spiders configured
  * Reports summary with by-spider breakdown
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from offerguide import auto_pipeline
from offerguide.memory import Store
from offerguide.platforms import RawJob
from offerguide.spiders._base import (
    DEFAULT_PER_HOST_GAP_S,
    SpiderError,
    SpiderResult,
    rate_limited_get,
    reset_host_throttle_for_tests,
)
from offerguide.spiders.awesome_jobs import (
    AwesomeJobsSpider,
    parse_markdown_tables,
)

# ═══════════════════════════════════════════════════════════════════
# spiders._base
# ═══════════════════════════════════════════════════════════════════


SAMPLE_MD = """\
# Title

> intro

## 校招正式批

### 互联网 && AI

| 公司     | 招聘状态&&投递链接 | 更新日期   | 地点 | 备注 |
| -------- | ------------------- | ---------- | ---- | ---- |
| 阿里巴巴 | [校招正式批](https://talent.alibaba.com/campus/x) | 2026/4/15   | 全国 | 7月开启 |
| 字节跳动 | [校招正式批](https://jobs.bytedance.com/zh/position) | 2026/4/15   | 全国 | 7月开启 |
| 腾讯     | [校招正式批](https://join.qq.com/post.html?query=p_2) | 2026/4/15   | 北京 |  |

### 银行&&保险

| 公司   | 招聘状态&&投递链接 | 更新日期 | 地点 | 备注 |
| ------ | ------------------- | -------- | ---- | ---- |
| 招商银行 | [校招正式批](https://career.cmbchina.com/) | 2026/4/15 | 全国 | |

## 实习信息

### 互联网 && AI

| 公司 | 投递链接 | 更新 | 地点 | 备注 |
| ---- | -------- | ---- | ---- | ---- |
| 美团 | [实习](https://campus.meituan.com/intern) | 2026/4/20 | 北京 |  |
"""


class TestParseMarkdown:
    def test_extracts_all_rows(self) -> None:
        rows = parse_markdown_tables(SAMPLE_MD)
        # 3 互联网 + 1 银行 + 1 实习 互联网 = 5
        assert len(rows) == 5

    def test_company_url_pair(self) -> None:
        rows = parse_markdown_tables(SAMPLE_MD)
        bytedance = next(r for r in rows if r.company == "字节跳动")
        assert bytedance.apply_url == "https://jobs.bytedance.com/zh/position"
        assert bytedance.apply_label == "校招正式批"

    def test_section_context_carried(self) -> None:
        rows = parse_markdown_tables(SAMPLE_MD)
        ali = next(r for r in rows if r.company == "阿里巴巴")
        assert "互联网" in ali.section
        assert "校招正式批" in ali.section

        cmb = next(r for r in rows if r.company == "招商银行")
        assert "银行" in cmb.section

    def test_skips_rows_without_url(self) -> None:
        md = "| 公司 | 投递链接 |\n| --- | --- |\n| X | 没有链接 |\n"
        rows = parse_markdown_tables(md)
        assert rows == []


class TestAwesomeJobsSpider:
    """Spider behavior with mocked HTTP."""

    @pytest.fixture
    def patched_get(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict = {"calls": []}

        @dataclass
        class _Resp:
            text: str
            status_code: int = 200

        def fake_get(url, **kwargs):
            captured["calls"].append(url)
            return _Resp(text=SAMPLE_MD)

        monkeypatch.setattr(
            "offerguide.spiders.awesome_jobs.rate_limited_get", fake_get
        )
        return captured

    def test_run_yields_jobs(self, patched_get) -> None:
        sp = AwesomeJobsSpider(sources=[("namewyf/Campus2026", "README.md")])
        result = sp.run(max_items=10)
        assert result.spider_name == "awesome_jobs"
        assert result.pages_fetched == 1
        assert len(result.raw_jobs) == 4  # 互联网×3 + 实习×1, 银行 filtered out
        assert result.errors == []

    def test_section_filter_drops_finance(self, patched_get) -> None:
        sp = AwesomeJobsSpider(sources=[("test/repo", "README.md")])
        result = sp.run(max_items=10)
        companies = [rj.company for rj in result.raw_jobs]
        assert "招商银行" not in companies
        assert "阿里巴巴" in companies

    def test_dedup_via_company_plus_url(self, patched_get) -> None:
        # The spider's internal dedup uses (company, url) — the same row
        # appearing across multiple sources should be emitted once.
        sp = AwesomeJobsSpider(
            sources=[("test/repo1", "README.md"), ("test/repo2", "README.md")],
        )
        result = sp.run(max_items=10)
        # 2 fetches but dedup => still 4 jobs
        assert result.pages_fetched == 2
        assert len(result.raw_jobs) == 4

    def test_max_items_caps_output(self, patched_get) -> None:
        sp = AwesomeJobsSpider(sources=[("t/r", "README.md")])
        result = sp.run(max_items=2)
        assert len(result.raw_jobs) == 2

    def test_raw_job_shape(self, patched_get) -> None:
        sp = AwesomeJobsSpider(sources=[("t/r", "README.md")])
        result = sp.run(max_items=3)
        bytedance = next(rj for rj in result.raw_jobs if rj.company == "字节跳动")
        assert bytedance.source == "awesome_jobs"
        assert bytedance.url == "https://jobs.bytedance.com/zh/position"
        assert bytedance.location == "全国"
        assert "公司: 字节跳动" in bytedance.raw_text
        assert "github_repo" in bytedance.extras


class TestRateLimitedGet:
    def test_per_host_throttle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two requests to same host within gap → second sleeps."""
        reset_host_throttle_for_tests()
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

        @dataclass
        class _R:
            status_code: int = 200
            text: str = "ok"

        def fake_get(url, **kw):
            return _R()

        monkeypatch.setattr(
            "offerguide.spiders._base.httpx.get", fake_get
        )

        rate_limited_get("https://example.com/a")
        rate_limited_get("https://example.com/b")
        # Second call should have slept ~ DEFAULT_PER_HOST_GAP_S
        assert any(s > 0 for s in slept)

    def test_different_hosts_no_throttle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reset_host_throttle_for_tests()
        slept: list[float] = []
        monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

        @dataclass
        class _R:
            status_code: int = 200
            text: str = "ok"

        monkeypatch.setattr(
            "offerguide.spiders._base.httpx.get", lambda u, **k: _R()
        )
        rate_limited_get("https://a.com/x")
        rate_limited_get("https://b.com/x")
        assert all(s == 0 for s in slept)

    def test_400_raises_spider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reset_host_throttle_for_tests()

        @dataclass
        class _R:
            status_code: int = 404
            text: str = "not found"

        monkeypatch.setattr(
            "offerguide.spiders._base.httpx.get", lambda u, **k: _R()
        )
        with pytest.raises(SpiderError, match="404"):
            rate_limited_get("https://example.com/x")


# ═══════════════════════════════════════════════════════════════════
# auto_pipeline
# ═══════════════════════════════════════════════════════════════════


class _FakeSpider:
    name = "fake"

    def __init__(self, jobs: list[RawJob]) -> None:
        self._jobs = jobs

    def run(self, *, max_items: int = 30) -> SpiderResult:
        return SpiderResult(
            spider_name=self.name, raw_jobs=list(self._jobs[:max_items])
        )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "w10.db")
    s.init_schema()
    return s


class TestRunSpiderSweep:
    def test_ingests_new_jobs(self, store: Store) -> None:
        spider = _FakeSpider([
            RawJob(
                source="fake", url=f"https://example.com/j{i}",
                title=f"job{i}", company=f"C{i}",
                raw_text=f"jd body {i}",
            )
            for i in range(3)
        ])
        result = auto_pipeline.run_spider_sweep(
            spider, store=store, auto_eval=False,
        )
        assert result.candidates_found == 3
        assert len(result.new_jobs) == 3
        assert result.duplicate_count == 0

    def test_dedup_on_second_run(self, store: Store) -> None:
        rj = RawJob(
            source="fake", url="https://example.com/j",
            title="t", company="C", raw_text="body",
        )
        spider = _FakeSpider([rj])
        first = auto_pipeline.run_spider_sweep(spider, store=store, auto_eval=False)
        second = auto_pipeline.run_spider_sweep(spider, store=store, auto_eval=False)
        assert len(first.new_jobs) == 1
        assert len(second.new_jobs) == 0
        assert second.duplicate_count == 1

    def test_skips_eval_when_no_runtime(self, store: Store) -> None:
        spider = _FakeSpider([
            RawJob(
                source="fake", url="https://example.com/j",
                title="t", company="C",
                raw_text="x" * 500,  # long enough to trigger eval if runtime present
            ),
        ])
        result = auto_pipeline.run_spider_sweep(
            spider, store=store, auto_eval=True,  # but runtime=None
        )
        assert len(result.new_jobs) == 1
        assert result.auto_evaluated == []  # no runtime → no eval

    def test_evaluate_threshold_short_text(self, store: Store) -> None:
        """JDs with raw_text shorter than MIN_TEXT_FOR_AUTO_EVAL should
        be ingested but not scored — saves tokens on metadata-only rows
        from awesome-style sources.

        We exercise ``evaluate_new_job`` directly rather than through a
        spider sweep so the test is independent of ingest behavior.
        """

        class _StubRuntime:
            def invoke(self, spec, inputs):
                pytest.fail("Runtime should not be called for short JDs")

        from offerguide.skills import SkillSpec
        score_spec = SkillSpec(
            name="score_match",
            version="v1",
            description="x",
            body="...",
        )
        # evaluate_new_job direct-call should return None for short text
        with store.connect() as c:
            c.execute(
                "INSERT INTO jobs(source,raw_text,content_hash,title,company) "
                "VALUES ('fake','short','h1','t','C')"
            )
            jid = int(c.execute("SELECT id FROM jobs").fetchone()[0])
        result = auto_pipeline.evaluate_new_job(
            store=store, job_id=jid, score_spec=score_spec,
            runtime=_StubRuntime(),  # type: ignore[arg-type]
            user_profile_text="resume",
        )
        assert result is None

    def test_high_score_pushes_to_inbox(
        self, store: Store, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When auto_eval scores a JD >= threshold, an inbox item appears."""
        long_text = "JD body " * 60  # > MIN_TEXT_FOR_AUTO_EVAL
        spider = _FakeSpider([
            RawJob(
                source="fake", url="https://example.com/j",
                title="后端实习", company="字节跳动",
                raw_text=long_text,
            ),
        ])

        # Fake runtime returns a high probability
        from dataclasses import dataclass as _dc

        @_dc
        class _FakeSkillResult:
            parsed: dict
            skill_run_id: int

        class _FakeRuntime:
            def invoke(self, spec, inputs):
                return _FakeSkillResult(
                    parsed={"probability": 0.78, "deal_breakers": []},
                    skill_run_id=42,
                )

        from offerguide.skills import SkillSpec
        score_spec = SkillSpec(
            name="score_match", version="v1", description="x",
            body="...",
        )

        result = auto_pipeline.run_spider_sweep(
            spider,
            store=store,
            runtime=_FakeRuntime(),  # type: ignore[arg-type]
            skills=[score_spec],
            user_profile_text="resume",
            auto_eval=True,
            inbox_threshold=0.55,
        )
        assert len(result.auto_evaluated) == 1
        assert len(result.pushed_to_inbox) == 1

        # And the inbox row is real
        with store.connect() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM inbox_items WHERE kind = 'consider_jd'"
            ).fetchone()[0]
        assert n == 1

    def test_below_threshold_no_inbox(
        self, store: Store
    ) -> None:
        long_text = "JD body " * 60
        spider = _FakeSpider([
            RawJob(
                source="fake", url="https://example.com/j",
                title="t", company="C", raw_text=long_text,
            ),
        ])

        from dataclasses import dataclass as _dc

        @_dc
        class _R:
            parsed: dict
            skill_run_id: int

        class _Rt:
            def invoke(self, spec, inputs):
                return _R(parsed={"probability": 0.20, "deal_breakers": []}, skill_run_id=1)

        from offerguide.skills import SkillSpec
        spec = SkillSpec(
            name="score_match", version="v1", description="x",
            body="...",
        )
        result = auto_pipeline.run_spider_sweep(
            spider, store=store,
            runtime=_Rt(),  # type: ignore[arg-type]
            skills=[spec], user_profile_text="resume",
            auto_eval=True, inbox_threshold=0.55,
        )
        assert len(result.auto_evaluated) == 1
        assert len(result.pushed_to_inbox) == 0


# ═══════════════════════════════════════════════════════════════════
# discover_jobs daemon job
# ═══════════════════════════════════════════════════════════════════


class TestDiscoverJobsJob:
    def test_skips_when_no_spiders(self, store: Store, monkeypatch) -> None:
        from offerguide.autonomous.jobs import discover_jobs

        # Patch spider set to be empty
        monkeypatch.setattr(
            "offerguide.autonomous.jobs.discover_jobs.build_default_spider_set",
            lambda: [],
        )

        @dataclass
        class _Ctx:
            store: Store
            settings: object = None
            llm: object = None
            runtime: object = None
            skills: list = None
            user_profile_text: str | None = None
            notifier: object = None

        result = discover_jobs.run(_Ctx(store=store, skills=[]))
        assert result == {"skipped": "no_spiders"}

    def test_runs_spiders_and_summarizes(
        self, store: Store, monkeypatch
    ) -> None:
        from offerguide.autonomous.jobs import discover_jobs

        spider = _FakeSpider([
            RawJob(
                source="fake", url=f"https://example.com/{i}",
                title=f"t{i}", company=f"C{i}", raw_text=f"jd {i}",
            )
            for i in range(3)
        ])
        monkeypatch.setattr(
            "offerguide.autonomous.jobs.discover_jobs.build_default_spider_set",
            lambda: [spider],
        )

        @dataclass
        class _Ctx:
            store: Store
            settings: object = None
            llm: object = None
            runtime: object = None
            skills: list = None
            user_profile_text: str | None = None
            notifier: object = None

        result = discover_jobs.run(_Ctx(store=store, skills=[]))
        assert result["spiders_run"] == 1
        assert result["total_new_jobs"] == 3
        assert result["auto_eval_enabled"] is False
        assert result["total_inboxed"] == 0


_ = DEFAULT_PER_HOST_GAP_S  # quiet unused-import lint
