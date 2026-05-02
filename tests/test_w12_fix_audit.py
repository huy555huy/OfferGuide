"""W12-fix — 端到端 audit 修复回归测试。

Audit 发现 3 个断点 (2026-05-02):
  1. spider raw_text 138 < MIN_TEXT_FOR_AUTO_EVAL (200) — 主链断
  2. daemon 从未启动 + 无健康可视化
  3. user_facts 表满但 0 SKILL retrieve 它

这一组测试锁住修复，避免回归：

- ``jd_enricher`` 模块: enrich_one + enrich_pending + status 持久化
- daemon_runs schema + scheduler.trigger_once 走 _wrap 记录
- SkillRuntime 调 invoke 前自动注入 user_facts (mock LLM 验证 system_msg)
- AwesomeJobsSpider 默认 max_stale_days 365 (Campus2026 的 2025/7/* 不被丢)
- _daemon_health helper 返回 7 个 job 名字 (即使全 'never')
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide import jd_enricher
from offerguide import user_facts as uf
from offerguide.config import Settings
from offerguide.profile import UserProfile
from offerguide.skills import SkillRuntime, discover_skills
from offerguide.spiders.awesome_jobs import AwesomeJobsSpider
from offerguide.ui.notify import ConsoleNotifier
from offerguide.ui.web import _daemon_health, create_app

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def store(tmp_path: Path):
    s = offerguide.Store(tmp_path / "fix.db")
    s.init_schema()
    return s


# ═══════════════════════════════════════════════════════════════════
# Fix 1: jd_enricher
# ═══════════════════════════════════════════════════════════════════


@dataclass
class _FakeLLMResp:
    content: str
    latency_ms: int = 100
    prompt_tokens: int = 10
    completion_tokens: int = 5
    cost_usd: float = 0.0
    model: str = "fake"
    raw: dict = None


class _FakeLLM:
    """LLM stub that returns canned JSON for jd_enricher extraction."""
    def __init__(self, response_dict: dict):
        self._resp = json.dumps(response_dict, ensure_ascii=False)

    def chat(self, messages, **kwargs):
        return _FakeLLMResp(content=self._resp)


_GOOD_EXTRACTED = {
    "page_kind": "single_jd",
    "best_role_title": "AI Agent 后端实习",
    "responsibilities": [
        "用 LangGraph 搭多 agent workflow", "评测 GAIA/SWE-bench", "调 prompt + RAG"
    ],
    "requirements": ["Python", "PyTorch", "LangGraph 经验"],
    "nice_to_have": ["DSPy", "GRPO"],
    "team_or_business": "Seed AI Agent",
    "location": "上海",
    "salary_or_level": None,
    "deadline": None,
    "rationale": "title 命中 role_hint",
}


class TestJDEnricher:
    def _seed_job(self, store, raw: str = "公司: X\n投递入口: https://x.com/a", url: str = "https://example.com/job") -> int:
        with store.connect() as c:
            cur = c.execute(
                "INSERT INTO jobs(source, content_hash, url, company, raw_text) "
                "VALUES ('test', ?, ?, 'X', ?) RETURNING id",
                (str(hash(raw)), url, raw),
            )
            return int(cur.fetchone()[0])

    def test_no_url_path(self, store) -> None:
        jid = self._seed_job(store, url="")
        result = jd_enricher.enrich_one(
            store, job_id=jid, llm=_FakeLLM(_GOOD_EXTRACTED),
        )
        assert result.status == "no_url"

    def test_no_llm_path(self, store) -> None:
        jid = self._seed_job(store)
        result = jd_enricher.enrich_one(store, job_id=jid, llm=None)
        assert result.status == "no_llm"

    def test_status_persisted(self, store) -> None:
        jid = self._seed_job(store, url="")
        jd_enricher.enrich_one(
            store, job_id=jid, llm=_FakeLLM(_GOOD_EXTRACTED),
        )
        with store.connect() as c:
            extras = c.execute(
                "SELECT json_extract(extras_json, '$.enrich_status') FROM jobs WHERE id=?",
                (jid,),
            ).fetchone()
        assert extras[0] == "no_url"

    def test_html_to_text_strips_scripts(self) -> None:
        html = (
            "<html><body><script>alert(1)</script>"
            "<style>.x{}</style>"
            "<h1>真正的内容</h1>段落 1</body></html>"
        )
        text = jd_enricher._html_to_text(html)
        assert "alert" not in text
        assert "真正的内容" in text

    def test_format_extracted_includes_responsibilities(self) -> None:
        out = jd_enricher._format_extracted_as_raw_text(
            extracted=_GOOD_EXTRACTED,
            original_raw="公司: X",
            company="字节跳动",
        )
        assert "工作职责" in out
        assert "用 LangGraph 搭多 agent workflow" in out
        assert "任职要求" in out
        # original_raw preserved at the bottom
        assert "公司: X" in out
        # length should be over MIN threshold
        assert len(out) >= jd_enricher.MIN_RAW_TEXT_AFTER_ENRICH

    def test_enrich_pending_skips_long_jobs(self, store) -> None:
        # Insert a job with already long raw_text — should not be picked up
        with store.connect() as c:
            c.execute(
                "INSERT INTO jobs(source, content_hash, url, company, raw_text) "
                "VALUES ('test', 'h_long', 'https://x.com', 'C', ?)",
                ("X" * 500,),
            )
        result = jd_enricher.enrich_pending(
            store, llm=_FakeLLM(_GOOD_EXTRACTED), limit=5,
        )
        assert result["scanned"] == 0  # long-raw_text job filtered out


# ═══════════════════════════════════════════════════════════════════
# Fix 2: daemon_runs + trigger_once goes through _wrap
# ═══════════════════════════════════════════════════════════════════


class TestDaemonRunsTracking:
    def test_schema_has_daemon_runs(self, store) -> None:
        with store.connect() as c:
            cols = {
                r[1] for r in c.execute("PRAGMA table_info(daemon_runs)").fetchall()
            }
        assert {"job_name", "started_at", "ended_at", "status", "summary_json"} <= cols

    def test_trigger_once_records_run(self, store) -> None:
        from offerguide.autonomous.scheduler import (
            AutonomousScheduler,
            JobContext,
            JobSpec,
        )
        ctx = JobContext(
            settings=Settings(), store=store, llm=None,
        )
        sched = AutonomousScheduler(ctx)
        sched.add(JobSpec(
            name="t1", func=lambda ctx: {"counted": 5},
            trigger="cron", trigger_kwargs={"hour": 0},
        ))
        result = sched.trigger_once("t1")
        assert result == {"counted": 5}

        with store.connect() as c:
            rows = c.execute(
                "SELECT job_name, status, summary_json FROM daemon_runs"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "t1"
        assert rows[0][1] == "ok"
        assert "counted" in rows[0][2]

    def test_trigger_once_records_error(self, store) -> None:
        from offerguide.autonomous.scheduler import (
            AutonomousScheduler,
            JobContext,
            JobSpec,
        )
        ctx = JobContext(settings=Settings(), store=store, llm=None)
        sched = AutonomousScheduler(ctx)

        def boom(ctx):
            raise RuntimeError("simulated")

        sched.add(JobSpec(
            name="bad", func=boom, trigger="cron", trigger_kwargs={"hour": 0},
        ))
        with pytest.raises(RuntimeError):
            sched.trigger_once("bad")
        with store.connect() as c:
            row = c.execute(
                "SELECT status, error_text FROM daemon_runs WHERE job_name='bad'"
            ).fetchone()
        assert row[0] == "error"
        assert "simulated" in row[1]


class TestDaemonHealthHelper:
    def test_returns_7_jobs_even_when_empty(self, store) -> None:
        out = _daemon_health(store)
        assert len(out) == 7
        names = {j["name"] for j in out}
        assert names == {
            "extract_facts", "discover_jobs", "jd_enrich",
            "corpus_classify", "silence_check",
            "corpus_refresh", "brief_update",
        }
        # All should be 'never'
        assert all(j["status"] == "never" for j in out)

    def test_reflects_run_history(self, store) -> None:
        with store.connect() as c:
            c.execute(
                "INSERT INTO daemon_runs(job_name, status, summary_json, ended_at) "
                "VALUES ('discover_jobs', 'ok', '{\"new\": 5}', julianday('now'))"
            )
        out = _daemon_health(store)
        disc = next(j for j in out if j["name"] == "discover_jobs")
        assert disc["status"] == "ok"
        assert "new=5" in disc["last_summary_str"]


# ═══════════════════════════════════════════════════════════════════
# Fix 3: user_facts injected into SkillRuntime
# ═══════════════════════════════════════════════════════════════════


class _CapturingLLM:
    """Captures the system message so we can verify user_facts injection."""
    def __init__(self):
        self.captured_system_msg = None

    def chat(self, messages, **kwargs):
        for m in messages:
            if m["role"] == "system":
                self.captured_system_msg = m["content"]
        return _FakeLLMResp(content='{"probability": 0.5, "reasoning": "x"}')


class TestUserFactsInjection:
    def test_facts_prepended_to_system_prompt(self, store) -> None:
        # Seed a fact
        uf.add_fact(
            store, fact_text="用户简历里有 LangGraph 经验且做过 RAG", kind="experience",
            confidence=0.9,
        )

        skills = discover_skills(SKILLS_ROOT)
        score_spec = next(s for s in skills if s.name == "score_match")

        capturing = _CapturingLLM()
        runtime = SkillRuntime(llm=capturing, store=store)
        runtime.invoke(
            score_spec,
            {"job_text": "字节跳动 LangGraph 后端实习", "user_profile": "上财应统硕士"},
        )

        # Verify system_msg contained the injected memory block
        assert capturing.captured_system_msg is not None
        assert "已知用户长期事实" in capturing.captured_system_msg
        assert "LangGraph" in capturing.captured_system_msg

        # AND the original SKILL body must still be there
        assert "score_match" in capturing.captured_system_msg.lower() or \
               "评分" in capturing.captured_system_msg

    def test_facts_used_count_bumped_after_invoke(self, store) -> None:
        uf.add_fact(
            store, fact_text="用户 RemeDi 项目 BERT 双塔 AUC 0.83",
            kind="project", confidence=0.8,
        )
        skills = discover_skills(SKILLS_ROOT)
        score_spec = next(s for s in skills if s.name == "score_match")
        runtime = SkillRuntime(llm=_CapturingLLM(), store=store)
        runtime.invoke(
            score_spec,
            {"job_text": "RemeDi 类项目要求", "user_profile": "x"},
        )
        facts = uf.list_facts(store)
        assert facts[0].used_count >= 1

    def test_inject_disabled_when_flag_false(self, store) -> None:
        uf.add_fact(
            store, fact_text="用户 LangGraph 经验丰富 (待注入)",
            kind="experience", confidence=0.9,
        )
        skills = discover_skills(SKILLS_ROOT)
        score_spec = next(s for s in skills if s.name == "score_match")
        capturing = _CapturingLLM()
        runtime = SkillRuntime(llm=capturing, store=store)
        runtime.invoke(
            score_spec,
            {"job_text": "LangGraph 后端", "user_profile": "x"},
            inject_long_term_memory=False,
        )
        # System msg must NOT include the memory block
        assert "已知用户长期事实" not in (capturing.captured_system_msg or "")

    def test_invocation_hash_unaffected_by_memory(self, store) -> None:
        """Critical: same canonical inputs must hash same regardless of
        evolving user_facts. Otherwise GEPA trainset dedup breaks."""
        skills = discover_skills(SKILLS_ROOT)
        score_spec = next(s for s in skills if s.name == "score_match")

        # Run 1 — empty memory
        runtime = SkillRuntime(llm=_CapturingLLM(), store=store)
        r1 = runtime.invoke(
            score_spec, {"job_text": "X", "user_profile": "Y"},
        )

        # Run 2 — memory now has facts (would change system_msg but not hash)
        uf.add_fact(
            store, fact_text="新加的 fact 影响 system_msg", kind="experience",
            confidence=0.8,
        )
        r2 = runtime.invoke(
            score_spec, {"job_text": "X", "user_profile": "Y"},
        )
        assert r1.input_hash == r2.input_hash


# ═══════════════════════════════════════════════════════════════════
# Fix 4: stale-filter default 365
# ═══════════════════════════════════════════════════════════════════


class TestStaleFilterDefault:
    def test_default_keeps_2025_july(self, monkeypatch) -> None:
        """2025-07-* entries should NOT be dropped by the new 365-day default
        on 2026-05-02 (~10 months back)."""
        from dataclasses import dataclass

        @dataclass
        class _R:
            text: str
            status_code: int = 200

        sample = (
            "## 互联网 && AI\n\n"
            "| 公司 | 投递链接 | 更新日期 | 地点 | 备注 |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| 字节跳动 | [校招](https://x.com/a) | 2025/7/20 | 全国 | x |\n"
            "| 阿里巴巴 | [校招](https://x.com/b) | 2025/7/1 | 全国 | x |\n"
        )
        monkeypatch.setattr(
            "offerguide.spiders.awesome_jobs.rate_limited_get",
            lambda url, **kw: _R(text=sample),
        )
        sp = AwesomeJobsSpider(sources=[("t/r", "README.md")])
        result = sp.run(max_items=10)
        companies = [rj.company for rj in result.raw_jobs]
        # 2025/7/* with default 365 + today=2026-05-02 → ~10 months → kept
        assert "字节跳动" in companies
        assert "阿里巴巴" in companies


# ═══════════════════════════════════════════════════════════════════
# UI integration — /dashboard renders daemon health
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app_setup(tmp_path: Path):
    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    profile = UserProfile(raw_resume_text="resume")
    skills = discover_skills(SKILLS_ROOT)
    app = create_app(
        settings=Settings(), store=store, profile=profile,
        skills=skills, runtime=None, notifier=ConsoleNotifier(),
    )
    return app, store


class TestDashboardDaemonCard:
    def test_dashboard_shows_daemon_health(self, app_setup) -> None:
        app, _ = app_setup
        resp = TestClient(app).get("/dashboard")
        assert resp.status_code == 200
        assert "Daemon 健康" in resp.text
        # All 7 job names listed
        for name in ("extract_facts", "discover_jobs", "jd_enrich",
                     "corpus_classify", "silence_check",
                     "corpus_refresh", "brief_update"):
            assert f"<code>{name}</code>" in resp.text
        # Empty state warning visible
        assert "所有 daemon 从未运行" in resp.text


_ = json  # keep import
