"""W14.7 — fixes for the 6 bugs surfaced by code review.

Each test pins down a bug that would have been caught by stricter typing /
real py3.11 CI / actual end-to-end exercise of maintenance tools and
evolution signal attribution.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

import offerguide
from offerguide import goals as _goals
from offerguide.skills._runtime import _hash_invocation


# ═══════════════════════════════════════════════════════════════════
# P1 #1 — maintenance.refresh_company_corpus called a non-existent method
# ═══════════════════════════════════════════════════════════════════


class TestMaintenanceCorpusRefreshUsesCollect:
    def test_calls_collect_not_refresh_company(self, monkeypatch, tmp_path):
        """maintenance._run_corpus_refresh_for_company must invoke
        CorpusCollector.collect — refresh_company never existed."""
        from offerguide.agent import maintenance as m
        from offerguide.agentic import corpus_collector as cc

        store = offerguide.Store(tmp_path / "x.db")
        store.init_schema()

        called = {}

        @dataclass
        class _FakeResult:
            hits_seen: int = 3
            hits_evaluated: int = 2
            inserted: int = 1
            skipped_dup: int = 1
            skipped_low_quality: int = 0

        class _FakeCollector:
            def __init__(self, *a, **kw): pass
            def collect(self, company, *, role_hint=None):
                called["collect"] = (company, role_hint)
                return _FakeResult()
            # Deliberately NOT defining refresh_company — verify we don't fall
            # back to it.

        monkeypatch.setattr(cc, "CorpusCollector", _FakeCollector)
        # Also stub the search builder so we don't hit network
        from offerguide.agentic import search as _search
        monkeypatch.setattr(_search, "build_default_search", lambda: None)

        ctx = m.MaintenanceCtx(
            store=store, llm=object(), runtime=object(), skills=[], search=None,
        )
        out = m._run_corpus_refresh_one_company(
            ctx, company="字节", role_hint="AI Agent",
        )
        assert called["collect"] == ("字节", "AI Agent")
        assert out.startswith("OK:")
        assert "hits=3" in out


# ═══════════════════════════════════════════════════════════════════
# P1 #2 — regenerate_company_brief reads attrs off BriefRow wrapper
# ═══════════════════════════════════════════════════════════════════


class TestRegenerateCompanyBriefReadsBriefField:
    def test_reads_through_brief_wrapper(self, monkeypatch, tmp_path):
        """BriefRow has the CompanyBrief in `.brief`; reading
        `result.confidence` directly used to AttributeError."""
        from offerguide.agent import maintenance as m
        from offerguide import briefs as briefs_mod

        store = offerguide.Store(tmp_path / "x.db")
        store.init_schema()

        # Mock CompanyBrief + BriefRow with the real shape
        class _FakeBrief:
            confidence = 0.72
            current_app_limit = 2
            summary = "字节 AI Lab 偏研究, 成长曲线快, 周末加班常态"

        class _FakeRow:
            brief = _FakeBrief()
            company = "字节"
            last_updated_at = 0.0
            update_count = 1

        monkeypatch.setattr(briefs_mod, "refresh_brief",
                            lambda store, company, llm, **kw: _FakeRow())

        ctx = m.MaintenanceCtx(
            store=store, llm=object(), runtime=object(), skills=[],
        )
        # Should not AttributeError — the bug was result.confidence (wrapper has none)
        out = m._run_brief_for_company(ctx, company="字节")
        assert out.startswith("OK:")
        assert "confidence=0.72" in out
        assert "app_limit=2" in out


# ═══════════════════════════════════════════════════════════════════
# P1 #3 — agent.loop must record result.skill_version, not spec.version
# ═══════════════════════════════════════════════════════════════════


class TestSkillInvocationsUseEffectiveVersion:
    def test_loop_records_actual_run_version(self):
        """When a canary/live variant is selected at invoke time, the
        skill_invocations dict (used for evolution-signal attribution)
        must capture that version, not the seed spec.version."""
        # Source-level guard: the structure that builds skill_invocations
        # must reference result.skill_version, NOT spec.version.
        loop_path = Path(__file__).parent.parent / "src/offerguide/agent/loop.py"
        src = loop_path.read_text()
        # Pin down the relevant section by string search.
        i = src.find("if skill_invocations is not None:")
        assert i != -1, "skill_invocations construction not found"
        block = src[i: i + 400]
        assert '"skill_version": result.skill_version' in block, (
            "agent loop must use result.skill_version (the version that ACTUALLY "
            "ran), not spec.version (always seed). Otherwise canary/live runs "
            "are mis-attributed."
        )
        assert '"skill_version": spec.version' not in block, (
            "Old seed-version attribution snuck back in"
        )


# ═══════════════════════════════════════════════════════════════════
# P1 #4 — goals.py used py3.12 f-string nesting; pyproject targets py3.11
# ═══════════════════════════════════════════════════════════════════


class TestGoalsPy311Compat:
    def test_module_parses_under_py311_grammar(self):
        """ast.parse with feature_version=(3, 11) confirms no py3.12+ syntax
        slipped in. The original `f"...{', julianday(\"now\") + ?'}..."` would
        SyntaxError under py3.11 (PEP 701 only applies from 3.12)."""
        goals_path = Path(__file__).parent.parent / "src/offerguide/goals.py"
        src = goals_path.read_text()
        # ast.parse(feature_version=(3, 11)) raises SyntaxError on 3.12-only forms
        ast.parse(src, filename=str(goals_path), feature_version=(3, 11))

    def test_write_self_observation_with_valid_for_days_none(self, tmp_path):
        """The branch without valid_until still inserts cleanly."""
        store = offerguide.Store(tmp_path / "x.db")
        store.init_schema()
        oid = _goals.write_self_observation(
            store,
            observation="agent 总爱在 23:30 后跑长 task",
            pattern_kind="repeated_mistake",
            valid_for_days=None,
        )
        assert oid > 0

    def test_write_self_observation_with_valid_for_days_set(self, tmp_path):
        """The branch with valid_until also inserts cleanly."""
        store = offerguide.Store(tmp_path / "x.db")
        store.init_schema()
        oid = _goals.write_self_observation(
            store,
            observation="本周已发了字节 brief, 7 天内别再调",
            pattern_kind="success_pattern",
            valid_for_days=7,
        )
        assert oid > 0
        # And the row really has a valid_until populated
        with store.connect() as conn:
            row = conn.execute(
                "SELECT valid_until FROM agent_self_observations WHERE id = ?",
                (oid,),
            ).fetchone()
        assert row is not None
        assert row[0] is not None  # julianday returned a real value


# ═══════════════════════════════════════════════════════════════════
# P2 #5 — _hash_invocation must encode the version actually run
# ═══════════════════════════════════════════════════════════════════


class TestHashInvocationEncodesEffectiveVersion:
    def test_same_inputs_diff_versions_yield_diff_hashes(self):
        """Seed and canary running on identical inputs must hash differently —
        otherwise the two runs collide in skill_runs.input_hash and we lose
        the ability to attribute / dedupe them per variant."""
        h_seed = _hash_invocation(
            skill_name="score_match", version="0.1.0",
            inputs={"job_text": "ai agent intern", "user_resume": "..."},
        )
        h_canary = _hash_invocation(
            skill_name="score_match", version="0.1.0+canary42",
            inputs={"job_text": "ai agent intern", "user_resume": "..."},
        )
        assert h_seed != h_canary

    def test_same_inputs_same_version_yield_same_hash(self):
        """Same prompt → same hash (the dedupe contract still holds within
        a single variant)."""
        a = _hash_invocation(
            skill_name="score_match", version="0.1.0",
            inputs={"job_text": "x", "user_resume": "y"},
        )
        b = _hash_invocation(
            skill_name="score_match", version="0.1.0",
            inputs={"job_text": "x", "user_resume": "y"},
        )
        assert a == b

    def test_callsite_passes_effective_version(self):
        """The single caller in invoke() must pass effective_version, not
        spec.version, otherwise the variant routing above is moot."""
        rt_path = Path(__file__).parent.parent / "src/offerguide/skills/_runtime.py"
        src = rt_path.read_text()
        # Find the call inside invoke()
        i = src.find("input_hash = _hash_invocation(")
        assert i != -1, "_hash_invocation call not found"
        call_block = src[i: i + 200]
        assert "version=effective_version" in call_block, (
            "callsite must hash with effective_version (the version actually "
            "selected by the variant registry), not spec.version (always seed)"
        )


# ═══════════════════════════════════════════════════════════════════
# P2 #6 — home_wake_agent must offload via asyncio.to_thread
# ═══════════════════════════════════════════════════════════════════


class TestHomeWakeAgentNonBlocking:
    def test_handler_offloads_blocking_agent_run(self):
        """In single-worker uvicorn (the default for local dev), a blocking
        sync call inside an async handler freezes the event loop for the full
        agent run (10-30s). The handler must offload via asyncio.to_thread,
        same as /api/agent/stream already does."""
        from offerguide.ui import web as web_mod
        src = inspect.getsource(web_mod)
        # Find the home_wake_agent handler
        i = src.find("async def home_wake_agent(")
        assert i != -1
        # Look at next ~2000 chars (covers handler body even with comments)
        body = src[i: i + 2000]
        # Must use to_thread — bare agent.run(...) would block event loop
        assert "asyncio.to_thread" in body, (
            "home_wake_agent must offload agent.run via asyncio.to_thread "
            "(handler is `async def`; sync agent.run blocks event loop ~10-30s)"
        )
        # And it must be awaited (not just scheduled)
        assert "await asyncio.to_thread" in body, (
            "the to_thread coro must be awaited so the route returns the actual run result"
        )
