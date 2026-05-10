"""W14.9 — quality pass: 6 real bugs from self-review.

Each bug has its own class. Tests pin the new behavior down so a regression
shows up immediately, without relying on the bug's original symptom (which
was usually "rare in single-user dev").
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import offerguide
from offerguide import goals as _goals  # noqa: F401  (kept for parity with other w14 tests)
from offerguide import inbox as inbox_mod
from offerguide.agent import maintenance as maintenance_mod
from offerguide.agent.loop import AgentLoop
from offerguide.config import Settings
from offerguide.llm import LLMResponse
from offerguide.llm.client import ToolCall
from offerguide.skills import SkillRuntime, discover_skills

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


@pytest.fixture
def store(tmp_path):
    s = offerguide.Store(tmp_path / "w14_9.db")
    s.init_schema()
    return s


# ═══════════════════════════════════════════════════════════════════
# #2 — inbox.decide() must be atomic; concurrent calls write only one signal
# ═══════════════════════════════════════════════════════════════════


class TestDecideRace:
    def test_concurrent_decide_only_one_thumbs_signal(self, store):
        """Two threads call decide() on the same pending item simultaneously.
        One should win; the other should raise ValueError. Crucially,
        evolution_signals should contain exactly ONE user_thumbs row — not
        two — because the bug doubled fitness from a double-clicked button."""
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="x", body="y",
            source_skill_name="score_match", source_skill_version="0.1.0",
        )

        results: list[Any] = []
        barrier = threading.Barrier(2)

        def _decide():
            barrier.wait()  # both threads start the SQL transaction together
            try:
                inbox_mod.decide(store, item.id, decision="approved")
                results.append("ok")
            except ValueError as e:
                results.append(("already_decided", str(e)))

        t1 = threading.Thread(target=_decide)
        t2 = threading.Thread(target=_decide)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Exactly one winner, one loser
        assert results.count("ok") == 1
        assert sum(1 for r in results if isinstance(r, tuple)) == 1

        # And the user_thumbs fan-out happened EXACTLY once
        with store.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM evolution_signals "
                "WHERE skill_name='score_match' AND signal_kind='user_thumbs'"
            ).fetchone()[0]
        assert n == 1, f"expected 1 user_thumbs signal, got {n}"

    def test_decide_missing_item_raises_keyerror(self, store):
        """Distinct from already-decided: caller can tell the diff."""
        with pytest.raises(KeyError):
            inbox_mod.decide(store, 9999, decision="approved")

    def test_decide_already_decided_raises_valueerror(self, store):
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="t", body="b",
            source_skill_name="score_match", source_skill_version="0.1.0",
        )
        inbox_mod.decide(store, item.id, decision="approved")
        with pytest.raises(ValueError):
            inbox_mod.decide(store, item.id, decision="rejected")


# ═══════════════════════════════════════════════════════════════════
# #6 — _row_to_item reads by column name, not position
# ═══════════════════════════════════════════════════════════════════


class TestRowDictAccess:
    def test_get_returns_correct_fields_by_name(self, store):
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="T", body="B",
            source_agent_run_id=42,
            source_skill_name="score_match",
            source_skill_version="v1",
        )
        fetched = inbox_mod.get(store, item.id)
        assert fetched is not None
        assert fetched.title == "T"
        assert fetched.body == "B"
        assert fetched.source_agent_run_id == 42
        assert fetched.source_skill_name == "score_match"
        assert fetched.source_skill_version == "v1"
        assert fetched.kind == "agent_suggestion"
        assert fetched.status == "pending"

    def test_list_items_returns_typed_inbox_items(self, store):
        for i in range(3):
            inbox_mod.enqueue_agent_suggestion(
                store, title=f"sug {i}", body="x",
                source_skill_name="score_match", source_skill_version="0.1.0",
            )
        rows = inbox_mod.list_items(store)
        assert len(rows) == 3
        for r in rows:
            assert r.kind == "agent_suggestion"
            assert r.source_skill_name == "score_match"


# ═══════════════════════════════════════════════════════════════════
# #5 — write_suggestion explicit source_skill_name beats heuristic
# ═══════════════════════════════════════════════════════════════════


class TestWriteSuggestionAttribution:
    def _make_loop(self, store):
        skills = discover_skills(SKILLS_ROOT)
        s = Settings(deepseek_api_key="x", default_model="stub")

        class _StubLLM:
            def chat(self, messages, **kw):
                return LLMResponse(content="{}", model="stub")
            def chat_with_tools(self, messages, **kw):
                return LLMResponse(content="", model="stub")

        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        return AgentLoop(
            llm=_StubLLM(), runtime=runtime, store=store, skills=skills,
            master_resume_text="x",
        ), runtime

    def test_explicit_source_skill_name_used(self, store):
        loop, _ = self._make_loop(store)
        # Pretend two SKILLs ran in this trajectory
        loop._current_run_id = 1
        loop._current_skill_invocations = {
            "tc1": {"skill_name": "score_match", "skill_version": "0.1.0", "skill_run_id": 100},
            "tc2": {"skill_name": "tailor_resume", "skill_version": "0.2.0", "skill_run_id": 101},
        }
        # Model writes a suggestion that's about score_match (the EARLIER call)
        tc = ToolCall(
            id="tcW", name="write_suggestion",
            arguments={
                "title": "score 偏低, 建议跳过",
                "body": "...",
                "source_skill_name": "score_match",
            },
        )
        out = loop._execute_write_suggestion(tc)
        assert "OK:" in out
        # The inbox item should be attributed to score_match (NOT tailor_resume)
        items = inbox_mod.list_items(store)
        assert len(items) == 1
        assert items[0].source_skill_name == "score_match"
        assert items[0].source_skill_version == "0.1.0"  # version from the actual run

    def test_falls_back_to_heuristic_when_not_provided(self, store):
        loop, _ = self._make_loop(store)
        loop._current_run_id = 1
        loop._current_skill_invocations = {
            "tc1": {"skill_name": "score_match", "skill_version": "0.1.0", "skill_run_id": 100},
            "tc2": {"skill_name": "tailor_resume", "skill_version": "0.2.0", "skill_run_id": 101},
        }
        tc = ToolCall(
            id="tcW", name="write_suggestion",
            arguments={"title": "T", "body": "B"},
        )
        loop._execute_write_suggestion(tc)
        items = inbox_mod.list_items(store)
        # Falls back to most-recent (tailor_resume) when model didn't say
        assert items[0].source_skill_name == "tailor_resume"

    def test_unknown_skill_name_honored_with_unknown_version(self, store):
        """Model names a SKILL that wasn't in the trajectory — treat the name
        as truth (the model knows what the suggestion is about) but mark
        version unknown so fitness doesn't bucket under a fake version."""
        loop, _ = self._make_loop(store)
        loop._current_run_id = 1
        loop._current_skill_invocations = {
            "tc1": {"skill_name": "score_match", "skill_version": "0.1.0", "skill_run_id": 100},
        }
        tc = ToolCall(
            id="tcW", name="write_suggestion",
            arguments={
                "title": "T", "body": "B",
                "source_skill_name": "compare_jobs",  # not in trajectory
            },
        )
        loop._execute_write_suggestion(tc)
        items = inbox_mod.list_items(store)
        assert items[0].source_skill_name == "compare_jobs"
        assert items[0].source_skill_version == "?"


# ═══════════════════════════════════════════════════════════════════
# #1 — Agent loop responds to cancel_event at iteration boundary
# ═══════════════════════════════════════════════════════════════════


class TestCooperativeCancellation:
    def test_run_aborts_when_cancel_event_pre_set(self, store):
        """Easy case: event is set before run starts. Loop should never call
        the LLM at all and persist agent_runs.status='cancelled'."""
        skills = discover_skills(SKILLS_ROOT)
        call_count = 0

        class _CountingLLM:
            def chat(self, *a, **kw):
                return LLMResponse(content="{}", model="stub")
            def chat_with_tools(self, *a, **kw):
                nonlocal call_count
                call_count += 1
                return LLMResponse(content="never reached", model="stub")
            def close(self): pass

        runtime = SkillRuntime(llm=_CountingLLM(), store=store)
        loop = AgentLoop(
            llm=_CountingLLM(), runtime=runtime, store=store, skills=skills,
            master_resume_text="x", critic_enabled=False,
        )

        ev = threading.Event()
        ev.set()  # already cancelled
        result = loop.run(goal="test", cancel_event=ev)

        assert call_count == 0, "LLM should never be called when pre-cancelled"
        # agent_runs row should be persisted with status='cancelled'
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM agent_runs WHERE id = ?", (result.run_id,),
            ).fetchone()
        assert row is not None and row[0] == "cancelled"
        # _cancelled event surfaced in trajectory
        assert any(e.kind == "_cancelled" for e in result.events)

    def test_run_completes_normally_when_cancel_event_unset(self, store):
        """Without a cancel set, run should proceed as usual."""
        skills = discover_skills(SKILLS_ROOT)

        class _OneShotLLM:
            def chat(self, *a, **kw):
                return LLMResponse(content="{}", model="stub")
            def chat_with_tools(self, *a, **kw):
                return LLMResponse(content="done", model="stub", finish_reason="stop")
            def close(self): pass

        runtime = SkillRuntime(llm=_OneShotLLM(), store=store)
        loop = AgentLoop(
            llm=_OneShotLLM(), runtime=runtime, store=store, skills=skills,
            master_resume_text="x", critic_enabled=False,
        )
        ev = threading.Event()  # not set
        result = loop.run(goal="test", cancel_event=ev)
        assert result.final_answer == "done"
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM agent_runs WHERE id = ?", (result.run_id,),
            ).fetchone()
        assert row[0] == "ok"


# ═══════════════════════════════════════════════════════════════════
# #4 — SSE queue is bounded; oldest events drop, _done is preserved
# ═══════════════════════════════════════════════════════════════════


class TestSSEQueueBounded:
    """Source-level guard: the SSE handler in web.py creates a bounded queue
    and the handler text contains the protective patterns. A pure unit test
    on the queue logic without spinning up the whole FastAPI handler is
    tricky; we lock the contract in source so a refactor that drops the
    bound is caught."""

    def test_sse_queue_constructed_with_maxsize(self):
        import inspect

        from offerguide.ui import web as web_mod
        src = inspect.getsource(web_mod)
        # The SSE handler must create a Queue with explicit maxsize
        assert "asyncio.Queue(maxsize=" in src, (
            "SSE queue must be bounded — unbounded queue can grow without "
            "limit if the client is slow / disconnected"
        )

    def test_sse_done_sentinel_protected_from_drop(self):
        import inspect

        from offerguide.ui import web as web_mod
        src = inspect.getsource(web_mod)
        # When the queue is full, _done events get put back so the user
        # doesn't lose the run summary
        assert '"_done"' in src and "put_nowait(old)" in src, (
            "drop-oldest policy must spare the _done sentinel — losing it "
            "leaves the SSE generator hanging waiting for a sentinel that "
            "will never arrive"
        )


# ═══════════════════════════════════════════════════════════════════
# #3 — maintenance forwards kwargs (no env mutation), daemon respects them
# ═══════════════════════════════════════════════════════════════════


class TestMaintenanceKwargForwarding:
    def test_jd_enrich_kwarg_reaches_daemon_no_env_mutation(self, store, monkeypatch):
        """The agent's max_jobs arg must flow into jd_enrich.run as `limit`,
        AND os.environ must not be mutated (the old code clobbered globals)."""
        captured: dict[str, Any] = {}
        before_env = os.environ.get("OFFERGUIDE_JD_ENRICH_MAX")

        def _fake_run(ctx, *, limit=None):
            captured["limit"] = limit
            captured["env_at_call_time"] = os.environ.get("OFFERGUIDE_JD_ENRICH_MAX")
            return {"scanned": 0, "ok": 0}

        # Patch the late import target
        from offerguide.autonomous.jobs import jd_enrich
        monkeypatch.setattr(jd_enrich, "run", _fake_run)
        # And the maintenance import shortcut
        monkeypatch.setattr(maintenance_mod, "_import_job", lambda name: jd_enrich.run)

        skills = discover_skills(SKILLS_ROOT)

        class _StubLLM:
            def chat(self, *a, **kw): return LLMResponse(content="{}", model="stub")

        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        ctx = maintenance_mod.MaintenanceCtx(
            store=store, llm=_StubLLM(), runtime=runtime, skills=skills,
        )
        out = maintenance_mod.execute_maintenance_tool(
            "enrich_thin_jds", {"max_jobs": 5}, ctx,
        )
        assert out.startswith("OK:")
        assert captured.get("limit") == 5, "max_jobs must flow into limit kwarg"
        # The old (buggy) implementation set this env var; the new one must not.
        assert captured.get("env_at_call_time") == before_env, (
            "OFFERGUIDE_JD_ENRICH_MAX env var must not be mutated — the old "
            "implementation set it process-globally, breaking concurrent runs"
        )
        # And after the call, env stays clean
        assert os.environ.get("OFFERGUIDE_JD_ENRICH_MAX") == before_env

    def test_extract_facts_kwarg_reaches_daemon(self, store, monkeypatch):
        captured: dict[str, Any] = {}

        def _fake_run(ctx, *, limit=None):
            captured["limit"] = limit
            return {"runs_scanned": 0, "inserted": 0}

        from offerguide.autonomous.jobs import extract_facts
        monkeypatch.setattr(extract_facts, "run", _fake_run)
        monkeypatch.setattr(maintenance_mod, "_import_job", lambda name: extract_facts.run)

        skills = discover_skills(SKILLS_ROOT)

        class _StubLLM:
            def chat(self, *a, **kw): return LLMResponse(content="{}", model="stub")

        runtime = SkillRuntime(llm=_StubLLM(), store=store)
        ctx = maintenance_mod.MaintenanceCtx(
            store=store, llm=_StubLLM(), runtime=runtime, skills=skills,
        )
        out = maintenance_mod.execute_maintenance_tool(
            "extract_facts_from_runs", {"max_runs": 7}, ctx,
        )
        assert out.startswith("OK:")
        assert captured.get("limit") == 7

    def test_daemon_default_when_no_kwarg(self, monkeypatch, store):
        """The cron entry calls run(ctx) with no kwargs and should still
        get the module-level MAX_PER_RUN default (backward compat)."""
        from offerguide.autonomous.jobs import jd_enrich

        captured: dict[str, Any] = {}

        def _fake_enrich_pending(store, llm, *, limit):
            captured["limit"] = limit
            return {"scanned": 0, "ok": 0, "js_rendered": 0,
                    "fetch_failed": 0, "extracted_thin": 0}

        monkeypatch.setattr(jd_enrich, "enrich_pending", _fake_enrich_pending)

        @dataclass
        class _Ctx:
            store: Any = None
            llm: Any = object()
            notifier: Any = None

        jd_enrich.run(_Ctx(store=store))  # no `limit=` kwarg
        assert captured.get("limit") == jd_enrich.MAX_PER_RUN
