"""W15 harness tests — memory tool, context, loop, tools, triggers, feedback, UI.

Single file (vs split into per-module) because the surfaces are tightly coupled
and the integration shape is what matters most. ~700 lines, organized by class.

Stub LLM: programmable response queue for chat_with_tools so we can simulate
multi-turn agent behavior without hitting the real API. Memory tool, context
management, triggers — all tested directly (no LLM needed).
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import offerguide
from offerguide.config import Settings
from offerguide.harness import (
    HarnessDeps,
    MemoryStore,
    SystemFacts,
    TriggerEvent,
    fire_event,
    make_cron_heartbeat,
    make_user_input_trigger,
    poll_pending,
)
from offerguide.harness import _schema as harness_schema
from offerguide.harness import (
    feedback as harness_feedback,
)
from offerguide.harness.context import (
    CLEAR_TOOL_RESULTS_TRIGGER_TOKENS,
    COMPACTION_TRIGGER_TOKENS,
    ContextManager,
)
from offerguide.harness.loop import RunResult
from offerguide.harness.loop import run as harness_run
from offerguide.harness.memory import MEMORY_TOOL_SCHEMA
from offerguide.harness.tools import ALL_TOOL_SCHEMAS, dispatch
from offerguide.llm import LLMResponse, ToolCall

# ═══════════════════════════════════════════════════════════════════
# Test infrastructure: stub LLM with programmable responses
# ═══════════════════════════════════════════════════════════════════


class StubLLM:
    """Programmable stub. Push LLMResponses onto `.queue` in order; each
    chat_with_tools call pops the next one."""

    def __init__(self) -> None:
        self.queue: list[LLMResponse] = []
        self.calls: list[dict[str, Any]] = []

    def chat_with_tools(self, messages, *, tools, **kw):
        self.calls.append({"messages": list(messages), "tools": list(tools), **kw})
        if not self.queue:
            # Default: end_turn with no tool calls
            return LLMResponse(content="(stub: queue empty)", model="stub")
        return self.queue.pop(0)

    def chat(self, messages, **kw):
        self.calls.append({"messages": list(messages), "kind": "chat", **kw})
        return LLMResponse(content="(stub chat)", model="stub")

    def push_text(self, text: str) -> None:
        self.queue.append(LLMResponse(
            content=text, model="stub",
            prompt_tokens=100, completion_tokens=20,
        ))

    def push_tool_call(self, *, name: str, arguments: dict[str, Any], call_id: str = "tc_0") -> None:
        self.queue.append(LLMResponse(
            content="", model="stub",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
            prompt_tokens=100, completion_tokens=10,
        ))


@pytest.fixture
def tmp_worldview() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def tmp_store(tmp_path) -> offerguide.Store:
    s = offerguide.Store(tmp_path / "harness_test.db")
    s.init_schema()
    harness_schema.init_harness_schema(s)
    return s


@pytest.fixture
def deps(tmp_store, tmp_worldview):
    settings = Settings(deepseek_api_key="x", default_model="stub")
    llm = StubLLM()
    return HarnessDeps(
        settings=settings,
        store=tmp_store,
        memory_store=MemoryStore(root=tmp_worldview),
        llm=llm,  # type: ignore[arg-type]  # stub
    )


# ═══════════════════════════════════════════════════════════════════
# Memory tool — 6 commands + path safety
# ═══════════════════════════════════════════════════════════════════


class TestMemoryTool:
    def test_bootstrap_creates_six_starter_files(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        files = set(m.list_files())
        assert "MEMORY.md" in files
        assert "candidate.md" in files
        assert "tracked-jobs.md" in files
        assert "upcoming-events.md" in files
        assert "reflections.md" in files
        assert "strategy.md" in files

    def test_view_returns_numbered_lines(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        result = m.execute({"command": "view", "path": "MEMORY.md"})
        assert result.startswith("OK MEMORY.md")
        assert "    1\t" in result  # line numbering format

    def test_view_with_view_range(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "x.md",
                   "file_text": "a\nb\nc\nd\ne\n"})
        result = m.execute({"command": "view", "path": "x.md",
                            "view_range": [2, 4]})
        assert "OK x.md (lines 2-4)" in result
        assert "    2\tb" in result and "    4\td" in result
        assert "    1\ta" not in result

    def test_view_dir_lists_files(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        result = m.execute({"command": "view", "path": "/"})
        assert "OK files:" in result
        assert "MEMORY.md" in result

    def test_create_writes_and_overwrites(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        r1 = m.execute({"command": "create", "path": "new.md",
                        "file_text": "hello"})
        assert "OK created new.md" in r1
        # Overwrite
        r2 = m.execute({"command": "create", "path": "new.md",
                        "file_text": "world"})
        assert "OK created" in r2
        check = m.execute({"command": "view", "path": "new.md"})
        assert "world" in check

    def test_str_replace_unique_match_required(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "x.md",
                   "file_text": "foo bar foo"})
        # Non-unique → ERROR
        r = m.execute({"command": "str_replace", "path": "x.md",
                       "old_str": "foo", "new_str": "BAR"})
        assert "ERROR" in r
        assert "matches 2 times" in r
        # Unique with context → OK
        r2 = m.execute({"command": "str_replace", "path": "x.md",
                        "old_str": "foo bar", "new_str": "BAR baz"})
        assert "OK replaced" in r2

    def test_str_replace_missing_file(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        r = m.execute({"command": "str_replace", "path": "ghost.md",
                       "old_str": "x", "new_str": "y"})
        assert r.startswith("ERROR:")

    def test_insert_at_line(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "x.md",
                   "file_text": "line1\nline3\n"})
        r = m.execute({"command": "insert", "path": "x.md",
                       "insert_line": 1, "insert_text": "line2"})
        assert "OK inserted" in r
        check = m.execute({"command": "view", "path": "x.md"})
        # Order: line1, line2, line3
        assert "line1" in check and "line2" in check and "line3" in check

    def test_insert_invalid_line(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "x.md",
                   "file_text": "only one line"})
        r = m.execute({"command": "insert", "path": "x.md",
                       "insert_line": 99, "insert_text": "off the end"})
        assert "ERROR" in r

    def test_delete(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "tmp.md", "file_text": "x"})
        r = m.execute({"command": "delete", "path": "tmp.md"})
        assert "OK deleted" in r
        assert "tmp.md" not in m.list_files()

    def test_rename(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "old.md", "file_text": "x"})
        r = m.execute({"command": "rename", "path": "old.md",
                       "new_path": "renamed/new.md"})
        assert "OK renamed" in r
        assert "renamed/new.md" in m.list_files()
        assert "old.md" not in m.list_files()

    def test_rename_refuses_overwrite(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "a.md", "file_text": "1"})
        m.execute({"command": "create", "path": "b.md", "file_text": "2"})
        r = m.execute({"command": "rename", "path": "a.md", "new_path": "b.md"})
        assert "ERROR" in r and "already exists" in r

    def test_path_traversal_blocked(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        for bad in ["../escape", "/etc/passwd", "../../../tmp/x"]:
            r = m.execute({"command": "view", "path": bad})
            # All forms should error or stay scoped
            # /etc/passwd → leading / stripped, becomes etc/passwd which doesn't exist
            assert "ERROR" in r

    def test_unknown_command(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        r = m.execute({"command": "wat"})
        assert r.startswith("ERROR:") and "wat" in r

    def test_auto_load_text_truncates_at_max_lines(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        # MEMORY.md from bootstrap is small; write a big one
        big = "\n".join(f"line {i}" for i in range(500))
        m.execute({"command": "create", "path": "MEMORY.md", "file_text": big})
        text = m.auto_load_text(max_lines=200)
        assert "line 0" in text
        assert "line 199" in text
        assert "line 200" not in text  # truncated
        assert "truncated" in text  # marker present

    def test_atomic_write_no_partial(self, tmp_worldview):
        """Atomic write means no .tmp file lingers after success."""
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "atom.md", "file_text": "hello"})
        # No leftover .tmp files
        leftover = [p for p in tmp_worldview.iterdir() if p.suffix == ".tmp"]
        assert leftover == []


# ═══════════════════════════════════════════════════════════════════
# Context manager — system message build, clearing, compaction
# ═══════════════════════════════════════════════════════════════════


class TestContextManager:
    def test_build_initial_system_includes_instructions_and_memory(self, deps):
        cm = ContextManager(llm=deps.llm, memory=deps.memory_store)
        sys_msg = cm.build_initial_system()
        # instructions.md content
        assert "求职 agent" in sys_msg
        # System facts
        assert "今天" in sys_msg
        # Worldview MEMORY.md auto-loaded (bootstrap'd)
        assert "你的主页" in sys_msg or "你脑子里的当前状态" in sys_msg

    def test_build_initial_system_with_empty_worldview(self, tmp_path, tmp_store):
        empty_dir = tmp_path / "empty_worldview"
        # Don't bootstrap — pass empty
        empty_dir.mkdir()
        # Note: MemoryStore __post_init__ will bootstrap, so we delete after
        m = MemoryStore(root=empty_dir)
        # Now wipe
        for f in list(m.root.iterdir()):
            f.unlink()
        cm = ContextManager(llm=StubLLM(), memory=m)  # type: ignore[arg-type]
        sys_msg = cm.build_initial_system()
        assert "worldview 是空的" in sys_msg or "第一次 wake" in sys_msg

    def test_system_facts_includes_calendar_phase(self):
        # Pin a date in May (暑期投递高峰末期)
        facts = SystemFacts(today=_dt.date(2026, 5, 6))
        rendered = facts.render()
        assert "2026-05-06" in rendered
        assert "校招阶段" in rendered

    def test_clearing_skipped_when_under_threshold(self, deps):
        cm = ContextManager(llm=deps.llm, memory=deps.memory_store)
        # Tiny conversation
        msgs = [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "y"},
            {"role": "tool", "tool_call_id": "t1", "content": "z"},
        ]
        new_msgs, cleared = cm.maybe_clear_tool_results(msgs)
        assert cleared == 0
        assert new_msgs == msgs

    def test_clearing_keeps_recent_tool_results(self, deps):
        cm = ContextManager(llm=deps.llm, memory=deps.memory_store)
        # Force trigger by faking last_prompt_tokens above threshold
        cm.last_prompt_tokens = CLEAR_TOOL_RESULTS_TRIGGER_TOKENS + 5_000
        # Build many tool result messages
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(8):
            msgs.append({"role": "assistant", "content": f"call {i}"})
            msgs.append({
                "role": "tool", "tool_call_id": f"t{i}",
                "content": "x" * 500,
            })
        new_msgs, cleared = cm.maybe_clear_tool_results(msgs)
        # Should clear all but last 4
        assert cleared >= 1
        # Most recent tool result content preserved (not "cleared" placeholder)
        last_tool = [m for m in new_msgs if m.get("role") == "tool"][-1]
        assert "cleared by harness" not in last_tool["content"]
        # Earliest tool result replaced with placeholder
        first_tool = next(m for m in new_msgs if m.get("role") == "tool")
        assert "cleared by harness" in first_tool["content"]

    def test_compaction_triggers_when_over_threshold(self, deps):
        cm = ContextManager(llm=deps.llm, memory=deps.memory_store)
        # Push a stub summary response
        deps.llm.push_text("(summary) user did X, decided Y")  # type: ignore[union-attr]
        cm.last_prompt_tokens = COMPACTION_TRIGGER_TOKENS + 5_000
        msgs = [
            {"role": "system", "content": "sys"},
            *[{"role": "user", "content": f"msg{i}"} for i in range(20)],
        ]
        new_msgs, compacted = cm.maybe_compact(msgs)
        assert compacted is True
        # New shape: [system, summary, last K]
        assert len(new_msgs) < len(msgs)
        assert new_msgs[0]["role"] == "system"
        # Summary is a user message (we frame it as "前情提要")
        assert "前情提要" in new_msgs[1]["content"]


# ═══════════════════════════════════════════════════════════════════
# Tool dispatch (without going through full loop)
# ═══════════════════════════════════════════════════════════════════


class TestToolDispatch:
    def test_unknown_tool_returns_error(self, deps):
        r = dispatch("nope", {}, deps)
        assert r.startswith("ERROR: unknown tool")

    def test_memory_dispatch_routes_to_memory_store(self, deps):
        r = dispatch("memory", {"command": "view", "path": "MEMORY.md"}, deps)
        assert r.startswith("OK MEMORY.md")

    def test_schedule_next_wake_inserts_row(self, deps):
        r = dispatch("schedule_next_wake",
                     {"delay_seconds": 3600, "reason": "test"}, deps)
        assert r.startswith("OK scheduled_wake#")
        with deps.store.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM harness_scheduled_wakes"
            ).fetchone()[0]
        assert n == 1

    def test_schedule_next_wake_validates_bounds(self, deps):
        # Too small
        r1 = dispatch("schedule_next_wake",
                      {"delay_seconds": 30, "reason": "x"}, deps)
        assert "ERROR" in r1 and "≥ 60" in r1
        # Too big
        r2 = dispatch("schedule_next_wake",
                      {"delay_seconds": 99 * 86400, "reason": "x"}, deps)
        assert "ERROR" in r2 and "30 days" in r2

    def test_record_event_inserts_row(self, deps):
        r = dispatch("record_event",
                     {"kind": "applied", "job_id": 42, "note": "投了"}, deps)
        assert r.startswith("OK event#")
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT kind, job_id, note FROM harness_events "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "applied"
        assert row[1] == 42

    def test_ask_user_writes_inbox_question(self, deps):
        r = dispatch("ask_user", {
            "question": "你想投北京吗?",
            "context": "你 cv 没说地点偏好",
            "options": [
                {"id": "yes", "label": "可以"},
                {"id": "no", "label": "不要"},
            ],
        }, deps)
        assert r.startswith("OK ask_user")
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT kind, title FROM inbox_items "
                "WHERE kind = 'question' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "question"

    def test_ask_user_requires_two_options(self, deps):
        r = dispatch("ask_user", {
            "question": "x", "context": "y", "options": [{"id": "1", "label": "a"}],
        }, deps)
        assert "ERROR" in r and "≥2" in r

    def test_notify_user_writes_inbox(self, deps):
        r = dispatch("notify_user", {
            "title": "找到一个好岗位",
            "body": "字节 NLP 实习, 高匹配",
            "priority": "important",
        }, deps)
        assert r.startswith("OK notify_user")

    def test_web_search_without_search_backend_errors(self, deps):
        # deps fixture doesn't set search → should ERROR
        r = dispatch("web_search", {"query": "test"}, deps)
        assert "ERROR" in r and "SearchBackend" in r

    def test_score_match_without_runtime_errors(self, deps):
        r = dispatch("score_match", {"job_id": 1}, deps)
        assert "ERROR" in r and "SkillRuntime" in r

    def test_all_tool_schemas_have_required_fields(self):
        # Sanity: each schema is well-formed OpenAI function spec
        for s in ALL_TOOL_SCHEMAS:
            assert s["type"] == "function"
            fn = s["function"]
            assert isinstance(fn["name"], str) and fn["name"]
            assert isinstance(fn["description"], str)
            assert fn["parameters"]["type"] == "object"

    def test_memory_tool_schema_first(self):
        # By design, memory comes first (agent's brain is primary)
        assert ALL_TOOL_SCHEMAS[0]["function"]["name"] == "memory"
        assert ALL_TOOL_SCHEMAS[0] is MEMORY_TOOL_SCHEMA


# ═══════════════════════════════════════════════════════════════════
# Master loop — single-threaded orchestration
# ═══════════════════════════════════════════════════════════════════


class TestHarnessLoop:
    def test_loop_records_run_in_harness_runs(self, deps):
        # Stub LLM returns end_turn immediately
        deps.llm.push_text("nothing to do, sleeping")  # type: ignore[union-attr]
        result = harness_run(
            trigger=make_cron_heartbeat(), deps=deps,
            max_iterations=3,
        )
        assert isinstance(result, RunResult)
        assert result.run_id is not None
        assert result.iterations == 1
        assert result.finish_reason == "end_turn"
        # Persisted
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT trigger_kind, status, iterations FROM harness_runs "
                "WHERE id = ?", (result.run_id,),
            ).fetchone()
        assert row[0] == "cron"
        assert row[1] == "ok"
        assert row[2] == 1

    def test_loop_executes_tool_call_then_terminates(self, deps):
        # Two-step: call schedule_next_wake, then end_turn
        deps.llm.push_tool_call(  # type: ignore[union-attr]
            name="schedule_next_wake",
            arguments={"delay_seconds": 600, "reason": "test"},
        )
        deps.llm.push_text("done")  # type: ignore[union-attr]
        result = harness_run(
            trigger=make_user_input_trigger("找点事做"), deps=deps,
            max_iterations=5,
        )
        assert result.iterations == 2
        assert result.finish_reason == "end_turn"
        assert "schedule_next_wake" in result.tool_call_log[0]
        # Side-effect: scheduled_wake row created
        with deps.store.connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM harness_scheduled_wakes"
            ).fetchone()[0]
        assert n == 1

    def test_loop_respects_max_iterations(self, deps):
        # Push 10 tool calls; max=3 caps it
        for i in range(10):
            deps.llm.push_tool_call(  # type: ignore[union-attr]
                name="memory",
                arguments={"command": "view", "path": "MEMORY.md"},
                call_id=f"t{i}",
            )
        result = harness_run(
            trigger=make_cron_heartbeat(), deps=deps,
            max_iterations=3,
        )
        assert result.iterations == 3
        assert result.finish_reason == "max_iterations"

    def test_loop_handles_no_llm(self, tmp_store, tmp_worldview):
        deps_no_llm = HarnessDeps(
            settings=Settings(deepseek_api_key="", default_model="stub"),
            store=tmp_store,
            memory_store=MemoryStore(root=tmp_worldview),
            llm=None,
        )
        result = harness_run(
            trigger=make_cron_heartbeat(), deps=deps_no_llm,
        )
        assert result.finish_reason == "no_llm"
        assert result.run_id is None

    def test_loop_passes_current_run_id_to_tools(self, deps):
        # Tool call → notify_user → check inbox row references the run
        deps.llm.push_tool_call(  # type: ignore[union-attr]
            name="notify_user",
            arguments={"title": "t", "body": "b"},
        )
        deps.llm.push_text("done")  # type: ignore[union-attr]
        result = harness_run(
            trigger=make_cron_heartbeat(), deps=deps,
            max_iterations=3,
        )
        # Check the inbox row got source_agent_run_id set to result.run_id
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT source_agent_run_id FROM inbox_items "
                "WHERE kind = 'agent_suggestion' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == result.run_id


# ═══════════════════════════════════════════════════════════════════
# Triggers — fire_event, poll_pending, scheduled wakes
# ═══════════════════════════════════════════════════════════════════


class TestTriggers:
    def test_fire_event_inserts_row(self, tmp_store):
        fire_event(tmp_store, event_kind="user_paste_jd",
                   detail={"job_id": 5, "note": "from bo zhi"})
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT kind, job_id, note, source FROM harness_events "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "user_paste_jd"
        assert row[1] == 5
        assert row[3] == "user"

    def test_poll_pending_picks_up_due_scheduled_wake(self, tmp_store):
        # Insert a scheduled wake with fire_at in the past
        with tmp_store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_scheduled_wakes(fire_at, reason) "
                "VALUES (julianday('now') - 0.001, 'test')"
            )
        pending = poll_pending(tmp_store)
        assert len(pending) >= 1
        assert pending[0].source == "scheduled"
        assert pending[0].trigger_event.kind == "scheduled"
        # Cleanup callback marks fired_at
        pending[0].cleanup()
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT fired_at FROM harness_scheduled_wakes "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] is not None

    def test_poll_pending_skips_future_scheduled(self, tmp_store):
        with tmp_store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_scheduled_wakes(fire_at, reason) "
                "VALUES (julianday('now') + 1, 'far future')"
            )
        pending = poll_pending(tmp_store)
        assert all(p.trigger_event.detail.get("reason") != "far future"
                   for p in pending)

    def test_poll_pending_picks_up_unprocessed_events(self, tmp_store):
        fire_event(tmp_store, event_kind="user_marked_applied",
                   detail={"job_id": 7})
        pending = poll_pending(tmp_store)
        events = [p for p in pending if p.source == "event"]
        assert len(events) >= 1
        assert events[0].trigger_event.detail["event"] == "user_marked_applied"

    def test_make_cron_heartbeat_returns_cron_kind(self):
        t = make_cron_heartbeat()
        assert isinstance(t, TriggerEvent)
        assert t.kind == "cron"
        assert "timestamp" in t.detail

    def test_make_user_input_truncates_long(self):
        t = make_user_input_trigger("x" * 5000)
        assert len(t.detail["message"]) <= 2000


# ═══════════════════════════════════════════════════════════════════
# Feedback — bridge to GEPA evolution_signals
# ═══════════════════════════════════════════════════════════════════


class TestFeedback:
    def test_accept_writes_positive_thumb(self, tmp_store):
        sig_id = harness_feedback.on_inbox_accepted(
            tmp_store, inbox_id=1, skill_run_id=None,
            user_text="不错",
        )
        assert sig_id > 0
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value, skill_name, notes "
                "FROM evolution_signals WHERE id = ?", (sig_id,),
            ).fetchone()
        assert row[0] == "user_thumbs"
        assert row[1] == 1.0
        # No skill_run_id → sentinel
        assert row[2] == "harness"
        notes = _json.loads(row[3])
        assert notes["related_inbox_id"] == 1
        assert notes["user_text"] == "不错"

    def test_reject_writes_negative_thumb(self, tmp_store):
        sig_id = harness_feedback.on_inbox_rejected(
            tmp_store, inbox_id=2, skill_run_id=None, user_text="不要",
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value FROM evolution_signals "
                "WHERE id = ?", (sig_id,),
            ).fetchone()
        assert row[0] == "user_thumbs"
        assert row[1] == -1.0

    def test_question_answered_records_option_id(self, tmp_store):
        sig_id = harness_feedback.on_question_answered(
            tmp_store, inbox_id=99, option_id="yes_beijing",
            free_text=None,
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, notes FROM evolution_signals "
                "WHERE id = ?", (sig_id,),
            ).fetchone()
        assert row[0] == "user_question_answer"
        notes = _json.loads(row[1])
        assert notes["metadata"]["option_id"] == "yes_beijing"

    def test_ignored_signal_has_lower_weight(self, tmp_store):
        sig_id = harness_feedback.on_inbox_ignored(
            tmp_store, inbox_id=5, days_ignored=10,
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value, signal_weight, notes "
                "FROM evolution_signals WHERE id = ?", (sig_id,),
            ).fetchone()
        assert row[0] == "user_ignored"
        assert row[1] < 0  # negative
        assert row[2] < 1.0  # weight reduced vs explicit thumbs
        notes = _json.loads(row[3])
        assert notes["metadata"]["days_ignored"] == 10

    def test_skill_meta_resolved_when_skill_run_exists(self, tmp_store):
        # Create a fake skill_run row
        with tmp_store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO skill_runs("
                "  skill_name, skill_version, input_hash, input_json, "
                "  output_json, latency_ms, cost_usd"
                ") VALUES ('score_match', 'v3', 'h', '{}', '{}', 0, 0) "
                "RETURNING id"
            )
            sr_id = int(cur.fetchone()[0])
        sig_id = harness_feedback.on_inbox_accepted(
            tmp_store, inbox_id=10, skill_run_id=sr_id,
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT skill_name, skill_version FROM evolution_signals "
                "WHERE id = ?", (sig_id,),
            ).fetchone()
        assert row[0] == "score_match"
        assert row[1] == "v3"


# ═══════════════════════════════════════════════════════════════════
# Web UI — home renders worldview + chat form, /debug renders
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def web_client(tmp_path):
    from offerguide.skills import discover_skills
    from offerguide.ui.notify import ConsoleNotifier
    from offerguide.ui.web import create_app

    store = offerguide.Store(tmp_path / "ui.db")
    store.init_schema()
    skills = discover_skills(Path(__file__).parent.parent / "src/offerguide/skills")
    s = Settings(deepseek_api_key="", default_model="stub")  # no LLM ok for read tests
    app = create_app(
        settings=s, store=store, profile=None, skills=skills,
        runtime=None, notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestHomeWithW15:
    def test_home_renders_worldview_section(self, web_client):
        client, _ = web_client
        # Hitting / will trigger MemoryStore bootstrap on real worldview dir;
        # the rendered template surfaces the 心智窗口 card.
        resp = client.get("/")
        assert resp.status_code == 200
        # New W15 sections present
        assert "🧠" in resp.text  # 心智窗口
        assert "💬" in resp.text  # chat input
        assert "submitAgentChat" in resp.text  # JS handler

    def test_home_chat_endpoint_requires_llm(self, web_client):
        client, _ = web_client
        resp = client.post("/api/home/chat", json={"message": "hi"})
        assert resp.status_code == 400
        assert "LLM" in resp.text

    def test_home_chat_endpoint_validates_message(self, tmp_path):
        # Need an LLM key for this validation path to be reached
        from offerguide.skills import discover_skills
        from offerguide.ui.notify import ConsoleNotifier
        from offerguide.ui.web import create_app

        store = offerguide.Store(tmp_path / "ui2.db")
        store.init_schema()
        s = Settings(deepseek_api_key="x", default_model="stub")
        skills = discover_skills(Path(__file__).parent.parent / "src/offerguide/skills")
        app = create_app(
            settings=s, store=store, profile=None, skills=skills,
            runtime=None, notifier=ConsoleNotifier(),
        )
        client = TestClient(app)
        r1 = client.post("/api/home/chat", json={"message": ""})
        assert r1.status_code == 400


class TestDebugView:
    def test_debug_renders_all_sections(self, web_client):
        client, _ = web_client
        resp = client.get("/debug")
        assert resp.status_code == 200
        assert "Harness Runs" in resp.text
        assert "Scheduled Wakes" in resp.text or "scheduled" in resp.text.lower()
        assert "Harness Events" in resp.text
        assert "Daemon Runs" in resp.text

    def test_debug_handles_empty_state(self, web_client):
        client, _ = web_client
        resp = client.get("/debug")
        assert resp.status_code == 200
        # All 4 sections show empty messages, not crash
        # (just verify status 200 + key headings present, done above)


# ═══════════════════════════════════════════════════════════════════
# W15.11 — inbox decide / answer wiring → evolution_signals
# ═══════════════════════════════════════════════════════════════════


class TestInboxFeedbackWiring:
    def test_decide_approved_records_positive_thumb(self, web_client):
        """User clicks 'approve' on agent suggestion → +1 thumb signal."""
        client, store = web_client
        from offerguide import inbox as inbox_mod
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="试试这家公司", body="字节 NLP 实习",
            source_agent_run_id=None,
        )
        resp = client.post(
            f"/inbox/{item.id}/decide",
            data={"decision": "approved"},
        )
        assert resp.status_code == 200
        with store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value, notes "
                "FROM evolution_signals ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row is not None
        assert row[0] == "user_thumbs"
        assert row[1] == 1.0
        notes = _json.loads(row[2])
        assert notes["related_inbox_id"] == item.id

    def test_decide_rejected_records_negative_thumb(self, web_client):
        client, store = web_client
        from offerguide import inbox as inbox_mod
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="不喜欢的公司", body="x",
        )
        client.post(f"/inbox/{item.id}/decide", data={"decision": "rejected"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value FROM evolution_signals "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "user_thumbs"
        assert row[1] == -1.0

    def test_decide_dismissed_records_ignored(self, web_client):
        client, store = web_client
        from offerguide import inbox as inbox_mod
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="无所谓", body="x",
        )
        client.post(f"/inbox/{item.id}/decide", data={"decision": "dismissed"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_weight FROM evolution_signals "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "user_ignored"
        # Weight should be reduced (vs explicit thumbs)
        assert row[1] < 1.0

    def test_decide_links_to_skill_run_when_payload_has_it(self, web_client):
        """With source_skill_run_id in payload, signal links to specific SKILL run."""
        client, store = web_client
        # Make a real skill_run row first
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO skill_runs("
                "  skill_name, skill_version, input_hash, input_json, "
                "  output_json, latency_ms, cost_usd"
                ") VALUES ('score_match', 'v5', 'h', '{}', '{}', 0, 0) "
                "RETURNING id"
            )
            sr_id = int(cur.fetchone()[0])

        from offerguide import inbox as inbox_mod
        item = inbox_mod.enqueue_agent_suggestion(
            store, title="字节高匹配", body="x",
            source_skill_run_id=sr_id,
        )
        client.post(f"/inbox/{item.id}/decide", data={"decision": "approved"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT skill_name, skill_version, skill_run_id "
                "FROM evolution_signals ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "score_match"
        assert row[1] == "v5"
        assert row[2] == sr_id

    def test_answer_question_records_signal(self, web_client):
        """User answers an agent's question → user_question_answer signal."""
        client, store = web_client
        from offerguide import inbox as inbox_mod
        item = inbox_mod.enqueue_question(
            store,
            question="你想投北京吗?",
            context="cv 没说地点偏好",
            options=[
                {"id": "yes", "label": "可以"},
                {"id": "no", "label": "不要"},
            ],
        )
        resp = client.post(
            f"/inbox/{item.id}/answer",
            data={"option_id": "no"},
            follow_redirects=False,
        )
        assert resp.status_code == 303  # redirect to /
        with store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, notes FROM evolution_signals "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "user_question_answer"
        notes = _json.loads(row[1])
        assert notes["metadata"]["option_id"] == "no"
        assert notes["related_inbox_id"] == item.id

    def test_decide_unknown_status_400(self, web_client):
        client, _ = web_client
        resp = client.post("/inbox/1/decide", data={"decision": "wat"})
        assert resp.status_code == 400

    def test_decide_404_doesnt_record_signal(self, web_client):
        client, store = web_client
        # Count signals before
        with store.connect() as conn:
            n_before = conn.execute(
                "SELECT COUNT(*) FROM evolution_signals"
            ).fetchone()[0]
        # Decide on nonexistent item
        resp = client.post("/inbox/99999/decide", data={"decision": "approved"})
        assert resp.status_code == 404
        with store.connect() as conn:
            n_after = conn.execute(
                "SELECT COUNT(*) FROM evolution_signals"
            ).fetchone()[0]
        # No signal recorded for failed decide
        assert n_after == n_before
