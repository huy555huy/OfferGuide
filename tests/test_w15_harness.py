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
from offerguide.agent_runtime import (
    AgentRuntimeDeps,
    MemoryStore,
    SystemFacts,
    TriggerEvent,
    make_user_input_trigger,
)
from offerguide.agent_runtime import _schema as harness_schema
from offerguide.agent_runtime import (
    feedback as harness_feedback,
)
from offerguide.agent_runtime.context import (
    CLEAR_TOOL_RESULTS_TRIGGER_TOKENS,
    COMPACTION_TRIGGER_TOKENS,
    ContextManager,
)
from offerguide.agent_runtime.loop import RunResult
from offerguide.agent_runtime.loop import run as agent_runtime_run
from offerguide.agent_runtime.memory import MEMORY_TOOL_SCHEMA
from offerguide.agent_runtime.tools import ALL_TOOL_SCHEMAS, dispatch
from offerguide.config import Settings
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
        self.queue.append(
            LLMResponse(
                content=text,
                model="stub",
                prompt_tokens=100,
                completion_tokens=20,
            )
        )

    def push_tool_call(
        self, *, name: str, arguments: dict[str, Any], call_id: str = "tc_0"
    ) -> None:
        self.queue.append(
            LLMResponse(
                content="",
                model="stub",
                tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
                prompt_tokens=100,
                completion_tokens=10,
            )
        )


@pytest.fixture
def tmp_worldview() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def tmp_store(tmp_path) -> offerguide.Store:
    s = offerguide.Store(tmp_path / "harness_test.db")
    s.init_schema()
    harness_schema.init_agent_runtime_schema(s)
    return s


@pytest.fixture
def deps(tmp_store, tmp_worldview):
    settings = Settings(deepseek_api_key="x", default_model="stub")
    llm = StubLLM()
    return AgentRuntimeDeps(
        settings=settings,
        store=tmp_store,
        memory_store=MemoryStore(root=tmp_worldview),
        llm=llm,  # type: ignore[arg-type]  # stub
    )


# ═══════════════════════════════════════════════════════════════════
# Memory tool — 6 commands + path safety
# ═══════════════════════════════════════════════════════════════════


class TestMemoryTool:
    def test_bootstrap_creates_starter_files(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        files = set(m.list_files())
        assert files == {
            "MEMORY.md",
            "candidate.md",
            "tracked-jobs.md",
            "upcoming-events.md",
        }

    def test_view_returns_numbered_lines(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        result = m.execute({"command": "view", "path": "MEMORY.md"})
        assert result.startswith("OK MEMORY.md")
        assert "    1\t" in result  # line numbering format

    def test_view_with_view_range(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "x.md", "file_text": "a\nb\nc\nd\ne\n"})
        result = m.execute({"command": "view", "path": "x.md", "view_range": [2, 4]})
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
        r1 = m.execute({"command": "create", "path": "new.md", "file_text": "hello"})
        assert "OK created new.md" in r1
        # Overwrite
        r2 = m.execute({"command": "create", "path": "new.md", "file_text": "world"})
        assert "OK created" in r2
        check = m.execute({"command": "view", "path": "new.md"})
        assert "world" in check

    def test_str_replace_unique_match_required(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "x.md", "file_text": "foo bar foo"})
        # Non-unique → ERROR
        r = m.execute(
            {"command": "str_replace", "path": "x.md", "old_str": "foo", "new_str": "BAR"}
        )
        assert "ERROR" in r
        assert "matches 2 times" in r
        # Unique with context → OK
        r2 = m.execute(
            {"command": "str_replace", "path": "x.md", "old_str": "foo bar", "new_str": "BAR baz"}
        )
        assert "OK replaced" in r2

    def test_str_replace_missing_file(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        r = m.execute(
            {"command": "str_replace", "path": "ghost.md", "old_str": "x", "new_str": "y"}
        )
        assert r.startswith("ERROR:")

    def test_insert_at_line(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "x.md", "file_text": "line1\nline3\n"})
        r = m.execute(
            {"command": "insert", "path": "x.md", "insert_line": 1, "insert_text": "line2"}
        )
        assert "OK inserted" in r
        check = m.execute({"command": "view", "path": "x.md"})
        # Order: line1, line2, line3
        assert "line1" in check and "line2" in check and "line3" in check

    def test_insert_invalid_line(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        m.execute({"command": "create", "path": "x.md", "file_text": "only one line"})
        r = m.execute(
            {"command": "insert", "path": "x.md", "insert_line": 99, "insert_text": "off the end"}
        )
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
        r = m.execute({"command": "rename", "path": "old.md", "new_path": "renamed/new.md"})
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
        cm = ContextManager(
            llm=deps.llm,
            memory=deps.memory_store,
            store=deps.store,
        )
        sys_msg = cm.build_initial_system()
        # instructions.md content
        assert "OfferGuide 求职 Agent" in sys_msg
        # System facts
        assert "今天" in sys_msg
        assert "三个核心任务" in sys_msg
        assert "Skill 自进化" in sys_msg
        # Worldview MEMORY.md auto-loaded (bootstrap'd)
        assert "已确认的长期求职上下文" in sys_msg

    def test_build_initial_system_includes_persistent_agenda(self, deps):
        deps.memory_store.execute(
            {
                "command": "create",
                "path": "agenda.md",
                "file_text": (
                    "# Agent Agenda\n\n"
                    "## 开放回路\n"
                    "- Open loop: follow up ByteDance by 2026-05-23\n\n"
                    "## 安静等待\n"
                    "- Do not notify about low-confidence roles.\n"
                ),
            }
        )
        cm = ContextManager(
            llm=deps.llm,
            memory=deps.memory_store,
            store=deps.store,
        )
        sys_msg = cm.build_initial_system()

        assert "已有开放事项（仅在仍与当前求职阶段相关时使用）" in sys_msg
        assert "Open loop: follow up ByteDance" in sys_msg
        assert "Do not notify about low-confidence roles" in sys_msg

    def test_build_initial_system_includes_active_goal_progress(self, deps):
        from offerguide import goals as _goals

        _goals.add_goal(
            deps.store,
            title="拿到 1 个 AI Agent 暑期实习 offer",
            target_date=_dt.date(2026, 7, 15),
            target_metric="1 offer",
            description="优先 AI Agent / LLM 应用方向",
        )
        cm = ContextManager(
            llm=deps.llm,
            memory=deps.memory_store,
            store=deps.store,
        )
        sys_msg = cm.build_initial_system(system_facts=SystemFacts(today=_dt.date(2026, 5, 16)))

        assert "拿到 1 个 AI Agent 暑期实习 offer" in sys_msg
        assert "target_date: 2026-07-15" in sys_msg
        assert "进行中申请=" in sys_msg
        assert "heuristic:" not in sys_msg

    def test_build_initial_system_excludes_completed_work_items(self, deps):
        open_id = harness_schema.ensure_work_item(
            deps.store,
            title="继续处理开放工作",
            source_kind="agent",
            source_ref="ctx-open",
        )
        done_id = harness_schema.ensure_work_item(
            deps.store,
            title="已经完成的工作",
            source_kind="agent",
            source_ref="ctx-done",
        )
        harness_schema.update_work_item(
            deps.store,
            work_item_id=done_id,
            status="done",
            summary="已完成",
        )

        cm = ContextManager(
            llm=deps.llm,
            memory=deps.memory_store,
            store=deps.store,
        )
        sys_msg = cm.build_initial_system()

        assert f"WorkItem #{open_id}" in sys_msg
        assert "继续处理开放工作" in sys_msg
        assert f"WorkItem #{done_id}" not in sys_msg
        assert "已经完成的工作" not in sys_msg

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
        assert "当前没有可用的长期记忆" in sys_msg

    def test_system_facts_do_not_invent_a_recruiting_phase(self):
        facts = SystemFacts(today=_dt.date(2026, 5, 6))
        rendered = facts.render()
        assert "2026-05-06" in rendered
        assert "校招阶段" not in rendered

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
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"t{i}",
                    "content": "x" * 500,
                }
            )
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

    def test_record_event_rejects_unbound_legacy_lifecycle_claim(self, deps):
        r = dispatch("record_event", {"kind": "applied", "job_id": 42, "note": "投了"}, deps)
        assert r == "ERROR: record_event requires integer application_id"
        with deps.store.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM harness_events").fetchone()[0]
        assert count == 0

    def test_update_work_item_marks_done_and_removes_from_active(self, deps):
        item_id = harness_schema.ensure_work_item(
            deps.store,
            title="评估用户粘贴的 JD",
            source_kind="user_input",
            source_ref="wi-done-test",
            summary="用户想知道 JD 值不值得投",
            priority=90,
        )

        r = dispatch(
            "update_work_item",
            {
                "work_item_id": item_id,
                "status": "done",
                "summary": "已完成 JD 判断并生成可读结论",
                "evidence": {
                    "job_id": 7,
                    "skill_run_id": 11,
                    "view": "/jobs/7/apply-pack",
                },
            },
            deps,
        )

        assert r.startswith(f"OK work_item#{item_id} done")
        active_ids = [item.id for item in harness_schema.list_active_work_items(deps.store)]
        assert item_id not in active_ids
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT status, closed_at, summary, evidence_json "
                "FROM agent_work_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        assert row[0] == "done"
        assert row[1] is not None
        assert "JD 判断" in row[2]
        assert _json.loads(row[3])["skill_run_id"] == 11

    def test_update_work_item_waiting_requires_next_action(self, deps):
        item_id = harness_schema.ensure_work_item(
            deps.store,
            title="等待投递反馈",
            source_kind="event",
            source_ref="wi-wait-test",
        )

        missing = dispatch(
            "update_work_item",
            {
                "work_item_id": item_id,
                "status": "waiting",
                "summary": "已安排后续检查",
            },
            deps,
        )
        assert "ERROR" in missing and "requires next_action" in missing

        ok = dispatch(
            "update_work_item",
            {
                "work_item_id": item_id,
                "status": "waiting",
                "summary": "已投递, 当前未记录新回复",
                "next_action": "明天面试前检查项目追问是否已准备",
                "due_seconds": 3600,
                "evidence": {"application_id": 3},
            },
            deps,
        )
        assert ok.startswith(f"OK work_item#{item_id} waiting")
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT status, closed_at, due_at, next_action FROM agent_work_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        assert row[0] == "waiting"
        assert row[1] is None
        assert row[2] is not None
        assert "明天面试前" in row[3]

    def test_ask_user_writes_inbox_question(self, deps):
        r = dispatch(
            "ask_user",
            {
                "question": "你想投北京吗?",
                "context": "你 cv 没说地点偏好",
                "options": [
                    {"id": "yes", "label": "可以"},
                    {"id": "no", "label": "不要"},
                ],
            },
            deps,
        )
        assert r.startswith("OK ask_user")
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT kind, title FROM inbox_items "
                "WHERE kind = 'question' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "question"

    def test_ask_user_requires_two_options(self, deps):
        r = dispatch(
            "ask_user",
            {
                "question": "x",
                "context": "y",
                "options": [{"id": "1", "label": "a"}],
            },
            deps,
        )
        assert "ERROR" in r and "≥2" in r

    def test_notify_user_writes_inbox(self, deps):
        r = dispatch(
            "notify_user",
            {
                "title": "找到一个好岗位",
                "body": "字节 NLP 实习, 高匹配",
                "priority": "important",
            },
            deps,
        )
        assert r.startswith("OK notify_user")

    def test_web_search_without_search_backend_errors(self, deps):
        # deps fixture doesn't set search → should ERROR
        r = dispatch("web_search", {"query": "test"}, deps)
        assert "ERROR" in r and "SearchBackend" in r

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

    def test_update_work_item_schema_is_available(self):
        names = [s["function"]["name"] for s in ALL_TOOL_SCHEMAS]
        assert "update_work_item" in names
        schema = next(
            s["function"] for s in ALL_TOOL_SCHEMAS if s["function"]["name"] == "update_work_item"
        )
        assert "durable work item" in schema["description"]
        assert "work_item_id" in schema["parameters"]["required"]


# ═══════════════════════════════════════════════════════════════════
# Master loop — single-threaded orchestration
# ═══════════════════════════════════════════════════════════════════


class TestHarnessLoop:
    def test_loop_records_run_in_harness_runs(self, deps):
        # Stub LLM returns end_turn immediately
        deps.llm.push_text("nothing to do, sleeping")  # type: ignore[union-attr]
        result = agent_runtime_run(
            trigger=TriggerEvent(kind="test"),
            deps=deps,
            max_iterations=3,
        )
        assert isinstance(result, RunResult)
        assert result.run_id is not None
        assert result.iterations == 1
        assert result.finish_reason == "end_turn"
        # Persisted
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT trigger_kind, status, iterations FROM harness_runs WHERE id = ?",
                (result.run_id,),
            ).fetchone()
        assert row[0] == "test"
        assert row[1] == "ok"
        assert row[2] == 1

    def test_loop_executes_tool_call_then_terminates(self, deps):
        # Two-step: reject an unbound pseudo-event, then end the turn.
        deps.llm.push_tool_call(  # type: ignore[union-attr]
            name="record_event",
            arguments={"kind": "note", "note": "test"},
            call_id="event_1",
        )
        deps.llm.push_text("done")  # type: ignore[union-attr]
        result = agent_runtime_run(
            trigger=make_user_input_trigger("找点事做"),
            deps=deps,
            max_iterations=5,
        )
        assert result.iterations == 2
        assert result.finish_reason == "end_turn"
        assert "record_event" in result.tool_call_log[0]
        with deps.store.connect() as conn:
            event_count = conn.execute("SELECT COUNT(*) FROM harness_events").fetchone()[0]
            work = conn.execute(
                "SELECT title, status, last_run_id FROM agent_work_items "
                "WHERE source_kind = 'user_input' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert event_count == 0
        assert work[0].startswith("User request:")
        assert work[1] == "in_progress"
        assert work[2] == result.run_id

    def test_loop_allows_agent_to_close_trigger_work_item(self, deps):
        message = "帮我看看这个 JD 值不值得投"
        item_id = harness_schema.prepare_trigger_work_items(
            deps.store,
            kind="user_input",
            detail={"message": message},
        )[0]
        deps.llm.push_tool_call(  # type: ignore[union-attr]
            name="update_work_item",
            arguments={
                "work_item_id": item_id,
                "status": "done",
                "summary": "已判断: 当前证据不足, 已告诉用户需要完整 JD",
                "evidence": {"user_message": message},
            },
            call_id="work_done_1",
        )
        deps.llm.push_text("需要完整 JD 才能判断。")  # type: ignore[union-attr]

        result = agent_runtime_run(
            trigger=make_user_input_trigger(message),
            deps=deps,
            max_iterations=5,
        )

        assert result.finish_reason == "end_turn"
        assert "update_work_item" in result.tool_call_log[0]
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT status, closed_at, last_run_id, summary FROM agent_work_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        assert row[0] == "done"
        assert row[1] is not None
        assert row[2] == result.run_id
        assert "证据不足" in row[3]

    def test_loop_respects_max_iterations(self, deps):
        # Push 10 tool calls; max=3 caps it
        for i in range(10):
            deps.llm.push_tool_call(  # type: ignore[union-attr]
                name="memory",
                arguments={"command": "view", "path": "MEMORY.md"},
                call_id=f"t{i}",
            )
        result = agent_runtime_run(
            trigger=TriggerEvent(kind="test"),
            deps=deps,
            max_iterations=3,
        )
        assert result.iterations == 3
        assert result.finish_reason == "max_iterations"

    def test_loop_handles_no_llm(self, tmp_store, tmp_worldview):
        deps_no_llm = AgentRuntimeDeps(
            settings=Settings(deepseek_api_key="", default_model="stub"),
            store=tmp_store,
            memory_store=MemoryStore(root=tmp_worldview),
            llm=None,
        )
        result = agent_runtime_run(
            trigger=TriggerEvent(kind="test"),
            deps=deps_no_llm,
        )
        assert result.finish_reason == "no_llm"
        assert result.run_id is None

    def test_loop_passes_current_run_id_to_tools(self, deps):
        # Tool call → notify_user → check inbox row references the run
        deps.llm.push_tool_call(  # type: ignore[union-attr]
            name="notify_user",
            arguments={"title": "t", "body": "b"},
            call_id="notify_1",
        )
        deps.llm.push_text("done")  # type: ignore[union-attr]
        result = agent_runtime_run(
            trigger=TriggerEvent(kind="test"),
            deps=deps,
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
# Synchronous user-input trigger
# ═══════════════════════════════════════════════════════════════════


class TestUserInputTrigger:
    def test_user_input_trigger_creates_work_item_for_run(self, deps):
        deps.llm.push_text("done")  # type: ignore[union-attr]
        result = agent_runtime_run(
            trigger=make_user_input_trigger("帮我看看这个 JD 值不值得投"),
            deps=deps,
        )

        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT title, source_kind, status, last_run_id, summary "
                "FROM agent_work_items ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0].startswith("User request:")
        assert row[1] == "user_input"
        assert row[2] == "in_progress"
        assert row[3] == result.run_id
        assert "JD" in row[4]

    def test_make_user_input_truncates_long(self):
        t = make_user_input_trigger("x" * 5000)
        assert len(t.detail["message"]) <= 2000


# ═══════════════════════════════════════════════════════════════════
# Feedback — bridge to GEPA evolution_signals
# ═══════════════════════════════════════════════════════════════════


class TestFeedback:
    def test_accept_writes_positive_thumb(self, tmp_store):
        sig_id = harness_feedback.on_inbox_accepted(
            tmp_store,
            inbox_id=1,
            skill_run_id=None,
            user_text="不错",
        )
        assert sig_id > 0
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value, skill_name, notes "
                "FROM evolution_signals WHERE id = ?",
                (sig_id,),
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
            tmp_store,
            inbox_id=2,
            skill_run_id=None,
            user_text="不要",
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value FROM evolution_signals WHERE id = ?",
                (sig_id,),
            ).fetchone()
        assert row[0] == "user_thumbs"
        assert row[1] == -1.0

    def test_question_answered_records_option_id(self, tmp_store):
        sig_id = harness_feedback.on_question_answered(
            tmp_store,
            inbox_id=99,
            option_id="yes_beijing",
            free_text=None,
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, notes FROM evolution_signals WHERE id = ?",
                (sig_id,),
            ).fetchone()
        assert row[0] == "user_question_answer"
        notes = _json.loads(row[1])
        assert notes["metadata"]["option_id"] == "yes_beijing"

    def test_ignored_signal_has_lower_weight(self, tmp_store):
        sig_id = harness_feedback.on_inbox_ignored(
            tmp_store,
            inbox_id=5,
            days_ignored=10,
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value, signal_weight, notes "
                "FROM evolution_signals WHERE id = ?",
                (sig_id,),
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
                ") VALUES ('example_skill', 'v3', 'h', '{}', '{}', 0, 0) "
                "RETURNING id"
            )
            sr_id = int(cur.fetchone()[0])
        sig_id = harness_feedback.on_inbox_accepted(
            tmp_store,
            inbox_id=10,
            skill_run_id=sr_id,
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT skill_name, skill_version FROM evolution_signals WHERE id = ?",
                (sig_id,),
            ).fetchone()
        assert row[0] == "example_skill"
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
        settings=s,
        store=store,
        master_source=None,
        skills=skills,
        runtime=None,
        notifier=ConsoleNotifier(),
    )
    return TestClient(app), store


class TestHomeWithW15:
    def test_home_renders_mission_control_shell(self, web_client):
        client, _ = web_client
        resp = client.get("/")
        assert resp.status_code == 200
        assert "OfferGuide · 指挥台" in resp.text
        assert "Mission Control" in resp.text
        assert "需决定" in resp.text
        assert 'href="/pipeline"' in resp.text
        assert 'href="/project-vault"' in resp.text
        assert 'href="/evolution"' not in resp.text
        assert "和 Agent 对话" in resp.text
        assert "发送给 Agent" in resp.text
        assert "trigger_kind=user_input" in resp.text
        assert "trigger_kind=user_button" not in resp.text

    def test_home_surfaces_recent_agent_artifacts(self, web_client):
        client, store = web_client
        harness_schema.init_agent_runtime_schema(store)
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_events(kind, note, source) "
                "VALUES ('project_record_saved', ?, 'agent')",
                (
                    _json.dumps(
                        {
                            "project_id": 3,
                            "title": "Deep Research Workspace",
                            "direction": "AI Agent",
                            "view": "/project-vault",
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        resp = client.get("/")
        assert resp.status_code == 200
        assert "最近产物" in resp.text
        assert "Deep Research Workspace" in resp.text
        assert "/project-vault" in resp.text

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
            settings=s,
            store=store,
            master_source=None,
            skills=skills,
            runtime=None,
            notifier=ConsoleNotifier(),
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
        assert "Harness Events" in resp.text

    def test_debug_handles_empty_state(self, web_client):
        client, _ = web_client
        resp = client.get("/debug")
        assert resp.status_code == 200
        # Empty runtime state renders without crashing.


# ═══════════════════════════════════════════════════════════════════
# W15.11 — inbox decide / answer wiring → evolution_signals
# ═══════════════════════════════════════════════════════════════════


class TestInboxFeedbackWiring:
    def test_decide_approved_records_positive_thumb(self, web_client):
        """User clicks 'approve' on agent suggestion → +1 thumb signal."""
        client, store = web_client
        from offerguide import inbox as inbox_mod

        item = inbox_mod.enqueue_agent_suggestion(
            store,
            title="试试这家公司",
            body="字节 NLP 实习",
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
            store,
            title="不喜欢的公司",
            body="x",
        )
        client.post(f"/inbox/{item.id}/decide", data={"decision": "rejected"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_value FROM evolution_signals ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "user_thumbs"
        assert row[1] == -1.0

    def test_decide_dismissed_records_ignored(self, web_client):
        client, store = web_client
        from offerguide import inbox as inbox_mod

        item = inbox_mod.enqueue_agent_suggestion(
            store,
            title="无所谓",
            body="x",
        )
        client.post(f"/inbox/{item.id}/decide", data={"decision": "dismissed"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT signal_kind, signal_weight FROM evolution_signals ORDER BY id DESC LIMIT 1"
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
                ") VALUES ('example_skill', 'v5', 'h', '{}', '{}', 0, 0) "
                "RETURNING id"
            )
            sr_id = int(cur.fetchone()[0])

        from offerguide import inbox as inbox_mod

        item = inbox_mod.enqueue_agent_suggestion(
            store,
            title="字节高匹配",
            body="x",
            source_skill_run_id=sr_id,
        )
        client.post(f"/inbox/{item.id}/decide", data={"decision": "approved"})
        with store.connect() as conn:
            row = conn.execute(
                "SELECT skill_name, skill_version, skill_run_id "
                "FROM evolution_signals ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "example_skill"
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
                "SELECT signal_kind, notes FROM evolution_signals ORDER BY id DESC LIMIT 1"
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
            n_before = conn.execute("SELECT COUNT(*) FROM evolution_signals").fetchone()[0]
        # Decide on nonexistent item
        resp = client.post("/inbox/99999/decide", data={"decision": "approved"})
        assert resp.status_code == 404
        with store.connect() as conn:
            n_after = conn.execute("SELECT COUNT(*) FROM evolution_signals").fetchone()[0]
        # No signal recorded for failed decide
        assert n_after == n_before


# ═══════════════════════════════════════════════════════════════════
# W15.12 — review fixes (Bug 1-7 + Smell 1, 6, 8)
# ═══════════════════════════════════════════════════════════════════


class TestReviewFixes:
    # ── Bug 1: max_iterations → status='truncated' (not 'ok')
    def test_max_iter_persists_truncated_status(self, deps):
        # Push enough tool_calls that loop runs 2 iter then caps
        for i in range(5):
            deps.llm.push_tool_call(  # type: ignore[union-attr]
                name="memory",
                arguments={"command": "view", "path": "MEMORY.md"},
                call_id=f"t{i}",
            )
        result = agent_runtime_run(
            trigger=TriggerEvent(kind="test"),
            deps=deps,
            max_iterations=2,
        )
        assert result.finish_reason == "max_iterations"
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM harness_runs WHERE id = ?",
                (result.run_id,),
            ).fetchone()
        assert row[0] == "truncated"  # NOT 'ok'

    def test_end_turn_persists_ok_status(self, deps):
        deps.llm.push_text("nothing to do")  # type: ignore[union-attr]
        result = agent_runtime_run(trigger=TriggerEvent(kind="test"), deps=deps)
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM harness_runs WHERE id = ?",
                (result.run_id,),
            ).fetchone()
        assert row[0] == "ok"

    # ── Bug 2: final_text accumulates content from tool-calling iterations
    def test_final_text_captures_reasoning_alongside_tool_calls(self, deps):
        # Iter 1: model returns BOTH content + tool_call (some models do this)
        deps.llm.queue.append(
            LLMResponse(  # type: ignore[union-attr]
                content="我先看一眼 worldview.",
                model="stub",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="memory",
                        arguments={"command": "view", "path": "MEMORY.md"},
                    )
                ],
                prompt_tokens=100,
                completion_tokens=20,
            )
        )
        # Iter 2: end_turn with final reasoning
        deps.llm.push_text("看完了, 不做事了.")  # type: ignore[union-attr]
        result = agent_runtime_run(trigger=TriggerEvent(kind="test"), deps=deps)
        # Both iterations' content present
        assert "我先看一眼" in result.final_text
        assert "看完了" in result.final_text

    # ── Bug 3: try/finally ensures harness_runs never stuck in 'running'
    def test_loop_crash_in_ctx_management_records_error(
        self,
        tmp_store,
        tmp_worldview,
        monkeypatch,
    ):
        from offerguide.agent_runtime import context as ctx_mod

        def _exploding_clear(*a, **kw):
            raise RuntimeError("simulated ctx crash")

        monkeypatch.setattr(
            ctx_mod.ContextManager,
            "maybe_clear_tool_results",
            _exploding_clear,
        )
        deps = AgentRuntimeDeps(
            settings=Settings(deepseek_api_key="x", default_model="stub"),
            store=tmp_store,
            memory_store=MemoryStore(root=tmp_worldview),
            llm=StubLLM(),  # type: ignore[arg-type]
        )
        result = agent_runtime_run(trigger=TriggerEvent(kind="test"), deps=deps)
        assert result.finish_reason == "loop_crash"
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT status, error_text FROM harness_runs WHERE id = ?",
                (result.run_id,),
            ).fetchone()
        # Status MUST be 'error', not 'running'
        assert row[0] == "error"
        assert "simulated ctx crash" in (row[1] or "")

    # ── Bug 4: paste:// URL stable across processes
    def test_paste_synthetic_url_is_stable(self, deps):
        from offerguide.agent_runtime.tools import _exec_fetch_jd

        text = "x" * 250  # ≥ 200 chars to pass validation

        # Same content twice → same URL → second is detected as dup
        r1 = _exec_fetch_jd({"url_or_text": text}, deps)
        r2 = _exec_fetch_jd({"url_or_text": text}, deps)

        assert "OK ingested" in r1
        # Second call updates the same canonical manual-job record.
        assert "updated existing job#" in r2

    # ── Bug 5: supporting tool cost flows into harness_runs.cost_usd
    def test_extra_cost_usd_reset_each_run(self, deps):
        # Pre-set extra_cost_usd to non-zero (simulating leak from prior run)
        deps.extra_cost_usd = 99.99
        deps.llm.push_text("done")  # type: ignore[union-attr]
        result = agent_runtime_run(trigger=TriggerEvent(kind="test"), deps=deps)
        # The 99.99 must NOT contaminate this run — loop resets to 0
        # at start, so cost_usd here is just stub LLM's $0.0 + 0
        assert result.cost_usd < 1.0  # nowhere near 99.99

    def test_sub_agent_cost_added_to_total(self, deps):
        # Manually set extra_cost_usd as if a sub-agent ran (simulates
        # what _exec_supporting tool does)
        deps.llm.push_tool_call(  # type: ignore[union-attr]
            name="memory",
            arguments={"command": "view", "path": "MEMORY.md"},
            call_id="t1",
        )
        # Hook: bump extra_cost_usd via a side effect on next call
        original_chat = deps.llm.chat_with_tools  # type: ignore[union-attr]

        def _patched(messages, **kw):
            r = original_chat(messages, **kw)
            deps.extra_cost_usd += 0.5  # simulate sub-agent cost
            return r

        deps.llm.chat_with_tools = _patched  # type: ignore[union-attr]
        deps.llm.push_text("done")  # type: ignore[union-attr]
        result = agent_runtime_run(trigger=TriggerEvent(kind="test"), deps=deps)
        assert result.cost_usd >= 0.5  # sub-agent cost included
        # Persisted too
        with deps.store.connect() as conn:
            row = conn.execute(
                "SELECT cost_usd, tool_calls_json FROM harness_runs WHERE id = ?",
                (result.run_id,),
            ).fetchone()
        assert float(row[0]) >= 0.5
        # tool_calls_json now has sub_agent_cost_usd field
        payload = _json.loads(row[1])
        assert payload.get("sub_agent_cost_usd", 0) >= 0.5

    # ── Bug 6: feedback notes JSON always valid even with huge inputs
    def test_feedback_notes_remains_valid_json_when_truncated(self, tmp_store):
        # Huge user_text that would naively dumps()→2000-char-truncate to invalid JSON
        huge_text = "我的反馈非常长 " * 500  # ~3500 chars
        sig_id = harness_feedback.on_inbox_accepted(
            tmp_store,
            inbox_id=1,
            skill_run_id=None,
            user_text=huge_text,
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT notes FROM evolution_signals WHERE id = ?",
                (sig_id,),
            ).fetchone()
        # Must be valid JSON — not truncated mid-string
        parsed = _json.loads(row[0])
        assert isinstance(parsed, dict)
        assert "user_text" in parsed
        assert "related_inbox_id" in parsed

    def test_feedback_notes_handles_huge_metadata(self, tmp_store):
        huge_meta = {f"key_{i}": "v" * 100 for i in range(50)}
        sig_id = harness_feedback.record(
            tmp_store,
            harness_feedback.FeedbackContext(
                signal_kind="user_thumbs",
                signal_value=1.0,
                related_inbox_id=1,
                metadata=huge_meta,
            ),
        )
        with tmp_store.connect() as conn:
            row = conn.execute(
                "SELECT notes FROM evolution_signals WHERE id = ?",
                (sig_id,),
            ).fetchone()
        parsed = _json.loads(row[0])  # must parse
        # Either preserved or marker-replaced
        assert isinstance(parsed, dict)
        assert "metadata" in parsed

    def test_auto_load_only_includes_confirmed_memory_summary(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        text = m.auto_load_text(max_lines=200)
        assert "已确认的求职上下文" in text
        assert "worldview/ 文件索引" not in text
        assert "candidate.md" not in text

    # ── Smell 8: view truncates large files
    def test_view_truncates_huge_file(self, tmp_worldview):
        m = MemoryStore(root=tmp_worldview)
        big = "\n".join(f"line {i}" for i in range(2000))
        m.execute({"command": "create", "path": "big.md", "file_text": big})
        result = m.execute({"command": "view", "path": "big.md"})
        assert "showing first 500" in result
        assert "use view_range=[start,end] for the rest" in result
        assert "line 0" in result
        assert "line 499" in result
        assert "line 999" not in result  # truncated

    # ── Q1: temperature is configurable per-call (W15.13)
    def test_temperature_default_passed_through(self, deps):
        deps.llm.push_text("done")  # type: ignore[union-attr]
        agent_runtime_run(trigger=TriggerEvent(kind="test"), deps=deps)
        # StubLLM stores all chat_with_tools calls
        assert deps.llm.calls  # type: ignore[union-attr]
        last = deps.llm.calls[-1]  # type: ignore[union-attr]
        assert last["temperature"] == 0.4  # default

    def test_temperature_override(self, deps):
        deps.llm.push_text("done")  # type: ignore[union-attr]
        agent_runtime_run(
            trigger=make_user_input_trigger("be creative"),
            deps=deps,
            temperature=0.7,
        )
        last = deps.llm.calls[-1]  # type: ignore[union-attr]
        assert last["temperature"] == 0.7

    # ── Q4: tighter compaction thresholds (W15.13)
    def test_compaction_threshold_tightened_to_30k(self):
        # The constants are exported. Verify they match the documented
        # tighter values so future drift is caught.
        from offerguide.agent_runtime.context import (
            CLEAR_TOOL_RESULTS_TRIGGER_TOKENS,
            COMPACTION_TRIGGER_TOKENS,
        )

        assert COMPACTION_TRIGGER_TOKENS == 30_000
        assert CLEAR_TOOL_RESULTS_TRIGGER_TOKENS == 12_000
        # Invariant: clear must trigger BEFORE compaction so the cheap
        # path runs first and may avoid an expensive LLM compaction call.
        assert CLEAR_TOOL_RESULTS_TRIGGER_TOKENS < COMPACTION_TRIGGER_TOKENS

    # ── W15.14: /api/jobs/{id}/track endpoint
    def test_track_job_idempotent(self, web_client):
        client, store = web_client
        # Create a job manually
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source, url, title, company, raw_text, content_hash) "
                "VALUES ('test', 'paste://t1', 't', 'co', 'x', 'hash1') RETURNING id"
            )
            job_id = int(cur.fetchone()[0])

        r1 = client.post(f"/api/jobs/{job_id}/track")
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["created"] is True
        assert "application_id" in d1

        # Second track call → idempotent (returns same application_id, created=False)
        r2 = client.post(f"/api/jobs/{job_id}/track")
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["created"] is False
        assert d2["application_id"] == d1["application_id"]

    def test_track_unknown_job_404(self, web_client):
        client, _ = web_client
        resp = client.post("/api/jobs/99999/track")
        assert resp.status_code == 404

    # ── Submission freezes exact material without inventing a follow-up deadline
    def test_marked_applied_freezes_the_exact_workspace(self, web_client):
        client, store = web_client
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source, url, title, company, raw_text, content_hash) "
                "VALUES ('test', 'paste://t2', 't', 'co', 'x', 'hash2') RETURNING id"
            )
            job_id = int(cur.fetchone()[0])
            application_id = int(
                conn.execute(
                    "INSERT INTO applications(job_id, status) "
                    "VALUES (?, 'considered') RETURNING id",
                    (job_id,),
                ).fetchone()[0]
            )

        import hashlib

        from offerguide.resume import (
            MasterResumeDocument,
            MasterResumeSource,
            ResumeJobContext,
            ResumeWorkspaceRepository,
            build_resume_context,
        )

        repo = ResumeWorkspaceRepository(store)
        master_pdf = store.db_path.parent / "master.pdf"
        master_pdf.write_bytes(b"master resume")
        master_sha = hashlib.sha256(master_pdf.read_bytes()).hexdigest()
        master_source = MasterResumeSource(
            source_path=str(master_pdf.resolve()),
            sha256=master_sha,
            extracted_text="candidate resume",
        )
        master_document = MasterResumeDocument(
            source_sha256=master_sha,
            semantic_text="candidate resume",
            confirmed_by_user=True,
        )
        repo.save_master(
            source_path=master_pdf,
            source_sha256=master_sha,
            extracted_text="candidate resume",
            semantic_document=master_document.model_dump(mode="json"),
            confirmed=True,
        )
        context = build_resume_context(
            job=ResumeJobContext(
                job_id=job_id,
                company="co",
                title="t",
                jd_text="x",
            ),
            master_source=master_source,
            master_document=master_document,
        )
        resume_pdf = store.db_path.parent / "resume.pdf"
        resume_pdf.write_bytes(b"rendered resume")
        repo.save_draft(
            application_id,
            job_snapshot={"job_id": job_id, "company": "co", "title": "t"},
            master_source_sha256=master_sha,
            context=context.model_dump(mode="json"),
            resume_document={
                "header": {
                    "name": {"spans": [{"text": "candidate"}]},
                    "lines": [],
                },
                "sections": [
                    {
                        "title": {"spans": [{"text": "Experience"}]},
                        "entries": [
                            {
                                "rows": [{"left": {"spans": [{"text": "Project"}]}}],
                                "blocks": [],
                            }
                        ],
                    }
                ],
            },
            pdf_path=resume_pdf,
            pdf_sha256=hashlib.sha256(resume_pdf.read_bytes()).hexdigest(),
            apply_pack={"assistant": {"message": "Relevant application message"}},
        )
        resp = client.post(f"/api/jobs/{job_id}/applied")
        assert resp.status_code == 200

        # Application created with status='applied'
        with store.connect() as conn:
            row = conn.execute(
                "SELECT status FROM applications WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        assert row[0] == "applied"
        workspace = repo.get(application_id)
        assert workspace is not None
        assert workspace.status == "submitted"

    # ── W15.14: /jobs page
    def test_jobs_page_renders(self, web_client):
        client, store = web_client
        # Empty state
        r = client.get("/jobs")
        assert r.status_code == 200
        assert "还没评估过任何岗位" in r.text

        # With a job
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source, url, title, company, raw_text, content_hash) "
                "VALUES ('test', 'paste://t3', '算法实习', '字节跳动', 'x', 'hash3') RETURNING id"
            )
            job_id = int(cur.fetchone()[0])
            conn.execute(
                "INSERT INTO applications(job_id, status) VALUES (?, 'considered')",
                (job_id,),
            )
        r2 = client.get("/jobs")
        assert r2.status_code == 200
        assert "字节跳动" in r2.text
        assert "算法实习" in r2.text
        assert "评估过 / 待决" in r2.text

    # ── W15.15: cache_hit_tokens parsing (DeepSeek + Anthropic formats)
    def test_cache_split_parses_deepseek_format(self):
        from offerguide.llm.client import _parse_cache_split

        usage = {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        }
        hit, miss = _parse_cache_split(usage, 1000)
        assert hit == 800
        assert miss == 200

    def test_cache_split_parses_anthropic_format(self):
        from offerguide.llm.client import _parse_cache_split

        usage = {
            "input_tokens": 200,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        }
        # prompt_tokens = 1000 (sum)
        hit, miss = _parse_cache_split(usage, 1000)
        assert hit == 800
        assert miss == 200

    def test_cache_split_falls_back_to_no_cache(self):
        """Old proxy / OpenAI without cache info — assume all miss."""
        from offerguide.llm.client import _parse_cache_split

        usage = {"prompt_tokens": 500, "completion_tokens": 100}
        hit, miss = _parse_cache_split(usage, 500)
        assert hit == 0
        assert miss == 500

    def test_estimate_cost_with_cache_hit_cheaper(self):
        """Cache-hit tokens should be charged at the cheaper rate."""
        from offerguide.llm.pricing import estimate_cost_usd

        # 1000 tokens, no cache hit
        no_cache = estimate_cost_usd(
            model="deepseek-v4-flash",
            prompt_tokens=1000,
            completion_tokens=0,
            cache_hit_tokens=0,
        )
        # 1000 tokens, all cache hit
        all_cache = estimate_cost_usd(
            model="deepseek-v4-flash",
            prompt_tokens=1000,
            completion_tokens=0,
            cache_hit_tokens=1000,
        )
        # Cache-hit price should be ~10% of no-cache
        assert all_cache < no_cache
        assert all_cache <= no_cache * 0.2  # at most 20% (we set ratio at 10%)

    # ── W15.15: daily budget guardrail
    def test_budget_under_cap_passes(self, tmp_store):
        from offerguide.llm.budget import enforce_daily_budget

        # Fresh store, no spend → should not raise
        enforce_daily_budget(tmp_store, cap_usd=5.0)  # no exception

    def test_budget_over_cap_raises(self, tmp_store):
        from offerguide.llm.budget import BudgetExceeded, enforce_daily_budget

        # Inject a huge agent_runtime_run cost
        with tmp_store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_runs(trigger_kind, started_at, cost_usd) "
                "VALUES ('cron', julianday('now'), 99.99)"
            )
        with pytest.raises(BudgetExceeded) as excinfo:
            enforce_daily_budget(tmp_store, cap_usd=5.0)
        assert excinfo.value.today_spent_usd >= 99.0
        assert excinfo.value.cap_usd == 5.0

    def test_budget_disabled_when_cap_zero(self, tmp_store):
        from offerguide.llm.budget import enforce_daily_budget

        # Inject huge cost
        with tmp_store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_runs(trigger_kind, started_at, cost_usd) "
                "VALUES ('cron', julianday('now'), 999.99)"
            )
        # cap=0 → disabled, should not raise
        enforce_daily_budget(tmp_store, cap_usd=0.0)

    def test_budget_includes_skill_runs_too(self, tmp_store):
        from offerguide.llm.budget import get_today_spend_usd

        with tmp_store.connect() as conn:
            conn.execute(
                "INSERT INTO skill_runs(skill_name, skill_version, input_hash, "
                "  input_json, output_json, cost_usd) "
                "VALUES ('example_skill', 'v1', 'h', '{}', '{}', 1.50)"
            )
            conn.execute(
                "INSERT INTO harness_runs(trigger_kind, started_at, cost_usd) "
                "VALUES ('test', julianday('now'), 0.30)"
            )
        spent = get_today_spend_usd(tmp_store)
        assert abs(spent - 1.80) < 0.01

    def test_loop_returns_budget_exceeded_finish_reason(self, tmp_store, tmp_worldview):
        """Harness loop refuses to start when over budget."""
        from offerguide.agent_runtime.loop import run as agent_runtime_run
        with tmp_store.connect() as conn:
            conn.execute(
                "INSERT INTO harness_runs(trigger_kind, started_at, cost_usd) "
                "VALUES ('test', julianday('now'), 99.99)"
            )
        deps = AgentRuntimeDeps(
            settings=Settings(deepseek_api_key="x", default_model="stub"),
            store=tmp_store,
            memory_store=MemoryStore(root=tmp_worldview),
            llm=StubLLM(),  # type: ignore[arg-type]
        )
        # Set a low cap via env
        import os as _os

        prev = _os.environ.get("OFFERGUIDE_DAILY_BUDGET_USD")
        _os.environ["OFFERGUIDE_DAILY_BUDGET_USD"] = "5.0"
        try:
            result = agent_runtime_run(trigger=TriggerEvent(kind="test"), deps=deps)
        finally:
            if prev is None:
                _os.environ.pop("OFFERGUIDE_DAILY_BUDGET_USD", None)
            else:
                _os.environ["OFFERGUIDE_DAILY_BUDGET_USD"] = prev
        assert result.finish_reason == "budget_exceeded"
        assert result.run_id is None
        assert "budget" in (result.error_text or "").lower()

    # ── W15.16: 术语去内核化 — user-facing UI 不应该暴露 internal jargon
    # (W21 redesign 2026-05-15: "Mission Control" 重新成了 home 的设计名,
    # 不再是 "internal jargon" 的同义词. 这个 assert 现在跟设计冲突, 删掉.)
    def test_home_no_mission_control_visible(self, web_client):
        """Root is now the W21 Mission Control workbench."""
        client, _ = web_client
        resp = client.get("/")
        assert "OfferGuide · 指挥台" in resp.text
        assert "Mission Control" in resp.text

    def test_home_has_dejargonized_button_labels(self, web_client):
        """W15.16 — '唤醒 agent' / 'trajectory' 这些 internal 术语该被替换."""
        client, _ = web_client
        resp = client.get("/")
        assert "发送给 Agent" in resp.text
        # 不应再出现 "唤醒 agent" 这种生硬翻译
        # (允许 details/comments — 检查可见 UI 部分)
        assert "Goal (自然语言)" not in resp.text
        assert "事件流 (live)" not in resp.text
        assert "最近 Agent 运行" not in resp.text
        # 老的 "trajectory" 链接文字被改
        assert "执行记录" in resp.text or "Trajectory" not in resp.text

    def test_navbar_keeps_the_three_stage_product_visible(self, web_client):
        """主导航突出找岗、投递材料和面试，不把内部实验室当主流程。"""
        client, _ = web_client
        resp = client.get("/")
        for href in ("/", "/pipeline", "/project-vault", "/debug"):
            assert f'href="{href}"' in resp.text
        assert 'href="/evolution"' not in resp.text

    def test_home_keeps_resume_editing_attached_to_a_job(self, web_client):
        client, _ = web_client
        resp = client.get("/")
        assert "修改当前岗位简历" in resp.text
        assert 'href="/tailor"' not in resp.text

    # ── W15.19: /metrics dogfood dashboard
    def test_metrics_page_renders_empty(self, web_client):
        client, _ = web_client
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "Dogfood Metrics" in resp.text
        assert "评估的岗位" in resp.text
        assert "投递漏斗" in resp.text

    def test_metrics_page_with_data(self, web_client):
        client, store = web_client
        # Inject a job + application + skill_run
        with store.connect() as conn:
            cur = conn.execute(
                "INSERT INTO jobs(source, url, title, company, raw_text, content_hash) "
                "VALUES ('test', 'paste://m1', 't', 'co', 'x', 'h_metrics_1') RETURNING id"
            )
            job_id = int(cur.fetchone()[0])
            conn.execute(
                "INSERT INTO applications(job_id, status, applied_at) "
                "VALUES (?, 'applied', julianday('now'))",
                (job_id,),
            )
            conn.execute(
                "INSERT INTO skill_runs(skill_name, skill_version, input_hash, "
                "  input_json, output_json, cost_usd, latency_ms) "
                "VALUES ('example_skill', 'v1', 'h', '{}', '{}', 0.0123, 100)"
            )
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Should show the application in funnel
        assert "已投" in resp.text
        # Should include the skill in leaderboard
        assert "example_skill" in resp.text

    def test_project_description_matches_the_three_stage_product(self):
        """项目说明应描述真实主流程，而不是旧的面试包装目标。"""
        from pathlib import Path

        root = Path(__file__).parent.parent
        description = root / "求职Agent项目说明.md"
        assert description.exists()
        text = description.read_text(encoding="utf-8")
        assert "投递前" in text
        assert "投递中" in text
        assert "投递后" in text
        assert "自进化" not in text

    def test_navbar_includes_recommended_link(self, web_client):
        client, _ = web_client
        resp = client.get("/")
        assert resp.status_code == 200
        assert "/pipeline" in resp.text
        assert "Pipeline" in resp.text

    def test_apply_pack_404_for_unknown_job(self, web_client):
        client, _ = web_client
        resp = client.get("/jobs/999999/apply-pack")
        assert resp.status_code == 404

    def test_apply_pack_renders_no_llm_key(self, web_client):
        """No LLM key → page renders with friendly error, not 500."""
        client, store = web_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
                "VALUES ('manual', 'AI 实习', '字节', ?, 'h_apack')",
                ("x" * 300,),
            )
            conn.commit()
        jid = 1
        resp = client.get(f"/jobs/{jid}/apply-pack")
        assert resp.status_code == 200
        # web_client fixture has no API key + no profile → some error message
        assert "投递包" in resp.text  # page header still renders
        assert "LLM key" in resp.text or "简历" in resp.text or "SkillRuntime" in resp.text

    def test_post_apply_pack_404_for_unknown_job(self, web_client):
        client, _ = web_client
        resp = client.get("/jobs/999999/post-apply-pack")
        assert resp.status_code == 404

    def test_post_apply_pack_requires_a_frozen_submitted_package(self, web_client):
        client, store = web_client
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (source, title, company, raw_text, content_hash) "
                "VALUES ('manual', 'AI 实习', '字节', ?, 'h_papack')",
                ("x" * 300,),
            )
            conn.commit()
        resp = client.get("/jobs/1/post-apply-pack")
        assert resp.status_code == 409
        assert "真实投递记录" in resp.text

    # ── Smell 6: dead code removed (no `_ = tools` statement at top level)
    def test_loop_module_no_dead_imports(self):
        from offerguide.agent_runtime import loop as loop_mod

        src = Path(loop_mod.__file__).read_text(encoding="utf-8")
        # Check non-comment lines only — the comment explaining the fix
        # legitimately mentions the old code.
        non_comment_lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
        body = "\n".join(non_comment_lines)
        assert "_ = tools" not in body
        assert "_ = _dt" not in body
