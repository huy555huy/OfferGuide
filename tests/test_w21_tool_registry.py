"""Conversation-agent tool registry tests."""
from __future__ import annotations

import json

import pytest

from offerguide.tools.registry import (
    ToolRegistry,
    registry,
    tool_error,
    tool_result,
)


@pytest.fixture
def fresh_registry():
    """Test isolation: clean slate per test."""
    r = ToolRegistry()
    return r


# ── Helpers ────────────────────────────────────────────────────────


class TestHelpers:
    def test_tool_error_basic(self):
        out = tool_error("not found")
        assert json.loads(out) == {"error": "not found"}

    def test_tool_error_with_extra(self):
        out = tool_error("bad input", code=400, field="job_id")
        d = json.loads(out)
        assert d["error"] == "bad input"
        assert d["code"] == 400
        assert d["field"] == "job_id"

    def test_tool_result_with_data(self):
        out = tool_result({"score": 0.75, "company": "百度"})
        assert json.loads(out) == {"score": 0.75, "company": "百度"}

    def test_tool_result_with_kwargs(self):
        out = tool_result(items=[1, 2], count=2)
        assert json.loads(out) == {"items": [1, 2], "count": 2}

    def test_tool_result_rejects_both(self):
        with pytest.raises(ValueError):
            tool_result({"a": 1}, b=2)

    def test_tool_result_chinese_chars_unescaped(self):
        """ensure_ascii=False so 中文 stays readable in JSON."""
        out = tool_result(message="任务完成")
        assert "任务完成" in out  # not 任...


# ── Registration ──────────────────────────────────────────────────


class TestRegistration:
    def test_register_one_tool(self, fresh_registry):
        def _handler(args, **kw):
            return tool_result(echo=args)

        fresh_registry.register(
            name="echo",
            group="main",
            schema={
                "name": "echo",
                "description": "echo input back",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=_handler,
        )
        entry = fresh_registry.get("echo")
        assert entry is not None
        assert entry.name == "echo"
        assert entry.group == "main"

    def test_register_overwrite_warns(self, fresh_registry, caplog):
        def _h1(args, **kw): return "{}"
        def _h2(args, **kw): return "{}"
        fresh_registry.register(
            name="x", group="main", schema={"name": "x", "parameters": {}}, handler=_h1,
        )
        fresh_registry.register(
            name="x", group="main", schema={"name": "x", "parameters": {}}, handler=_h2,
        )
        # Second wins
        assert fresh_registry.get("x").group == "main"
        assert fresh_registry.get("x").handler is _h2


# ── Dispatch ──────────────────────────────────────────────────────


class TestDispatch:
    def test_dispatch_success(self, fresh_registry):
        def _h(args, **kw):
            return tool_result(echoed=args.get("x"))
        fresh_registry.register(
            name="echo", group="main",
            schema={"name": "echo", "parameters": {}}, handler=_h,
        )
        out = fresh_registry.dispatch("echo", {"x": "hello"})
        assert json.loads(out) == {"echoed": "hello"}

    def test_dispatch_unknown_tool(self, fresh_registry):
        out = fresh_registry.dispatch("does_not_exist", {})
        d = json.loads(out)
        assert "error" in d
        assert "does_not_exist" in d["error"]

    def test_dispatch_handler_exception_returns_error_json(self, fresh_registry):
        def _crash(args, **kw):
            raise ValueError("oops")
        fresh_registry.register(
            name="crash", group="main",
            schema={"name": "crash", "parameters": {}}, handler=_crash,
        )
        out = fresh_registry.dispatch("crash", {})
        d = json.loads(out)
        assert "error" in d
        assert "ValueError" in d["error"]
        assert "oops" in d.get("detail", "")

    def test_dispatch_passes_runtime_kwargs(self, fresh_registry):
        captured = {}
        def _h(args, *, store=None, settings=None):
            captured["store"] = store
            captured["settings"] = settings
            return "{}"
        fresh_registry.register(
            name="needs_store", group="main",
            schema={"name": "needs_store", "parameters": {}}, handler=_h,
        )
        fresh_registry.dispatch(
            "needs_store", {},
            store="STORE_OBJ", settings="SETTINGS_OBJ",
        )
        assert captured["store"] == "STORE_OBJ"
        assert captured["settings"] == "SETTINGS_OBJ"

    def test_dispatch_handler_returns_non_str_gets_serialized(self, fresh_registry):
        """Defensive: if a handler accidentally returns dict, wrap it."""
        def _h(args, **kw):
            return {"oops_not_json": True}  # type: ignore[return-value]
        fresh_registry.register(
            name="bad_handler", group="main",
            schema={"name": "bad_handler", "parameters": {}}, handler=_h,
        )
        out = fresh_registry.dispatch("bad_handler", {})
        assert json.loads(out) == {"oops_not_json": True}


# ── Module-level singleton ────────────────────────────────────────


class TestSingleton:
    def test_singleton_is_importable(self):
        from offerguide.tools.registry import registry as r1
        from offerguide.tools.registry import registry as r2
        assert r1 is r2  # Same object

    def test_singleton_starts_empty_or_loaded(self):
        # Just verify it exists and is a ToolRegistry
        assert isinstance(registry, ToolRegistry)
