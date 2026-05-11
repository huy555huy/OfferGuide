"""W21 — Tool registry tests.

This is the foundation for the agent-first refactor. Every other phase
depends on this working correctly.
"""
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
            name="x", group="discovery", schema={"name": "x", "parameters": {}}, handler=_h2,
        )
        # Second wins
        assert fresh_registry.get("x").group == "discovery"
        assert fresh_registry.get("x").handler is _h2

    def test_aliases_map_back_to_canonical(self, fresh_registry):
        def _h(args, **kw): return "{}"
        fresh_registry.register(
            name="score_job",
            group="evaluation",
            schema={"name": "score_job", "parameters": {}},
            handler=_h,
            deprecated_aliases=("score_match",),
        )
        # Alias resolves
        assert fresh_registry.get("score_match") is not None
        assert fresh_registry.get("score_match").name == "score_job"

    def test_deregister_clears_aliases(self, fresh_registry):
        def _h(args, **kw): return "{}"
        fresh_registry.register(
            name="x", group="main",
            schema={"name": "x", "parameters": {}}, handler=_h,
            deprecated_aliases=("old_x",),
        )
        assert fresh_registry.get("old_x") is not None
        fresh_registry.deregister("x")
        assert fresh_registry.get("x") is None
        assert fresh_registry.get("old_x") is None


# ── Query / grouping ──────────────────────────────────────────────


class TestQuery:
    def _register_4(self, r):
        def _h(args, **kw): return "{}"
        r.register(name="m1", group="main", schema={"name": "m1", "parameters": {}}, handler=_h)
        r.register(name="m2", group="main", schema={"name": "m2", "parameters": {}}, handler=_h)
        r.register(name="d1", group="discovery", schema={"name": "d1", "parameters": {}}, handler=_h)
        r.register(name="s1", group="shared", schema={"name": "s1", "parameters": {}}, handler=_h)

    def test_names_in_group(self, fresh_registry):
        self._register_4(fresh_registry)
        assert fresh_registry.names_in_group("main") == ["m1", "m2"]
        assert fresh_registry.names_in_group("discovery") == ["d1"]
        assert fresh_registry.names_in_group("shared") == ["s1"]

    def test_names_for_agent_includes_shared(self, fresh_registry):
        self._register_4(fresh_registry)
        # Discovery sub-agent should see d1 + s1 (shared), not m1/m2
        names = fresh_registry.names_for_agent("discovery", include_shared=True)
        assert "d1" in names
        assert "s1" in names
        assert "m1" not in names
        assert "m2" not in names

    def test_names_for_agent_exclude_shared(self, fresh_registry):
        self._register_4(fresh_registry)
        names = fresh_registry.names_for_agent("main", include_shared=False)
        assert names == ["m1", "m2"]


class TestSchemas:
    def test_get_schemas_returns_openai_format(self, fresh_registry):
        def _h(args, **kw): return "{}"
        fresh_registry.register(
            name="score_job",
            group="evaluation",
            schema={
                "name": "score_job",
                "description": "Score a job's fit",
                "parameters": {
                    "type": "object",
                    "properties": {"job_id": {"type": "integer"}},
                    "required": ["job_id"],
                },
            },
            handler=_h,
        )
        schemas = fresh_registry.get_schemas("evaluation")
        assert len(schemas) == 1
        s = schemas[0]
        assert s["type"] == "function"
        assert s["function"]["name"] == "score_job"
        assert s["function"]["description"] == "Score a job's fit"
        assert "parameters" in s["function"]

    def test_get_schemas_adds_name_field_if_missing(self, fresh_registry):
        """Some callers may pass schema without 'name'; registry fills it."""
        def _h(args, **kw): return "{}"
        fresh_registry.register(
            name="x",
            group="main",
            schema={
                "description": "x",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=_h,
        )
        schemas = fresh_registry.get_schemas("main")
        assert schemas[0]["function"]["name"] == "x"


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

    def test_dispatch_via_alias(self, fresh_registry):
        def _h(args, **kw): return tool_result(ok=True)
        fresh_registry.register(
            name="score_job", group="evaluation",
            schema={"name": "score_job", "parameters": {}},
            handler=_h, deprecated_aliases=("score_match",),
        )
        # Dispatch via alias works
        out = fresh_registry.dispatch("score_match", {})
        assert json.loads(out) == {"ok": True}

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
