"""W5' (Phase 0) — SKILL runtime strict input handling.

Locks down two invariants:
1. Strict mode (default) rejects any extra input keys not declared in spec.inputs.
2. Render, hash, and persist all read from the SAME canonical input dict, so
   two functionally-identical invocations always produce the same input_hash
   and the same persisted skill_runs.input_json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import offerguide
from offerguide.llm import LLMResponse
from offerguide.skills import SkillRuntime, load_skill

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


class _StubLLM:
    def __init__(self, content: str = "{}") -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages,
        *,
        model=None,
        temperature=0.3,
        json_mode=False,
        extra=None,
    ) -> LLMResponse:
        self.calls.append({"messages": list(messages)})
        return LLMResponse(content=self._content, model="stub")


def test_strict_mode_rejects_extra_inputs(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM(), store)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unexpected inputs"):
        rt.invoke(
            spec,
            {
                "job_text": "x",
                "user_profile": "y",
                "typo_extra_key": "oops",
            },
        )


def test_non_strict_mode_accepts_extras(tmp_path: Path) -> None:
    """Opt-out path for debug context that is allowed to drift."""
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM(), store)  # type: ignore[arg-type]

    # Extras silently dropped (not passed to LLM, not in hash)
    result = rt.invoke(
        spec,
        {"job_text": "x", "user_profile": "y", "debug_marker": "1"},
        strict_inputs=False,
    )
    assert result.skill_run_id > 0


def test_strict_mode_still_reports_missing_required(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM(), store)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="missing"):
        rt.invoke(spec, {"job_text": "only one"})


def test_input_hash_ignores_extras_when_non_strict(tmp_path: Path) -> None:
    """Two non-strict calls with the same declared inputs but different extras
    must hash equally — extras are dropped during canonicalization."""
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM(), store)  # type: ignore[arg-type]

    a = rt.invoke(
        spec,
        {"job_text": "x", "user_profile": "y", "_dbg": "alpha"},
        strict_inputs=False,
    )
    b = rt.invoke(
        spec,
        {"job_text": "x", "user_profile": "y", "_dbg": "beta"},
        strict_inputs=False,
    )
    assert a.input_hash == b.input_hash


def test_input_hash_changes_when_declared_input_changes(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM(), store)  # type: ignore[arg-type]

    a = rt.invoke(spec, {"job_text": "x", "user_profile": "y"})
    b = rt.invoke(spec, {"job_text": "x", "user_profile": "z"})
    assert a.input_hash != b.input_hash


def test_render_and_persist_use_same_canonical_inputs(tmp_path: Path) -> None:
    """The user message rendered to the LLM and the skill_runs.input_json blob
    must contain the SAME keys — neither over- nor under-set against canonical."""
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    llm = _StubLLM()
    rt = SkillRuntime(llm, store)  # type: ignore[arg-type]

    rt.invoke(
        spec,
        {"job_text": "前端实习生", "user_profile": "胡阳"},
        strict_inputs=False,
    )

    user_msg = llm.calls[0]["messages"][1]["content"]
    assert "### job_text" in user_msg
    assert "### user_profile" in user_msg

    with store.connect() as conn:
        (input_json,) = conn.execute("SELECT input_json FROM skill_runs").fetchone()
    persisted = json.loads(input_json)
    assert set(persisted.keys()) == {"job_text", "user_profile"}
    assert persisted["job_text"] == "前端实习生"


def test_render_iterates_inputs_in_declared_order(tmp_path: Path) -> None:
    """Order matters for hash stability — declared order is the canonical order."""
    spec = load_skill(SKILLS_ROOT / "score_match")
    # spec.inputs is ('job_text', 'user_profile'); render must show job_text first
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    llm = _StubLLM()
    rt = SkillRuntime(llm, store)  # type: ignore[arg-type]

    # Pass in reversed order
    rt.invoke(spec, {"user_profile": "胡阳", "job_text": "前端"})

    user_msg = llm.calls[0]["messages"][1]["content"]
    # job_text must appear before user_profile in the rendered output
    assert user_msg.index("### job_text") < user_msg.index("### user_profile")
