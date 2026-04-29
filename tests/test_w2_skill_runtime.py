"""SkillRuntime tests — render → call (stubbed) → parse → persist to skill_runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import offerguide
from offerguide.llm import LLMResponse
from offerguide.skills import SkillRuntime, load_skill
from offerguide.skills.score_match.helpers import ScoreMatchResult, calibration_floor

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


class _StubLLM:
    """Stand-in for LLMClient — records prompts and returns canned responses."""

    def __init__(self, content: str) -> None:
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
        self.calls.append({
            "messages": list(messages),
            "model": model,
            "temperature": temperature,
            "json_mode": json_mode,
        })
        return LLMResponse(
            content=self._content,
            model=model or "stub-model",
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=5,
        )


def test_runtime_renders_system_and_user_messages(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()

    canned = json.dumps({
        "probability": 0.42,
        "reasoning": "技术匹配中等，经验偏弱",
        "dimensions": {"tech": 0.5, "exp": 0.3, "company_tier": 0.6},
        "deal_breakers": [],
    })
    llm = _StubLLM(canned)
    rt = SkillRuntime(llm, store)  # type: ignore[arg-type]

    result = rt.invoke(
        spec,
        {"job_text": "AI Agent at ByteDance", "user_profile": "Statistics master"},
    )

    # Two messages: system = SKILL body, user = labelled inputs
    msgs = llm.calls[0]["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "校准概率" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "### job_text" in msgs[1]["content"]
    assert "### user_profile" in msgs[1]["content"]
    assert "AI Agent at ByteDance" in msgs[1]["content"]
    assert "## 输出" in msgs[1]["content"]

    # JSON mode is on by default for our scoring SKILLs
    assert llm.calls[0]["json_mode"] is True

    # Output parsed and run persisted
    assert result.parsed is not None
    assert result.parsed["probability"] == 0.42
    parsed = ScoreMatchResult.model_validate(result.parsed)
    assert parsed.dimensions.tech == 0.5
    assert result.skill_name == "score_match"
    assert result.skill_run_id > 0

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT skill_name, skill_version, input_hash FROM skill_runs"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "score_match"
    assert rows[0][1] == "0.2.0"
    assert len(rows[0][2]) == 64  # sha256 hex


def test_runtime_validates_required_inputs(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM("{}"), store)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="missing"):
        rt.invoke(spec, {"job_text": "x"})  # user_profile missing


def test_runtime_handles_invalid_json_response(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM("not json"), store)  # type: ignore[arg-type]

    result = rt.invoke(
        spec, {"job_text": "x", "user_profile": "y"}, json_mode=True
    )
    assert result.parsed is None
    assert result.raw_text == "not json"
    # Run still persisted — important for GEPA, who needs to see failures too.
    with store.connect() as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM skill_runs").fetchone()
    assert n == 1


def test_runtime_input_hash_stable(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM("{}"), store)  # type: ignore[arg-type]

    inputs = {"job_text": "x", "user_profile": "y"}
    a = rt.invoke(spec, inputs)
    b = rt.invoke(spec, inputs)
    assert a.input_hash == b.input_hash


def test_runtime_input_hash_diverges_on_input_change(tmp_path: Path) -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    store = offerguide.Store(tmp_path / "s.db")
    store.init_schema()
    rt = SkillRuntime(_StubLLM("{}"), store)  # type: ignore[arg-type]

    a = rt.invoke(spec, {"job_text": "x", "user_profile": "y"})
    b = rt.invoke(spec, {"job_text": "x", "user_profile": "z"})
    assert a.input_hash != b.input_hash


def test_score_match_helpers_calibration_floor() -> None:
    assert calibration_floor(0.9, n_samples=0) == 0.5
    # As n grows, the prior weakens
    assert 0.5 < calibration_floor(0.9, n_samples=10) < 0.9
    assert calibration_floor(0.9, n_samples=1000) == pytest.approx(0.9, abs=0.02)


def test_score_match_pydantic_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ScoreMatchResult.model_validate(
            {
                "probability": 1.5,  # > 1
                "reasoning": "x",
                "dimensions": {"tech": 0.5, "exp": 0.5, "company_tier": 0.5},
            }
        )


def test_score_match_pydantic_forbids_extra_keys() -> None:
    with pytest.raises(ValidationError):
        ScoreMatchResult.model_validate(
            {
                "probability": 0.5,
                "reasoning": "x",
                "dimensions": {"tech": 0.5, "exp": 0.5, "company_tier": 0.5},
                "uninvited": "extra",
            }
        )


def test_skill_load_now_at_v0_2() -> None:
    """Sanity check that the W2 prompt update bumped the version."""
    spec = load_skill(SKILLS_ROOT / "score_match")
    assert spec.version == "0.2.0"
    assert spec.parent_version == "0.1.0"
    assert "校准概率" in spec.body
    assert "deal_breakers" in spec.output_schema
