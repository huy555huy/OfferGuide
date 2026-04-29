"""DSPy module wrapper tests — pure structural checks, no LLM calls."""

from __future__ import annotations

from pathlib import Path

import dspy
import pytest

from offerguide.evolution.dspy_module import (
    build_signature,
    build_student,
    read_instructions,
)
from offerguide.skills import load_skill
from offerguide.skills._spec import SkillSpec

SKILLS_ROOT = Path(__file__).parent.parent / "src/offerguide/skills"


def test_build_signature_uses_skill_body_as_instructions() -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    sig = build_signature(spec)
    assert sig.instructions == spec.body
    # Inputs and one output (response_json)
    assert set(sig.input_fields.keys()) == set(spec.inputs)
    assert "response_json" in sig.output_fields


def test_build_signature_accepts_explicit_instructions_override() -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    sig = build_signature(spec, instructions="alternate prompt")
    assert sig.instructions == "alternate prompt"


def test_build_signature_rejects_skill_with_no_inputs() -> None:
    bad = SkillSpec(
        name="x",
        description="desc",
        version="0.0.1",
        body="body",
        inputs=(),
    )
    with pytest.raises(ValueError, match="no declared inputs"):
        build_signature(bad)


def test_build_student_returns_dspy_module_instance() -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    student = build_student(spec)
    assert isinstance(student, dspy.Module)
    assert hasattr(student, "predict")
    assert isinstance(student.predict, dspy.Predict)


def test_build_student_returns_fresh_instances() -> None:
    """Each call should return a new module — GEPA mutates `predict.signature.instructions`
    in place, so a shared instance would leak state across runs."""
    spec = load_skill(SKILLS_ROOT / "score_match")
    a = build_student(spec)
    b = build_student(spec)
    assert a is not b
    assert a.predict is not b.predict
    # Mutate one; the other should be unaffected
    a.predict.signature = a.predict.signature.with_instructions("mutated")
    assert "mutated" not in b.predict.signature.instructions


def test_read_instructions_extracts_current_signature() -> None:
    spec = load_skill(SKILLS_ROOT / "score_match")
    student = build_student(spec)
    assert read_instructions(student) == spec.body
    student.predict.signature = student.predict.signature.with_instructions("evolved!")
    assert read_instructions(student) == "evolved!"
